"""
Разбор чертежа: разметка листа зрением плюс чтение внешней моделью Qwen3-VL.

Как устроена ветка и почему именно так.

1. **Зрение размечает, модель читает.** Геометрия (`drawing_vision`) выправляет
   наклон, находит рамку формата, штамп и таблицы; модель читает лист целиком
   и каждую найденную область отдельно. Прогон шести способов на настоящем
   листе показал, почему нужны оба: сплошное чтение листа теряет таблицу
   целиком (спецификация — ноль строк из пяти), а нарезка по найденной сетке
   поднимает её до полного восстановления структуры.

2. **Модель внешняя.** Веса живут на отдельном хосте с GPU; конвейер ходит
   туда по HTTP. Если адрес не задан или хост молчит, чертёж остаётся
   неразобранным, и это видно: фрагмент не превращается в `drawing`, в логе
   стоит причина. Подменять разбор сплошным OCR по листу больше нечем —
   на живом листе он давал 24% попаданий при 85% мусора.

3. **Правила проверяют модель.** Категорию, допуск, единицу и характер
   размера определяют те же правила ГОСТ, что и для векторного чертежа
   (`vector_drawing`). Совпала категория модели с правилом — уверенность
   высокая; разошлись — поле остаётся, но уверенность честно ниже.

4. **Детектор YOLO и Florence в ветке больше не участвуют.** Стоковые веса
   DOTA не знают категорий чертежа и возвращали ноль областей, из-за чего лист
   доезжал до сплошного распознавания. Их код из проекта удалён; появятся
   веса под машиностроительный чертёж — детектор можно вернуть из истории
   git как источник областей вместо геометрии.

5. **Таблица составных частей не помещается в контракт.** Полей вида
   «строка спецификации» в нём нет, поэтому таблицы уходят в
   `structured_payload.tables` целиком, а в поля — только то, для чего
   категория предусмотрена.

Связи. Вызывается из `ingest.tasks` после нормализации:
`DrawingProcessor.enrich_fragments` заменяет фрагменты `image` на `drawing`
и, если документ — скан или фото чертежа, разбирает листы целиком
(страницы из `raster_drawing_pages` или `drawing_pages` в
`classification.signals`, не больше `DRAWING_MAX_SHEETS`). Векторный чертёж
приходит сюда уже готовым фрагментом `drawing` из текстового слоя
(`vector_drawing`) и не трогается. Лист, который `vector_drawing` признал
текстом в рамке ЕСКД (`text_in_frame`, флаг `DRAWING_TEXT_IN_FRAME_ENABLED`),
чертёжным не считается и полей чертежа не получает.

Вызовы модели (`QwenVisionClient`, `QWEN_ENDPOINT`): `read_sheet` — лист
целиком, затем по области, найденной геометрией (всего не больше
`DRAWING_MAX_REGIONS`): `read_title_block` для штампа, `read_table` для
остальных таблиц. Время на лист ограничено `DRAWING_TIME_BUDGET`. Если до
срока задачи не укладывается даже один вызов (`VLM_TIMEOUT`), фрагмент
остаётся с пометкой `DRAWING_SKIPPED_NO_TIME`, а неразобранные листы скана —
с `DRAWING_SHEETS_SKIPPED`: документ сохраняется, а не теряется целиком.

На листе, весь текст которого переписан моделью, записи распознавания того же
листа правятся её текстом (`core/drawing_transcript.py`,
`DRAWING_TRANSCRIPT_FIXES_OCR`), а повторы и шум распознавания убираются
(`core/drawing_stale_sources.py`, `DRAWING_SUPPRESS_STALE_ON_FULL_READ`).
"""

import logging
import time
from typing import Any, Dict, List, Optional

from core import filetypes
from celery.exceptions import SoftTimeLimitExceeded

from core import deadline
from core.config import settings
from core.models.contract import Content, DrawingField, Fragment, Position, Provenance
from core.providers import (
    dimension_fidelity, drawing_text, drawing_vision, raster_drawing, title_block,
    vector_drawing,
)
from core.providers.qwen_client import CATEGORIES, QwenVisionClient
from core.providers.storage import StorageProvider, StorageProviderFactory
from core.providers.vector_drawing import structure_ratio
from core.drawing_quality import apply_drawing_quality
from core.drawing_stale_sources import suppress_stale_sources
from core.drawing_transcript import apply_transcripts

logger = logging.getLogger(__name__)

# Жанр и источник из классификации лестницы (core.ladder.context). Строками,
# чтобы не тянуть сюда модуль лестницы ради двух констант.
GENRE_DRAWING = "drawing"
SOURCE_RASTER = "raster_only"

MODE_VISION_VLM = "vision_vlm"   # разметка зрением + чтение моделью
MODE_SHEET_ONLY = "sheet_only"   # только чтение листа целиком
MODE_NONE = "none"               # разобрать не удалось

# Способ получения полей -> (метод в контракте, уровень лестницы).
# Внешняя мультимодальная модель — это уровень 7 по определению лестницы.
_PROVENANCE_BY_MODE = {
    MODE_VISION_VLM: ("vision_plus_vlm", 7),
    MODE_SHEET_ONLY: ("vlm_full_page", 7),
}

# Уверенность в поле складывается из согласия модели с правилами ГОСТ.
_AGREED = 0.9            # модель и правило назвали одну категорию
_MODEL_ONLY = 0.75       # правило не знает такой категории (позиция, штамп)
_DISAGREED = 0.6         # разошлись: поле остаётся, но доверия меньше
_FROM_REGION = 0.85      # прочитано в области, найденной геометрией
_HANDWRITTEN = 0.5       # дописано от руки: проверить правилами нечем

# Классификатор «чертёж или текст» (DRAWING_PAGE_CLASSIFIER_ENABLED). В
# текстовом документе картинка меньше этой доли листа — логотип, печать,
# подпись, гриф «Утверждаю», QR-код. Qwen-VL делает из них «основную надпись»
# чертежа («Ген», «202д», «НТА-прибор»). Вид или схема в ТЗ и ПМ занимают
# десятую часть листа и больше и разбираются как чертёж.
_TEXT_DOC_MIN_FIGURE_AREA = 0.06
# Документ текстовый, если в среднем на страницу у него столько символов
# текста и таблиц. Скан сборочного чертежа, который растровый детектор не
# узнал, — это штамп и пара подписей: до порога он не дотягивает.
_TEXT_DOC_MIN_CHARS_PER_PAGE = 300


class DrawingProcessor:
    """Изображение чертежа -> структурированные поля -> фрагмент `drawing`."""

    def __init__(
        self,
        storage: Optional[StorageProvider] = None,
        qwen: Optional[QwenVisionClient] = None,
    ):
        self.storage = storage or StorageProviderFactory.default()
        self.qwen = qwen or QwenVisionClient()
        self.vision_enabled = settings.DRAWING_VISION_ENABLED

    # ------------------------------------------------------------- публичное
    def process(
        self, image_ref: str, metadata: Dict[str, Any], page: int = 1,
        transcribe: bool = False, log_page: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Возвращает:
        {'detected_count', 'parsed_fields', 'avg_confidence', 'completeness',
         'mode', 'payload'}

        `log_page` — номер страницы только для лога: у вырезки MinerU номер
        страницы живёт во фрагменте, а не в `page` (там всегда 1, потому что
        вырезка — отдельная картинка). На разбор не влияет.
        """
        if not self.qwen.available:
            logger.error(
                "QWEN_ENDPOINT не задан — чертёж %s не разобран. Ветка чертежей "
                "работает только с внешней моделью.", image_ref,
            )
            return _empty(MODE_NONE)

        image = self._open(image_ref, page)
        if image is None:
            return _empty(MODE_NONE)

        started = time.monotonic()
        sheet = page if log_page is None else log_page
        layout = (
            drawing_vision.analyse(image) if self.vision_enabled
            else drawing_vision.plain(image)
        )

        fields: List[Dict[str, Any]] = []
        payload: Dict[str, Any] = {"layout": layout.as_dict(), "tables": []}
        payload["coordinate_reference"] = "deskewed_page"
        missing_regions: List[Dict[str, Any]] = []

        pass_started = time.monotonic()
        reading = self.qwen.read_sheet(layout.image, hint=layout.hint())
        self._log_pass(
            sheet, "лист целиком", pass_started,
            ("надписей %d" % len(reading.annotations)) if reading is not None
            else "лист не прочитан",
        )
        # Весь текст листа отдельным проходом (DRAWING_IMAGE_FULL_READ):
        # разметка надписей по категориям теряет штамп, графы, номер листа
        # и примечания, а без них лист нельзя найти ни по обозначению, ни
        # по требованиям.
        lines = None
        if transcribe:
            pass_started = time.monotonic()
            lines = self._transcribe(layout.image, payload, missing_regions, started)
            tiles = int(payload.get("transcript_tiles") or 0)
            self._log_pass(
                sheet, "весь текст листа", pass_started,
                ("строк текста %d%s" % (len(lines), ", частей %d" % tiles if tiles else ""))
                if lines else "текст не прочитан",
            )
        if lines:
            payload["transcript_lines"] = lines
        if reading is None and not lines:
            logger.error("Модель не прочитала лист %s", image_ref)
            self._log_sheet(image_ref, sheet, MODE_NONE, 0, 0, 0, 0, started)
            return _empty(MODE_NONE)
        if reading is not None:
            payload["sheet_type"] = reading.sheet_type
            fields.extend(self._fields_from_annotations(reading.annotations))

        # Штамп важнее прочих таблиц: если времени хватит не на все области,
        # прочитан будет он.
        regions = sorted(
            layout.regions,
            key=lambda r: r.kind != drawing_vision.KIND_TITLE_BLOCK,
        )
        regions_read, skipped = 0, 0
        for region in regions:
            if self._out_of_time(started):
                skipped += 1
                missing_regions.append({"kind": region.kind, "bbox": region.bbox, "reason": "time_budget"})
                continue
            pass_started = time.monotonic()
            before = len(fields)
            if region.kind == drawing_vision.KIND_TITLE_BLOCK:
                read = self._read_title_block(layout, region, fields, payload)
                with_side = bool((payload.get("side_column") or {}).get("read"))
                self._log_pass(
                    sheet, "штамп", pass_started,
                    ("граф %d%s" % (len(fields) - before,
                                    ", с боковой колонкой" if with_side else ""))
                    if read else "штамп не прочитан",
                )
            else:
                piece = drawing_vision.crop(layout.image, region.bbox)
                read = self._read_table(piece, region, payload)
                self._log_pass(
                    sheet, "таблица", pass_started,
                    ("строк %d" % len(payload["tables"][-1]["rows"])) if read
                    else "таблица не прочитана",
                )
            regions_read += int(read)
            if not read:
                missing_regions.append({"kind": region.kind, "bbox": region.bbox, "reason": "region_not_read"})
        if skipped:
            logger.warning(
                "На %d областей листа %s не хватило бюджета времени (%d с)",
                skipped, image_ref, settings.DRAWING_TIME_BUDGET,
            )
            payload["regions_skipped"] = skipped

        fields = _deduplicate(fields)
        if settings.DRAWING_DIMENSION_FIDELITY:
            # Один размер, прочитанный и на листе целиком, и на вырезке вида:
            # остаётся прочтение со знаком «±» и со справочной звёздочкой.
            fields = dimension_fidelity.merge_readings(fields)
        mode = MODE_VISION_VLM if (self.vision_enabled and layout.regions) else MODE_SHEET_ONLY
        if not fields and not payload["tables"] and not payload.get("transcript_lines"):
            self._log_sheet(image_ref, sheet, MODE_NONE, 0, len(lines or []),
                            regions_read, len(regions), started)
            return _empty(MODE_NONE)

        payload["regions_read"] = regions_read
        payload["model"] = self.qwen.model
        self._log_sheet(image_ref, sheet, mode, len(fields), len(lines or []),
                        regions_read, len(regions), started)
        confidence = (
            sum(f["confidence"] for f in fields) / len(fields) if fields else 0.0
        )
        return {
            "detected_count": len(fields),
            "parsed_fields": fields,
            "avg_confidence": round(confidence, 3),
            # Покрытие найденных таблиц — независимый сигнал, но не
            # доказательство чтения всех размеров/видов листа.
            "completeness": None,
            "structure_score": structure_ratio(fields),
            "completeness_mechanisms": {
                "geometry_regions": {
                    "expected": len(regions), "read": regions_read,
                    "coverage": regions_read / len(regions) if regions else None,
                },
                "whole_sheet": {"coverage": None, "reason": "not_independently_measured"},
            },
            "missing_regions": missing_regions,
            "warnings": ["DRAWING_COMPLETENESS_UNKNOWN"],
            "mode": mode,
            "payload": payload,
        }

    # ------------------------------------------------------------------- лог
    # Доказательство работы модели на чертеже читается прямо из лога: по
    # строке на каждый вызов Qwen-VL и строка-итог на лист. Содержимого
    # листа в лог не попадает — только номер страницы, вид прохода, счётчики,
    # имя модели и время.
    def _log_pass(self, sheet: Any, kind: str, started: float, produced: str) -> None:
        """Один вызов модели: что за проход, что дал и сколько шёл."""
        logger.info(
            "Qwen-VL: стр. %s, проход «%s», %s, модель %s, %.1f с",
            sheet, kind, produced, self.qwen.model, time.monotonic() - started,
        )

    def _log_sheet(
        self, image_ref: str, sheet: Any, mode: str, fields: int, lines: int,
        regions_read: int, regions: int, started: float,
    ) -> None:
        """Итог по листу: один вызов `process` — одна строка."""
        logger.info(
            "Qwen-VL итог: лист %s стр. %s, режим %s, полей %d, строк текста %d, "
            "областей %d из %d, модель %s, %.1f с",
            image_ref, sheet, mode, fields, lines, regions_read, regions,
            self.qwen.model, time.monotonic() - started,
        )

    def build_fragment(
        self,
        drawing_result: Dict[str, Any],
        source: Fragment,
        metadata: Dict[str, Any],
    ) -> Optional[Fragment]:
        """Строит фрагмент типа `drawing` на месте фрагмента-изображения."""
        fields = drawing_result.get("parsed_fields") or []
        payload = dict(drawing_result.get("payload") or {})
        # Текст листа идёт в content.text, а в payload остаётся только число
        # строк: иначе потребитель индексировал бы его дважды.
        transcript = payload.pop("transcript_lines", None) or []
        if transcript:
            payload["transcript_line_count"] = len(transcript)
        if not fields and not payload.get("tables") and not transcript:
            return None

        mode = drawing_result.get("mode", MODE_SHEET_ONLY)
        method, level = _PROVENANCE_BY_MODE.get(mode, ("vlm_full_page", 7))
        payload.update({
            "detected_count": drawing_result.get("detected_count", 0),
            "recognition_mode": mode,
        })
        fragment = Fragment(
            fragment_id=source.fragment_id,
            type="drawing",
            content=Content(
                # Переписанный текст листа — то, по чему лист находят поиском.
                text="\n".join(transcript) or None,
                image_ref=source.content.image_ref,
                structured_drawing_fields=[DrawingField(**f) for f in fields],
                structured_payload=payload,
            ),
            position=source.position.model_copy(deep=True) if source.position else Position(page=1, bbox=[0, 0, 1, 1]),
            section_title=source.section_title,
            section_path=list(source.section_path),
            confidence=round(float(drawing_result.get("avg_confidence", 0.0)), 3),
            completeness=drawing_result.get("completeness"),
            structure_score=drawing_result.get("structure_score"),
            completeness_mechanisms=drawing_result.get("completeness_mechanisms") or {},
            missing_regions=[
                {**region, "page": source.position.page if source.position else 0,
                 "sheet": source.position.sheet if source.position else None}
                for region in drawing_result.get("missing_regions") or []
            ],
            warnings=list(dict.fromkeys(source.warnings + list(drawing_result.get("warnings") or []))),
            provenance=Provenance(method=method, strategy_level=level, source="drawing",
                                  source_unit_id=source.provenance.source_unit_id),
            graph_nodes=list(source.graph_nodes),
            relations=list(source.relations),
        )
        if fragment.position:
            fragment.position.bbox_exact = False
            fragment.position.page_start = fragment.position.page_end = fragment.position.page
        apply_drawing_quality(fragment)
        return fragment

    def enrich_fragments(
        self, fragments: List[Fragment], metadata: Dict[str, Any]
    ) -> List[Fragment]:
        """
        Заменяет image-фрагменты на drawing-фрагменты там, где получилось, и
        отдельно разбирает лист целиком, если сам документ — растровый
        чертёж. Без второго обработчик молчал на сканах и фотографиях
        чертежей: фрагмента `image` в таком документе нет.
        """
        result: List[Fragment] = []
        text_document = settings.DRAWING_PAGE_CLASSIFIER_ENABLED and _is_text_document(fragments, metadata)
        for fragment in fragments:
            # Векторный чертёж уже разобран парсером по текстовому слою —
            # там точные значения и углы, читать нечего.
            if fragment.type == "drawing":
                result.append(fragment)
                continue
            if fragment.type != "image" or not fragment.content.image_ref:
                result.append(fragment)
                continue
            if text_document and _is_small_picture(fragment):
                # Логотип или печать в тексте: чертежом не становится и
                # вызова модели не стоит. Фрагмент остаётся картинкой.
                result.append(fragment)
                continue
            left = deadline.remaining()
            if left is not None and left < settings.VLM_TIMEOUT:
                # До срока задачи не уложится даже один вызов модели: лучше
                # сохранить документ с пометкой, чем потерять его целиком.
                fragment.review_required = True
                fragment.warnings = list(dict.fromkeys(fragment.warnings + ["DRAWING_SKIPPED_NO_TIME"]))
                result.append(fragment)
                continue
            try:
                drawing_result = self.process(
                    fragment.content.image_ref, metadata,
                    log_page=fragment.position.page if fragment.position else None,
                )
                enriched = self.build_fragment(drawing_result, fragment, metadata)
            except SoftTimeLimitExceeded:
                raise
            except Exception as exc:  # noqa: BLE001 — чертёж не должен ронять документ
                logger.error("Обработка чертежа %s не удалась: %s",
                             fragment.content.image_ref, exc)
                enriched = None
            if enriched is None and self._source_is_raster_drawing(metadata):
                fragment.review_required = True
                fragment.warnings = list(dict.fromkeys(fragment.warnings + ["DRAWING_NOT_RECOGNIZED"]))
            result.append(enriched or fragment)

        had_drawings = any(f.type == "drawing" for f in result)
        sheets = self._sheet_drawings(result, metadata)
        pdf_full_read = self._full_read_pdf(metadata)
        if sheets and (pdf_full_read or (had_drawings and settings.DRAWING_SHEET_MISSING_PAGES)):
            # Дочитанный лист встаёт перед фрагментами своей страницы, а не
            # в начало документа: порядок страниц сохраняется.
            for sheet in sheets:
                page = sheet.position.page
                index = next((i for i, f in enumerate(result)
                              if f.position is not None and f.position.page >= page), len(result))
                result.insert(index, sheet)
        elif sheets:
            result[0:0] = sheets
        elif not any(f.type == "drawing" for f in result) and self._source_is_raster_drawing(metadata):
            for fragment in result:
                fragment.review_required = True
                fragment.warnings = list(dict.fromkeys(fragment.warnings + ["DRAWING_NOT_RECOGNIZED"]))
        if settings.DRAWING_TRANSCRIPT_FIXES_OCR:
            result = apply_transcripts(result)
        if settings.DRAWING_SUPPRESS_STALE_ON_FULL_READ:
            # После правки абзацев: исправленный абзац к этому моменту —
            # текст модели, и убирается как повтор, а не как расхождение.
            result = suppress_stale_sources(result)
        return result

    # -------------------------------------------------- лист-картинка целиком
    def _sheet_drawings(
        self, fragments: List[Fragment], metadata: Dict[str, Any]
    ) -> List[Fragment]:
        """
        Разбор листов целиком, когда документ — скан или фото чертежа.

        У скана PDF разбирается каждый чертёжный лист (по классификации
        лестницы), а не только файл как одна картинка: PDF как растр не
        открывается, и ветка молча вернула бы пустой результат.
        """
        full_read = self._full_read_image(metadata) or self._full_read_pdf(metadata)
        drawings = [fragment for fragment in fragments if fragment.type == "drawing"]
        drawn_pages = {int(f.position.page) for f in drawings if f.position is not None}
        # Нарезанные MinerU картинки листа — это виды, а не лист: штамп,
        # номер листа и примечания остаются за ними. При полном чтении
        # картинки лист читается целиком и в этом случае.
        only_undrawn = False
        if not full_read and drawings:
            # Листы без единого фрагмента drawing (повёрнутый лист, который
            # MinerU принял за таблицу) иначе не читались моделью вовсе.
            # Фрагмент без страницы не даёт сказать, какой лист уже прочитан.
            if (not settings.DRAWING_SHEET_MISSING_PAGES
                    or len(drawn_pages) == 0 or any(f.position is None for f in drawings)
                    or not self._source_is_raster_drawing(metadata)):
                return []
            only_undrawn = True
        if not full_read and not self._source_is_raster_drawing(metadata):
            return []

        source_uri = str(metadata.get("source_path") or "")
        if not source_uri:
            return []

        signals = (metadata.get("classification") or {}).get("signals") or {}
        pages = list(signals.get("raster_drawing_pages") or signals.get("drawing_pages") or [1])
        if only_undrawn:
            pages = [page for page in pages if int(page) not in drawn_pages]
            if not pages:
                return []
        sheets: List[Fragment] = []
        skipped = pages[settings.DRAWING_MAX_SHEETS:]
        for index, page in enumerate(pages[: settings.DRAWING_MAX_SHEETS]):
            left = deadline.remaining()
            if left is not None and left < settings.VLM_TIMEOUT:
                logger.warning("Разбор листа %s стр. %d пропущен: до срока задачи %.0f с",
                               source_uri, page, left)
                skipped = pages[index:]
                break
            try:
                drawing_result = self.process(source_uri, metadata, page, transcribe=full_read)
            except SoftTimeLimitExceeded:
                raise
            except Exception as exc:  # noqa: BLE001 — чертёж не должен ронять документ
                logger.error("Разбор листа %s стр. %d не удался: %s", source_uri, page, exc)
                continue
            anchor = Fragment(
                # Нулевой номер: лист целиком идёт перед своими фрагментами и не
                # сталкивается с нумерацией нормализатора, которая с единицы.
                fragment_id=f"{metadata.get('doc_id') or 'doc'}-frag-000"
                + ("" if int(page) == 1 else f"-{page}"),
                type="image",
                content=Content(image_ref=source_uri),
                position=Position(page=int(page), bbox=[0.0, 0.0, 1.0, 1.0], order=0),
                confidence=0.0, completeness=0.0,
                provenance=Provenance(method="vision_plus_vlm", strategy_level=7, source="drawing"),
            )
            sheet = self.build_fragment(drawing_result, anchor, metadata)
            if sheet is not None:
                sheets.append(sheet)
        if skipped and sheets:
            # Неразобранные листы не пропадают молча: пометка ведёт на проверку.
            note = "DRAWING_SHEETS_SKIPPED:" + ",".join(str(page) for page in skipped)
            sheets[0].review_required = True
            sheets[0].warnings = list(dict.fromkeys(sheets[0].warnings + [note]))
        return sheets

    def _full_read_image(self, metadata: Dict[str, Any]) -> bool:
        """
        Лист-картинка читается целиком со всем текстом
        (DRAWING_IMAGE_FULL_READ): по классификации это растровый чертёж,
        либо лестница назвала лист текстом, но растровый детектор видит на
        нём рамку формата (лист со штампом, где MinerU нашёл только
        таблицу: таблица и примечания дают много строк, и детектор зовёт
        такой лист страницей текста, но рамка у него есть).
        """
        if not settings.DRAWING_IMAGE_FULL_READ:
            return False
        if not filetypes.is_image(str(metadata.get("file_type") or "")):
            return False
        if self._source_is_raster_drawing(metadata):
            return True
        data = self._read(str(metadata.get("source_path") or ""))
        if not data:
            return False
        verdict = raster_drawing.analyse(data)
        return verdict.is_drawing or bool(verdict.signals.get("frame"))

    def _full_read_pdf(self, metadata: Dict[str, Any]) -> bool:
        """
        Скан чертежа в PDF читается целиком со всем текстом листа
        (DRAWING_PDF_FULL_READ): вырезки видов, которые нарезал MinerU, не
        несут ни штампа, ни технических требований, ни размеров вне вырезок.
        """
        if not settings.DRAWING_PDF_FULL_READ:
            return False
        if str(metadata.get("file_type") or "").lower().lstrip(".") != "pdf":
            return False
        return self._source_is_raster_drawing(metadata)

    def _source_is_raster_drawing(self, metadata: Dict[str, Any]) -> bool:
        """Растровый чертёж по классификации лестницы, иначе — по самому файлу."""
        classification = metadata.get("classification") or {}
        genre = classification.get("genre")
        if genre:
            return genre == GENRE_DRAWING and classification.get("source") == SOURCE_RASTER

        if not filetypes.is_image(str(metadata.get("file_type") or "")):
            return False
        data = self._read(str(metadata.get("source_path") or ""))
        if not data:
            return False
        return raster_drawing.analyse(data).is_drawing

    def _transcribe(self, image, payload, missing_regions, started) -> Optional[List[str]]:
        """
        Весь текст листа. С DRAWING_TRANSCRIBE_TILES плотный лист, текст
        которого не помещается в один ответ, дочитывается частями; часть,
        которую прочитать не удалось, — непрочитанная область листа, и лист
        не считается переписанным целиком (core/drawing_quality.py).
        """
        if not settings.DRAWING_TRANSCRIBE_TILES:
            return self._fidelity(self.qwen.transcribe(image))
        budget = settings.DRAWING_TIME_BUDGET
        transcript = self.qwen.transcribe_sheet(image, until=started + budget if budget else None)
        if transcript is None:
            return None
        if transcript.tiles:
            payload["transcript_tiles"] = transcript.tiles
        if not transcript.complete:
            payload["transcript_complete"] = False
            missing_regions.extend(
                {"kind": "transcript_tile", "bbox": box, "reason": reason}
                for box, reason in transcript.failed
            )
        return self._fidelity(transcript.lines)

    @staticmethod
    def _fidelity(lines: Optional[List[str]]) -> Optional[List[str]]:
        """Строки транскрипции с восстановленными числами и знаками."""
        if lines is None or not settings.DRAWING_DIMENSION_FIDELITY:
            return lines
        return dimension_fidelity.repair_lines(lines)

    def _out_of_time(self, started: float) -> bool:
        budget = settings.DRAWING_TIME_BUDGET
        return bool(budget) and (time.monotonic() - started) > budget

    # ------------------------------------------------------------ чтение
    def _read_title_block(
        self,
        layout: drawing_vision.SheetLayout,
        region: drawing_vision.Region,
        fields: List[Dict[str, Any]],
        payload: Dict[str, Any],
    ) -> bool:
        """Штамп: графы превращаются в поля, таблица целиком — в payload."""
        if settings.DRAWING_TITLE_BLOCK_PASS:
            return self._read_title_block_fields(layout, region, fields, payload)
        piece = drawing_vision.crop(layout.image, region.bbox)
        block = self.qwen.read_title_block(piece)
        if block is None:
            logger.warning("Штамп не прочитан")
            return False

        payload["title_block"] = block.get("fields", {})
        if block.get("rows"):
            payload["tables"].append({
                "kind": drawing_vision.KIND_TITLE_BLOCK,
                "bbox": region.bbox,
                "rows": block["rows"],
            })

        for name, value in (block.get("fields") or {}).items():
            text = drawing_text.normalize(str(value))
            if settings.DRAWING_DIMENSION_FIDELITY and "масштаб" in str(name).lower():
                # «7:1» в графе «Масштаб» — это «1:1»: курсивная «1» похожа
                # на «7», а ряд масштабов по ГОСТ 2.302 узкий.
                text = dimension_fidelity.repair_scale(text)
            readable, _ = drawing_text.looks_like_text(text)
            if not readable:
                continue
            category = "material" if _is_material(name, text) else "title_block"
            fields.append(_field(
                category=category,
                text=text,
                bbox=region.bbox,
                confidence=_FROM_REGION,
                label=str(name),
            ))
        return True

    def _read_title_block_fields(
        self,
        layout: drawing_vision.SheetLayout,
        region: drawing_vision.Region,
        fields: List[Dict[str, Any]],
        payload: Dict[str, Any],
    ) -> bool:
        """
        Отдельный проход по штампу (DRAWING_TITLE_BLOCK_PASS).

        Два отличия от обычного чтения области. Вырезка штампа идёт в
        модель в своём, высоком разрешении: на ужатой вырезке путаются
        фамилии и код документа. И колонка дополнительных граф у левого
        поля читается отдельным запросом — в вырезку штампа она не входит
        никогда, а «Инв. № подл.» и «Перв. примен.» лежат именно там.
        """
        bbox = drawing_vision.title_block_crop(layout, region)
        payload["title_block_bbox"] = bbox
        piece = drawing_vision.crop(
            layout.image, bbox,
            upscale=settings.QWEN_TITLE_BLOCK_UPSCALE,
            max_side=settings.QWEN_TITLE_BLOCK_MAX_SIDE,
        )
        parsed = self.qwen.read_title_block_fields(piece)
        side = self._read_side_column(layout, payload)
        if parsed is None and side is None:
            logger.warning("Штамп не прочитан")
            return False
        parsed = _merge_blocks(parsed, side)
        if settings.DRAWING_DIMENSION_FIDELITY:
            # «7:1» в графе «Масштаб» — это «1:1»: курсивная «1» похожа на
            # «7», а ряд масштабов по ГОСТ 2.302 узкий. Правка делается до
            # payload, чтобы в structured_payload и в полях стояло одно и
            # то же значение, а запись MinerU с тем же «7:1» не оказалась
            # повтором исправленного прочтения.
            _repair_scale_graph(parsed)

        payload["title_block"] = dict(parsed.get("fields") or {})
        handwritten = dict(parsed.get("handwritten") or {})
        if handwritten:
            payload["title_block_handwritten"] = handwritten
        if parsed.get("rows"):
            payload["tables"].append({
                "kind": drawing_vision.KIND_TITLE_BLOCK,
                "bbox": region.bbox,
                "rows": parsed["rows"],
            })

        for name, value, by_hand in title_block.entries(parsed):
            text = drawing_text.normalize(value)
            readable, _ = drawing_text.looks_like_text(text)
            if not readable:
                continue
            category = "material" if _is_material(name, text) else "title_block"
            fields.append(_field(
                category=category,
                text=title_block.mark(text, by_hand),
                bbox=region.bbox,
                # Рукописное прочитано с той же вырезки, но проверить его
                # правилами нечем: уверенность ниже печатной графы.
                confidence=_HANDWRITTEN if by_hand else _FROM_REGION,
                label=name,
            ))
        return True

    def _read_side_column(
        self, layout: drawing_vision.SheetLayout, payload: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Колонка дополнительных граф у левого поля, повёрнутая под чтение."""
        if not settings.DRAWING_SIDE_COLUMN_READ:
            return None
        bbox = drawing_vision.side_column(layout)
        if bbox is None:
            return None
        piece = drawing_vision.upright(drawing_vision.crop(
            layout.image, bbox, upscale=settings.QWEN_TITLE_BLOCK_UPSCALE,
            padding=0.0, max_side=settings.QWEN_TITLE_BLOCK_MAX_SIDE,
        ))
        parsed = self.qwen.read_side_column(piece)
        payload["side_column"] = {"bbox": bbox, "read": parsed is not None}
        return parsed

    def _read_table(
        self, piece, region: drawing_vision.Region, payload: Dict[str, Any]
    ) -> bool:
        """
        Таблица со листа (спецификация, ведомость) уходит в payload целиком.
        Полей вида «строка спецификации» в контракте нет, и раскладывать её
        по `note` значило бы засорять индекс мусором.
        """
        table = self.qwen.read_table(piece)
        if table is None or not table.get("rows"):
            return False
        payload["tables"].append({
            "kind": drawing_vision.KIND_TABLE,
            "bbox": region.bbox,
            "title": table.get("title", ""),
            "headers": table.get("headers", []),
            "rows": table["rows"],
        })
        return True

    def _fields_from_annotations(
        self, annotations: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Надписи, прочитанные моделью на листе целиком -> поля контракта."""
        fields: List[Dict[str, Any]] = []
        for annotation in annotations:
            text = drawing_text.normalize(str(annotation.get("value") or ""))
            readable, reason = drawing_text.looks_like_text(text)
            if not readable:
                logger.debug("Надпись отброшена (%s): %r", reason, text)
                continue

            claimed = str(annotation.get("category") or "").strip().lower()
            by_rules, _ = vector_drawing.classify(text)
            category, confidence = _reconcile(claimed, by_rules)
            fields.append(_field(
                category=category,
                text=text,
                bbox=_bbox(annotation.get("bbox")),
                confidence=confidence,
            ))
        return fields

    # ------------------------------------------------------------ хранилище
    def _open(self, image_ref: str, page: int = 1):
        data = self._read(image_ref)
        if not data:
            return None
        if data[:5] == b"%PDF-":
            # Скан чертежа в PDF: лист рендерится в растр, иначе модель
            # получала бы «картинку», которая не открывается.
            return _render_pdf_page(data, page)
        image = raster_drawing.open_image(data)
        if image is None:
            logger.error("Чертёж %s не открылся как растр", image_ref)
        return image

    def _read(self, uri: str) -> bytes:
        try:
            return self.storage.read_bytes(uri)
        except Exception as exc:  # noqa: BLE001
            logger.error("Не удалось прочитать %s: %s", uri, exc)
            return b""


# ===========================================================================
# Поля
# ===========================================================================

def _is_text_document(fragments: List[Fragment], metadata: Dict[str, Any]) -> bool:
    """
    Текстовый документ (договор, ТЗ, ТУ, письмо, протокол), а не чертёж:
    лестница не назвала его чертежом, и текста в нём в среднем не меньше
    `_TEXT_DOC_MIN_CHARS_PER_PAGE` символов на страницу.
    """
    if (metadata.get("classification") or {}).get("genre") == GENRE_DRAWING:
        return False
    chars, pages = 0, set()
    for fragment in fragments:
        if fragment.position is not None:
            pages.add(fragment.position.page)
        if fragment.type in ("text", "table"):
            chars += _visible_chars(fragment.content.text or "")
            table = fragment.content.table_data
            if table is not None:
                # У табличного фрагмента текста нет — всё в ячейках.
                cells = list(table.headers) + [cell for row in table.rows for cell in row]
                chars += sum(_visible_chars(str(cell)) for cell in cells if cell is not None)
    return bool(pages) and chars / len(pages) >= _TEXT_DOC_MIN_CHARS_PER_PAGE


def _visible_chars(text: str) -> int:
    return len("".join(text.split()))


def _is_small_picture(fragment: Fragment) -> bool:
    """Картинка меньше `_TEXT_DOC_MIN_FIGURE_AREA` листа (bbox в долях листа)."""
    bbox = fragment.position.bbox if fragment.position is not None else None
    if not bbox or len(bbox) != 4:
        return False
    x0, y0, x1, y1 = (float(v) for v in bbox)
    if max(x0, y0, x1, y1) > 1.0 or min(x0, y0, x1, y1) < 0.0:
        return False      # не в долях листа — размер неизвестен
    return abs(x1 - x0) * abs(y1 - y0) < _TEXT_DOC_MIN_FIGURE_AREA


def _render_pdf_page(data: bytes, page: int):
    """Страница PDF как растр в рабочем разрешении чертежей."""
    try:
        import fitz
        from PIL import Image
    except ImportError:  # pragma: no cover
        return None
    try:
        document = fitz.open(stream=data, filetype="pdf")
    except Exception as exc:  # noqa: BLE001
        logger.error("PDF чертежа не открылся: %s", exc)
        return None
    try:
        if not 1 <= page <= document.page_count:
            return None
        pixmap = document[page - 1].get_pixmap(dpi=settings.DRAWING_RENDER_DPI)
        return Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
    finally:
        document.close()


def _field(
    category: str,
    text: str,
    bbox: List[float],
    confidence: float,
    label: str = "",
) -> Dict[str, Any]:
    """Поле контракта: значение от модели, разметка — правилами ГОСТ."""
    value = f"{label}: {text}" if label else text
    # Марка стали — не посадка: в «12Х18Н10Т» правило посадок видит «H10».
    no_tolerance = settings.DRAWING_MATERIAL_NO_TOLERANCE and category == "material"
    return {
        "category": category,
        "value": value,
        "tolerance": None if no_tolerance else vector_drawing.extract_tolerance(text),
        "nature": vector_drawing.determine_nature(text),
        "unit": vector_drawing.extract_unit(text, category),
        "confidence": round(max(0.0, min(1.0, confidence)), 3),
        "bbox": bbox,
        "source_of_tolerance": "unknown" if no_tolerance else vector_drawing.tolerance_source(text),
        # Надпись прочитана с самого листа. Значений контракта это не
        # расширяет: ровно то же ставит ветка детектора.
        "provenance": "detected_from_annotation",
    }


def _reconcile(claimed: str, by_rules: str) -> tuple:
    """
    Категория поля: модель против правил. Правила детерминированы и знают
    ГОСТ, модель видит лист — поэтому спор решается не в чью-то пользу, а
    понижением уверенности.
    """
    if claimed not in CATEGORIES:
        return by_rules, _MODEL_ONLY if by_rules != "note" else 0.4
    if claimed == by_rules:
        return claimed, _AGREED
    if by_rules == "note":
        # Правила такой категории просто не знают: номер позиции, графа
        # штампа. Это не спор.
        return claimed, _MODEL_ONLY
    return claimed, _DISAGREED


def _bbox(raw: Any) -> List[float]:
    """Координаты от модели: проверяются, а не принимаются на веру."""
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        return [0.0, 0.0, 1.0, 1.0]
    try:
        coords = [float(c) for c in raw]
    except (TypeError, ValueError):
        return [0.0, 0.0, 1.0, 1.0]
    if max(coords) > 1.5:            # модель ответила пикселями — не угадываем
        return [0.0, 0.0, 1.0, 1.0]
    clamp = lambda c: max(0.0, min(1.0, c))  # noqa: E731
    x1, y1, x2, y2 = (clamp(c) for c in coords)
    if x2 <= x1 or y2 <= y1:
        return [0.0, 0.0, 1.0, 1.0]
    return [round(x1, 4), round(y1, 4), round(x2, 4), round(y2, 4)]


def _deduplicate(fields: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Одна надпись, прочитанная и на листе, и в области, — одно поле.
    Остаётся то, у которого выше уверенность.
    """
    best: Dict[tuple, Dict[str, Any]] = {}
    order: List[tuple] = []
    for field in fields:
        key = (field["category"], field["value"].strip().lower())
        if key not in best:
            best[key] = field
            order.append(key)
        elif field["confidence"] > best[key]["confidence"]:
            best[key] = field
    return [best[key] for key in order]


def _merge_blocks(
    block: Optional[Dict[str, Any]], side: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Графы штампа и графы боковой колонки в один разбор. Штамп главнее:
    одну и ту же графу (обозначение бывает и там и там) он переопределяет,
    а боковая колонка добавляет то, чего в штампе нет.
    """
    merged: Dict[str, Any] = {"fields": {}, "handwritten": {}, "unknown": {}, "rows": []}
    for part in (side, block):
        if not part:
            continue
        for key in ("fields", "handwritten", "unknown"):
            merged[key].update(part.get(key) or {})
        merged["rows"] = part.get("rows") or merged["rows"]
    # Одно и то же значение, названное и печатным, и рукописным, остаётся
    # одним — рукописным. Разные значения сохраняются оба: выбрать между
    # «1085-02» из штампа и «1085» из боковой колонки нечем, а потерять
    # лучшее из двух хуже, чем отдать оба.
    for key, value in list(merged["fields"].items()):
        if merged["handwritten"].get(key, "").strip() == value.strip():
            merged["fields"].pop(key)
    return merged


def _repair_scale_graph(parsed: Dict[str, Any]) -> None:
    """
    Значение графы «Масштаб» в разборе штампа — приводится к ряду
    ГОСТ 2.302 (`dimension_fidelity.repair_scale`). Правится только эта
    графа: отношение в любом другом месте листа масштабом не является.
    """
    for part_name in ("fields", "handwritten"):
        part = parsed.get(part_name) or {}
        value = part.get(title_block.SCALE)
        if isinstance(value, str) and value:
            part[title_block.SCALE] = dimension_fidelity.repair_scale(value)


def _is_material(name: str, text: str) -> bool:
    lowered = f"{name} {text}".lower()
    return "материал" in lowered or vector_drawing.classify(text)[0] == "material"


def _empty(mode: str) -> Dict[str, Any]:
    return {
        "detected_count": 0,
        "parsed_fields": [],
        "avg_confidence": 0.0,
        "completeness": None,
        "structure_score": 0.0,
        "missing_regions": [],
        "warnings": ["DRAWING_NOT_RECOGNIZED"],
        "completeness_mechanisms": {},
        "mode": mode,
        "payload": {},
    }
