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
   доезжал до сплошного распознавания. Сервисы и каркас дообучения в
   `training/` оставлены: появятся веса под машиностроительный чертёж —
   детектор вернётся как источник областей вместо геометрии.

5. **Таблица составных частей не помещается в контракт.** Полей вида
   «строка спецификации» в нём нет, поэтому таблицы уходят в
   `structured_payload.tables` целиком, а в поля — только то, для чего
   категория предусмотрена.
"""

import logging
import time
from typing import Any, Dict, List, Optional

from core import filetypes
from celery.exceptions import SoftTimeLimitExceeded

from core import deadline
from core.config import settings
from core.models.contract import Content, DrawingField, Fragment, Position, Provenance
from core.providers import drawing_text, drawing_vision, raster_drawing, vector_drawing
from core.providers.qwen_client import CATEGORIES, QwenVisionClient
from core.providers.storage import StorageProvider, StorageProviderFactory
from core.providers.vector_drawing import structure_ratio
from core.drawing_quality import apply_drawing_quality

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
        self, image_ref: str, metadata: Dict[str, Any], page: int = 1
    ) -> Dict[str, Any]:
        """
        Возвращает:
        {'detected_count', 'parsed_fields', 'avg_confidence', 'completeness',
         'mode', 'payload'}
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
        layout = (
            drawing_vision.analyse(image) if self.vision_enabled
            else drawing_vision.plain(image)
        )

        fields: List[Dict[str, Any]] = []
        payload: Dict[str, Any] = {"layout": layout.as_dict(), "tables": []}
        payload["coordinate_reference"] = "deskewed_page"
        missing_regions: List[Dict[str, Any]] = []

        reading = self.qwen.read_sheet(layout.image, hint=layout.hint())
        if reading is None:
            logger.error("Модель не прочитала лист %s", image_ref)
            return _empty(MODE_NONE)
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
            piece = drawing_vision.crop(layout.image, region.bbox)
            if region.kind == drawing_vision.KIND_TITLE_BLOCK:
                read = self._read_title_block(piece, region, fields, payload)
            else:
                read = self._read_table(piece, region, payload)
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
        mode = MODE_VISION_VLM if (self.vision_enabled and layout.regions) else MODE_SHEET_ONLY
        if not fields and not payload["tables"]:
            return _empty(MODE_NONE)

        payload["regions_read"] = regions_read
        payload["model"] = self.qwen.model
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

    def build_fragment(
        self,
        drawing_result: Dict[str, Any],
        source: Fragment,
        metadata: Dict[str, Any],
    ) -> Optional[Fragment]:
        """Строит фрагмент типа `drawing` на месте фрагмента-изображения."""
        fields = drawing_result.get("parsed_fields") or []
        payload = dict(drawing_result.get("payload") or {})
        if not fields and not payload.get("tables"):
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
        for fragment in fragments:
            # Векторный чертёж уже разобран парсером по текстовому слою —
            # там точные значения и углы, читать нечего.
            if fragment.type == "drawing":
                result.append(fragment)
                continue
            if fragment.type != "image" or not fragment.content.image_ref:
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
                drawing_result = self.process(fragment.content.image_ref, metadata)
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

        sheets = self._sheet_drawings(result, metadata)
        if sheets:
            result[0:0] = sheets
        elif not any(f.type == "drawing" for f in result) and self._source_is_raster_drawing(metadata):
            for fragment in result:
                fragment.review_required = True
                fragment.warnings = list(dict.fromkeys(fragment.warnings + ["DRAWING_NOT_RECOGNIZED"]))
        return result

    # -------------------------------------------------- лист-картинка целиком
    def _sheet_drawings(
        self, fragments: List[Fragment], metadata: Dict[str, Any]
    ) -> List[Fragment]:
        """
        Разбор листов целиком, когда документ — скан или фото чертежа.

        У скана PDF разбирается каждый чертёжный лист (по классификации
        лестницы), а не только файл как одна картинка: раньше PDF не
        открывался как растр, и ветка молча возвращала пустой результат.
        """
        if any(fragment.type == "drawing" for fragment in fragments):
            return []
        if not self._source_is_raster_drawing(metadata):
            return []

        source_uri = str(metadata.get("source_path") or "")
        if not source_uri:
            return []

        signals = (metadata.get("classification") or {}).get("signals") or {}
        pages = list(signals.get("raster_drawing_pages") or signals.get("drawing_pages") or [1])
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
                drawing_result = self.process(source_uri, metadata, page)
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

    def _out_of_time(self, started: float) -> bool:
        budget = settings.DRAWING_TIME_BUDGET
        return bool(budget) and (time.monotonic() - started) > budget

    # ------------------------------------------------------------ чтение
    def _read_title_block(
        self,
        piece,
        region: drawing_vision.Region,
        fields: List[Dict[str, Any]],
        payload: Dict[str, Any],
    ) -> bool:
        """Штамп: графы превращаются в поля, таблица целиком — в payload."""
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
    return {
        "category": category,
        "value": value,
        "tolerance": vector_drawing.extract_tolerance(text),
        "nature": vector_drawing.determine_nature(text),
        "unit": vector_drawing.extract_unit(text, category),
        "confidence": round(max(0.0, min(1.0, confidence)), 3),
        "bbox": bbox,
        "source_of_tolerance": vector_drawing.tolerance_source(text),
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
