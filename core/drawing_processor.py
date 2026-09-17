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
from core.config import settings
from core.models.contract import Content, DrawingField, Fragment, Position, Provenance
from core.providers import drawing_text, drawing_vision, raster_drawing, vector_drawing
from core.providers.qwen_client import CATEGORIES, QwenVisionClient
from core.providers.storage import StorageProvider, StorageProviderFactory
from core.providers.vector_drawing import structure_ratio

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
    def process(self, image_ref: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
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

        image = self._open(image_ref)
        if image is None:
            return _empty(MODE_NONE)

        started = time.monotonic()
        layout = (
            drawing_vision.analyse(image) if self.vision_enabled
            else drawing_vision.plain(image)
        )

        fields: List[Dict[str, Any]] = []
        payload: Dict[str, Any] = {"layout": layout.as_dict(), "tables": []}

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
                continue
            piece = drawing_vision.crop(layout.image, region.bbox)
            if region.kind == drawing_vision.KIND_TITLE_BLOCK:
                regions_read += int(self._read_title_block(piece, region, fields, payload))
            else:
                regions_read += int(self._read_table(piece, region, payload))
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
            # Полнота — доля полей, которым нашлась категория. Связей «размер
            # — к какому элементу» ветка по-прежнему не строит, и полнотой
            # это не считается.
            "completeness": structure_ratio(fields),
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
        return Fragment(
            fragment_id=source.fragment_id,
            type="drawing",
            content=Content(
                image_ref=source.content.image_ref,
                structured_drawing_fields=[DrawingField(**f) for f in fields],
                structured_payload=payload,
            ),
            position=source.position or Position(page=1, bbox=[0, 0, 1, 1]),
            section_title=source.section_title,
            confidence=round(float(drawing_result.get("avg_confidence", 0.0)), 3),
            completeness=round(float(drawing_result.get("completeness", 0.0)), 3),
            provenance=Provenance(method=method, strategy_level=level, source="drawing"),
            graph_nodes=list(source.graph_nodes),
            relations=list(source.relations),
        )

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
            try:
                drawing_result = self.process(fragment.content.image_ref, metadata)
                enriched = self.build_fragment(drawing_result, fragment, metadata)
            except Exception as exc:  # noqa: BLE001 — чертёж не должен ронять документ
                logger.error("Обработка чертежа %s не удалась: %s",
                             fragment.content.image_ref, exc)
                enriched = None
            result.append(enriched or fragment)

        sheet = self._sheet_drawing(result, metadata)
        if sheet is not None:
            result.insert(0, sheet)
        return result

    # -------------------------------------------------- лист-картинка целиком
    def _sheet_drawing(
        self, fragments: List[Fragment], metadata: Dict[str, Any]
    ) -> Optional[Fragment]:
        """Разбор всего листа, когда документ — это скан или фото чертежа."""
        if any(fragment.type == "drawing" for fragment in fragments):
            return None
        if not self._source_is_raster_drawing(metadata):
            return None

        source_uri = str(metadata.get("source_path") or "")
        if not source_uri:
            return None

        try:
            drawing_result = self.process(source_uri, metadata)
        except Exception as exc:  # noqa: BLE001 — чертёж не должен ронять документ
            logger.error("Разбор листа %s не удался: %s", source_uri, exc)
            return None

        anchor = Fragment(
            # Нулевой номер: лист целиком идёт перед своими фрагментами и не
            # сталкивается с нумерацией нормализатора, которая с единицы.
            fragment_id=f"{metadata.get('doc_id') or 'doc'}-frag-000",
            type="image",
            content=Content(image_ref=source_uri),
            position=Position(page=1, bbox=[0.0, 0.0, 1.0, 1.0], order=0),
            confidence=0.0, completeness=0.0,
            provenance=Provenance(method="vision_plus_vlm", strategy_level=7, source="drawing"),
        )
        return self.build_fragment(drawing_result, anchor, metadata)

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
    def _open(self, image_ref: str):
        data = self._read(image_ref)
        if not data:
            return None
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
        "completeness": 0.0,
        "mode": mode,
        "payload": {},
    }
