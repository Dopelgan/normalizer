"""
Структура листа, этапы S0–S1: рамка страницы и общий пул прочтений.

Всё, что проходы уже прочитали на одной странице, сводится в один список
кандидатов `Candidate` в координатах страницы: поля фрагмента листа целиком,
поля вырезок видов MinerU (их рамки — доли вырезки, здесь они переводятся в
доли страницы), поля текстового слоя векторного PDF, сводные графы штампа,
таблицы, строки прохода «весь текст листа» и строки распознавания
(`position.lines`, флаг RECOGNITION_LINES), если они есть.

Перевод рамок вырезки — тот же, что у размещения полей
(`core/providers/field_placement.py`, DRAWING_FIELD_PLACEMENT_V2): если
фрагмент уже размещён на странице (`coordinate_reference == "page"`), рамки
берутся как есть; иначе — помощником размещения, а без него (ветка размещения
ещё не в дереве) — переносом рамки вырезки без учёта поворота.

Вызовов модели нет: только то, что уже прочитано.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from core.models.contract import Fragment

# Источник значения (`provenance.source` элемента).
SOURCE_VECTOR = "vector"
SOURCE_TEXT_LAYER = "text_layer"
SOURCE_OCR = "ocr"
SOURCE_VLM = "vlm"

# Проходы (`provenance.passes`).
PASS_SHEET = "sheet"
PASS_TRANSCRIBE = "transcribe"
PASS_TITLE_BLOCK = "title_block"
PASS_SIDE_COLUMN = "side_column"
PASS_TABLE = "table"
PASS_VIEW_CROP = "view_crop"
PASS_REREAD = "reread"
PASS_LINKS = "links"

WHOLE = [0.0, 0.0, 1.0, 1.0]
# Рамка больше этой доли страницы — область, где надпись прочитана, а не её место.
REGION_AREA = 0.25
REGION_SIDE = 0.5

# Источник прочтения D3 (`DrawingField.readings[].source`) -> (источник, проход).
_READING_SOURCES = {
    "sheet": (SOURCE_VLM, PASS_SHEET),
    "crop": (SOURCE_VLM, PASS_VIEW_CROP),
    "transcript": (SOURCE_VLM, PASS_TRANSCRIBE),
    "ocr": (SOURCE_OCR, PASS_SHEET),
    "vector": (SOURCE_VECTOR, PASS_SHEET),
    "crop_reread": (SOURCE_VLM, PASS_REREAD),
}

_LABELLED = re.compile(r"^\s*([A-Za-zА-Яа-яЁё][^:]{0,40}?)\s*:\s*(.+)$", re.S)
_HANDWRITTEN = re.compile(r"\s*\((?:рукописно|от руки)\)\s*$", re.I)


@dataclass
class Candidate:
    """Одно прочтение надписи одним проходом, в координатах страницы."""

    value: str
    category: str
    bbox: List[float]
    source: str
    pass_: str
    confidence: float = 0.0
    # Мост к плоским полям (`FieldRef`): (порядковый номер фрагмента страницы
    # среди фрагментов с `structured_drawing_fields`, индекс поля в нём);
    # `None` — прочтение не из поля.
    field_ref: Optional[Tuple[int, int]] = None
    # Графа штампа «Метка: значение» -> метка как прочитана.
    label: str = ""
    handwritten: bool = False
    tolerance: Optional[str] = None
    nature: str = "executive"
    agreement: Optional[str] = None
    confirmed: bool = False
    # Связь размера (`DrawingField.link`) в координатах страницы.
    link: Optional[Dict[str, Any]] = None
    # Другие прочтения той же надписи, уже известные проходам (D3 `readings`,
    # `alternatives` размещения полей): [{value, source, pass, confidence}].
    readings: List[Dict[str, Any]] = field(default_factory=list)
    # Рамка стоит на самой надписи (вектор, размещение N0 с точной рамкой).
    exact: bool = False

    @property
    def placed(self) -> bool:
        """Рамка — место надписи, а не область прочтения."""
        return is_place(self.bbox)


@dataclass
class PagePool:
    """Пул прочтений одной страницы и всё, что нужно следующим этапам."""

    page: int
    candidates: List[Candidate] = field(default_factory=list)
    # Строки с рамками для привязки (S3): текстовый слой и распознавание.
    lines: List[Dict[str, Any]] = field(default_factory=list)
    # Строки прохода «весь текст листа» (без рамок).
    transcript: List[str] = field(default_factory=list)
    tables: List[Dict[str, Any]] = field(default_factory=list)
    title_fields: Dict[str, str] = field(default_factory=dict)
    title_handwritten: Dict[str, str] = field(default_factory=dict)
    title_rows: List[List[str]] = field(default_factory=list)
    title_bbox: Optional[List[float]] = None
    side_bbox: Optional[List[float]] = None
    # Рамки вырезок видов на странице: кандидаты в области видов (S5).
    crop_frames: List[List[float]] = field(default_factory=list)
    # Виды из `value_links` (DRAWING_VALUE_VIEW_LINKS), если были.
    value_views: List[Dict[str, Any]] = field(default_factory=list)
    frame: Dict[str, Any] = field(default_factory=dict)
    sheet_type: str = "unknown"
    total_fields: int = 0


# ===========================================================================
# Геометрия
# ===========================================================================

def area(bbox: Sequence[float]) -> float:
    if not bbox or len(bbox) != 4:
        return 0.0
    return max(0.0, float(bbox[2]) - float(bbox[0])) * max(0.0, float(bbox[3]) - float(bbox[1]))


def is_place(bbox: Sequence[float]) -> bool:
    """Рамка указывает на надпись: не весь лист и не большая область."""
    if not bbox or len(bbox) != 4:
        return False
    x1, y1, x2, y2 = (float(v) for v in bbox)
    if [x1, y1, x2, y2] == WHOLE or x2 <= x1 or y2 <= y1:
        return False
    return (x2 - x1) <= REGION_SIDE and (y2 - y1) <= REGION_SIDE and area(bbox) <= REGION_AREA


def centre(bbox: Sequence[float]) -> Tuple[float, float]:
    return (float(bbox[0]) + float(bbox[2])) / 2, (float(bbox[1]) + float(bbox[3])) / 2


def clamp_box(bbox: Sequence[float]) -> List[float]:
    out = [round(max(0.0, min(1.0, float(v))), 4) for v in bbox]
    if out[2] < out[0]:
        out[0], out[2] = out[2], out[0]
    if out[3] < out[1]:
        out[1], out[3] = out[3], out[1]
    return out


def _frame_to_page(bbox: Sequence[float], frame: Sequence[float], layout: Dict[str, Any]) -> List[float]:
    """
    Рамка в долях выправленной картинки фрагмента -> доли страницы.

    Помощник размещения полей учитывает поворот и наклон картинки; пока его
    ветки нет в дереве, рамка просто переносится в рамку вырезки.
    """
    try:
        from core.providers import field_placement  # ветка DRAWING_FIELD_PLACEMENT_V2
    except ImportError:
        field_placement = None
    if field_placement is not None:
        size = layout.get("source_size")
        size = size if isinstance(size, (list, tuple)) and len(size) == 2 and min(size) > 0 else None
        return field_placement.box_to_page(
            bbox, frame, float(layout.get("angle") or 0.0), int(layout.get("turn") or 0), size)
    fx1, fy1, fx2, fy2 = (float(v) for v in frame)
    w, h = fx2 - fx1, fy2 - fy1
    x1, y1, x2, y2 = (float(v) for v in bbox)
    return clamp_box([fx1 + x1 * w, fy1 + y1 * h, fx1 + x2 * w, fy1 + y2 * h])


def _point_to_page(point: Sequence[float], frame: Sequence[float]) -> List[float]:
    fx1, fy1, fx2, fy2 = (float(v) for v in frame)
    return [round(fx1 + float(point[0]) * (fx2 - fx1), 4), round(fy1 + float(point[1]) * (fy2 - fy1), 4)]


# ===========================================================================
# Фрагменты страницы
# ===========================================================================

def is_vector(fragment: Fragment) -> bool:
    return fragment.provenance.method == "vector_text_layer"


def is_whole_sheet(fragment: Fragment) -> bool:
    """Фрагмент листа целиком (как у `reading_reconcile._is_whole_sheet`)."""
    bbox = list(fragment.position.bbox or []) if fragment.position else []
    payload = fragment.content.structured_payload or {}
    return [float(v) for v in bbox] == WHOLE or bool(payload.get("transcript_line_count"))


def page_of(fragment: Fragment) -> Optional[int]:
    if fragment.position is None or fragment.position.page is None:
        return None
    return int(fragment.position.page)


def whole_sheet(drawings: Sequence[Fragment]) -> Optional[Fragment]:
    """Фрагмент листа целиком: носитель структуры. Самый населённый из таких."""
    model = [f for f in drawings if not is_vector(f)]
    whole = [f for f in model if is_whole_sheet(f)]
    pool = whole or model or list(drawings)
    if not pool:
        return None
    return max(pool, key=lambda f: len(f.content.structured_drawing_fields or []))


def _on_page(fragment: Fragment) -> bool:
    """Рамки полей уже в долях страницы: вектор или размещение N0."""
    payload = fragment.content.structured_payload or {}
    return is_vector(fragment) or payload.get("coordinate_reference") == "page"


def _mapper(fragment: Fragment, carrier: bool):
    """Функция «рамка фрагмента -> рамка страницы» и точка -> точка."""
    payload = fragment.content.structured_payload or {}
    layout = payload.get("layout") if isinstance(payload.get("layout"), dict) else {}
    frame = list(fragment.position.bbox) if fragment.position and fragment.position.bbox else WHOLE
    frame = [float(v) for v in frame]
    if _on_page(fragment) or (carrier and frame == WHOLE):
        return (lambda b: clamp_box(b)), (lambda p: [round(float(p[0]), 4), round(float(p[1]), 4)])
    return (lambda b: _frame_to_page(b, frame, layout)), (lambda p: _point_to_page(p, frame))


def _split_label(value: str) -> Tuple[str, str]:
    match = _LABELLED.match(value or "")
    if not match:
        return "", value
    return match.group(1).strip(), match.group(2).strip()


def _pass_of(fragment: Fragment, carrier: bool, bbox: Sequence[float], category: str,
             labelled: bool, payload: Dict[str, Any]) -> str:
    if not carrier:
        return PASS_VIEW_CROP
    if category in ("title_block", "material") and labelled:
        side = payload.get("side_column") or {}
        if side.get("bbox") and _same_box(bbox, side["bbox"]):
            return PASS_SIDE_COLUMN
        return PASS_TITLE_BLOCK
    return PASS_SHEET


def _same_box(a: Sequence[float], b: Sequence[float]) -> bool:
    return len(a) == 4 and len(b) == 4 and all(abs(float(x) - float(y)) < 1e-3 for x, y in zip(a, b))


def collect(drawings: Sequence[Fragment], others: Sequence[Fragment] = ()) -> PagePool:
    """
    S0–S1: пул прочтений страницы. `drawings` — чертёжные фрагменты страницы
    в порядке документа (по нему считаются `field_refs`), `others` —
    прочие фрагменты той же страницы (текст и таблицы: строки распознавания).
    """
    carrier = whole_sheet(drawings)
    page = page_of(carrier) if carrier is not None else None
    pool = PagePool(page=page or 0)
    if carrier is not None:
        payload = carrier.content.structured_payload or {}
        layout = payload.get("layout") if isinstance(payload.get("layout"), dict) else {}
        pool.frame = {
            "coordinate_reference": "deskewed_page",
            "angle": float(layout.get("angle") or 0.0),
            "turn": int(layout.get("turn") or 0),
            "inner_frame": layout.get("frame"),
            "page_px": layout.get("source_size"),
        }
        pool.sheet_type = str(payload.get("sheet_type") or "unknown")
        pool.transcript = [line for line in (carrier.content.text or "").split("\n") if line.strip()]
        values = payload.get("value_links") or {}
        if isinstance(values, dict):
            pool.value_views = [dict(view) for view in values.get("views") or []]

    ordinal = -1
    for fragment in drawings:
        if fragment.content.structured_drawing_fields is not None:
            ordinal += 1
        own = fragment.content.structured_drawing_fields or []
        is_carrier = fragment is carrier
        to_page, point_to_page = _mapper(fragment, is_carrier)
        payload = fragment.content.structured_payload or {}
        if not is_carrier and not is_vector(fragment) and fragment.position and fragment.position.bbox:
            frame = [float(v) for v in fragment.position.bbox]
            if frame != WHOLE:
                pool.crop_frames.append(clamp_box(frame))
        if is_carrier:
            _title_parts(pool, payload, to_page)
        for index, item in enumerate(own):
            pool.candidates.append(_candidate(item, (ordinal, index), fragment, is_carrier,
                                              payload, to_page, point_to_page))
        for table in payload.get("tables") or []:
            if not isinstance(table, dict):
                continue
            if table.get("kind") == "title_block":
                if is_carrier:
                    pool.title_rows.extend(_rows(table.get("rows")))
                continue
            box = table.get("bbox")
            pool.tables.append({
                "bbox": to_page(box) if isinstance(box, (list, tuple)) and len(box) == 4 else None,
                "title": str(table.get("title") or ""),
                "headers": [str(h) for h in table.get("headers") or []],
                "rows": _rows(table.get("rows")),
                "pass": PASS_TABLE,
            })
        pool.total_fields += len(own)

    for fragment in list(drawings) + list(others):
        pool.lines.extend(_lines_of(fragment))
    for fragment in drawings:
        if is_vector(fragment):
            for item in fragment.content.structured_drawing_fields or []:
                if is_place(item.bbox):
                    pool.lines.append({"text": item.value, "bbox": clamp_box(item.bbox),
                                       "source": SOURCE_VECTOR})
    return pool


def _title_parts(pool: PagePool, payload: Dict[str, Any], to_page) -> None:
    pool.title_fields = {str(k): str(v) for k, v in (payload.get("title_block") or {}).items()
                         if isinstance(v, (str, int, float)) and str(v).strip()}
    pool.title_handwritten = {str(k): str(v) for k, v in
                              (payload.get("title_block_handwritten") or {}).items()
                              if isinstance(v, (str, int, float)) and str(v).strip()}
    box = payload.get("title_block_bbox")
    if not box:
        layout = payload.get("layout") if isinstance(payload.get("layout"), dict) else {}
        for region in layout.get("regions") or []:
            if isinstance(region, dict) and region.get("kind") == "title_block":
                box = region.get("bbox")
                break
    if isinstance(box, (list, tuple)) and len(box) == 4:
        pool.title_bbox = to_page(box)
    side = payload.get("side_column") or {}
    if isinstance(side, dict) and isinstance(side.get("bbox"), (list, tuple)) and len(side["bbox"]) == 4:
        pool.side_bbox = to_page(side["bbox"])


def _rows(raw: Any) -> List[List[str]]:
    rows: List[List[str]] = []
    for row in raw or []:
        if isinstance(row, dict):
            row = list(row.values())
        if isinstance(row, (list, tuple)):
            rows.append([" ".join(str(cell if cell is not None else "").split()) for cell in row])
    return rows


def _candidate(item, ref: Tuple[int, int], fragment: Fragment, carrier: bool, payload: Dict[str, Any],
               to_page, point_to_page) -> Candidate:
    raw_bbox = [float(v) for v in (item.bbox or WHOLE)]
    vector = is_vector(fragment)
    bbox = to_page(raw_bbox) if is_place(raw_bbox) or not carrier else clamp_box(raw_bbox)
    if not carrier and not is_place(raw_bbox) and not _on_page(fragment):
        # Поле вырезки без места: место — вся вырезка.
        frame = list(fragment.position.bbox) if fragment.position and fragment.position.bbox else WHOLE
        bbox = clamp_box(frame)
    label, value = _split_label(item.value) if item.category in ("title_block", "material") else ("", item.value)
    handwritten = bool(_HANDWRITTEN.search(value))
    if handwritten:
        value = _HANDWRITTEN.sub("", value).strip()
    source = SOURCE_VECTOR if vector else SOURCE_VLM
    pass_ = PASS_SHEET if vector else _pass_of(fragment, carrier, raw_bbox, item.category, bool(label), payload)
    link = None
    if item.link is not None:
        link = {
            "target_label": item.link.target_label,
            "target_bbox": (to_page(item.link.target_bbox) if not _on_page(fragment)
                            else clamp_box(item.link.target_bbox)),
            "leader": [point_to_page(p) for p in item.link.leader if len(p) == 2],
            "method": item.link.method,
            "confidence": float(item.link.confidence),
        }
    readings: List[Dict[str, Any]] = []
    for reading in item.readings or []:
        src, pas = _READING_SOURCES.get(str(reading.get("source")), (SOURCE_VLM, PASS_SHEET))
        if not carrier and src == SOURCE_VLM and pas == PASS_SHEET:
            pas = PASS_VIEW_CROP
        readings.append({"value": str(reading.get("value") or ""), "source": src, "pass": pas,
                         "confidence": None})
    for alternative in getattr(item, "alternatives", None) or []:
        readings.append({"value": str(alternative), "source": source, "pass": pass_, "confidence": None})
    exact = vector or bool(getattr(item, "bbox_exact", None))
    agreement = item.agreement
    if agreement is None and getattr(item, "needs_review", None):
        # Размещение полей (N0) нашло расхождение прочтений — на проверку.
        agreement = "conflict"
    confidence = float(item.confidence or 0.0)
    if getattr(item, "low_quality", None):
        confidence = min(confidence, 0.5)
    return Candidate(
        value=value, category=item.category, bbox=bbox, source=source, pass_=pass_,
        confidence=confidence, field_ref=ref, label=label,
        handwritten=handwritten, tolerance=item.tolerance, nature=item.nature,
        agreement=agreement, confirmed=bool(item.confirmed), link=link,
        readings=readings, exact=exact,
    )


def _lines_of(fragment: Fragment) -> List[Dict[str, Any]]:
    """Строки распознавания фрагмента (`position.lines`, флаг RECOGNITION_LINES)."""
    lines = getattr(fragment.position, "lines", None) if fragment.position else None
    out: List[Dict[str, Any]] = []
    for line in lines or []:
        text = getattr(line, "text", None) if not isinstance(line, dict) else line.get("text")
        bbox = getattr(line, "bbox", None) if not isinstance(line, dict) else line.get("bbox")
        source = getattr(line, "source", None) if not isinstance(line, dict) else line.get("source")
        if not text or not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        kind = SOURCE_TEXT_LAYER if source == "text_layer" else SOURCE_OCR
        out.append({"text": str(text), "bbox": clamp_box(bbox), "source": kind})
    return out
