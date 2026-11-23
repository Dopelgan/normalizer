"""
Структура листа чертежа `drawing-sheet-v1` (`Content.drawing_sheet`).

Один объект на страницу, только у фрагмента «лист целиком». Собирается
детерминированно из прочтений, которые проходы уже дали; плоские
`structured_drawing_fields` рядом не меняются — структура их дополняет и
ссылается на них через `field_refs`.

Правила схемы:

* координаты — доли выправленного листа `[x1, y1, x2, y2]`
  (`coordinate_reference: deskewed_page`), как у `DrawingField.bbox`;
* у каждого элемента есть `id` (`<вид>-<n>` в порядке чтения) и `anchor` —
  отпечаток вида, нормализованного значения и центра рамки на сетке 1/64:
  по нему решения специалиста переживают перечитывание листа;
* варианты прочтения не выбрасываются, а лежат в `alternatives`;
* ключ со значением `None` в ответ не попадает: так структура короче, а
  отсутствующий ключ потребитель читает как `null`;
* версия — строка `schema` внутри объекта; добавление ключей версию не
  меняет, поэтому лишние ключи при разборе пропускаются, а не отвергаются;
* размер ограничен 48 КБ (`cap_sheet_size`): точка индекса RAG читается
  с пределом 64 КБ.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_serializer

SCHEMA_VERSION = "drawing-sheet-v1"
# Предел размера структуры одного листа в байтах UTF-8 компактного JSON.
SIZE_CAP_BYTES = 48 * 1024
# Сетка центра рамки для `anchor`: 1/64 листа — около 4,6 мм на A3, этого
# хватает, чтобы перечитанная надпись попала в ту же клетку.
ANCHOR_GRID = 64

ElementSource = Literal["vector", "text_layer", "ocr", "vlm"]
PassName = Literal[
    "sheet", "transcribe", "title_block", "side_column", "table", "view_crop", "reread", "links",
]
ReviewState = Literal["auto", "needs_check", "confirmed", "corrected", "rejected"]
SheetType = Literal["detail", "assembly", "scheme", "unknown"]
ToleranceSource = Literal["explicit", "general_tolerance_table", "fit_notation", "unknown"]

_ID = re.compile(r"^[a-z_]+-\d+$")
# Число повторов: целое или как написано на листе («N отв.» -> "N").
_COUNT_MAX_CHARS = 16
_SPACES = re.compile(r"\s+")


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _normalize_bbox(value: List[float]) -> List[float]:
    """Порядок (левый-верх, правый-низ) и зажим в [0, 1], как у `Position.bbox`."""
    if len(value) != 4:
        raise ValueError(f"bbox должен содержать 4 координаты, получено {len(value)}")
    x1, y1, x2, y2 = (float(c) for c in value)
    return [_clamp(min(x1, x2)), _clamp(min(y1, y2)), _clamp(max(x1, x2)), _clamp(max(y1, y2))]


def _count(value: Any) -> Any:
    """Целое >= 1 или непустая короткая запись («N»); цифры строкой -> целое."""
    if value is None or isinstance(value, bool):
        return None if value is None else _bad_count(value)
    if isinstance(value, int):
        return value if value >= 1 else _bad_count(value)
    text = " ".join(str(value).split())
    if text.isdigit():
        return _count(int(text))
    if not text or len(text) > _COUNT_MAX_CHARS:
        return _bad_count(value)
    return text


def _bad_count(value: Any):
    raise ValueError(f"count — целое >= 1 или короткая запись, получено {value!r}")


def _normalize_points(value: List[List[float]]) -> List[List[float]]:
    points = []
    for point in value:
        if len(point) != 2:
            raise ValueError(f"точка линии должна содержать 2 координаты, получено {len(point)}")
        points.append([_clamp(point[0]), _clamp(point[1])])
    return points


class _SheetModel(BaseModel):
    """Общая основа: лишние ключи пропускаются, `None` в ответ не выводится."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    @model_serializer(mode="wrap")
    def _without_none(self, handler):
        data = handler(self)
        if isinstance(data, dict):
            return {key: value for key, value in data.items() if value is not None}
        return data


# ===========================================================================
# Основа элемента
# ===========================================================================

class ElementProvenance(_SheetModel):
    """Чем получено значение и какими проходами оно прочитано."""

    source: ElementSource
    passes: List[PassName] = Field(default_factory=list)


class ElementReview(_SheetModel):
    state: ReviewState = "auto"


class Alternative(_SheetModel):
    """Другое прочтение той же надписи. `pass` — слово Python, отсюда `pass_`."""

    value: str
    source: ElementSource
    pass_: PassName = Field(alias="pass")
    confidence: float = Field(0.0, ge=0, le=1)

    @model_serializer(mode="wrap")
    def _without_none(self, handler):
        # Ключ в ответе — `pass` при любом `by_alias`: фрагмент сохраняется
        # обычным `model_dump()`, а API отдаёт по псевдонимам.
        data = handler(self)
        if not isinstance(data, dict):
            return data
        return {
            ("pass" if key == "pass_" else key): value
            for key, value in data.items() if value is not None
        }


class FieldRef(_SheetModel):
    """
    Мост к плоским полям: `fragment` — порядковый номер (с нуля) фрагмента
    этой страницы среди фрагментов с `structured_drawing_fields` в порядке
    ответа, `field` — индекс поля в его `structured_drawing_fields`.
    """

    fragment: int = Field(ge=0)
    field: int = Field(ge=0)


class Element(_SheetModel):
    """Общие ключи всякого элемента листа."""

    id: str
    anchor: str = Field(pattern=r"^[0-9a-f]{40}$")
    bbox: List[float]
    bbox_exact: bool = False
    provenance: ElementProvenance
    confidence: float = Field(0.0, ge=0, le=1)
    review: ElementReview = Field(default_factory=ElementReview)
    alternatives: List[Alternative] = Field(default_factory=list)
    field_refs: List[FieldRef] = Field(default_factory=list)

    @field_validator("id")
    @classmethod
    def _id_format(cls, value: str) -> str:
        if not _ID.match(value):
            raise ValueError(f"id элемента должен иметь вид <вид>-<n>, получено {value!r}")
        return value

    @field_validator("bbox")
    @classmethod
    def _bbox(cls, value: List[float]) -> List[float]:
        return _normalize_bbox(value)


# ===========================================================================
# Рамка и штамп
# ===========================================================================

class Frame(_SheetModel):
    """Лист и его выправление: угол, поворот, формат, внутренняя рамка."""

    coordinate_reference: Literal["deskewed_page"] = "deskewed_page"
    angle: float = 0.0
    turn: Literal[0, 90, 180, 270] = 0
    format: Optional[str] = None
    format_confidence: float = Field(0.0, ge=0, le=1)
    inner_frame: Optional[List[float]] = None
    # Размер страницы в пикселях рендера [ширина, высота].
    page_px: Optional[List[int]] = None

    @field_validator("inner_frame")
    @classmethod
    def _inner(cls, value: Optional[List[float]]) -> Optional[List[float]]:
        return None if value is None else _normalize_bbox(value)

    @field_validator("page_px")
    @classmethod
    def _page_px(cls, value: Optional[List[int]]) -> Optional[List[int]]:
        if value is not None and (len(value) != 2 or min(value) <= 0):
            raise ValueError("page_px — два положительных числа [ширина, высота]")
        return value


class TypedValue(Element):
    """Графа штампа: значение из сводного прочтения, `bbox` — рамка клетки."""

    value: str
    handwritten: bool = False


class TitleRole(_SheetModel):
    """
    Роль в штампе. Имя — персональные данные: в связи графа не идёт.
    `alternatives` — другие прочтения имени, `bbox` — клетка имени,
    `field_refs` — поля, из которых роль собрана.
    """

    role: str
    name: str
    date: Optional[str] = None
    handwritten: bool = False
    personal: Literal[True] = True
    bbox: Optional[List[float]] = None
    alternatives: List[str] = Field(default_factory=list)
    field_refs: List["FieldRef"] = Field(default_factory=list)

    @field_validator("bbox")
    @classmethod
    def _bbox(cls, value: Optional[List[float]]) -> Optional[List[float]]:
        return None if value is None else _normalize_bbox(value)


class TitleChange(_SheetModel):
    """Строка таблицы изменений штампа."""

    rev: Optional[str] = None
    doc: Optional[str] = None
    sign: Optional[str] = None
    date: Optional[str] = None


class TitleBlock(_SheetModel):
    """
    Основная надпись. Ключи `fields`: designation, name, material, mass,
    scale, sheet, sheets, format, organisation, inventory_no и другие графы —
    набор открыт.
    """

    bbox: Optional[List[float]] = None
    fields: Dict[str, TypedValue] = Field(default_factory=dict)
    roles: List[TitleRole] = Field(default_factory=list)
    changes: List[TitleChange] = Field(default_factory=list)

    @field_validator("bbox")
    @classmethod
    def _bbox(cls, value: Optional[List[float]]) -> Optional[List[float]]:
        return None if value is None else _normalize_bbox(value)


# ===========================================================================
# Виды, элементы конструкции, размеры
# ===========================================================================

class View(Element):
    label: Optional[str] = None
    title_raw: Optional[str] = None
    kind: Literal["main", "view", "section", "detail"] = "view"
    scale: Optional[str] = None
    # Число одинаковых мест вида («4 места»).
    count: Optional[int] = Field(None, ge=1)
    # Рамка заголовка вида («Б (4:1)»), когда `bbox` — область вида.
    title_bbox: Optional[List[float]] = None

    @field_validator("title_bbox")
    @classmethod
    def _title_bbox(cls, value: Optional[List[float]]) -> Optional[List[float]]:
        return None if value is None else _normalize_bbox(value)


class Feature(Element):
    """Элемент детали. На сканах выводится из размера, связи и выноски."""

    kind: Literal[
        "hole", "thread", "fillet", "chamfer", "bend", "edge", "surface", "contour", "weld",
    ]
    label: Optional[str] = None
    # Сколько таких элементов: целое или запись листа («N» у «N отв.»).
    count: Optional[Union[int, str]] = None
    view_id: Optional[str] = None
    # id элементов, из которых элемент выведен.
    derived_from: List[str] = Field(default_factory=list)

    @field_validator("count", mode="before")
    @classmethod
    def _count_value(cls, value: Any) -> Any:
        return _count(value)


class Tolerance(_SheetModel):
    """Отклонения в единицах размера; `fit` — обозначение посадки («H7»)."""

    upper: Optional[float] = None
    lower: Optional[float] = None
    fit: Optional[str] = None
    source: ToleranceSource = "unknown"


class Formula(_SheetModel):
    """Параметрический размер: `(X+38)` и переменные `["X"]`."""

    expr: str
    variables: List[str] = Field(default_factory=list)


class LinkInfo(_SheetModel):
    method: Literal["vector", "vlm", "rule"]
    confidence: float = Field(0.0, ge=0, le=1)


class Dimension(Element):
    value_raw: str
    kind: Literal["linear", "diameter", "radius", "angle", "thread", "chamfer"]
    nominal: Optional[float] = None
    unit: Optional[str] = None
    tolerance: Optional[Tolerance] = None
    # Справочный размер («*» по примечанию) и размер, обеспечиваемый инструментом.
    reference: bool = False
    tool_provided: bool = False
    # «N отв.», «4 места»: сколько раз размер повторяется — целое или
    # запись листа, когда число задано буквой («N»).
    count: Optional[Union[int, str]] = None
    formula: Optional[Formula] = None
    view_id: Optional[str] = None
    feature_id: Optional[str] = None
    # Линия от надписи к элементу [[x, y], ...]; при превышении размера
    # снимается первой.
    leader: List[List[float]] = Field(default_factory=list)
    link: Optional[LinkInfo] = None
    note_refs: List[str] = Field(default_factory=list)

    @field_validator("leader")
    @classmethod
    def _leader(cls, value: List[List[float]]) -> List[List[float]]:
        return _normalize_points(value)

    @field_validator("count", mode="before")
    @classmethod
    def _count_value(cls, value: Any) -> Any:
        return _count(value)


class TableRowRef(_SheetModel):
    table_id: str
    row: int = Field(ge=0)


class PositionMark(Element):
    """Позиция сборочного чертежа (номер в кружке с выноской)."""

    number: str
    leader: List[List[float]] = Field(default_factory=list)
    target_bbox: Optional[List[float]] = None
    view_id: Optional[str] = None
    table_row: Optional[TableRowRef] = None

    @field_validator("leader")
    @classmethod
    def _leader(cls, value: List[List[float]]) -> List[List[float]]:
        return _normalize_points(value)

    @field_validator("target_bbox")
    @classmethod
    def _target(cls, value: Optional[List[float]]) -> Optional[List[float]]:
        return None if value is None else _normalize_bbox(value)


class NoteRefs(_SheetModel):
    """На что ссылается пункт технических требований (id элементов)."""

    notes: List[str] = Field(default_factory=list)
    dimensions: List[str] = Field(default_factory=list)
    features: List[str] = Field(default_factory=list)
    symbols: List[str] = Field(default_factory=list)


class Note(Element):
    number: Optional[int] = Field(None, ge=0)
    text: str
    # `footnote` — сноска «1) - …» под требованиями: номер свой, с пунктами не
    # сравнивается; пункты со знаком «1)» ссылаются на неё через `refs.notes`.
    kind: Literal["item", "footnote"] = "item"
    # Пункт, определяющий знак у размеров («*» — справочный и т. п.).
    defines_marker: Optional[Literal["*", "**"]] = None
    refs: NoteRefs = Field(default_factory=NoteRefs)


class TableCell(_SheetModel):
    text: str
    bbox: Optional[List[float]] = None

    @field_validator("bbox")
    @classmethod
    def _bbox(cls, value: Optional[List[float]]) -> Optional[List[float]]:
        return None if value is None else _normalize_bbox(value)


class TableRow(_SheetModel):
    cells: List[TableCell] = Field(default_factory=list)


class SheetTable(Element):
    kind: Literal["spec", "change", "material", "other"] = "other"
    headers: List[str] = Field(default_factory=list)
    rows: List[TableRow] = Field(default_factory=list)


class Symbol(Element):
    """Знак шероховатости или сварки; `general` — знак в углу листа."""

    kind: Literal["roughness", "weld"]
    value: str
    scope: Optional[Literal["general", "surface"]] = None
    feature_id: Optional[str] = None
    standard: Optional[str] = None


class SheetText(Element):
    """Прочитанная надпись, не подошедшая ни к одному виду: ничего не теряется молча."""

    text: str


class SheetStats(_SheetModel):
    elements: int = Field(0, ge=0)
    merged_duplicates: int = Field(0, ge=0)
    alternatives: int = Field(0, ge=0)
    needs_check: int = Field(0, ge=0)
    unlinked_dimensions: int = Field(0, ge=0)
    truncated: bool = False


class DrawingSheet(_SheetModel):
    schema_: Literal["drawing-sheet-v1"] = Field(SCHEMA_VERSION, alias="schema")
    page: int = Field(ge=0)
    sheet_type: SheetType = "unknown"
    frame: Frame = Field(default_factory=Frame)
    title_block: Optional[TitleBlock] = None
    views: List[View] = Field(default_factory=list)
    features: List[Feature] = Field(default_factory=list)
    dimensions: List[Dimension] = Field(default_factory=list)
    positions: List[PositionMark] = Field(default_factory=list)
    notes: List[Note] = Field(default_factory=list)
    tables: List[SheetTable] = Field(default_factory=list)
    symbols: List[Symbol] = Field(default_factory=list)
    texts: List[SheetText] = Field(default_factory=list)
    stats: SheetStats = Field(default_factory=SheetStats)

    @field_validator("sheet_type", mode="before")
    @classmethod
    def _sheet_type(cls, value: Any) -> str:
        # Модель листа может назвать вид иначе — это не повод терять лист.
        return value if value in ("detail", "assembly", "scheme") else "unknown"

    @model_serializer(mode="wrap")
    def _without_none(self, handler):
        # `schema` занято у BaseModel; в ответе ключ всегда `schema`.
        data = handler(self)
        if isinstance(data, dict) and "schema_" in data:
            data = {"schema": data.pop("schema_"), **data}
        return {key: value for key, value in data.items() if value is not None}


# ===========================================================================
# Помощники
# ===========================================================================

def normalize_anchor_value(value: str) -> str:
    """Значение для отпечатка: NFKC, без пробелов, без регистра, запятая = точка."""
    text = unicodedata.normalize("NFKC", value or "")
    return _SPACES.sub("", text).casefold().replace(",", ".")


def element_anchor(kind: str, value: str, bbox: List[float]) -> str:
    """
    Отпечаток элемента: sha1 вида, нормализованного значения и центра рамки
    на сетке 1/64. Тот же отпечаток считает RAG, сопоставляя решения.
    """
    x1, y1, x2, y2 = _normalize_bbox(bbox)
    cell = lambda c: min(ANCHOR_GRID - 1, int(c * ANCHOR_GRID))  # noqa: E731
    key = f"{kind}|{normalize_anchor_value(value)}|{cell((x1 + x2) / 2)},{cell((y1 + y2) / 2)}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()


def sheet_json(sheet: DrawingSheet) -> str:
    """Компактный JSON структуры — по нему считается предел размера."""
    return json.dumps(
        sheet.model_dump(mode="json"), ensure_ascii=False, separators=(",", ":"),
    )


def sheet_size_bytes(sheet: DrawingSheet) -> int:
    return len(sheet_json(sheet).encode("utf-8"))


def _json_bytes(value: Any) -> int:
    return len(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


def cap_sheet_size(
    sheet: DrawingSheet, limit: int = SIZE_CAP_BYTES,
) -> Tuple[DrawingSheet, bool]:
    """
    Укладывает структуру в предел: сначала снимает линии выносок (размеры и
    позиции), затем надписи без вида — с конца, по одной. Если что-то снято,
    ставит `stats.truncated`. Возвращает новую структуру и признак, что она
    уложилась; не уложившуюся вызывающий не отдаёт.
    """
    if sheet_size_bytes(sheet) <= limit:
        return sheet, True

    capped = sheet.model_copy(deep=True)
    capped.stats.truncated = True
    for item in [*capped.dimensions, *capped.positions]:
        item.leader = []
    size = sheet_size_bytes(capped)
    if size <= limit:
        return capped, True

    # Снимаем надписи с конца, считая байты каждой вместе с разделителем,
    # чтобы не сериализовать лист заново на каждом шаге.
    texts = list(capped.texts)
    while texts and size > limit:
        dropped = texts.pop()
        size -= _json_bytes(dropped.model_dump(mode="json")) + (1 if texts else 0)
    capped.texts = texts
    size = sheet_size_bytes(capped)
    return capped, size <= limit
