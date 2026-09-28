"""
Модели контракта RAG <-> Parser (PARSER API SPEC DRAFT 2026-09-07,
изменения контракта от 2026-09-22 — см. docs/PARSER-CONTRACT-CHANGES.md).

Правила контракта, которые здесь закодированы:

* наружу отдаются `document_metadata` + `fragments`;
* все обязательные поля присутствуют в JSON, даже если значение `null`
  (поэтому нигде не используется `exclude_none`);
* `confidence` и `completeness` лежат в диапазоне [0, 1];
* `position.bbox` — нормализованные координаты [x1, y1, x2, y2] в [0, 1];
* поля доступа (container / legal_entity / sensitivity) Parser не формирует;
* синхронная ручка работает с `dialog_id`, фоновая — только с `event_id`:
  в фоне диалога нет, а событие жизненного цикла есть;
* ошибка — объект с кодом причины, а не строка: по строке потребитель не
  может решить, повторять запрос или отправлять файл человеку.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_serializer, field_validator

FragmentType = Literal["text", "table", "formula", "drawing", "image", "structured"]
DocumentStatus = Literal["pending", "indexed", "error"]
RequestStatus = Literal["queued", "processing", "completed", "partial", "error"]
Operation = Literal["create", "update"]
# Итог разбора одного файла. Отличается от `DocumentStatus`: тот описывает
# состояние записи в хранилище, а этот — чем закончился именно этот разбор.
DocumentOutcome = Literal["completed", "partial", "error"]

# Минимальный набор кодов причины по контракту. Список расширяемый, но
# сужать его нельзя: потребитель принимает решение о повторе по коду.
ReasonCode = Literal[
    "SOURCE_NOT_FOUND",
    "SOURCE_READ_ERROR",
    "UNSUPPORTED_FORMAT",
    "OCR_FAILED",
    "EXTRACTION_FAILED",
    "TABLE_EXTRACTION_FAILED",
    "PARSER_TIMEOUT",
    "INTERNAL_ERROR",
]

FactKind = Literal[
    "number", "range", "date", "duration", "text", "boolean", "identifier", "enum"
]
FactOperator = Literal["=", "<=", ">=", "<", ">", "range"]


def _iso_utc(value: datetime) -> str:
    """ISO-8601 с суффиксом Z, как в примерах контракта."""
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ===========================================================================
# Ошибка
# ===========================================================================

class ParserError(BaseModel):
    """
    Структурированная ошибка разбора.

    `retryable` отвечает на единственный вопрос потребителя: повторять или
    нет. Неподдерживаемый формат повторять бессмысленно, недоступное
    хранилище — наоборот.
    """

    reason_code: ReasonCode = "INTERNAL_ERROR"
    message: str = ""
    retryable: bool = False
    attempt: int = Field(1, ge=1)

    @classmethod
    def coerce(cls, value: Any) -> Optional["ParserError"]:
        """
        Приводит к объекту то, что пришло строкой.

        Нужно ради совместимости: в Redis могут лежать документы, сложенные
        до перехода на объект ошибки, и терять их из-за формата нельзя.
        """
        if value is None or isinstance(value, cls):
            return value
        if isinstance(value, dict):
            return cls(**value)
        return cls(reason_code="INTERNAL_ERROR", message=str(value))


# ===========================================================================
# Вложенные структуры фрагмента
# ===========================================================================

class Position(BaseModel):
    """Положение фрагмента в исходном документе."""

    page: int = Field(..., ge=0)
    sheet: Optional[str] = None
    bbox: List[float] = Field(default_factory=lambda: [0.0, 0.0, 1.0, 1.0])
    order: Optional[int] = None

    @field_validator("bbox")
    @classmethod
    def _normalize_bbox(cls, v: List[float]) -> List[float]:
        if not v:
            return [0.0, 0.0, 1.0, 1.0]
        if len(v) != 4:
            raise ValueError(f"bbox должен содержать 4 координаты, получено {len(v)}")
        x1, y1, x2, y2 = (float(c) for c in v)
        # Порядок координат приводим к (левый-верх, правый-низ) и зажимаем в [0,1].
        lo_x, hi_x = min(x1, x2), max(x1, x2)
        lo_y, hi_y = min(y1, y2), max(y1, y2)
        clamp = lambda c: max(0.0, min(1.0, c))  # noqa: E731
        return [clamp(lo_x), clamp(lo_y), clamp(hi_x), clamp(hi_y)]


class Provenance(BaseModel):
    """
    Происхождение фрагмента: чем получен и из какого источника.

    `source_file_id` и `source_page` добавлены контрактом: без них RAG не
    может открыть страницу-доказательство и не отличает повторный разбор
    того же файла от нового документа.
    """

    method: str          # cad_source | text_layer_extraction | mineru_ocr | ...
    # Уровень лестницы стратегий, 1..7 по возрастанию стоимости:
    # 1 исходник САПР, 2 текстовый слой, 3 табличный разбор,
    # 4 распознавание без структуры, 5 детекция областей,
    # 6 восстановление растра плюс 5, 7 мультимодальная модель.
    strategy_level: int = Field(..., ge=1, le=7)
    source: str          # vector_pdf | scanned_pdf | document_parser | drawing | ...
    source_file_id: Optional[str] = None
    source_page: Optional[int] = None


class GraphNode(BaseModel):
    id: str
    type: str
    name: str


class Relation(BaseModel):
    source: str
    target: str
    type: str


class TableData(BaseModel):
    headers: List[str] = Field(default_factory=list)
    rows: List[List[Any]] = Field(default_factory=list)
    total_row: Optional[Dict[str, Any]] = None


class FormulaData(BaseModel):
    """Разобранное выражение формулы (`content.parsed_expression`)."""

    base_var: str = ""
    operations: List[Dict[str, Any]] = Field(default_factory=list)


class DrawingField(BaseModel):
    """Структурированное поле чертежа."""

    category: str
    value: str
    tolerance: Optional[str] = None
    nature: Literal["executive", "reference", "tool_provided"] = "executive"
    # Пусто, когда единица не названа и не следует из категории. Подставлять
    # «мм» по умолчанию нельзя: так помечался и текст, у которого единицы нет.
    unit: str = ""
    confidence: float = Field(0.0, ge=0, le=1)
    bbox: List[float] = Field(default_factory=lambda: [0.0, 0.0, 1.0, 1.0])
    source_of_tolerance: Literal[
        "explicit", "general_tolerance_table", "fit_notation", "unknown"
    ] = "unknown"
    provenance: Literal[
        "detected_from_annotation", "reference_table", "expert_verified",
        "full_page_ocr", "vector_text_layer",
    ] = "detected_from_annotation"


class Content(BaseModel):
    """
    Содержимое фрагмента. Ключи присутствуют всегда — какой именно заполнен,
    определяется полем `type` фрагмента.
    """

    text: Optional[str] = None
    table_data: Optional[TableData] = None
    image_ref: Optional[str] = None
    formula_mathml: Optional[str] = None
    parsed_expression: Optional[FormulaData] = None
    structured_drawing_fields: Optional[List[DrawingField]] = None
    structured_payload: Optional[Dict[str, Any]] = None


# ===========================================================================
# Извлечённые факты
# ===========================================================================

class FactValue(BaseModel):
    """
    Значение факта. `raw` хранит исходную запись — она остаётся
    доказательством, когда разбор числа спорный.

    Выдумывать значение или единицу, которых нет в документе, нельзя:
    пустое поле честнее подставленного по умолчанию.
    """

    kind: FactKind
    raw: str
    number: Optional[float] = None
    min: Optional[float] = None
    max: Optional[float] = None
    unit: Optional[str] = None
    operator: Optional[FactOperator] = None

    @field_validator("unit")
    @classmethod
    def _empty_unit_is_none(cls, v: Optional[str]) -> Optional[str]:
        return v or None

    @field_validator("max")
    @classmethod
    def _ordered_range(cls, v: Optional[float], info) -> Optional[float]:
        low = info.data.get("min")
        if v is not None and low is not None and v < low:
            raise ValueError("max диапазона меньше min")
        return v


class ExtractedFact(BaseModel):
    """
    Явно извлечённое значение из фрагмента.

    Parser извлекает факт, но не решает, нужен ли он для конкретного
    пользовательского вопроса: отбор — работа RAG.
    """

    fact_id: str
    fragment_id: str
    # Расширяемый ключ типа факта: фиксированного списка под один проект нет.
    key: str = "document.parameter"
    label: str = ""
    value: FactValue
    confidence: float = Field(1.0, ge=0, le=1)
    # Координаты внутри фрагмента: для таблиц — {row, column, header}.
    # Без них факт из таблицы нельзя показать в исходной ячейке.
    provenance: Optional[Dict[str, Any]] = None


# ===========================================================================
# Фрагмент и документ
# ===========================================================================

class Fragment(BaseModel):
    fragment_id: str
    type: FragmentType
    content: Content
    position: Optional[Position] = None
    section_title: Optional[str] = None
    confidence: float = Field(..., ge=0, le=1)
    completeness: float = Field(..., ge=0, le=1)
    provenance: Provenance
    extracted_facts: List[ExtractedFact] = Field(default_factory=list)
    graph_nodes: List[GraphNode] = Field(default_factory=list)
    relations: List[Relation] = Field(default_factory=list)


class DocumentMetadata(BaseModel):
    doc_id: str
    doc_version: Optional[str] = None
    language: str = "ru"
    source_path: str = ""
    updated_at: Optional[datetime] = None
    status: DocumentStatus = "pending"

    @field_serializer("updated_at")
    def _ser_updated_at(self, value: Optional[datetime], _info) -> Optional[str]:
        return _iso_utc(value) if value else None


class DocumentResult(BaseModel):
    """
    Результат по одному файлу.

    `processed_pages` и `failed_pages` заполняются всегда, а не только при
    `status: partial`: по ним потребитель индексирует разобранное и ставит
    файл на проверку, не гадая, что именно пропало.
    """

    s3_fileid: str
    status: DocumentOutcome = "completed"
    # `sha256:<hex>` содержимого файла и нормализованного текста. Второй
    # отвечает на вопрос «изменился ли смысл», когда байты файла изменились.
    content_hash: Optional[str] = None
    normalized_content_hash: Optional[str] = None
    mime_type: Optional[str] = None
    pages: Optional[int] = None
    processed_pages: List[int] = Field(default_factory=list)
    failed_pages: List[int] = Field(default_factory=list)
    document_metadata: DocumentMetadata
    fragments: List[Fragment] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    error: Optional[ParserError] = None

    @field_validator("error", mode="before")
    @classmethod
    def _error_object(cls, v: Any) -> Any:
        return ParserError.coerce(v)


# ===========================================================================
# Ответы ручек
# ===========================================================================

class ParseResponse(BaseModel):
    """
    Ответ синхронной ручки POST /internal/v1/parse.

    `dialog_id` здесь остаётся: синхронный разбор вызывается из
    пользовательского диалога.
    """

    request_id: str
    dialog_id: str
    documents: List[DocumentResult] = Field(default_factory=list)


class SyncPendingResponse(BaseModel):
    """
    Ответ 504 синхронной ручки: работа не уложилась в таймаут.

    Это по-прежнему синхронный ответ, поэтому `dialog_id` в нём есть.
    Готовый результат забирается по /internal/v1/parse/results/{request_id}.
    """

    request_id: str
    dialog_id: str
    status: RequestStatus
    documents: List[DocumentResult] = Field(default_factory=list)
    error: Optional[ParserError] = None

    @field_validator("error", mode="before")
    @classmethod
    def _error_object(cls, v: Any) -> Any:
        return ParserError.coerce(v)


class BackgroundAcceptedResponse(BaseModel):
    """
    Ответ POST /internal/v1/parse/background (202 Accepted).

    `dialog_id` из фонового контракта убран: связь события, запроса и
    результата держит `event_id`, пришедший от Backend через RAG, и заменять
    его новым идентификатором нельзя.

    `operation` — общая операция пакета. Когда файлы просят разное, общей
    операции нет, и поле приходит пустым: выдумывать её нельзя.
    """

    request_id: str
    event_id: str = ""
    operation: Optional[Operation] = None
    accepted: bool = True
    status: RequestStatus = "queued"


class ResultsResponse(BaseModel):
    """
    Envelope фонового результата: и для поллинга
    GET /internal/v1/parse/results/{request_id}, и для отправки в
    POST /internal/v1/parser/results на стороне RAG.
    """

    request_id: str
    event_id: str = ""
    operation: Optional[Operation] = None
    status: RequestStatus
    documents: List[DocumentResult] = Field(default_factory=list)
    error: Optional[ParserError] = None

    @field_validator("error", mode="before")
    @classmethod
    def _error_object(cls, v: Any) -> Any:
        return ParserError.coerce(v)
