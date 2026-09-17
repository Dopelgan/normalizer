"""
Модели контракта RAG <-> Parser (PARSER API SPEC DRAFT 2026-09-07).

Правила контракта, которые здесь закодированы:

* наружу отдаются `document_metadata` + `fragments`;
* все обязательные поля присутствуют в JSON, даже если значение `null`
  (поэтому нигде не используется `exclude_none`);
* `confidence` и `completeness` лежат в диапазоне [0, 1];
* `position.bbox` — нормализованные координаты [x1, y1, x2, y2] в [0, 1];
* поля доступа (container / legal_entity / sensitivity) Parser не формирует.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_serializer, field_validator

FragmentType = Literal["text", "table", "formula", "drawing", "image", "structured"]
DocumentStatus = Literal["pending", "indexed", "error"]
RequestStatus = Literal["queued", "processing", "completed", "partial", "error"]
Operation = Literal["create", "update"]


def _iso_utc(value: datetime) -> str:
    """ISO-8601 с суффиксом Z, как в примерах контракта."""
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


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
    """Происхождение фрагмента: чем получен и из какого источника."""

    method: str          # cad_source | text_layer_extraction | mineru_ocr | ...
    # Уровень лестницы стратегий, 1..7 по возрастанию стоимости:
    # 1 исходник САПР, 2 текстовый слой, 3 табличный разбор,
    # 4 распознавание без структуры, 5 детекция областей,
    # 6 восстановление растра плюс 5, 7 мультимодальная модель.
    strategy_level: int = Field(..., ge=1, le=7)
    source: str          # vector_pdf | scanned_pdf | document_parser | drawing | ...


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
    s3_fileid: str
    document_metadata: DocumentMetadata
    fragments: List[Fragment] = Field(default_factory=list)
    error: Optional[str] = None


# ===========================================================================
# Ответы ручек
# ===========================================================================

class ParseResponse(BaseModel):
    """Ответ синхронной ручки POST /internal/v1/parse."""

    request_id: str
    dialog_id: str
    documents: List[DocumentResult] = Field(default_factory=list)


class BackgroundAcceptedResponse(BaseModel):
    """
    Ответ POST /internal/v1/parse/background (202 Accepted).

    `operation` — общая операция пакета. Когда файлы просят разное, общей
    операции нет, и поле приходит пустым: выдумывать её нельзя.
    """

    request_id: str
    dialog_id: str
    operation: Optional[Operation] = None
    accepted: bool = True
    status: RequestStatus = "queued"


class ResultsResponse(BaseModel):
    """Ответ GET /internal/v1/parse/results/{request_id}."""

    request_id: str
    dialog_id: str
    operation: Optional[Operation] = None
    status: RequestStatus
    documents: List[DocumentResult] = Field(default_factory=list)
    error: Optional[str] = None
