"""Внутренние модели результата парсинга (между парсером и нормализатором)."""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator

BlockType = Literal["text", "table", "image", "formula", "drawing"]


class ParsedBlock(BaseModel):
    """
    Один блок, извлечённый парсером.

    `bbox` всегда нормализован в [0, 1] относительно размера страницы —
    приведение выполняет парсер, который единственный знает `page_size`.
    """

    type: BlockType
    text: Optional[str] = None
    table_data: Optional[Dict[str, Any]] = None   # {headers, rows, total_row}
    table_html: Optional[str] = None              # исходный HTML таблицы от MinerU
    image_ref: Optional[str] = None               # URI изображения (для type=image)
    page: int = 0
    # Имя листа: лист книги XLSX, лист печати DXF. По контракту уезжает в
    # position.sheet — лист есть самостоятельная единица.
    sheet_name: Optional[str] = None
    bbox: List[float] = Field(default_factory=lambda: [0.0, 0.0, 1.0, 1.0])
    order: Optional[int] = None
    section_title: Optional[str] = None
    confidence: float = 1.0
    is_fallback: bool = False                     # получен через Tesseract
    page_size: Optional[List[float]] = None       # [width, height] исходной страницы
    # Чем реально получен текст блока: text_layer | text_layer_recovered |
    # mineru_ocr | tesseract. Нужен, чтобы provenance не выдумывался по типу
    # фрагмента, а отражал фактический путь.
    method: Optional[str] = None
    # Разобранные поля чертежа (для type="drawing"). Заполняются парсером,
    # который единственный держит в руках исходный PDF и его текстовый слой.
    drawing_fields: Optional[List[Dict[str, Any]]] = None

    @field_validator("bbox")
    @classmethod
    def _check_bbox(cls, v: List[float]) -> List[float]:
        if not v or len(v) != 4:
            return [0.0, 0.0, 1.0, 1.0]
        x1, y1, x2, y2 = (float(c) for c in v)
        clamp = lambda c: max(0.0, min(1.0, c))  # noqa: E731
        return [clamp(min(x1, x2)), clamp(min(y1, y2)), clamp(max(x1, x2)), clamp(max(y1, y2))]

    @field_validator("confidence")
    @classmethod
    def _check_conf(cls, v: float) -> float:
        return max(0.0, min(1.0, float(v)))

    def area(self) -> float:
        """Доля площади страницы, занятая блоком."""
        x1, y1, x2, y2 = self.bbox
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)


class ParseResult(BaseModel):
    blocks: List[ParsedBlock] = Field(default_factory=list)
    is_fallback: bool = False                     # весь результат получен фоллбэком
    # Формат отдал всё своё содержимое и потерять его стратегия не могла
    # (DOCX, TXT, XLSX, DXF). Полнота таких фрагментов не измеряется по
    # площади блоков на листе — листа у них нет.
    exhaustive: bool = False
    parser_name: str = "mineru"                   # mineru | tesseract_full
    source_kind: str = "vector_pdf"               # vector_pdf | scanned_pdf | image | office
    raw_output: Dict[str, Any] = Field(default_factory=dict)
    failed_pages: List[int] = Field(default_factory=list)
    page_count: int = 0
    # Символов (без пробелов) в текстовом слое PDF по страницам. Пусто, если
    # слоя нет или PyMuPDF недоступен. Это эталон для расчёта полноты.
    text_layer_chars: Dict[int, int] = Field(default_factory=dict)
    # Сводка по сшивке слоя с блоками парсера.
    text_layer_stats: Dict[str, Any] = Field(default_factory=dict)
    # Как выбирался уровень лестницы: классификация, попытки, оценки.
    ladder: Dict[str, Any] = Field(default_factory=dict)
    # Деградации разбора: недоступность сервиса, победа фоллбэка, таблица,
    # которую не удалось разобрать. Это не свойства документа, а сообщения о
    # том, что система отработала хуже, чем умеет, — и молчать о них нельзя:
    # лежащий mineru-api иначе неотличим от плохого скана.
    degraded: List[str] = Field(default_factory=list)

    def pages(self) -> List[int]:
        return sorted({b.page for b in self.blocks})
