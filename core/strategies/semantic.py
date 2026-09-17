"""Нарезка по смысловым границам: заголовкам и номерам разделов."""

import re
from typing import Any, Dict, List, Optional

from core.config import settings
from core.models.parse_result import ParsedBlock
from core.providers.chunking_strategy import ChunkingStrategy, ChunkItem

HEADER_PATTERNS = [
    re.compile(r"^(Глава|Раздел|Приложение|Chapter|Section|Appendix)\s+\d+", re.IGNORECASE),
    re.compile(r"^\d+(\.\d+)*\.?\s+\S"),          # "1. Введение", "2.3 Требования"
    re.compile(r"^[А-ЯЁA-Z][А-ЯЁA-Z\s\-]{6,}$"),  # СТРОКА ЗАГЛАВНЫМИ
    re.compile(r"^(ГОСТ|ОСТ|ТУ|СНиП|СП)\s", re.IGNORECASE),
]


class SemanticChunkingStrategy(ChunkingStrategy):
    """
    Новый фрагмент начинается на заголовке (пометка парсера `section_title`
    или текст, похожий на заголовок) либо при переполнении `chunk_size`.
    Заголовок запоминается и проставляется всем фрагментам своего раздела.
    """

    name = "semantic"

    def __init__(self, chunk_size: Optional[int] = None, overlap: Optional[int] = None):
        self.chunk_size = int(chunk_size or settings.DEFAULT_CHUNK_SIZE)
        self.overlap = int(settings.DEFAULT_OVERLAP if overlap is None else overlap)

    @staticmethod
    def is_header(block: ParsedBlock) -> bool:
        if block.section_title:
            return True
        text = (block.text or "").strip()
        if not text or len(text) > 200:
            return False
        return any(p.match(text) for p in HEADER_PATTERNS)

    def chunk(self, blocks: List[ParsedBlock], metadata: Dict[str, Any]) -> List[ChunkItem]:
        text_blocks = self.text_blocks(blocks)
        if not text_blocks:
            return []

        items: List[ChunkItem] = []
        order = 0
        buffer: List[ParsedBlock] = []
        buffer_len = 0
        section: Optional[str] = None

        def flush(buf: List[ParsedBlock], order_: int, title: Optional[str]) -> int:
            if not buf:
                return order_
            text = " ".join(b.text.strip() for b in buf if b.text)
            if text.strip():
                items.append(self._item(text, buf, order_, section_title=title))
                order_ += 1
            return order_

        for block in text_blocks:
            text = (block.text or "").strip()
            if not text:
                continue

            header = self.is_header(block)
            page_changed = bool(buffer) and block.page != buffer[-1].page
            overflow = bool(buffer) and buffer_len + len(text) > self.chunk_size

            if buffer and (header or page_changed or overflow):
                order = flush(buffer, order, section)
                buffer, buffer_len = [], 0

            if header:
                section = block.section_title or text[:200]

            buffer.append(block)
            buffer_len += len(text) + 1

        order = flush(buffer, order, section)
        return items
