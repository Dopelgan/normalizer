"""Нарезка на фрагменты примерно одинакового размера."""

from typing import Any, Dict, List, Optional

from core.config import settings
from core.models.parse_result import ParsedBlock
from core.providers.chunking_strategy import ChunkingStrategy, ChunkItem


class FixedSizeChunkingStrategy(ChunkingStrategy):
    """
    Склеивает текст постранично и режет на куски заданного размера с
    перекрытием. Границы страниц не пересекаются: иначе один фрагмент
    пришлось бы приписать сразу двум страницам и `position` стал бы враньём.
    """

    name = "fixed"

    def __init__(self, chunk_size: Optional[int] = None, overlap: Optional[int] = None):
        self.chunk_size = int(chunk_size or settings.DEFAULT_CHUNK_SIZE)
        self.overlap = int(settings.DEFAULT_OVERLAP if overlap is None else overlap)

    def chunk(self, blocks: List[ParsedBlock], metadata: Dict[str, Any]) -> List[ChunkItem]:
        text_blocks = self.text_blocks(blocks)
        if not text_blocks:
            return []

        items: List[ChunkItem] = []
        order = 0

        for _page, page_blocks in self.group_by_page(text_blocks):
            spans: List[tuple] = []   # (start, end, block) в координатах склейки
            pieces: List[str] = []
            cursor = 0
            for block in page_blocks:
                text = block.text.strip()
                if not text:
                    continue
                pieces.append(text)
                spans.append((cursor, cursor + len(text), block))
                cursor += len(text) + 1   # +1 на пробел-разделитель

            page_text = " ".join(pieces)
            if not page_text:
                continue

            for start, end in self.split_positions(page_text, self.chunk_size, self.overlap):
                fragment = page_text[start:end].strip()
                if not fragment:
                    continue
                covered = [b for (s, e, b) in spans if s < end and e > start]
                if not covered:
                    covered = [page_blocks[0]]
                item = self._item(fragment, covered, order)
                item.start_char, item.end_char = start, end
                items.append(item)
                order += 1

        return items
