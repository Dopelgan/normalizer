"""Нарезка по абзацам с укрупнением мелких и дроблением крупных."""

from typing import Any, Dict, List, Optional

from core.config import settings
from core.models.parse_result import ParsedBlock
from core.providers.chunking_strategy import ChunkingStrategy, ChunkItem, merge_bbox


class ParagraphChunkingStrategy(ChunkingStrategy):
    """
    Каждый текстовый блок парсера — абзац. Подряд идущие абзацы одной
    страницы склеиваются, пока укладываются в `chunk_size`; слишком длинный
    абзац режется с перекрытием. Смена страницы всегда закрывает фрагмент.

    Фрагмент закрывается ещё и на заметном вертикальном разрыве между
    блоками: иначе далёкие друг от друга абзацы попадают в один фрагмент,
    и его bbox растягивается на пол-листа — подсветка источника в RAG
    начинает указывать мимо.
    """

    name = "paragraph"

    # Доля высоты страницы, после которой абзацы считаются разными частями.
    VERTICAL_GAP = 0.035

    def __init__(self, chunk_size: Optional[int] = None, overlap: Optional[int] = None):
        self.chunk_size = int(chunk_size or settings.DEFAULT_CHUNK_SIZE)
        self.overlap = int(settings.DEFAULT_OVERLAP if overlap is None else overlap)
        self.min_chunk_size = int(settings.MIN_CHUNK_SIZE)

    def chunk(self, blocks: List[ParsedBlock], metadata: Dict[str, Any]) -> List[ChunkItem]:
        text_blocks = self.text_blocks(blocks)
        if not text_blocks:
            return []

        items: List[ChunkItem] = []
        order = 0

        for _page, page_blocks in self.group_by_page(text_blocks):
            buffer: List[ParsedBlock] = []
            buffer_len = 0

            def flush(buf: List[ParsedBlock], order_: int) -> int:
                if not buf:
                    return order_
                text = " ".join(b.text.strip() for b in buf if b.text)
                if text.strip():
                    items.append(self._item(text, buf, order_))
                    order_ += 1
                return order_

            previous: Optional[ParsedBlock] = None
            for block in page_blocks:
                text = block.text.strip()
                if not text:
                    continue

                if previous is not None and self._far_apart(previous, block):
                    order = flush(buffer, order)
                    buffer, buffer_len = [], 0
                previous = block

                if len(text) > self.chunk_size:
                    order = flush(buffer, order)
                    buffer, buffer_len = [], 0
                    for start, end in self.split_positions(text, self.chunk_size, self.overlap):
                        piece = text[start:end].strip()
                        if not piece:
                            continue
                        item = self._item(piece, [block], order)
                        item.start_char, item.end_char = start, end
                        items.append(item)
                        order += 1
                    continue

                if buffer and buffer_len + len(text) > self.chunk_size:
                    order = flush(buffer, order)
                    buffer, buffer_len = [], 0

                buffer.append(block)
                buffer_len += len(text) + 1

            order = flush(buffer, order)

        return self._absorb_tiny(items)

    # ------------------------------------------------------------ внутреннее
    def _far_apart(self, previous: ParsedBlock, current: ParsedBlock) -> bool:
        """Между блоками пустое место заметно больше межабзацного."""
        if previous.bbox == [0.0, 0.0, 1.0, 1.0] or current.bbox == [0.0, 0.0, 1.0, 1.0]:
            return False
        return (current.bbox[1] - previous.bbox[3]) > self.VERTICAL_GAP

    def _absorb_tiny(self, items: List[ChunkItem]) -> List[ChunkItem]:
        """
        Короткий фрагмент подклеивается к соседнему на той же странице, а не
        выбрасывается: раньше строки вроде `POST /cart/add` просто исчезали.
        Одиночный короткий фрагмент страницы остаётся как есть.
        """
        if self.min_chunk_size <= 0:
            return items

        merged: List[ChunkItem] = []
        for item in items:
            if (
                merged
                and len(item.text) < self.min_chunk_size
                and merged[-1].page == item.page
            ):
                host = merged[-1]
                host.text = f"{host.text} {item.text}".strip()
                host.blocks = host.blocks + item.blocks
                host.bbox = merge_bbox([b.bbox for b in host.blocks])
                continue
            merged.append(item)

        for order, item in enumerate(merged):
            item.order = order
        return merged
