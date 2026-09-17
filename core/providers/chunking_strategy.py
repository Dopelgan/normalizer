"""
Базовый интерфейс стратегий нарезки.

Стратегия работает со списком `ParsedBlock` (а не с плоской строкой), потому
что только так сохраняются номер страницы, bbox и порядок — без них нельзя
заполнить `position` из контракта и показать источник фрагмента.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from core.models.parse_result import ParsedBlock


@dataclass
class ChunkItem:
    """Результат нарезки: текст плюс всё, что нужно для `position`."""

    text: str
    page: int = 0
    bbox: List[float] = field(default_factory=lambda: [0.0, 0.0, 1.0, 1.0])
    order: int = 0
    section_title: Optional[str] = None
    blocks: List[ParsedBlock] = field(default_factory=list)
    start_char: Optional[int] = None
    end_char: Optional[int] = None

    @property
    def is_fallback(self) -> bool:
        return any(b.is_fallback for b in self.blocks)

    @property
    def confidence(self) -> float:
        """
        Средневзвешенная по объёму текста, а не минимум: одна слабая рамка
        не должна топить уверенность всей страницы. Короткий блок с низким
        баллом весит ровно столько, сколько в нём символов.
        """
        if not self.blocks:
            return 1.0
        weights = [max(1, len(b.text or "")) for b in self.blocks]
        total = sum(weights)
        return sum(b.confidence * w for b, w in zip(self.blocks, weights)) / total

    @property
    def method(self) -> Optional[str]:
        """Способ получения, которым добыто большинство символов фрагмента."""
        if not self.blocks:
            return None
        weight: Dict[str, int] = {}
        for block in self.blocks:
            if not block.method:
                continue
            weight[block.method] = weight.get(block.method, 0) + max(1, len(block.text or ""))
        if not weight:
            return None
        return max(weight, key=weight.get)


def merge_bbox(bboxes: List[List[float]]) -> List[float]:
    """Объединяющий прямоугольник. Пустой список -> вся страница."""
    valid = [b for b in bboxes if b and len(b) == 4]
    if not valid:
        return [0.0, 0.0, 1.0, 1.0]
    return [
        min(b[0] for b in valid),
        min(b[1] for b in valid),
        max(b[2] for b in valid),
        max(b[3] for b in valid),
    ]


class ChunkingStrategy(ABC):
    """Интерфейс стратегии нарезки."""

    name = "base"

    @abstractmethod
    def chunk(self, blocks: List[ParsedBlock], metadata: Dict[str, Any]) -> List[ChunkItem]:
        """Нарезает текстовые блоки на фрагменты, сохраняя порядок."""

    # ------------------------------------------------------- общие помощники
    @staticmethod
    def text_blocks(blocks: List[ParsedBlock]) -> List[ParsedBlock]:
        """Только непустые текстовые блоки, в исходном порядке."""
        selected = [b for b in blocks if b.type == "text" and b.text and b.text.strip()]
        return sorted(
            selected,
            key=lambda b: (b.page, b.order if b.order is not None else 0),
        )

    @staticmethod
    def group_by_page(blocks: List[ParsedBlock]) -> List[tuple]:
        """[(page, [blocks...]), ...] с сохранением порядка страниц."""
        groups: Dict[int, List[ParsedBlock]] = {}
        for block in blocks:
            groups.setdefault(block.page, []).append(block)
        return [(page, groups[page]) for page in sorted(groups)]

    @staticmethod
    def split_positions(text: str, size: int, overlap: int) -> List[tuple]:
        """
        Границы нарезки строки с перекрытием. Гарантирует продвижение вперёд
        на каждой итерации, поэтому зациклиться не может ни при каких size/overlap.
        """
        size = max(1, int(size))
        overlap = max(0, min(int(overlap), size - 1))
        length = len(text)
        spans: List[tuple] = []
        start = 0
        while start < length:
            end = min(start + size, length)
            if end < length:
                brk = ChunkingStrategy._find_break(text, start, end)
                if start < brk < end:
                    end = brk
            spans.append((start, end))
            if end >= length:
                break
            # Перекрытие не должно отбрасывать назад к уже пройденной границе:
            # иначе следующий шаг выберет тот же разрыв и выдаст почти такой же
            # кусок — на длинных строках без границ предложения это давало
            # десятки дублей подряд.
            start = end - overlap if end - overlap > start else end
        return spans

    @staticmethod
    def _find_break(text: str, start: int, end: int) -> int:
        """Граница предложения, иначе граница слова, иначе исходный end."""
        window_start = max(start, end - 120)
        segment = text[window_start:end]
        for marker in (". ", "! ", "? ", ".\n", "\n\n"):
            idx = segment.rfind(marker)
            if idx > 0:
                return window_start + idx + len(marker)
        space = text.rfind(" ", start, end)
        if space > start:
            return space + 1
        return end

    def _item(
        self,
        text: str,
        blocks: List[ParsedBlock],
        order: int,
        section_title: Optional[str] = None,
    ) -> ChunkItem:
        page = blocks[0].page if blocks else 0
        title = section_title
        if title is None:
            for block in blocks:
                if block.section_title:
                    title = block.section_title
                    break
        return ChunkItem(
            text=text.strip(),
            page=page,
            bbox=merge_bbox([b.bbox for b in blocks]),
            order=order,
            section_title=title,
            blocks=list(blocks),
        )
