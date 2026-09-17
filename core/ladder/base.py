"""Интерфейс стратегии разбора и общие помощники уровней лестницы."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import List

from core.ladder.context import DocumentContext
from core.models.parse_result import ParsedBlock, ParseResult

logger = logging.getLogger(__name__)


class Strategy(ABC):
    """
    Один уровень лестницы. Стоимость закодирована в `level`: маршрутизатор
    начинает с самого дешёвого применимого и поднимается выше только если
    результат не дотянул до порога.
    """

    level: int = 0
    name: str = "base"
    #: Текстовая метка способа получения, уезжает в provenance фрагментов.
    method: str = "unknown"
    #: Извлечение исчерпывающее: формат отдал всё, что в нём есть, и
    #: потерять содержимое стратегия не могла. Для распознавания это не так,
    #: и там полноту приходится измерять.
    exhaustive: bool = False

    @abstractmethod
    def applicable(self, context: DocumentContext) -> bool:
        """Можно ли применить стратегию к этому документу."""

    @abstractmethod
    def run(self, context: DocumentContext) -> ParseResult:
        """Разобрать документ. Исключение означает неудачу уровня."""

    # --------------------------------------------------------- помощники
    def build_result(
        self,
        blocks: List[ParsedBlock],
        source_kind: str,
        page_count: int = 0,
        is_fallback: bool = False,
        **extra,
    ) -> ParseResult:
        """Результат уровня с проставленным способом получения у блоков."""
        for order, block in enumerate(blocks):
            if block.order is None:
                block.order = order
            if block.method is None:
                block.method = self.method
        return ParseResult(
            blocks=blocks,
            is_fallback=is_fallback,
            exhaustive=self.exhaustive,
            parser_name=self.name,
            source_kind=source_kind,
            page_count=page_count or len({b.page for b in blocks}),
            **extra,
        )

    def __repr__(self) -> str:  # pragma: no cover — только для логов
        return f"<{type(self).__name__} level={self.level}>"


def text_chars(result: ParseResult) -> int:
    """Символы без пробелов во всех блоках результата, включая таблицы."""
    total = 0
    for block in result.blocks:
        if block.table_data:
            cells = [str(c) for row in (block.table_data.get("rows") or []) for c in row]
            cells += [str(h) for h in (block.table_data.get("headers") or [])]
            total += len("".join("".join(c.split()) for c in cells))
        total += len("".join((block.text or "").split()))
    return total


def has_structure(result: ParseResult) -> bool:
    """Есть ли в результате что-то кроме плоского текста."""
    return any(b.type in ("table", "formula", "drawing", "image") for b in result.blocks)
