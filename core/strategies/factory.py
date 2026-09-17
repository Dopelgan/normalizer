"""Фабрика и селектор стратегий нарезки."""

from typing import Any, Dict, Type

from core.config import settings
from core.providers.chunking_strategy import ChunkingStrategy
from core.strategies.fixed_size import FixedSizeChunkingStrategy
from core.strategies.paragraph import ParagraphChunkingStrategy
from core.strategies.semantic import SemanticChunkingStrategy


class ChunkingStrategyFactory:
    _strategies: Dict[str, Type[ChunkingStrategy]] = {
        "fixed": FixedSizeChunkingStrategy,
        "paragraph": ParagraphChunkingStrategy,
        "semantic": SemanticChunkingStrategy,
    }

    @classmethod
    def get(cls, name: str, **kwargs) -> ChunkingStrategy:
        strategy_class = cls._strategies.get((name or "").lower())
        if strategy_class is None:
            raise ValueError(
                f"Неизвестная стратегия нарезки: {name!r}. "
                f"Доступны: {', '.join(sorted(cls._strategies))}"
            )
        return strategy_class(**kwargs)


class StrategySelector:
    """Выбор стратегии по метаданным документа и характеру источника."""

    @staticmethod
    def select(metadata: Dict[str, Any]) -> str:
        explicit = metadata.get("chunking_strategy")
        if explicit:
            return str(explicit)

        doc_type = str(metadata.get("doc_type", "")).lower()
        if doc_type in ("technical", "spec", "standard", "гост"):
            return "semantic"
        if doc_type in ("article", "paper", "report"):
            return "paragraph"

        # У скана нет структуры блоков — единственная честная стратегия
        # это фиксированный размер.
        if metadata.get("source_kind") in ("scanned_pdf", "image"):
            return "fixed"
        return "paragraph"

    @staticmethod
    def build(metadata: Dict[str, Any]) -> ChunkingStrategy:
        return ChunkingStrategyFactory.get(
            StrategySelector.select(metadata),
            chunk_size=metadata.get("chunk_size", settings.DEFAULT_CHUNK_SIZE),
            overlap=metadata.get("overlap", settings.DEFAULT_OVERLAP),
        )
