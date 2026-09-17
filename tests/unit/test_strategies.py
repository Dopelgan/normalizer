"""Стратегии нарезки: порядок, страницы, координаты, отсутствие зацикливаний."""

import pytest

from core.models.parse_result import ParsedBlock
from core.providers.chunking_strategy import ChunkingStrategy, merge_bbox
from core.strategies import (
    ChunkingStrategyFactory,
    FixedSizeChunkingStrategy,
    ParagraphChunkingStrategy,
    SemanticChunkingStrategy,
    StrategySelector,
)


def blocks(*specs):
    out = []
    for order, (text, page, bbox) in enumerate(specs):
        out.append(ParsedBlock(type="text", text=text, page=page, bbox=bbox, order=order))
    return out


class TestSplitPositions:
    def test_covers_whole_text(self):
        text = "a" * 1000
        spans = ChunkingStrategy.split_positions(text, 100, 20)
        assert spans[0][0] == 0
        assert spans[-1][1] == len(text)

    @pytest.mark.parametrize(
        "size,overlap",
        [(10, 0), (10, 9), (10, 50), (1, 0), (500, 50), (3, 3)],
    )
    def test_always_terminates(self, size, overlap):
        """Раньше overlap >= chunk_size давал бесконечный цикл."""
        spans = ChunkingStrategy.split_positions("слово " * 200, size, overlap)
        assert spans
        assert all(end > start for start, end in spans)
        # монотонное продвижение
        assert all(b[0] > a[0] for a, b in zip(spans, spans[1:]))

    def test_overlap_repeats_text(self):
        text = "".join(str(i % 10) for i in range(300))
        spans = ChunkingStrategy.split_positions(text, 100, 30)
        assert len(spans) > 1
        assert spans[1][0] < spans[0][1]

    def test_no_near_duplicate_spans(self):
        """
        Регрессия: если граница предложения откатывала конец ближе, чем на
        перекрытие, следующий шаг выбирал её же и выдавал почти такой же
        кусок. На 260 символах получалось 240 спанов вместо десятка.
        """
        text = "abc def ghi. " * 20
        spans = ChunkingStrategy.split_positions(text, 20, 50)

        assert len(spans) < 30
        starts = [start for start, _ in spans]
        assert starts == sorted(set(starts))          # каждый шаг вперёд
        assert len({span for span in spans}) == len(spans)
        assert spans[-1][1] == len(text)



class TestFixedSize:
    def test_pages_not_merged(self):
        strategy = FixedSizeChunkingStrategy(chunk_size=10_000, overlap=0)
        items = strategy.chunk(blocks(
            ("Первая страница.", 1, [0.1, 0.1, 0.9, 0.2]),
            ("Вторая страница.", 2, [0.1, 0.1, 0.9, 0.2]),
        ), {})
        assert [i.page for i in items] == [1, 2]
        assert "Вторая" not in items[0].text

    def test_bbox_is_union_of_blocks(self):
        strategy = FixedSizeChunkingStrategy(chunk_size=10_000, overlap=0)
        items = strategy.chunk(blocks(
            ("Первый блок.", 1, [0.1, 0.1, 0.5, 0.2]),
            ("Второй блок.", 1, [0.2, 0.3, 0.9, 0.4]),
        ), {})
        assert len(items) == 1
        assert items[0].bbox == [0.1, 0.1, 0.9, 0.4]

    def test_order_is_sequential(self):
        strategy = FixedSizeChunkingStrategy(chunk_size=40, overlap=5)
        items = strategy.chunk(blocks(("Текст документа. " * 20, 1, [0, 0, 1, 1])), {})
        assert [i.order for i in items] == list(range(len(items)))


class TestParagraph:
    def test_small_paragraphs_merged(self):
        strategy = ParagraphChunkingStrategy(chunk_size=200, overlap=0)
        items = strategy.chunk(blocks(
            ("Первый абзац.", 1, [0.1, 0.1, 0.9, 0.2]),
            ("Второй абзац.", 1, [0.1, 0.2, 0.9, 0.3]),
        ), {})
        assert len(items) == 1
        assert "Первый" in items[0].text and "Второй" in items[0].text

    def test_long_paragraph_split(self):
        strategy = ParagraphChunkingStrategy(chunk_size=50, overlap=10)
        items = strategy.chunk(blocks(("Очень длинный абзац. " * 20, 1, [0, 0, 1, 1])), {})
        assert len(items) > 1
        assert all(len(i.text) <= 80 for i in items)

    def test_page_change_closes_chunk(self):
        strategy = ParagraphChunkingStrategy(chunk_size=10_000, overlap=0)
        items = strategy.chunk(blocks(
            ("Один.", 1, [0, 0, 1, 1]),
            ("Два.", 2, [0, 0, 1, 1]),
        ), {})
        assert len(items) == 2


class TestSemantic:
    def test_splits_on_headers(self):
        strategy = SemanticChunkingStrategy(chunk_size=10_000)
        items = strategy.chunk(blocks(
            ("1. Введение", 1, [0, 0, 1, 0.1]),
            ("Текст введения.", 1, [0, 0.1, 1, 0.2]),
            ("2. Требования", 1, [0, 0.2, 1, 0.3]),
            ("Текст требований.", 1, [0, 0.3, 1, 0.4]),
        ), {})
        assert len(items) == 2
        assert items[0].section_title == "1. Введение"
        assert items[1].section_title == "2. Требования"

    def test_section_title_from_parser_marker(self):
        marked = ParsedBlock(
            type="text", text="Технические требования", page=1,
            bbox=[0, 0, 1, 0.1], order=0, section_title="Технические требования",
        )
        body = ParsedBlock(type="text", text="Содержимое.", page=1, bbox=[0, 0.1, 1, 0.2], order=1)
        items = SemanticChunkingStrategy(chunk_size=10_000).chunk([marked, body], {})
        assert items[0].section_title == "Технические требования"

    def test_gost_reference_is_header(self):
        block = ParsedBlock(type="text", text="ГОСТ 1050-2013 Прокат", page=1,
                            bbox=[0, 0, 1, 0.1], order=0)
        assert SemanticChunkingStrategy.is_header(block)


class TestFactoryAndSelector:
    def test_factory_returns_requested(self):
        assert isinstance(ChunkingStrategyFactory.get("fixed"), FixedSizeChunkingStrategy)
        assert isinstance(ChunkingStrategyFactory.get("paragraph"), ParagraphChunkingStrategy)
        assert isinstance(ChunkingStrategyFactory.get("semantic"), SemanticChunkingStrategy)

    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError, match="Неизвестная стратегия"):
            ChunkingStrategyFactory.get("magic")

    def test_selector(self):
        assert StrategySelector.select({"doc_type": "technical"}) == "semantic"
        assert StrategySelector.select({"doc_type": "article"}) == "paragraph"
        assert StrategySelector.select({"source_kind": "scanned_pdf"}) == "fixed"
        assert StrategySelector.select({"chunking_strategy": "fixed"}) == "fixed"

    def test_build_returns_configured_strategy(self):
        strategy = StrategySelector.build({"doc_type": "article", "chunk_size": 123})
        assert isinstance(strategy, ParagraphChunkingStrategy)
        assert strategy.chunk_size == 123


def test_merge_bbox_handles_empty():
    assert merge_bbox([]) == [0.0, 0.0, 1.0, 1.0]
    assert merge_bbox([[0.2, 0.2, 0.4, 0.4], [0.1, 0.5, 0.3, 0.7]]) == [0.1, 0.2, 0.4, 0.7]


def test_non_text_blocks_ignored():
    table = ParsedBlock(type="table", page=1, bbox=[0, 0, 1, 1], table_data={"headers": [], "rows": []})
    text = ParsedBlock(type="text", text="Только этот текст.", page=1, bbox=[0, 0, 1, 1])
    items = FixedSizeChunkingStrategy(chunk_size=1000).chunk([table, text], {})
    assert len(items) == 1
    assert items[0].text == "Только этот текст."
