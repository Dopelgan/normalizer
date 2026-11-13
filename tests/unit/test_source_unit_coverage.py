"""Число исходных единиц и выбор стратегии на весь документ без вызовов модели."""
import io

import fitz
import openpyxl
import pytest
from PIL import Image

from core.ladder.base import Strategy
from core.ladder.context import Classification, DocumentContext, GENRE_TEXT, SOURCE_NATIVE
from core.ladder.router import LadderRouter
from core.ladder.strategies.native_text import PdfTextLayerStrategy
from core.models.parse_result import ParsedBlock, ParseResult
from ingest.tasks import page_report


def pdf_bytes(pages):
    with fitz.open() as document:
        for _ in range(pages):
            document.new_page()
        return document.tobytes()


def source(data, file_type):
    context = DocumentContext(uri="fixture", file_type=file_type, data=data)
    context._classification = Classification(source=SOURCE_NATIVE, genre=GENRE_TEXT)
    return context


def test_pdf_count_comes_from_all_source_pages_even_without_text():
    context = source(pdf_bytes(3), "pdf")
    assert context.source_page_count == 3
    assert context.text_layer is None


def test_xlsx_count_includes_empty_and_hidden_sheets():
    workbook = openpyxl.Workbook()
    workbook.active["A1"] = "Первая"
    workbook.create_sheet("Пустая")
    hidden = workbook.create_sheet("Скрытая")
    hidden.sheet_state = "hidden"
    hidden["A1"] = "Третья"
    data = io.BytesIO()
    workbook.save(data)
    assert source(data.getvalue(), "xlsx").source_page_count == 3


def test_multiframe_image_count_is_not_assumed_one():
    data = io.BytesIO()
    first = Image.new("RGB", (32, 32), "white")
    second = Image.new("RGB", (32, 32), "black")
    first.save(data, format="TIFF", save_all=True, append_images=[second])
    assert source(data.getvalue(), "tiff").source_page_count == 2


def test_unreadable_source_count_remains_unknown():
    assert source(b"not a PDF", "pdf").source_page_count is None


def test_short_overlay_on_scanned_page_is_not_complete_native_text():
    raster = io.BytesIO()
    Image.new("RGB", (300, 400), "gray").save(raster, format="PNG")
    with fitz.open() as document:
        page = document.new_page(width=300, height=400)
        for index in range(10):
            page.insert_text((10, 20 + index * 18), "Native text with substantial body content")
        page = document.new_page(width=300, height=400)
        page.insert_image(page.rect, stream=raster.getvalue())
        page.insert_text((10, 20), "SCAN OVERLAY")
        data = document.tobytes()
    context = source(data, "pdf")
    assert context.source_page_count == 2
    assert context.text_layer.overlay_only(2)
    result = PdfTextLayerStrategy().run(context)
    assert result.pages() == [1]


class FixedStrategy(Strategy):
    exhaustive = True

    def __init__(self, level, pages, fallback=False):
        self.level, self.pages, self.fallback = level, pages, fallback
        self.name = f"fixture-{level}"
        self.ran = False

    def applicable(self, context):
        return True

    def run(self, context):
        self.ran = True
        return ParseResult(blocks=[ParsedBlock(type="text", page=p, text="Документ " * 30)
                                  for p in self.pages],
                           page_count=len(self.pages), is_fallback=self.fallback)


@pytest.mark.parametrize('fallback', [False, True])
def test_first_page_cannot_stop_or_outrank_whole_document(fallback):
    first_page = FixedStrategy(2, [1])
    whole = FixedStrategy(5, [1, 2, 3], fallback=fallback)
    result = LadderRouter([first_page, whole], threshold=0.75).parse(source(pdf_bytes(3), "pdf"))
    assert first_page.ran and whole.ran
    assert result.pages() == [1, 2, 3]
    assert result.source_page_count == result.page_count == 3
    assert page_report(result)["unknown_pages"] == []


def test_partial_candidate_preserves_real_source_count_and_unknown_pages():
    result = LadderRouter([FixedStrategy(2, [1])]).parse(source(pdf_bytes(3), "pdf"))
    assert result.source_page_count == 3
    assert page_report(result)["unknown_pages"] == [2, 3]


def test_unknown_source_boundary_does_not_stop_next_strategy():
    cheap = FixedStrategy(2, [1])
    later = FixedStrategy(5, [1])
    LadderRouter([cheap, later], threshold=0.75).parse(source(b"invalid image", "png"))
    assert cheap.ran and later.ran


@pytest.mark.parametrize('sparse_fallback,reverse', [(False, False), (True, False), (True, True)])
def test_page_presence_does_not_override_known_content_loss(sparse_fallback, reverse):
    class SparseStrategy(FixedStrategy):
        def run(self, context):
            result = super().run(context)
            for block in result.blocks:
                block.text = "X"
            return result

    rich = FixedStrategy(5 if reverse else 4, [1], fallback=True)
    sparse = SparseStrategy(4 if reverse else 5, [1, 2], fallback=sparse_fallback)
    result = LadderRouter([rich, sparse], threshold=0.75).parse(source(pdf_bytes(2), "pdf"))
    assert rich.ran and sparse.ran
    assert result.pages() == [1]
    assert "Документ" in result.blocks[0].text
    assert page_report(result)["unknown_pages"] == [2]
