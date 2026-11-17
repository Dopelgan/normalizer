"""
LADDER_LAYOUT_PAGE_SUBSET: на векторном PDF с полным слоем в MinerU уходят
только страницы со структурой (сетка таблицы и её соседи, растр, графика,
формулы); текст остальных страниц сшивка берёт из слоя. Ни одной такой
страницы — запроса нет. Флаг выключен — запрос прежний.
"""

import pytest

from core import telemetry
from core.config import settings
from core.ladder import layout_pages
from core.ladder.context import DocumentContext
from core.ladder.router import LadderRouter
from core.ladder.strategies.native_text import PdfTextLayerStrategy
from core.ladder.strategies.recognition import LayoutStrategy
from core.providers import mineru_parser
from core.providers.mineru_parser import MinerUParserProvider

fitz = pytest.importorskip("fitz")

LINE = "plain running text of the document body line"
URI = "documents/doc.pdf"


def make_pdf(kinds):
    """Страница на каждый элемент `kinds`: text | table | image | formula."""
    document = fitz.open()
    for number, kind in enumerate(kinds, start=1):
        page = document.new_page()
        for row in range(10):
            page.insert_text((50, 60 + row * 14), f"page {number} {LINE}", fontsize=10)
        if kind == "table":
            for row in range(5):
                y = 300 + row * 20
                page.draw_line((50, y), (450, y))
            for x in (50, 250, 450):
                page.draw_line((x, 300), (x, 380))
        elif kind == "image":
            pixmap = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 40, 40), False)
            pixmap.clear_with(128)
            page.insert_image(fitz.Rect(50, 300, 250, 500), pixmap=pixmap)
        elif kind == "formula":
            # Шрифт Symbol не маркером списка: формула.
            page.insert_text((50, 320), "a + b = c", fontsize=10, fontname="symb")
    return document.tobytes()


def context(data):
    return DocumentContext(uri=URI, file_type="pdf", metadata={}, data=data)


def page_texts(data):
    document = fitz.open(stream=data, filetype="pdf")
    try:
        return [page.get_text().split("\n")[0] for page in document]
    finally:
        document.close()


class TestPagesForLayout:
    def test_table_page_and_neighbours(self):
        data = make_pdf(["text", "text", "table", "text", "text", "text"])
        assert layout_pages.pages_for_layout(context(data)) == [2, 3, 4]

    def test_image_and_formula_pages(self):
        data = make_pdf(["text", "image", "text", "formula", "text"])
        assert layout_pages.pages_for_layout(context(data)) == [2, 4]

    def test_plain_text_needs_no_page(self):
        assert layout_pages.pages_for_layout(context(make_pdf(["text"] * 3))) == []

    def test_incomplete_layer_goes_whole(self):
        document = fitz.open()
        document.new_page()  # страница без текста: слой неполон
        page = document.new_page()
        for row in range(10):
            page.insert_text((50, 60 + row * 14), LINE, fontsize=10)
        assert layout_pages.pages_for_layout(context(document.tobytes())) is None


class TestPageSubset:
    def test_page_map_round_trip(self):
        data = make_pdf(["text"] * 6)
        subset = mineru_parser._page_subset(data, [5, 2, 3], 6)
        assert subset.pages == [2, 3, 5]
        # Номера страниц те же: на месте ненужных — пустые листы. От номера
        # страницы зависит имя вырезки MinerU, а от него — `fragment_id`.
        texts = page_texts(subset.data)
        assert [t.split(" ")[1] if t else "" for t in texts] == ["", "2", "3", "", "5", ""]
        assert [subset.original(p) for p in range(1, 8)] == [None, 2, 3, None, 5, None, None]

    def test_contiguous_table_pages_stay_adjacent(self):
        subset = mineru_parser._page_subset(make_pdf(["text"] * 6), [2, 3, 4], 6)
        texts = page_texts(subset.data)
        assert [t.split(" ")[1] if t else "" for t in texts] == ["", "2", "3", "4", "", ""]

    def test_all_pages_send_the_file_as_is(self):
        assert mineru_parser._page_subset(make_pdf(["text"] * 3), [1, 2, 3], 3) is None


def table_page(index, html=None):
    """Страница ответа MinerU: таблица (с содержимым или пустая рамка-продолжение)."""
    block = {"type": "table", "bbox": [50, 300, 450, 380]}
    if html:
        block["blocks"] = [{"lines": [{"spans": [{"html": html}]}]}]
    return {"page_idx": index, "page_size": [595, 842], "para_blocks": [block]}


TABLE_HTML = ("<table><tr><th>A</th><th>B</th></tr><tr><td>1</td><td>2</td></tr>"
              "<tr><td>3</td><td>4</td></tr><tr><td>5</td><td>6</td></tr></table>")


@pytest.fixture
def provider(temp_storage, mocker):
    MinerUParserProvider.reset_availability()
    calls = []
    instance = MinerUParserProvider(temp_storage)

    def request(url, data, files, timeout, **_):
        sent = files[0][1][1]
        calls.append(sent)
        texts = page_texts(sent)
        first = next(i for i, text in enumerate(texts) if text)
        # Таблица на первой непустой странице, на остальных — пустые рамки
        # продолжения (так MinerU отдаёт таблицу через несколько листов);
        # рамка есть и на пустых листах — их блоки должны отброситься.
        pages = [table_page(i, TABLE_HTML if i == first else None) for i in range(len(texts))]
        return {"results": {"doc.pdf": {"middle_json": {"pdf_info": pages}}}}

    mocker.patch.object(instance, "_request_mineru", side_effect=request)
    instance.calls = calls
    yield instance
    MinerUParserProvider.reset_availability()


def store(storage, data):
    storage.write_file(URI, data)
    return data


class TestProviderSubset:
    def test_table_over_three_pages_stays_one_table(self, provider, temp_storage):
        store(temp_storage, make_pdf(["text", "table", "table", "table", "text", "text"]))
        result = provider.parse(URI, "pdf", {}, only_pages=[2, 3, 4])
        # Листы таблицы во временном файле подряд: MinerU склеивает её как
        # прежде.
        assert len(provider.calls) == 1
        texts = page_texts(provider.calls[0])
        assert [t.split(" ")[1] if t else "" for t in texts] == ["", "2", "3", "4", "", ""]
        parsed = [b for b in result.blocks if b.type == "table" and b.table_data]
        assert len(parsed) == 1 and parsed[0].page == 2
        assert sorted(b.page for b in result.blocks if b.type == "table") == [2, 3, 4]
        # Текст каждой страницы на месте, в том числе не отправленной.
        for page in range(1, 7):
            assert any(b.page == page and f"page {page} " in (b.text or "")
                       for b in result.blocks), page

    def test_blocks_of_blank_pages_are_dropped(self, provider, temp_storage):
        store(temp_storage, make_pdf(["text", "table", "text", "text", "table", "text"]))
        result = provider.parse(URI, "pdf", {}, only_pages=[2, 5])
        texts = page_texts(provider.calls[0])
        assert [t.split(" ")[1] if t else "" for t in texts] == ["", "2", "", "", "5", ""]
        # Рамки с пустых листов отброшены, номера страниц — исходные.
        tables = sorted((b.page, bool(b.table_data)) for b in result.blocks if b.type == "table")
        assert tables == [(2, True), (5, False)]
        assert result.page_count == 6
        assert {b.page for b in result.blocks} == {1, 2, 3, 4, 5, 6}

    def test_no_page_needed_no_request(self, provider, temp_storage):
        store(temp_storage, make_pdf(["text", "text"]))
        with telemetry.collect() as timings:
            result = provider.parse(URI, "pdf", {}, only_pages=[])
        assert provider.calls == []
        assert timings["parse.layout_pages_sent"] == 0
        assert {b.page for b in result.blocks} == {1, 2}
        assert not result.is_fallback

    def test_without_subset_the_file_goes_as_is(self, provider, temp_storage):
        data = store(temp_storage, make_pdf(["text", "table", "text"]))
        provider.parse(URI, "pdf", {})
        provider.parse(URI, "pdf", {}, only_pages=[1, 2, 3])
        assert provider.calls == [data, data]


class TestRouter:
    def run(self, monkeypatch, provider, flag, kinds):
        monkeypatch.setattr(settings, "LADDER_LAYOUT_PAGE_SUBSET", flag)
        data = make_pdf(kinds)
        provider.storage.write_file(URI, data)
        router = LadderRouter([PdfTextLayerStrategy(), LayoutStrategy(provider=provider)],
                              threshold=1.01)
        return router.parse(context(data))

    def test_flag_off_request_identical(self, monkeypatch, provider):
        result = self.run(monkeypatch, provider, False, ["text", "table", "text", "text"])
        assert provider.calls == [provider.storage.read_bytes(URI)]
        assert "layout_pages" not in result.ladder

    def test_flag_on_sends_only_needed_pages(self, monkeypatch, provider):
        result = self.run(monkeypatch, provider, True, ["text", "text", "table", "text", "text"])
        assert len(provider.calls) == 1
        texts = page_texts(provider.calls[0])
        assert [t.split(" ")[1] if t else "" for t in texts] == ["", "2", "3", "4", ""]
        assert result.ladder["layout_pages"] == {"sent": [2, 3, 4], "total": 5}
        assert result.ladder["chosen_level"] == 5
        assert {b.page for b in result.blocks} == {1, 2, 3, 4, 5}

    def test_flag_on_plain_text_calls_no_mineru(self, monkeypatch, provider):
        result = self.run(monkeypatch, provider, True, ["text", "text"])
        assert provider.calls == []
        assert result.ladder["layout_pages"]["sent"] == []
