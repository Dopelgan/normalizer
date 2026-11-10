"""
Причинные регрессии потерь данных в разборе (10.11.2026):
формулы XLSX без сохранённого значения, обрыв OCR после сбойной пачки
страниц и таймаут MinerU, выключавший разбор с разметкой для всех документов.
"""

import io

import openpyxl
import pytest
import requests
import requests_mock
from PIL import Image

from core.config import settings
from core.providers import tesseract_fallback
from core.providers.mineru_parser import MinerUParserProvider
from core.providers.tesseract_fallback import OcrUnavailable, TesseractFallbackProvider

PARSE_URL = f"{settings.MINERU_ENDPOINT.rstrip('/')}/file_parse"


# ------------------------------------------------------------------ XLSX

def _xlsx_with_uncached_formula() -> bytes:
    """openpyxl пишет формулы без кэша значений — как выгрузки и скрипты."""
    book = openpyxl.Workbook()
    sheet = book.active
    sheet.append(["Позиция", "Кол-во", "Цена", "Сумма"])
    sheet.append(["Датчик", 2, 150, "=B2*C2"])
    sheet.append(["Итого", None, None, "=SUM(D2:D2)"])
    buffer = io.BytesIO()
    book.save(buffer)
    return buffer.getvalue()


def test_uncached_xlsx_formula_is_kept_as_text():
    from core.ladder.strategies.tabular import SpreadsheetStrategy

    blocks = SpreadsheetStrategy()._from_xlsx(_xlsx_with_uncached_formula())
    rows = blocks[0].table_data["rows"]
    assert rows[0] == ["Датчик", "2", "150", "=B2*C2"]
    assert rows[1][3] == "=SUM(D2:D2)"


def test_plain_xlsx_values_unchanged():
    from core.ladder.strategies.tabular import SpreadsheetStrategy

    book = openpyxl.Workbook()
    book.active.append(["А", 1])
    book.active.append(["Б", 2.5])
    buffer = io.BytesIO()
    book.save(buffer)
    blocks = SpreadsheetStrategy()._from_xlsx(buffer.getvalue())
    assert blocks[0].table_data["headers"] == ["А", "1"]
    assert blocks[0].table_data["rows"] == [["Б", "2.5"]]


# ---------------------------------------------------- сбойная пачка страниц

def _fake_render(fail_batches):
    def render(path, first_page, last_page, dpi):
        if first_page in fail_batches:
            raise RuntimeError("битая страница")
        return [Image.new("L", (100, 100), 255) for _ in range(first_page, last_page + 1)]
    return render


@pytest.fixture
def pdf_provider(monkeypatch):
    provider = TesseractFallbackProvider(storage=object())
    monkeypatch.setattr(provider, "_materialize", lambda uri: ("/tmp/doc.pdf", lambda: None))
    monkeypatch.setattr(provider, "_is_image", staticmethod(lambda path: False))
    monkeypatch.setattr(provider, "_page_count", staticmethod(lambda path: 12))
    monkeypatch.setattr(settings, "OCR_BATCH_PAGES", 5)
    monkeypatch.setattr(
        provider, "_blocks_from_image",
        lambda image, page: [tesseract_fallback.ParsedBlock(type="text", text=f"стр {page}", page=page)],
    )
    return provider


def test_failed_render_batch_does_not_drop_rest_of_document(pdf_provider, monkeypatch):
    monkeypatch.setattr(tesseract_fallback, "convert_from_path", _fake_render({6}))
    blocks = pdf_provider.parse_all_pages("documents/doc.pdf")
    assert sorted({b.page for b in blocks}) == [1, 2, 3, 4, 5, 11, 12]
    assert pdf_provider.failed_pages == [6, 7, 8, 9, 10]


def test_text_layer_continues_after_failed_batch(pdf_provider, monkeypatch):
    monkeypatch.setattr(tesseract_fallback, "convert_from_path", _fake_render({1}))
    pages = [page for page, _ in pdf_provider._pdf_pages("/tmp/doc.pdf")]
    assert pages == [6, 7, 8, 9, 10, 11, 12]
    assert pdf_provider.failed_pages == [1, 2, 3, 4, 5]


def test_unknown_page_count_does_not_invent_failed_pages(pdf_provider, monkeypatch):
    """Число страниц неизвестно: пачки за концом документа — не неудачи (ревью 2b0d2e1)."""
    monkeypatch.setattr(pdf_provider, "_page_count", staticmethod(lambda path: 0))

    def render(path, first_page, last_page, dpi):
        if first_page > 3:
            raise RuntimeError("за концом документа")
        return [Image.new("L", (10, 10), 255) for _ in range(first_page, min(last_page, 3) + 1)]

    monkeypatch.setattr(tesseract_fallback, "convert_from_path", render)
    blocks = pdf_provider.parse_all_pages("documents/doc.pdf")
    assert sorted({b.page for b in blocks}) == [1, 2, 3]
    assert pdf_provider.failed_pages == []


def test_time_budget_truncates_honestly(pdf_provider, monkeypatch):
    monkeypatch.setattr(tesseract_fallback, "convert_from_path", _fake_render(set()))
    monkeypatch.setattr(settings, "OCR_TIME_BUDGET", -1)
    assert pdf_provider.parse_all_pages("documents/doc.pdf") == []
    assert pdf_provider.truncated_at == (0, 12)


def test_nothing_rendered_is_ocr_failure_not_empty_document(pdf_provider, monkeypatch):
    monkeypatch.setattr(tesseract_fallback, "convert_from_path", _fake_render({1, 6, 11}))
    with pytest.raises(OcrUnavailable):
        list(pdf_provider._pdf_pages("/tmp/doc.pdf"))


# ------------------------------------------------------------- MinerU

@pytest.fixture
def mineru(temp_storage):
    MinerUParserProvider.reset_availability()
    temp_storage.write_file("documents/test.pdf", b"%PDF-1.4 fake")
    yield MinerUParserProvider(temp_storage)
    MinerUParserProvider.reset_availability()


def test_slow_document_does_not_disable_mineru_for_others(mineru, mocker):
    mocker.patch.object(mineru.tesseract, "parse_all_pages", return_value=[])
    with requests_mock.Mocker() as m:
        m.post(PARSE_URL, exc=requests.exceptions.ReadTimeout("долго"))
        result = mineru.parse("documents/test.pdf", "pdf")
    assert result.parser_name == "tesseract_full"
    assert MinerUParserProvider.is_available() is True


def test_unreachable_mineru_is_still_marked_unavailable(mineru, mocker):
    mocker.patch.object(mineru.tesseract, "parse_all_pages", return_value=[])
    with requests_mock.Mocker() as m:
        m.post(PARSE_URL, exc=requests.exceptions.ConnectionError("нет связи"))
        mineru.parse("documents/test.pdf", "pdf")
    assert MinerUParserProvider.is_available() is False


def test_timeout_grows_with_page_count(mineru, monkeypatch):
    import fitz

    document = fitz.open()
    for _ in range(100):
        document.new_page()
    _, _, pages = MinerUParserProvider._read_pdf(document.tobytes(), "pdf")
    monkeypatch.setattr(settings, "MINERU_TIMEOUT_PER_PAGE", 15)
    monkeypatch.setattr(settings, "PARSE_TASK_SOFT_TIME_LIMIT", 3600)
    assert pages == 100
    assert mineru._timeout_for(pages) == 1440  # не больше 40 % бюджета задачи
    assert mineru._timeout_for(30) == 450
    assert mineru._timeout_for(1) == mineru.timeout


def test_parse_task_has_its_own_time_budget():
    from ingest.tasks import process_document

    assert process_document.soft_time_limit == settings.PARSE_TASK_SOFT_TIME_LIMIT
    assert settings.MINERU_TIMEOUT_PER_PAGE * 30 < settings.PARSE_TASK_SOFT_TIME_LIMIT


def test_timeout_is_reported_as_timeout_not_outage(mineru, mocker):
    mocker.patch.object(mineru.tesseract, "parse_all_pages", return_value=[])
    with requests_mock.Mocker() as m:
        m.post(PARSE_URL, exc=requests.exceptions.ReadTimeout("долго"))
        result = mineru.parse("documents/test.pdf", "pdf")
    assert any(note.startswith("mineru_timeout") for note in result.degraded)


# ----------------------------------------------- многостраничное восстановление

def test_restoration_keeps_every_page_of_a_scan():
    import fitz
    from core.ladder.restoration import restore_pages

    document = fitz.open()
    for index in range(3):
        page = document.new_page(width=300, height=400)
        page.insert_text((50, 100), f"page {index + 1}")
    restored, applied, sample = restore_pages(document.tobytes(), max_pages=50)
    assert fitz.open(stream=restored, filetype="pdf").page_count == 3
    assert sample and applied


# ------------------------------------------- общий срок задачи (ревью 6ff80ed)

def test_deadline_bounds_budgets_only_inside_task():
    from core import deadline

    assert deadline.bounded(1200) == 1200
    with deadline.scope(100):
        assert 99 < deadline.bounded(1200) <= 100
        assert deadline.bounded(1200, reserve=30) <= 70
    assert deadline.remaining() is None


def test_mineru_timeout_leaves_half_of_remaining_time(mineru):
    from core import deadline

    with deadline.scope(200):
        assert mineru._timeout_for(100) <= 100


class _Level:
    level, name, exhaustive = 4, "fake", False

    def __init__(self, exc):
        self.exc = exc

    def applicable(self, context):
        return True

    def run(self, context):
        raise self.exc


def _context():
    from types import SimpleNamespace

    classification = SimpleNamespace(source="raster", genre="text", raster_quality=0.5,
                                     as_dict=lambda: {})
    return SimpleNamespace(uri="documents/x.pdf", file_type="pdf", kind="pdf",
                           classification=classification, source_page_count=1,
                           text_layer=None)


def test_router_does_not_swallow_celery_soft_time_limit():
    from celery.exceptions import SoftTimeLimitExceeded
    from core.ladder.router import LadderRouter

    router = LadderRouter(strategies=[_Level(SoftTimeLimitExceeded())], enabled_levels=[4])
    with pytest.raises(SoftTimeLimitExceeded):
        router.parse(_context())


def test_router_skips_levels_without_time_left():
    from core import deadline
    from core.ladder.router import LadderRouter
    from core.providers.document_parser import ParserFailed

    level = _Level(AssertionError("уровень не должен запускаться"))
    router = LadderRouter(strategies=[level], enabled_levels=[4])
    with deadline.scope(1), pytest.raises(ParserFailed, match="бюджета времени"):
        router.parse(_context())


def test_ocr_render_does_not_swallow_soft_time_limit(pdf_provider, monkeypatch):
    from celery.exceptions import SoftTimeLimitExceeded

    def render(*args, **kwargs):
        raise SoftTimeLimitExceeded()

    monkeypatch.setattr(tesseract_fallback, "convert_from_path", render)
    with pytest.raises(SoftTimeLimitExceeded):
        pdf_provider.parse_all_pages("documents/doc.pdf")


def test_soft_limit_keeps_best_candidate_already_found():
    """Мягкий лимит на дорогом уровне не теряет результат дешёвого (ревью f7e47ff)."""
    from celery.exceptions import SoftTimeLimitExceeded
    from core.ladder.router import LadderRouter
    from core.models.parse_result import ParsedBlock, ParseResult

    class _Ok(_Level):
        def run(self, context):
            return ParseResult(blocks=[ParsedBlock(type="text", text="текст " * 50, page=1,
                                                   confidence=0.5)],
                               is_fallback=True, parser_name="fake", source_kind="scanned_pdf")

    cheap, costly = _Ok(None), _Level(SoftTimeLimitExceeded())
    costly.level, costly.name = 5, "costly"
    router = LadderRouter(strategies=[cheap, costly], enabled_levels=[4, 5])
    result = router.parse(_context())
    assert result.blocks and any("срок задачи истёк" in note for note in result.degraded)


def test_restoration_stops_before_task_deadline():
    import fitz
    from core import deadline
    from core.ladder.restoration import restore_pages

    document = fitz.open()
    for _ in range(3):
        document.new_page(width=200, height=200)
    with deadline.scope(10):
        restored, applied, _ = restore_pages(document.tobytes(), max_pages=50)
    assert restored is None and applied == []  # ни одной страницы не начато


# ------------------------------------------------- сканы чертежей в PDF (10.11)

def _scanned_drawing_pdf(pages: int = 1) -> bytes:
    """Скан чертежа: рамка ГОСТ с полем подшивки 20 мм (9,5 % ширины), без текстового слоя."""
    import fitz
    from PIL import ImageDraw

    document = fitz.open()
    for _ in range(pages):
        sheet = Image.new("L", (1654, 2339), 255)
        draw = ImageDraw.Draw(sheet)
        draw.rectangle([157, 39, 1615, 2300], outline=0, width=4)  # 20 мм слева, 5 мм остальные
        for y in range(1990, 2300, 40):
            draw.line([(800, y), (1615, y)], fill=0, width=2)  # штамп
        draw.line([(800, 1990), (800, 2300)], fill=0, width=2)
        draw.arc([400, 400, 1200, 1400], 90, 270, fill=0, width=6)
        for y in (300, 1500, 1700):
            draw.line([(250, y), (1500, y)], fill=0, width=2)  # размерные и выносные линии
        draw.line([(1450, 200), (1450, 1900)], fill=0, width=2)
        buffer = io.BytesIO()
        sheet.save(buffer, format="PNG")
        page = document.new_page(width=595, height=842)
        page.insert_image(page.rect, stream=buffer.getvalue())
    return document.tobytes()


def test_scanned_pdf_drawing_is_classified_as_drawing(monkeypatch):
    """Скан PDF без векторных линий проверяется растровым детектором (раньше — нет)."""
    from core.ladder.context import GENRE_DRAWING, DocumentContext
    from core.providers import raster_drawing

    calls = []

    def verdict(data):
        calls.append(len(data))
        return raster_drawing.RasterVerdict(raster_drawing.KIND_DRAWING, 0.85, "тест", {})

    monkeypatch.setattr(raster_drawing, "analyse", verdict)
    context = DocumentContext(uri="x", file_type="pdf", data=_scanned_drawing_pdf(pages=2))
    assert context.classification.genre == GENRE_DRAWING
    assert context.classification.signals["raster_drawing_pages"] == [1, 2]
    assert len(calls) == 2


def test_single_false_drawing_page_does_not_relabel_text_scan(monkeypatch):
    """Одна «чертёжная» страница из многих не делает скан смешанным (гриф!)."""
    from core.ladder.context import GENRE_MIXED, GENRE_DRAWING, DocumentContext
    from core.providers import raster_drawing

    kinds = iter([raster_drawing.KIND_DRAWING] + [raster_drawing.KIND_TEXT_SCAN] * 7)
    monkeypatch.setattr(raster_drawing, "analyse",
                        lambda data: raster_drawing.RasterVerdict(next(kinds), 0.8, "тест", {}))
    context = DocumentContext(uri="x", file_type="pdf", data=_scanned_drawing_pdf(pages=8))
    assert context.classification.genre not in (GENRE_MIXED, GENRE_DRAWING)
    assert context.classification.signals.get("drawing_share", 0) == 0


def test_drawing_branch_renders_pdf_scan_pages(monkeypatch):
    from core.drawing_processor import DrawingProcessor

    data = _scanned_drawing_pdf(pages=2)

    class Storage:
        def read_bytes(self, uri):
            return data

    processor = DrawingProcessor(Storage())
    image = processor._open("src/sheet.pdf", page=2)
    assert image is not None and image.size[0] > 1000
    assert processor._open("src/sheet.pdf", page=3) is None
