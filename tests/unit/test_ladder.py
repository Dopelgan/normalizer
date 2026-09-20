"""Лестница стратегий: классификация, уровни 1-3, правила подъёма и выбора."""

import io

import pytest

from core import filetypes
from core.ladder.base import Strategy
from core.ladder.context import (
    GENRE_DRAWING,
    GENRE_MIXED,
    GENRE_SPREADSHEET,
    GENRE_TEXT,
    SOURCE_CAD,
    SOURCE_NATIVE,
    SOURCE_RASTER,
    SOURCE_VECTOR_TEXT,
    DocumentContext,
)
from core.ladder.context import assess_raster
from core.ladder.router import LadderParserProvider, LadderRouter
from core.ladder.strategies import recognition
from core.ladder.strategies.recognition import RestoreThenLayoutStrategy
from core.ladder.scoring import score_result
from core.ladder.strategies.cad_source import CadSourceStrategy
from core.ladder.strategies.native_text import (
    DocxStrategy,
    PdfTextLayerStrategy,
    PlainTextStrategy,
)
from core.ladder.strategies.tabular import SpreadsheetStrategy
from core.providers import vector_drawing
from core.models.parse_result import ParsedBlock, ParseResult
from core.providers.document_parser import ParserFailed


def context(data: bytes, file_type: str, **metadata) -> DocumentContext:
    return DocumentContext(uri="documents/x", file_type=file_type,
                           metadata=metadata, data=data)


class TestLayoutSkippedWithoutMineru:
    """
    Когда сервис разбора лежит, уровень 5 выродится в тот же полный OCR,
    что уже сделал уровень 4. Раньше это стоило второго прохода OCR на
    каждом скане — вдвое дольше и без единого нового символа.
    """

    def setup_method(self):
        from core.providers.mineru_parser import MinerUParserProvider

        MinerUParserProvider.reset_availability()

    teardown_method = setup_method

    def test_skipped_while_service_is_down(self):
        from core.ladder.strategies.recognition import LayoutStrategy
        from core.providers.mineru_parser import MinerUParserProvider

        ctx = context(b"%PDF-1.4", "pdf")
        strategy = LayoutStrategy()
        assert strategy.applicable(ctx) is True

        MinerUParserProvider.mark_unavailable()
        assert strategy.applicable(ctx) is False

    def test_availability_returns_after_ttl(self, monkeypatch):
        from core.providers.mineru_parser import MinerUParserProvider

        MinerUParserProvider.mark_unavailable()
        assert MinerUParserProvider.is_available() is False
        monkeypatch.setattr(
            "core.providers.mineru_parser.time.time",
            lambda: MinerUParserProvider._unavailable_until + 1,
        )
        assert MinerUParserProvider.is_available() is True


class TestOcrPageLimits:
    """Потолки OCR: страницы и размер листа."""

    def test_large_page_is_downscaled(self, monkeypatch):
        from PIL import Image

        from core.config import settings
        from core.providers.tesseract_fallback import TesseractFallbackProvider

        monkeypatch.setattr(settings, "OCR_MAX_SIDE", 1000)
        image = Image.new("RGB", (6614, 4677), "white")
        fitted = TesseractFallbackProvider._fit(image)
        assert max(fitted.size) == 1000
        assert fitted.size[0] > fitted.size[1]

    def test_small_page_is_left_alone(self, monkeypatch):
        from PIL import Image

        from core.config import settings
        from core.providers.tesseract_fallback import TesseractFallbackProvider

        monkeypatch.setattr(settings, "OCR_MAX_SIDE", 3500)
        image = Image.new("RGB", (1200, 900), "white")
        assert TesseractFallbackProvider._fit(image) is image


class TestContextType:
    """
    Род содержимого решает, какой уровень лестницы возьмётся за документ,
    поэтому тип уточняется по сигнатуре, а не берётся из расширения на веру.
    """

    def test_declared_type_is_corrected_by_signature(self):
        docx = pytest.importorskip("docx")
        import io as _io

        document = docx.Document()
        document.add_paragraph("Спецификация к договору")
        buffer = _io.BytesIO()
        document.save(buffer)

        ctx = context(buffer.getvalue(), "pdf")
        assert ctx.file_type == "docx"
        assert ctx.kind == filetypes.KIND_OFFICE_TEXT
        assert ctx.declared_type == "pdf"
        assert "docx" in (ctx.type_mismatch or "")

    def test_matching_type_is_left_alone(self):
        ctx = context(b"%PDF-1.4 ...", "pdf")
        assert ctx.file_type == "pdf"
        assert ctx.type_mismatch is None

    def test_unknown_signature_keeps_declared_type(self):
        ctx = context("a;b;c".encode(), "csv")
        assert ctx.file_type == "csv"
        assert ctx.kind == filetypes.KIND_SPREADSHEET


# ===========================================================================
# Фикстуры документов
# ===========================================================================

@pytest.fixture
def dxf_bytes():
    ezdxf = pytest.importorskip("ezdxf")
    document = ezdxf.new(setup=True)
    space = document.modelspace()
    space.add_text("Ø20h7", height=3.5).set_placement((10, 50))
    space.add_text("R15", height=3.5).set_placement((40, 30))
    # Повёрнутая подпись размера — в растре она и теряется.
    space.add_text("85±0,2", height=3.5, dxfattribs={"rotation": 90}).set_placement((5, 20))
    space.add_text("Сталь 45 ГОСТ 1050-88", height=3.5).set_placement((180, 5))
    buffer = io.StringIO()
    document.write(buffer)
    return buffer.getvalue().encode("utf-8")


@pytest.fixture
def xlsx_bytes():
    openpyxl = pytest.importorskip("openpyxl")
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = "Ведомость"
    sheet.append(["Обозначение", "Наименование", "Кол."])
    sheet.append(["АБВГ.301261.005", "Вал ступенчатый", "2"])
    second = workbook.create_sheet("Материалы")
    second.append(["Материал", "ГОСТ"])
    second.append(["Сталь 45", "1050-88"])
    buffer = io.BytesIO()
    workbook.save(buffer)
    return buffer.getvalue()


@pytest.fixture
def docx_bytes():
    docx = pytest.importorskip("docx")
    document = docx.Document()
    document.add_heading("Раздел 1", level=1)
    document.add_paragraph("Текст раздела достаточной длины для фрагмента.")
    table = document.add_table(rows=2, cols=2)
    table.cell(0, 0).text = "Навык"
    table.cell(0, 1).text = "Балл"
    table.cell(1, 0).text = "Go"
    table.cell(1, 1).text = "2"
    buffer = io.BytesIO()
    document.save(buffer)
    return buffer.getvalue()


@pytest.fixture
def text_pdf_bytes():
    pymupdf = pytest.importorskip("pymupdf")
    document = pymupdf.open()
    for _ in range(2):
        page = document.new_page(width=600, height=800)
        for row in range(24):
            page.insert_text((60, 60 + row * 24), "plain running text line here", fontsize=11)
    data = document.tobytes()
    document.close()
    return data


@pytest.fixture
def ruled_pdf_bytes():
    """PDF с текстом и линейной графикой — таблицами."""
    pymupdf = pytest.importorskip("pymupdf")
    document = pymupdf.open()
    page = document.new_page(width=600, height=800)
    for row in range(10):
        page.insert_text((60, 60 + row * 24), "text with a table below", fontsize=11)
    for row in range(8):
        page.draw_line(pymupdf.Point(60, 400 + row * 20), pymupdf.Point(540, 400 + row * 20))
    data = document.tobytes()
    document.close()
    return data


# ===========================================================================
# КЛ-1, КЛ-2, КЛ-3
# ===========================================================================

class TestClassification:
    def test_cad_source(self, dxf_bytes):
        classification = context(dxf_bytes, "dxf").classification
        assert classification.source == SOURCE_CAD
        assert classification.genre == GENRE_DRAWING

    def test_spreadsheet(self, xlsx_bytes):
        classification = context(xlsx_bytes, "xlsx").classification
        assert classification.source == SOURCE_NATIVE
        assert classification.genre == GENRE_SPREADSHEET

    def test_plain_text(self):
        classification = context(b"just text", "txt").classification
        assert classification.source == SOURCE_NATIVE
        assert classification.genre == GENRE_TEXT

    def test_vector_pdf_without_graphics_is_plain_text(self, text_pdf_bytes):
        classification = context(text_pdf_bytes, "pdf").classification
        assert classification.source == SOURCE_VECTOR_TEXT
        assert classification.genre == GENRE_TEXT

    def test_vector_pdf_with_rules_is_mixed(self, ruled_pdf_bytes):
        """Линейная графика означает структуру, которую слой не опишет."""
        classification = context(ruled_pdf_bytes, "pdf").classification
        assert classification.source == SOURCE_VECTOR_TEXT
        assert classification.genre == GENRE_MIXED

    def test_scanned_pdf_is_raster(self):
        pymupdf = pytest.importorskip("pymupdf")
        document = pymupdf.open()
        document.new_page(width=600, height=800)
        data = document.tobytes()
        document.close()
        classification = context(data, "pdf").classification
        assert classification.source == SOURCE_RASTER


class TestDrawingShare:
    """Чертёжность документа считается долей листов, а не их наличием."""

    @staticmethod
    def _verdicts(drawing_pages: int, total: int):
        return {
            page: vector_drawing.DrawingPage(page, 0.9 if page <= drawing_pages else 0.0)
            for page in range(1, total + 1)
        }

    def test_one_drawing_page_of_three_is_not_a_drawing_album(self, mocker, text_pdf_bytes):
        """
        Целочисленное `len // 2` объявляло чертежом документ с одной
        чертёжной страницей из трёх. Жанр правит меткой доступа: приложение
        к договору переводило в `confidential` весь договор.
        """
        mocker.patch.object(
            vector_drawing, "detect_drawing_pages", return_value=self._verdicts(1, 3)
        )
        classification = context(text_pdf_bytes, "pdf").classification
        assert classification.genre == GENRE_MIXED
        assert classification.signals["drawing_share"] == 0.333

    def test_two_of_three_is_a_drawing_album(self, mocker, text_pdf_bytes):
        mocker.patch.object(
            vector_drawing, "detect_drawing_pages", return_value=self._verdicts(2, 3)
        )
        assert context(text_pdf_bytes, "pdf").classification.genre == GENRE_DRAWING

    def test_single_page_drawing_stays_a_drawing(self, mocker, text_pdf_bytes):
        mocker.patch.object(
            vector_drawing, "detect_drawing_pages", return_value=self._verdicts(1, 1)
        )
        assert context(text_pdf_bytes, "pdf").classification.genre == GENRE_DRAWING


class TestRasterAssessment:
    def test_quality_and_signals(self):
        Image = pytest.importorskip("PIL.Image", reason="нужен Pillow")
        pytest.importorskip("numpy")
        from PIL import Image, ImageDraw

        image = Image.new("L", (1600, 1200), color=255)
        draw = ImageDraw.Draw(image)
        for row in range(20):
            draw.text((40, 40 + row * 40), "contrasty sample line", fill=0)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")

        classification = context(buffer.getvalue(), "png").classification
        assert classification.source == SOURCE_RASTER
        assert 0.0 <= classification.raster_quality <= 1.0
        assert classification.signals["width"] == 1600

    def test_tiny_low_contrast_image_needs_restoration(self):
        pytest.importorskip("numpy")
        from PIL import Image

        # Маленькая и почти однотонная — читать нечего без выправления.
        image = Image.new("L", (200, 140), color=140)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")

        classification = context(buffer.getvalue(), "png").classification
        assert classification.needs_restoration is True


# ===========================================================================
# Уровни 1-3
# ===========================================================================

class TestLevelOneCad:
    def test_fields_are_read_without_recognition(self, dxf_bytes):
        ctx = context(dxf_bytes, "dxf")
        strategy = CadSourceStrategy()
        assert strategy.applicable(ctx)

        result = strategy.run(ctx)
        block = result.blocks[0]
        assert block.type == "drawing"
        assert block.method == "cad_source"
        values = {f["value"] for f in block.drawing_fields}
        assert {"Ø20h7", "R15", "85±0,2"} <= values

    def test_rotated_dimension_keeps_its_tolerance(self, dxf_bytes):
        result = CadSourceStrategy().run(context(dxf_bytes, "dxf"))
        rotated = [f for f in result.blocks[0].drawing_fields if f["value"] == "85±0,2"]
        assert rotated[0]["tolerance"] == "±0,2"

    def test_categories_are_recognised(self, dxf_bytes):
        result = CadSourceStrategy().run(context(dxf_bytes, "dxf"))
        categories = {f["category"] for f in result.blocks[0].drawing_fields}
        assert {"size", "radius", "material"} <= categories

    def test_sheet_is_a_unit(self, dxf_bytes):
        result = CadSourceStrategy().run(context(dxf_bytes, "dxf"))
        assert result.blocks[0].sheet_name == "Model"

    def test_not_applicable_to_pdf(self, text_pdf_bytes):
        assert not CadSourceStrategy().applicable(context(text_pdf_bytes, "pdf"))


class TestLevelTwo:
    def test_pdf_text_layer(self, text_pdf_bytes):
        ctx = context(text_pdf_bytes, "pdf")
        result = PdfTextLayerStrategy().run(ctx)
        assert result.page_count == 2
        assert all(b.method == "text_layer" for b in result.blocks)
        assert result.text_layer_chars

    def test_docx_keeps_tables_and_headings(self, docx_bytes):
        result = DocxStrategy().run(context(docx_bytes, "docx"))
        types = [b.type for b in result.blocks]
        assert "table" in types
        table = [b for b in result.blocks if b.type == "table"][0]
        assert table.table_data["headers"] == ["Навык", "Балл"]
        assert result.blocks[0].section_title == "Раздел 1"

    def test_markdown_heading_becomes_section(self):
        data = "# Заголовок\n\nПервый абзац.\n\nВторой абзац.".encode()
        result = PlainTextStrategy().run(context(data, "md"))
        assert len(result.blocks) == 3
        assert result.blocks[1].section_title == "Заголовок"

    def test_cp1251_text_is_decoded(self):
        result = PlainTextStrategy().run(context("Русский текст".encode("cp1251"), "txt"))
        assert "Русский текст" in result.blocks[0].text


class TestLevelThree:
    def test_every_sheet_is_a_unit(self, xlsx_bytes):
        result = SpreadsheetStrategy().run(context(xlsx_bytes, "xlsx"))
        assert [b.sheet_name for b in result.blocks] == ["Ведомость", "Материалы"]
        assert all(b.type == "table" for b in result.blocks)

    def test_headers_and_rows(self, xlsx_bytes):
        result = SpreadsheetStrategy().run(context(xlsx_bytes, "xlsx"))
        table = result.blocks[0].table_data
        assert table["headers"] == ["Обозначение", "Наименование", "Кол."]
        assert table["rows"][0][1] == "Вал ступенчатый"

    def test_semicolon_csv(self):
        data = "имя;значение\nальфа;1\nбета;2\n".encode("cp1251")
        result = SpreadsheetStrategy().run(context(data, "csv"))
        assert result.blocks[0].table_data["headers"] == ["имя", "значение"]

    def test_exhaustive_extraction_scores_full(self, xlsx_bytes):
        """Формат отдал всё — полнота единица, а не оценка по объёму текста."""
        ctx = context(xlsx_bytes, "xlsx")
        score = score_result(SpreadsheetStrategy().run(ctx), ctx, exhaustive=True)
        assert score.completeness == 1.0
        assert score.value >= 0.9


# ===========================================================================
# Маршрутизатор
# ===========================================================================

class FakeStrategy(Strategy):
    """Управляемая стратегия: нужна, чтобы проверять правила подъёма."""

    def __init__(self, level, confidence=1.0, blocks=1, fail=False, structured=False,
                 fallback=False):
        self.level = level
        self.name = f"fake-{level}"
        self.method = f"fake_{level}"
        self.exhaustive = True
        self._confidence = confidence
        self._blocks = blocks
        self._fail = fail
        self._structured = structured
        self._fallback = fallback
        self.ran = False

    def applicable(self, context):
        return True

    def run(self, context):
        self.ran = True
        if self._fail:
            raise ValueError("уровень не справился")
        blocks = [
            ParsedBlock(type="table" if self._structured else "text",
                        text="Достаточно длинный текст фрагмента документа.",
                        table_data={"headers": ["a"], "rows": [["1"]]} if self._structured else None,
                        page=1, confidence=self._confidence, method=self.method)
            for _ in range(self._blocks)
        ]
        return self.build_result(
            blocks, source_kind="test", page_count=1, is_fallback=self._fallback
        )


class TestRouter:
    def test_stops_at_first_sufficient_level(self):
        cheap = FakeStrategy(2, confidence=1.0)
        expensive = FakeStrategy(5, confidence=1.0)
        result = LadderRouter([cheap, expensive], threshold=0.75).parse(
            context(b"text", "txt")
        )
        assert cheap.ran and not expensive.ran
        assert result.ladder["chosen_level"] == 2

    def test_fallback_does_not_stop_the_ladder(self):
        """
        Плоское распознавание не вернёт ни таблицы, ни формулы, и его высокая
        оценка означает лишь «символов много». На листе с формулами именно
        так лестница и останавливалась, не доходя до разбора с детекцией.
        """
        ocr = FakeStrategy(4, confidence=1.0, fallback=True)
        layout = FakeStrategy(5, confidence=1.0, structured=True)
        result = LadderRouter([ocr, layout], threshold=0.75).parse(
            context(b"text", "png")
        )
        assert ocr.ran and layout.ran
        assert result.ladder["chosen_level"] == 5

    def test_fallback_wins_when_nothing_better_exists(self):
        ocr = FakeStrategy(4, confidence=1.0, fallback=True)
        broken = FakeStrategy(5, fail=True)
        result = LadderRouter([ocr, broken], threshold=0.75).parse(
            context(b"text", "png")
        )
        assert result.ladder["chosen_level"] == 4

    def test_structural_result_outranks_a_higher_scoring_fallback(self):
        """
        Балл меряет объём и связность текста, а не структуру. Плоское
        распознавание набирает его на любой странице, и сравнение по одному
        баллу отдавало победу фоллбэку: разбор с таблицами проигрывал
        листу текста, из которого нечего взять.
        """
        ocr = FakeStrategy(4, confidence=1.0, fallback=True)
        layout = FakeStrategy(5, confidence=0.85, structured=True)
        result = LadderRouter([ocr, layout], threshold=0.75).parse(
            context(b"text", "png")
        )
        scores = {a["level"]: a["score"] for a in result.ladder["attempts"]}
        assert scores[4] > scores[5]
        assert result.ladder["chosen_level"] == 5

    def test_structural_result_outranks_fallback_below_threshold_too(self):
        """
        Регрессия на разобранный случай `formulas.png`: скан с формулами.
        Tesseract набирал 0.87 объёмом каши `д = —fz+ay`, разбор с детекцией
        областей — 0.39, потому что формулы дают мало символов. Старшинство
        структурного разбора кончалось у порога, и в индекс уезжала каша.
        Теперь порог решает только, останавливаться ли, а не кто старше.
        """
        ocr = FakeStrategy(4, confidence=1.0, fallback=True)
        layout = FakeStrategy(5, confidence=0.2, structured=True)
        result = LadderRouter([ocr, layout], threshold=0.75).parse(
            context(b"text", "png")
        )
        scores = {a["level"]: a["score"] for a in result.ladder["attempts"]}
        assert scores[4] > scores[5]
        assert result.ladder["chosen_level"] == 5
        assert result.ladder["fallback_won"] is False

    def test_fallback_wins_when_structural_lost_the_text(self):
        """
        Оговорка к старшинству: разбор, потерявший текст, его не получает.
        Иначе MinerU, вернувший со страницы три символа, побеждал бы
        распознавание всей страницы просто по признаку «структурный».
        """
        ocr = FakeStrategy(4, confidence=1.0, fallback=True, blocks=10)
        thin = FakeStrategy(5, confidence=1.0, structured=True, blocks=1)
        result = LadderRouter([ocr, thin], threshold=0.75).parse(
            context(b"text", "png")
        )
        assert result.ladder["chosen_level"] == 4

    def test_fallback_win_is_recorded_as_degradation(self):
        """
        Победивший фоллбэк при живом структурном уровне — деградация. Раньше
        документ выглядел разобранным, и отличить это от «документ такой»
        можно было только чтением логов.
        """
        ocr = FakeStrategy(4, confidence=1.0, fallback=True, blocks=10)
        thin = FakeStrategy(5, confidence=1.0, structured=True, blocks=1)
        result = LadderRouter([ocr, thin], threshold=0.75).parse(
            context(b"text", "png")
        )
        assert result.ladder["fallback_won"] is True
        assert result.degraded and "фоллбэк выиграл" in result.degraded[0]

    def test_attempts_report_volume_and_kind(self):
        """Отчёт лестницы должен давать чем сравнивали, а не только балл."""
        ocr = FakeStrategy(4, confidence=1.0, fallback=True)
        layout = FakeStrategy(5, confidence=1.0, structured=True)
        result = LadderRouter([ocr, layout], threshold=0.75).parse(
            context(b"text", "png")
        )
        by_level = {a["level"]: a for a in result.ladder["attempts"]}
        assert by_level[4]["is_fallback"] is True
        assert by_level[5]["is_fallback"] is False
        assert by_level[4]["chars"] > 0 and by_level[5]["chars"] > 0

    def test_escalates_when_below_threshold(self):
        weak = FakeStrategy(2, confidence=0.2)
        strong = FakeStrategy(5, confidence=1.0)
        result = LadderRouter([weak, strong], threshold=0.75).parse(context(b"text", "txt"))
        assert weak.ran and strong.ran
        assert result.ladder["chosen_level"] == 5

    def test_best_result_wins_not_the_last(self):
        """Результаты уровней не смешиваются — выбирается лучший по оценке."""
        good = FakeStrategy(2, confidence=0.70)
        worse = FakeStrategy(5, confidence=0.30)
        result = LadderRouter([good, worse], threshold=0.99).parse(context(b"text", "txt"))
        assert result.ladder["chosen_level"] == 2
        assert [a["level"] for a in result.ladder["attempts"]] == [2, 5]

    def test_failed_level_does_not_stop_the_ladder(self):
        broken = FakeStrategy(2, fail=True)
        working = FakeStrategy(5, confidence=1.0)
        result = LadderRouter([broken, working], threshold=0.75).parse(context(b"t", "txt"))
        assert result.ladder["chosen_level"] == 5
        assert "error" in result.ladder["attempts"][0]

    def test_all_levels_failing_is_an_error(self):
        strategies = [FakeStrategy(2, fail=True), FakeStrategy(5, fail=True)]
        with pytest.raises(ParserFailed):
            LadderRouter(strategies, threshold=0.75).parse(context(b"t", "txt"))

    def test_no_applicable_strategy_is_an_error(self):
        with pytest.raises(ParserFailed):
            LadderRouter([], threshold=0.75).parse(context(b"t", "txt"))

    def test_disabled_levels_are_excluded(self):
        expensive = FakeStrategy(5)
        router = LadderRouter([FakeStrategy(2), expensive], enabled_levels=[2])
        assert [s.level for s in router.strategies] == [2]

    def test_attempts_are_reported(self):
        router = LadderRouter([FakeStrategy(2, confidence=0.1), FakeStrategy(5)],
                              threshold=0.75)
        result = router.parse(context(b"t", "txt"))
        assert result.ladder["classification"]["source"] == SOURCE_NATIVE
        assert len(result.ladder["attempts"]) == 2


class TestRestorationComparison:
    """В-5: «стало ли лучше» меряется в одной шкале с исходной оценкой."""

    @staticmethod
    def _scanned_page(size=(600, 400)):
        """Страница со строками текста и лёгким зерном — как обычный скан."""
        Image = pytest.importorskip("PIL.Image")
        numpy = pytest.importorskip("numpy")
        width, height = size
        sheet = numpy.full((height, width), 245, dtype=numpy.uint8)
        for top in range(10, height - 10, 26):
            sheet[top:top + 9, 30:width - 30] = 40
        rng = numpy.random.default_rng(3)
        grain = rng.integers(-4, 4, sheet.shape)
        sheet = numpy.clip(sheet.astype(numpy.int16) + grain, 0, 255).astype(numpy.uint8)
        buffer = io.BytesIO()
        Image.fromarray(sheet, mode="L").save(buffer, format="PNG")
        return buffer.getvalue()

    @staticmethod
    def _upscaled(data: bytes, factor: int):
        Image = pytest.importorskip("PIL.Image")
        image = Image.open(io.BytesIO(data))
        image = image.resize((image.width * factor, image.height * factor), Image.LANCZOS)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()

    def test_upscaling_alone_does_not_count_as_improvement(self):
        """
        В оценку входит разрешение: увеличение вдвое даёт вчетверо больше
        пикселей и «улучшение» из воздуха. Ровно так сравнивались рендер PDF
        в 150 dpi и увеличенный вдвое PNG.
        """
        pytest.importorskip("numpy")
        original = self._scanned_page()
        before, signals = assess_raster(original)
        bigger = self._upscaled(original, 2)

        naive, _ = assess_raster(bigger)
        same_scale = RestoreThenLayoutStrategy._quality_after(bigger, signals)

        # Порог подъёма — 0.02: наивное сравнение перешагивает его на одном
        # только размере, сравнение в одной шкале — нет.
        assert naive > before + 0.02
        assert same_scale <= before + 0.02

    def test_unknown_source_size_falls_back_to_plain_assessment(self):
        pytest.importorskip("numpy")
        data = self._scanned_page((300, 200))
        plain, _ = assess_raster(data)
        assert RestoreThenLayoutStrategy._quality_after(data, {}) == plain


class TestStrategyConstruction:
    """Создание уровней: выключенные не строятся, хранилище одно на всех."""

    def test_disabled_levels_are_not_constructed(self, mocker):
        """
        Конструктор уровня 4 поднимает распознавание, уровня 5 — клиента
        парсера. Раньше создавались все девять уровней и только потом
        отбрасывались лишние — плата за выключенный уровень бралась всё равно.
        """
        tesseract = mocker.patch.object(recognition, "TesseractFallbackProvider")
        mineru = mocker.patch.object(recognition, "MinerUParserProvider")
        router = LadderRouter(enabled_levels=[1, 2])
        assert {s.level for s in router.strategies} == {1, 2}
        tesseract.assert_not_called()
        mineru.assert_not_called()

    def test_storage_reaches_the_levels_that_read_files(self):
        """Своё хранилище у каждого уровня — это свой клиент S3 на документ."""
        storage = object()
        router = LadderRouter(enabled_levels=[4, 5, 6], storage=storage)
        by_level = {s.level: s for s in router.strategies}
        assert by_level[4].provider.storage is storage
        assert by_level[5].provider.storage is storage
        assert by_level[6].layout.provider.storage is storage

    def test_provider_hands_its_storage_to_the_ladder(self):
        storage = object()
        provider = LadderParserProvider(storage=storage)
        layout = next(s for s in provider.router.strategies if s.level == 5)
        assert layout.provider.storage is storage


class TestRouterOnRealDocuments:
    def test_plain_pdf_stops_at_text_layer(self, text_pdf_bytes):
        """Простой текстовый PDF не должен тянуть за собой тяжёлый разбор."""
        router = LadderRouter(enabled_levels=[2, 5])
        result = router.parse(context(text_pdf_bytes, "pdf"))
        assert result.ladder["chosen_level"] == 2

    def test_structured_pdf_does_not_settle_for_text_layer(self, ruled_pdf_bytes):
        """
        У документа со структурой один текстовый слой — структурная
        неполнота, и оценка уровня 2 обязана остаться ниже порога.
        """
        ctx = context(ruled_pdf_bytes, "pdf")
        score = score_result(PdfTextLayerStrategy().run(ctx), ctx, exhaustive=True)
        assert not score.structure_ok
        assert score.value < 0.75

    def test_spreadsheet_routes_to_level_three(self, xlsx_bytes):
        result = LadderRouter().parse(context(xlsx_bytes, "xlsx"))
        assert result.ladder["chosen_level"] == 3

    def test_cad_routes_to_level_one(self, dxf_bytes):
        result = LadderRouter().parse(context(dxf_bytes, "dxf"))
        assert result.ladder["chosen_level"] == 1
