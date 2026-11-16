"""
LADDER_SKIP_OCR_ON_TEXT_LAYER: уровень 4 не запускается на PDF с полным
текстовым слоем, его роль (мерка сохранности, запас для чертёжной ветки,
пометка о деградации) берёт уровень 2.
"""

import pytest

from core import telemetry
from core.config import settings
from core.ladder import drawing_route, router as router_module
from core.ladder.base import Strategy
from core.ladder.context import DocumentContext
from core.ladder.router import LadderRouter
from core.ladder.text_layer_complete import CompleteLayer, complete_text_layer
from core.models.parse_result import ParsedBlock


def pdf(pages_text):
    fitz = pytest.importorskip("fitz")
    document = fitz.open()
    for text in pages_text:
        page = document.new_page()
        for row, line in enumerate(text.split("\n")):
            if line:
                page.insert_text((50, 60 + row * 14), line, fontsize=10)
    return document.tobytes()


def context(data=b"%PDF-1.4", file_type="pdf"):
    return DocumentContext(uri="documents/x", file_type=file_type, metadata={}, data=data)


LINE = "plain running text of the document body line"
FULL_PAGE = "\n".join([LINE] * 10)          # ~440 символов


class TestCompleteTextLayer:
    def test_every_page_with_text_is_complete(self):
        layer = complete_text_layer(context(pdf([FULL_PAGE, FULL_PAGE, FULL_PAGE])))
        assert layer is not None
        assert layer.pages == 3 and layer.chars > 1000

    def test_one_page_without_text_keeps_ocr(self):
        assert complete_text_layer(context(pdf([FULL_PAGE, "", FULL_PAGE]))) is None

    def test_thin_layer_keeps_ocr(self):
        assert complete_text_layer(context(pdf([LINE, LINE, LINE]))) is None

    def test_image_is_never_complete(self):
        from PIL import Image
        import io

        buffer = io.BytesIO()
        Image.new("RGB", (200, 100), "white").save(buffer, format="PNG")
        assert complete_text_layer(context(buffer.getvalue(), "png")) is None


class Fixed(Strategy):
    """Уровень с заданным объёмом текста."""

    def __init__(self, level, chars=400, fail=False, fallback=False, structured=False,
                 confidence=None):
        self.level = level
        self.name = f"fixed-{level}"
        self.method = f"fixed_{level}"
        self.exhaustive = True
        self._chars, self._fail, self._fallback, self._structured = chars, fail, fallback, structured
        # Структурный уровень по умолчанию увереннее слоя: на равенстве
        # сохранности побеждает по баллу, как MinerU на векторном PDF.
        self._confidence = confidence if confidence is not None else (0.95 if structured else 0.6)
        self.ran = False

    def applicable(self, context):
        return True

    def run(self, context):
        self.ran = True
        if self._fail:
            raise ValueError("уровень не справился")
        block = ParsedBlock(
            type="table" if self._structured else "text", text="ж" * self._chars,
            table_data={"headers": ["a"], "rows": [["1"]]} if self._structured else None,
            page=1, confidence=self._confidence, method=self.method,
        )
        return self.build_result([block], source_kind="vector_pdf", page_count=1,
                                 is_fallback=self._fallback)


@pytest.fixture
def skip_on(monkeypatch):
    monkeypatch.setattr(settings, "LADDER_SKIP_OCR_ON_TEXT_LAYER", True)
    monkeypatch.setattr(router_module, "complete_text_layer",
                        lambda ctx: CompleteLayer(pages=1, chars=400))


def ladder(*strategies):
    return LadderRouter(list(strategies), threshold=0.75)


class TestRouterSkip:
    def test_level4_is_skipped_and_timed_as_zero(self, skip_on):
        ocr = Fixed(4, fallback=True)
        with telemetry.collect() as timings:
            result = ladder(Fixed(2), ocr, Fixed(5, structured=True)).parse(context())
        assert not ocr.ran
        assert timings["parse.level4_ms"] == 0
        assert [a["level"] for a in result.ladder["attempts"]] == [2, 5]
        assert result.ladder["chosen_level"] == 5

    def test_flag_off_runs_level4(self, monkeypatch):
        monkeypatch.setattr(settings, "LADDER_SKIP_OCR_ON_TEXT_LAYER", False)
        monkeypatch.setattr(router_module, "complete_text_layer",
                            lambda ctx: CompleteLayer(pages=1, chars=400))
        ocr = Fixed(4, fallback=True)
        ladder(Fixed(2), ocr, Fixed(5, structured=True)).parse(context())
        assert ocr.ran

    def test_incomplete_layer_keeps_level4(self, monkeypatch):
        monkeypatch.setattr(settings, "LADDER_SKIP_OCR_ON_TEXT_LAYER", True)
        monkeypatch.setattr(router_module, "complete_text_layer", lambda ctx: None)
        ocr = Fixed(4, fallback=True)
        ladder(Fixed(2), ocr, Fixed(5, structured=True)).parse(context())
        assert ocr.ran

    def test_without_level2_level4_is_kept(self, skip_on):
        ocr = Fixed(4, fallback=True)
        ladder(ocr, Fixed(5, structured=True)).parse(context())
        assert ocr.ran

    def test_level5_failing_after_skip_is_marked_like_a_fallback_win(self, skip_on):
        result = ladder(Fixed(2), Fixed(4, fallback=True), Fixed(5, fail=True)).parse(context())
        assert result.ladder["chosen_level"] == 2
        assert result.ladder["fallback_won"] is True
        assert any(note.startswith("фоллбэк выиграл сравнение: fixed-2") for note in result.degraded)

    def test_layer_is_the_measure_of_kept_text(self, skip_on):
        """Структурный уровень, потерявший больше половины текста слоя, старшинства не получает."""
        result = ladder(Fixed(2, chars=1000), Fixed(4, fallback=True),
                        Fixed(5, chars=100, structured=True)).parse(context())
        assert result.ladder["chosen_level"] == 2

    def test_same_choice_as_with_ocr_of_the_same_volume(self, monkeypatch, skip_on):
        """OCR векторного PDF приносит столько же текста, сколько слой: выбор тот же."""
        cases = [(1000, 900), (1000, 100), (400, 400)]
        for layer_chars, layout_chars in cases:
            on = ladder(Fixed(2, chars=layer_chars), Fixed(4, chars=layer_chars, fallback=True),
                        Fixed(5, chars=layout_chars, structured=True)).parse(context())
            monkeypatch.setattr(settings, "LADDER_SKIP_OCR_ON_TEXT_LAYER", False)
            off = ladder(Fixed(2, chars=layer_chars), Fixed(4, chars=layer_chars, fallback=True),
                         Fixed(5, chars=layout_chars, structured=True)).parse(context())
            monkeypatch.setattr(settings, "LADDER_SKIP_OCR_ON_TEXT_LAYER", True)
            assert on.ladder["chosen_level"] == off.ladder["chosen_level"], (layer_chars, layout_chars)

    def test_drawing_route_gets_the_layer_as_reserve(self, skip_on, monkeypatch):
        monkeypatch.setattr(drawing_route, "deferred", lambda ctx: True)
        layer = Fixed(2, chars=1000)
        result = ladder(layer, Fixed(4, fallback=True),
                        Fixed(5, chars=100, structured=True)).parse(context())
        assert result.ladder["chosen_level"] == 5
        assert result.fallback_reserve is not None
        assert result.fallback_reserve.parser_name == "fixed-2"
