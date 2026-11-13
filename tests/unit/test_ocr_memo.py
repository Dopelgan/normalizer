"""
Одна страница скана — одно распознавание на документ.

Лестница на скане запускает уровень 4 (plain_ocr, `parse_all_pages`) и
уровень 5, где после MinerU тот же Tesseract снова читает те же страницы
(`text_layer`). Оба прохода рендерят страницы одинаково и дают один и тот
же результат — второй проход был бы чистой потерей времени (~40 % Tesseract).
"""

from PIL import Image, ImageDraw

from core.config import settings
from core.ladder.base import Strategy
from core.ladder.context import DocumentContext
from core.ladder.router import LadderRouter
from core.ladder.strategies.recognition import PlainOcrStrategy
from core.models.parse_result import ParsedBlock
from core.providers import tesseract_fallback
from core.providers.tesseract_fallback import TesseractFallbackProvider, ocr_memo


def _page(text_x: int = 10) -> Image.Image:
    image = Image.new("RGB", (1000, 700), "white")
    ImageDraw.Draw(image).rectangle((text_x, 10, text_x + 200, 40), fill="black")
    return image


class _Tesseract:
    """Подмена `image_to_data`: считает вызовы и отдаёт одну строку."""

    def __init__(self, words=("Лист", "1")):
        self.calls = 0
        self.words = list(words)

    def __call__(self, image, lang=None, config=None, output_type=None):
        self.calls += 1
        n = len(self.words)
        return {
            "text": self.words, "conf": [90] * n,
            "block_num": [1] * n, "par_num": [1] * n, "line_num": [1] * n,
            "left": [10 + 60 * i for i in range(n)], "top": [10] * n,
            "width": [50] * n, "height": [20] * n,
        }


def _patch(monkeypatch, fake):
    monkeypatch.setattr(settings, "OCR_PREPROCESS_ENABLED", False)
    # Память прохода считается по странице и языку. Здесь проверяется
    # сама память, а не выбор языка, поэтому построчный выбор
    # (OCR_PER_LINE_LANGUAGE) закреплён выключенным: с ним на страницу
    # приходится по вызову на язык, и счёт вызовов стал бы про другое.
    monkeypatch.setattr(settings, "OCR_PER_LINE_LANGUAGE", False)
    monkeypatch.setattr(tesseract_fallback.pytesseract, "image_to_data", fake)


def test_same_page_read_once_across_providers(monkeypatch):
    fake = _Tesseract()
    _patch(monkeypatch, fake)
    with ocr_memo():
        first = TesseractFallbackProvider(storage=object()).ocr_lines(_page())
        second = TesseractFallbackProvider(storage=object()).ocr_lines(_page())
    assert fake.calls == 1
    assert first == second and first
    # Копии: правка рамки у одного прохода не протекает в другой.
    first[0].bbox[0] = 0.99
    assert second[0].bbox[0] != 0.99


def test_without_memo_every_call_reads(monkeypatch):
    fake = _Tesseract()
    _patch(monkeypatch, fake)
    provider = TesseractFallbackProvider(storage=object())
    provider.ocr_lines(_page())
    provider.ocr_lines(_page())
    assert fake.calls == 2


def test_different_pages_and_modes_are_not_confused(monkeypatch):
    fake = _Tesseract()
    _patch(monkeypatch, fake)
    provider = TesseractFallbackProvider(storage=object())
    with ocr_memo():
        provider.ocr_lines(_page(10))
        provider.ocr_lines(_page(400))
        provider.ocr_lines(_page(10), psm=11)
        provider.ocr_lines(_page(10).convert("L"))
    assert fake.calls == 4


def test_empty_result_is_not_remembered(monkeypatch):
    """Пустой ответ неотличим от сбоя Tesseract — страница читается снова."""
    fake = _Tesseract(words=())
    _patch(monkeypatch, fake)
    provider = TesseractFallbackProvider(storage=object())
    with ocr_memo():
        assert provider.ocr_lines(_page()) == []
        assert provider.ocr_lines(_page()) == []
    assert fake.calls == 2


def test_memo_is_dropped_after_block(monkeypatch):
    fake = _Tesseract()
    _patch(monkeypatch, fake)
    provider = TesseractFallbackProvider(storage=object())
    with ocr_memo():
        with ocr_memo():  # вложенный блок пользуется внешней памятью
            provider.ocr_lines(_page())
        provider.ocr_lines(_page())
    provider.ocr_lines(_page())
    assert fake.calls == 2
    assert tesseract_fallback._MEMO.get() is None


class _LayerLevel(Strategy):
    """Уровень 5 в миниатюре: структура плюс слой распознавания тех же страниц."""

    level = 5
    name = "layout"
    method = "mineru_ocr"

    def __init__(self, storage):
        self.storage = storage
        self.layer = None

    def applicable(self, context):
        return True

    def run(self, context):
        self.layer = TesseractFallbackProvider(storage=self.storage).text_layer(
            context.uri, context.file_type
        )
        blocks = [ParsedBlock(type="table", text=" ".join(l.text for l in self.layer.lines(1)),
                              table_data={"headers": ["a"], "rows": [["1"]]},
                              page=1, confidence=0.95, method=self.method)]
        return self.build_result(blocks, source_kind="image", page_count=1)


def _run_ladder(monkeypatch, temp_storage):
    import io

    buffer = io.BytesIO()
    _page().save(buffer, format="PNG")
    temp_storage.write_file("documents/sheet.png", buffer.getvalue())
    fake = _Tesseract()
    _patch(monkeypatch, fake)
    layout = _LayerLevel(temp_storage)
    router = LadderRouter([PlainOcrStrategy(storage=temp_storage), layout], threshold=0.75)
    context = DocumentContext(uri="documents/sheet.png", file_type="png",
                              metadata={}, storage=temp_storage, data=buffer.getvalue())
    return router.parse(context), layout, fake


def test_ladder_reads_scan_once_and_keeps_both_results(monkeypatch, temp_storage):
    result, layout, fake = _run_ladder(monkeypatch, temp_storage)
    assert fake.calls == 1
    # Уровень 4 отработал и вошёл в сравнение — выбор тот же.
    attempts = result.ladder["attempts"]
    assert [a["strategy"] for a in attempts] == ["plain_ocr", "layout"]
    assert attempts[0]["chars"] > 0
    assert result.ladder["chosen_strategy"] == "layout"
    assert [line.text for line in layout.layer.lines(1)] == ["Лист 1"]
    assert tesseract_fallback._MEMO.get() is None


def test_palette_is_part_of_the_page(monkeypatch):
    fake = _Tesseract()
    _patch(monkeypatch, fake)
    provider = TesseractFallbackProvider(storage=object())
    light = _page().convert("P")
    dark = light.copy()
    dark.putpalette([255 - v for v in light.getpalette()])
    with ocr_memo():
        provider.ocr_lines(light)
        provider.ocr_lines(dark)
    assert fake.calls == 2


def test_memo_is_dropped_when_parse_fails(monkeypatch):
    class Broken(Strategy):
        level, name, method = 5, "broken", "broken"

        def applicable(self, context):
            return True

        def run(self, context):
            TesseractFallbackProvider(storage=object()).ocr_lines(_page())
            raise RuntimeError("сбой")

    _patch(monkeypatch, _Tesseract())
    import pytest
    from core.providers.document_parser import ParserFailed

    with pytest.raises(ParserFailed):
        LadderRouter([Broken()], threshold=0.75).parse(
            DocumentContext(uri="documents/x.png", file_type="png", metadata={}, data=b"x")
        )
    assert tesseract_fallback._MEMO.get() is None
