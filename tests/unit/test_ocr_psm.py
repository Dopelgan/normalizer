"""
Режим сегментации страницы Tesseract: настройка есть, поведение не изменено.

Замер (`ocr-psm-eval-20261114`) показал, что чертёжному листу режим
`--psm 6` («один однородный блок») не подходит: строка идёт через весь лист
и склеивает надписи из разных его мест. Настройка `OCR_PSM` вводится, чтобы
режим можно было сменить без правки кода, но значение по умолчанию равно
тому, что зашито в `PSM_BLOCK` сегодня, — при незаданной настройке лист
читается ровно как раньше, байт в байт.

Чтение настройки живёт в `core/providers/tesseract_fallback.py`, который в
этой правке не трогается (он принадлежит другой ветке). Тест держит
договор между двумя правками: значение по умолчанию и совпадение с
`PSM_BLOCK`.
"""

from PIL import Image, ImageDraw

from core.config import settings
from core.providers import tesseract_fallback
from core.providers.tesseract_fallback import PSM_BLOCK, TesseractFallbackProvider


def test_psm_default_keeps_today_behaviour():
    """По умолчанию режим тот же, что зашит в коде: поведение не меняется."""
    assert settings.OCR_PSM == PSM_BLOCK == 6


def test_psm_is_an_integer_setting():
    """Режим — номер, а не строка: он подставляется в `--psm {n}`."""
    assert isinstance(settings.OCR_PSM, int)


def test_page_is_read_in_psm_6_while_setting_is_default(monkeypatch):
    """
    Главное: при незаданной настройке страница уходит в Tesseract ровно с
    тем же `--psm 6`, что и до появления настройки. Тест ловит не значение
    настройки, а строку, которая реально доходит до Tesseract.
    """
    seen = []

    def fake(image, lang=None, config=None, output_type=None):
        seen.append(config)
        return {
            "text": ["Лист"], "conf": [90], "block_num": [1],
            "par_num": [1], "line_num": [1], "left": [10], "top": [10],
            "width": [50], "height": [20],
        }

    monkeypatch.setattr(settings, "OCR_PREPROCESS_ENABLED", False)
    monkeypatch.setattr(tesseract_fallback.pytesseract, "image_to_data", fake)

    image = Image.new("RGB", (1000, 700), "white")
    ImageDraw.Draw(image).rectangle((10, 10, 210, 40), fill="black")
    TesseractFallbackProvider(storage=object()).ocr_lines(image)

    assert seen == [f"--psm {settings.OCR_PSM}"] == ["--psm 6"]
