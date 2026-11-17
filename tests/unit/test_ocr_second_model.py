"""
Вторая русская модель (`OCR_RUS_SECOND_MODEL`) и окончание порядкового
числительного латиницей (`OCR_ORDINAL_SUFFIX_FIX`).

Дообученная под шрифт ГОСТ модель на машинописи свидетельств и писем
путает буквы: «С-ВЫТ» вместо «С-ВЫГ», «бесперепективно», «3-ro разряда».
Системная модель те же слова читает верно. Проверяется строка языков,
которая реально доходит до Tesseract, и что без файла модели и без флага
поведение прежнее. Тексты фикстур — короткие строки того же вида, без
документов заказчика.
"""

import logging

import pytest
from PIL import Image, ImageDraw

from core.config import settings
from core.providers import ocr_cleanup, ocr_model, tesseract_fallback
from core.providers.ocr_cleanup import Vocabulary, clean_text
from core.providers.tesseract_fallback import TesseractFallbackProvider


def _tessdata(tmp_path, extra=()):
    for name in ocr_model.REQUIRED + tuple(f"{n}.traineddata" for n in extra):
        (tmp_path / name).write_bytes(b"traineddata")
    return str(tmp_path)


@pytest.fixture(autouse=True)
def _fresh_warnings(monkeypatch):
    monkeypatch.setattr(ocr_model, "_warned", set())


class TestLanguages:
    def test_off_by_default(self, monkeypatch, tmp_path):
        monkeypatch.setattr(settings, "OCR_TESSDATA_DIR", _tessdata(tmp_path, ["rusfast"]))
        assert settings.OCR_RUS_SECOND_MODEL == ""
        assert ocr_model.languages("rus+eng") == "rus+eng"

    def test_second_model_goes_last(self, monkeypatch, tmp_path):
        # После `eng`: иначе латиница (адреса, «RA.RU») читается русской
        # моделью кириллицей — замерено на свидетельстве.
        monkeypatch.setattr(settings, "OCR_TESSDATA_DIR", _tessdata(tmp_path, ["rusfast"]))
        monkeypatch.setattr(settings, "OCR_RUS_SECOND_MODEL", "rusfast")
        assert ocr_model.languages("rus+eng") == "rus+eng+rusfast"
        assert ocr_model.languages("rus") == "rus+rusfast"

    def test_only_with_russian(self, monkeypatch, tmp_path):
        # Построчный выбор языка читает «eng» отдельно: русская модель там чужая.
        monkeypatch.setattr(settings, "OCR_TESSDATA_DIR", _tessdata(tmp_path, ["rusfast"]))
        monkeypatch.setattr(settings, "OCR_RUS_SECOND_MODEL", "rusfast")
        assert ocr_model.languages("eng") == "eng"
        assert ocr_model.languages("rus+eng+rusfast") == "rus+eng+rusfast"

    def test_missing_file_keeps_languages_and_warns_once(self, monkeypatch, tmp_path, caplog):
        # Неизвестный язык роняет весь вызов Tesseract: без файла его не просим.
        monkeypatch.setattr(settings, "OCR_TESSDATA_DIR", _tessdata(tmp_path))
        monkeypatch.setattr(settings, "OCR_RUS_SECOND_MODEL", "rusfast")
        with caplog.at_level(logging.WARNING, logger=ocr_model.__name__):
            assert ocr_model.languages("rus+eng") == "rus+eng"
            assert ocr_model.languages("rus+eng") == "rus+eng"
        assert sum("OCR_RUS_SECOND_MODEL" in r.getMessage() for r in caplog.records) == 1

    def test_system_catalog_is_not_guessed(self, monkeypatch):
        monkeypatch.setattr(settings, "OCR_TESSDATA_DIR", "")
        monkeypatch.setattr(settings, "OCR_RUS_SECOND_MODEL", "rusfast")
        assert ocr_model.languages("rus+eng") == "rus+eng"


def test_page_read_with_second_model(monkeypatch, tmp_path):
    """До Tesseract доходит строка языков со второй моделью — не только настройка."""
    seen = []

    def fake(image, lang=None, config=None, output_type=None):
        seen.append(lang)
        return {
            "text": ["С-ВЫГ/27-09-2022/1"], "conf": [90], "block_num": [1],
            "par_num": [1], "line_num": [1], "left": [10], "top": [10],
            "width": [50], "height": [20],
        }

    monkeypatch.setattr(settings, "OCR_PREPROCESS_ENABLED", False)
    monkeypatch.setattr(settings, "OCR_TESSDATA_DIR", _tessdata(tmp_path, ["rusfast"]))
    monkeypatch.setattr(tesseract_fallback.pytesseract, "image_to_data", fake)
    image = Image.new("RGB", (1000, 700), "white")
    ImageDraw.Draw(image).rectangle((10, 10, 210, 40), fill="black")

    TesseractFallbackProvider(storage=object()).ocr_lines(image)
    monkeypatch.setattr(settings, "OCR_RUS_SECOND_MODEL", "rusfast")
    image2 = image.copy()
    ImageDraw.Draw(image2).rectangle((10, 100, 210, 130), fill="black")
    TesseractFallbackProvider(storage=object()).ocr_lines(image2)

    assert seen == ["rus+eng", "rus+eng+rusfast"]


VOCAB = Vocabulary(rus=frozenset("эталон разряда гост издание".split()), eng=frozenset())


class TestOrdinalSuffix:
    @pytest.mark.parametrize("raw, expected", [
        ("Эталон 3-ro разряда ГОСТ", "Эталон 3-го разряда ГОСТ"),
        ("в 72-x годах", "в 72-х годах"),
        ("3-e издание", "3-е издание"),
    ])
    def test_latin_ending_becomes_russian(self, monkeypatch, raw, expected):
        monkeypatch.setattr(settings, "OCR_ORDINAL_SUFFIX_FIX", True)
        assert clean_text(raw, VOCAB)[0] == expected

    @pytest.mark.parametrize("raw", [
        "Исполнение 03-a эталон",     # «а» — не окончание, литера исполнения
        "код 0005-t эталон",          # «т» — не окончание
        "Исполнение 0001-E эталон",   # заглавная — литера обозначения
        "Эталон 3-го разряда",        # уже русское
        "ISO 9001-e drawing",         # латинская строка
    ])
    def test_other_tokens_untouched(self, monkeypatch, raw):
        monkeypatch.setattr(settings, "OCR_ORDINAL_SUFFIX_FIX", True)
        assert clean_text(raw, VOCAB)[0] == raw

    def test_off_by_default(self):
        assert settings.OCR_ORDINAL_SUFFIX_FIX is False
        assert clean_text("Эталон 3-ro разряда", VOCAB)[0] == "Эталон 3-ro разряда"

    def test_counted_as_latin_word(self, monkeypatch):
        monkeypatch.setattr(settings, "OCR_ORDINAL_SUFFIX_FIX", True)
        assert clean_text("Эталон 3-ro разряда", VOCAB)[1].latin_words == 1
        assert ocr_cleanup._fix_ordinal("12-my") == "12-му"
