"""
Подготовка скана к OCR: поворот, выравнивание, снятие линий таблиц.

Без подготовки повёрнутый лист даёт 0 % текста, числа в ячейках таблицы с
сеткой не читаются ни в одном режиме Tesseract, а зерно скана вдвое снижает
долю прочитанных реквизитов.
"""

import os
import random
import re
import shutil

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from core.config import settings
from core.providers.ocr_prepare import (
    PreparedPage,
    _line_mask,
    _rotate,
    _skew_angle,
    _speckle_share,
    prepare_page,
)

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def _black_box(image: Image.Image):
    a = np.asarray(image.convert("L")) < 128
    ys, xs = np.nonzero(a)
    w, h = image.size
    return [xs.min() / w, ys.min() / h, (xs.max() + 1) / w, (ys.max() + 1) / h]


@pytest.mark.parametrize("angles", [[-90], [90], [180], [3.0], [-90, 2.5]])
def test_bbox_maps_back_to_source_after_rotation(angles):
    source = Image.new("L", (800, 1200), 255)
    ImageDraw.Draw(source).rectangle([100, 150, 300, 220], fill=0)
    expected = _black_box(source)

    page = PreparedPage(image=source, source_size=source.size)
    for angle in angles:
        _rotate(page, angle)
    mapped = page.to_source(_black_box(page.image))

    assert mapped == pytest.approx(expected, abs=0.02)


def test_unrotated_page_keeps_bbox():
    page = PreparedPage(image=Image.new("L", (10, 10), 255), source_size=(10, 10))
    assert page.to_source([0.1, 0.2, 0.3, 0.4]) == [0.1, 0.2, 0.3, 0.4]


def test_table_grid_removed_but_letters_kept():
    image = Image.new("L", (1200, 800), 255)
    draw = ImageDraw.Draw(image)
    for x in range(100, 1101, 250):
        draw.line([(x, 100), (x, 700)], fill=0, width=3)
    for y in range(100, 701, 120):
        draw.line([(100, y), (1100, y)], fill=0, width=3)
    letters = Image.new("L", image.size, 255)
    ImageDraw.Draw(letters).rectangle([150, 140, 165, 180], fill=0)  # штрих цифры
    image.paste(letters.crop((150, 140, 166, 181)), (150, 140))

    black = np.asarray(image) < 128
    lines = _line_mask(black)
    cleaned = black & ~lines

    assert lines.sum() > 0
    assert cleaned[140:181, 150:166].mean() > 0.9  # штрих цифры остался
    assert cleaned[100, 400:900].sum() == 0  # горизонтальная линия ушла
    assert cleaned[300:600, 350].sum() == 0  # вертикальная линия ушла


def test_thick_letter_bars_are_not_lines():
    """Верхняя черта крупной «Т» длинная, но толстая — это не линия сетки."""
    image = Image.new("L", (1600, 1000), 255)
    ImageDraw.Draw(image).rectangle([100, 100, 1500, 140], fill=0)  # черта 40 px
    black = np.asarray(image) < 128
    assert _line_mask(black)[100:141, 100:1500].sum() == 0


def test_clean_thin_strokes_survive_preparation(monkeypatch):
    """
    Тонкий штрих мелкого шрифта на чистом листе не стирается: медианный
    фильтр включается только на зернистом скане (иначе плотная
    страница 16 px даёт 0 % текста).
    """
    monkeypatch.setattr("core.providers.ocr_prepare._orientation", lambda _: 0)
    image = Image.new("L", (800, 600), 255)
    draw = ImageDraw.Draw(image)
    for x in range(100, 700, 6):
        draw.line([(x, 200), (x, 215)], fill=0, width=1)  # штрихи толщиной 1 px
    prepared = prepare_page(image)
    assert (np.asarray(prepared.image)[200:216, 100:700] < 128).sum() > 1000


def test_speckle_share_separates_clean_and_grainy_scans():
    clean = Image.new("L", (400, 400), 255)
    ImageDraw.Draw(clean).rectangle([50, 50, 350, 80], fill=0)
    grainy = clean.copy()
    pixels = grainy.load()
    rnd = random.Random(3)
    for _ in range(4000):
        pixels[rnd.randrange(400), rnd.randrange(100, 400)] = 0
    assert _speckle_share(clean) < 0.01
    assert _speckle_share(grainy) > 0.05


def test_blank_page_is_not_rotated():
    assert _skew_angle(Image.new("L", (1654, 2339), 255), 6.0) == 0.0


def test_skew_stays_within_limit():
    image = Image.new("L", (1200, 1600), 255)
    draw = ImageDraw.Draw(image)
    for y in range(200, 1400, 60):
        draw.rectangle([150, y, 1050, y + 14], fill=0)
    angle = _skew_angle(image.rotate(-5.8, expand=True, fillcolor=255), 6.0)
    assert -6.0 <= angle <= 6.0


def test_prepare_failure_falls_back_to_original(monkeypatch):
    from core.providers.tesseract_fallback import TesseractFallbackProvider

    def boom(_):
        raise RuntimeError("сбой подготовки")

    monkeypatch.setattr(
        "core.providers.tesseract_fallback.prepare_page", boom
    )
    assert TesseractFallbackProvider._prepare(Image.new("L", (50, 50), 255)) is None


def test_preprocess_can_be_disabled(monkeypatch):
    from core.providers.tesseract_fallback import TesseractFallbackProvider

    monkeypatch.setattr(settings, "OCR_PREPROCESS_ENABLED", False)
    assert TesseractFallbackProvider._prepare(Image.new("L", (50, 50), 255)) is None


# --------------------------------------------------- сквозной OCR (Tesseract)

def _has_russian_tesseract() -> bool:
    if not shutil.which("tesseract") or not os.path.exists(FONT):
        return False
    try:
        import pytesseract

        return "rus" in pytesseract.get_languages(config="")
    except Exception:  # noqa: BLE001
        return False


requires_ocr = pytest.mark.skipif(
    not _has_russian_tesseract(), reason="нужен tesseract с rus и шрифт DejaVu"
)

FACTS = ["ПИ-2026/0457", "НТЛ-Прибор", "15.03.2026", "КТ-118", "12,49", "0,032", "1450", "Годен"]


def _protocol() -> Image.Image:
    """Лист протокола испытаний: шапка, реквизиты и таблица с сеткой (A4, 200 dpi)."""
    font = lambda size: ImageFont.truetype(FONT, size)  # noqa: E731
    image = Image.new("L", (1654, 2339), 255)
    draw = ImageDraw.Draw(image)
    draw.text((200, 130), "ПРОТОКОЛ ИСПЫТАНИЙ № ПИ-2026/0457", font=font(42), fill=0)
    draw.text((200, 215), "Организация: ООО «НТЛ-Прибор»  Дата: 15.03.2026", font=font(30), fill=0)
    draw.text((200, 300), "Стенд: КТ-118      Заключение: Годен", font=font(30), fill=0)
    rows = [["Параметр", "Норма", "Факт"], ["Давление, кПа", "12,5", "12,49"],
            ["Ток утечки, мА", "0,03", "0,032"], ["Ресурс, ч", "1500", "1450"]]
    widths, top, height = [480, 280, 280], 450, 64
    for r, row in enumerate(rows):
        x = 200
        for c, cell in enumerate(row):
            draw.rectangle([x, top + r * height, x + widths[c], top + (r + 1) * height], outline=0, width=2)
            draw.text((x + 14, top + r * height + 14), cell, font=font(26), fill=0)
            x += widths[c]
    return image


def _scan(image: Image.Image, rotate: int = 0, skew: float = 0.0) -> Image.Image:
    if skew:
        image = image.rotate(skew, expand=True, fillcolor=255, resample=Image.BICUBIC)
    if rotate:
        image = image.rotate(rotate, expand=True, fillcolor=255)
    image = image.filter(ImageFilter.GaussianBlur(0.7))
    pixels = image.load()
    rnd = random.Random(7)
    for _ in range(image.width * image.height // 80):
        pixels[rnd.randrange(image.width), rnd.randrange(image.height)] = rnd.choice((0, 200))
    return image.convert("RGB")


def _recall(lines) -> float:
    text = re.sub(r"\s+", " ", " ".join(line.text for line in lines)).lower()
    return sum(fact.lower() in text for fact in FACTS) / len(FACTS)


@requires_ocr
@pytest.mark.parametrize("rotate,skew", [(0, 0.0), (90, 0.0), (180, 0.0), (0, 3.0)])
def test_scanned_protocol_keeps_requisites_and_table_numbers(rotate, skew):
    from core.providers.tesseract_fallback import TesseractFallbackProvider

    provider = TesseractFallbackProvider(storage=object())
    lines = provider.ocr_lines(_scan(_protocol(), rotate=rotate, skew=skew))

    assert _recall(lines) >= 0.75
    text = " ".join(line.text for line in lines)
    assert "0,032" in text or "1450" in text  # числа из ячеек таблицы


@requires_ocr
def test_rotated_scan_was_unreadable_without_preparation(monkeypatch):
    """Причинный контроль: без подготовки повёрнутый лист не читается."""
    from core.providers.tesseract_fallback import TesseractFallbackProvider

    monkeypatch.setattr(settings, "OCR_PREPROCESS_ENABLED", False)
    provider = TesseractFallbackProvider(storage=object())
    assert _recall(provider.ocr_lines(_scan(_protocol(), rotate=90))) < 0.3


@requires_ocr
def test_small_upscaled_image_still_readable():
    """
    Мелкий растр `_fit` увеличивает до рабочего размера; штрихи букв на нём
    толстые и длинные, их нельзя снимать как линии таблицы.
    """
    from core.providers.tesseract_fallback import TesseractFallbackProvider

    small = Image.new("RGB", (160, 50), "white")
    ImageDraw.Draw(small).text((6, 12), "ТЕСТ 1450", font=ImageFont.truetype(FONT, 20), fill="black")
    text = " ".join(line.text for line in TesseractFallbackProvider(storage=object()).ocr_lines(small))
    assert "1450" in text


@requires_ocr
def test_rotated_scan_lines_are_in_source_coordinates():
    """Строки повёрнутого листа возвращаются в координатах исходного файла."""
    from core.providers.tesseract_fallback import TesseractFallbackProvider

    scan = _scan(_protocol(), rotate=90)  # заголовок оказался у левого края
    lines = TesseractFallbackProvider(storage=object()).ocr_lines(scan)
    title = next(line for line in lines if "ПРОТОКОЛ" in line.text.upper())
    assert title.bbox[2] < 0.35  # в исходном (повёрнутом) файле — слева
    assert title.bbox[3] - title.bbox[1] > title.bbox[2] - title.bbox[0]  # вытянут по вертикали


class TestOsdRotation:
    """Ответ OSD разбирается в (угол, уверенность); нет ответа — `None`, а не 0."""

    def test_answer_is_parsed(self, monkeypatch):
        import pytesseract

        from core.providers.ocr_prepare import osd_rotation

        monkeypatch.setattr(
            pytesseract, "image_to_osd",
            lambda *a, **k: "Page number: 0\nRotate: 270\nOrientation confidence: 7.31\n",
        )
        assert osd_rotation(Image.new("L", (10, 10), 255)) == (270, 7.31)

    def test_failure_is_unknown_not_zero(self, monkeypatch):
        import pytesseract

        from core.providers.ocr_prepare import _orientation, osd_rotation

        def broken(*a, **k):
            raise RuntimeError("tesseract не установлен")

        monkeypatch.setattr(pytesseract, "image_to_osd", broken)
        assert osd_rotation(Image.new("L", (10, 10), 255)) is None
        assert _orientation(Image.new("L", (10, 10), 255)) == 0


class TestLineRemovalGuards:
    """
    Предохранители снятия линий: отказ, защита штрихов, устойчивый порог.

    Снятие линий задумано под скан таблицы с сеткой, а на чертёжном листе
    отнимает больше, чем даёт (замер — `ocr-prepare-eval-20261114`). Все
    три настройки по умолчанию выключены, и при выключенных подготовка
    работает ровно как раньше — это здесь и проверяется.
    """

    @staticmethod
    def _sheet_with_grid_and_text():
        """Лист с длинной тонкой линией и буквой, лежащей на этой линии."""
        image = Image.new("L", (900, 600), 255)
        draw = ImageDraw.Draw(image)
        # Линия рамки: длинная (больше 6 % ширины) и тонкая (2 px).
        draw.rectangle([40, 300, 860, 301], fill=0)
        # Вертикальный штрих, проходящий через линию насквозь.
        draw.rectangle([400, 240, 403, 360], fill=0)
        # Немного текста, чтобы лист не был пустым.
        draw.text((60, 80), "ПРОТОКОЛ 12-345", fill=0)
        return image

    def _prepared(self, monkeypatch, **overrides):
        monkeypatch.setattr("core.providers.ocr_prepare._orientation", lambda _: 0)
        monkeypatch.setattr("core.providers.ocr_prepare._skew_angle", lambda *a: 0.0)
        for name, value in overrides.items():
            monkeypatch.setattr(settings, name, value)
        return prepare_page(self._sheet_with_grid_and_text())

    def test_flags_off_prepare_unchanged(self, monkeypatch):
        """С выключенными настройками — тот же лист и те же снятые пиксели."""
        before = self._prepared(monkeypatch)
        after = self._prepared(
            monkeypatch,
            OCR_LINE_REMOVAL_SKIP=False,
            OCR_LINE_STROKE_GUARD=False,
            OCR_OTSU_THRESHOLD_STEP=0,
        )
        assert after.removed_line_pixels == before.removed_line_pixels
        assert np.array_equal(np.asarray(after.image), np.asarray(before.image))
        assert before.removed_line_pixels > 0  # линия действительно снималась

    def test_skip_setting_removes_nothing(self, monkeypatch):
        page = self._prepared(monkeypatch, OCR_LINE_REMOVAL_SKIP=True)
        assert page.removed_line_pixels == 0

    def test_keep_lines_argument_removes_nothing(self, monkeypatch):
        monkeypatch.setattr("core.providers.ocr_prepare._orientation", lambda _: 0)
        monkeypatch.setattr("core.providers.ocr_prepare._skew_angle", lambda *a: 0.0)
        page = prepare_page(self._sheet_with_grid_and_text(), keep_lines=True)
        assert page.removed_line_pixels == 0

    def test_guard_keeps_the_stroke_that_crosses_the_line(self, monkeypatch):
        """
        Штрих, лежащий на линии, вместе с линией не пропадает.

        Без защиты пиксели линии под штрихом закрашиваются белым, и штрих
        разрывается. С защитой столбец штриха остаётся.
        """
        plain = self._prepared(monkeypatch)
        guarded = self._prepared(
            monkeypatch, OCR_LINE_STROKE_GUARD=True, OCR_LINE_STROKE_GUARD_REACH=4
        )
        assert guarded.removed_line_pixels < plain.removed_line_pixels
        column = 401
        plain_px = np.asarray(plain.image)
        guarded_px = np.asarray(guarded.image)
        # Разрыв штриха на линии: без защиты белое, с защитой чернила.
        assert plain_px[300, column] == 255
        assert guarded_px[300, column] < 128

    def test_threshold_step_makes_otsu_stable(self):
        """
        Порог, округлённый до кратного, не прыгает от 0,2 % изменения размера.

        На настоящих листах сдвиг длинной стороны 3507 → 3500 px двигал
        порог ровно на уровень (151 → 152), а с ним — маску линий.
        """
        from core.providers.ocr_line_guard import quantise

        assert quantise(151, 0) == 151  # выключено — как было
        assert quantise(151, 8) == quantise(152, 8)
        assert quantise(149, 8) == quantise(150, 8)
        assert 0 <= quantise(254, 8) <= 255

    def test_guard_leaves_a_lonely_line_alone(self):
        """У одинокой линии сетки защищать нечего — она снимается целиком."""
        from core.providers.ocr_line_guard import stroke_guard

        black = np.zeros((40, 400), dtype=bool)
        black[20, 10:390] = True
        lines = black.copy()
        assert stroke_guard(black, lines, 4).sum() == lines.sum()
