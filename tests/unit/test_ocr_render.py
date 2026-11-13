"""
Выбор разрешения растрирования страницы под OCR.

Проверяется то, ради чего модуль написан: страница рисуется в разрешении
своего собственного растра, но не выше того, что сохранит `OCR_MAX_SIDE`, —
иначе нарисованное тут же уменьшается обратно, и вся цена уходит впустую.
И отдельно — что с выключенным флагом не меняется ничего.
"""

import pytest

from core.config import settings
from core.providers import ocr_render


@pytest.fixture(autouse=True)
def _clear_probe_cache():
    """Разрешение и размер страницы кешируются по пути — между тестами сбрасываем."""
    ocr_render.source_dpi.cache_clear()
    ocr_render.page_inches.cache_clear()
    yield
    ocr_render.source_dpi.cache_clear()
    ocr_render.page_inches.cache_clear()


def _page(monkeypatch, native_ppi, inches, *, from_source=True,
          max_side=3500, ocr_dpi=200, ceiling=600):
    monkeypatch.setattr(settings, "OCR_DPI_FROM_SOURCE", from_source)
    monkeypatch.setattr(settings, "OCR_SMALL_PRINT_RESCAN", from_source)
    monkeypatch.setattr(settings, "OCR_DPI", ocr_dpi)
    monkeypatch.setattr(settings, "OCR_MAX_SIDE", max_side)
    monkeypatch.setattr(settings, "OCR_DPI_CEILING", ceiling)
    monkeypatch.setattr(
        ocr_render, "source_dpi", lambda path, page: native_ppi,
    )
    monkeypatch.setattr(
        ocr_render, "page_inches", lambda path, page: inches,
    )


# ---------------------------------------------------------------- флаг выключен

def test_flag_off_keeps_configured_dpi(monkeypatch):
    """Выключенный флаг — прежнее поведение, одно число на все страницы."""
    _page(monkeypatch, native_ppi=600, inches=16.53, from_source=False)
    assert ocr_render.render_dpi("/tmp/a.pdf", 1) == 200


def test_flag_off_never_asks_the_file(monkeypatch):
    """С выключенным флагом файл не осматривается вовсе: ни одного запуска poppler."""
    monkeypatch.setattr(settings, "OCR_DPI_FROM_SOURCE", False)
    monkeypatch.setattr(settings, "OCR_DPI", 200)

    def refuse(*_args, **_kwargs):
        raise AssertionError("с выключенным флагом файл осматривать нельзя")

    monkeypatch.setattr(ocr_render, "_probe", refuse)
    assert ocr_render.render_dpi("/tmp/a.pdf", 1) == 200


def test_rescan_is_off_without_its_flag(monkeypatch):
    """Второй проход тоже за флагом: без него страницу никто не перечитывает."""
    _page(monkeypatch, native_ppi=600, inches=16.53, from_source=False)
    assert ocr_render.rescan("/tmp/a.pdf", 1, 200, median_word_height=8) is None


def test_the_two_flags_are_independent(monkeypatch):
    """
    Флаги не связаны: второй проход можно включить, не трогая безусловный
    выбор разрешения, — именно так его и предлагается включать.
    """
    _page(monkeypatch, native_ppi=601, inches=16.53, from_source=False)
    monkeypatch.setattr(settings, "OCR_SMALL_PRINT_RESCAN", True)
    assert ocr_render.render_dpi("/tmp/spec.pdf", 1) == 200
    assert ocr_render.rescan("/tmp/spec.pdf", 1, 200, median_word_height=8) is not None


# ------------------------------------------------------------- выбор разрешения

def test_a4_scan_at_300_is_rendered_at_300(monkeypatch):
    """
    Страница A4 со своим растром 300 ppi: потолок 3500 сохраняет 299 dpi,
    поэтому рисуется 299 — почти всё своё разрешение и ничего сверх.
    """
    _page(monkeypatch, native_ppi=300, inches=11.69)
    assert ocr_render.render_dpi("/tmp/a4.pdf", 1) == 299


def test_a3_scan_is_capped_at_what_the_cap_keeps(monkeypatch):
    """
    A3 со своим растром 300 ppi: потолок 3500 сохраняет только 211 dpi.
    Рисовать 300 здесь — чистый расход: `_fit` уменьшит лист обратно.
    """
    _page(monkeypatch, native_ppi=300, inches=16.53)
    assert ocr_render.render_dpi("/tmp/a3.pdf", 1) == 211


def test_raising_the_cap_lets_the_native_resolution_through(monkeypatch):
    """Снятый потолок — рисуем всё своё разрешение, до верхней границы."""
    _page(monkeypatch, native_ppi=601, inches=16.53, max_side=0)
    assert ocr_render.render_dpi("/tmp/a3.pdf", 1) == 600


def test_never_above_the_source_resolution(monkeypatch):
    """Выше собственного растра не рисуем: там не детали, а интерполяция."""
    _page(monkeypatch, native_ppi=150, inches=11.69, max_side=0)
    assert ocr_render.render_dpi("/tmp/tag.pdf", 1) == 200


def test_configured_dpi_is_the_floor(monkeypatch):
    """
    Растр мельче настроенного разрешения не опускает его: страница со
    150-ppi растром не должна читаться хуже, чем сегодня.
    """
    _page(monkeypatch, native_ppi=150, inches=16.53)
    assert ocr_render.render_dpi("/tmp/tag.pdf", 1) == 200


def test_unknown_source_falls_back_to_configured(monkeypatch):
    """Векторная страница или нечитаемый файл — решает `OCR_DPI`."""
    _page(monkeypatch, native_ppi=None, inches=11.69)
    assert ocr_render.render_dpi("/tmp/vector.pdf", 1) == 200


def test_unknown_page_size_keeps_the_source_resolution(monkeypatch):
    """Размер страницы не прочитался — потолок не переводится, остаётся растр."""
    _page(monkeypatch, native_ppi=400, inches=None)
    assert ocr_render.render_dpi("/tmp/a4.pdf", 1) == 400


def test_ceiling_limits_a_very_high_source(monkeypatch):
    """Растр 1200 ppi не поднимает разрешение выше верхней границы."""
    _page(monkeypatch, native_ppi=1200, inches=11.69, max_side=0, ceiling=600)
    assert ocr_render.render_dpi("/tmp/a4.pdf", 1) == 600


# --------------------------------------------------------------- пачки рендера

def test_batch_runs_keeps_one_batch_when_resolution_is_the_same(monkeypatch):
    """Одинаковые страницы — пачки прежней длины, без лишних разрывов."""
    monkeypatch.setattr(ocr_render, "render_dpi", lambda path, page: 200)
    assert list(ocr_render.batch_runs("/tmp/a.pdf", 1, 7, 5)) == [(1, 5, 200), (6, 7, 200)]


def test_batch_runs_splits_where_resolution_changes(monkeypatch):
    """
    Пачка рвётся на смене разрешения: иначе всю пачку пришлось бы рисовать
    по самой требовательной странице.
    """
    monkeypatch.setattr(
        ocr_render, "render_dpi", lambda path, page: 600 if page in (3, 4) else 200,
    )
    assert list(ocr_render.batch_runs("/tmp/a.pdf", 1, 6, 5)) == [
        (1, 2, 200), (3, 4, 600), (5, 6, 200),
    ]


def test_batch_runs_handles_a_single_page(monkeypatch):
    monkeypatch.setattr(ocr_render, "render_dpi", lambda path, page: 300)
    assert list(ocr_render.batch_runs("/tmp/a.pdf", 4, 4, 5)) == [(4, 4, 300)]


# ------------------------------------------------------------- мелкая печать

@pytest.mark.parametrize("height,expected", [
    (0, False), (8, True), (11, True), (14, True), (15, False), (23, False), (36, False),
])
def test_small_print_threshold(monkeypatch, height, expected):
    """Ниже порога — мелкая печать; 0 (ничего не прочиталось) ею не считается."""
    monkeypatch.setattr(settings, "OCR_SMALL_PRINT_HEIGHT", 15)
    assert ocr_render.small_print(height) is expected


def test_rescan_raises_both_dpi_and_cap(monkeypatch):
    """
    Второй проход поднимает и разрешение, и потолок: одно разрешение без
    потолка бесполезно — `_fit` уменьшит лист обратно.
    """
    _page(monkeypatch, native_ppi=601, inches=16.53)
    got = ocr_render.rescan("/tmp/spec.pdf", 1, current_dpi=200, median_word_height=8)
    assert got is not None
    dpi, max_side = got
    assert dpi == 600
    # Потолок ровно под выбранное разрешение: 600 dpi * 16.53 дюйма.
    assert max_side == pytest.approx(9918, abs=2)


def test_no_rescan_when_lines_are_big_enough(monkeypatch):
    """Строки нормальной высоты второй проход не заказывают."""
    _page(monkeypatch, native_ppi=601, inches=16.53)
    assert ocr_render.rescan("/tmp/spec.pdf", 1, 200, median_word_height=23) is None


def test_no_rescan_when_the_source_has_nothing_more(monkeypatch):
    """
    Мелкая печать на растре 200 ppi: перечитывать нечего, деталей в файле
    больше нет. Второй проход был бы потрачен на интерполяцию.
    """
    _page(monkeypatch, native_ppi=200, inches=11.69)
    assert ocr_render.rescan("/tmp/fax.pdf", 1, 200, median_word_height=9) is None


# ---------------------------------------------------------- чтение из файла

def test_source_dpi_reads_the_largest_image(monkeypatch):
    """
    Разрешение страницы задаёт самый крупный растр, а не печать или логотип:
    по мелкой картинке оценка уехала бы в любую сторону.
    """
    listing = (
        "page   num  type   width height color comp bpc  enc interp  object ID "
        "x-ppi y-ppi size ratio\n"
        "----------------------------------------------------------------\n"
        "   1     0 image     120   120  rgb     3   8  jpeg   no         5  0"
        "  1200  1200  12K 1.0%\n"
        "   1     1 image    4960  3507  rgb     3   8  jpeg   no         1  0"
        "   300   300 1414K 2.8%\n"
    )
    monkeypatch.setattr(ocr_render, "_probe", lambda command: listing)
    assert ocr_render.source_dpi("/tmp/a3.pdf", 1) == 300


def test_source_dpi_is_none_without_images(monkeypatch):
    """Векторная страница: картинок нет, разрешения нет."""
    monkeypatch.setattr(ocr_render, "_probe", lambda command: "page   num  type\n----\n")
    assert ocr_render.source_dpi("/tmp/vector.pdf", 1) is None


def test_page_inches_takes_the_long_side(monkeypatch):
    """Длинная сторона A3 в пунктах — это 16.53 дюйма."""
    monkeypatch.setattr(
        ocr_render, "_probe",
        lambda command: "Pages:           1\nPage    1 size: 1190.4 x 841.68 pts (A3)\n",
    )
    assert ocr_render.page_inches("/tmp/a3.pdf", 1) == pytest.approx(16.53, abs=0.01)


def test_probe_failure_is_not_fatal(monkeypatch):
    """Poppler не отработал — страница читается в настроенном разрешении."""
    monkeypatch.setattr(settings, "OCR_DPI_FROM_SOURCE", True)
    monkeypatch.setattr(settings, "OCR_DPI", 200)
    monkeypatch.setattr(ocr_render, "_probe", lambda command: "")
    assert ocr_render.render_dpi("/tmp/broken.pdf", 1) == 200
