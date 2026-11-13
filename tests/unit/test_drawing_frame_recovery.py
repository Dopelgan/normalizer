"""
Восстановление рамки формата на листах, где сплошной линии в 0,7 стороны нет.

Геометрия — с двух настоящих листов (см. `tests/fixtures/real_sheet_lines`),
которые 13.11 вернулись человеку с «рамка нет». Выдуманных координат здесь
нет: порог не подгоняется под примеры, а проверяется на том, что на этих
листах действительно нарисовано.
"""

import pytest

from core.config import settings
from core.providers import drawing_vision as vision

pytest.importorskip("numpy")
pytest.importorskip("PIL")

from tests.fixtures import real_sheet_lines as real  # noqa: E402
from tests.fixtures import synthetic_sheet as sheet  # noqa: E402

# Имя механизма -> его флаг в настройках.
FLAGS = {
    "stitch": "DRAWING_FRAME_STITCH",
    "four_sides": "DRAWING_FRAME_FOUR_SIDES",
    "from_title_block": "DRAWING_FRAME_FROM_TITLE_BLOCK",
}


@pytest.fixture
def flags(monkeypatch):
    """Включатель механизмов: по умолчанию все выключены."""
    def set_flags(**wanted):
        for name, flag in FLAGS.items():
            monkeypatch.setattr(settings, flag, bool(wanted.get(name, False)))
    set_flags()
    return set_flags


@pytest.fixture(params=real.SHEETS)
def data(request):
    return real.sheet(request.param)


# ===========================================================================
# Что именно не проходит на этих листах
# ===========================================================================

class TestReproduction:
    def test_no_single_line_reaches_the_threshold(self, data):
        """
        Причина отказа — длина, а не отсутствие линий. Длиннейшая сплошная
        горизонталь на обоих листах короче `_FRAME_LINE`, при том что
        вертикали порог проходят: рвутся именно горизонтали рамки.
        """
        assert data["longest_raw_h"] < vision._FRAME_LINE
        assert data["longest_raw_v"] >= vision._FRAME_LINE

    def test_plain_detector_finds_no_frame(self, data):
        height, width = data["mask_shape"]
        assert vision._frame(
            real.lines(data, "raw_h"), real.lines(data, "raw_v"), width, height,
        ) is None
        assert data["frame_before"] is None

    def test_lines_on_the_sheet_are_plentiful(self, data):
        """Линий на листе хватает — поэтому таблицы и штамп находятся."""
        assert len(data["raw_h"]) >= 30 and len(data["raw_v"]) >= 60


# ===========================================================================
# Флаги выключены — не меняется ничего
# ===========================================================================

class TestFlagsOff:
    def test_recovery_is_not_attempted(self, flags, data):
        assert vision._recovery_enabled() is False

    def test_synthetic_sheet_layout_is_unchanged(self, flags):
        """Лист, на котором рамка и так находится, флаги не трогают."""
        off = vision.analyse(sheet.build())
        flags(stitch=True, four_sides=True, from_title_block=True)
        on = vision.analyse(sheet.build())
        assert off.frame == on.frame
        assert "frame_recovery" not in off.signals
        # Рамка нашлась обычным способом — восстановление не включалось.
        assert "frame_recovery" not in on.signals

    def test_blank_sheet_stays_without_frame(self, flags):
        from PIL import Image

        flags(stitch=True, four_sides=True, from_title_block=True)
        empty = vision.analyse(Image.new("L", (1200, 900), 255))
        assert empty.frame is None
        assert empty.signals["frame_recovery"]["source"] is None


# ===========================================================================
# Способ 1: сшивка разорванной линии
# ===========================================================================

class TestStitch:
    def test_broken_frame_line_becomes_one_long_line(self, data):
        """
        Та же полоса листа: обычный детектор видит короткие куски, сшивка —
        одну линию длиной больше порога. Длина считается по чернилам, так
        что разрывы в неё не засчитываются.
        """
        mask = real.band_mask(data)
        width = data["mask_shape"][1]
        pieces = vision._lines(mask, axis=0)
        stitched = vision._stitched(mask, axis=0)
        assert max(piece[2] - piece[1] for piece in pieces) < vision._FRAME_LINE * width
        assert max(line[3] for line in stitched) >= vision._FRAME_LINE * width

    def test_stitched_length_never_exceeds_the_ink(self, data):
        """Сшитая линия не может быть длиннее листа и своих же чернил."""
        width = data["mask_shape"][1]
        for position, start, end, ink in real.lines(data, "stitched_h"):
            assert 0 <= ink <= end - start + 1 <= width

    def test_frame_is_recovered_with_a_gost_format(self, data):
        height, width = data["mask_shape"]
        box, detail = vision._frame_by_sides(
            real.lines(data, "stitched_h"), real.lines(data, "stitched_v"),
            real.Mask((height, width)), whole_sheet=True,
        )
        # Положения линий в выгрузке округлены до десятой пикселя, поэтому
        # сравнение с допуском в эту же десятую, а не точное.
        assert box == pytest.approx(data["frame_after"], abs=1e-3)
        assert detail == data["format_mm"]

    def test_recovered_frame_is_the_inner_gost_frame(self, data):
        """
        Найденное — внутренняя рамка: поле подшивки слева шире правого поля
        (20 против 5 мм по ГОСТ 2.301), и рамка лежит внутри листа.
        """
        left, top, right, bottom = data["frame_after"]
        assert left > 1 - right
        assert 0 < left and right < 1 and 0 < top and bottom < 1


# ===========================================================================
# Способ 2: согласие четырёх сторон
# ===========================================================================

class TestFourSides:
    def test_lower_threshold_still_needs_all_four_sides(self, data):
        """Порог ниже, но сторон по-прежнему четыре: трёх не хватает."""
        height, width = data["mask_shape"]
        horizontals = real.lines(data, "stitched_h")
        verticals = real.lines(data, "stitched_v")
        box, _ = vision._frame_by_sides(
            horizontals, verticals, real.Mask((height, width)), whole_sheet=False)
        assert box is not None
        # Без линий у нижнего края рамка не собирается ни при каком пороге.
        without_bottom = [
            line for line in horizontals if line[0] < (1 - vision._EDGE_BAND) * height
        ]
        refused, why = vision._frame_by_sides(
            without_bottom, verticals, real.Mask((height, width)), whole_sheet=False)
        assert refused is None
        assert "низ" in why

    def test_single_long_side_is_never_a_frame(self, data):
        """Одна сторона — не рамка, сколько бы её ни было прочерчено."""
        height, width = data["mask_shape"]
        verticals = real.lines(data, "stitched_v")
        longest = max(real.lines(data, "stitched_h"), key=lambda line: line[3])
        box, why = vision._frame_by_sides(
            [longest], verticals, real.Mask((height, width)), whole_sheet=False)
        assert box is None and "нет линии у края" in why


# ===========================================================================
# Отказы: неверная рамка хуже отсутствующей
# ===========================================================================

class TestRefusals:
    def test_reason_is_recorded_for_every_mechanism(self, flags):
        from PIL import Image

        flags(stitch=True, four_sides=True, from_title_block=True)
        layout = vision.analyse(Image.new("L", (1200, 900), 255))
        refused = layout.signals["frame_recovery"]["refused"]
        assert set(refused) == {"stitch", "four_sides", "title_block"}
        assert all(isinstance(why, str) and why for why in refused.values())

    def test_refusal_reaches_the_payload(self, flags):
        from PIL import Image

        flags(stitch=True, four_sides=True, from_title_block=True)
        layout = vision.analyse(Image.new("L", (1200, 900), 255))
        assert layout.as_dict()["frame_recovery"]["source"] is None

    def test_rectangle_without_a_gost_proportion_is_refused(self, data):
        """
        Квадрат ни одному формату ГОСТ 2.301 не отвечает: вытянутость
        листа — это и есть признак, по которому формат узнаётся.
        """
        height, width = data["mask_shape"]
        square = min(height, width) - 2
        paper, why = vision._format_of(square, square, width, height, 0.02)
        assert paper is None and "ГОСТ 2.301" in why

    def test_frame_without_a_binding_margin_is_refused(self, data):
        """
        Обводка таблицы стоит на листе симметрично, у рамки слева поле
        подшивки 20 мм против 5 мм справа. Без этой разницы — отказ.
        """
        height, width = data["mask_shape"]
        mask = real.Mask((height, width))
        margin = int(0.02 * width)
        box = [margin, int(0.02 * height), width - margin, height - int(0.02 * height)]
        why = vision._rejected(box, (395, 287), mask, [], [])
        assert "поля подшивки" in why

    def test_title_block_without_a_stamp_is_refused(self, data):
        height, width = data["mask_shape"]
        box, why = vision._frame_by_title_block(
            None, real.lines(data, "stitched_h"), real.lines(data, "stitched_v"),
            real.Mask((height, width)),
        )
        assert box is None and why == "штамп не найден"


# ===========================================================================
# Рамка не подменяет уже найденную
# ===========================================================================

class TestNoOverwrite:
    def test_recovery_only_runs_without_a_frame(self, flags, monkeypatch):
        """
        Восстановление не вызывается, когда рамка нашлась обычным способом:
        подменить верную рамка на угаданную оно не может в принципе.
        """
        calls = []
        monkeypatch.setattr(
            vision, "_recover_frame",
            lambda *args, **kwargs: calls.append(args) or None,
        )
        flags(stitch=True, four_sides=True, from_title_block=True)
        layout = vision.analyse(sheet.build())
        assert layout.frame is not None
        assert calls == []
