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
# Поля скана вокруг листа
# ===========================================================================

class TestScanMargin:
    """
    Картинка — не лист: сканер добавляет вокруг листа поле неизвестной
    ширины, и доля листа, занятая рамкой, падает ниже номинальной. Пока это
    считалось отказом, верная рамка A3 отвергалась начиная с 8 мм поля, а
    прямоугольник с краем бумаги вместо правой стороны проходил и получал
    ярлык «A4 альбомный».
    """

    @pytest.mark.parametrize("margin_mm", [0, 5, 8, 12, 20, 40])
    def test_true_frame_survives_scan_margin(self, margin_mm):
        for paper in vision._FRAMES_MM:
            sheet = vision._sheet_of(paper)
            share_w = paper[0] / (sheet[0] + 2 * margin_mm)
            share_h = paper[1] / (sheet[1] + 2 * margin_mm)
            assert vision._fills(share_w, share_h, paper), (paper, margin_mm)

    def test_a_whole_format_of_white_around_is_refused(self):
        """Лишнего белого больше, чем на формат вокруг, — это не скан листа."""
        paper = (395, 287)                      # A3
        sheet = vision._sheet_of(paper)
        step = vision._SHEET_STEP
        share_w = paper[0] / (sheet[0] * step * 1.05)
        share_h = paper[1] / (sheet[1] * step * 1.05)
        assert not vision._fills(share_w, share_h, paper)

    def test_frame_cannot_exceed_the_picture(self):
        assert not vision._fills(1.02, 0.95, (395, 287))

    def test_degenerate_aspects_are_told_apart_by_size(self):
        """
        A0, A4 альбомный и A1 по пропорции почти не отличаются (1,4169,
        1,4205, 1,4216), поэтому формат выбирается по занятой доле листа.
        """
        aspects = {mm: mm[0] / mm[1] for mm in ((1159, 818), (277, 195), (816, 574))}
        assert max(aspects.values()) / min(aspects.values()) < 1.01
        for paper in aspects:
            # Картинка — ровно лист этого формата, рамка на своём месте.
            sheet = vision._sheet_of(paper)
            scale = 1000 / sheet[0]
            width, height = int(sheet[0] * scale), int(sheet[1] * scale)
            box_w, box_h = paper[0] * scale, paper[1] * scale
            chosen, why = vision._format_of(box_w, box_h, width, height, 0.02)
            assert chosen == paper, (paper, chosen, why)

    @staticmethod
    def _sheet_with_margin(paper, margin_mm):
        """Рамка `paper` посреди картинки с полем скана `margin_mm` вокруг листа."""
        sheet = vision._sheet_of(paper)
        full_w, full_h = sheet[0] + 2 * margin_mm, sheet[1] + 2 * margin_mm
        scale = 1000.0 / full_w
        width, height = int(full_w * scale), int(full_h * scale)
        box_w, box_h = paper[0] * scale, paper[1] * scale
        left, top = (width - box_w) / 2 / width, (height - box_h) / 2 / height
        return [left, top, 1 - left, 1 - top], (height, width), box_w, box_h

    @pytest.mark.parametrize("paper", vision._FRAMES_MM)
    @pytest.mark.parametrize("margin_mm", [0, 2, 3, 4, 8, 15, 30])
    def test_a_wrong_label_is_never_called_certain(self, paper, margin_mm):
        """
        Главное про ярлык формата: ошибиться он может, соврать уверенно — нет.

        Поле ГОСТ задано в миллиметрах, поэтому у крупного листа рамка
        занимает большую долю своего листа (A4 альбомный 0,933, A3 0,940,
        A2 0,958, A1 0,970, A0 0,975). Поле скана наблюдаемую долю только
        занижает, и ярлык съезжает вниз по ряду: A2 на A3 с 3 мм и на A4
        альбомный с 8 мм, A1 с 4 мм, A0 с 2 мм. Рамка при этом остаётся
        верной — проверяется, что такой съехавший ярлык не объявляется
        определённым.
        """
        frame, shape, box_w, box_h = self._sheet_with_margin(paper, margin_mm)
        height, width = shape
        chosen, why = vision._format_of(
            box_w, box_h, width, height,
            vision._format_tolerance(box_w, box_h, shape))
        assert chosen is not None, why
        options = vision._format_options(frame, shape)
        # Истинный формат объяснить картинку обязан всегда.
        assert tuple(paper) in options, (paper, margin_mm, options)
        if chosen != tuple(paper):
            assert len(options) > 1, (paper, margin_mm, chosen)
            assert vision._format_certain(frame, shape) is False

    @pytest.mark.parametrize("margin_mm", [0, 3, 8, 15])
    def test_degenerate_cluster_is_never_certain(self, margin_mm):
        """
        A0, A4 альбомный и A1 отличаются по пропорции на 0,3 %, а поле скана
        неизвестно: A1 с полем 17 мм даёт ровно то же, что A4 альбомный без
        поля. Такие листы по картинке не различимы, и ни один из них не
        может быть объявлен определённым.
        """
        for paper in ((1159, 818), (277, 195), (816, 574)):
            frame, shape, _, _ = self._sheet_with_margin(paper, margin_mm)
            assert vision._format_certain(frame, shape) is False, (paper, margin_mm)

    def test_an_isolated_aspect_stays_certain(self):
        """
        A4 книжный пропорцию ни с кем не делит (ближайший сосед в 2,9 %),
        поэтому его формат определён при любом поле скана.
        """
        for margin_mm in (0, 3, 8, 15, 30):
            frame, shape, _, _ = self._sheet_with_margin((185, 287), margin_mm)
            assert vision._format_options(frame, shape) == [(185, 287)]
            assert vision._format_certain(frame, shape) is True

    def test_size_no_longer_refuses_a_format(self, data):
        """
        Доля листа только упорядочивает форматы. Настоящая рамка с полем
        скана обязана получить формат, а не отказ.
        """
        height, width = data["mask_shape"]
        left, top, right, bottom = data["frame_after"]
        box_w, box_h = (right - left) * width, (bottom - top) * height
        for margin in (1.0, 1.05, 1.2):                 # картинка шире листа
            paper, why = vision._format_of(
                box_w, box_h, int(width * margin), int(height * margin), 0.02)
            assert paper is not None, (margin, why)


# ===========================================================================
# Причина отказа
# ===========================================================================

class TestRefusalReport:
    def test_reasons_are_counted_not_overwritten(self):
        """
        Кандидатов перебирается много. В поле должна попасть не причина
        последнего по счёту, а все с числом кандидатов, от частой к редкой.
        """
        refused = vision._Refusals()
        for _ in range(3):
            refused.add("пропорция не отвечает формату ГОСТ 2.301")
        refused.add("нет поля подшивки слева")
        summary = refused.summary("по умолчанию")
        assert summary.startswith("пропорция не отвечает формату ГОСТ 2.301 (кандидатов 3)")
        assert "нет поля подшивки слева (кандидатов 1)" in summary

    def test_default_is_used_when_no_candidate_was_judged(self):
        assert vision._Refusals().summary("сторон не нашлось") == "сторон не нашлось"

    def test_real_sheet_format_is_among_the_options(self, data):
        """
        На настоящих листах формат рамки — A3. Однозначно он не определён
        (A2 с полем скана в несколько миллиметров даёт ту же картинку), но
        истинный формат обязан остаться среди возможных, и самым вероятным.
        """
        shape = data["mask_shape"]
        options = vision._format_options(data["frame_after"], shape)
        assert (395, 287) in options
        assert data["format_mm"] == "395x287"

    def test_cropped_sheet_loses_the_size_argument(self, data):
        """
        Рамка, упирающаяся в край картинки: поля листа в кадре нет, доля
        формат не ограничивает, и возможных форматов становится только
        больше.
        """
        shape = data["mask_shape"]
        inside = vision._format_options(data["frame_after"], shape)
        touching = list(data["frame_after"])
        touching[2] = 0.9999
        assert len(vision._format_options(touching, shape)) >= len(inside)
        assert vision._format_certain(touching, shape) is False

    def test_mask_failure_is_recorded(self, flags, monkeypatch):
        flags(stitch=True, four_sides=True, from_title_block=True)
        monkeypatch.setattr(vision, "_thin_line_mask", lambda image: None)
        signals = {}
        assert vision._recover_frame(object(), [], signals) is None
        assert signals["frame_recovery"]["refused"] == {"mask": "маска листа не построилась"}


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
