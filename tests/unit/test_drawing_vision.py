"""Разметка листа зрением: выравнивание, рамка формата, штамп, таблицы."""

import pytest

from core.providers import drawing_vision as vision

pytest.importorskip("numpy")
pytest.importorskip("PIL")

from tests.fixtures import synthetic_sheet as sheet  # noqa: E402


@pytest.fixture(scope="module")
def layout():
    return vision.analyse(sheet.build())


class TestFrame:
    def test_format_frame_is_found(self, layout):
        assert layout.frame is not None
        left, top, right, bottom = layout.frame
        assert left < 0.1 and top < 0.1
        assert right > 0.9 and bottom > 0.9

    def test_blank_sheet_has_no_frame(self):
        from PIL import Image

        empty = vision.analyse(Image.new("L", (1200, 900), 255))
        assert empty.frame is None
        assert empty.regions == []


class TestRegions:
    def test_title_block_found_in_bottom_right(self, layout):
        block = layout.title_block
        assert block is not None
        assert sheet.overlap(block.bbox, sheet.title_block_bbox()) > 0.7
        assert block.rows >= 3 and block.cols >= 3

    def test_table_found_and_is_not_the_title_block(self, layout):
        tables = [r for r in layout.regions if r.kind == vision.KIND_TABLE]
        assert tables, "таблица спецификации не найдена"
        assert sheet.overlap(tables[0].bbox, sheet.table_bbox()) > 0.7

    def test_cells_cover_the_grid(self, layout):
        block = layout.title_block
        assert len(block.cells) == block.rows * block.cols

    def test_hint_names_what_was_found(self, layout):
        hint = layout.hint()
        assert "рамка" in hint and "штамп" in hint


class TestDeskew:
    @pytest.mark.parametrize("angle", [0.8, -1.2])
    def test_tilt_is_measured_and_corrected(self, angle):
        tilted = vision.analyse(sheet.build(angle=angle))
        # Знак положительный против часовой, как у PIL.rotate.
        assert tilted.angle == pytest.approx(-angle, abs=0.35)
        assert tilted.title_block is not None

    def test_straight_sheet_is_not_rotated(self):
        straight = sheet.build()
        result = vision.analyse(straight)
        assert result.angle == 0.0
        assert result.image is straight

    def test_limit_zero_disables_correction(self):
        result = vision.analyse(sheet.build(angle=1.0), deskew_limit=0.0)
        assert result.angle == 0.0


class TestCrop:
    def test_region_is_cut_and_enlarged(self, layout):
        block = layout.title_block
        piece = vision.crop(layout.image, block.bbox, upscale=2.0)
        width = (block.bbox[2] - block.bbox[0]) * layout.image.width
        assert piece.width > width          # увеличено
        assert piece.width <= 2.4 * width   # и не больше запрошенного с полями

    def test_crop_survives_bbox_outside_the_sheet(self, layout):
        piece = vision.crop(layout.image, [-0.5, -0.5, 1.5, 1.5])
        assert piece.width > 0 and piece.height > 0


class TestPlain:
    def test_disabled_vision_returns_sheet_as_is(self):
        image = sheet.build()
        result = vision.plain(image)
        assert result.image is image
        assert result.regions == []
        assert result.hint() == ""


def _sideways(turn, with_table=False):
    """Лист, отсканированный боком: содержимое повёрнуто на `turn` против часовой."""
    return sheet.build(with_table=with_table).rotate(turn, expand=True, fillcolor=255)


class TestOrientation:
    """
    Альбомный лист, отсканированный книжным (или вверх ногами), без поворота
    теряет штамп: в правом нижнем углу его нет, и модель читает графы боком.
    """

    @pytest.fixture
    def no_osd(self, monkeypatch):
        monkeypatch.setattr("core.providers.ocr_prepare.osd_rotation", lambda _: None)

    @pytest.mark.parametrize("scanned, fix", [(90, 270), (180, 180), (270, 90)])
    def test_sideways_sheet_is_turned_by_title_block(self, no_osd, scanned, fix):
        layout = vision.analyse(_sideways(scanned))
        assert layout.turn == fix
        assert layout.signals["turn_source"] == "geometry"
        assert layout.image.size == (sheet.WIDTH, sheet.HEIGHT)
        assert layout.title_block is not None
        assert sheet.overlap(layout.title_block.bbox, sheet.title_block_bbox()) > 0.7
        assert "повёрнут" in layout.hint()
        assert layout.as_dict()["turn"] == fix

    @pytest.mark.parametrize("scanned", [90, 270])
    def test_confident_osd_settles_ambiguous_geometry(self, monkeypatch, scanned):
        # Спецификация в углу даёт второй «штамп» при другом повороте: одной
        # геометрией лист не поворачивается, уверенный OSD решает.
        monkeypatch.setattr("core.providers.ocr_prepare.osd_rotation", lambda _: (scanned, 6.0))
        layout = vision.analyse(_sideways(scanned, with_table=True))
        assert layout.turn == (-scanned) % 360
        assert layout.signals["turn_source"] == "osd"
        assert sheet.overlap(layout.title_block.bbox, sheet.title_block_bbox()) > 0.7
        tables = [r for r in layout.regions if r.kind == vision.KIND_TABLE]
        assert tables and sheet.overlap(tables[0].bbox, sheet.table_bbox()) > 0.7

    def test_ambiguous_geometry_without_osd_leaves_sheet_alone(self, no_osd):
        scanned = _sideways(90, with_table=True)
        layout = vision.analyse(scanned)
        assert layout.turn == 0
        assert layout.image is scanned

    def test_osd_that_disagrees_with_title_block_is_ignored(self, monkeypatch):
        monkeypatch.setattr("core.providers.ocr_prepare.osd_rotation", lambda _: (180, 9.0))
        layout = vision.analyse(_sideways(90))
        assert layout.turn == 0

    def test_unsure_osd_falls_back_to_strict_geometry(self, monkeypatch):
        monkeypatch.setattr("core.providers.ocr_prepare.osd_rotation", lambda _: (180, 0.5))
        layout = vision.analyse(_sideways(90))
        assert layout.turn == 270
        assert layout.signals["turn_source"] == "geometry"

    def test_upright_sheet_never_asks_osd(self, monkeypatch):
        def must_not_run(_):
            raise AssertionError("OSD спрошен для листа со штампом на месте")

        monkeypatch.setattr("core.providers.ocr_prepare.osd_rotation", must_not_run)
        straight = sheet.build()
        layout = vision.analyse(straight)
        assert layout.turn == 0 and layout.image is straight

    def test_margin_column_alone_does_not_turn_the_sheet(self, no_osd):
        # Узкая колонка дополнительных граф у левого поля (12×145 мм) — не
        # штамп: верный лист с ненайденным штампом не поворачивается.
        from PIL import Image, ImageDraw

        image = Image.new("L", (sheet.WIDTH, sheet.HEIGHT), 255)
        draw = ImageDraw.Draw(image)
        draw.rectangle([20, 20, sheet.WIDTH - 20, sheet.HEIGHT - 20], outline=0, width=3)
        draw.rectangle([60, 30, sheet.WIDTH - 30, sheet.HEIGHT - 30], outline=0, width=3)
        x0, x1, y0 = 64, 104, sheet.HEIGHT - 30 - 560
        draw.rectangle([x0, y0, x1, sheet.HEIGHT - 34], outline=0, width=2)
        draw.line([x0 + 20, y0, x0 + 20, sheet.HEIGHT - 34], fill=0, width=2)
        for offset in (100, 220, 320, 420):
            draw.line([x0, y0 + offset, x1, y0 + offset], fill=0, width=2)
        layout = vision.analyse(image)
        assert layout.turn == 0 and layout.image is image

    def test_switch_disables_turning(self, no_osd, monkeypatch):
        monkeypatch.setattr(vision.settings, "DRAWING_AUTO_ROTATE", False)
        assert vision.analyse(_sideways(90)).turn == 0

    def test_turn_is_dropped_when_title_block_does_not_appear(self, no_osd, monkeypatch):
        scanned = _sideways(90)
        real = vision._find_regions
        calls = []

        def second_pass_finds_nothing(image):
            calls.append(image.size)
            return real(image) if len(calls) == 1 else (None, [], {})

        monkeypatch.setattr(vision, "_find_regions", second_pass_finds_nothing)
        layout = vision.analyse(scanned)
        assert layout.turn == 0 and layout.image is scanned
        assert layout.signals["turn_rejected"] == "geometry"

    @pytest.mark.parametrize("turn", [90, 180, 270])
    def test_bbox_follows_pil_rotation(self, turn):
        from PIL import Image

        image = Image.new("L", (200, 100), 255)
        image.putpixel((150, 20), 0)                # точка в доле (0.75, 0.2)
        turned = image.rotate(turn, expand=True, fillcolor=255)
        x, y = next(
            (i, j) for j in range(turned.height) for i in range(turned.width)
            if turned.getpixel((i, j)) < 128
        )
        box = vision._turn_bbox([0.75, 0.2, 0.75, 0.2], turn)
        assert box[0] == pytest.approx(x / turned.width, abs=0.02)
        assert box[1] == pytest.approx(y / turned.height, abs=0.02)


class TestOrientationA4:
    """
    Штамп A4 во всю ширину рамки распадается на куски, а
    его левая часть слипается с колонкой граф на поле подшивки в высокую
    таблицу, которая при повороте на 90° выглядит штампом.
    """

    @pytest.mark.parametrize("osd", [None, (270, 9.0), (90, 9.0)])
    def test_upright_a4_is_never_turned(self, monkeypatch, osd):
        monkeypatch.setattr("core.providers.ocr_prepare.osd_rotation", lambda _: osd)
        image = sheet.build_a4_portrait()
        layout = vision.analyse(image)
        assert layout.turn == 0 and layout.image is image
        assert layout.title_block is not None

    @pytest.mark.parametrize("scanned", [90, 180, 270])
    def test_sideways_a4_is_not_turned_the_wrong_way(self, monkeypatch, scanned):
        monkeypatch.setattr("core.providers.ocr_prepare.osd_rotation", lambda _: None)
        layout = vision.analyse(sheet.build_a4_portrait().rotate(scanned, expand=True, fillcolor=255))
        assert layout.turn in (0, (-scanned) % 360)

    def test_inner_frame_skips_the_binding_margin(self):
        found = vision._find_regions(sheet.build_a4_portrait())
        inner = found[2]["inner_frame"]
        assert inner[0] == pytest.approx(20 / 210, abs=0.01)   # рамка, а не обрез листа
        assert inner[3] > 0.95                                  # линия штампа — не рамка
