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
