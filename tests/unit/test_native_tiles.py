"""
Крупный лист плитками в родном разрешении (DRAWING_NATIVE_TILES).

Модель подменена: проверяется выбор сетки по формату листа, сведение
перекрытий, правило пары с проходом с рамками, отказ плиток, вырезка штампа
из родного рендера и то, что без флага всё прежнее (вызов листа и ключ кэша).
"""

import json

import pytest

from core.config import settings
from core.ladder import result_cache
from core.providers import native_tiles, qwen_client
from core.providers.qwen_client import QwenVisionClient, SheetReading

Image = pytest.importorskip("PIL.Image")

MM = 25.4
# Форматы ГОСТ 2.301, альбомно, мм.
A4 = (297, 210)
A3 = (420, 297)
A2 = (594, 420)
A1 = (841, 594)


def px(size_mm, dpi):
    return (round(size_mm[0] / MM * dpi), round(size_mm[1] / MM * dpi))


def item(value, category="note"):
    return {"category": category, "value": value}


# ---------------------------------------------------------------- сетка
class TestPlan:
    @pytest.mark.parametrize("fmt, grid", [
        (A4, (1, 1)), (A3, (2, 1)), (A2, (2, 2)),
    ])
    def test_grid_by_format_at_300_dpi(self, fmt, grid):
        layout = native_tiles.plan(px(fmt, 300), 300)
        assert (layout.columns, layout.rows) == grid
        assert layout.dpi >= settings.DRAWING_NATIVE_TILES_MIN_DPI
        assert layout.max_side <= settings.DRAWING_NATIVE_TILES_MAX_SIDE
        assert len(layout.boxes) == grid[0] * grid[1]

    def test_portrait_a3_splits_rows(self):
        layout = native_tiles.plan(px((297, 420), 300), 300)
        assert (layout.columns, layout.rows) == (1, 2)

    def test_a1_render_is_capped_and_grid_is_six(self):
        # A1 в 300 dpi — 70 Мп; рендер ограничен RENDER_MAX_MP, и сетка
        # подбирается под то разрешение, которое получилось.
        dpi = native_tiles.render_dpi(px(A1, 200), 200)
        assert 200 < dpi < 300
        assert px(A1, dpi)[0] * px(A1, dpi)[1] <= settings.DRAWING_NATIVE_TILES_RENDER_MAX_MP * 1e6 * 1.01
        layout = native_tiles.plan(px(A1, dpi), dpi)
        assert (layout.columns, layout.rows) == (3, 2)
        assert layout.max_side <= settings.DRAWING_NATIVE_TILES_MAX_SIDE

    def test_boxes_overlap_and_cover_sheet(self):
        boxes = native_tiles.grid_boxes(2, 2, 0.1)
        assert boxes[0][0] == 0.0 and boxes[0][1] == 0.0 and boxes[-1][2:] == [1.0, 1.0]
        assert boxes[0][2] > boxes[1][0]  # соседи по строке перекрываются
        assert boxes[0][3] > boxes[2][1]  # соседи по столбцу перекрываются

    @pytest.mark.parametrize("fmt, reason", [(A4, ""), (A3, "small_print"), (A2, "small_print"),
                                             (A1, "small_print")])
    def test_wanted_by_cap_height(self, fmt, reason):
        # Рабочий растр — 200 dpi, проход листа ужимает до 2000 px.
        assert native_tiles.wanted(px(fmt, 200), 200, None) == reason

    def test_cut_answer_makes_a4_dense(self):
        assert native_tiles.wanted(px(A4, 200), 200, SheetReading(cut=True)) == "dense"


# ---------------------------------------------------------------- сведение
LEFT, RIGHT = [0.0, 0.0, 0.55, 1.0], [0.45, 0.0, 1.0, 1.0]
FAR_LEFT, FAR_RIGHT = [0.0, 0.0, 0.37, 1.0], [0.63, 0.0, 1.0, 1.0]


class TestMerge:
    def test_same_text_in_overlap_counts_once(self, mocker):
        mocker.patch.object(settings, "DRAWING_KEEP_REPEATS", True)
        merged = native_tiles.merge([(LEFT, [item("R1 Резистор")]), (RIGHT, [item("R1  резистор")])])
        assert [m["value"] for m in merged] == ["R1 Резистор"]
        assert merged[0]["area"] == [0.45, 0.0, 0.55, 1.0]

    def test_repeats_in_one_tile_and_far_tiles_are_kept_with_keep_repeats(self, mocker):
        mocker.patch.object(settings, "DRAWING_KEEP_REPEATS", True)
        merged = native_tiles.merge([
            (FAR_LEFT, [item("⌀8", "size"), item("⌀8", "size")]),
            (FAR_RIGHT, [item("⌀8", "size")]),
        ])
        assert len(merged) == 3

    def test_without_keep_repeats_one_record_once(self, mocker):
        # Проход листа без DRAWING_KEEP_REPEATS повторов не пишет: плитки не
        # добавляют их тоже.
        mocker.patch.object(settings, "DRAWING_KEEP_REPEATS", False)
        merged = native_tiles.merge([
            (FAR_LEFT, [item("⌀8", "size"), item("⌀8", "size")]),
            (FAR_RIGHT, [item("⌀8", "size"), item("20±0,1", "size")]),
        ])
        assert [m["value"] for m in merged] == ["⌀8", "20±0,1"]


class TestPairRule:
    def test_located_wins_and_only_unpaired_are_added(self):
        located = SheetReading(sheet_type="scheme", annotations=[
            {"category": "title_block", "value": "АБВГ 421451003", "bbox": [0.7, 0.9, 0.9, 0.92]},
            {"category": "size", "value": "20", "bbox": [0.1, 0.1, 0.2, 0.2]},
        ])
        tiles = native_tiles.TileReading(sheet_type="detail", items=[
            {**item("АБВГ 421451.003", "note"), "area": [0, 0, 1, 1]},  # пара по записи без знаков
            {**item("20", "note"), "area": [0, 0, 1, 1]},              # точная пара
            {**item("Копировал"), "area": [0, 0, 1, 1]},               # подпись графы: шум
            {**item("C1 Конденсатор К10-17б 0,1 мкФ"), "area": [0, 0, 0.5, 1]},
        ], stats={"tiles": 2, "read": 2})
        merged = qwen_client.with_tiles(tiles, located)
        assert merged.sheet_type == "scheme"
        assert merged.annotations[:2] == located.annotations
        assert merged.annotations[2:] == [{"category": "note", "value": "C1 Конденсатор К10-17б 0,1 мкФ"}]
        assert merged.native_tiles["added"] == 1

    def test_no_located_reading(self):
        tiles = native_tiles.TileReading(sheet_type="detail", items=[{**item("⌀44H7", "size"), "area": [0, 0, 1, 1]}])
        merged = qwen_client.with_tiles(tiles, None)
        assert merged.sheet_type == "detail"
        assert merged.annotations == [{"category": "size", "value": "⌀44H7"}]


# ---------------------------------------------------------------- чтение
class FakeTiles:
    """Ответы плиток по очереди; None — отказ, исключение — падение."""

    def __init__(self, *answers):
        self.answers = list(answers)
        self.sizes = []

    def __call__(self, image, max_side):
        self.sizes.append((image.size, max_side))
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer


def a3_sheet(render_calls=None):
    """A3 в рабочих 200 dpi; новый рендер — в запрошенном dpi."""
    base = Image.new("RGB", px(A3, 200), "white")

    def render(dpi):
        if render_calls is not None:
            render_calls.append(dpi)
        return Image.new("RGB", px(A3, dpi), "white")

    return native_tiles.NativeSheet(base, 200, render)


@pytest.fixture
def tiles_on(mocker):
    mocker.patch.object(settings, "DRAWING_NATIVE_TILES", True)
    mocker.patch.object(settings, "DRAWING_NATIVE_TILES_PARALLEL", 1)
    mocker.patch.object(settings, "DRAWING_SHEET_CONTENT_PASS", False)


def tile(*values):
    return SheetReading(sheet_type="scheme", annotations=[item(v) for v in values])


class TestReadSheet:
    def test_tiles_add_unpaired_to_located(self, mocker, tiles_on):
        calls = []
        located = SheetReading(sheet_type="scheme", annotations=[
            {"category": "note", "value": "R1 Резистор", "bbox": [0.1, 0.1, 0.3, 0.12]}])
        client = QwenVisionClient(endpoint="http://gpu-host:8000")
        mocker.patch.object(client, "_read_sheet_located", return_value=located)
        fake = FakeTiles(tile("R1 Резистор", "R2 Резистор"), tile("R2 Резистор", "C1 Конденсатор"))
        mocker.patch.object(client, "read_sheet_tile", side_effect=fake)
        reading = client.read_sheet(Image.new("RGB", (10, 10)), native=a3_sheet(calls))
        assert calls == [300.0]
        assert [a["value"] for a in reading.annotations] == ["R1 Резистор", "R2 Резистор", "C1 Конденсатор"]
        assert reading.annotations[0]["bbox"] == [0.1, 0.1, 0.3, 0.12]
        assert reading.native_tiles["grid"] == [2, 1] and reading.native_tiles["read"] == 2
        # Плитка уходит модели в родном разрешении, а не ужатой до листа.
        assert all(max(size) > settings.QWEN_SHEET_MAX_SIDE for size, _ in fake.sizes)

    def test_one_tile_failed_others_kept(self, mocker, tiles_on):
        client = QwenVisionClient(endpoint="http://gpu-host:8000")
        mocker.patch.object(client, "_read_sheet_located", return_value=SheetReading(sheet_type="scheme"))
        mocker.patch.object(client, "read_sheet_tile",
                            side_effect=FakeTiles(RuntimeError("обрыв"), tile("C1 Конденсатор")))
        reading = client.read_sheet(Image.new("RGB", (10, 10)), native=a3_sheet())
        assert [a["value"] for a in reading.annotations] == ["C1 Конденсатор"]
        assert reading.native_tiles["read"] == 1

    @pytest.mark.parametrize("stop", ["cancelled", "soft_limit"])
    def test_cancel_and_soft_limit_are_not_swallowed(self, mocker, tiles_on, stop):
        from celery.exceptions import SoftTimeLimitExceeded

        from core import deadline

        error = deadline.Cancelled() if stop == "cancelled" else SoftTimeLimitExceeded()
        client = QwenVisionClient(endpoint="http://gpu-host:8000")
        mocker.patch.object(client, "_read_sheet_located", return_value=SheetReading(sheet_type="scheme"))
        mocker.patch.object(client, "read_sheet_tile", side_effect=FakeTiles(error, tile("C1")))
        with pytest.raises(type(error)):
            client.read_sheet(Image.new("RGB", (10, 10)), native=a3_sheet())

    def test_all_tiles_refused_keeps_previous_reading(self, mocker, tiles_on):
        located = SheetReading(sheet_type="scheme", annotations=[item("R1")], cut=True)
        client = QwenVisionClient(endpoint="http://gpu-host:8000")
        mocker.patch.object(client, "_read_sheet_located", return_value=located)
        mocker.patch.object(client, "read_sheet_tile", side_effect=FakeTiles(None, None))
        assert client.read_sheet(Image.new("RGB", (10, 10)), native=a3_sheet()) is located

    def test_spent_budget_makes_no_tile_calls(self, mocker, tiles_on):
        located = SheetReading(sheet_type="scheme", annotations=[item("R1")])
        client = QwenVisionClient(endpoint="http://gpu-host:8000")
        mocker.patch.object(client, "_read_sheet_located", return_value=located)
        fake = mocker.patch.object(client, "read_sheet_tile")
        until = qwen_client.gpu_gate.budget_now() + 1
        assert client.read_sheet(Image.new("RGB", (10, 10)), until=until, native=a3_sheet()) is located
        fake.assert_not_called()

    def test_a4_without_cut_is_not_tiled(self, mocker, tiles_on):
        located = SheetReading(sheet_type="detail", annotations=[item("R1")])
        client = QwenVisionClient(endpoint="http://gpu-host:8000")
        mocker.patch.object(client, "_read_sheet_located", return_value=located)
        fake = mocker.patch.object(client, "read_sheet_tile")
        native = native_tiles.NativeSheet(Image.new("RGB", px(A4, 200), "white"), 200)
        assert client.read_sheet(Image.new("RGB", (10, 10)), native=native) is located
        fake.assert_not_called()


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload
        self.status_code = 200
        self.text = ""

    def json(self):
        return self._payload

    def raise_for_status(self):
        pass


class FakeSession:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests = []

    def post(self, url, json=None, headers=None, timeout=None):
        self.requests.append(json)
        return self.responses.pop(0)


class TestTilePrompt:
    def test_short_answer_without_boxes(self):
        body = {"sheet_type": "scheme", "annotations": [["n", "R1 Резистор"], ["s", "⌀8"]]}
        session = FakeSession(FakeResponse({"choices": [{"message": {"content": json.dumps(body)}}]}))
        reading = QwenVisionClient(endpoint="http://gpu-host:8000", session=session).read_sheet_tile(
            Image.new("RGB", (3000, 2000), "white"), 3400)
        assert [a["value"] for a in reading.annotations] == ["R1 Резистор", "⌀8"]
        request = session.requests[0]
        assert request["response_format"]["json_schema"]["name"] == "sheet_content"
        prompt = request["messages"][1]["content"][1]["text"]
        assert "bbox" not in prompt


# ---------------------------------------------------------------- штамп
class TestNativeCrop:
    def test_stamp_crop_comes_from_native_render(self):
        base = Image.new("RGB", (1000, 700), "white")
        rendered = []

        def render(dpi):
            rendered.append(dpi)
            return Image.new("RGB", (3000, 2100), "black")

        native = native_tiles.NativeSheet(base, 100, render)
        piece = native.crop([0.5, 0.74, 1.0, 1.0], 3.0, 4000, padding=0.0)
        assert rendered == [300.0]
        assert piece.getpixel((piece.width // 2, piece.height // 2)) == (0, 0, 0)
        # Увеличение делится на то, во сколько раз родной растр крупнее.
        assert abs(piece.width - 1500) <= 2

    def test_failed_render_falls_back_to_working_raster(self):
        base = Image.new("RGB", (1000, 700), "white")

        def render(dpi):
            raise RuntimeError("нет памяти")

        native = native_tiles.NativeSheet(base, 100, render)
        image, dpi = native.image()
        assert image is base and dpi == 100

    def test_render_cancel_is_not_swallowed(self):
        from core import deadline

        def render(dpi):
            raise deadline.Cancelled()

        native = native_tiles.NativeSheet(Image.new("RGB", (1000, 700), "white"), 100, render)
        with pytest.raises(deadline.Cancelled):
            native.image()


# ---------------------------------------------------------------- без флага
class TestFlagOff:
    def test_cache_key_settings_unchanged_without_flag(self, mocker):
        mocker.patch.object(settings, "DRAWING_NATIVE_TILES", False)
        snapshot = result_cache.settings_snapshot()
        assert not [name for name in snapshot if name.startswith("DRAWING_NATIVE_TILES")]
        mocker.patch.object(settings, "DRAWING_NATIVE_TILES", True)
        snapshot = result_cache.settings_snapshot()
        assert snapshot["DRAWING_NATIVE_TILES"] is True
        assert snapshot["DRAWING_NATIVE_TILES_MAX_SIDE"] == settings.DRAWING_NATIVE_TILES_MAX_SIDE

    def test_processor_makes_no_native_sheet_without_flag(self, mocker, temp_storage):
        from core.drawing_processor import DrawingProcessor
        from core.providers import drawing_vision

        mocker.patch.object(settings, "DRAWING_NATIVE_TILES", False)
        processor = DrawingProcessor(temp_storage, qwen=object())
        image = Image.new("RGB", px(A3, 200), "white")
        assert processor._native_sheet("x.png", 1, image, drawing_vision.plain(image)) is None

    def test_read_sheet_without_native_is_one_call(self, mocker):
        mocker.patch.object(settings, "DRAWING_SHEET_CONTENT_PASS", False)
        client = QwenVisionClient(endpoint="http://gpu-host:8000")
        located = SheetReading(sheet_type="scheme", cut=True)
        mocker.patch.object(client, "_read_sheet_located", return_value=located)
        fake = mocker.patch.object(client, "read_sheet_tile")
        assert client.read_sheet(Image.new("RGB", (10, 10))) is located
        fake.assert_not_called()


class TestPdfRender:
    def test_pdf_sheet_renders_again_at_native_dpi(self, mocker, temp_storage):
        fitz = pytest.importorskip("fitz")
        from core.drawing_processor import DrawingProcessor, _render_pdf_page
        from core.providers import drawing_vision

        mocker.patch.object(settings, "DRAWING_NATIVE_TILES", True)
        document = fitz.open()
        document.new_page(width=A3[0] / MM * 72, height=A3[1] / MM * 72)
        data = document.tobytes()
        document.close()
        temp_storage.write_file("assets/doc-1/sheet.pdf", data)
        image = _render_pdf_page(data, 1)
        assert image.info["render_dpi"] == settings.DRAWING_RENDER_DPI
        processor = DrawingProcessor(temp_storage, qwen=object())
        native = processor._native_sheet("assets/doc-1/sheet.pdf", 1, image, drawing_vision.plain(image))
        rendered, dpi = native.image()
        assert dpi == settings.DRAWING_NATIVE_TILES_DPI
        assert abs(rendered.size[0] - px(A3, dpi)[0]) <= 2


class StampQwen:
    available = True
    model = "fake-vl"

    def __init__(self):
        self.pieces = []

    def read_title_block_fields(self, image):
        self.pieces.append(image)
        return None

    def read_side_column(self, image):
        return None


@pytest.mark.parametrize("flag", [True, False])
def test_stamp_pass_reads_native_crop_with_gost_fallback(mocker, temp_storage, flag):
    # Рамка штампа не найдена: с флагом штамп — по месту ГОСТ 2.104 и из
    # родного рендера; без флага (и без DRAWING_TITLE_BLOCK_PASS) прежний путь.
    from core.drawing_processor import DrawingProcessor
    from core.providers import drawing_vision

    mocker.patch.object(settings, "DRAWING_NATIVE_TILES", flag)
    mocker.patch.object(settings, "DRAWING_TITLE_BLOCK_PASS", False)
    qwen = StampQwen()
    qwen.read_title_block = lambda image: qwen.pieces.append(image)
    processor = DrawingProcessor(temp_storage, qwen=qwen)
    base = Image.new("RGB", (1000, 700), "white")
    layout = drawing_vision.plain(base)
    region = drawing_vision.fallback_title_block(layout)
    native = native_tiles.NativeSheet(base, 100, lambda dpi: Image.new("RGB", (3000, 2100), "black"))
    payload = {"tables": []}
    processor._read_title_block(layout, region, [], payload, native=native if flag else None)
    piece = qwen.pieces[0]
    black = piece.getpixel((piece.width // 2, piece.height // 2)) == (0, 0, 0)
    assert black is flag
    assert ("title_block_bbox" in payload) is flag
