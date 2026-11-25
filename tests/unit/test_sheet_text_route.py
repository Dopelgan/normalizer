"""
C8 (DRAWING_ROUTE_SHEET_TEXT_VLM): штамп и текст страницы-чертежа — модели;
U2 (DRAWING_COLLAPSE_REREAD, DRAWING_VLM_REJECT_DETAILS): схлопнувшееся чтение
листа перечитывается, отказы модели хранятся с причиной.

Строки — синтетические: проверяется маршрут, а не содержание листа.
"""

import json

import pytest

from core.config import settings
from core.drawing_processor import DrawingProcessor
from core.models.contract import Content, DrawingField, Fragment, Position, Provenance
from core.providers import qwen_client, title_block
from core.providers.qwen_client import SheetReading

pytest.importorskip("numpy")
pytest.importorskip("PIL")

from tests.fixtures import synthetic_sheet as sheet  # noqa: E402
from tests.unit.test_qwen_client import FakeResponse, FakeSession, answer, client, picture  # noqa: E402

SHEET_URI = "assets/doc-1/sheet.png"
OCR_TT = "1. Сварку выполнять по схеме сварки. 2. Размер 128 обеспечить. 3. Зазор 20+10 мм."
VLM_TT = ["1. Сварку выполнять по схеме сборки.", "2. Размер 138 обеспечить.", "3. Зазор 20±10 мм."]


class FakeQwen:
    def __init__(self, lines=None, block=None):
        self.available = True
        self.model = "fake-vl"
        self.lines = lines
        self.block = block
        self.transcribed = []
        self.stamps = 0

    def transcribe(self, image):
        self.transcribed.append(image.size)
        return self.lines

    def read_title_block_fields(self, image):
        self.stamps += 1
        return self.block

    def read_side_column(self, image):
        return None


@pytest.fixture
def storage(temp_storage):
    temp_storage.write_file(SHEET_URI, sheet.png_bytes())
    return temp_storage


@pytest.fixture
def flags(monkeypatch):
    monkeypatch.setattr(settings, "DRAWING_ROUTE_SHEET_TEXT_VLM", True)
    monkeypatch.setattr(settings, "DRAWING_TITLE_BLOCK_PASS", True)
    monkeypatch.setattr(settings, "DRAWING_VISION_ENABLED", False)
    monkeypatch.setattr(settings, "DRAWING_DIMENSION_FIDELITY", False)


def _picture(bbox, kind="drawing", fields=None):
    return Fragment(
        fragment_id="doc-1-frag-001", type=kind,
        content=Content(image_ref=SHEET_URI, structured_drawing_fields=fields or []),
        position=Position(page=1, bbox=bbox, order=1), confidence=0.8, completeness=1.0,
        provenance=Provenance(method="vlm", strategy_level=7, source="drawing"),
    )


def _text(text, bbox, fid="doc-1-frag-002"):
    return Fragment(
        fragment_id=fid, type="text", content=Content(text=text),
        position=Position(page=1, bbox=bbox, order=2), confidence=0.6, completeness=1.0,
        provenance=Provenance(method="ocr_layer", strategy_level=5, source="scanned_pdf"),
    )


def _route(storage, qwen, fragments):
    from core import sheet_text_route

    processor = DrawingProcessor(storage, qwen=qwen)
    return sheet_text_route.apply(processor, fragments, {"source_path": SHEET_URI, "doc_id": "doc-1"})


STAMP = {"fields": {"Обозначение": "АБВГ.000000.001 СБ", "Наименование": "Узел сварной"}}


class TestSheetTextRoute:
    def test_flag_off_changes_nothing(self, storage, monkeypatch):
        monkeypatch.setattr(settings, "DRAWING_ROUTE_SHEET_TEXT_VLM", False)
        qwen = FakeQwen(lines=VLM_TT, block=title_block.parse(STAMP))
        fragments = [_picture([0.05, 0.05, 0.7, 0.75]), _text(OCR_TT, [0.72, 0.1, 0.95, 0.4])]
        out = _route(storage, qwen, fragments)
        assert [f.fragment_id for f in out] == ["doc-1-frag-001", "doc-1-frag-002"]
        assert out[1].content.text == OCR_TT and out[1].content.structured_payload is None
        assert qwen.transcribed == [] and qwen.stamps == 0

    def test_drawing_page_text_and_stamp_go_to_model(self, storage, flags):
        qwen = FakeQwen(lines=VLM_TT, block=title_block.parse(STAMP))
        fragments = [_picture([0.05, 0.05, 0.7, 0.75]), _text(OCR_TT, [0.72, 0.1, 0.95, 0.4])]
        out = _route(storage, qwen, fragments)
        stamp = out[0]
        assert stamp.type == "drawing" and stamp.content.structured_payload["stamp_only"] is True
        assert any("Узел сварной" in f.value for f in stamp.content.structured_drawing_fields)
        tt = next(f for f in out if f.fragment_id == "doc-1-frag-002")
        assert tt.content.text == "\n".join(VLM_TT)
        assert tt.provenance.method == "vlm_sheet_text"
        assert tt.content.structured_payload["sheet_text_route"]["source"] == "vlm_crop"
        # Вырезка блока, а не лист целиком.
        assert len(qwen.transcribed) == 1 and qwen.transcribed[0][0] < sheet.WIDTH

    @pytest.mark.parametrize("lines, reason", [(None, "vlm_no_answer"), ([], "vlm_empty"),
                                               (["1."], "vlm_shorter")])
    def test_model_refusal_keeps_ocr(self, storage, flags, lines, reason):
        qwen = FakeQwen(lines=lines, block=None)
        fragments = [_picture([0.05, 0.05, 0.7, 0.75]), _text(OCR_TT, [0.72, 0.1, 0.95, 0.4])]
        out = _route(storage, qwen, fragments)
        tt = next(f for f in out if f.fragment_id == "doc-1-frag-002")
        assert tt.content.text == OCR_TT and tt.provenance.method == "ocr_layer"
        assert tt.content.structured_payload["sheet_text_route"] == {
            "ocr_chars": tt.content.structured_payload["sheet_text_route"]["ocr_chars"],
            "vlm_chars": tt.content.structured_payload["sheet_text_route"]["vlm_chars"],
            "source": "ocr_fallback", "reason": reason}

    def test_text_inside_picture_is_left_to_drawing_pass(self, storage, flags):
        qwen = FakeQwen(lines=VLM_TT)
        fragments = [_picture([0.05, 0.05, 0.7, 0.75], fields=[DrawingField(
            category="title_block", value="Обозначение: X", bbox=[0.6, 0.9, 0.9, 0.95], confidence=0.8)]),
            _text("подпись вида", [0.1, 0.1, 0.3, 0.15])]
        out = _route(storage, qwen, fragments)
        assert qwen.transcribed == [] and qwen.stamps == 0
        assert out[1].content.text == "подпись вида"

    def test_small_picture_without_frame_is_not_a_drawing_page(self, storage, flags):
        # Картинка в тексте (логотип) — не чертёж: страница идёт прежним путём.
        qwen = FakeQwen(lines=VLM_TT, block=title_block.parse(STAMP))
        fragments = [_picture([0.05, 0.05, 0.2, 0.12], kind="image"), _text(OCR_TT, [0.1, 0.2, 0.9, 0.5])]
        out = _route(storage, qwen, fragments)
        assert qwen.transcribed == [] and qwen.stamps == 0
        assert out[1].content.text == OCR_TT

    def test_small_picture_on_framed_sheet_is_a_drawing_page(self, storage, flags, monkeypatch):
        # Рамка формата на странице: рисунок мелкий, но страница — лист чертежа.
        monkeypatch.setattr(settings, "DRAWING_VISION_ENABLED", True)
        qwen = FakeQwen(lines=VLM_TT, block=None)
        fragments = [_picture([0.1, 0.1, 0.3, 0.3]), _text(OCR_TT, [0.4, 0.1, 0.7, 0.4])]
        out = _route(storage, qwen, fragments)
        assert next(f for f in out if f.fragment_id == "doc-1-frag-002").content.text == "\n".join(VLM_TT)

    def test_blocks_are_bounded(self, storage, flags, monkeypatch):
        monkeypatch.setattr(settings, "DRAWING_ROUTE_SHEET_TEXT_MAX_BLOCKS", 1)
        qwen = FakeQwen(lines=VLM_TT)
        fragments = [_picture([0.05, 0.05, 0.7, 0.75]), _text(OCR_TT, [0.72, 0.1, 0.95, 0.3]),
                     _text(OCR_TT, [0.72, 0.4, 0.95, 0.6], fid="doc-1-frag-003")]
        _route(storage, qwen, fragments)
        assert len(qwen.transcribed) == 1


class TestCollapseReread:
    class Qwen:
        def __init__(self, tiles_items):
            self.tiles_items = tiles_items
            self.calls = 0

        def reread_sheet_by_tiles(self, image, whole, hint="", until=None):
            self.calls += 1
            return SheetReading(sheet_type=whole.sheet_type, tiles=4,
                                annotations=whole.annotations + [{"category": "note", "value": str(i)}
                                                                 for i in range(self.tiles_items)])

    def _layout(self):
        from core.providers import drawing_vision
        from PIL import Image

        return drawing_vision.plain(Image.new("RGB", (200, 100), "white"))

    def _reading(self, n):
        return SheetReading(sheet_type="assembly", annotations=[{"category": "note", "value": str(i)}
                                                                for i in range(n)])

    def test_collapsed_reading_is_reread_once(self, storage, monkeypatch):
        monkeypatch.setattr(settings, "DRAWING_TIME_BUDGET", 0)
        qwen = self.Qwen(tiles_items=100)
        payload = {}
        reading = self._reading(17)
        out = DrawingProcessor(storage, qwen=qwen)._guard_collapsed_sheet(
            self._layout(), reading, 124, 0.0, payload)
        assert qwen.calls == 1 and len(out.annotations) == 117
        assert payload["collapse_reread"]["kept"] == "tiles"

    @pytest.mark.parametrize("items, lines", [(40, 124), (3, 10)])
    def test_normal_reading_is_kept(self, storage, monkeypatch, items, lines):
        # 40 надписей на 124 строки — обычный лист; 10 строк — мало для вывода.
        qwen = self.Qwen(tiles_items=100)
        reading = self._reading(items)
        out = DrawingProcessor(storage, qwen=qwen)._guard_collapsed_sheet(
            self._layout(), reading, lines, 0.0, {})
        assert out is reading and qwen.calls == 0

    def test_reread_that_finds_less_keeps_first(self, storage, monkeypatch):
        monkeypatch.setattr(settings, "DRAWING_TIME_BUDGET", 0)
        qwen = self.Qwen(tiles_items=0)
        reading = self._reading(5)
        payload = {}
        out = DrawingProcessor(storage, qwen=qwen)._guard_collapsed_sheet(self._layout(), reading, 60, 0.0, payload)
        assert out is reading and payload["collapse_reread"]["kept"] == "first"

    def test_no_reread_without_time(self, storage, monkeypatch):
        from core.providers import gpu_gate

        monkeypatch.setattr(settings, "DRAWING_TIME_BUDGET", 100)
        monkeypatch.setattr(gpu_gate, "budget_now", lambda: 1000.0)
        qwen = self.Qwen(tiles_items=100)
        payload = {}
        out = DrawingProcessor(storage, qwen=qwen)._guard_collapsed_sheet(
            self._layout(), self._reading(2), 60, 0.0, payload)
        assert qwen.calls == 0 and out.annotations == self._reading(2).annotations
        assert payload["collapse_reread"]["skipped"] == "time_budget"

    def test_sheet_read_by_native_tiles_is_not_reread(self, storage, monkeypatch):
        monkeypatch.setattr(settings, "DRAWING_TIME_BUDGET", 0)
        qwen = self.Qwen(tiles_items=100)
        reading = self._reading(5)
        reading.native_tiles = {"grid": [2, 2], "read": 4}
        out = DrawingProcessor(storage, qwen=qwen)._guard_collapsed_sheet(
            self._layout(), reading, 124, 0.0, {})
        assert out is reading and qwen.calls == 0

    def test_reread_keeps_native_tiles(self, storage, monkeypatch):
        # Пустой ответ при плитках в родном разрешении перечитывается частями,
        # но учёт плиток остаётся у нового прочтения.
        monkeypatch.setattr(settings, "DRAWING_TIME_BUDGET", 0)
        qwen = self.Qwen(tiles_items=3)
        reading = self._reading(0)
        reading.native_tiles = {"grid": [2, 2], "read": 0}
        out = DrawingProcessor(storage, qwen=qwen)._guard_empty_sheet(self._layout(), reading, 60, 0.0)
        assert out is not reading and out.native_tiles == {"grid": [2, 2], "read": 0}


class TestRejectDetails:
    @pytest.mark.parametrize("content, finish, reason", [
        ('{"lines": ["a"', "length", "cut"), ("не JSON", "stop", "not_json"),
        ("{}", "stop", "empty_object"), ('{"a": 1}', "stop", "check_failed")])
    def test_reason(self, content, finish, reason):
        assert qwen_client.reject_reason(content, finish) == reason

    def test_raw_and_reason_logged_and_kept_with_flag(self, mocker, tmp_path):
        mocker.patch.object(qwen_client.time, "sleep")
        log = tmp_path / "rejected.jsonl"
        mocker.patch.object(qwen_client.settings, "DRAWING_VLM_STABLE", True)
        mocker.patch.object(qwen_client.settings, "DRAWING_VLM_REJECTED_LOG", str(log))
        mocker.patch.object(qwen_client.settings, "DRAWING_VLM_REJECT_DETAILS", True)
        session = FakeSession(FakeResponse(answer("{}")), FakeResponse(answer({"fields": {"Масса": "3,74"}})))
        model = client(session)
        model.read_title_block_fields(picture())
        row = json.loads(log.read_text(encoding="utf-8").splitlines()[0])
        assert row["reason"] == "empty_object" and row["raw"] == "{}"
        assert [r["reason"] for r in model.rejected_answers] == ["empty_object"]

        from core import sheet_text_route

        drawing = _picture([0, 0, 1, 1])
        sheet_text_route.attach_rejected(model, [_text("x", [0.1, 0.1, 0.2, 0.2]), drawing])
        assert drawing.content.structured_payload["vlm_rejected"][0]["reason"] == "empty_object"

    def test_without_flag_log_row_is_as_before(self, mocker, tmp_path):
        mocker.patch.object(qwen_client.time, "sleep")
        log = tmp_path / "rejected.jsonl"
        mocker.patch.object(qwen_client.settings, "DRAWING_VLM_STABLE", True)
        mocker.patch.object(qwen_client.settings, "DRAWING_VLM_REJECTED_LOG", str(log))
        mocker.patch.object(qwen_client.settings, "DRAWING_VLM_REJECT_DETAILS", False)
        session = FakeSession(FakeResponse(answer("{}")), FakeResponse(answer({"fields": {"Масса": "3,74"}})))
        model = client(session)
        model.read_title_block_fields(picture())
        row = json.loads(log.read_text(encoding="utf-8").splitlines()[0])
        assert set(row) == {"key", "finish", "head", "length"}
        assert not hasattr(model, "rejected_answers")


class TestRouteLimits:
    def test_block_error_keeps_stamp_and_ocr(self, storage, flags):
        class Failing(FakeQwen):
            def transcribe(self, image):
                raise RuntimeError("обрыв соединения")

        qwen = Failing(block=title_block.parse(STAMP))
        fragments = [_picture([0.05, 0.05, 0.7, 0.75]), _text(OCR_TT, [0.72, 0.1, 0.95, 0.4])]
        out = _route(storage, qwen, fragments)
        assert out[0].content.structured_payload["stamp_only"] is True
        tt = next(f for f in out if f.fragment_id == "doc-1-frag-002")
        assert tt.content.text == OCR_TT
        assert tt.content.structured_payload["sheet_text_route"] == {"source": "ocr_fallback",
                                                                     "reason": "vlm_error"}

    def test_spent_drawing_budget_stops_route(self, storage, flags, monkeypatch):
        from core.providers import gpu_gate

        clock = iter([0.0] + [1000.0] * 10)
        monkeypatch.setattr(settings, "DRAWING_TIME_BUDGET", 100)
        monkeypatch.setattr(gpu_gate, "budget_now", lambda: next(clock))
        qwen = FakeQwen(lines=VLM_TT, block=title_block.parse(STAMP))
        fragments = [_picture([0.05, 0.05, 0.7, 0.75]), _text(OCR_TT, [0.72, 0.1, 0.95, 0.4])]
        out = _route(storage, qwen, fragments)
        assert qwen.transcribed == [] and qwen.stamps == 0
        assert len(out) == 2 and out[1].content.text == OCR_TT

    def test_render_dpi_capped_by_megapixels(self, monkeypatch):
        from core import sheet_text_route

        monkeypatch.setattr(settings, "DRAWING_ROUTE_SHEET_TEXT_DPI", 300)
        monkeypatch.setattr(settings, "DRAWING_ROUTE_SHEET_TEXT_RENDER_MAX_MP", 48.0)
        assert sheet_text_route.render_dpi(595, 842) == 300  # A4
        a0 = sheet_text_route.render_dpi(2384, 3370)
        assert a0 < 300
        assert (2384 / 72 * a0) * (3370 / 72 * a0) <= 48e6 * 1.001

    def test_pdf_page_render_respects_cap(self, temp_storage, monkeypatch):
        fitz = pytest.importorskip("fitz")
        from core import sheet_text_route

        document = fitz.open()
        document.new_page(width=1190, height=842)  # A3
        temp_storage.write_file("documents/sheet.pdf", document.tobytes())
        monkeypatch.setattr(settings, "DRAWING_ROUTE_SHEET_TEXT_RENDER_MAX_MP", 2.0)
        image = sheet_text_route._page_image(DrawingProcessor(temp_storage, qwen=FakeQwen()),
                                             "documents/sheet.pdf", 1)
        assert image.width * image.height <= 2e6 * 1.01
        assert image.info["render_dpi"] < settings.DRAWING_ROUTE_SHEET_TEXT_DPI

    def test_frame_check_runs_on_working_resolution(self, storage, monkeypatch):
        from PIL import Image

        from core import sheet_text_route
        from core.providers import drawing_vision

        monkeypatch.setattr(settings, "DRAWING_VISION_ENABLED", True)
        monkeypatch.setattr(settings, "DRAWING_RENDER_DPI", 150)
        seen = []
        monkeypatch.setattr(drawing_vision, "analyse",
                            lambda image: seen.append(image.size) or drawing_vision.SheetLayout(image=image))
        page = Image.new("RGB", (3000, 2000), "white")
        page.info["render_dpi"] = 300
        processor = DrawingProcessor(storage, qwen=FakeQwen())
        sheet_text_route._is_drawing_page([_picture([0.1, 0.1, 0.2, 0.2])], page, processor)
        assert seen == [(1500, 1000)]
