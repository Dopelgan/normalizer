"""
Дочитывание плотного листа частями (DRAWING_TRANSCRIBE_TILES).

Электросхема или табличка дают больше строк, чем помещается в потолок ответа
модели: JSON обрывается, и раньше весь текст листа пропадал как «не-JSON».
Модель здесь поддельная: у неё ограниченный ответ, и лист, на котором текста
больше, она обрывает посреди строки, как vLLM с finish_reason=length.
"""

import base64
import io
import json

import pytest
import requests

from core import deadline
from core.config import settings
from core.drawing_processor import DrawingProcessor
from core.drawing_quality import document_quality
from core.models.contract import Content, Fragment, Position, Provenance
from core.providers.qwen_client import (
    TILE_CUT, TILE_NO_TIME, TILE_NOT_READ, QwenVisionClient, Transcript,
    merge_lines, salvage_lines, tile_boxes,
)

pytest.importorskip("numpy")
pytest.importorskip("PIL")
from PIL import Image  # noqa: E402

from tests.fixtures import synthetic_sheet as sheet  # noqa: E402

# Четверти листа окрашены по-разному: по цветам на картинке поддельная
# модель понимает, какую часть листа ей прислали.
COLORS = {
    "top_left": (255, 0, 0), "top_right": (0, 255, 0),
    "bottom_left": (0, 0, 255), "bottom_right": (255, 255, 0),
}
TEXT = {
    name: [f"{name} строка {index}" for index in range(6)] + ["XT1", "XT1"]
    for name in COLORS
}


def quartered(size=(800, 600)):
    width, height = size
    image = Image.new("RGB", size)
    half_w, half_h = width // 2, height // 2
    image.paste(COLORS["top_left"], (0, 0, half_w, half_h))
    image.paste(COLORS["top_right"], (half_w, 0, width, half_h))
    image.paste(COLORS["bottom_left"], (0, half_h, half_w, height))
    image.paste(COLORS["bottom_right"], (half_w, half_h, width, height))
    return image


def shown_parts(body):
    """Четверти, которые занимают заметную долю присланной картинки."""
    url = body["messages"][1]["content"][0]["image_url"]["url"]
    image = Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))).convert("RGB")
    counts = dict((color, count) for count, color in image.getcolors(1 << 24))
    total = image.size[0] * image.size[1]
    return [name for name, color in COLORS.items() if counts.get(color, 0) / total > 0.2]


def reply(content, finish="stop"):
    return {"choices": [{"message": {"content": content}, "finish_reason": finish}]}


class Response:
    def __init__(self, payload, status=200):
        self.payload, self.status_code, self.text = payload, status, ""

    def json(self):
        return self.payload


class TruncatingModel:
    """
    Отвечает строками тех четвертей, что видны на картинке. Если строк больше
    `capacity`, ответ обрывается посреди очередной строки (finish=length).
    `fail` — четверти, на которых модель не отвечает (обрыв связи).
    """

    def __init__(self, capacity=12, fail=(), cut_wider_than=None):
        self.capacity = capacity
        self.fail = set(fail)
        self.cut_wider_than = cut_wider_than
        self.calls = []

    def post(self, url, json=None, headers=None, timeout=None):
        parts = shown_parts(json)
        self.calls.append(parts)
        if parts and set(parts) <= self.fail:
            raise requests.ConnectionError("обрыв")
        lines = [line for name in parts for line in TEXT[name]]
        width = self._width(json)
        too_long = len(lines) > self.capacity or (
            self.cut_wider_than is not None and width > self.cut_wider_than)
        if not too_long:
            return Response(reply(_json(lines)))
        kept = lines[: self.capacity] if len(lines) > self.capacity else lines[:-1]
        head = '{\n"lines": [\n' + ",\n".join(_json_str(line) for line in kept)
        return Response(reply(head + ',\n"' + lines[len(kept)][:5], finish="length"))

    @staticmethod
    def _width(body):
        url = body["messages"][1]["content"][0]["image_url"]["url"]
        return Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))).size[0]


def _json(lines):
    return json.dumps({"lines": lines}, ensure_ascii=False)


def _json_str(line):
    return json.dumps(line, ensure_ascii=False)


def client(model):
    return QwenVisionClient(endpoint="http://qwen:8000/v1", model="fake-vl", session=model)


@pytest.fixture
def tiles(monkeypatch):
    monkeypatch.setattr(settings, "DRAWING_TRANSCRIBE_TILES", True)
    monkeypatch.setattr(settings, "QWEN_RETRIES", 0)
    monkeypatch.setattr(settings, "DRAWING_TRANSCRIBE_TILE_DEPTH", 1)


ALL_LINES = [line for name in COLORS for line in TEXT[name]]


class TestSalvage:
    def test_finished_lines_are_kept_and_the_cut_one_dropped(self):
        assert salvage_lines('{\n"lines": [\n"ТАДУ 421451.003-01 Э4",\n"XT1",\n"Каб') == [
            "ТАДУ 421451.003-01 Э4", "XT1"]

    def test_cut_right_after_a_comma(self):
        assert salvage_lines('```json\n{"lines": ["А", 12, "Б",') == ["А", "12", "Б"]

    def test_no_lines_key_gives_nothing(self):
        assert salvage_lines('{"annotations": [') == []
        assert salvage_lines("") == []

    def test_escaped_quotes_survive(self):
        assert salvage_lines('{"lines": ["Кабель \\"ВВГ\\"", "оборв') == ['Кабель "ВВГ"']


    def test_model_loop_is_collapsed(self):
        content = '{"lines": ["Блок", ' + '"1", ' * 500 + '"1'
        assert salvage_lines(content) == ["Блок", "1", "1", "1"]


class TestMerge:
    def test_overlap_duplicate_dropped_repeat_inside_part_kept(self):
        merged = merge_lines([["XT1", "XT1", "QF1"], ["QF1", "XT1", "QF2"]])
        assert merged == ["XT1", "XT1", "QF1", "QF2"]

    def test_spacing_and_case_do_not_make_a_new_line(self):
        assert merge_lines([["Блок  модульный"], ["блок модульный"]]) == ["Блок  модульный"]

    def test_partial_line_from_a_tile_is_not_lost(self):
        merged = merge_lines([["Кабель ВВГнг"], ["Кабель ВВГнг(А)-LS 5x6"]])
        assert merged == ["Кабель ВВГнг", "Кабель ВВГнг(А)-LS 5x6"]


class TestTileBoxes:
    def test_grid_covers_the_box_with_overlap(self):
        boxes = tile_boxes([0, 0, 1, 1], 2, 0.06)
        assert len(boxes) == 4
        assert boxes[0] == [0.0, 0.0, pytest.approx(0.53), pytest.approx(0.53)]
        assert boxes[3] == [pytest.approx(0.47), pytest.approx(0.47), 1.0, 1.0]

    def test_inner_box_stays_inside(self):
        for box in tile_boxes([0.5, 0.5, 1.0, 1.0], 2, 0.06):
            assert 0.5 <= box[0] < box[2] <= 1.0 and 0.5 <= box[1] < box[3] <= 1.0


class TestTranscribeSheet:
    def test_flag_off_cut_reply_is_lost_as_before(self):
        model = TruncatingModel(capacity=12)
        assert client(model).transcribe(quartered()) is None

    def test_cut_sheet_is_read_by_tiles_without_loss(self, tiles):
        model = TruncatingModel(capacity=12)
        transcript = client(model).transcribe_sheet(quartered())

        assert transcript.complete and transcript.tiles == 4
        # Каждая строка листа на месте; повтор внутри четверти сохранён, а
        # одинаковые строки разных четвертей сливаются (их текст не теряется).
        for name in COLORS:
            assert all(line in transcript.lines for line in TEXT[name])
        assert transcript.lines.count("XT1") == 2
        assert len(transcript.lines) == len(set(ALL_LINES)) + 1
        assert len(model.calls) == 5

    def test_sheet_that_fits_is_read_once(self, tiles):
        model = TruncatingModel(capacity=100)
        transcript = client(model).transcribe_sheet(quartered())
        assert transcript == Transcript(lines=ALL_LINES)
        assert len(model.calls) == 1

    def test_failed_tile_keeps_the_rest_and_is_reported(self, tiles):
        model = TruncatingModel(capacity=12, fail=["bottom_right"])
        transcript = client(model).transcribe_sheet(quartered())

        assert not transcript.complete
        assert [reason for _, reason in transcript.failed] == [TILE_NOT_READ]
        assert transcript.failed[0][0][0] > 0.4 and transcript.failed[0][0][1] > 0.4
        for name in ("top_left", "top_right", "bottom_left"):
            assert all(line in transcript.lines for line in TEXT[name])
        # Строки нижней правой четверти, успевшие попасть в оборванный ответ
        # листа, не выбрасываются: здесь их нет, ответ оборвался раньше.
        assert all(line in transcript.lines for line in ALL_LINES[:12])

    def test_tile_cut_at_last_depth_keeps_its_lines_and_is_partial(self, tiles):
        model = TruncatingModel(capacity=12, cut_wider_than=300)
        transcript = client(model).transcribe_sheet(quartered())
        assert {reason for _, reason in transcript.failed} == {TILE_CUT}
        assert len(transcript.failed) == 4
        # Из каждой оборванной четверти сохранены законченные строки.
        for name in COLORS:
            assert TEXT[name][0] in transcript.lines

    def test_cut_tile_is_split_again(self, tiles, monkeypatch):
        monkeypatch.setattr(settings, "DRAWING_TRANSCRIBE_TILE_DEPTH", 2)
        model = TruncatingModel(capacity=100, cut_wider_than=300)
        transcript = client(model).transcribe_sheet(quartered())
        assert transcript.complete
        assert transcript.tiles == 4 + 16
        assert len(model.calls) == 1 + 4 + 16

    def test_no_time_left_marks_tiles_unread(self, tiles):
        model = TruncatingModel(capacity=12)
        with deadline.scope(1):
            transcript = client(model).transcribe_sheet(quartered())
        assert {reason for _, reason in transcript.failed} == {TILE_NO_TIME}
        # Строки оборванного ответа листа остаются.
        assert transcript.lines == ALL_LINES[:12]
        assert len(model.calls) == 1

    def test_sheet_budget_spent_stops_tiles(self, tiles):
        import time
        model = TruncatingModel(capacity=12)
        transcript = client(model).transcribe_sheet(quartered(), until=time.monotonic() - 1)
        assert {reason for _, reason in transcript.failed} == {TILE_NO_TIME}
        assert len(model.calls) == 1

    def test_wrong_shape_is_not_tiled(self, tiles):
        class Shapeless(TruncatingModel):
            def post(self, url, json=None, headers=None, timeout=None):
                self.calls.append([])
                return Response(reply('{"annotations": []}'))
        model = Shapeless()
        assert client(model).transcribe_sheet(quartered()) is None
        assert len(model.calls) == 1

    def test_no_answer_on_sheet_is_none(self, tiles):
        model = TruncatingModel(fail=list(COLORS))
        assert client(model).transcribe_sheet(quartered()) is None


# ---------------------------------------------------------------- лист в целом
SHEET_URI = "src/sheet.png"


class FakeQwen:
    def __init__(self, transcript):
        self.available = True
        self.model = "fake-vl"
        self.transcript = transcript
        self.asked = []

    def read_sheet(self, image, hint=""):
        return None

    def read_title_block(self, image):
        return None

    def read_table(self, image):
        return None

    def transcribe(self, image):
        self.asked.append("transcribe")
        return None

    def transcribe_sheet(self, image, until=None):
        self.asked.append("transcribe_sheet")
        return self.transcript


@pytest.fixture
def storage(temp_storage):
    temp_storage.write_file(SHEET_URI, sheet.png_bytes())
    return temp_storage


@pytest.fixture
def sheet_flags(monkeypatch):
    monkeypatch.setattr(settings, "DRAWING_IMAGE_FULL_READ", True)
    monkeypatch.setattr(settings, "DRAWING_UNMEASURED_NOT_PARTIAL", True)
    monkeypatch.setattr(settings, "DRAWING_TRANSCRIBE_TILES", True)


def metadata():
    return {"doc_id": "doc-1", "source_path": SHEET_URI, "file_type": "jpg",
            "classification": {"genre": "drawing", "source": "raster_only"}}


def sheet_fragments(storage, transcript):
    processor = DrawingProcessor(storage, qwen=FakeQwen(transcript))
    return processor.enrich_fragments([], metadata()), processor.qwen


class TestSheetFragment:
    def test_flag_off_uses_single_pass(self, storage, sheet_flags, monkeypatch):
        monkeypatch.setattr(settings, "DRAWING_TRANSCRIBE_TILES", False)
        out, qwen = sheet_fragments(storage, Transcript(lines=["А"]))
        assert qwen.asked == ["transcribe"]
        assert out == []

    def test_tiled_sheet_is_a_whole_page(self, storage, sheet_flags):
        out, _ = sheet_fragments(storage, Transcript(lines=ALL_LINES, tiles=4))
        assert out[0].content.text == "\n".join(ALL_LINES)
        payload = out[0].content.structured_payload
        assert payload["transcript_tiles"] == 4
        assert "transcript_complete" not in payload
        partial, _ = document_quality(out)
        assert not partial

    def test_failed_tile_is_a_lost_region(self, storage, sheet_flags):
        failed = [([0.47, 0.47, 1.0, 1.0], TILE_NOT_READ)]
        out, _ = sheet_fragments(storage, Transcript(lines=ALL_LINES[:20], tiles=4, failed=failed))
        fragment = out[0]
        assert fragment.content.text.startswith(ALL_LINES[0])
        assert fragment.content.structured_payload["transcript_complete"] is False
        assert {"kind": "transcript_tile", "bbox": [0.47, 0.47, 1.0, 1.0],
                "reason": TILE_NOT_READ, "page": 1, "sheet": None} in fragment.missing_regions
        partial, _ = document_quality(out)
        assert partial

    def test_incomplete_sheet_does_not_cover_other_regions(self, sheet_flags):
        def drawing(missing, payload):
            return Fragment(
                fragment_id=f"d-{len(missing)}", type="drawing",
                content=Content(text="ТАДУ", structured_payload=payload),
                position=Position(page=1), confidence=0.5, completeness=None,
                missing_regions=[{"kind": "table", "bbox": [0, 0, 1, 1],
                                  "reason": "region_not_read", "page": 1} for _ in missing],
                provenance=Provenance(method="vision_plus_vlm", strategy_level=7, source="drawing"),
            )
        def tile_lost(payload):
            fragment = drawing([], payload)
            fragment.missing_regions = [{"kind": "transcript_tile", "bbox": [0, 0, 1, 1],
                                         "reason": TILE_NOT_READ, "page": 1}]
            return fragment
        whole = drawing([], {"transcript_line_count": 3})
        # Другой фрагмент страницы с целым текстом не покрывает недочитанную часть.
        assert document_quality([whole, tile_lost({"transcript_line_count": 3,
                                                   "transcript_complete": False})])[0]
        cut = drawing([], {"transcript_line_count": 3, "transcript_complete": False})
        assert not document_quality([whole, drawing([1], {})])[0]
        assert document_quality([cut, drawing([1], {})])[0]

