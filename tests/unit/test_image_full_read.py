"""
Полное чтение листа-картинки (DRAWING_IMAGE_FULL_READ).

Лист JPG, из которого MinerU нарезал виды, без флага читается моделью только
по вырезкам: штамп, номер листа и примечания остаются за ними. Лист, который
лестница назвала текстом, в модель не попадает вовсе. Здесь проверяется, что
при включённом флаге лист читается целиком со всем текстом, а при
выключенном — только по вырезкам.
"""

import json

import pytest

from core.config import settings
from core.drawing_processor import DrawingProcessor
from core.drawing_quality import document_quality
from core.models.contract import Content, DrawingField, Fragment, Position, Provenance
from core.providers import raster_drawing
from core.providers.qwen_client import QwenVisionClient, SheetReading

pytest.importorskip("numpy")
pytest.importorskip("PIL")

from tests.fixtures import synthetic_sheet as sheet  # noqa: E402

SHEET_URI = "src/sheet.png"
CROP_URI = "assets/doc-1/crop.png"
LINES = ["ТАДУ 408721.917.00 СБ", "Лист 3", "1. * Размеры для справок."]


class FakeQwen:
    def __init__(self, reading=None, lines=None):
        self.available = True
        self.model = "fake-vl"
        self.reading = reading
        self.lines = lines
        self.asked = []

    def read_sheet(self, image, hint=""):
        self.asked.append("sheet")
        return self.reading

    def read_title_block(self, image):
        self.asked.append("title_block")
        return None

    def read_table(self, image):
        self.asked.append("table")
        return None

    def transcribe(self, image):
        self.asked.append("transcribe")
        return self.lines


def reading(*values):
    return SheetReading(sheet_type="assembly", annotations=[
        {"category": "size", "value": value, "bbox": [0.1, 0.1, 0.2, 0.2]} for value in values
    ])


def crop_fragment():
    return Fragment(
        fragment_id="doc-1-frag-001", type="image", content=Content(image_ref=CROP_URI),
        position=Position(page=1, bbox=[0.3, 0.2, 0.7, 0.5], order=1),
        confidence=0.9, completeness=1.0,
        provenance=Provenance(method="mineru_image", strategy_level=5, source="raster_only"),
    )


def metadata(genre="drawing", file_type="jpg"):
    return {"doc_id": "doc-1", "source_path": SHEET_URI, "file_type": file_type,
            "classification": {"genre": genre, "source": "raster_only"}}


@pytest.fixture
def storage(temp_storage):
    temp_storage.write_file(SHEET_URI, sheet.png_bytes())
    temp_storage.write_file(CROP_URI, sheet.png_bytes())
    return temp_storage


@pytest.fixture
def full_read(monkeypatch):
    monkeypatch.setattr(settings, "DRAWING_IMAGE_FULL_READ", True)


def model(storage, **kwargs):
    qwen = FakeQwen(reading=reading("20"), lines=LINES, **kwargs)
    return DrawingProcessor(storage, qwen=qwen), qwen


class TestWholeSheetWithCrops:
    def test_flag_off_crops_hide_the_sheet(self, storage):
        processor, qwen = model(storage)
        out = processor.enrich_fragments([crop_fragment()], metadata())
        assert [f.content.image_ref for f in out] == [CROP_URI]
        assert "transcribe" not in qwen.asked

    def test_flag_on_sheet_is_read_whole_with_its_text(self, storage, full_read):
        processor, qwen = model(storage)
        out = processor.enrich_fragments([crop_fragment()], metadata())

        assert [f.content.image_ref for f in out] == [SHEET_URI, CROP_URI]
        sheet_fragment = out[0]
        assert sheet_fragment.type == "drawing"
        assert sheet_fragment.content.text == "\n".join(LINES)
        payload = sheet_fragment.content.structured_payload
        assert payload["transcript_line_count"] == len(LINES)
        # Текст лежит один раз: в content.text, а не ещё и в payload.
        assert "transcript_lines" not in payload
        # Вырезка читается одна, без прохода «весь текст».
        assert qwen.asked.count("transcribe") == 1

    def test_transcript_alone_makes_a_fragment(self, storage, full_read):
        processor, _ = model(storage)
        processor.qwen.reading = None
        out = processor.enrich_fragments([], metadata())
        assert out[0].type == "drawing"
        assert out[0].content.text.startswith("ТАДУ 408721.917.00 СБ")
        assert not out[0].content.structured_drawing_fields

    def test_nothing_read_is_still_not_recognized(self, storage, full_read):
        processor, _ = model(storage)
        processor.qwen.reading, processor.qwen.lines = None, None
        out = processor.enrich_fragments([], metadata())
        assert out == []


class TestSheetCalledText:
    def test_flag_off_text_sheet_skips_the_model(self, storage):
        processor, qwen = model(storage)
        assert processor.enrich_fragments([], metadata(genre="text")) == []
        assert not qwen.asked

    def test_flag_on_frame_on_text_sheet_is_read(self, storage, full_read):
        assert raster_drawing.analyse(sheet.png_bytes()).is_drawing
        processor, _ = model(storage)
        out = processor.enrich_fragments([], metadata(genre="text"))
        assert [f.type for f in out] == ["drawing"]
        assert out[0].content.text

    def test_flag_on_text_scan_is_left_alone(self, storage, full_read, mocker):
        mocker.patch.object(raster_drawing, "analyse", return_value=raster_drawing.RasterVerdict(
            raster_drawing.KIND_TEXT_SCAN, 0.75))
        processor, qwen = model(storage)
        assert processor.enrich_fragments([], metadata(genre="text")) == []
        assert not qwen.asked

    def test_flag_on_text_scan_in_a_frame_is_read(self, storage, full_read, mocker):
        # Лист со штампом и таблицей: строк много, детектор зовёт его
        # текстом, но рамка формата у него есть.
        mocker.patch.object(raster_drawing, "analyse", return_value=raster_drawing.RasterVerdict(
            raster_drawing.KIND_TEXT_SCAN, 0.75, signals={"frame": True}))
        processor, _ = model(storage)
        out = processor.enrich_fragments([], metadata(genre="text"))
        assert [f.type for f in out] == ["drawing"]

    def test_flag_on_pdf_is_not_an_image_input(self, storage, full_read):
        processor, qwen = model(storage)
        assert processor.enrich_fragments([], metadata(genre="text", file_type="pdf")) == []
        assert not qwen.asked


def drawing(page=1, missing=(), text=None, lines=0):
    payload = {"transcript_line_count": lines} if lines else {}
    return Fragment(
        fragment_id=f"d{page}-{len(missing)}-{lines}", type="drawing",
        content=Content(text=text, structured_payload=payload, structured_drawing_fields=[
            DrawingField(category="size", value="20", confidence=0.5)]),
        position=Position(page=page), confidence=0.5, completeness=None,
        review_required=True, warnings=["DRAWING_COMPLETENESS_UNKNOWN"],
        missing_regions=[{"kind": "table", "bbox": [0, 0, 1, 1], "reason": "region_not_read",
                          "page": page} for _ in missing],
        provenance=Provenance(method="vision_plus_vlm", strategy_level=7, source="drawing"),
    )


class TestCoveredRegions:
    @pytest.fixture(autouse=True)
    def unmeasured(self, monkeypatch):
        monkeypatch.setattr(settings, "DRAWING_UNMEASURED_NOT_PARTIAL", True)

    def test_flag_off_unread_region_is_loss(self):
        partial, _ = document_quality([drawing(text="ТАДУ", lines=1), drawing(missing=[1])])
        assert partial

    def test_region_on_transcribed_sheet_is_not_loss(self, full_read):
        partial, _ = document_quality([drawing(text="ТАДУ", lines=1), drawing(missing=[1])])
        assert not partial

    def test_region_on_other_page_is_still_loss(self, full_read):
        partial, _ = document_quality([drawing(page=2, text="ТАДУ", lines=1), drawing(missing=[1])])
        assert partial

    def test_empty_transcript_covers_nothing(self, full_read):
        partial, _ = document_quality([drawing(missing=[1])])
        assert partial

    def test_not_recognized_is_loss_whatever_the_transcript(self, full_read):
        lost = drawing(text="ТАДУ", lines=1)
        lost.warnings.append("DRAWING_NOT_RECOGNIZED")
        partial, _ = document_quality([lost])
        assert partial


class FakeSession:
    def __init__(self, content):
        self.content = content
        self.bodies = []

    def post(self, url, json=None, headers=None, timeout=None):
        self.bodies.append(json)
        content = self.content

        class Response:
            status_code = 200
            text = ""

            @staticmethod
            def json():
                return {"choices": [{"message": {"content": content}}]}
        return Response()


class TestTranscribeCall:
    def picture(self):
        from PIL import Image
        return Image.new("RGB", (4000, 2800), "white")

    def test_lines_are_returned_and_limits_applied(self):
        session = FakeSession(json.dumps({"lines": ["ТАДУ 421451.001 СБ", "", 7, None]},
                                         ensure_ascii=False))
        client = QwenVisionClient(endpoint="http://gpu:8000", session=session)
        assert client.transcribe(self.picture()) == ["ТАДУ 421451.001 СБ", "7"]
        assert session.bodies[0]["max_tokens"] == settings.DRAWING_TRANSCRIBE_MAX_TOKENS

    def test_answer_without_lines_is_none(self):
        client = QwenVisionClient(endpoint="http://gpu:8000", session=FakeSession('{"text": "x"}'))
        assert client.transcribe(self.picture()) is None
