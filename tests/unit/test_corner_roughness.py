"""
Знак общей шероховатости в правом верхнем углу листа (DRAWING_CORNER_ROUGHNESS).

Проход листа целиком мелкий знак «√Ra 6,3 (√)» в углу не читает. Угол
вырезается и читается одним коротким вызовом; пустой ответ не добавляет
ничего, а без бюджета листа вызова нет. Модель поддельная.
"""

from io import BytesIO

import pytest

from core.config import settings
from core.providers import sense_rules
from core.providers.qwen_client import SheetReading

pytest.importorskip("PIL")
from PIL import Image  # noqa: E402


class CornerQwen:
    available = True
    model = "fake-vl"

    def __init__(self, answer, sheet_fields=()):
        self.answer = answer
        self.sheet_fields = list(sheet_fields)
        self.corner_calls = []

    def read_sheet(self, image, hint="", **_):
        return SheetReading(sheet_type="detail", annotations=list(self.sheet_fields))

    def read_title_block(self, image):
        return None

    def read_table(self, image):
        return None

    def read_corner_roughness(self, image):
        self.corner_calls.append(image.size)
        return self.answer


def run(temp_storage, monkeypatch, answer, sheet_fields=(), flag=True, budget_left=True):
    from core.drawing_processor import DrawingProcessor
    from core.models.contract import Content, Fragment, Position, Provenance

    monkeypatch.setattr(settings, "DRAWING_CORNER_ROUGHNESS", flag)
    monkeypatch.setattr(settings, "DRAWING_VISION_ENABLED", False)
    if not budget_left:
        # Бюджет листа почти истрачен: на вызов модели его уже не хватит.
        monkeypatch.setattr(settings, "DRAWING_TIME_BUDGET", 10)
        monkeypatch.setattr(settings, "VLM_TIMEOUT", 30)
    data = BytesIO()
    Image.new("RGB", (1654, 1169), "white").save(data, format="PNG")
    temp_storage.write_file("assets/doc/sheet.png", data.getvalue())
    fragment = Fragment(
        fragment_id="doc-frag-1", type="image", content=Content(image_ref="assets/doc/sheet.png"),
        position=Position(page=1, bbox=[0.0, 0.0, 1.0, 1.0], order=1), confidence=0.9, completeness=1.0,
        provenance=Provenance(method="mineru_image", strategy_level=2, source="vector_pdf"),
    )
    qwen = CornerQwen(answer, sheet_fields)
    out = DrawingProcessor(temp_storage, qwen=qwen).enrich_fragments([fragment], {})
    sheet = next(f for f in out if f.content.image_ref == "assets/doc/sheet.png")
    return sheet, qwen


SIZE = {"category": "size", "value": "⌀18", "bbox": [0.4, 0.4, 0.43, 0.42]}


def roughness(sheet):
    return [f for f in sheet.content.structured_drawing_fields or [] if f.category == "roughness"]


def test_corner_read_adds_general_roughness_once(temp_storage, monkeypatch):
    sheet, qwen = run(temp_storage, monkeypatch, {"value": "Ra 6,3", "check_mark": True}, [SIZE])
    assert len(qwen.corner_calls) == 1
    (rough,) = roughness(sheet)
    # Голое значение, как у прочих полей шероховатости: подпись — категория.
    assert rough.value == "Ra 6,3 (√)" and rough.category == "roughness"
    x1, y1, x2, y2 = rough.bbox
    assert x1 >= 0.69 and y2 <= 0.13          # правый верхний угол листа
    # Рамка — место знака у правого края, а не весь угол: центр правее 0,85.
    assert (x1 + x2) / 2 > 0.85 and x2 - x1 < 0.15


def test_corner_read_empty_answer_adds_nothing(temp_storage, monkeypatch):
    sheet, qwen = run(temp_storage, monkeypatch, {"value": "", "unsure": True}, [SIZE])
    assert len(qwen.corner_calls) == 1 and roughness(sheet) == []


def test_corner_read_skips_known_roughness(temp_storage, monkeypatch):
    known = {"category": "roughness", "value": "Ra 6,3", "bbox": [0.85, 0.03, 0.9, 0.06]}
    sheet, _ = run(temp_storage, monkeypatch, {"value": "Ra 6,3", "check_mark": True}, [SIZE, known])
    assert [f.value for f in roughness(sheet)] == ["Ra 6,3"]


def test_corner_read_without_budget_makes_no_call(temp_storage, monkeypatch):
    sheet, qwen = run(temp_storage, monkeypatch, {"value": "Ra 6,3"}, [SIZE], budget_left=False)
    assert qwen.corner_calls == [] and roughness(sheet) == []
    assert sheet.content.structured_payload["corner_roughness"] == "no_time"


def test_flag_off_makes_no_call(temp_storage, monkeypatch):
    assert settings.DRAWING_CORNER_ROUGHNESS is False
    sheet, qwen = run(temp_storage, monkeypatch, {"value": "Ra 6,3"}, [SIZE], flag=False)
    assert qwen.corner_calls == [] and roughness(sheet) == []
    assert "corner_roughness" not in (sheet.content.structured_payload or {})


@pytest.mark.parametrize("answer, expected", [
    ({"value": "Ra 6,3", "check_mark": True}, "Ra 6,3 (√)"),
    ({"value": "√Ra 3.2", "check_mark": False}, "Ra 3,2"),
    ({"value": "Rz 20"}, "Rz 20"),
    ({"value": "Ra 6,3", "unsure": True}, None),
    ({"value": "6,3"}, None),
    ({"value": "S0,5*"}, None),
    (None, None),
])
def test_corner_answer_parsing(answer, expected):
    assert sense_rules.corner_roughness(answer) == expected


def test_corner_box_follows_frame():
    assert sense_rules.corner_box(None) == [0.7, 0.0, 1.0, 0.12]
    assert sense_rules.corner_box([0.1, 0.1, 0.9, 0.9]) == [0.66, 0.1, 0.9, 0.196]


# ---------------------------------------------------------------- правки ревью

def _image_fragment(temp_storage, ref, page=1, bbox=(0.0, 0.0, 1.0, 1.0), size=(1654, 1169)):
    from core.models.contract import Content, Fragment, Position, Provenance

    data = BytesIO()
    Image.new("RGB", size, "white").save(data, format="PNG")
    temp_storage.write_file(ref, data.getvalue())
    return Fragment(
        fragment_id=f"doc-{ref}", type="image", content=Content(image_ref=ref),
        position=Position(page=page, bbox=list(bbox), order=page), confidence=0.9, completeness=1.0,
        provenance=Provenance(method="mineru_image", strategy_level=2, source="vector_pdf"),
    )


class SheetTypeQwen(CornerQwen):
    def __init__(self, answer, sheet_type="detail", fail=False):
        super().__init__(answer, [SIZE])
        self.sheet_type = sheet_type
        self.fail = fail

    def read_sheet(self, image, hint="", **_):
        return SheetReading(sheet_type=self.sheet_type, annotations=list(self.sheet_fields))

    def read_corner_roughness(self, image):
        self.corner_calls.append(image.size)
        if self.fail:
            raise RuntimeError("модель недоступна")
        return self.answer


def _process(temp_storage, monkeypatch, fragments, qwen, layout=None, processor=None):
    from core.drawing_processor import DrawingProcessor
    from core.providers import drawing_vision

    monkeypatch.setattr(settings, "DRAWING_CORNER_ROUGHNESS", True)
    monkeypatch.setattr(settings, "DRAWING_VISION_ENABLED", False)
    if layout is not None:
        monkeypatch.setattr(drawing_vision, "plain", layout)
    processor = processor or DrawingProcessor(temp_storage, qwen=qwen)
    return processor.enrich_fragments(fragments, {}), processor


def test_turned_sheet_uses_frame_of_turned_image(temp_storage, monkeypatch):
    from core.providers import drawing_vision

    frame = [0.05, 0.03, 0.97, 0.95]

    def turned(image):
        # Рамка найдена на уже повёрнутом листе — в его долях.
        return drawing_vision.SheetLayout(image=image, turn=90, frame=list(frame))

    qwen = SheetTypeQwen({"value": "Ra 3,2"})
    out, _ = _process(temp_storage, monkeypatch, [_image_fragment(temp_storage, "assets/doc/a.png")],
                      qwen, layout=turned)
    (rough,) = roughness(out[0])
    assert rough.bbox == sense_rules.corner_sign_box(sense_rules.corner_box(frame))


def test_missing_frame_uses_sheet_corner(temp_storage, monkeypatch):
    qwen = SheetTypeQwen({"value": "Ra 3,2"})
    out, _ = _process(temp_storage, monkeypatch, [_image_fragment(temp_storage, "assets/doc/a.png")], qwen)
    (rough,) = roughness(out[0])
    assert rough.bbox == sense_rules.corner_sign_box(sense_rules.corner_box(None))


def test_vlm_exception_keeps_sheet(temp_storage, monkeypatch):
    qwen = SheetTypeQwen({"value": "Ra 3,2"}, fail=True)
    out, _ = _process(temp_storage, monkeypatch, [_image_fragment(temp_storage, "assets/doc/a.png")], qwen)
    assert len(qwen.corner_calls) == 1 and roughness(out[0]) == []
    assert [f.value for f in out[0].content.structured_drawing_fields] == ["⌀18"]


def test_unknown_sheet_type_makes_no_call(temp_storage, monkeypatch):
    qwen = SheetTypeQwen({"value": "Ra 3,2"}, sheet_type="unknown")
    out, _ = _process(temp_storage, monkeypatch, [_image_fragment(temp_storage, "assets/doc/a.png")], qwen)
    assert qwen.corner_calls == [] and roughness(out[0]) == []


def test_multi_page_reads_each_sheet_once(temp_storage, monkeypatch):
    qwen = SheetTypeQwen({"value": "Ra 3,2"})
    fragments = [_image_fragment(temp_storage, "assets/doc/a.png", page=1),
                 _image_fragment(temp_storage, "assets/doc/b.png", page=2)]
    out, _ = _process(temp_storage, monkeypatch, fragments, qwen)
    assert len(qwen.corner_calls) == 2
    assert [len(roughness(f)) for f in out if f.type == "drawing"] == [1, 1]


def test_whole_images_reset_between_documents(temp_storage, monkeypatch):
    qwen = SheetTypeQwen({"value": "Ra 3,2"})
    _, processor = _process(temp_storage, monkeypatch,
                            [_image_fragment(temp_storage, "assets/doc/a.png")], qwen)
    assert len(qwen.corner_calls) == 1 and processor._whole_images == set()
    # Следующий документ: та же ссылка — уже вырезка вида, не лист целиком.
    crop = _image_fragment(temp_storage, "assets/doc/a.png", bbox=(0.1, 0.1, 0.5, 0.5))
    _process(temp_storage, monkeypatch, [crop], qwen, processor=processor)
    assert len(qwen.corner_calls) == 1


def test_corner_call_does_not_repeat_read_timeout(monkeypatch):
    from core.providers.qwen_client import QwenVisionClient as QwenClient

    client = QwenClient.__new__(QwenClient)
    seen = {}

    def ask_text(image, prompt, max_side, max_tokens=None, **kwargs):
        seen.update(kwargs)
        return ('{"value": "Ra 3,2"}', "stop")

    monkeypatch.setattr(client, "_ask_text", ask_text, raising=False)
    monkeypatch.setattr(QwenClient, "available", property(lambda self: True), raising=False)
    assert client.read_corner_roughness(Image.new("RGB", (10, 10)))["value"] == "Ra 3,2"
    assert seen.get("read_timeout_retries") == 0
