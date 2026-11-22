"""
Сжатый ответ прохода «лист целиком» (DRAWING_SHEET_COMPACT_JSON).

Надпись приходит массивом `[код, надпись, x1, y1, x2, y2]` с рамкой в
тысячных долях. Разбор обязан свести его к тем же объектам
`{category, value, bbox}`, что и прежний вид, принимать и прежний вид,
спасать целые надписи из оборванного ответа и оставлять неизвестный код
правилам — как любую выдуманную моделью категорию.
"""

import json

import pytest

from core.config import settings
from core.drawing_processor import DrawingProcessor
from core.providers import qwen_client
from core.providers.qwen_client import QwenVisionClient

Image = pytest.importorskip("PIL.Image")

OBJECTS = {"sheet_type": "detail", "annotations": [
    {"category": "size", "value": "⌀44H7", "bbox": [0.125, 0.25, 0.3, 0.35]},
    {"category": "roughness", "value": "Ra 3,2", "bbox": [0.4, 0.301, 0.5, 0.352]},
    {"category": "title_block", "value": "ТАДУ 687000.001", "bbox": [0.7, 0.9, 0.95, 0.95]},
    {"category": "position", "value": "3", "bbox": [0.05, 0.05, 0.06, 0.06]},
]}
COMPACT = {"sheet_type": "detail", "annotations": [
    ["s", "⌀44H7", 125, 250, 300, 350],
    ["r", "Ra 3,2", 400, 301, 500, 352],
    ["tb", "ТАДУ 687000.001", 700, 900, 950, 950],
    ["p", 3, 50, 50, 60, 60],
]}


def _client(monkeypatch, content, finish="stop"):
    client = QwenVisionClient(endpoint="http://vlm.test/v1", model="qwen")
    prompts = []

    def post(self, body):
        prompts.append(body["messages"][1]["content"][1]["text"])
        return {"choices": [{"message": {"content": content}, "finish_reason": finish}]}

    monkeypatch.setattr(QwenVisionClient, "_post", post)
    return client, prompts


def _image():
    return Image.new("RGB", (200, 100), "white")


@pytest.fixture
def compact(monkeypatch):
    monkeypatch.setattr(settings, "DRAWING_SHEET_COMPACT_JSON", True)
    monkeypatch.setattr(settings, "DRAWING_SHEET_TILES", False)


def test_both_shapes_parse_to_the_same_structure(monkeypatch, compact):
    client, _ = _client(monkeypatch, json.dumps(OBJECTS, ensure_ascii=False))
    from_objects = client.read_sheet(_image())
    client, _ = _client(monkeypatch, json.dumps(COMPACT, ensure_ascii=False))
    from_arrays = client.read_sheet(_image())
    assert from_arrays.sheet_type == from_objects.sheet_type
    assert from_arrays.lost == from_objects.lost == 0
    assert len(from_arrays.annotations) == len(from_objects.annotations)
    for mine, theirs in zip(from_arrays.annotations, from_objects.annotations):
        assert mine["category"] == theirs["category"]
        assert mine["value"] == theirs["value"]
        assert mine["bbox"] == pytest.approx(theirs["bbox"], abs=0.001)


def test_fields_downstream_are_equal(monkeypatch, compact):
    processor = DrawingProcessor.__new__(DrawingProcessor)
    client, _ = _client(monkeypatch, json.dumps(OBJECTS, ensure_ascii=False))
    from_objects = processor._fields_from_annotations(client.read_sheet(_image()).annotations)
    client, _ = _client(monkeypatch, json.dumps(COMPACT, ensure_ascii=False))
    from_arrays = processor._fields_from_annotations(client.read_sheet(_image()).annotations)
    assert json.dumps(from_arrays, ensure_ascii=False) == json.dumps(from_objects, ensure_ascii=False)


def test_truncated_array_keeps_whole_items(monkeypatch, compact):
    text = json.dumps(COMPACT, ensure_ascii=False)
    cut = text[: text.index('["p"') + 8]          # обрыв посреди четвёртой надписи
    client, _ = _client(monkeypatch, cut, finish="length")
    reading = client.read_sheet(_image())
    assert reading.cut
    assert [a["value"] for a in reading.annotations] == ["⌀44H7", "Ra 3,2", "ТАДУ 687000.001"]
    assert reading.sheet_type == "detail"


def test_unknown_code_goes_to_fallback(monkeypatch, compact):
    processor = DrawingProcessor.__new__(DrawingProcessor)
    as_object = {"annotations": [{"category": "zz", "value": "⌀20", "bbox": [0.1, 0.1, 0.2, 0.2]}]}
    as_array = {"annotations": [["zz", "⌀20", 100, 100, 200, 200]]}
    client, _ = _client(monkeypatch, json.dumps(as_object, ensure_ascii=False))
    expected = processor._fields_from_annotations(client.read_sheet(_image()).annotations)
    client, _ = _client(monkeypatch, json.dumps(as_array, ensure_ascii=False))
    got = processor._fields_from_annotations(client.read_sheet(_image()).annotations)
    assert got == expected
    assert got[0]["category"] == "size"


def test_lone_code_is_schema_word(compact):
    annotations, lost = qwen_client.as_annotations(["s", "⌀20", ["s", "20", 1, 2, 3, 4]], compact=True)
    assert [a["value"] for a in annotations] == ["⌀20", "20"]
    assert lost == {"слово схемы": 1}


def test_prompt_follows_the_flag(monkeypatch):
    monkeypatch.setattr(settings, "DRAWING_SHEET_TILES", False)
    monkeypatch.setattr(settings, "DRAWING_SHEET_COMPACT_JSON", False)
    client, prompts = _client(monkeypatch, json.dumps(OBJECTS, ensure_ascii=False))
    client.read_sheet(_image(), hint="штамп в правом нижнем углу")
    monkeypatch.setattr(settings, "DRAWING_SHEET_COMPACT_JSON", True)
    client.read_sheet(_image(), hint="штамп в правом нижнем углу")
    plain, short = prompts
    assert '"bbox"' in plain and '"bbox"' not in short
    assert "тысячные доли" in short
    assert short.endswith("Подсказка по разметке листа: штамп в правом нижнем углу\n")


def test_flag_off_arrays_are_read_as_before(monkeypatch):
    monkeypatch.setattr(settings, "DRAWING_SHEET_COMPACT_JSON", False)
    annotations, lost = qwen_client.as_annotations([["s", "⌀20", 100, 100, 200, 200]])
    # Как и раньше: строки массива — надписи без места, числа — потеря.
    assert [a["value"] for a in annotations] == ["s", "⌀20"]
    assert lost == {"int": 4}
