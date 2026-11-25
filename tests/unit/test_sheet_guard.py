"""
Ограждённый проход «лист целиком» (DRAWING_SHEET_GUARDED, карточка D2).

Прежде любой ответ модели с текстом шёл в кэш: пустой ответ, ответ,
зациклившийся на одной надписи, и ответ, оборванный потолком токенов,
замораживались, и повторный разбор отдавал тот же сбой навсегда. Модель
здесь поддельная: отвечает по сценарию и запоминает тела запросов и сроки.
"""

import json
from pathlib import Path

import pytest
import requests

from core.config import settings
from core.providers import qwen_client
from core.providers.qwen_client import (
    REJECT_CUT, REJECT_LOOP, QwenVisionClient, sheet_answer_accepted, sheet_verdict,
)

pytest.importorskip("PIL")
from PIL import Image  # noqa: E402


class Response:
    def __init__(self, payload, status=200):
        self.payload, self.status_code, self.text = payload, status, ""

    def json(self):
        return self.payload


def reply(content, finish="stop"):
    return {"choices": [{"message": {"content": content}, "finish_reason": finish}]}


def sheet(values, finish="stop"):
    items = [{"category": "size", "value": v, "bbox": [0.1, 0.1, 0.2, 0.2]} for v in values]
    return reply(json.dumps({"sheet_type": "detail", "annotations": items}, ensure_ascii=False), finish)


GOOD = ["⌀280±0,3", "⌀315", "18*", "8 отв."]
LOOP = ["⌀8"] * 12 + ["⌀10", "⌀12"]


class Model:
    """Отвечает по очереди ответами `script`; последний повторяется."""

    def __init__(self, *script):
        self.script = list(script)
        self.bodies, self.timeouts = [], []

    def post(self, url, json=None, headers=None, timeout=None):
        self.bodies.append(json)
        self.timeouts.append(timeout)
        answer = self.script.pop(0) if len(self.script) > 1 else self.script[0]
        if isinstance(answer, Exception):
            raise answer
        if isinstance(answer, int):
            return Response({}, status=answer)
        return Response(answer)


def client(model):
    return QwenVisionClient(endpoint="http://qwen:8000/v1", model="fake-vl", session=model)


def image():
    return Image.new("RGB", (400, 300), (255, 255, 255))


def varied_image():
    """Лист, у которого каждая часть своя: одинаковые части делят ключ кэша."""
    img = Image.new("RGB", (400, 300))
    img.putdata([(x * 255 // 400, y * 255 // 300, (x + y) % 256)
                 for y in range(300) for x in range(400)])
    return img


def cached_files(path: Path):
    return sorted(path.rglob("*.json"))


@pytest.fixture(autouse=True)
def setup(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "QWEN_RETRIES", 2)
    monkeypatch.setattr(settings, "QWEN_TIMEOUT", 180)
    monkeypatch.setattr(settings, "DRAWING_SMALL_PRINT_CARE", False)
    monkeypatch.setattr(settings, "DRAWING_DIMENSION_FIDELITY", False)
    monkeypatch.setattr(settings, "DRAWING_SHEET_TILES", False)
    monkeypatch.setattr(settings, "DRAWING_SHEET_HIGH_TOKENS", True)
    monkeypatch.setattr(settings, "DRAWING_SHEET_MAX_TOKENS", 12288)
    monkeypatch.setattr(settings, "VLM_RESPONSE_CACHE", True)
    monkeypatch.setattr(settings, "VLM_RESPONSE_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(qwen_client.time, "sleep", lambda seconds: None)


@pytest.fixture
def guarded(monkeypatch):
    monkeypatch.setattr(settings, "DRAWING_SHEET_GUARDED", True)
    monkeypatch.setattr(settings, "DRAWING_SHEET_SCHEMA", True)


class TestVerdict:
    def test_cut_answer_is_rejected(self):
        assert sheet_verdict("length", [{"value": "⌀8"}]) == REJECT_CUT

    def test_looping_answer_is_rejected(self):
        assert sheet_verdict("stop", [{"value": v} for v in LOOP]) == REJECT_LOOP

    def test_repeated_size_on_a_real_sheet_is_not_a_loop(self):
        values = ["⌀8"] * 4 + [f"{n}" for n in range(20)]
        assert sheet_verdict("stop", [{"value": v} for v in values]) is None

    def test_four_holes_on_a_small_sheet_are_not_a_loop(self):
        values = ["⌀6"] * 4 + [str(n) for n in range(8)]
        assert sheet_verdict("stop", [{"value": v} for v in values]) is None

    def test_loop_keeps_the_same_text_in_other_places(self):
        same = {"category": "size", "value": "⌀8", "bbox": [0.1, 0.1, 0.2, 0.2]}
        elsewhere = {"category": "size", "value": "⌀8", "bbox": [0.7, 0.7, 0.8, 0.8]}
        kept = qwen_client._first_of_each([same] * 10 + [elsewhere, {"value": "⌀10"}])
        assert kept == [same, elsewhere, {"value": "⌀10"}]

    def test_short_list_is_never_a_loop(self):
        assert sheet_verdict("stop", [{"value": "⌀8"}] * 9) is None

    def test_empty_answer_is_not_cached(self):
        assert not sheet_answer_accepted(json.dumps({"annotations": []}), "stop")

    def test_compact_items_are_read_too(self):
        assert sheet_verdict("stop", [["s", "⌀8", 1, 2, 3, 4]] * 12) == REJECT_LOOP


class TestFlagOff:
    def test_body_and_cache_as_before(self, tmp_path):
        model = Model(sheet(LOOP, finish="length"))
        reading = client(model).read_sheet(image())
        assert model.bodies[0]["response_format"] == {"type": "json_object"}
        assert model.timeouts == [180]
        assert reading.rejected == ""
        # Без флага кэшируется и сбой: поведение прода.
        assert len(cached_files(tmp_path)) == 1


class TestGuarded:
    def test_good_answer_is_cached_as_today(self, guarded, tmp_path):
        model = Model(sheet(GOOD))
        first = client(model).read_sheet(image())
        again = client(Model(500)).read_sheet(image())
        assert [a["value"] for a in first.annotations] == GOOD
        assert [a["value"] for a in again.annotations] == GOOD
        assert len(model.bodies) == 1 and len(cached_files(tmp_path)) == 1

    def test_looping_answer_is_rejected_not_cached_and_read_by_tiles(self, guarded, tmp_path):
        model = Model(sheet(LOOP), sheet(GOOD))
        # Части идут параллельно: на одинаковых белых частях ответ одной
        # успевал лечь в кэш раньше, чем вторая сходила в модель (4 вызова из 5).
        reading = client(model).read_sheet(varied_image())
        assert reading.rejected == REJECT_LOOP
        # Лист и четыре части; лист прочитан частями, хотя DRAWING_SHEET_TILES выключен.
        assert len(model.bodies) == 5 and reading.tiles == 4
        values = [a["value"] for a in reading.annotations]
        assert values.count("⌀8") <= 5          # хвост цикла не дошёл до полей
        assert "⌀280±0,3" in values
        # В кэше только ответы частей, ответа по листу нет.
        assert len(cached_files(tmp_path)) == 4  # листа нет, по файлу на каждую из четырёх частей

    def test_cut_answer_is_rejected_and_read_by_tiles(self, guarded, tmp_path):
        model = Model(sheet(GOOD, finish="length"), sheet(GOOD))
        reading = client(model).read_sheet(image())
        assert reading.rejected == REJECT_CUT and reading.tiles == 4
        assert reading.complete

    def test_second_rejection_leaves_the_sheet_unread(self, guarded):
        model = Model(sheet(LOOP))
        reading = client(model).read_sheet(image())
        assert reading.rejected == REJECT_LOOP
        assert len(reading.failed) == 4 and not reading.complete

    def test_frozen_failure_from_old_cache_is_not_taken(self, guarded, tmp_path, monkeypatch):
        monkeypatch.setattr(settings, "DRAWING_SHEET_SCHEMA", False)
        monkeypatch.setattr(settings, "DRAWING_SHEET_GUARDED", False)
        client(Model(reply(json.dumps({"annotations": []})))).read_sheet(image())
        assert len(cached_files(tmp_path)) == 1
        monkeypatch.setattr(settings, "DRAWING_SHEET_GUARDED", True)
        model = Model(sheet(GOOD))
        reading = client(model).read_sheet(image())
        assert len(model.bodies) == 1
        assert [a["value"] for a in reading.annotations] == GOOD

    def test_timeout_from_token_ceiling(self, guarded):
        model = Model(sheet(GOOD))
        client(model).read_sheet(image())
        assert model.timeouts == [12288 // 25 + 1]

    def test_timeout_never_below_qwen_timeout(self, guarded, monkeypatch):
        monkeypatch.setattr(settings, "DRAWING_SHEET_HIGH_TOKENS", False)
        monkeypatch.setattr(settings, "QWEN_MAX_TOKENS", 2048)
        model = Model(sheet(GOOD))
        client(model).read_sheet(image())
        assert model.timeouts == [180]

    def test_read_timeout_retried_once(self, guarded):
        model = Model(requests.ReadTimeout("долго"))
        assert client(model).read_sheet(image()) is None
        assert len(model.bodies) == 2

    def test_schema_limits_categories_items_and_length(self, guarded):
        model = Model(sheet(GOOD))
        client(model).read_sheet(image())
        schema = model.bodies[0]["response_format"]["json_schema"]["schema"]
        items = schema["properties"]["annotations"]
        assert items["maxItems"] == settings.DRAWING_SHEET_MAX_ITEMS
        assert items["items"]["properties"]["value"]["maxLength"] == settings.DRAWING_SHEET_VALUE_MAX_CHARS
        assert set(items["items"]["properties"]["category"]["enum"]) == set(qwen_client.CATEGORIES)

    def test_schema_refused_by_server_falls_back_to_free_json(self, guarded):
        model = Model(400, sheet(GOOD))
        reading = client(model).read_sheet(image())
        assert [a["value"] for a in reading.annotations] == GOOD
        assert model.bodies[1]["response_format"] == {"type": "json_object"}

    def test_other_passes_unchanged(self, guarded):
        model = Model(reply(json.dumps({"lines": ["Масса 3,74"]})))
        assert client(model).transcribe(image()) == ["Масса 3,74"]
        assert model.bodies[0]["response_format"] == {"type": "json_object"}
        assert model.timeouts == [180]


class TestEmptySheet:
    """Пустой ответ при непустом листе — перечитывание частями (drawing_processor)."""

    def _process(self, monkeypatch, transcript_lines):
        from core.drawing_processor import DrawingProcessor

        monkeypatch.setattr(settings, "DRAWING_SHEET_GUARDED", True)
        monkeypatch.setattr(settings, "DRAWING_VISION_ENABLED", False)
        empty = reply(json.dumps({"sheet_type": "detail", "annotations": []}))
        lines = reply(json.dumps({"lines": transcript_lines}))

        class Routed(Model):
            def post(self, url, json=None, headers=None, timeout=None):
                self.bodies.append(json)
                prompt = json["messages"][1]["content"][1]["text"]
                if '"lines"' in prompt:
                    return Response(lines)
                return Response(empty if len(self.bodies) == 1 else sheet(GOOD))

        model = Routed(empty)
        processor = DrawingProcessor(storage=None, qwen=client(model))
        processor._open = lambda image_ref, page=1: image()
        result = processor.process("sheet.png", {}, transcribe=True)
        return result, model

    def test_empty_answer_with_text_on_the_sheet_is_read_by_tiles(self, monkeypatch):
        result, _ = self._process(monkeypatch, [f"строка {n}" for n in range(6)])
        values = [field["value"] for field in result["parsed_fields"]]
        assert "⌀280±0,3" in values
        assert "sheet_guard" not in result["payload"]

    def test_empty_answer_on_an_empty_sheet_is_kept(self, monkeypatch):
        result, model = self._process(monkeypatch, ["Лист 1"])
        assert result["parsed_fields"] == []
        # Пустой лист — не сбой: частями не читается.
        assert sum('"lines"' not in b["messages"][1]["content"][1]["text"] for b in model.bodies) == 1


class TestWhitespaceCut:
    """
    Опыт D1 на проде: под схемой модель выдала весь потолок пробелами — ни
    одной надписи, finish=length. Прежде такой лист терялся целиком (None),
    хотя обрыв под оградой дочитывается частями.
    """

    def test_cut_answer_without_items_is_read_by_tiles(self, guarded):
        blank = reply("{\n" + " \n" * 50 + "\"sheet_type\": \"scheme\", \"annotations\": [ {", finish="length")
        model = Model(blank, sheet(GOOD))
        reading = client(model).read_sheet(image())
        assert reading is not None
        assert reading.rejected == REJECT_CUT and reading.tiles == 4
        assert [a["value"] for a in reading.annotations][:len(GOOD)] == GOOD

    def test_flag_off_keeps_the_old_answer(self):
        blank = reply("{\n" + " \n" * 50, finish="length")
        assert client(Model(blank)).read_sheet(image()) is None


class TestTableGuard:
    """Пилот 17.11: проход по таблице вернул одну шапку при 16 строках у MinerU."""

    @staticmethod
    def table(rows):
        return reply(json.dumps({"title": "", "headers": ["Цвет"], "rows": [[f"Rt{n}"] for n in range(rows)]}))

    def test_short_table_is_not_cached_and_asked_again(self, guarded, tmp_path):
        model = Model(self.table(1), self.table(16))
        table = client(model).read_table(image(), expect_rows=16)
        assert len(table["rows"]) == 16 and len(model.bodies) == 2
        assert len(cached_files(tmp_path)) == 1

    def test_second_short_answer_keeps_the_longer_one(self, guarded):
        model = Model(self.table(2), self.table(1))
        assert len(client(model).read_table(image(), expect_rows=16)["rows"]) == 2

    def test_without_guard_one_call_as_before(self, tmp_path):
        model = Model(self.table(1))
        assert len(client(model).read_table(image(), expect_rows=16)["rows"]) == 1
        assert len(model.bodies) == 1 and len(cached_files(tmp_path)) == 1
