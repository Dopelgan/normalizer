"""
Один и тот же запрос к модели не задаётся дважды (VLM_SAME_REQUEST_ONCE, карточка T5).

Кэш ответов не покрывал два случая: отказ ограждённого прохода листа
(оборванный или зациклившийся ответ в кэш не пишется, и на каждом
переразборе модель снова тратит на него весь потолок токенов) и одинаковые
запросы, идущие одновременно. Модель поддельная: отвечает по сценарию и
считает вызовы.
"""

import json
import threading

import pytest
import requests

from core import telemetry
from core.config import settings
from core.providers import qwen_client
from core.providers.qwen_client import REJECT_LOOP, QwenVisionClient, sheet_answer_repeats

pytest.importorskip("PIL")
from PIL import Image  # noqa: E402


class Response:
    def __init__(self, payload, status=200):
        self.payload, self.status_code, self.text = payload, status, ""

    def json(self):
        return self.payload


def reply(content, finish="stop", usage=None):
    payload = {"choices": [{"message": {"content": content}, "finish_reason": finish}]}
    if usage is not None:
        payload["usage"] = usage
    return payload


def sheet(values, finish="stop"):
    items = [{"category": "size", "value": v, "bbox": [0.1, 0.1, 0.2, 0.2]} for v in values]
    return reply(json.dumps({"sheet_type": "detail", "annotations": items}, ensure_ascii=False), finish)


GOOD = ["⌀280±0,3", "⌀315", "18*", "8 отв."]
LOOP = ["⌀8"] * 12 + ["⌀10", "⌀12"]
EMPTY = reply(json.dumps({"sheet_type": "detail", "annotations": []}))


class Model:
    """Отвечает по очереди ответами `script`; последний повторяется. `gate` держит первый вызов."""

    def __init__(self, *script, gate=None):
        self.script = list(script)
        self.bodies = []
        self.gate = gate
        self.entered = threading.Event()
        self.lock = threading.Lock()

    def post(self, url, json=None, headers=None, timeout=None):
        with self.lock:
            self.bodies.append(json)
            answer = self.script.pop(0) if len(self.script) > 1 else self.script[0]
        self.entered.set()
        if self.gate is not None:
            self.gate.wait(5)
        if isinstance(answer, Exception):
            raise answer
        if isinstance(answer, int):
            return Response({}, status=answer)
        return Response(answer)


def client(model):
    return QwenVisionClient(endpoint="http://qwen:8000/v1", model="fake-vl", session=model)


def image(colour=(255, 255, 255)):
    return Image.new("RGB", (400, 300), colour)


@pytest.fixture(autouse=True)
def setup(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "QWEN_RETRIES", 0)
    monkeypatch.setattr(settings, "QWEN_TIMEOUT", 180)
    monkeypatch.setattr(settings, "DRAWING_SMALL_PRINT_CARE", False)
    monkeypatch.setattr(settings, "DRAWING_DIMENSION_FIDELITY", False)
    monkeypatch.setattr(settings, "DRAWING_SHEET_TILES", False)
    monkeypatch.setattr(settings, "DRAWING_SHEET_HIGH_TOKENS", True)
    monkeypatch.setattr(settings, "DRAWING_SHEET_COMPACT_JSON", False)
    monkeypatch.setattr(settings, "VLM_RESPONSE_CACHE", True)
    monkeypatch.setattr(settings, "VLM_RESPONSE_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(settings, "DRAWING_SHEET_GUARDED", True)
    monkeypatch.setattr(settings, "VLM_SAME_REQUEST_ONCE", True)


def values(reading):
    return [a["value"] for a in reading.annotations]


class TestRepeatsVerdict:
    def test_cut_and_loop_repeat_empty_and_good_do_not(self):
        cut = sheet(GOOD, finish="length")["choices"][0]["message"]["content"]
        assert sheet_answer_repeats(cut, "length")
        assert sheet_answer_repeats("{\"annotations\": [{\"val", "length")  # обрыв без единой надписи
        assert sheet_answer_repeats(json.dumps({"annotations": [{"value": v} for v in LOOP]}), "stop")
        assert not sheet_answer_repeats(json.dumps({"annotations": []}), "stop")
        assert not sheet_answer_repeats(json.dumps({"annotations": [{"value": v} for v in GOOD]}), "stop")


class TestRejectedAnswerIsNotAskedAgain:
    def test_looping_sheet_reparsed_without_calling_the_model(self, tmp_path):
        first = client(Model(sheet(LOOP), sheet(GOOD))).read_sheet(image())
        assert first.rejected == REJECT_LOOP and first.tiles == 4
        # Второй разбор того же листа: модель не нужна — отказ отложен, части в кэше.
        again_model = Model(500)
        with telemetry.collect() as timings:
            again = client(again_model).read_sheet(image())
        assert again_model.bodies == []
        assert again.rejected == REJECT_LOOP and again.tiles == 4
        assert values(again) == values(first)
        assert timings["parse.vlm_reused"] >= 1
        # Отложенный отказ не виден ни кэшу, ни отчёту по кэшу (*.json).
        assert len(list(tmp_path.rglob("*.rejected"))) == 1
        assert all(not p.name.endswith(".rejected.json") for p in tmp_path.rglob("*.json"))

    def test_flag_off_asks_the_model_again(self, monkeypatch):
        monkeypatch.setattr(settings, "VLM_SAME_REQUEST_ONCE", False)
        client(Model(sheet(LOOP), sheet(GOOD))).read_sheet(image())
        again_model = Model(sheet(LOOP), sheet(GOOD))
        client(again_model).read_sheet(image())
        assert len(again_model.bodies) == 1  # лист заново, части из кэша

    def test_without_cache_the_model_is_always_asked(self, monkeypatch):
        monkeypatch.setattr(settings, "VLM_RESPONSE_CACHE", False)
        client(Model(sheet(LOOP), sheet(GOOD))).read_sheet(image())
        again_model = Model(sheet(LOOP), sheet(GOOD))
        client(again_model).read_sheet(image())
        assert len(again_model.bodies) >= 2

    def test_above_zero_temperature_asks_again(self, monkeypatch):
        monkeypatch.setattr(settings, "QWEN_TEMPERATURE", 0.3)
        client(Model(sheet(LOOP), sheet(GOOD))).read_sheet(image())
        again_model = Model(sheet(GOOD))
        again = client(again_model).read_sheet(image())
        assert len(again_model.bodies) == 1 and values(again) == GOOD

    def test_empty_answer_is_asked_again(self, tmp_path):
        client(Model(EMPTY)).read_sheet(image())
        again_model = Model(sheet(GOOD))
        again = client(again_model).read_sheet(image())
        assert len(again_model.bodies) == 1 and values(again) == GOOD
        assert list(tmp_path.rglob("*.rejected")) == []

    def test_kept_answer_is_judged_by_current_code(self, monkeypatch):
        client(Model(sheet(LOOP), sheet(GOOD))).read_sheet(image())
        # Порог цикла подняли: отложенный ответ теперь принят, вызова нет.
        monkeypatch.setattr(qwen_client, "_LOOP_SHARE", 0.9)
        again_model = Model(500)
        again = client(again_model).read_sheet(image())
        assert again_model.bodies == [] and again.rejected == "" and again.tiles == 0

    def test_unguarded_pass_does_not_use_kept_answers(self, monkeypatch):
        monkeypatch.setattr(settings, "DRAWING_SHEET_SCHEMA", False)
        client(Model(sheet(LOOP), sheet(GOOD))).read_sheet(image())
        monkeypatch.setattr(settings, "DRAWING_SHEET_GUARDED", False)
        again_model = Model(sheet(GOOD))
        again = client(again_model).read_sheet(image())
        assert len(again_model.bodies) == 1 and values(again) == GOOD


def _read_in_threads(model, count=2):
    results = [None] * count

    def run(index):
        results[index] = client(model).read_sheet(image())

    threads = [threading.Thread(target=run, args=(n,)) for n in range(count)]
    threads[0].start()
    assert model.entered.wait(5)
    for thread in threads[1:]:
        thread.start()
    threading.Event().wait(0.3)  # второй поток дошёл до ожидания
    model.gate.set()
    for thread in threads:
        thread.join(5)
    return results


class TestSameRequestInFlight:
    def test_concurrent_identical_sheets_ask_the_model_once(self):
        model = Model(sheet(GOOD), gate=threading.Event())
        results = _read_in_threads(model)
        assert len(model.bodies) == 1
        assert [values(r) for r in results] == [GOOD, GOOD]

    def test_flag_off_asks_twice(self, monkeypatch):
        monkeypatch.setattr(settings, "VLM_SAME_REQUEST_ONCE", False)
        model = Model(sheet(GOOD), gate=threading.Event())
        _read_in_threads(model)
        assert len(model.bodies) == 2

    def test_failure_of_the_first_is_not_inherited(self):
        model = Model(requests.ConnectionError("обрыв"), sheet(GOOD), gate=threading.Event())
        results = _read_in_threads(model)
        assert len(model.bodies) == 2
        assert results[0] is None and values(results[1]) == GOOD

    def test_different_sheets_do_not_wait_for_each_other(self):
        model = Model(sheet(GOOD))
        client(model).read_sheet(image())
        client(model).read_sheet(image((0, 0, 0)))
        assert len(model.bodies) == 2
        assert qwen_client._flights == {}


class TestTokens:
    def test_tokens_of_a_call_are_counted(self):
        model = Model(reply("{}", usage={"prompt_tokens": 4400, "completion_tokens": 2083}))
        with telemetry.collect() as timings:
            client(model)._post({})
        assert timings["parse.vlm_prompt_tokens"] == 4400
        assert timings["parse.vlm_completion_tokens"] == 2083

    def test_answer_without_usage_counts_nothing(self):
        with telemetry.collect() as timings:
            client(Model(reply("{}")))._post({})
        assert "parse.vlm_completion_tokens" not in timings
