"""
Устойчивость проходов чертежа (аудит 25.11): N-01 знаки в надписях, N-02
проверка ответа перед кэшем, N-03 неудача листа не теряет штамп, N-05
повторы таймаута и срок вызова, N-07 калиброванное подтверждение, N-13
спасение оборванного JSON, N-26 рассуждение выключено у всех проходов,
N-27 один вид ответа штампа. Каждый флаг: без него — прежнее поведение.

Модель поддельная, значения придуманы.
"""

import json
import os
import time
from concurrent.futures import Future

import pytest
import requests
from celery.exceptions import SoftTimeLimitExceeded

from core import deadline
from core.config import settings
from core.ladder import result_cache
from core.models.contract import Content, DrawingField, Fragment, Position, Provenance
from core.providers import drawing_text, drawing_vision, qwen_client
from core.providers.qwen_client import QwenVisionClient

pytest.importorskip("numpy")
Image = pytest.importorskip("PIL.Image")


# ------------------------------------------------------------ поддельная модель
class Response:
    def __init__(self, payload, status=200):
        self.payload, self.status_code, self.text = payload, status, ""

    def json(self):
        return self.payload


def reply(content, finish="stop"):
    return {"choices": [{"message": {"content": content}, "finish_reason": finish}]}


class Model:
    """Отвечает по очереди; последний ответ повторяется. Запоминает тела и сроки."""

    def __init__(self, *script):
        self.script = list(script)
        self.bodies, self.timeouts = [], []

    def post(self, url, json=None, headers=None, timeout=None):
        self.bodies.append(json)
        self.timeouts.append(timeout)
        answer = self.script.pop(0) if len(self.script) > 1 else self.script[0]
        if isinstance(answer, Exception):
            raise answer
        return Response(answer)


def client(model):
    return QwenVisionClient(endpoint="http://qwen.test/v1", model="fake-vl", session=model)


def picture():
    return Image.new("RGB", (300, 200), "white")


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "VLM_RESPONSE_CACHE", True)
    monkeypatch.setattr(settings, "VLM_RESPONSE_CACHE_DIR", str(tmp_path / "vlm"))
    monkeypatch.setattr(settings, "QWEN_RETRIES", 0)
    return tmp_path / "vlm"


def cached_files(path):
    return sorted(path.rglob("*.json")) if path.exists() else []


# ------------------------------------------------------------------- N-01
SYMBOLIC = ["⌀ 5", "⌀5*", "(5)", "±2°", "(+0,2)", "√Ra 6,3 (√)", "+0,5 / -0,2", "□ 8"]
NOISE = ["———", "(√)", "......1", "------2", "|||"]


def test_symbol_inscriptions_dropped_without_flag(monkeypatch):
    monkeypatch.setattr(settings, "DRAWING_TEXT_SYMBOLS", False)
    for text in ["⌀ 5", "⌀5*", "(5)", "±2°", "(+0,2)", "√Ra 6,3 (√)"]:
        assert drawing_text.looks_like_text(text)[0] is False


def test_symbol_inscriptions_kept_with_flag(monkeypatch):
    monkeypatch.setattr(settings, "DRAWING_TEXT_SYMBOLS", True)
    for text in SYMBOLIC:
        assert drawing_text.looks_like_text(text) == (True, ""), text


def test_noise_still_rejected_with_flag(monkeypatch):
    monkeypatch.setattr(settings, "DRAWING_TEXT_SYMBOLS", True)
    for text in NOISE:
        assert drawing_text.looks_like_text(text)[0] is False, text


# ------------------------------------------------------------------- N-02
def test_namespace_empty_keeps_old_key_and_set_changes_it(monkeypatch):
    body = {"model": "m", "messages": []}
    monkeypatch.setattr(settings, "VLM_RESPONSE_CACHE_NAMESPACE", "")
    old = qwen_client.cache_key("http://q/v1", body, "img")
    monkeypatch.setattr(settings, "VLM_RESPONSE_CACHE_NAMESPACE", "reprocess-2")
    assert qwen_client.cache_key("http://q/v1", body, "img") != old


def test_ttl_expires_old_entries_without_deleting(cache, monkeypatch):
    qwen_client.cache_put("ab" + "0" * 62, ('{"a": 1}', "stop"))
    path = cached_files(cache)[0]
    old = time.time() - 3 * 86400
    os.utime(path, (old, old))
    monkeypatch.setattr(settings, "VLM_RESPONSE_CACHE_TTL_DAYS", 0)
    assert qwen_client.cache_get("ab" + "0" * 62) is not None
    monkeypatch.setattr(settings, "VLM_RESPONSE_CACHE_TTL_DAYS", 1)
    assert qwen_client.cache_get("ab" + "0" * 62) is None
    assert path.exists()


CUT_STAMP = '{"fields": {"Обозначение": "АБВГ.123456.001", "Масса": "2,5"}, "rows": [["Обозн'


@pytest.mark.parametrize("validate,stored", [(False, 1), (True, 0)])
def test_cut_answer_is_cached_only_without_validation(cache, monkeypatch, validate, stored):
    monkeypatch.setattr(settings, "VLM_CACHE_VALIDATE", validate)
    client(Model(reply(CUT_STAMP, "length"))).read_title_block(picture())
    assert len(cached_files(cache)) == stored


@pytest.mark.parametrize("validate,stored", [(False, 1), (True, 0)])
def test_answer_without_required_key_is_not_cached(cache, monkeypatch, validate, stored):
    monkeypatch.setattr(settings, "VLM_CACHE_VALIDATE", validate)
    client(Model(reply('{"title": "Спецификация"}'))).read_table(picture())
    assert len(cached_files(cache)) == stored


def test_frozen_empty_answer_is_asked_again_with_validation(cache, monkeypatch):
    monkeypatch.setattr(settings, "VLM_CACHE_VALIDATE", False)
    client(Model(reply("{}"))).read_title_block(picture())
    assert len(cached_files(cache)) == 1
    monkeypatch.setattr(settings, "VLM_CACHE_VALIDATE", True)
    good = Model(reply('{"fields": {"Обозначение": "АБВГ.1"}, "rows": []}'))
    block = client(good).read_title_block(picture())
    assert len(good.bodies) == 1
    assert block["fields"] == {"Обозначение": "АБВГ.1"}


def sheet_answer(count):
    items = [{"category": "size", "value": f"⌀{10 + n}", "bbox": [0.1, 0.1, 0.2, 0.2]} for n in range(count)]
    return reply(json.dumps({"sheet_type": "detail", "annotations": items}, ensure_ascii=False))


@pytest.mark.parametrize("count,stored", [(3, 0), (20, 1)])
def test_poor_sheet_answer_is_not_cached_against_text_layer(cache, monkeypatch, count, stored):
    monkeypatch.setattr(settings, "VLM_CACHE_VALIDATE", True)
    token = qwen_client.set_expected_text_lines(40)
    try:
        reading = client(Model(sheet_answer(count)))._read_sheet_piece(picture(), "лист", whole_sheet=True)
    finally:
        qwen_client.reset_expected_text_lines(token)
    assert len(reading.annotations) == count     # разбору ответ отдаётся как есть
    assert len(cached_files(cache)) == stored


def test_poor_sheet_answer_cached_without_validation(cache, monkeypatch):
    monkeypatch.setattr(settings, "VLM_CACHE_VALIDATE", False)
    token = qwen_client.set_expected_text_lines(40)
    try:
        client(Model(sheet_answer(3)))._read_sheet_piece(picture(), "лист", whole_sheet=True)
    finally:
        qwen_client.reset_expected_text_lines(token)
    assert len(cached_files(cache)) == 1


# ------------------------------------------------------------------- N-13
CUT_TABLE = ('{"title": "Спецификация", "headers": ["Поз.", "Кол."], '
             '"rows": [["1", "2"], ["2", "4"], ["3", "')


def test_cut_table_lost_without_salvage(monkeypatch):
    monkeypatch.setattr(settings, "DRAWING_ASK_SALVAGE", False)
    monkeypatch.setattr(settings, "QWEN_RETRIES", 0)
    assert client(Model(reply(CUT_TABLE, "length"))).read_table(picture()) is None


def test_cut_table_rows_salvaged(monkeypatch):
    monkeypatch.setattr(settings, "DRAWING_ASK_SALVAGE", True)
    monkeypatch.setattr(settings, "QWEN_RETRIES", 0)
    table = client(Model(reply(CUT_TABLE, "length"))).read_table(picture())
    assert table == {"title": "Спецификация", "headers": ["Поз.", "Кол."],
                     "rows": [["1", "2"], ["2", "4"]], "cut": True}


def test_cut_stamp_inner_object_not_taken(monkeypatch):
    monkeypatch.setattr(settings, "QWEN_RETRIES", 0)
    monkeypatch.setattr(settings, "DRAWING_ASK_SALVAGE", False)
    before = client(Model(reply(CUT_STAMP, "length"))).read_title_block(picture())
    assert before["fields"] == {}                 # вложенный объект принят за ответ
    monkeypatch.setattr(settings, "DRAWING_ASK_SALVAGE", True)
    after = client(Model(reply(CUT_STAMP, "length"))).read_title_block(picture())
    assert after["fields"] == {"Обозначение": "АБВГ.123456.001", "Масса": "2,5"}
    assert after["cut"] is True


def test_cut_transcript_lines_salvaged(monkeypatch):
    monkeypatch.setattr(settings, "QWEN_RETRIES", 0)
    cut = '{"lines": ["Сталь 20", "Ra 3,2", "Неоконч'
    monkeypatch.setattr(settings, "DRAWING_ASK_SALVAGE", False)
    assert client(Model(reply(cut, "length"))).transcribe(picture()) is None
    monkeypatch.setattr(settings, "DRAWING_ASK_SALVAGE", True)
    assert client(Model(reply(cut, "length"))).transcribe(picture()) == ["Сталь 20", "Ra 3,2"]


# ------------------------------------------------------------------- N-05
@pytest.mark.parametrize("robust,calls", [(False, 3), (True, 2)])
def test_read_timeout_retried_at_most_once(monkeypatch, robust, calls):
    monkeypatch.setattr(settings, "DRAWING_ROBUST_PASSES", robust)
    monkeypatch.setattr(settings, "QWEN_RETRIES", 2)
    monkeypatch.setattr(qwen_client.time, "sleep", lambda _s: None)
    model = Model(requests.ReadTimeout("долго"))
    assert client(model).read_title_block(picture()) is None
    assert len(model.bodies) == calls


def test_call_timeout_clamped_to_task_deadline_and_sheet_budget(monkeypatch):
    monkeypatch.setattr(settings, "QWEN_RETRIES", 0)
    ok = reply('{"fields": {}, "rows": []}')
    monkeypatch.setattr(settings, "DRAWING_ROBUST_PASSES", False)
    model = Model(ok)
    with deadline.scope(50):
        client(model).read_title_block(picture())
    assert model.timeouts == [settings.QWEN_TIMEOUT]

    monkeypatch.setattr(settings, "DRAWING_ROBUST_PASSES", True)
    model = Model(ok)
    with deadline.scope(50):
        client(model).read_title_block(picture())
    assert 40 < model.timeouts[0] <= 50

    model = Model(ok)
    token = qwen_client.set_call_until(time.monotonic() + 30)
    try:
        client(model).read_title_block(picture())
    finally:
        qwen_client.reset_call_until(token)
    assert 20 < model.timeouts[0] <= 30
    # Не короче нижней границы, даже когда бюджет почти исчерпан.
    token = qwen_client.set_call_until(time.monotonic() + 1)
    try:
        model = Model(ok)
        client(model).read_title_block(picture())
    finally:
        qwen_client.reset_call_until(token)
    assert model.timeouts[0] == settings.DRAWING_ROBUST_MIN_CALL_TIMEOUT


def _soft_limit_run(temp_storage, monkeypatch, robust):
    from core import drawing_processor as dp

    monkeypatch.setattr(settings, "DRAWING_ROBUST_PASSES", robust)
    monkeypatch.setattr(settings, "VLM_SHEET_CONCURRENCY", 1)
    processor = dp.DrawingProcessor(temp_storage, qwen=QwenVisionClient(endpoint="http://vlm.test/v1"))

    def process(uri, metadata, page, transcribe=False, page_fields=None):
        if page == 2:
            raise SoftTimeLimitExceeded()
        return f"r{page}"

    monkeypatch.setattr(processor, "process", process)
    result = processor._read_sheets("src/scan.pdf", {}, [1, 2, 3], False, [None] * 3)
    return ["NO_TIME" if r is dp._NO_TIME else r for r in result]


def test_soft_limit_loses_all_sheets_without_flag(temp_storage, monkeypatch):
    with pytest.raises(SoftTimeLimitExceeded):
        _soft_limit_run(temp_storage, monkeypatch, False)


def test_soft_limit_keeps_sheets_already_read(temp_storage, monkeypatch):
    assert _soft_limit_run(temp_storage, monkeypatch, True) == ["r1", "NO_TIME", "NO_TIME"]


def test_unfinished_parallel_sheet_counts_as_no_time():
    from core import drawing_processor as dp

    done, failed, running = Future(), Future(), Future()
    done.set_result("r1")
    failed.set_exception(RuntimeError("сбой"))
    assert dp._finished(done) == "r1"
    assert dp._finished(failed) is dp._NO_TIME
    assert dp._finished(running) is dp._NO_TIME


# ------------------------------------------------------------------- N-03
SHEET_URI = "assets/doc-1/sheet.png"
TITLE = drawing_vision.Region(kind=drawing_vision.KIND_TITLE_BLOCK, bbox=[0.6, 0.8, 1.0, 1.0])


class SheetFails:
    """Лист не прочитан (таймаут), штамп читается."""

    available = True
    model = "fake-vl"

    def __init__(self):
        self.asked = []

    def read_sheet(self, image, hint="", until=None):
        self.asked.append("sheet")
        return None

    def read_title_block(self, image):
        self.asked.append("title_block")
        return {"fields": {"Обозначение": "АБВГ.123456.001"}, "rows": []}


@pytest.fixture
def stamp_sheet(temp_storage, monkeypatch):
    from tests.fixtures import synthetic_sheet

    temp_storage.write_file(SHEET_URI, synthetic_sheet.png_bytes())
    fixed = drawing_vision.SheetLayout(image=Image.new("RGB", (800, 600), "white"),
                                       frame=[0.0, 0.0, 1.0, 1.0], regions=[TITLE])
    monkeypatch.setattr(drawing_vision, "analyse", lambda _image: fixed)
    monkeypatch.setattr(settings, "DRAWING_TITLE_BLOCK_PASS", False)
    monkeypatch.setattr(settings, "DRAWING_PASSES_PARALLEL", 1)
    return temp_storage


def test_failed_sheet_drops_stamp_without_flag(stamp_sheet, monkeypatch):
    from core.drawing_processor import MODE_NONE, DrawingProcessor

    monkeypatch.setattr(settings, "DRAWING_ROBUST_PASSES", False)
    qwen = SheetFails()
    result = DrawingProcessor(stamp_sheet, qwen=qwen).process(SHEET_URI, {})
    assert result["mode"] == MODE_NONE
    assert qwen.asked == ["sheet"]


def test_failed_sheet_keeps_stamp_with_flag(stamp_sheet, monkeypatch):
    from core.drawing_processor import MODE_NONE, DrawingProcessor

    monkeypatch.setattr(settings, "DRAWING_ROBUST_PASSES", True)
    qwen = SheetFails()
    result = DrawingProcessor(stamp_sheet, qwen=qwen).process(SHEET_URI, {})
    assert result["mode"] != MODE_NONE
    assert qwen.asked == ["sheet", "title_block"]
    assert any("АБВГ.123456.001" in field["value"] for field in result["parsed_fields"])
    assert {"kind": "sheet", "bbox": [0.0, 0.0, 1.0, 1.0], "reason": "sheet_not_read"} \
        in result["missing_regions"]


# ------------------------------------------------------------------- N-07
def fragment(text, **field):
    values = {"category": "size", "value": "⌀44H7", "confidence": 0.9, **field}
    return Fragment(
        fragment_id="f", type="drawing",
        content=Content(text=text, structured_drawing_fields=[DrawingField(**values)],
                        structured_payload={}),
        position=Position(page=1), confidence=0.9, completeness=None,
        provenance=Provenance(method="vision_plus_vlm", strategy_level=7, source="drawing"),
    )


def confirmed(item):
    from core.drawing_quality import apply_drawing_quality

    apply_drawing_quality(item)
    return item.content.structured_drawing_fields[0].confirmed


def test_vlm_maximum_never_confirmed_without_flag(monkeypatch):
    monkeypatch.setattr(settings, "DRAWING_CONFIRMED_CALIBRATED", False)
    assert confirmed(fragment("Отверстие ⌀44H7 сквозное")) is False


def test_threshold_is_a_setting(monkeypatch):
    monkeypatch.setattr(settings, "DRAWING_CONFIRMED_CALIBRATED", False)
    monkeypatch.setattr(settings, "DRAWING_CONFIRMED_MIN_CONFIDENCE", 0.85)
    assert confirmed(fragment("")) is True


@pytest.mark.parametrize("text,field,expected", [
    ("Отверстие ⌀ 44H7 сквозное", {}, True),                     # переписанный текст
    ("Сталь 20", {}, False),                                      # второго источника нет
    ("", {"agreement": "agreed"}, True),                          # согласие проходов
    ("", {"provenance": "vector_text_layer"}, True),              # текстовый слой
    ("⌀44H7", {"provenance": "full_page_ocr"}, False),            # источник тот же
    ("⌀44H7", {"confidence": 0.75}, False),                       # ниже максимума модели
    ("⌀44H7", {"value": "⌀44H7 (dev)"}, False),                   # выдумка
])
def test_calibrated_confirmation(monkeypatch, text, field, expected):
    monkeypatch.setattr(settings, "DRAWING_CONFIRMED_CALIBRATED", True)
    assert confirmed(fragment(text, **field)) is expected


def test_calibrated_confirmation_survives_composed_record(monkeypatch):
    monkeypatch.setattr(settings, "DRAWING_CONFIRMED_CALIBRATED", True)
    item = fragment("Отверстие ⌀44H7")
    assert confirmed(item) is True
    item.content.text = "Размеры: ⌀44H7"
    item.content.structured_payload["record_composed"] = True
    assert confirmed(item) is True


# ------------------------------------------------------------- N-26, N-27
PASSES = ["read_title_block", "read_title_block_fields", "read_side_column", "read_table",
          "transcribe", "read_sheet_content"]


@pytest.mark.parametrize("method", PASSES)
def test_thinking_disabled_on_every_pass_when_stable(monkeypatch, method):
    monkeypatch.setattr(settings, "QWEN_RETRIES", 0)
    monkeypatch.setattr(settings, "DRAWING_VLM_STABLE", True)
    model = Model(reply('{"fields": {"a": "1"}, "rows": [], "lines": [], "annotations": []}'))
    getattr(client(model), method)(picture())
    assert model.bodies and all(
        body["chat_template_kwargs"] == {"enable_thinking": False} for body in model.bodies)


def test_sheet_and_short_passes_disable_thinking(monkeypatch):
    monkeypatch.setattr(settings, "QWEN_RETRIES", 0)
    monkeypatch.setattr(settings, "DRAWING_VLM_STABLE", True)
    model = Model(sheet_answer(2))
    vlm = client(model)
    vlm._read_sheet_piece(picture(), "лист")
    vlm.read_corner_roughness(picture())
    vlm.read_values(picture(), 2)
    assert len(model.bodies) == 3
    assert all("chat_template_kwargs" in body for body in model.bodies)


@pytest.mark.parametrize("method", ["read_title_block", "read_title_block_fields"])
def test_single_form_stamp_prompt_drops_rows(monkeypatch, method):
    monkeypatch.setattr(settings, "QWEN_RETRIES", 0)

    def prompt(single):
        monkeypatch.setattr(settings, "DRAWING_STAMP_SINGLE_FORM", single)
        model = Model(reply('{"fields": {"Обозначение": "АБВГ.1"}}'))
        getattr(client(model), method)(picture())
        return model.bodies[0]["messages"][1]["content"][1]["text"]

    assert '"rows"' in prompt(False)
    single = prompt(True)
    assert '"rows"' not in single and "rows —" not in single
    assert '"fields"' in single


# --------------------------------------------------------- ключ кэша лестницы
@pytest.mark.parametrize("flag", ["DRAWING_ROBUST_PASSES", "DRAWING_ASK_SALVAGE",
                                  "DRAWING_TEXT_SYMBOLS", "DRAWING_STAMP_SINGLE_FORM"])
def test_flag_off_keeps_ladder_key_and_on_changes_it(monkeypatch, flag):
    monkeypatch.setattr(settings, flag, False)
    off = result_cache.settings_snapshot()
    assert flag not in off
    monkeypatch.setattr(settings, flag, True)
    assert result_cache.settings_snapshot()[flag] is True
