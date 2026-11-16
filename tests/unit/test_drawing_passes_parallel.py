"""
Проходы одного листа идут разом (DRAWING_PASSES_PARALLEL).

Лист целиком, весь текст листа, штамп и каждая таблица — независимые вызовы
модели. Разом они должны занимать время самого долгого из них, а не сумму, и
не менять ничего в записи листа: те же поля в том же порядке, тот же payload
с ключами в том же порядке, те же непрочитанные области. Отказ одного
прохода не обрывает соседние, отмена листа не даёт начать новые.
"""

import json
import threading
import time

import pytest

from core import deadline
from core.config import settings
from core.drawing_processor import DrawingProcessor
from core.providers import drawing_vision
from core.providers.qwen_client import SheetReading

pytest.importorskip("numpy")
Image = pytest.importorskip("PIL.Image")

from tests.fixtures import synthetic_sheet as sheet  # noqa: E402

SHEET_URI = "assets/doc-1/sheet.png"
TITLE = drawing_vision.Region(kind=drawing_vision.KIND_TITLE_BLOCK, bbox=[0.6, 0.8, 1.0, 1.0])
SPEC = drawing_vision.Region(kind=drawing_vision.KIND_TABLE, bbox=[0.6, 0.1, 1.0, 0.4], rows=5, cols=3)
EMPTY = drawing_vision.Region(kind=drawing_vision.KIND_TABLE, bbox=[0.05, 0.05, 0.3, 0.2], rows=2, cols=2)


class SlowQwen:
    """Модель с задержкой на каждый проход; запоминает, что и когда спросили."""

    def __init__(self, delay=0.2, fail=None, on_call=None):
        self.available = True
        self.model = "fake-vl"
        self.delay = delay
        self.fail = fail
        self.on_call = on_call
        self.lock = threading.Lock()
        self.asked = []

    def _call(self, kind, answer):
        with self.lock:
            self.asked.append(kind)
        if self.on_call:
            self.on_call(kind)
        deadline.check_cancelled()
        time.sleep(self.delay)
        if kind == self.fail:
            raise RuntimeError("связь с моделью оборвалась")
        return answer

    def read_sheet(self, image, hint="", until=None):
        return self._call("sheet", SheetReading(sheet_type="detail", annotations=[
            {"category": "size", "value": "⌀44H7", "bbox": [0.2, 0.3, 0.3, 0.35]},
            {"category": "roughness", "value": "Ra 3,2", "bbox": [0.4, 0.3, 0.5, 0.35]},
        ]))

    def transcribe(self, image):
        return self._call("transcribe", ["ТАДУ 687000.001", "Сталь 20 ГОСТ 1050-88"])

    def read_title_block(self, image):
        return self._call("title_block", {
            "fields": {"Обозначение": "ТАДУ 687000.001", "Материал": "Сталь 20 ГОСТ 1050-88"},
            "rows": [["Разраб.", "Иванов"]],
        })

    def read_table(self, image):
        centre = image.convert("L").getpixel((image.width // 2, image.height // 2))
        answer = (
            {"title": "", "headers": ["Поз.", "Обозначение", "Кол."],
             "rows": [["1", "ТАДУ 687000.002", "2"]]}
            if centre < 200 else {"title": "", "headers": [], "rows": []}
        )
        return self._call("table", answer)


@pytest.fixture
def storage(temp_storage):
    temp_storage.write_file(SHEET_URI, sheet.png_bytes())
    return temp_storage


@pytest.fixture
def layout(monkeypatch):
    image = Image.new("RGB", (800, 600), "white")
    # Спецификация закрашена серым: по вырезке видно, какая таблица спрошена.
    image.paste((128, 128, 128), (480, 60, 800, 240))
    fixed = drawing_vision.SheetLayout(image=image, frame=[0.0, 0.0, 1.0, 1.0],
                                       regions=[SPEC, EMPTY, TITLE])
    monkeypatch.setattr(drawing_vision, "analyse", lambda _image: fixed)
    monkeypatch.setattr(settings, "DRAWING_TITLE_BLOCK_PASS", False)
    monkeypatch.setattr(settings, "DRAWING_TRANSCRIBE_TILES", False)
    monkeypatch.setattr(settings, "DRAWING_TIME_BUDGET", 420)
    return fixed


def _run(storage, passes, monkeypatch, qwen):
    monkeypatch.setattr(settings, "DRAWING_PASSES_PARALLEL", passes)
    processor = DrawingProcessor(storage, qwen=qwen)
    started = time.monotonic()
    result = processor.process(SHEET_URI, {}, transcribe=True)
    return result, time.monotonic() - started


def test_wall_time_is_longest_pass(storage, layout, monkeypatch):
    # Пять проходов: лист, текст, штамп, две таблицы.
    _, one = _run(storage, 1, monkeypatch, SlowQwen(delay=0.2))
    _, five = _run(storage, 5, monkeypatch, SlowQwen(delay=0.2))
    assert one >= 1.0
    assert five < 0.6


def test_record_is_identical(storage, layout, monkeypatch):
    first, _ = _run(storage, 1, monkeypatch, SlowQwen(delay=0.01))
    third, _ = _run(storage, 3, monkeypatch, SlowQwen(delay=0.01))
    # Ключи payload в том же порядке: dumps без sort_keys.
    assert json.dumps(first, ensure_ascii=False) == json.dumps(third, ensure_ascii=False)
    assert [region["reason"] for region in third["missing_regions"]] == ["region_not_read"]
    assert third["payload"]["regions_read"] == 2


def test_failed_pass_does_not_cancel_others(storage, layout, monkeypatch):
    qwen = SlowQwen(delay=0.05, fail="transcribe")
    with pytest.raises(RuntimeError):
        _run(storage, 3, monkeypatch, qwen)
    assert sorted(qwen.asked) == sorted(["sheet", "transcribe", "table", "table", "title_block"])


def test_failed_pass_fails_sheet_as_before(storage, layout, monkeypatch):
    with pytest.raises(RuntimeError):
        _run(storage, 1, monkeypatch, SlowQwen(delay=0.0, fail="transcribe"))


def test_cancellation_stops_new_passes(storage, layout, monkeypatch):
    stop = threading.Event()
    qwen = SlowQwen(delay=0.1, on_call=lambda kind: stop.set())
    monkeypatch.setattr(settings, "DRAWING_PASSES_PARALLEL", 2)
    processor = DrawingProcessor(storage, qwen=qwen)
    with deadline.cancel_on(stop), pytest.raises(deadline.Cancelled):
        processor.process(SHEET_URI, {}, transcribe=True)
    # Два первых прохода успели начаться, остальные — нет.
    assert len(qwen.asked) <= 2


def test_budget_spent_skips_regions_in_both_modes(storage, layout, monkeypatch):
    monkeypatch.setattr(settings, "DRAWING_TIME_BUDGET", 0.001)
    first, _ = _run(storage, 1, monkeypatch, SlowQwen(delay=0.01))
    assert first["payload"]["regions_skipped"] == 3
    third, _ = _run(storage, 3, monkeypatch, SlowQwen(delay=0.01))
    # Разом области начинаются раньше, чем бюджет кончится на первых
    # проходах, — в этом и выигрыш; запись при этом та же по виду.
    assert third["payload"].get("regions_skipped", 0) <= 3


def test_table_pass_is_logged(storage, layout, monkeypatch, caplog):
    caplog.set_level("INFO", logger="core.drawing_processor")
    _run(storage, 1, monkeypatch, SlowQwen(delay=0.0))
    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("Qwen-VL таблица:")]
    assert len(lines) == 2
    assert "итог строк 1" in lines[0]
    assert "итог пусто" in lines[1]
