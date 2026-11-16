"""
Места на видеокарте для фоновых вызовов разбора (GPU_GATE_ENABLED).

Счётчик общий для всех воркеров и живёт в Redis (здесь — fakeredis):
держателей не больше предела, умерший держатель освобождает место по
истечении аренды, предел зависит от часа по UTC, лежащий Redis не
останавливает разбор. Поле `priority` уходит в vLLM, но не в ключ кэша.
Выключенный флаг не трогает ни Redis, ни тело запроса.
"""

import json
import threading
import time
from datetime import datetime, timezone

import pytest

from core import telemetry
from core.config import settings
from core.providers import gpu_gate, qwen_client
from core.providers.qwen_client import QwenVisionClient

fakeredis = pytest.importorskip("fakeredis")
Image = pytest.importorskip("PIL.Image")


@pytest.fixture
def redis_client(monkeypatch):
    client = fakeredis.FakeRedis()
    gpu_gate.set_client(client)
    monkeypatch.setattr(settings, "GPU_GATE_ENABLED", True)
    monkeypatch.setattr(settings, "GPU_GATE_YIELD_ON_WAITING", False)
    monkeypatch.setattr(gpu_gate, "_POLL_SECONDS", 0.01)
    yield client
    gpu_gate.set_client(None)


class Response:
    status_code = 200
    text = ""

    def __init__(self, content):
        self._payload = {"choices": [{"message": {"content": content}, "finish_reason": "stop"}]}

    def json(self):
        return self._payload


class Session:
    """Подмена HTTP-сессии: запоминает тела запросов."""

    def __init__(self, delay=0.0):
        self.bodies = []
        self.delay = delay
        self.lock = threading.Lock()
        self.in_flight = 0
        self.max_in_flight = 0

    def post(self, url, json=None, headers=None, timeout=None):
        with self.lock:
            self.bodies.append(json)
            self.in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            time.sleep(self.delay)
            return Response('{"lines": ["ТАДУ 687000.001"]}')
        finally:
            with self.lock:
                self.in_flight -= 1


def _client(session):
    return QwenVisionClient(endpoint="http://vlm.test/v1", model="qwen", session=session)


def _image():
    return Image.new("RGB", (64, 64), "white")


# ---------------------------------------------------------------- счётчик
def test_holder_over_limit_waits(redis_client):
    assert gpu_gate.try_acquire(gpu_gate.VLM, "a", lease=60, slots=2)
    assert gpu_gate.try_acquire(gpu_gate.VLM, "b", lease=60, slots=2)
    assert not gpu_gate.try_acquire(gpu_gate.VLM, "c", lease=60, slots=2)
    # Отступивший претендент не остаётся в счёте.
    assert gpu_gate.holders(gpu_gate.VLM) == 2
    gpu_gate.release(gpu_gate.VLM, "a")
    assert gpu_gate.try_acquire(gpu_gate.VLM, "c", lease=60, slots=2)


def test_expired_lease_frees_slot(redis_client):
    # Держатель умер, не отдав место: оно освобождается по истечении аренды.
    assert gpu_gate.try_acquire(gpu_gate.MINERU, "dead", lease=0.05, slots=1)
    assert not gpu_gate.try_acquire(gpu_gate.MINERU, "next", lease=60, slots=1)
    time.sleep(0.1)
    assert gpu_gate.try_acquire(gpu_gate.MINERU, "next", lease=60, slots=1)


def test_slots_follow_the_clock(monkeypatch):
    monkeypatch.setattr(settings, "GPU_GATE_BUSY_HOURS_UTC", "4-13")
    day = datetime(2026, 11, 16, 8, 0, tzinfo=timezone.utc)
    night = datetime(2026, 11, 16, 13, 0, tzinfo=timezone.utc)
    assert gpu_gate.limit(gpu_gate.VLM, day) == settings.GPU_GATE_VLM_SLOTS_DAY
    assert gpu_gate.limit(gpu_gate.VLM, night) == settings.GPU_GATE_VLM_SLOTS_NIGHT
    assert gpu_gate.limit(gpu_gate.MINERU, day) == settings.GPU_GATE_MINERU_SLOTS_DAY
    assert gpu_gate.limit(gpu_gate.MINERU, night) == settings.GPU_GATE_MINERU_SLOTS_NIGHT
    # Окно через полночь.
    monkeypatch.setattr(settings, "GPU_GATE_BUSY_HOURS_UTC", "22-6")
    assert gpu_gate.is_busy(datetime(2026, 11, 16, 23, 0, tzinfo=timezone.utc))
    assert gpu_gate.is_busy(datetime(2026, 11, 16, 5, 0, tzinfo=timezone.utc))
    assert not gpu_gate.is_busy(datetime(2026, 11, 16, 12, 0, tzinfo=timezone.utc))


def test_concurrent_calls_never_exceed_slots(redis_client, monkeypatch):
    monkeypatch.setattr(settings, "GPU_GATE_VLM_SLOTS_DAY", 2)
    monkeypatch.setattr(settings, "GPU_GATE_VLM_SLOTS_NIGHT", 2)
    session = Session(delay=0.05)
    client = _client(session)
    threads = [
        threading.Thread(target=client.transcribe, args=(Image.new("RGB", (64, 64), (i, i, i)),))
        for i in range(6)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(session.bodies) == 6
    assert session.max_in_flight == 2
    assert gpu_gate.holders(gpu_gate.VLM) == 0


def test_redis_down_call_proceeds(monkeypatch):
    class Broken:
        def pipeline(self, *args, **kwargs):
            raise ConnectionError("redis down")

        def zrem(self, *args):
            raise ConnectionError("redis down")

    gpu_gate.set_client(Broken())
    monkeypatch.setattr(settings, "GPU_GATE_ENABLED", True)
    monkeypatch.setattr(settings, "GPU_GATE_YIELD_ON_WAITING", False)
    try:
        session = Session()
        assert _client(session).transcribe(_image()) == ["ТАДУ 687000.001"]
        assert len(session.bodies) == 1
    finally:
        gpu_gate.set_client(None)


def test_flag_off_touches_nothing(monkeypatch):
    class Untouchable:
        def __getattr__(self, name):
            raise AssertionError("Redis тронут при выключенном флаге")

    gpu_gate.set_client(Untouchable())
    monkeypatch.setattr(settings, "GPU_GATE_ENABLED", False)
    try:
        with telemetry.collect() as timings:
            session = Session()
            _client(session).transcribe(_image())
        assert "priority" not in session.bodies[0]
        assert "parse.gate_wait_ms" not in timings
    finally:
        gpu_gate.set_client(None)


# --------------------------------------------------------------- приоритет
def test_priority_absent_when_unset(monkeypatch):
    monkeypatch.setattr(settings, "QWEN_REQUEST_PRIORITY", None)
    session = Session()
    _client(session).transcribe(_image())
    assert "priority" not in session.bodies[0]


def test_priority_sent_but_not_in_cache_key(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "VLM_RESPONSE_CACHE", True)
    monkeypatch.setattr(settings, "VLM_RESPONSE_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(settings, "VLM_CACHE_V1_FALLBACK", False)
    keys = []
    real_key = qwen_client.cache_key
    monkeypatch.setattr(qwen_client, "cache_key",
                        lambda *args: keys.append(real_key(*args)) or keys[-1])

    monkeypatch.setattr(settings, "QWEN_REQUEST_PRIORITY", None)
    plain = Session()
    _client(plain).transcribe(_image())

    monkeypatch.setattr(settings, "QWEN_REQUEST_PRIORITY", 5)
    for path in tmp_path.rglob("*.json"):
        path.unlink()
    with_priority = Session()
    _client(with_priority).transcribe(_image())

    assert keys[0] == keys[1]
    assert with_priority.bodies[0]["priority"] == 5
    body = dict(with_priority.bodies[0])
    body.pop("priority")
    assert json.dumps(body, sort_keys=True) == json.dumps(plain.bodies[0], sort_keys=True)
    # Ответ, записанный с приоритетом, находится и без него.
    monkeypatch.setattr(settings, "QWEN_REQUEST_PRIORITY", None)
    third = Session()
    assert _client(third).transcribe(_image()) == ["ТАДУ 687000.001"]
    assert third.bodies == []


# ------------------------------------------------- бюджет листа и тайминги
def test_wait_is_not_budget_and_goes_to_timings(redis_client, monkeypatch):
    monkeypatch.setattr(settings, "GPU_GATE_VLM_SLOTS_DAY", 1)
    monkeypatch.setattr(settings, "GPU_GATE_VLM_SLOTS_NIGHT", 1)
    assert gpu_gate.try_acquire(gpu_gate.VLM, "other-worker", lease=60, slots=1)
    threading.Timer(0.3, gpu_gate.release, args=(gpu_gate.VLM, "other-worker")).start()

    with telemetry.collect() as timings, gpu_gate.budget_clock():
        wall_started = time.monotonic()
        budget_started = gpu_gate.budget_now()
        with gpu_gate.slot(gpu_gate.VLM, lease=60):
            pass
        wall = time.monotonic() - wall_started
        budget = gpu_gate.budget_now() - budget_started

    assert wall >= 0.29
    assert budget < 0.1
    assert timings["parse.gate_wait_ms"] >= 290


def test_parallel_waits_count_once(redis_client):
    clock = gpu_gate._WaitClock()
    started = time.monotonic()
    clock.enter()
    clock.enter()
    time.sleep(0.1)
    clock.leave()
    clock.leave()
    assert clock.waited() == pytest.approx(time.monotonic() - started, abs=0.03)


def test_yield_waits_while_vllm_has_queue(redis_client, monkeypatch):
    monkeypatch.setattr(settings, "GPU_GATE_YIELD_ON_WAITING", True)
    answers = iter([3.0, 1.0, 0.0])
    monkeypatch.setattr(gpu_gate, "_waiting_requests", lambda: next(answers))
    with gpu_gate.slot(gpu_gate.VLM, lease=60):
        pass
    with pytest.raises(StopIteration):
        next(answers)


def test_waiting_requests_parses_metrics(monkeypatch):
    class Metrics:
        status_code = 200
        text = (
            "# HELP vllm:num_requests_waiting Number of requests waiting.\n"
            'vllm:num_requests_waiting{engine="0",model_name="qwen"} 2.0\n'
            'vllm:num_requests_running{engine="0",model_name="qwen"} 7.0\n'
        )

    urls = []
    monkeypatch.setattr(settings, "QWEN_ENDPOINT", "http://vlm.test:8000/v1")
    monkeypatch.setattr(gpu_gate, "_metrics_cache", (0.0, 0.0))
    monkeypatch.setattr(gpu_gate.requests, "get",
                        lambda url, timeout: urls.append(url) or Metrics())
    assert gpu_gate._waiting_requests() == 2.0
    assert urls == ["http://vlm.test:8000/metrics"]
