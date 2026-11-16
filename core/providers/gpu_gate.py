"""
Места на видеокарте для фоновых вызовов разбора (GPU_GATE_ENABLED).

Зачем. На одной карте живут и ответы пользователям (vLLM), и разбор
документов: чтение листов Qwen-VL, перечитывание таблиц, MinerU. Семь
воркеров разбора, каждый со своими листами и частями листа, легко ставят в
очередь vLLM десятки запросов разом — и вопрос пользователя ждёт за ними.
Днём разбор должен уступать, ночью — брать карту целиком.

Как. Один общий счётчик на все воркеры — отсортированное множество в Redis
Celery: элемент — держатель места, оценка — момент, когда его аренда
истекает. Умерший воркер места не держит: его аренда кончается сама. Аренда
берётся на срок одного HTTP-вызова (таймаут плюс запас), поэтому повторы
и паузы между ними места не занимают.

Взятие места без Lua: добавить себя, посчитать, и если держателей больше
предела — убрать себя и ждать. Два претендента могут разом увидеть
переполнение и оба отступить, но держателей больше предела не бывает
никогда: каждый оставшийся видел счёт, в который уже входили все, кто
остался до него.

Redis недоступен — вызов идёт без места (одно предупреждение на процесс):
очередь к модели хуже, чем разбор, вставший целиком.

Время ожидания места не съедает бюджет листа `DRAWING_TIME_BUDGET`: ветка
чертежей меряет бюджет часами `budget_now()`, из которых вычтено время,
когда лист стоял: его потоки ждали места и ни один его вызов модели не
шёл. Ждать места дольше остатка срока задачи вызов не станет — пойдёт без
места, чтобы документ не упал по сроку. В тайминги документа ожидание
идёт отдельным ключом `parse.gate_wait_ms`. С выключенным флагом часы
совпадают с `time.monotonic()`, и ничего не меняется.
"""

from __future__ import annotations

import contextvars
import logging
import random
import re
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Callable, Iterator, Optional, Tuple

import requests

from core import deadline, telemetry
from core.config import settings

logger = logging.getLogger(__name__)

VLM = "vlm"
MINERU = "mineru"

# Ключ в Redis на вид места.
_KEY = "normalizer:gpu_gate:{kind}"
# Пауза между попытками взять место; случайная добавка разводит воркеры.
_POLL_SECONDS = 0.25
# Срок жизни ключа: дольше любой аренды и только продлевается — иначе
# короткая аренда укоротила бы ключ под длинной, и счёт пропал бы вместе
# с ним, пока длинный вызов ещё идёт.
_KEY_TTL = 4 * 3600
# Сколько не ходить в лежащий Redis после отказа: каждая попытка — это
# таймаут соединения на каждом вызове модели.
_REDIS_RETRY_SECONDS = 30.0
# Сколько ждать, пока у vLLM есть очередь, прежде чем идти всё равно.
_YIELD_MAX_SECONDS = 30.0
# Сколько считать прочитанное число ждущих запросов vLLM свежим.
_METRICS_TTL = 2.0
_WAITING = re.compile(r"^vllm:num_requests_waiting(?:\{[^}]*\})?\s+([0-9.eE+-]+)", re.M)

_client_lock = threading.Lock()
_client: Any = None
_warned = False
_redis_down_until = 0.0
_metrics_lock = threading.Lock()
_metrics_cache: Tuple[float, float] = (0.0, 0.0)  # (когда прочитано, сколько ждут)
_timings_lock = threading.Lock()


# ===========================================================================
# Redis
# ===========================================================================

def _redis():
    global _client
    with _client_lock:
        if _client is None:
            import redis

            _client = redis.Redis.from_url(
                settings.CELERY_BROKER_URL, socket_timeout=2, socket_connect_timeout=2,
            )
        return _client


def set_client(client: Any) -> None:
    """Подмена клиента (тесты с fakeredis); None — вернуть настоящий."""
    global _client, _warned, _redis_down_until
    with _client_lock:
        _client = client
    _warned = False
    _redis_down_until = 0.0


def _fail_open(exc: Exception) -> None:
    global _warned, _redis_down_until
    _redis_down_until = time.monotonic() + _REDIS_RETRY_SECONDS
    if not _warned:
        _warned = True
        logger.warning("Места на видеокарте не учитываются — Redis недоступен: %s", exc)


# ===========================================================================
# Предел по часам
# ===========================================================================

def _busy_hours() -> Tuple[int, int]:
    """`GPU_GATE_BUSY_HOURS_UTC` «4-13» -> (4, 13): с 4:00 до 12:59 UTC."""
    text = str(settings.GPU_GATE_BUSY_HOURS_UTC or "").strip()
    try:
        start, end = (int(part) for part in text.split("-", 1))
    except ValueError:
        return 0, 0
    return start % 24, end % 24


def is_busy(now: Optional[datetime] = None) -> bool:
    """Рабочие часы заказчика: место за ответами пользователей."""
    hour = (now or datetime.now(timezone.utc)).astimezone(timezone.utc).hour
    start, end = _busy_hours()
    if start == end:
        return False
    if start < end:
        return start <= hour < end
    return hour >= start or hour < end     # окно через полночь: «22-6»


def limit(kind: str, now: Optional[datetime] = None) -> int:
    """Сколько мест вида `kind` сейчас."""
    busy = is_busy(now)
    if kind == MINERU:
        value = settings.GPU_GATE_MINERU_SLOTS_DAY if busy else settings.GPU_GATE_MINERU_SLOTS_NIGHT
    else:
        value = settings.GPU_GATE_VLM_SLOTS_DAY if busy else settings.GPU_GATE_VLM_SLOTS_NIGHT
    return max(1, int(value))


# ===========================================================================
# Взять и отдать место
# ===========================================================================

def try_acquire(kind: str, token: str, lease: float, slots: Optional[int] = None) -> bool:
    """
    Одна попытка взять место. Бросает исключение Redis — решает вызывающий.
    Истёкшие аренды (умерший воркер) снимаются здесь же.
    """
    key = _KEY.format(kind=kind)
    now = time.time()
    client = _redis()
    pipe = client.pipeline(transaction=True)
    pipe.zremrangebyscore(key, "-inf", now)
    pipe.zadd(key, {token: now + lease})
    pipe.zcard(key)
    # Брошенный ключ уберётся сам; срок только продлевается.
    pipe.expire(key, max(_KEY_TTL, int(lease) + 60))
    count = pipe.execute()[2]
    if count <= (slots if slots is not None else limit(kind)):
        return True
    client.zrem(key, token)
    return False


def release(kind: str, token: str) -> None:
    try:
        _redis().zrem(_KEY.format(kind=kind), token)
    except Exception as exc:  # noqa: BLE001 — аренда истечёт сама
        logger.debug("Место %s не отдано: %s", kind, exc)


def holders(kind: str) -> int:
    """Сколько мест вида `kind` занято сейчас (для лога и тестов)."""
    key = _KEY.format(kind=kind)
    client = _redis()
    client.zremrangebyscore(key, "-inf", time.time())
    return int(client.zcard(key))


def _waiting_requests() -> float:
    """
    `vllm:num_requests_waiting` с `/metrics` модели; кэш на 2 с, чтобы семь
    воркеров не опрашивали vLLM на каждый вызов. Ошибка — 0: уступать
    некому, раз очереди не видно.
    """
    global _metrics_cache
    with _metrics_lock:
        read_at, value = _metrics_cache
        if time.monotonic() - read_at < _METRICS_TTL:
            return value
    base = (settings.QWEN_ENDPOINT or "").rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    value = 0.0
    if base:
        try:
            response = requests.get(f"{base}/metrics", timeout=2)
            if response.status_code == 200:
                value = sum(float(found) for found in _WAITING.findall(response.text))
        except (requests.RequestException, ValueError):
            value = 0.0
    with _metrics_lock:
        _metrics_cache = (time.monotonic(), value)
    return value


def _yield_to_answers() -> None:
    """Пока у vLLM есть очередь, фоновый вызов ждёт до 30 с, потом идёт."""
    if not settings.GPU_GATE_YIELD_ON_WAITING:
        return
    until = time.monotonic() + _YIELD_MAX_SECONDS
    while time.monotonic() < until and _waiting_requests() > 0:
        deadline.check_cancelled()
        time.sleep(_POLL_SECONDS)


@contextmanager
def slot(kind: str, lease: float) -> Iterator[None]:
    """
    Место на видеокарте на один вызов. `lease` — на сколько секунд берётся
    аренда: таймаут вызова плюс запас. Без флага — ничего не делает.
    """
    if not settings.GPU_GATE_ENABLED or time.monotonic() < _redis_down_until:
        yield
        return
    token = uuid.uuid4().hex
    started = time.monotonic()
    clock = _clock.get()
    if clock is not None:
        clock.enter()
    held = False
    try:
        if kind == VLM:
            _yield_to_answers()
        while True:
            try:
                if try_acquire(kind, token, lease):
                    held = True
                    break
            except Exception as exc:  # noqa: BLE001 — Redis лежит: идём без места
                _fail_open(exc)
                break
            deadline.check_cancelled()
            left = deadline.remaining()
            if left is not None and left < lease:
                # Дальше ждать — значит не успеть сам вызов до срока задачи:
                # документ упал бы по сроку вместо обычной деградации. Вызов
                # идёт без места.
                logger.warning("Место %s не дождались: до срока задачи %.0f с, вызов идёт без места",
                               kind, left)
                break
            time.sleep(_POLL_SECONDS * (1.0 + random.random()))
    finally:
        if clock is not None:
            clock.leave()
        waited_ms = (time.monotonic() - started) * 1000.0
        with _timings_lock:
            telemetry.add("parse.gate_wait_ms", waited_ms)
    if clock is not None:
        clock.start_call()
    try:
        yield
    finally:
        if clock is not None:
            clock.end_call()
        if held:
            release(kind, token)


def call(kind: str, lease: float, function: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """`function(*args, **kwargs)` внутри `slot(kind, lease)`."""
    with slot(kind, lease):
        return function(*args, **kwargs)


# ===========================================================================
# Часы бюджета листа без ожидания мест
# ===========================================================================

class _WaitClock:
    """
    Сколько секунд лист стоял из-за мест: кто-то из его потоков ждал места,
    и ни один вызов листа при этом не шёл. Пока хоть один проход или часть
    листа работает с моделью, лист движется, и это время идёт в бюджет —
    иначе при параллельных проходах кто-то ждёт почти всегда, и бюджет листа
    фактически выключен. Три потока, ждавшие одну минуту, — минута, не три.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._waiting = 0
        self._calls = 0
        self._since = 0.0       # начало текущего простоя
        self._total = 0.0

    def _stalled(self) -> bool:
        return self._waiting > 0 and self._calls == 0

    def _change(self, waiting: int = 0, calls: int = 0) -> None:
        with self._lock:
            before = self._stalled()
            self._waiting += waiting
            self._calls += calls
            after = self._stalled()
            now = time.monotonic()
            if not before and after:
                self._since = now
            elif before and not after:
                self._total += now - self._since

    def enter(self) -> None:
        self._change(waiting=1)

    def leave(self) -> None:
        self._change(waiting=-1)

    def start_call(self) -> None:
        self._change(calls=1)

    def end_call(self) -> None:
        self._change(calls=-1)

    def waited(self) -> float:
        with self._lock:
            extra = time.monotonic() - self._since if self._stalled() else 0.0
            return self._total + extra


_clock: contextvars.ContextVar[Optional[_WaitClock]] = contextvars.ContextVar(
    "gpu_gate_wait_clock", default=None
)


@contextmanager
def budget_clock() -> Iterator[None]:
    """Область одного листа: ожидание мест внутри неё вычитается из `budget_now`."""
    token = _clock.set(_WaitClock())
    try:
        yield
    finally:
        _clock.reset(token)


def budget_now() -> float:
    """`time.monotonic()` без времени, которое лист ждал места на видеокарте."""
    clock = _clock.get()
    now = time.monotonic()
    return now - clock.waited() if clock is not None else now
