"""
Общий дедлайн задачи разбора документа.

Бюджеты уровней лестницы (таймаут MinerU, время OCR), посчитанные на
каждый вызов отдельно, в сумме превышают лимит задачи Celery: большой
скан обрывается жёстким таймаутом без записи об ошибке. Здесь хранится один
срок на всю задачу; уровни берут `bounded(свой_бюджет)` и не начинаются,
если времени не осталось.

Вне задачи (тесты, ручной вызов) срок не задан и бюджеты не урезаются.
"""

from __future__ import annotations

import contextvars
import threading
import time
from contextlib import contextmanager
from typing import Iterator, Optional

_deadline: contextvars.ContextVar[Optional[float]] = contextvars.ContextVar(
    "parse_deadline", default=None
)

# Отмена работы, начатой в потоке (листы чертежа, VLM_SHEET_CONCURRENCY).
# Обрыв задачи Celery приходит в главный поток; потоки листов о нём не
# знают и продолжали бы ходить в модель уже после того, как задача
# закончилась. Событие ставит тот, кто запустил потоки, а вызов модели
# проверяет его перед каждым запросом (`check_cancelled`).
_cancel: contextvars.ContextVar[Optional[threading.Event]] = contextvars.ContextVar(
    "parse_cancel", default=None
)


class Cancelled(Exception):
    """Работа отменена тем, кто её запустил: дальше не идти."""


@contextmanager
def scope(seconds: float) -> Iterator[None]:
    """Задать срок на `seconds` секунд вперёд на время блока."""
    token = _deadline.set(time.monotonic() + seconds)
    try:
        yield
    finally:
        _deadline.reset(token)


def remaining() -> Optional[float]:
    """Сколько секунд осталось; None — срок не задан."""
    deadline = _deadline.get()
    return None if deadline is None else deadline - time.monotonic()


@contextmanager
def cancel_on(event: threading.Event) -> Iterator[None]:
    """Внутри блока `check_cancelled` бросает `Cancelled`, когда `event` поставлен."""
    token = _cancel.set(event)
    try:
        yield
    finally:
        _cancel.reset(token)


def check_cancelled() -> None:
    """Бросить `Cancelled`, если работа отменена. Вне `cancel_on` ничего не делает."""
    event = _cancel.get()
    if event is not None and event.is_set():
        raise Cancelled()


def bounded(budget: float, reserve: float = 0.0) -> float:
    """Собственный бюджет шага, но не больше остатка срока за вычетом запаса."""
    left = remaining()
    if left is None:
        return budget
    return max(0.0, min(budget, left - reserve))
