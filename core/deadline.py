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
import time
from contextlib import contextmanager
from typing import Iterator, Optional

_deadline: contextvars.ContextVar[Optional[float]] = contextvars.ContextVar(
    "parse_deadline", default=None
)


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


def bounded(budget: float, reserve: float = 0.0) -> float:
    """Собственный бюджет шага, но не больше остатка срока за вычетом запаса."""
    left = remaining()
    if left is None:
        return budget
    return max(0.0, min(budget, left - reserve))
