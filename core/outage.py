"""
Отказ сервиса — повод разобрать документ позже, а не хуже (PARSE_REQUEUE_ON_OUTAGE).

Когда MinerU или vLLM лежит, разбор сегодня не останавливается: скан
остаётся плоским OCR без таблиц, листы чертежа — непрочитанными, и документ
уходит в индекс таким навсегда. Это свойство минуты, а не документа. С
флагом задача разбора видит отказ и ставит документ в очередь заново через
`PARSE_OUTAGE_RETRY_SECONDS`; RAG всё это время видит файл «в обработке».

Отказом считается только то, что говорит о сервисе: нет соединения или 5xx.
Таймаут чтения — свойство документа (долгий скан, плотный лист) и остаётся
сегодняшней деградацией одного листа или уровня.

Ожидание ограничено: если сервис не поднялся за
`PARSE_OUTAGE_MAX_WAIT_SECONDS` с первого отказа, документ разбирается как
сегодня, с прежней пометкой о деградации. Первый отказ запоминается в
аргументах задачи (`outage_since`), поэтому срок общий на все её повторы.

`ServiceOutage` — наследник `BaseException`, а не `Exception`: по дороге к
задаче он проходит лестницу уровней и ветку чертежей, где неудача уровня или
листа ловится общим `except Exception` и превращается в деградацию — ровно
то, чего здесь быть не должно. Ловит его только задача разбора. Вне задачи
(области `scope` нет) он не бросается никогда.
"""

from __future__ import annotations

import contextvars
import logging
import threading
import time
from contextlib import contextmanager
from typing import Iterator, Optional

from core.config import settings

logger = logging.getLogger(__name__)


class ServiceOutage(BaseException):
    """Сервис разбора недоступен: документ разобрать позже."""


class _State:
    def __init__(self, since: float):
        self.since = since
        self.lock = threading.Lock()
        self.vlm_ok = 0
        self.vlm_down = 0


_state: contextvars.ContextVar[Optional[_State]] = contextvars.ContextVar(
    "parse_outage", default=None
)


@contextmanager
def scope(since: float = 0.0) -> Iterator[None]:
    """
    Область задачи разбора. `since` — когда документ впервые застал отказ
    (0 — не заставал). Вызовы модели в потоках листов видят ту же область:
    копия контекста держит тот же объект.
    """
    token = _state.set(_State(since))
    try:
        yield
    finally:
        _state.reset(token)


def requeue_allowed() -> bool:
    """Отказ сервиса сейчас ставит документ в очередь, а не деградирует его."""
    if not settings.PARSE_REQUEUE_ON_OUTAGE:
        return False
    state = _state.get()
    if state is None:
        return False
    if not state.since:
        return True
    return time.time() - state.since < settings.PARSE_OUTAGE_MAX_WAIT_SECONDS


def mineru_down(reason: str) -> None:
    """MinerU недоступен для растрового документа: в очередь, если можно."""
    if requeue_allowed():
        raise ServiceOutage(f"MinerU недоступен: {reason}")


def note_vlm(ok: bool) -> None:
    """Итог вызова модели: ответ получен или сервис отказал (нет связи, 5xx)."""
    state = _state.get()
    if state is None:
        return
    with state.lock:
        if ok:
            state.vlm_ok += 1
        else:
            state.vlm_down += 1


def check_vlm() -> None:
    """
    Модель не ответила документу ни разу и отказывала как сервис: в
    очередь, если можно. Один ответ из многих — уже не отказ сервиса, и
    непрочитанные листы остаются сегодняшней деградацией.
    """
    state = _state.get()
    if state is None or not requeue_allowed():
        return
    with state.lock:
        ok, down = state.vlm_ok, state.vlm_down
    if down and not ok:
        raise ServiceOutage(f"Qwen-VL недоступен: отказов {down}, ответов нет")
