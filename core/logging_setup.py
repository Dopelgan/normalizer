"""
Настройка логов одна на все сервисы.

Раньше каждая точка входа звала `logging.basicConfig` со своим форматом, и
строки четырёх сервисов складывались в одну ленту без общего признака: по
ней нельзя было ни отфильтровать сервис, ни собрать тайминги.

Формат выбирается настройкой LOG_FORMAT: `json` — строка на событие,
пригодная для сборщика логов; `text` — как было, для чтения глазами во
время разработки.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any, Dict

from core import telemetry

# Поля, которые LogRecord несёт всегда: в JSON они не нужны, туда уходит
# только то, что дописал вызывающий.
_RESERVED = frozenset(vars(logging.LogRecord("", 0, "", 0, "", (), None)))
_EXTRA_RESERVED = frozenset({"message", "asctime", "taskName"})


class JsonFormatter(logging.Formatter):
    """Одно событие — одна строка JSON."""

    def __init__(self, service: str):
        super().__init__()
        self.service = service

    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
            "level": record.levelname,
            "service": self.service,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # `extra={"fields": {...}}` — основной способ дописать структуру.
        fields = getattr(record, "fields", None)
        if isinstance(fields, dict):
            payload.update(fields)
        # Всё остальное, что пришло через extra, тоже не теряем.
        for key, value in vars(record).items():
            if key in _RESERVED or key in _EXTRA_RESERVED or key == "fields":
                continue
            payload.setdefault(key, value)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def configure(service: str) -> None:
    """
    Настроить логи процесса и назвать сервис.

    Вызывается точкой входа до создания приложения: имя сервиса попадает и
    в каждую строку лога, и в метки метрик.
    """
    from core.config import settings

    telemetry.set_service(service)

    level = getattr(logging, str(settings.LOG_LEVEL).upper(), logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    if str(settings.LOG_FORMAT).lower() == "json":
        handler.setFormatter(JsonFormatter(service))
    else:
        handler.setFormatter(logging.Formatter(
            f"%(asctime)s [%(levelname)s] {service} %(name)s: %(message)s"
        ))

    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)
    # Логи доступа uvicorn дублируют ручки метрик и health на каждый опрос.
    logging.getLogger("uvicorn.access").setLevel(
        max(level, logging.WARNING) if settings.QUIET_ACCESS_LOG else level
    )
