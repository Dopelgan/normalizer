"""
Замер времени операций — один на все сервисы.

Повод конкретный: пятьдесят PDF без текстового слоя не уложились в таймаут
Quality Gate, и по логам нельзя было сказать, что именно отняло время —
чтение из хранилища, открытие PDF, быстрый OCR или поиск почти-дублей.
Таймаут подняли вслепую, с 120 до 600 секунд, и это лечит симптом.

Здесь у каждого заметного шага появляется длительность, и она уходит сразу
в три места:

* в лог — строкой JSON, которую разбирает любой сборщик логов;
* в ответ и в базу — полем `timings`, чтобы по конкретному файлу было
  видно, где он провёл время, не поднимая логи;
* в метрики Prometheus — гистограммой по этапам, чтобы видеть не один
  случай, а распределение.

Замер не имеет права ронять обработку: и Prometheus, и сборщик таймингов
необязательны, а ошибка внутри `measure` не должна подменять собой ошибку
измеряемого кода.
"""

from __future__ import annotations

import contextvars
import logging
import time
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional

logger = logging.getLogger("normalizer.timing")

# Имя сервиса в логах и метриках. Точка входа выставляет своё.
_SERVICE = "normalizer"

# Сборщик таймингов текущей операции. contextvars, а не глобальная
# переменная: у FastAPI на одном процессе несколько запросов сразу, и
# тайминги одного не должны попадать в ответ другому.
_COLLECTOR: contextvars.ContextVar[Optional[Dict[str, float]]] = contextvars.ContextVar(
    "normalizer_timings", default=None
)

# Границы гистограммы в секундах. Подобраны под то, что мы меряем: от
# чтения файла (миллисекунды) до разбора чертежа (минуты).
_BUCKETS = (0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120, 300, 600)

try:  # pragma: no cover — метрики необязательны
    from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

    _DURATION = Histogram(
        "normalizer_stage_duration_seconds",
        "Длительность этапа обработки",
        ("service", "stage", "outcome"),
        buckets=_BUCKETS,
    )
    _FAILURES = Counter(
        "normalizer_stage_failures_total",
        "Этапы, завершившиеся исключением",
        ("service", "stage", "error"),
    )
    METRICS_AVAILABLE = True
except ImportError:  # pragma: no cover
    CONTENT_TYPE_LATEST = "text/plain; charset=utf-8"
    _DURATION = None
    _FAILURES = None
    METRICS_AVAILABLE = False

    def generate_latest(*_args, **_kwargs):  # type: ignore[misc]
        return b""


# ===========================================================================
# Имя сервиса
# ===========================================================================

def set_service(name: str) -> None:
    """Вызывается точкой входа: gateway, quality_gate, parser, worker."""
    global _SERVICE
    _SERVICE = name or "normalizer"


def service() -> str:
    return _SERVICE


# ===========================================================================
# Сбор таймингов операции
# ===========================================================================

@contextmanager
def collect() -> Iterator[Dict[str, float]]:
    """
    Область, внутри которой длительности этапов складываются в один словарь.

    Словарь возвращается вызывающему — он и уходит в ответ и в базу. Этот
    же словарь видят вложенные `measure`, в том числе в чужих модулях: им
    ничего не нужно знать о сборе, достаточно мерить своё.
    """
    bucket: Dict[str, float] = {}
    token = _COLLECTOR.set(bucket)
    try:
        yield bucket
    finally:
        _COLLECTOR.reset(token)


def collected() -> Optional[Dict[str, float]]:
    """Текущий сборщик или None, если замер идёт вне области сбора."""
    return _COLLECTOR.get()


def add(stage: str, duration_ms: float) -> None:
    """Дописать длительность в сборщик. Повторный этап суммируется."""
    bucket = _COLLECTOR.get()
    if bucket is None:
        return
    bucket[stage] = round(bucket.get(stage, 0.0) + duration_ms, 2)


# ===========================================================================
# Замер
# ===========================================================================

@contextmanager
def measure(stage: str, **fields: Any) -> Iterator[Dict[str, Any]]:
    """
    Измерить этап.

    Возвращает словарь, в который измеряемый код дописывает всё, что стоит
    видеть рядом с длительностью: исход, число страниц, выбранный уровень
    лестницы. Поле `outcome` попадает и в метку метрики, поэтому кладут
    туда короткое слово, а не текст причины.

        with telemetry.measure("quality_gate.ocr", s3_fileid=fid) as span:
            probe = quick_ocr(...)
            span["outcome"] = "accept" if probe else "unmeasured"
    """
    span: Dict[str, Any] = dict(fields)
    started = time.perf_counter()
    error: Optional[BaseException] = None
    try:
        yield span
    except BaseException as exc:  # noqa: BLE001 — измеряем и отдаём как было
        error = exc
        raise
    finally:
        duration_ms = (time.perf_counter() - started) * 1000
        outcome = "error" if error is not None else str(span.get("outcome") or "ok")
        try:
            _publish(stage, duration_ms, outcome, span, error)
        except Exception as exc:  # noqa: BLE001 — замер не роняет обработку
            logger.debug("Не удалось записать тайминг %s: %s", stage, exc)


def _publish(
    stage: str,
    duration_ms: float,
    outcome: str,
    span: Dict[str, Any],
    error: Optional[BaseException],
) -> None:
    add(stage, duration_ms)

    if _DURATION is not None:
        _DURATION.labels(_SERVICE, stage, outcome).observe(duration_ms / 1000)
    if error is not None and _FAILURES is not None:
        _FAILURES.labels(_SERVICE, stage, type(error).__name__).inc()

    payload = {
        "event": "stage",
        "service": _SERVICE,
        "stage": stage,
        "duration_ms": round(duration_ms, 2),
        "outcome": outcome,
    }
    payload.update({k: v for k, v in span.items() if k != "outcome"})
    if error is not None:
        payload["error"] = f"{type(error).__name__}: {error}"

    logger.info(
        "%s: %s за %.0f мс (%s)", _SERVICE, stage, duration_ms, outcome,
        extra={"fields": payload},
    )


def timed(stage: str, **fields: Any):
    """Декоратор для случаев, когда функция целиком и есть этап."""

    def wrapper(function):
        from functools import wraps

        @wraps(function)
        def inner(*args, **kwargs):
            with measure(stage, **fields):
                return function(*args, **kwargs)

        return inner

    return wrapper


# ===========================================================================
# Отдача метрик
# ===========================================================================

def metrics_payload() -> bytes:
    return generate_latest()


def metrics_content_type() -> str:
    return CONTENT_TYPE_LATEST


def install_metrics_endpoint(app: Any, path: str = "/metrics") -> None:
    """Ручка /metrics на приложении FastAPI. Без prometheus_client — 501."""
    from fastapi import Response

    @app.get(path, include_in_schema=False)
    def metrics() -> Response:  # pragma: no cover — тривиальная отдача
        if not METRICS_AVAILABLE:
            return Response(
                content="prometheus_client не установлен\n",
                media_type="text/plain; charset=utf-8",
                status_code=501,
            )
        return Response(content=metrics_payload(), media_type=metrics_content_type())


def start_metrics_server(port: int) -> bool:
    """
    Отдельный HTTP-порт метрик для процесса без своего веб-сервера — воркера.
    Возвращает False, если метрики недоступны или порт занят.
    """
    if not METRICS_AVAILABLE or not port:
        return False
    try:  # pragma: no cover — сеть в тестах не поднимаем
        from prometheus_client import start_http_server

        start_http_server(port)
        logger.info("Метрики воркера на порту %d", port)
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("Метрики воркера не поднялись на порту %d: %s", port, exc)
        return False
