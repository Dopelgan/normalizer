"""Приложение Celery: очереди, маршрутизация, таймауты."""

import logging

from celery import Celery
from celery.signals import worker_process_shutdown
from kombu import Queue

from core.config import settings

app = Celery(
    "normalizer",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
)

app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone=settings.TZ,
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,          # GPU-задачи не копим в воркере
    task_time_limit=settings.CELERY_TASK_TIME_LIMIT,
    task_soft_time_limit=settings.CELERY_TASK_SOFT_TIME_LIMIT,
    result_expires=settings.RESULT_TTL_SECONDS,
)

# Очередь ml_gpu объявляется здесь, иначе `celery worker -Q ml_gpu` слушает
# очередь, в которую никто не пишет.
app.conf.task_default_queue = "default"
app.conf.task_queues = (
    Queue("default", routing_key="default"),
    Queue(settings.CELERY_ML_QUEUE, routing_key=settings.CELERY_ML_QUEUE),
)
app.conf.task_routes = {
    "ingest.tasks.process_document": {"queue": settings.CELERY_ML_QUEUE},
}

app.autodiscover_tasks(["ingest"])

# Задачи живут в пакете ingest — импорт нужен, чтобы они зарегистрировались
# при запуске воркера как `celery -A core.celery_app`.
try:  # pragma: no cover
    import ingest.tasks  # noqa: F401,E402
except Exception as _exc:  # noqa: BLE001
    # API-контейнеру задачи при импорте не нужны, но молчать нельзя:
    # ровно так когда-то и потерялась регистрация process_document.
    logging.getLogger(__name__).warning(
        "Задачи ingest.tasks не импортировались: %s", _exc
    )

@worker_process_shutdown.connect
def _close_http_pools(**_kwargs):  # pragma: no cover — сигнал Celery
    """
    Закрыть пул соединений к модели чертежей вместе с процессом. Сессия
    общая на процесс и живёт между задачами намеренно; остаётся закрыть её
    один раз, когда закрывать уже есть кому.
    """
    from core.providers.qwen_client import close_shared_session

    close_shared_session()


# Псевдоним: часть кода исторически ожидает имя celery_app.
celery_app = app

