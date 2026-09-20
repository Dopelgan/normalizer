"""Приложение Celery: очереди, маршрутизация, таймауты."""

import logging

from celery import Celery
from celery.signals import worker_process_shutdown, worker_ready
from kombu import Queue

from core.config import settings
from core.workspace import configure_process_tempdir

# Воркер — главный производитель временных файлов (выкачанные из S3
# исходники, растры страниц), поэтому каталог задаётся до старта задач.
configure_process_tempdir()

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

# Очереди объявляются здесь, иначе `celery worker -Q ml_gpu` слушает
# очередь, в которую никто не пишет.
#
# Приём и разбор живут в разных очередях намеренно. Приём — это чтение,
# открытие формата и быстрый OCR двух страниц: работа лёгкая, её можно
# вести в несколько потоков. Разбор занимает GPU и идёт по одному
# документу. В общей очереди пачка сканов на приёме встаёт за чертежом,
# который модель читает семь минут, — и приём, который сам по себе быстр,
# ждёт вместе с ней.
app.conf.task_default_queue = "default"
app.conf.task_queues = (
    Queue("default", routing_key="default"),
    Queue(settings.CELERY_ML_QUEUE, routing_key=settings.CELERY_ML_QUEUE),
    Queue(settings.CELERY_INTAKE_QUEUE, routing_key=settings.CELERY_INTAKE_QUEUE),
)
app.conf.task_routes = {
    "ingest.tasks.process_document": {"queue": settings.CELERY_ML_QUEUE},
    "core.intake.tasks.process_intake_file": {"queue": settings.CELERY_INTAKE_QUEUE},
}

app.autodiscover_tasks(["ingest", "core.intake"])

# Задачи живут в пакете ingest — импорт нужен, чтобы они зарегистрировались
# при запуске воркера как `celery -A core.celery_app`.
for _module in ("ingest.tasks", "core.intake.tasks"):  # pragma: no cover
    try:
        __import__(_module)
    except Exception as _exc:  # noqa: BLE001
        # API-контейнеру задачи при импорте не нужны, но молчать нельзя:
        # ровно так когда-то и потерялась регистрация process_document.
        logging.getLogger(__name__).warning(
            "Задачи %s не импортировались: %s", _module, _exc
        )


@worker_ready.connect
def _start_metrics(**_kwargs):  # pragma: no cover — сигнал Celery
    """
    Метрики воркера. У сервисов FastAPI их отдаёт ручка /metrics, а у
    воркера своего веб-сервера нет — поднимаем отдельный порт.
    """
    from core import telemetry

    telemetry.start_metrics_server(settings.METRICS_WORKER_PORT)

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

