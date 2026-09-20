"""
Задача Celery: приём одного файла.

Одна задача на файл, а не на пакет. Пакет из пятидесяти сканов, обёрнутый
в одну задачу, сначала доходит до потолка времени, а потом теряет вердикты
по всем пятидесяти разом — притом что сорок из них были готовы. По файлу
на задачу: тяжёлый скан задерживает только себя, остальные вердикты уже
лежат в состоянии запроса и доступны поллингом.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Sequence

from celery.exceptions import SoftTimeLimitExceeded

from core.celery_app import app
from core.config import settings
from core.gateway.service import QUARANTINE
from core.intake import state
from core.intake.pipeline import IntakePipeline
from core.models.intake import FileVerdict, IntakeFile

logger = logging.getLogger(__name__)

STAGE = "intake"


@app.task(
    bind=True,
    name="core.intake.tasks.process_intake_file",
    queue=settings.CELERY_INTAKE_QUEUE,
    acks_late=True,
    max_retries=0,
    soft_time_limit=settings.INTAKE_TASK_SOFT_TIME_LIMIT,
    time_limit=settings.INTAKE_TASK_TIME_LIMIT,
)
def process_intake_file(
    self, request_id: str, s3_fileid: str, operation: str = "create"
) -> Dict[str, Any]:
    """Принимает один файл и кладёт итог в состояние запроса."""
    state.mark_started(request_id)
    item = IntakeFile(s3_fileid=s3_fileid, operation=operation)

    try:
        result = IntakePipeline().run(request_id, item)
    except SoftTimeLimitExceeded:
        logger.error("Приём %s не уложился в потолок времени", s3_fileid)
        result = _unfinished(
            s3_fileid,
            f"Приём не уложился в {settings.INTAKE_TASK_SOFT_TIME_LIMIT} с. "
            f"Файл необычно тяжёл — нужно решение человека.",
        )
    except Exception as exc:  # noqa: BLE001 — вердикт обязан быть у каждого файла
        logger.error("Приём %s не удался: %s", s3_fileid, exc, exc_info=True)
        # Ошибка обработки — не приговор документу: он уходит в карантин,
        # а не в отказ. Отказ означал бы, что файл проверен и не годится.
        result = _unfinished(s3_fileid, f"Приём завершился ошибкой: {exc}")

    processed = state.put_file_result(request_id, s3_fileid, result)
    logger.info(
        "Приём %s: файл %s -> %s (%d готово)",
        request_id, s3_fileid, result["outcome"], processed,
        extra={"fields": {
            "event": "intake_file", "request_id": request_id,
            "s3_fileid": s3_fileid, "outcome": result["outcome"],
            "timings_ms": result.get("timings_ms") or {},
        }},
    )
    return {"s3_fileid": s3_fileid, "outcome": result["outcome"]}


def _unfinished(s3_fileid: str, reason: str) -> Dict[str, Any]:
    verdict = FileVerdict(
        s3_fileid=s3_fileid, outcome=QUARANTINE, stage=STAGE,
        layer="QG-0", reason=reason, confidence=0.0,
    )
    return {
        "s3_fileid": s3_fileid,
        "outcome": QUARANTINE,
        "verdicts": [verdict.model_dump(mode="json")],
        "forwarded": False,
        "forward_error": None,
        "timings_ms": {},
    }


def dispatch(request_id: str, files: Sequence[IntakeFile]) -> List[str]:
    """Ставит по задаче на каждый файл пакета. Возвращает их идентификаторы."""
    task_ids: List[str] = []
    for item in files:
        task = process_intake_file.delay(
            request_id=request_id,
            s3_fileid=item.s3_fileid,
            operation=item.operation,
        )
        task_ids.append(task.id)
        logger.info(
            "Задача приёма %s поставлена для файла %s (%s)",
            task.id, item.s3_fileid, item.operation,
        )
    return task_ids
