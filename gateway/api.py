"""
HTTP-интерфейс Data Gateway — вход всей цепочки приёма.

    POST /internal/v1/intake                          приём пакета (202)
    GET  /internal/v1/intake/results/{request_id}     готовность и вердикты
    POST /internal/v1/intake/sync                     приём без очереди
    GET  /internal/v1/intake/quarantine               очередь администратору
    POST /internal/v1/intake/quarantine/{id}/resolve  решение по карантину
    GET  /internal/v1/intake/report                   отчёт по массовому приёму
    GET  /internal/v1/intake/suggestions              предложения по правилам
    GET  /health, GET /metrics

Приём вынесен из HTTP-запроса в очередь. Ручка проверяет тело, ставит по
задаче на файл и отвечает 202 — вердикты забираются поллингом по
`/internal/v1/intake/results/{request_id}`.

Повод: пятьдесят PDF без текстового слоя не укладывались ни в таймаут
между Data Gateway и Quality Gate (120 с), ни в таймаут внешнего
потребителя (60 с). Потребитель, не дождавшись ответа, слал запрос заново,
и конвейер получал ту же пачку по второму кругу. Таймауты можно поднимать
и дальше, но следующий набор файлов будет тяжелее — границу нужно убрать,
а не отодвинуть.

Порядок этапов прежний: сначала отсев непригодных данных, затем проверка
технической пригодности на Quality Gate, и только утверждённые файлы уходят
в нормализатор. Личная фотография технически безупречна и любую проверку
качества прошла бы — поэтому отсев обязан идти первым. Оба этапа теперь
выполняются в одной задаче (`core.intake.pipeline`), без сетевого запроса
между ними.

Передача принятых файлов в нормализатор включается настройкой
INTAKE_FORWARD_TO_PARSER; по умолчанию цепочка останавливается на приёме.

Тело запроса — список файлов, у каждого своя операция:

    {"request_id": "...",
     "files": [{"s3_fileid": "a", "operation": "create"},
               {"s3_fileid": "b", "operation": "update"}]}

Операция принадлежит файлу, а не пакету: в одном приёме приходят и новые
документы, и обновления уже принятых.
"""

import logging
from typing import List, Optional

from fastapi import FastAPI, HTTPException

from core import logging_setup, telemetry
from core.db.session import SessionLocal
from core.intake import state
from core.intake.pipeline import IntakePipeline
from core.models.intake import (
    FileVerdict,
    IntakeAcceptedResponse,
    IntakeRequest,
    IntakeResponse,
    IntakeResultsResponse,
    QuarantineItem,
    ResolveRequest,
)
from core.repositories import IntakeRepository
from core.workspace import configure_process_tempdir

# Временные файлы процесса — в том сервиса, а не на слой контейнера.
configure_process_tempdir()
logging_setup.configure("data_gateway")

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Data Gateway",
    description="Отсев данных, не относящихся к корпоративным знаниям",
    version="2.0.0",
)
telemetry.install_metrics_endpoint(app)

STAGE = "data_gateway"
RESULTS_PATH = "/internal/v1/intake/results"


# ===========================================================================
# Приём
# ===========================================================================

@app.post(
    "/internal/v1/intake",
    response_model=IntakeAcceptedResponse,
    status_code=202,
)
def intake(request: IntakeRequest) -> IntakeAcceptedResponse:
    """
    Ставит пакет в очередь приёма и сразу отвечает 202.

    Повторный запрос по незавершённому request_id — конфликт: иначе
    потребитель, у которого истёк собственный таймаут, запускает ту же
    работу поверх ещё не законченной.
    """
    from core.intake.tasks import dispatch

    if state.is_active(request.request_id):
        raise HTTPException(
            status_code=409,
            detail=(
                f"Приём {request.request_id} уже выполняется. Заберите "
                f"результат по {RESULTS_PATH}/{request.request_id} "
                f"или используйте новый request_id."
            ),
        )

    state.init_request(
        request.request_id, [item.model_dump() for item in request.files]
    )
    dispatch(request.request_id, request.files)

    logger.info(
        "Приём %s поставлен в очередь: %d файлов",
        request.request_id, len(request.files),
        extra={"fields": {
            "event": "intake_queued", "request_id": request.request_id,
            "total": len(request.files),
        }},
    )
    return IntakeAcceptedResponse(
        request_id=request.request_id,
        total=len(request.files),
        poll_url=f"{RESULTS_PATH}/{request.request_id}",
    )


@app.get(RESULTS_PATH + "/{request_id}", response_model=IntakeResultsResponse)
def intake_results(request_id: str) -> IntakeResultsResponse:
    """Готовность приёма и вердикты по каждому файлу."""
    snapshot = state.get_state(request_id)
    if snapshot is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Приём {request_id} не найден: он не создавался или истёк "
                f"срок хранения результата."
            ),
        )
    return IntakeResultsResponse(**snapshot)


@app.post("/internal/v1/intake/sync", response_model=IntakeResponse)
def intake_sync(request: IntakeRequest) -> IntakeResponse:
    """
    Приём без очереди: ответ отдаётся, когда готовы вердикты по всем файлам.

    Оставлен для ручной проверки и для установок, где очередь не поднята.
    На тяжёлых файлах он и упирается в таймауты, ради которых сделана
    очередь, — поэтому штатный вход не здесь.
    """
    pipeline = IntakePipeline()
    response = IntakeResponse(request_id=request.request_id)
    for item in request.files:
        result = pipeline.run(request.request_id, item)
        verdicts = [FileVerdict(**v) for v in result["verdicts"]]
        response.verdicts.extend(verdicts)
        _bucket(response, result["outcome"], item.s3_fileid)
        response.forwarded = response.forwarded or bool(result.get("forwarded"))
        response.forward_error = response.forward_error or result.get("forward_error")

    logger.info(
        "Data Gateway %s: принято %d, карантин %d, отклонено %d",
        request.request_id, len(response.accepted),
        len(response.quarantined), len(response.rejected),
    )
    return response


def _bucket(response: IntakeResponse, outcome: str, s3_fileid: str) -> None:
    """Раскладка исхода по спискам. Неизвестный исход — в карантин."""
    buckets = {
        "accept": response.accepted,
        "quarantine": response.quarantined,
        "reject": response.rejected,
    }
    bucket = buckets.get(outcome)
    if bucket is None:
        logger.error(
            "Неизвестный исход %r по файлу %s — считаем карантином",
            outcome, s3_fileid,
        )
        bucket = response.quarantined
    bucket.append(s3_fileid)


# ===========================================================================
# Карантин и отчёты
# ===========================================================================

@app.get("/internal/v1/intake/quarantine", response_model=List[QuarantineItem])
def quarantine(limit: int = 100) -> List[QuarantineItem]:
    """Очередь администратору: что не пропущено и почему."""
    session = SessionLocal()
    try:
        return [
            QuarantineItem(
                decision_id=row.id, s3_fileid=row.s3_fileid, stage=row.stage,
                layer=row.layer, category=row.category, reason=row.reason,
                confidence=row.confidence,
                created_at=row.created_at.isoformat() if row.created_at else None,
            )
            for row in IntakeRepository(session).quarantine_queue(limit)
        ]
    finally:
        session.close()


@app.post("/internal/v1/intake/quarantine/{record_id}/resolve")
def resolve(record_id: str, request: ResolveRequest) -> dict:
    """Решение человека по карантину. На этих решениях система и учится."""
    session = SessionLocal()
    try:
        with session.begin():
            record = IntakeRepository(session).resolve(
                record_id, request.outcome, request.resolved_by
            )
            if record is None:
                raise HTTPException(status_code=404, detail=f"Запись {record_id} не найдена")
            return {
                "decision_id": record.id,
                "s3_fileid": record.s3_fileid,
                "resolved_outcome": record.resolved_outcome,
                "resolved_by": record.resolved_by,
            }
    finally:
        session.close()


@app.get("/internal/v1/intake/report")
def report(stage: Optional[str] = None) -> dict:
    """Отчёт по итогам приёма: сколько принято, в карантине, отклонено."""
    session = SessionLocal()
    try:
        return IntakeRepository(session).summary(stage)
    finally:
        session.close()


@app.get("/internal/v1/intake/suggestions")
def suggestions() -> dict:
    """
    Что стоит поправить в правилах отсева. Система предлагает — меняет
    человек: молча подстраивать правила под себя она не должна.
    """
    session = SessionLocal()
    try:
        return {"suggestions": IntakeRepository(session).rule_suggestions()}
    finally:
        session.close()


@app.get("/health")
def health() -> dict:
    return {"status": "healthy", "service": "data_gateway"}
