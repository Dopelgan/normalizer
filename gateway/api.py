"""
HTTP-интерфейс Data Gateway — вход всей цепочки приёма.

    POST /internal/v1/intake                          приём пакета файлов
    GET  /internal/v1/intake/quarantine               очередь администратору
    POST /internal/v1/intake/quarantine/{id}/resolve  решение по карантину
    GET  /internal/v1/intake/report                   отчёт по массовому приёму
    GET  /internal/v1/intake/suggestions              предложения по правилам
    GET  /health

Порядок жёсткий: сначала отсев непригодных данных здесь, затем проверка
технической пригодности на Quality Gate, и только утверждённые файлы
уходят в нормализатор. Личная фотография технически безупречна и любую
проверку качества прошла бы — поэтому отсев обязан идти первым.

ВНИМАНИЕ: последнее звено сейчас отключено. Передача из Quality Gate в
нормализатор закомментирована (`quality/api.py::check`), поэтому приём
заканчивается вердиктами, разбор не запускается, а `forwarded` в ответе
остаётся `false`.

Тело запроса — список файлов, у каждого своя операция:

    {"request_id": "...",
     "files": [{"s3_fileid": "a", "operation": "create"},
               {"s3_fileid": "b", "operation": "update"}]}

Операция принадлежит файлу, а не пакету: в одном приёме приходят и новые
документы, и обновления уже принятых.
"""

import logging
from typing import List, Optional

import requests
from fastapi import FastAPI, HTTPException

from core.config import settings
from core.db.session import SessionLocal
from core.gateway.profile import load_profile
from core.gateway.service import DataGateway
from core.models.intake import (
    FileVerdict,
    IntakeFile,
    IntakeRequest,
    IntakeResponse,
    QuarantineItem,
    ResolveRequest,
)
from core.providers.file_locator import FileLocator, SourceFileNotFound
from core.providers.storage import StorageProviderFactory
from core.repositories import IntakeRepository
from core.repositories.intake_repo import decision_id
from core.workspace import configure_process_tempdir

# Временные файлы процесса — в том сервиса, а не на слой контейнера.
configure_process_tempdir()

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Data Gateway",
    description="Отсев данных, не относящихся к корпоративным знаниям",
    version="1.0.0",
)

STAGE = "data_gateway"


@app.post("/internal/v1/intake", response_model=IntakeResponse)
def intake(request: IntakeRequest) -> IntakeResponse:
    storage = StorageProviderFactory.default()
    locator = FileLocator(storage)
    gateway = DataGateway(profile=load_profile(), storage=storage)
    session = SessionLocal()
    response = IntakeResponse(request_id=request.request_id)

    try:
        for item in request.files:
            s3_fileid = item.s3_fileid
            uri, size = None, None
            try:
                located = locator.locate(s3_fileid)
                uri = located.uri
                size = storage.size(uri) if hasattr(storage, "size") else None
            except SourceFileNotFound as exc:
                verdict = FileVerdict(
                    s3_fileid=s3_fileid, outcome="reject", stage=STAGE,
                    layer="G-1", reason=str(exc),
                )
            else:
                result = gateway.evaluate(s3_fileid, uri, size)
                verdict = FileVerdict(
                    s3_fileid=s3_fileid, outcome=result.outcome, stage=STAGE,
                    layer=result.layer, reason=result.reason,
                    category=result.category, confidence=result.confidence,
                    decision_id=decision_id(s3_fileid, STAGE),
                )
                with session.begin():
                    IntakeRepository(session).record(
                        s3_fileid=s3_fileid, stage=STAGE, outcome=result.outcome,
                        reason=result.reason, layer=result.layer,
                        category=result.category, confidence=result.confidence,
                        signals=result.signals, source_path=uri,
                    )

            response.verdicts.append(verdict)
            _bucket(response, verdict)

        if response.accepted:
            forwarded = _forward_to_quality_gate(
                request.request_id, request.subset(response.accepted)
            )
            response.forwarded = forwarded.get("forwarded", False)
            response.forward_error = forwarded.get("error")
            # Quality Gate мог отклонить что-то из принятого здесь — его
            # вердикты дополняют наши, а не заменяют их.
            response.verdicts.extend(forwarded.get("verdicts", []))
            _reconcile(response, forwarded)

        logger.info(
            "Data Gateway %s: принято %d, карантин %d, отклонено %d",
            request.request_id, len(response.accepted),
            len(response.quarantined), len(response.rejected),
        )
        return response
    finally:
        session.close()


def _bucket(response: IntakeResponse, verdict: FileVerdict) -> None:
    """Раскладка вердикта по спискам. Неизвестный исход — в карантин."""
    buckets = {
        "accept": response.accepted,
        "quarantine": response.quarantined,
        "reject": response.rejected,
    }
    bucket = buckets.get(verdict.outcome)
    if bucket is None:
        logger.error(
            "Неизвестный исход %r по файлу %s — считаем карантином",
            verdict.outcome, verdict.s3_fileid,
        )
        bucket = response.quarantined
    bucket.append(verdict.s3_fileid)


def _forward_to_quality_gate(request_id: str, files: List[IntakeFile]) -> dict:
    url = f"{settings.QUALITY_GATE_ENDPOINT.rstrip('/')}/internal/v1/quality"
    payload = {
        "request_id": request_id,
        "files": [item.model_dump() for item in files],
    }
    try:
        response = requests.post(url, json=payload, timeout=settings.INTAKE_TIMEOUT)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as exc:
        logger.error("Quality Gate недоступен: %s", exc)
        return {"forwarded": False, "error": f"Quality Gate недоступен: {exc}"}
    except ValueError as exc:
        return {"forwarded": False, "error": f"Quality Gate вернул не-JSON: {exc}"}

    try:
        verdicts = [FileVerdict(**v) for v in data.get("verdicts", [])]
    except Exception as exc:  # noqa: BLE001 — чужой ответ не обязан быть нашим
        logger.error("Quality Gate вернул вердикты не по контракту: %s", exc)
        return {"forwarded": False, "error": f"Quality Gate ответил не по контракту: {exc}"}

    return {
        "forwarded": bool(data.get("forwarded")),
        "error": data.get("forward_error"),
        "verdicts": verdicts,
        "accepted": data.get("accepted", []),
        "quarantined": data.get("quarantined", []),
        "rejected": data.get("rejected", []),
    }


def _reconcile(response: IntakeResponse, forwarded: dict) -> None:
    """Итоговые списки — по решению последнего этапа, который видел файл."""
    quarantined = set(forwarded.get("quarantined") or [])
    rejected = set(forwarded.get("rejected") or [])
    if not quarantined and not rejected:
        return
    response.accepted = [f for f in response.accepted
                         if f not in quarantined and f not in rejected]
    response.quarantined.extend(sorted(quarantined))
    response.rejected.extend(sorted(rejected))


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
