"""
HTTP-интерфейс Quality Gate.

    POST /internal/v1/quality   проверка пригодности пакета
    GET  /health

Принятые файлы отправляются в нормализатор. Отклонённые и карантинные туда
не попадают — в этом и смысл проверки.

ВНИМАНИЕ: передача в нормализатор сейчас закомментирована в `check()` —
цепочка приёма намеренно останавливается здесь, и `/internal/v1/parse/background`
не вызывается. `forwarded` в ответе остаётся `false`.

Тело запроса такое же, как у Data Gateway: список файлов, у каждого своя
операция. Дальше в нормализатор уходят только принятые — со своими
операциями, а не с одной на весь пакет.
"""

import logging
from typing import Dict, List, Tuple

import requests
from fastapi import FastAPI
from sqlalchemy import select

from core.config import settings
from core.db.models import IntakeDecisionDB
from core.db.session import SessionLocal
from core.models.intake import FileVerdict, IntakeFile, IntakeRequest, IntakeResponse
from core.providers.file_locator import FileLocator, SourceFileNotFound
from core.providers.storage import StorageProviderFactory
from core.quality.service import QualityGate
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
    title="Quality Gate",
    description="Техническая пригодность документа перед нормализацией",
    version="1.0.0",
)

STAGE = "quality_gate"


def _known(session) -> Tuple[Dict[str, str], List[Tuple[str, int]]]:
    """Контрольные суммы и отпечатки уже принятого — для поиска дублей."""
    rows = session.execute(
        select(IntakeDecisionDB)
        .where(IntakeDecisionDB.outcome == "accept")
        .order_by(IntakeDecisionDB.created_at.desc())
        .limit(1000)
    ).scalars()
    hashes = {row.file_hash: row.s3_fileid for row in rows if row.file_hash}
    return hashes, IntakeRepository(session).fingerprints()


@app.post("/internal/v1/quality", response_model=IntakeResponse)
def check(request: IntakeRequest) -> IntakeResponse:
    storage = StorageProviderFactory.default()
    locator = FileLocator(storage)
    session = SessionLocal()
    response = IntakeResponse(request_id=request.request_id)

    try:
        # Чтение тоже идёт в транзакции и ею же закрывается. Иначе SQLAlchemy
        # открывает транзакцию сам на первом же SELECT, и следующий
        # `with session.begin()` падает с «A transaction is already begun»,
        # превращая любой запрос с найденным файлом в 500.
        with session.begin():
            known_hashes, known_fingerprints = _known(session)

        gate = QualityGate(
            storage=storage,
            known_hashes=known_hashes,
            known_fingerprints=known_fingerprints,
        )

        for item in request.files:
            s3_fileid = item.s3_fileid
            try:
                located = locator.locate(s3_fileid)
                uri = located.uri
            except SourceFileNotFound as exc:
                verdict = _reject(s3_fileid, str(exc))
            else:
                result = gate.evaluate(s3_fileid, uri)
                verdict = FileVerdict(
                    s3_fileid=s3_fileid,
                    outcome=result.outcome,
                    stage=STAGE,
                    layer=result.stage,
                    reason=result.reason,
                    confidence=result.confidence,
                    warnings=result.warnings,
                    routing=result.routing,
                    decision_id=decision_id(s3_fileid, STAGE),
                )
                with session.begin():
                    IntakeRepository(session).record(
                        s3_fileid=s3_fileid, stage=STAGE, outcome=result.outcome,
                        reason=result.reason, layer=result.stage,
                        confidence=result.confidence,
                        signals={**result.signals, "warnings": result.warnings,
                                 "routing": result.routing},
                        source_path=uri, file_hash=result.file_hash,
                    )
                # Принятый файл становится известным сразу: два одинаковых
                # файла в одном пакете иначе проходят оба — снимок дублей
                # снят до начала разбора.
                if result.outcome == "accept" and result.file_hash:
                    known_hashes.setdefault(result.file_hash, s3_fileid)
                    gate.known_hashes = known_hashes

            response.verdicts.append(verdict)
            _bucket(response, verdict)

        # Передача принятых файлов в нормализатор временно отключена:
        # обработка останавливается на Quality Gate и до
        # `/internal/v1/parse/background` не доходит. Сам `_forward` и его
        # настройки оставлены нетронутыми — чтобы вернуть цепочку, достаточно
        # раскомментировать эти три строки.
        # if response.accepted:
        #     response.forwarded, response.forward_error = _forward(
        #         request.request_id, request.subset(response.accepted)
        #     )

        logger.info(
            "Quality Gate %s: принято %d, карантин %d, отклонено %d",
            request.request_id, len(response.accepted),
            len(response.quarantined), len(response.rejected),
        )
        return response
    finally:
        session.close()


def _reject(s3_fileid: str, reason: str) -> FileVerdict:
    return FileVerdict(
        s3_fileid=s3_fileid, outcome="reject", stage=STAGE, layer="QG-1", reason=reason
    )


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


def _forward(request_id: str, files: List[IntakeFile]):
    """
    Только утверждённые файлы уходят в нормализатор.

    Сейчас не вызывается: вызов в `check()` закомментирован, обработка
    останавливается на Quality Gate. Функция оставлена рабочей.
    """
    url = f"{settings.PARSER_ENDPOINT.rstrip('/')}/internal/v1/parse/background"
    payload = {
        "request_id": request_id,
        "files": [item.model_dump() for item in files],
    }
    try:
        response = requests.post(url, json=payload, timeout=settings.INTAKE_TIMEOUT)
        response.raise_for_status()
        return True, None
    except requests.RequestException as exc:
        logger.error("Нормализатор не принял запрос %s: %s", request_id, exc)
        return False, str(exc)


@app.get("/health")
def health() -> dict:
    return {"status": "healthy", "service": "quality_gate"}
