"""
HTTP-интерфейс Parser по контракту RAG <-> Parser.

    POST /internal/v1/parse                      синхронная обработка
    POST /internal/v1/parse/background           фоновая обработка (202)
    GET  /internal/v1/parse/results/{request_id} готовность и результат
    GET  /health

Ручки рассчитаны на внутреннюю сеть контейнеров: авторизации в MVP-контракте
нет. Наружу порты публиковать не следует.
"""

import asyncio
import logging
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from core.config import settings
from core.models.api import BackgroundParseRequest, HealthResponse, ParseRequest
from core.models.intake import IntakeFile
from core.models.contract import (
    BackgroundAcceptedResponse,
    DocumentResult,
    ParseResponse,
    ResultsResponse,
)
from core.result_aggregator import get_request_state, init_request_state
from core.workspace import configure_process_tempdir

# Временные файлы процесса — в том сервиса, а не на слой контейнера.
configure_process_tempdir()

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Parser API",
    description="Нормализация документов: текст, таблицы, формулы, чертежи",
    version="1.0.0",
)

SYNC_POLL_INTERVAL = 0.5
ACTIVE_STATUSES = ("queued", "processing")


# ===========================================================================
# Вспомогательное
# ===========================================================================

def _dispatch(request_id: str, dialog_id: str, items: List[IntakeFile]) -> None:
    """Ставит по задаче на каждый файл пакета — со своей операцией."""
    from ingest.tasks import process_document

    for item in items:
        task = process_document.delay(
            request_id=request_id,
            dialog_id=dialog_id,
            s3_fileid=item.s3_fileid,
            operation=item.operation,
        )
        logger.info(
            "Задача %s поставлена для файла %s (%s)",
            task.id, item.s3_fileid, item.operation,
        )


def _reject_if_active(request_id: str) -> None:
    """Повторное использование request_id незавершённого запроса — конфликт."""
    state = get_request_state(request_id)
    if state and state["status"] in ACTIVE_STATUSES:
        raise HTTPException(
            status_code=409,
            detail=f"Запрос {request_id} уже выполняется. Используйте новый request_id.",
        )


def _documents(state: Dict[str, Any]) -> List[DocumentResult]:
    documents: List[DocumentResult] = []
    for raw in state.get("documents", []):
        try:
            documents.append(DocumentResult(**raw))
        except Exception as exc:  # noqa: BLE001
            logger.error("Документ в состоянии запроса не проходит валидацию: %s", exc)
    return documents


# ===========================================================================
# Ручки контракта
# ===========================================================================

@app.post("/internal/v1/parse", response_model=ParseResponse)
async def parse_sync(request: ParseRequest):
    """
    Синхронная обработка: ответ отдаётся, когда готовы все файлы пакета.
    Если обработка не уложилась в PROCESSING_TIMEOUT_SECONDS, возвращается
    504 с текущим состоянием — результат при этом никуда не пропадает и
    остаётся доступен через /internal/v1/parse/results/{request_id}.
    """
    _reject_if_active(request.request_id)

    items = [IntakeFile(s3_fileid=fileid) for fileid in request.s3_fileid]
    await asyncio.to_thread(
        init_request_state,
        request.request_id, request.dialog_id, request.s3_fileid, "create",
    )
    await asyncio.to_thread(_dispatch, request.request_id, request.dialog_id, items)

    deadline = asyncio.get_event_loop().time() + settings.PROCESSING_TIMEOUT_SECONDS
    while True:
        state = await asyncio.to_thread(get_request_state, request.request_id)
        if state and state["status"] not in ACTIVE_STATUSES:
            return ParseResponse(
                request_id=request.request_id,
                dialog_id=request.dialog_id,
                documents=_documents(state),
            )
        if asyncio.get_event_loop().time() >= deadline:
            logger.warning("Синхронный запрос %s не уложился в таймаут", request.request_id)
            partial = ResultsResponse(
                request_id=request.request_id,
                dialog_id=request.dialog_id,
                status="processing",
                documents=_documents(state) if state else [],
                error="Обработка не завершилась в отведённое время, "
                      "результат доступен через /internal/v1/parse/results/{request_id}",
            )
            return JSONResponse(status_code=504, content=partial.model_dump(mode="json"))
        await asyncio.sleep(SYNC_POLL_INTERVAL)


@app.post("/internal/v1/parse/background", response_model=BackgroundAcceptedResponse, status_code=202)
async def parse_background(request: BackgroundParseRequest):
    """Фоновая обработка: сразу 202, результат забирается поллингом."""
    _reject_if_active(request.request_id)

    items = request.items
    operations = {item.s3_fileid: item.operation for item in items}
    # Операция пакета осталась в контракте одним полем: когда файлы просят
    # разное, общей операции у пакета нет, и выдумывать её нельзя.
    common = next(iter(set(operations.values()))) if len(set(operations.values())) == 1 else None

    await asyncio.to_thread(
        init_request_state,
        request.request_id, request.dialog_id,
        [item.s3_fileid for item in items], common, operations=operations,
    )
    await asyncio.to_thread(_dispatch, request.request_id, request.dialog_id, items)

    return BackgroundAcceptedResponse(
        request_id=request.request_id,
        dialog_id=request.dialog_id,
        operation=common,
        accepted=True,
        status="queued",
    )


@app.get("/internal/v1/parse/results/{request_id}", response_model=ResultsResponse)
async def parse_results(request_id: str):
    """Статус и результат фоновой задачи."""
    state = await asyncio.to_thread(get_request_state, request_id)
    if state is None:
        raise HTTPException(
            status_code=404,
            detail=f"Запрос {request_id} не найден: он не создавался или истёк срок хранения.",
        )

    return ResultsResponse(
        request_id=state["request_id"],
        dialog_id=state["dialog_id"],
        operation=state.get("operation"),
        status=state["status"],
        documents=_documents(state),
        error=_first_error(state) if state["status"] in ("error", "partial") else None,
    )


def _first_error(state: Dict[str, Any]) -> str:
    for raw in state.get("documents", []):
        if raw.get("error"):
            return str(raw["error"])
        metadata = raw.get("document_metadata") or {}
        if metadata.get("status") == "error":
            return f"Документ {metadata.get('doc_id')} обработан с ошибкой"
    return "Часть документов обработана с ошибкой"


# ===========================================================================
# Служебное
# ===========================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Проверяет зависимости, а не только то, что процесс жив."""
    def _probe() -> Dict[str, bool]:
        checks = {"redis": False, "database": False}
        try:
            from core.result_aggregator import get_client
            get_client().ping()
            checks["redis"] = True
        except Exception as exc:  # noqa: BLE001
            logger.warning("Redis недоступен: %s", exc)
        try:
            from sqlalchemy import text
            from core.db.session import engine
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            checks["database"] = True
        except Exception as exc:  # noqa: BLE001
            logger.warning("БД недоступна: %s", exc)
        return checks

    dependencies = await asyncio.to_thread(_probe)
    status = "ok" if all(dependencies.values()) else "degraded"
    return HealthResponse(status=status, service="parser", dependencies=dependencies)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    # errors() кладёт в ctx само исключение, а json.dumps его не умеет:
    # без jsonable_encoder любая ошибка своего валидатора превращала
    # честные 422 в 500.
    errors = jsonable_encoder(exc.errors())
    logger.warning("Ошибка валидации %s: %s", request.url.path, errors)
    return JSONResponse(
        status_code=422,
        content={"detail": "Validation error", "errors": errors},
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error("Необработанная ошибка на %s: %s", request.url.path, exc, exc_info=True)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run("ingest.api:app", host="0.0.0.0", port=settings.API_PORT, reload=True)
