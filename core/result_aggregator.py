"""
Состояние пакетного запроса в Redis.

Один запрос RAG может содержать несколько файлов, которые обрабатываются
разными задачами Celery параллельно. Чтобы не ловить гонку на «прочитал
JSON — дописал — записал обратно», результаты складываются в hash по одному
полю на файл: HSET атомарен, а число готовых документов — это HLEN.

Ключи:
    req:{request_id}:meta   строка JSON — event_id/dialog_id, операции, файлы
    req:{request_id}:docs   hash   s3_fileid -> JSON документа
    req:{request_id}:start  строка — счётчик начатых задач
    req:{request_id}:pub    флаг «результат уже опубликован» (SET NX)
    req:{request_id}:since  hash   s3_fileid -> время последнего старта задачи
"""

import hashlib
import json
import logging
import time
from typing import Any, Dict, List, Optional

import redis

from core.config import settings

logger = logging.getLogger(__name__)

_client: Optional[redis.Redis] = None


def get_client() -> redis.Redis:
    """Ленивое подключение — чтобы импорт модуля не требовал живого Redis."""
    global _client
    if _client is None:
        _client = redis.Redis.from_url(settings.EVENT_BROKER_URL, decode_responses=True)
    return _client


def set_client(client: redis.Redis) -> None:
    """Подмена клиента (используется в тестах с fakeredis)."""
    global _client
    _client = client


def _meta_key(request_id: str) -> str:
    return f"req:{request_id}:meta"


def _docs_key(request_id: str) -> str:
    return f"req:{request_id}:docs"


def _start_key(request_id: str) -> str:
    return f"req:{request_id}:start"


def _published_key(request_id: str) -> str:
    return f"req:{request_id}:pub"


def _since_key(request_id: str) -> str:
    return f"req:{request_id}:since"


# ===========================================================================
# Запись
# ===========================================================================

def init_request_state(
    request_id: str,
    s3_fileids: List[str],
    operation: Optional[str] = None,
    ttl: Optional[int] = None,
    operations: Optional[Dict[str, str]] = None,
    dialog_id: str = "",
    event_id: str = "",
) -> Dict[str, Any]:
    """
    Создаёт запись о запросе. Возвращает мету.

    `operation` — общая операция пакета, если она одна на все файлы;
    `operations` — операция по каждому файлу, когда они разные.

    `dialog_id` заполняется только синхронным разбором, `event_id` — только
    фоновым: в фоне диалога нет, и подставлять туда `event_id` под именем
    диалога, как делалось раньше, значит врать в обе стороны.
    """
    client = get_client()
    ttl = ttl or settings.RESULT_TTL_SECONDS
    meta = {
        "request_id": request_id,
        "dialog_id": dialog_id,
        "event_id": event_id,
        "operation": operation,
        "operations": dict(operations or {}),
        "s3_fileids": list(s3_fileids),
        "total": len(s3_fileids),
        "created_at": time.time(),
    }
    pipe = client.pipeline()
    pipe.delete(
        _docs_key(request_id), _start_key(request_id), _published_key(request_id),
        _since_key(request_id),
    )
    pipe.set(_meta_key(request_id), json.dumps(meta), ex=ttl)
    pipe.execute()
    return meta


def claim_publication(request_id: str) -> bool:
    """
    Право опубликовать результат пакета — ровно одному вызову.

    Задачи Celery завершаются параллельно и могут одновременно увидеть
    «обработаны все», а `acks_late` допускает ещё и повторное выполнение
    задачи. Без этого флага подписчик получает результат по несколько раз.
    """
    ok = get_client().set(
        _published_key(request_id), "1", nx=True, ex=settings.RESULT_TTL_SECONDS
    )
    return bool(ok)


def mark_started(request_id: str, s3_fileid: Optional[str] = None) -> None:
    """
    Отмечает, что очередная задача взялась за работу (queued -> processing).

    Время старта запоминается по файлу и обновляется при каждом запуске
    (повтор, ожидание чужого разбора): по нему видно, что задача пропала,
    не дойдя до finalize.
    """
    client = get_client()
    pipe = client.pipeline()
    pipe.incr(_start_key(request_id))
    pipe.expire(_start_key(request_id), settings.RESULT_TTL_SECONDS)
    if s3_fileid:
        pipe.hset(_since_key(request_id), s3_fileid, str(time.time()))
        pipe.expire(_since_key(request_id), settings.RESULT_TTL_SECONDS)
    pipe.execute()


def put_document_result(request_id: str, s3_fileid: str, document: Dict[str, Any]) -> int:
    """
    Кладёт результат по одному файлу. Возвращает число готовых документов.
    Повторный вызов для того же файла не увеличивает счётчик — задача
    Celery может выполниться дважды, и это не должно ломать агрегацию.
    """
    client = get_client()
    pipe = client.pipeline()
    pipe.hset(_docs_key(request_id), s3_fileid, json.dumps(document, default=str))
    pipe.expire(_docs_key(request_id), settings.RESULT_TTL_SECONDS)
    pipe.hlen(_docs_key(request_id))
    result = pipe.execute()
    return int(result[-1])


def delete_request_state(request_id: str) -> None:
    get_client().delete(
        _meta_key(request_id), _docs_key(request_id), _start_key(request_id),
        _published_key(request_id), _since_key(request_id),
    )


# ===========================================================================
# Чтение
# ===========================================================================

def get_meta(request_id: str) -> Optional[Dict[str, Any]]:
    raw = get_client().get(_meta_key(request_id))
    return json.loads(raw) if raw else None


def get_documents(request_id: str) -> Dict[str, Dict[str, Any]]:
    raw = get_client().hgetall(_docs_key(request_id)) or {}
    documents: Dict[str, Dict[str, Any]] = {}
    for fileid, payload in raw.items():
        try:
            documents[fileid] = json.loads(payload)
        except (TypeError, ValueError):
            logger.error("Битый JSON документа %s в запросе %s", fileid, request_id)
    return documents


def get_request_state(request_id: str) -> Optional[Dict[str, Any]]:
    """
    Полное состояние запроса:
    {request_id, dialog_id, event_id, operation, operations, total,
     processed, status, documents}
    """
    meta = get_meta(request_id)
    if meta is None:
        return None

    documents = get_documents(request_id)
    order = meta.get("s3_fileids") or list(documents)
    documents.update(_expire_lost_files(request_id, order, documents))
    ordered = [documents[f] for f in order if f in documents]
    # На случай, если файл вернулся под другим ключом.
    ordered += [d for f, d in documents.items() if f not in order]

    total = int(meta.get("total") or 0)
    processed = len(documents)
    started = get_client().get(_start_key(request_id))

    return {
        "request_id": meta.get("request_id", request_id),
        "dialog_id": meta.get("dialog_id", ""),
        "event_id": meta.get("event_id", ""),
        "operation": meta.get("operation"),
        "operations": meta.get("operations") or {},
        "total": total,
        "processed": processed,
        "status": compute_status(total, ordered, bool(started)),
        "documents": ordered,
    }


def parse_stale_after_seconds() -> int:
    """Дольше этого задача разбора жить не может: жёсткий лимит плюс запас."""
    return settings.PARSE_TASK_TIME_LIMIT + settings.PARSE_STALE_MARGIN_SECONDS


def _expire_lost_files(
    request_id: str, order: List[str], documents: Dict[str, Dict[str, Any]]
) -> Dict[str, Dict[str, Any]]:
    """
    Файлы, задача которых пропала: стартовала дольше жёсткого лимита назад
    и результата не оставила. Жёсткий лимит, OOM и SIGKILL до finalize не
    доходят, и без этого запрос висел «processing» до истечения TTL (сутки),
    а RAG всё это время его опрашивал. Таким файлам записывается ошибка
    PARSER_TIMEOUT (retryable): RAG повторяет запрос, повтор того же
    request_id с ошибкой запускает разбор заново.

    HSETNX: результат, который успел лечь параллельно, не затирается.
    """
    pending = [f for f in order if f not in documents]
    if not pending:
        return {}
    client = get_client()
    since = client.hmget(_since_key(request_id), pending)
    now = time.time()
    limit = parse_stale_after_seconds()
    expired: Dict[str, Dict[str, Any]] = {}
    for fileid, raw in zip(pending, since):
        try:
            started_at = float(raw) if raw else None
        except ValueError:
            started_at = None
        if started_at is None or now - started_at <= limit:
            continue
        document = _timeout_document(fileid, now - started_at)
        if client.hsetnx(_docs_key(request_id), fileid, json.dumps(document, default=str)):
            client.expire(_docs_key(request_id), settings.RESULT_TTL_SECONDS)
            logger.error(
                "Запрос %s: задача файла %s пропала (старт %.0f с назад, лимит %d с) — "
                "отдаём PARSER_TIMEOUT", request_id, fileid, now - started_at, limit,
            )
            expired[fileid] = document
        else:
            stored = client.hget(_docs_key(request_id), fileid)
            if stored:
                expired[fileid] = json.loads(stored)
    return expired


def _timeout_document(s3_fileid: str, elapsed: float) -> Dict[str, Any]:
    """Документ-ошибка по контракту для файла, задача которого пропала."""
    from core.models.contract import DocumentMetadata, DocumentResult, ParserError

    return DocumentResult(
        s3_fileid=s3_fileid,
        status="error",
        document_metadata=DocumentMetadata(
            # Та же формула, что ingest.tasks.generate_doc_id (импорт задач
            # в API тянет весь конвейер разбора).
            doc_id=f"doc-{hashlib.sha256(s3_fileid.encode('utf-8')).hexdigest()[:32]}",
            language=settings.DEFAULT_LANGUAGE,
            status="error",
        ),
        fragments=[],
        error=ParserError(
            reason_code="PARSER_TIMEOUT",
            message=(
                f"Задача разбора не вернула результат за {int(elapsed)} с "
                f"(лимит {settings.PARSE_TASK_TIME_LIMIT} с): воркер остановлен "
                "или убит. Повторите запрос."
            ),
            retryable=True,
        ),
    ).model_dump(mode="json")


def compute_status(total: int, documents: List[Dict[str, Any]], started: bool) -> str:
    """queued | processing | completed | partial | error по контракту."""
    processed = len(documents)
    if processed == 0:
        return "processing" if started else "queued"
    if processed < total:
        return "processing"

    failed = sum(1 for d in documents if _is_failed(d))
    if failed == processed:
        return "error"
    if failed or any(_is_partial(d) for d in documents):
        return "partial"
    return "completed"


def _is_failed(document: Dict[str, Any]) -> bool:
    if document.get("error"):
        return True
    if document.get("status") == "error":
        return True
    metadata = document.get("document_metadata") or {}
    return metadata.get("status") == "error"


def _is_partial(document: Dict[str, Any]) -> bool:
    """Файл разобран не целиком: часть страниц не прочиталась."""
    return bool(document.get("status") == "partial" or document.get("failed_pages"))
