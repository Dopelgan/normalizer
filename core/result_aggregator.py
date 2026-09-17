"""
Состояние пакетного запроса в Redis.

Один запрос RAG может содержать несколько файлов, которые обрабатываются
разными задачами Celery параллельно. Чтобы не ловить гонку на «прочитал
JSON — дописал — записал обратно», результаты складываются в hash по одному
полю на файл: HSET атомарен, а число готовых документов — это HLEN.

Ключи:
    req:{request_id}:meta   строка JSON — dialog_id, операции, список файлов
    req:{request_id}:docs   hash   s3_fileid -> JSON документа
    req:{request_id}:start  строка — счётчик начатых задач
    req:{request_id}:pub    флаг «результат уже опубликован» (SET NX)
"""

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


# ===========================================================================
# Запись
# ===========================================================================

def init_request_state(
    request_id: str,
    dialog_id: str,
    s3_fileids: List[str],
    operation: Optional[str] = None,
    ttl: Optional[int] = None,
    operations: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Создаёт запись о запросе. Возвращает мету.

    `operation` — общая операция пакета, если она одна на все файлы;
    `operations` — операция по каждому файлу, когда они разные.
    """
    client = get_client()
    ttl = ttl or settings.RESULT_TTL_SECONDS
    meta = {
        "request_id": request_id,
        "dialog_id": dialog_id,
        "operation": operation,
        "operations": dict(operations or {}),
        "s3_fileids": list(s3_fileids),
        "total": len(s3_fileids),
        "created_at": time.time(),
    }
    pipe = client.pipeline()
    pipe.delete(_docs_key(request_id), _start_key(request_id), _published_key(request_id))
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


def mark_started(request_id: str) -> None:
    """Отмечает, что очередная задача взялась за работу (queued -> processing)."""
    client = get_client()
    pipe = client.pipeline()
    pipe.incr(_start_key(request_id))
    pipe.expire(_start_key(request_id), settings.RESULT_TTL_SECONDS)
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
        _published_key(request_id),
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
    {request_id, dialog_id, operation, operations, total, processed,
     status, documents}
    """
    meta = get_meta(request_id)
    if meta is None:
        return None

    documents = get_documents(request_id)
    order = meta.get("s3_fileids") or list(documents)
    ordered = [documents[f] for f in order if f in documents]
    # На случай, если файл вернулся под другим ключом.
    ordered += [d for f, d in documents.items() if f not in order]

    total = int(meta.get("total") or 0)
    processed = len(documents)
    started = get_client().get(_start_key(request_id))

    return {
        "request_id": meta.get("request_id", request_id),
        "dialog_id": meta.get("dialog_id", ""),
        "operation": meta.get("operation"),
        "operations": meta.get("operations") or {},
        "total": total,
        "processed": processed,
        "status": compute_status(total, ordered, bool(started)),
        "documents": ordered,
    }


def compute_status(total: int, documents: List[Dict[str, Any]], started: bool) -> str:
    """queued | processing | completed | partial | error по контракту."""
    processed = len(documents)
    if processed == 0:
        return "processing" if started else "queued"
    if processed < total:
        return "processing"

    failed = sum(1 for d in documents if _is_failed(d))
    if failed == 0:
        return "completed"
    if failed == processed:
        return "error"
    return "partial"


def _is_failed(document: Dict[str, Any]) -> bool:
    if document.get("error"):
        return True
    metadata = document.get("document_metadata") or {}
    return metadata.get("status") == "error"
