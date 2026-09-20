"""
Состояние приёма в Redis: что поставлено в очередь и чем закончилось.

Устроено так же, как состояние разбора (`core.result_aggregator`), и по той
же причине: файлы одного запроса обрабатываются разными задачами
параллельно, поэтому результат складывается в hash по одному полю на файл.
HSET атомарен, а число готовых файлов — это HLEN, и гонки «прочитал JSON,
дописал, записал обратно» не возникает.

Ключи:
    intake:{request_id}:meta    строка JSON — список файлов и операции
    intake:{request_id}:files   hash   s3_fileid -> JSON итога по файлу
    intake:{request_id}:start   счётчик начатых задач
    intake:hash:{sha256}        заявка на содержимое (см. claim_content)
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional

from core.config import settings
from core.result_aggregator import get_client

logger = logging.getLogger(__name__)

STATUS_QUEUED = "queued"
STATUS_PROCESSING = "processing"
STATUS_COMPLETED = "completed"


def _meta_key(request_id: str) -> str:
    return f"intake:{request_id}:meta"


def _files_key(request_id: str) -> str:
    return f"intake:{request_id}:files"


def _start_key(request_id: str) -> str:
    return f"intake:{request_id}:start"


def _hash_key(file_hash: str) -> str:
    return f"intake:hash:{file_hash}"


# ===========================================================================
# Запись
# ===========================================================================

def init_request(
    request_id: str,
    files: List[Dict[str, str]],
    ttl: Optional[int] = None,
) -> Dict[str, Any]:
    """Создаёт запись о приёме пакета и возвращает её мету."""
    client = get_client()
    ttl = ttl or settings.INTAKE_STATE_TTL_SECONDS
    meta = {
        "request_id": request_id,
        "files": list(files),
        "s3_fileids": [item["s3_fileid"] for item in files],
        "total": len(files),
        "created_at": time.time(),
    }
    pipe = client.pipeline()
    pipe.delete(_files_key(request_id), _start_key(request_id))
    pipe.set(_meta_key(request_id), json.dumps(meta, ensure_ascii=False), ex=ttl)
    pipe.execute()
    return meta


def mark_started(request_id: str) -> None:
    """Очередная задача взялась за работу: queued -> processing."""
    client = get_client()
    pipe = client.pipeline()
    pipe.incr(_start_key(request_id))
    pipe.expire(_start_key(request_id), settings.INTAKE_STATE_TTL_SECONDS)
    pipe.execute()


def put_file_result(request_id: str, s3_fileid: str, result: Dict[str, Any]) -> int:
    """
    Кладёт итог по одному файлу. Возвращает число готовых файлов.
    Повторное выполнение задачи Celery счётчик не увеличивает.
    """
    client = get_client()
    pipe = client.pipeline()
    pipe.hset(
        _files_key(request_id), s3_fileid,
        json.dumps(result, ensure_ascii=False, default=str),
    )
    pipe.expire(_files_key(request_id), settings.INTAKE_STATE_TTL_SECONDS)
    pipe.hlen(_files_key(request_id))
    return int(pipe.execute()[-1])


def claim_content(file_hash: str, s3_fileid: str) -> Optional[str]:
    """
    Заявка на содержимое: кто первым объявил этот хеш своим.

    Возвращает `None`, если заявка наша, и чужой s3_fileid, если файл с
    таким содержимым уже принят. Нужна против гонки: два одинаковых файла
    одного пакета обрабатываются параллельно, и проверка по базе оба
    пропускает — записи друг друга они ещё не видят.
    """
    if not file_hash:
        return None
    client = get_client()
    key = _hash_key(file_hash)
    if client.set(key, s3_fileid, nx=True, ex=settings.INTAKE_STATE_TTL_SECONDS):
        return None
    owner = client.get(key)
    return owner if owner and owner != s3_fileid else None


def release_content(file_hash: str, s3_fileid: str) -> None:
    """Снять свою заявку: файл в итоге не принят, и хеш держать незачем."""
    if not file_hash:
        return
    client = get_client()
    key = _hash_key(file_hash)
    if client.get(key) == s3_fileid:
        client.delete(key)


def delete_request(request_id: str) -> None:
    get_client().delete(
        _meta_key(request_id), _files_key(request_id), _start_key(request_id)
    )


# ===========================================================================
# Чтение
# ===========================================================================

def get_meta(request_id: str) -> Optional[Dict[str, Any]]:
    raw = get_client().get(_meta_key(request_id))
    return json.loads(raw) if raw else None


def get_files(request_id: str) -> Dict[str, Dict[str, Any]]:
    raw = get_client().hgetall(_files_key(request_id)) or {}
    results: Dict[str, Dict[str, Any]] = {}
    for fileid, payload in raw.items():
        try:
            results[fileid] = json.loads(payload)
        except (TypeError, ValueError):
            logger.error("Битый JSON итога %s в приёме %s", fileid, request_id)
    return results


def get_state(request_id: str) -> Optional[Dict[str, Any]]:
    """
    Полное состояние приёма: что поставлено, что готово, чем кончилось.

    Списки принятого, карантина и отклонённого собираются здесь, а не
    хранятся отдельно: иначе они расходятся с вердиктами по файлам, и
    ответ противоречит сам себе.
    """
    meta = get_meta(request_id)
    if meta is None:
        return None

    results = get_files(request_id)
    order = meta.get("s3_fileids") or list(results)
    ordered = [results[f] for f in order if f in results]
    ordered += [r for f, r in results.items() if f not in order]

    total = int(meta.get("total") or 0)
    processed = len(results)
    started = get_client().get(_start_key(request_id))

    buckets: Dict[str, List[str]] = {"accept": [], "quarantine": [], "reject": []}
    verdicts: List[Dict[str, Any]] = []
    forwarded = False
    forward_error: Optional[str] = None
    for item in ordered:
        bucket = buckets.get(item.get("outcome") or "quarantine")
        if bucket is None:
            bucket = buckets["quarantine"]
        bucket.append(item["s3_fileid"])
        verdicts.extend(item.get("verdicts") or [])
        forwarded = forwarded or bool(item.get("forwarded"))
        forward_error = forward_error or item.get("forward_error")

    return {
        "request_id": meta.get("request_id", request_id),
        "total": total,
        "processed": processed,
        "status": compute_status(total, processed, bool(started)),
        "files": ordered,
        "accepted": buckets["accept"],
        "quarantined": buckets["quarantine"],
        "rejected": buckets["reject"],
        "verdicts": verdicts,
        "forwarded": forwarded,
        "forward_error": forward_error,
    }


def compute_status(total: int, processed: int, started: bool) -> str:
    if total and processed >= total:
        return STATUS_COMPLETED
    if processed or started:
        return STATUS_PROCESSING
    return STATUS_QUEUED


def is_active(request_id: str) -> bool:
    """Идёт ли ещё приём по этому request_id."""
    state = get_state(request_id)
    return bool(state and state["status"] != STATUS_COMPLETED)
