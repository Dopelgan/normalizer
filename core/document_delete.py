"""
Удаление документа из всех хранилищ Parser по `s3_fileid` (ACL-02).

Документ удалён в Backend и отозван из RAG — его текст не должен оставаться и
здесь: иначе повторный `create` того же ключа отдаёт сохранённый результат
«из БД», даже когда исходника уже нет. Порядок:

1. надгробие `deleted:<doc_id>` в Redis (время удаления, срок
   `NORMALIZER_DELETE_TOMBSTONE_SECONDS`): разбор того же файла, начатый или
   поставленный в очередь до удаления, свой результат не сохраняет
   (`ingest/tasks.py` → `tombstoned`, `discard_tombstoned`);
2. строки `documents` и `chunks` (каскадом и явно) — по doc_id и по всем
   строкам с тем же `s3_fileid`; решения приёма `intake_decisions`;
3. копии результата в Redis: поле файла в `req:*:docs` заменяется документом
   с ошибкой без содержимого — счётчики запроса и его мета не трогаются;
4. картинки MinerU под `assets/<id>/` каждой удалённой строки (`ASSETS_PREFIX`);
5. записи кэша лестницы (`core/ladder/result_cache.py::drop_for`): записанные
   этим документом — всегда; записи тех же байтов — только если другого
   живого документа с тем же `file_hash` нет.

Повторный вызов безопасен: чего уже нет, то пропускается, итог — `absent`.
Ручка — `DELETE /internal/v1/documents/{s3_fileid}` (`ingest/api.py`), флаг
`NORMALIZER_DELETE_ENABLED`; уборка сирот — `tools/purge_orphans.py`.
"""

import hashlib
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from core.config import settings
from core.db.models import ChunkDB, DocumentDB, IntakeDecisionDB

logger = logging.getLogger(__name__)

DELETED_MESSAGE = "Документ удалён: результат разбора не сохраняется"


def doc_id_for(s3_fileid: str) -> str:
    """Тот же doc_id, что `ingest.tasks.generate_doc_id`: один файл — один doc_id."""
    digest = hashlib.sha256(s3_fileid.encode("utf-8")).hexdigest()
    return f"doc-{digest[:32]}"


# ------------------------------------------------------------------ надгробие
def _tombstone_key(doc_id: str) -> str:
    return f"deleted:{doc_id}"


def write_tombstone(client, doc_id: str, now: Optional[float] = None) -> float:
    stamp = time.time() if now is None else now
    client.set(
        _tombstone_key(doc_id), repr(stamp),
        ex=max(1, int(settings.NORMALIZER_DELETE_TOMBSTONE_SECONDS)),
    )
    return stamp


def tombstoned(client, doc_id: str, started_at: float) -> bool:
    """Документ удалён после начала разбора (или постановки запроса в очередь)."""
    raw = client.get(_tombstone_key(doc_id))
    if raw is None:
        return False
    try:
        return float(raw) >= started_at
    except (TypeError, ValueError):
        return True


def parse_started_at(client, request_id: str, task_started: float) -> float:
    """
    Начало разбора для сравнения с надгробием: создание запроса, если оно
    раньше старта задачи. Задача из очереди, поставленная до удаления,
    стартует уже после него — сравнивать надо со временем запроса.
    """
    try:
        meta = json.loads(client.get(f"req:{request_id}:meta") or "{}")
        created = float(meta.get("created_at") or task_started)
    except (TypeError, ValueError):
        created = task_started
    return min(created, task_started)


def deleted_document(s3_fileid: str, doc_id: str) -> Dict[str, Any]:
    """Документ с ошибкой без содержимого — на место результата удалённого файла."""
    from datetime import datetime, timezone

    return {
        "s3_fileid": s3_fileid,
        "status": "error",
        "document_metadata": {
            "doc_id": doc_id,
            "language": settings.DEFAULT_LANGUAGE,
            "source_path": "",
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "status": "error",
        },
        "fragments": [],
        "error": {"reason_code": "SOURCE_NOT_FOUND", "message": DELETED_MESSAGE,
                  "retryable": False},
    }


# ------------------------------------------------------------------- отчёт
@dataclass
class DeleteReport:
    s3_fileid: str
    doc_id: str
    documents: int = 0
    chunks: int = 0
    intake_decisions: int = 0
    redis_results: int = 0
    assets: int = 0
    cache_entries: int = 0
    cache_kept_shared: bool = False
    errors: Dict[str, str] = field(default_factory=dict)
    failed_keys: List[str] = field(default_factory=list)

    @property
    def removed_anything(self) -> bool:
        return any((
            self.documents, self.chunks, self.intake_decisions,
            self.redis_results, self.assets, self.cache_entries,
        ))

    @property
    def status(self) -> str:
        if self.errors:
            return "partial"
        return "deleted" if self.removed_anything else "absent"

    def as_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["status"] = self.status
        return data


def _drop_rows(session, s3_fileid: str, doc_id: str,
               report: DeleteReport) -> Tuple[Set[str], Optional[str]]:
    """Строки БД. Возвращает удалённые doc_id и file_hash документа."""
    with session.begin():
        doc_ids = {doc_id}
        file_hash: Optional[str] = None
        rows = session.query(DocumentDB).filter(DocumentDB.s3_fileid == s3_fileid).all()
        doc_ids.update(row.id for row in rows)
        own = session.get(DocumentDB, doc_id)
        for row in [own] + rows:
            if row is not None and row.file_hash:
                file_hash = row.file_hash
                break

        report.chunks = (
            session.query(ChunkDB)
            .filter(ChunkDB.doc_id.in_(doc_ids))
            .delete(synchronize_session=False)
        )
        report.documents = (
            session.query(DocumentDB)
            .filter(DocumentDB.id.in_(doc_ids))
            .delete(synchronize_session=False)
        )
        report.intake_decisions = (
            session.query(IntakeDecisionDB)
            .filter(IntakeDecisionDB.s3_fileid == s3_fileid)
            .delete(synchronize_session=False)
        )
        if file_hash:
            shared = (
                session.query(DocumentDB.id)
                .filter(DocumentDB.file_hash == file_hash)
                .filter(DocumentDB.id.notin_(doc_ids))
                .first()
            )
            report.cache_kept_shared = shared is not None
    return doc_ids, file_hash


def scrub_redis(client, s3_fileid: str, doc_id: str) -> int:
    """
    Результат файла в каждом `req:*:docs` заменяется документом без
    содержимого. Только поле этого файла: счётчики, мета и другие файлы
    запроса не трогаются — запрос, который ещё собирается, не ломается.
    """
    replaced = 0
    stub = json.dumps(deleted_document(s3_fileid, doc_id), ensure_ascii=False)
    for key in client.scan_iter(match="req:*:docs", count=500):
        current = client.hget(key, s3_fileid)
        if current is None or DELETED_MESSAGE in current:
            continue  # поля нет или оно уже заменено — повтор ничего не меняет
        client.hset(key, s3_fileid, stub)
        replaced += 1
    return replaced


def drop_assets(storage, doc_ids, report: DeleteReport) -> None:
    for one in sorted(doc_ids):
        prefix = f"{settings.ASSETS_PREFIX.rstrip('/')}/{one}/"
        try:
            report.assets += storage.delete_prefix(prefix)
        except Exception as exc:  # noqa: BLE001
            logger.error("Удаление %s: картинки %s не удалены: %s", one, prefix, exc)
            report.errors["assets"] = type(exc).__name__
            report.failed_keys.extend(getattr(exc, "failed_keys", []) or [prefix])


def delete_document(
    s3_fileid: str,
    *,
    session_factory: Callable[[], Any],
    redis_client: Any,
    storage: Any,
) -> DeleteReport:
    """
    Удалить документ отовсюду. Каждое хранилище — отдельный шаг: сбой одного
    не останавливает остальные и попадает в `errors` (итог `partial`, повтор
    безопасен).
    """
    doc_id = doc_id_for(s3_fileid)
    report = DeleteReport(s3_fileid=s3_fileid, doc_id=doc_id)

    # Надгробие первым: разбор, который сохранится после этой точки, его увидит.
    try:
        write_tombstone(redis_client, doc_id)
    except Exception as exc:  # noqa: BLE001
        logger.error("Удаление %s: надгробие не записано: %s", doc_id, exc)
        report.errors["tombstone"] = type(exc).__name__

    doc_ids: Set[str] = {doc_id}
    file_hash: Optional[str] = None
    session = session_factory()
    try:
        doc_ids, file_hash = _drop_rows(session, s3_fileid, doc_id, report)
    except Exception as exc:  # noqa: BLE001
        logger.error("Удаление %s: строки БД не удалены: %s", doc_id, exc)
        report.errors["database"] = type(exc).__name__
    finally:
        session.close()

    try:
        report.redis_results = scrub_redis(redis_client, s3_fileid, doc_id)
    except Exception as exc:  # noqa: BLE001
        logger.error("Удаление %s: копии в Redis не удалены: %s", doc_id, exc)
        report.errors["redis"] = type(exc).__name__

    drop_assets(storage, doc_ids, report)

    if "database" not in report.errors:
        try:
            from core.ladder import result_cache

            # Общие байты нужны другому документу: его запись по хешу остаётся,
            # записи этого документа (по doc_id) уходят всё равно.
            shared_hash = None if report.cache_kept_shared else file_hash
            for one in sorted(doc_ids):
                report.cache_entries += result_cache.drop_for(one, shared_hash)
                shared_hash = None
        except Exception as exc:  # noqa: BLE001
            logger.error("Удаление %s: записи кэша лестницы не удалены: %s", doc_id, exc)
            report.errors["ladder_cache"] = type(exc).__name__

    logger.info(
        "Документ %s удалён: %s", doc_id,
        {k: v for k, v in report.as_dict().items() if k != "s3_fileid"},
    )
    return report


def discard_tombstoned(s3_fileid: str, doc_id: str, *, storage: Any,
                       session_factory: Callable[[], Any]) -> None:
    """
    Разбор закончился после удаления документа: убрать то, что он успел
    оставить до сохранения, — строку `pending`, картинки и записи кэша по doc_id.
    """
    report = DeleteReport(s3_fileid=s3_fileid, doc_id=doc_id)
    session = session_factory()
    try:
        with session.begin():
            session.query(ChunkDB).filter(ChunkDB.doc_id == doc_id).delete(
                synchronize_session=False)
            session.query(DocumentDB).filter(DocumentDB.id == doc_id).delete(
                synchronize_session=False)
    except Exception as exc:  # noqa: BLE001
        logger.error("Отброс %s: строка БД не удалена: %s", doc_id, exc)
    finally:
        session.close()
    drop_assets(storage, {doc_id}, report)
    try:
        from core.ladder import result_cache

        result_cache.drop_for(doc_id, None)
    except Exception as exc:  # noqa: BLE001
        logger.error("Отброс %s: записи кэша не удалены: %s", doc_id, exc)
