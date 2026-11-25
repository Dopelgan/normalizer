"""
Удаление документа из всех хранилищ Parser по `s3_fileid` (ACL-02).

Документ удалён в Backend и отозван из RAG — его текст не должен оставаться и
здесь: иначе повторный `create` того же ключа отдаёт сохранённый результат
«из БД», даже когда исходника уже нет. Удаляется:

- строки `documents` и `chunks` (каскадом и явно);
- решения приёма `intake_decisions` этого `s3_fileid`;
- копии результата в Redis: поле файла в `req:*:docs`; запрос, в котором
  больше ничего нет, — целиком (`core/result_aggregator.py`);
- картинки MinerU под `assets/<doc_id>/` (`ASSETS_PREFIX`);
- записи кэша лестницы (`core/ladder/result_cache.py::drop_for`) — только если
  другого живого документа с тем же `file_hash` нет: общие байты нужны ему.

Повторный вызов безопасен: чего уже нет, то пропускается, итог — `absent`.
Ручка — `DELETE /internal/v1/documents/{s3_fileid}` (`ingest/api.py`), флаг
`NORMALIZER_DELETE_ENABLED`; уборка сирот — `tools/purge_orphans.py`.
"""

import hashlib
import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, Optional

from core.config import settings
from core.db.models import ChunkDB, DocumentDB, IntakeDecisionDB

logger = logging.getLogger(__name__)


def doc_id_for(s3_fileid: str) -> str:
    """Тот же doc_id, что `ingest.tasks.generate_doc_id`: один файл — один doc_id."""
    digest = hashlib.sha256(s3_fileid.encode("utf-8")).hexdigest()
    return f"doc-{digest[:32]}"


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


def _drop_rows(session, s3_fileid: str, doc_id: str, report: DeleteReport) -> Optional[str]:
    """Строки БД. Возвращает file_hash удалённого документа и решает про кэш."""
    with session.begin():
        doc_ids = {doc_id}
        file_hash: Optional[str] = None
        for row in session.query(DocumentDB).filter(DocumentDB.s3_fileid == s3_fileid).all():
            doc_ids.add(row.id)
        document = session.get(DocumentDB, doc_id)
        if document is not None:
            file_hash = document.file_hash
        else:
            for other_id in doc_ids - {doc_id}:
                row = session.get(DocumentDB, other_id)
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
            if shared is not None:
                report.cache_kept_shared = True
    return file_hash


def _drop_redis(client, s3_fileid: str) -> int:
    """Поле файла в каждом `req:*:docs`; опустевший запрос — целиком."""
    removed = 0
    for key in client.scan_iter(match="req:*:docs", count=500):
        if not client.hdel(key, s3_fileid):
            continue
        removed += 1
        if client.hlen(key) == 0:
            request_id = key[len("req:"):-len(":docs")]
            client.delete(*(
                f"req:{request_id}:{suffix}"
                for suffix in ("meta", "docs", "start", "pub", "since")
            ))
    return removed


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

    file_hash: Optional[str] = None
    session = session_factory()
    try:
        file_hash = _drop_rows(session, s3_fileid, doc_id, report)
    except Exception as exc:  # noqa: BLE001
        logger.error("Удаление %s: строки БД не удалены: %s", doc_id, exc)
        report.errors["database"] = type(exc).__name__
    finally:
        session.close()

    try:
        report.redis_results = _drop_redis(redis_client, s3_fileid)
    except Exception as exc:  # noqa: BLE001
        logger.error("Удаление %s: копии в Redis не удалены: %s", doc_id, exc)
        report.errors["redis"] = type(exc).__name__

    prefix = f"{settings.ASSETS_PREFIX.rstrip('/')}/{doc_id}/"
    try:
        report.assets = storage.delete_prefix(prefix)
    except Exception as exc:  # noqa: BLE001
        logger.error("Удаление %s: картинки %s не удалены: %s", doc_id, prefix, exc)
        report.errors["assets"] = type(exc).__name__

    if "database" not in report.errors and not report.cache_kept_shared:
        try:
            from core.ladder import result_cache

            report.cache_entries = result_cache.drop_for(doc_id, file_hash)
        except Exception as exc:  # noqa: BLE001
            logger.error("Удаление %s: записи кэша лестницы не удалены: %s", doc_id, exc)
            report.errors["ladder_cache"] = type(exc).__name__

    logger.info(
        "Документ %s удалён: %s", doc_id,
        {k: v for k, v in report.as_dict().items() if k not in ("s3_fileid",)},
    )
    return report
