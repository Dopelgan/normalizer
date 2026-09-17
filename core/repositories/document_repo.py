"""Репозиторий документов."""

from typing import Any, Dict, Optional

from sqlalchemy.orm import Session

from core.db.models import DocumentDB, utcnow


class DocumentRepository:
    """CRUD поверх таблицы documents. Сессией управляет вызывающий код."""

    def __init__(self, session: Session):
        self.session = session

    def create(self, doc_data: Dict[str, Any]) -> DocumentDB:
        doc = DocumentDB(**doc_data)
        self.session.add(doc)
        return doc

    def get(self, doc_id: str, for_update: bool = False) -> Optional[DocumentDB]:
        """for_update=True добавляет SELECT ... FOR UPDATE (блокировка строки)."""
        query = self.session.query(DocumentDB).filter(DocumentDB.id == doc_id)
        if for_update:
            query = query.with_for_update()
        return query.first()

    def get_by_s3_fileid(self, s3_fileid: str) -> Optional[DocumentDB]:
        return (
            self.session.query(DocumentDB)
            .filter(DocumentDB.s3_fileid == s3_fileid)
            .first()
        )

    def update_status(self, doc_id: str, status: str) -> bool:
        """Обновляет статус и updated_at. True, если строка найдена."""
        updated = (
            self.session.query(DocumentDB)
            .filter(DocumentDB.id == doc_id)
            .update({"status": status, "updated_at": utcnow()}, synchronize_session=False)
        )
        return updated > 0

    def update_extra(self, doc_id: str, extra: Dict[str, Any]) -> None:
        """Сливает словарь в extra_data (MutableDict отследит изменение)."""
        doc = self.get(doc_id, for_update=True)
        if doc is None:
            return
        current = dict(doc.extra_data or {})
        current.update(extra)
        doc.extra_data = current
        doc.updated_at = utcnow()

    def update_metadata(self, doc_id: str, **kwargs) -> None:
        doc = self.get(doc_id, for_update=True)
        if doc is None:
            return
        for key, value in kwargs.items():
            if hasattr(doc, key) and key not in ("id", "created_at"):
                setattr(doc, key, value)
        doc.updated_at = utcnow()

    def is_stale(self, doc: DocumentDB, timeout_seconds: int) -> bool:
        """Документ завис в pending дольше таймаута — можно перезапускать."""
        if doc.status != "pending":
            return False
        stamp = doc.updated_at or doc.created_at
        if stamp is None:
            return True
        return (utcnow() - stamp).total_seconds() > timeout_seconds
