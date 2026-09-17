"""Репозиторий фрагментов (чанков)."""

from typing import Any, Dict, List

from sqlalchemy.orm import Session

from core.db.models import ChunkDB


class ChunkRepository:
    def __init__(self, session: Session):
        self.session = session

    def create_chunk(self, chunk_dict: Dict[str, Any]) -> ChunkDB:
        chunk = ChunkDB(**chunk_dict)
        self.session.add(chunk)
        return chunk

    def create_chunks_bulk(self, chunks_list: List[Dict[str, Any]]) -> List[ChunkDB]:
        chunks = [ChunkDB(**data) for data in chunks_list]
        self.session.add_all(chunks)
        return chunks

    def delete_by_doc_id(self, doc_id: str) -> int:
        return (
            self.session.query(ChunkDB)
            .filter(ChunkDB.doc_id == doc_id)
            .delete(synchronize_session=False)
        )

    def get_by_doc_id(self, doc_id: str) -> List[ChunkDB]:
        return (
            self.session.query(ChunkDB)
            .filter(ChunkDB.doc_id == doc_id)
            .order_by(ChunkDB.order_index, ChunkDB.id)
            .all()
        )

    def get_by_doc_id_and_type(self, doc_id: str, chunk_type: str) -> List[ChunkDB]:
        return (
            self.session.query(ChunkDB)
            .filter(ChunkDB.doc_id == doc_id, ChunkDB.chunk_type == chunk_type)
            .order_by(ChunkDB.order_index, ChunkDB.id)
            .all()
        )

    def get_by_ids(self, chunk_ids: List[str]) -> List[ChunkDB]:
        if not chunk_ids:
            return []
        return self.session.query(ChunkDB).filter(ChunkDB.id.in_(chunk_ids)).all()

    def count_by_doc_id(self, doc_id: str) -> int:
        return self.session.query(ChunkDB).filter(ChunkDB.doc_id == doc_id).count()
