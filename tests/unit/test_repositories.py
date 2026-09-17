"""Репозитории документов и фрагментов."""

from datetime import timedelta

import pytest

from core.db.models import ChunkDB, DocumentDB, utcnow
from core.repositories import ChunkRepository, DocumentRepository


def doc_payload(doc_id, **kwargs):
    payload = {
        "id": doc_id,
        "s3_fileid": f"file-{doc_id}",
        "source_path": "documents/test.pdf",
        "language": "ru",
        "status": "pending",
        "extra_data": {},
    }
    payload.update(kwargs)
    return payload


def chunk_payload(doc_id, index=1, **kwargs):
    payload = {
        "id": f"{doc_id}-frag-{index:03d}",
        "doc_id": doc_id,
        "chunk_type": "text",
        "content": {"text": f"Фрагмент {index}"},
        "position": {"page": 1, "sheet": None, "bbox": [0, 0, 1, 1], "order": index},
        "confidence": 0.9,
        "completeness": 0.8,
        "provenance": {"method": "mineru", "strategy_level": 2, "source": "vector_pdf"},
        "graph_nodes": [],
        "relations": [],
        "order_index": index,
        "extra_data": {},
    }
    payload.update(kwargs)
    return payload


class TestDocumentRepository:
    def test_create_and_get(self, db_session):
        repo = DocumentRepository(db_session)
        repo.create(doc_payload("doc-1"))
        db_session.commit()

        doc = repo.get("doc-1")
        assert doc is not None
        assert doc.status == "pending"
        assert doc.created_at is not None
        assert doc.updated_at is not None

    def test_update_status_touches_updated_at(self, db_session):
        repo = DocumentRepository(db_session)
        repo.create(doc_payload("doc-1"))
        db_session.commit()
        before = repo.get("doc-1").updated_at

        repo.update_status("doc-1", "indexed")
        db_session.commit()
        doc = repo.get("doc-1")

        assert doc.status == "indexed"
        assert doc.updated_at >= before

    def test_update_extra_merges(self, db_session):
        repo = DocumentRepository(db_session)
        repo.create(doc_payload("doc-1", extra_data={"parser": "mineru"}))
        db_session.commit()

        repo.update_extra("doc-1", {"pages": 10})
        db_session.commit()

        extra = repo.get("doc-1").extra_data
        assert extra["pages"] == 10
        assert extra["parser"] == "mineru"

    def test_update_metadata_ignores_protected_fields(self, db_session):
        repo = DocumentRepository(db_session)
        repo.create(doc_payload("doc-1"))
        db_session.commit()
        created = repo.get("doc-1").created_at

        repo.update_metadata("doc-1", id="hacked", created_at=None, language="en")
        db_session.commit()

        doc = repo.get("doc-1")
        assert doc.id == "doc-1"
        assert doc.created_at == created
        assert doc.language == "en"

    def test_lookup_by_s3_fileid(self, db_session):
        repo = DocumentRepository(db_session)
        repo.create(doc_payload("doc-1", s3_fileid="file-id-001"))
        db_session.commit()
        assert repo.get_by_s3_fileid("file-id-001").id == "doc-1"

    def test_is_stale(self, db_session):
        repo = DocumentRepository(db_session)
        repo.create(doc_payload("doc-1"))
        db_session.commit()
        doc = repo.get("doc-1")

        assert repo.is_stale(doc, timeout_seconds=900) is False
        doc.updated_at = utcnow() - timedelta(seconds=1000)
        assert repo.is_stale(doc, timeout_seconds=900) is True

        doc.status = "indexed"
        assert repo.is_stale(doc, timeout_seconds=900) is False

    def test_missing_document_is_noop(self, db_session):
        repo = DocumentRepository(db_session)
        assert repo.update_status("нет", "indexed") is False
        repo.update_extra("нет", {"a": 1})     # не должно падать
        repo.update_metadata("нет", language="en")


class TestChunkRepository:
    def test_bulk_create_and_order(self, db_session):
        DocumentRepository(db_session).create(doc_payload("doc-1"))
        repo = ChunkRepository(db_session)
        repo.create_chunks_bulk([chunk_payload("doc-1", i) for i in (3, 1, 2)])
        db_session.commit()

        chunks = repo.get_by_doc_id("doc-1")
        assert [c.order_index for c in chunks] == [1, 2, 3]
        assert repo.count_by_doc_id("doc-1") == 3

    def test_delete_by_doc_id(self, db_session):
        DocumentRepository(db_session).create(doc_payload("doc-1"))
        repo = ChunkRepository(db_session)
        repo.create_chunk(chunk_payload("doc-1"))
        db_session.commit()

        assert repo.delete_by_doc_id("doc-1") == 1
        db_session.commit()
        assert repo.get_by_doc_id("doc-1") == []

    def test_filter_by_type(self, db_session):
        DocumentRepository(db_session).create(doc_payload("doc-1"))
        repo = ChunkRepository(db_session)
        repo.create_chunks_bulk([
            chunk_payload("doc-1", 1),
            chunk_payload("doc-1", 2, chunk_type="table", content={"table_data": {}}),
        ])
        db_session.commit()

        assert len(repo.get_by_doc_id_and_type("doc-1", "table")) == 1
        assert len(repo.get_by_doc_id_and_type("doc-1", "text")) == 1

    def test_get_by_ids(self, db_session):
        DocumentRepository(db_session).create(doc_payload("doc-1"))
        repo = ChunkRepository(db_session)
        repo.create_chunks_bulk([chunk_payload("doc-1", i) for i in (1, 2)])
        db_session.commit()

        assert len(repo.get_by_ids(["doc-1-frag-001"])) == 1
        assert repo.get_by_ids([]) == []

    def test_cascade_delete_with_document(self, db_session):
        doc_repo = DocumentRepository(db_session)
        chunk_repo = ChunkRepository(db_session)
        doc_repo.create(doc_payload("doc-1"))
        chunk_repo.create_chunk(chunk_payload("doc-1"))
        db_session.commit()

        db_session.delete(doc_repo.get("doc-1"))
        db_session.commit()
        assert db_session.query(ChunkDB).count() == 0
        assert db_session.query(DocumentDB).count() == 0

    def test_json_fields_roundtrip(self, db_session):
        DocumentRepository(db_session).create(doc_payload("doc-1"))
        repo = ChunkRepository(db_session)
        repo.create_chunk(chunk_payload(
            "doc-1", content={"text": None, "table_data": {"headers": ["A"], "rows": [["1"]]}},
        ))
        db_session.commit()

        chunk = repo.get_by_doc_id("doc-1")[0]
        assert chunk.content["table_data"]["rows"] == [["1"]]
        assert chunk.position["bbox"] == [0, 0, 1, 1]
