"""Преобразование фрагмент <-> строка БД."""

from datetime import datetime, timezone

from core.db.models import DocumentDB
from core.mappers import document_to_metadata, fragment_to_row, row_to_fragment
from core.models.contract import Content, Fragment, Position, Provenance, TableData
from core.repositories import ChunkRepository, DocumentRepository


def make_fragment(**kwargs) -> Fragment:
    defaults = dict(
        fragment_id="doc-1-frag-001",
        type="text",
        content=Content(text="Текст фрагмента"),
        position=Position(page=2, sheet="A1", bbox=[0.1, 0.2, 0.9, 0.4], order=5),
        section_title="Раздел 1",
        confidence=0.93,
        completeness=0.71,
        provenance=Provenance(method="text_layer_extraction", strategy_level=2, source="vector_pdf"),
    )
    defaults.update(kwargs)
    return Fragment(**defaults)


class TestFragmentToRow:
    def test_fields_mapped(self):
        row = fragment_to_row(make_fragment(), "doc-1", {"needs_review": True})
        assert row["id"] == "doc-1-frag-001"
        assert row["doc_id"] == "doc-1"
        assert row["chunk_type"] == "text"
        assert row["order_index"] == 5
        assert row["position"]["sheet"] == "A1"
        assert row["extra_data"]["needs_review"] is True

    def test_service_flags_do_not_leak_into_contract(self):
        fragment = make_fragment()
        row = fragment_to_row(fragment, "doc-1", {"needs_review": True})
        assert "needs_review" not in row["content"]
        assert "needs_review" not in fragment.model_dump()


class TestRoundtrip:
    def test_text_fragment(self, db_session):
        DocumentRepository(db_session).create({
            "id": "doc-1", "source_path": "documents/a.pdf", "language": "ru", "status": "pending",
        })
        repo = ChunkRepository(db_session)
        repo.create_chunk(fragment_to_row(make_fragment(), "doc-1"))
        db_session.commit()

        restored = row_to_fragment(repo.get_by_doc_id("doc-1")[0])
        assert restored.model_dump() == make_fragment().model_dump()

    def test_table_fragment(self, db_session):
        DocumentRepository(db_session).create({
            "id": "doc-1", "source_path": "documents/a.pdf", "language": "ru", "status": "pending",
        })
        fragment = make_fragment(
            type="table",
            content=Content(table_data=TableData(headers=["A"], rows=[["1"]])),
        )
        repo = ChunkRepository(db_session)
        repo.create_chunk(fragment_to_row(fragment, "doc-1"))
        db_session.commit()

        restored = row_to_fragment(repo.get_by_doc_id("doc-1")[0])
        assert restored.type == "table"
        assert restored.content.table_data.rows == [["1"]]


class TestDocumentMetadata:
    def test_mapping(self):
        document = DocumentDB(
            id="doc-1", source_path="documents/a.pdf", language="ru", status="indexed",
            updated_at=datetime(2026, 9, 8, 10, 0, tzinfo=timezone.utc).replace(tzinfo=None),
        )
        metadata = document_to_metadata(document)
        assert metadata.doc_id == "doc-1"
        assert metadata.status == "indexed"
        assert metadata.model_dump(mode="json")["updated_at"] == "2026-09-08T10:00:00Z"

    def test_unknown_status_falls_back_to_pending(self):
        document = DocumentDB(id="doc-1", source_path="a", language="ru", status="processing")
        assert document_to_metadata(document).status == "pending"
