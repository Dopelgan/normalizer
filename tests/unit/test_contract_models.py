"""Модели контракта: валидаторы и требование «все поля присутствуют»."""

import json
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from core.models.contract import (
    Content,
    DocumentMetadata,
    DocumentResult,
    Fragment,
    ParseResponse,
    Position,
    Provenance,
    TableData,
)


def make_fragment(**kwargs) -> Fragment:
    defaults = dict(
        fragment_id="doc-001-frag-001",
        type="text",
        content=Content(text="Нормализованный текст"),
        position=Position(page=1, bbox=[0.1, 0.2, 0.9, 0.3], order=1),
        confidence=0.97,
        completeness=1.0,
        provenance=Provenance(
            method="text_layer_extraction", strategy_level=2, source="vector_pdf"
        ),
    )
    defaults.update(kwargs)
    return Fragment(**defaults)


class TestPosition:
    def test_bbox_clamped_and_ordered(self):
        position = Position(page=1, bbox=[0.9, 1.4, 0.1, -0.2])
        assert position.bbox == [0.1, 0.0, 0.9, 1.0]

    def test_bbox_wrong_length_rejected(self):
        with pytest.raises(ValidationError):
            Position(page=1, bbox=[0.1, 0.2, 0.3])

    def test_empty_bbox_defaults_to_full_page(self):
        assert Position(page=1, bbox=[]).bbox == [0.0, 0.0, 1.0, 1.0]


class TestFragment:
    def test_confidence_out_of_range_rejected(self):
        with pytest.raises(ValidationError):
            make_fragment(confidence=1.4)
        with pytest.raises(ValidationError):
            make_fragment(completeness=-0.1)

    def test_unknown_type_rejected(self):
        with pytest.raises(ValidationError):
            make_fragment(type="chart")

    def test_all_content_keys_present_even_when_null(self):
        """Контракт требует присутствия всех полей, даже со значением null."""
        payload = make_fragment().model_dump(mode="json")
        assert set(payload["content"]) == {
            "text", "table_data", "image_ref", "formula_mathml",
            "parsed_expression", "structured_drawing_fields", "structured_payload",
        }
        assert payload["content"]["table_data"] is None
        assert payload["graph_nodes"] == []
        assert payload["relations"] == []

    def test_no_access_fields(self):
        payload = make_fragment().model_dump(mode="json")
        for forbidden in ("container", "legal_entity", "sensitivity", "access"):
            assert forbidden not in payload


class TestDocumentMetadata:
    def test_updated_at_iso8601_with_z(self):
        metadata = DocumentMetadata(
            doc_id="doc-001",
            language="ru",
            source_path="/data/documents/file-001.pdf",
            updated_at=datetime(2026, 9, 8, 10, 0, 0, tzinfo=timezone.utc),
            status="indexed",
        )
        assert metadata.model_dump(mode="json")["updated_at"] == "2026-09-08T10:00:00Z"

    def test_status_restricted(self):
        with pytest.raises(ValidationError):
            DocumentMetadata(doc_id="d", status="processing")


class TestResponseShape:
    def test_matches_contract_example(self):
        response = ParseResponse(
            request_id="req-001",
            dialog_id="dialog-001",
            documents=[
                DocumentResult(
                    s3_fileid="file-id-001",
                    document_metadata=DocumentMetadata(
                        doc_id="doc-001",
                        language="ru",
                        source_path="/data/documents/file-001.pdf",
                        updated_at=datetime(2026, 9, 8, 10, 0, tzinfo=timezone.utc),
                        status="indexed",
                    ),
                    fragments=[make_fragment()],
                )
            ],
        )
        payload = json.loads(response.model_dump_json())

        assert set(payload) == {"request_id", "dialog_id", "documents"}
        document = payload["documents"][0]
        assert set(document) >= {"s3_fileid", "document_metadata", "fragments"}
        assert set(document["document_metadata"]) == {
            "doc_id", "doc_version", "language", "source_path", "updated_at", "status",
        }
        fragment = document["fragments"][0]
        assert set(fragment) == {
            "fragment_id", "type", "content", "position", "section_title",
            "confidence", "completeness", "provenance", "graph_nodes", "relations",
        }
        assert set(fragment["position"]) == {"page", "sheet", "bbox", "order"}
        assert set(fragment["provenance"]) == {"method", "strategy_level", "source"}

    def test_table_fragment(self):
        fragment = make_fragment(
            type="table",
            content=Content(table_data=TableData(headers=["A", "B"], rows=[["1", "2"]])),
        )
        payload = fragment.model_dump(mode="json")
        assert payload["content"]["table_data"]["headers"] == ["A", "B"]
        assert payload["content"]["table_data"]["total_row"] is None
