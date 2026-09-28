"""Ручки Parser по контракту RAG <-> Parser."""

import pytest
from fastapi.testclient import TestClient

from core.result_aggregator import init_request_state, put_document_result
from ingest.api import app


@pytest.fixture
def dispatched(mocker):
    """Подменяет постановку задач Celery — воркер в юнит-тестах не нужен."""
    return mocker.patch("ingest.api._dispatch")


@pytest.fixture
def client(fake_redis, dispatched):
    with TestClient(app) as test_client:
        yield test_client


def document(fileid, status="indexed", error=None):
    return {
        "s3_fileid": fileid,
        "status": "error" if error else "completed",
        "document_metadata": {
            "doc_id": f"doc-{fileid}",
            "doc_version": None,
            "language": "ru",
            "source_path": f"documents/{fileid}.pdf",
            "updated_at": "2026-09-08T10:00:00Z",
            "status": status,
        },
        "fragments": [],
        "error": error,
    }


class TestBackground:
    def test_returns_202_and_queued(self, client, dispatched):
        response = client.post("/internal/v1/parse/background", json={
            "request_id": "req-002",
            "event_id": "event-123",
            "operation": "create",
            "s3_fileid": ["file-id-003"],
        })
        assert response.status_code == 202
        assert response.json() == {
            "request_id": "req-002",
            "event_id": "event-123",
            "operation": "create",
            "accepted": True,
            "status": "queued",
        }
        dispatched.assert_called_once()

    def test_dialog_id_is_not_part_of_background_contract(self, client):
        """В фоне `dialog_id` не передаётся и не проверяется."""
        response = client.post("/internal/v1/parse/background", json={
            "request_id": "req-002b",
            "event_id": "event-124",
            "dialog_id": "event-124",
            "s3_fileid": ["file-id-003"],
        })
        assert response.status_code == 202
        assert "dialog_id" not in response.json()

    def test_event_id_not_replaced(self, client, dispatched):
        """`event_id` от Backend уезжает в задачу как есть, без нового UUID."""
        client.post("/internal/v1/parse/background", json={
            "request_id": "req-002c", "event_id": "event-125",
            "s3_fileid": ["file-1"],
        })
        assert dispatched.call_args[0][3] == "event-125"

    def test_operation_update_passed_through(self, client, dispatched):
        client.post("/internal/v1/parse/background", json={
            "request_id": "req-003", "event_id": "event-1", "operation": "update",
            "s3_fileid": ["file-1"],
        })
        items = dispatched.call_args[0][1]
        assert [item.operation for item in items] == ["update"]

    def test_operation_defaults_to_create(self, client):
        response = client.post("/internal/v1/parse/background", json={
            "request_id": "req-004", "event_id": "event-1", "s3_fileid": ["f1"],
        })
        assert response.json()["operation"] == "create"

    def test_unknown_operation_rejected(self, client):
        response = client.post("/internal/v1/parse/background", json={
            "request_id": "req-005", "event_id": "event-1",
            "operation": "delete", "s3_fileid": ["f1"],
        })
        assert response.status_code == 422

    def test_per_file_operation(self, client, dispatched):
        """Операция принадлежит файлу: в одном пакете и create, и update."""
        response = client.post("/internal/v1/parse/background", json={
            "request_id": "req-003b",
            "files": [
                {"s3_fileid": "file-1", "operation": "create"},
                {"s3_fileid": "file-2", "operation": "update"},
            ],
        })
        assert response.status_code == 202
        # Общей операции у пакета нет — и придумывать её ручка не должна.
        assert response.json()["operation"] is None
        items = dispatched.call_args[0][1]
        assert [(i.s3_fileid, i.operation) for i in items] == [
            ("file-1", "create"), ("file-2", "update"),
        ]

    def test_files_without_event_id_accepted(self, client):
        """У цепочки приёма lifecycle-события нет — `event_id` необязателен."""
        response = client.post("/internal/v1/parse/background", json={
            "request_id": "req-003c",
            "files": [{"s3_fileid": "file-9", "operation": "create"}],
        })
        assert response.status_code == 202
        assert response.json()["event_id"] == ""

    def test_empty_batch_rejected(self, client):
        response = client.post("/internal/v1/parse/background", json={
            "request_id": "req-003d",
        })
        assert response.status_code == 422

    def test_duplicate_active_request_conflicts(self, client):
        payload = {"request_id": "req-006", "event_id": "event-1", "s3_fileid": ["f1"]}
        assert client.post("/internal/v1/parse/background", json=payload).status_code == 202
        assert client.post("/internal/v1/parse/background", json=payload).status_code == 409


class TestResults:
    def test_processing(self, client):
        init_request_state("req-007", ["f1", "f2"], operation="create")
        put_document_result("req-007", "f1", document("f1"))

        body = client.get("/internal/v1/parse/results/req-007").json()
        assert body["status"] == "processing"
        assert body["operation"] == "create"
        assert len(body["documents"]) == 1

    def test_completed(self, client):
        init_request_state("req-008", ["f1"], operation="create")
        put_document_result("req-008", "f1", document("f1"))

        body = client.get("/internal/v1/parse/results/req-008").json()
        assert body["status"] == "completed"
        assert body["error"] is None
        assert body["documents"][0]["s3_fileid"] == "f1"
        assert body["documents"][0]["document_metadata"]["doc_id"] == "doc-f1"

    def test_partial_carries_error(self, client):
        init_request_state("req-009", ["f1", "f2"], operation="create")
        put_document_result("req-009", "f1", document("f1"))
        put_document_result("req-009", "f2", document("f2", status="error", error="файл не найден"))

        body = client.get("/internal/v1/parse/results/req-009").json()
        assert body["status"] == "partial"
        assert body["error"]["message"] == "файл не найден"
        assert body["error"]["reason_code"] == "INTERNAL_ERROR"

    def test_unknown_request_404(self, client):
        assert client.get("/internal/v1/parse/results/нет-такого").status_code == 404

    def test_response_shape_matches_contract(self, client):
        init_request_state("req-010", ["f1"], operation="create")
        put_document_result("req-010", "f1", document("f1"))
        body = client.get("/internal/v1/parse/results/req-010").json()
        assert set(body) == {
            "request_id", "event_id", "operation", "status", "documents", "error",
        }


class TestSync:
    def test_returns_documents_when_ready(self, client, mocker):
        def fake_dispatch(request_id, items, dialog_id="", event_id=""):
            for item in items:
                put_document_result(request_id, item.s3_fileid, document(item.s3_fileid))

        mocker.patch("ingest.api._dispatch", side_effect=fake_dispatch)

        response = client.post("/internal/v1/parse", json={
            "request_id": "req-001",
            "dialog_id": "dialog-001",
            "s3_fileid": ["file-id-001", "file-id-002"],
        })
        assert response.status_code == 200
        body = response.json()
        assert set(body) == {"request_id", "dialog_id", "documents"}
        assert [d["s3_fileid"] for d in body["documents"]] == ["file-id-001", "file-id-002"]

    def test_timeout_returns_504_with_state(self, client, monkeypatch):
        from core.config import settings

        monkeypatch.setattr(settings, "PROCESSING_TIMEOUT_SECONDS", 0)
        response = client.post("/internal/v1/parse", json={
            "request_id": "req-011", "dialog_id": "dlg", "s3_fileid": ["f1"],
        })
        assert response.status_code == 504
        body = response.json()
        assert body["status"] == "processing"
        # Синхронный ответ, поэтому `dialog_id` на месте, а ошибка — объект.
        assert body["dialog_id"] == "dlg"
        assert body["error"]["reason_code"] == "PARSER_TIMEOUT"
        assert body["error"]["retryable"] is True


class TestValidation:
    def test_empty_file_list_rejected(self, client):
        response = client.post("/internal/v1/parse", json={
            "request_id": "req-012", "dialog_id": "dlg", "s3_fileid": [],
        })
        assert response.status_code == 422

    def test_missing_dialog_id_rejected(self, client):
        response = client.post("/internal/v1/parse", json={
            "request_id": "req-013", "s3_fileid": ["f1"],
        })
        assert response.status_code == 422

    def test_extra_fields_ignored(self, client):
        response = client.post("/internal/v1/parse/background", json={
            "request_id": "req-014", "event_id": "event-1", "s3_fileid": ["f1"],
            "какое-то-новое-поле": 1,
        })
        assert response.status_code == 202


class TestHealth:
    def test_reports_dependencies(self, client):
        body = client.get("/health").json()
        assert body["status"] in ("ok", "degraded")
        assert set(body["dependencies"]) == {"redis", "database"}
        assert body["dependencies"]["redis"] is True

    def test_degraded_when_db_down(self, client, mocker):
        mocker.patch("core.db.session.engine.connect", side_effect=OSError("нет БД"))
        body = client.get("/health").json()
        assert body["status"] == "degraded"
        assert body["dependencies"]["database"] is False
