"""Состояние пакетного запроса в Redis."""

import pytest

from core.result_aggregator import (
    compute_status,
    delete_request_state,
    get_documents,
    get_request_state,
    init_request_state,
    mark_started,
    put_document_result,
)


def document(fileid, status="indexed", error=None):
    return {
        "s3_fileid": fileid,
        "document_metadata": {"doc_id": f"doc-{fileid}", "status": status},
        "fragments": [],
        "error": error,
    }


@pytest.mark.usefixtures("fake_redis")
class TestState:
    def test_initial_state_is_queued(self):
        init_request_state("req-1", "dlg-1", ["f1", "f2"])
        state = get_request_state("req-1")
        assert state["status"] == "queued"
        assert state["total"] == 2
        assert state["processed"] == 0

    def test_started_switches_to_processing(self):
        init_request_state("req-1", "dlg-1", ["f1", "f2"])
        mark_started("req-1")
        assert get_request_state("req-1")["status"] == "processing"

    def test_partial_progress(self):
        init_request_state("req-1", "dlg-1", ["f1", "f2"])
        put_document_result("req-1", "f1", document("f1"))
        state = get_request_state("req-1")
        assert state["status"] == "processing"
        assert state["processed"] == 1

    def test_completed(self):
        init_request_state("req-1", "dlg-1", ["f1", "f2"])
        put_document_result("req-1", "f1", document("f1"))
        put_document_result("req-1", "f2", document("f2"))
        assert get_request_state("req-1")["status"] == "completed"

    def test_partial_when_one_failed(self):
        init_request_state("req-1", "dlg-1", ["f1", "f2"])
        put_document_result("req-1", "f1", document("f1"))
        put_document_result("req-1", "f2", document("f2", status="error", error="boom"))
        assert get_request_state("req-1")["status"] == "partial"

    def test_error_when_all_failed(self):
        init_request_state("req-1", "dlg-1", ["f1"])
        put_document_result("req-1", "f1", document("f1", status="error", error="boom"))
        assert get_request_state("req-1")["status"] == "error"

    def test_documents_keep_request_order(self):
        init_request_state("req-1", "dlg-1", ["f1", "f2", "f3"])
        for fileid in ("f3", "f1", "f2"):
            put_document_result("req-1", fileid, document(fileid))
        order = [d["s3_fileid"] for d in get_request_state("req-1")["documents"]]
        assert order == ["f1", "f2", "f3"]

    def test_duplicate_result_does_not_inflate_counter(self):
        """Задача Celery может выполниться дважды — счётчик не должен врать."""
        init_request_state("req-1", "dlg-1", ["f1", "f2"])
        put_document_result("req-1", "f1", document("f1"))
        put_document_result("req-1", "f1", document("f1"))
        state = get_request_state("req-1")
        assert state["processed"] == 1
        assert state["status"] == "processing"

    def test_reinit_clears_previous_documents(self):
        init_request_state("req-1", "dlg-1", ["f1"])
        put_document_result("req-1", "f1", document("f1"))
        init_request_state("req-1", "dlg-1", ["f1", "f2"])
        assert get_documents("req-1") == {}

    def test_missing_state(self):
        assert get_request_state("нет-такого") is None

    def test_delete(self):
        init_request_state("req-1", "dlg-1", ["f1"])
        delete_request_state("req-1")
        assert get_request_state("req-1") is None

    def test_operation_preserved(self):
        init_request_state("req-2", "dlg", ["f1"], operation="update")
        assert get_request_state("req-2")["operation"] == "update"


class TestComputeStatus:
    def test_queued_vs_processing(self):
        assert compute_status(2, [], started=False) == "queued"
        assert compute_status(2, [], started=True) == "processing"

    def test_all_good(self):
        docs = [document("f1"), document("f2")]
        assert compute_status(2, docs, started=True) == "completed"

    def test_mixed(self):
        docs = [document("f1"), document("f2", status="error")]
        assert compute_status(2, docs, started=True) == "partial"
