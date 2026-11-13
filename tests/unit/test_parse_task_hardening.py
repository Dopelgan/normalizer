"""
Живучесть разбора под Celery: повторная доставка долгой задачи, пропавшая
задача (жёсткий лимит, OOM, SIGKILL) и повтор POST готового запроса.
"""

import time
from datetime import timedelta

import pytest
from fastapi.testclient import TestClient

from core import result_aggregator
from core.config import settings
from core.db.models import utcnow
from core.models.contract import DocumentResult
from core.repositories import DocumentRepository
from core.result_aggregator import (
    get_request_state,
    init_request_state,
    mark_started,
    put_document_result,
)
from ingest import tasks
from ingest.tasks import generate_doc_id, process_document
from tests.unit.test_api import document
from tests.unit.test_tasks import pipeline, task_db  # noqa: F401 — фикстуры


# ===========================================================================
# Повторная доставка долгой задачи брокером
# ===========================================================================

class TestVisibilityTimeout:
    def test_broker_returns_task_well_after_hard_limit(self):
        from core.celery_app import app

        visibility = app.conf.broker_transport_options["visibility_timeout"]
        assert visibility >= 2 * settings.PARSE_TASK_TIME_LIMIT

    def test_configured_value_cannot_go_below_floor(self, monkeypatch):
        from core.celery_app import broker_visibility_timeout

        monkeypatch.setattr(settings, "CELERY_VISIBILITY_TIMEOUT", 100)
        assert broker_visibility_timeout() == 2 * settings.PARSE_TASK_TIME_LIMIT
        monkeypatch.setattr(settings, "CELERY_VISIBILITY_TIMEOUT", 100_000)
        assert broker_visibility_timeout() == 100_000

    def test_visibility_outlives_stale_threshold(self):
        """Повторная доставка должна видеть документ уже «зависшим»."""
        from core.celery_app import broker_visibility_timeout

        assert broker_visibility_timeout() > result_aggregator.parse_stale_after_seconds()


def _set_pending(task_db, fileid, age_seconds):  # noqa: F811
    session = task_db()
    with session.begin():
        DocumentRepository(session).create({
            "id": generate_doc_id(fileid), "s3_fileid": fileid,
            "source_path": f"documents/{fileid}.pdf", "language": "ru",
            "status": "pending", "extra_data": {},
            "updated_at": utcnow() - timedelta(seconds=age_seconds),
        })
    session.close()


class TestBusyDocument:
    def test_live_pending_document_is_not_parsed_twice(  # noqa: F811
        self, pipeline, task_db, mocker
    ):
        """
        Документ, который другая задача разбирает уже 20 минут, не считается
        зависшим (порог 900 с) и не разбирается повторно — это был бы второй
        OCR того же файла. Задача ждёт первую.
        """
        requeue = mocker.patch.object(tasks.process_document, "apply_async")
        _set_pending(task_db, "file-busy", age_seconds=1200)
        init_request_state("req-busy", ["file-busy"])

        result = process_document("req-busy", "file-busy")

        assert result["status"] == "waiting_for_peer"
        assert pipeline.calls == []
        requeue.assert_called_once()
        assert requeue.call_args.kwargs["kwargs"]["request_id"] == "req-busy"
        assert requeue.call_args.kwargs["countdown"] == tasks.PEER_WAIT_SECONDS
        # Вызов напрямую, без брокера — очередь разбора по умолчанию.
        assert requeue.call_args.kwargs["queue"] == settings.CELERY_ML_QUEUE
        # Документ остаётся у первой задачи, не помечается ошибкой.
        session = task_db()
        assert DocumentRepository(session).get(generate_doc_id("file-busy")).status == "pending"
        session.close()

    def test_stale_pending_document_is_restarted(self, pipeline, task_db, mocker):  # noqa: F811
        requeue = mocker.patch.object(tasks.process_document, "apply_async")
        _set_pending(
            task_db, "file-dead",
            age_seconds=result_aggregator.parse_stale_after_seconds() + 10,
        )
        init_request_state("req-dead", ["file-dead"])

        result = process_document("req-dead", "file-dead")

        assert result["status"] == "success"
        assert len(pipeline.calls) == 1
        requeue.assert_not_called()

    def test_waiting_does_not_count_as_lost_task(  # noqa: F811
        self, pipeline, task_db, fake_redis, mocker
    ):
        """Пока ожидающая задача стоит в очереди, файл не «пропавший»."""
        mocker.patch.object(tasks.process_document, "apply_async")
        _set_pending(task_db, "file-wait", age_seconds=10)
        init_request_state("req-wait", ["file-wait"])

        process_document("req-wait", "file-wait")

        assert not fake_redis.hexists("req:req-wait:since", "file-wait")

    def test_requeue_failure_does_not_mark_peer_document_error(  # noqa: F811
        self, pipeline, task_db, mocker
    ):
        """
        Сбой постановки ожидания уходит в общий обработчик ошибок; документ,
        который разбирает другая задача, при этом не помечается error —
        иначе следующий повтор начал бы второй разбор.
        """
        mocker.patch.object(
            tasks.process_document, "apply_async", side_effect=ConnectionError("broker")
        )
        mocker.patch.object(tasks.process_document, "retry", side_effect=RuntimeError("retry"))
        _set_pending(task_db, "file-peer", age_seconds=10)
        init_request_state("req-peer", ["file-peer"])

        with pytest.raises(RuntimeError):
            process_document("req-peer", "file-peer")

        session = task_db()
        assert DocumentRepository(session).get(generate_doc_id("file-peer")).status == "pending"
        session.close()

    def test_failed_attempt_handed_to_retry_is_not_lost(  # noqa: F811
        self, pipeline, task_db, fake_redis, mocker
    ):
        mocker.patch.object(tasks.process_document, "retry", side_effect=RuntimeError("retry"))
        pipeline.parse = mocker.Mock(side_effect=ValueError("boom"))
        init_request_state("req-retry", ["file-retry"])

        with pytest.raises(RuntimeError):
            process_document("req-retry", "file-retry")

        assert not fake_redis.hexists("req:req-retry:since", "file-retry")


# ===========================================================================
# Пропавшая задача: запрос не висит «processing» сутки
# ===========================================================================

class TestLostTask:
    def _age_start(self, fake_redis, request_id, fileid, seconds):
        fake_redis.hset(f"req:{request_id}:since", fileid, str(time.time() - seconds))

    def test_task_lost_after_hard_limit_reports_parser_timeout(self, fake_redis):
        init_request_state("req-lost", ["f-ok", "f-lost"])
        put_document_result("req-lost", "f-ok", document("f-ok"))
        mark_started("req-lost", "f-lost")
        self._age_start(
            fake_redis, "req-lost", "f-lost",
            result_aggregator.parse_stale_after_seconds() + 1,
        )

        state = get_request_state("req-lost")

        assert state["status"] == "partial"
        assert state["processed"] == 2
        lost = state["documents"][1]
        assert lost["s3_fileid"] == "f-lost"
        assert lost["status"] == "error"
        assert lost["error"]["reason_code"] == "PARSER_TIMEOUT"
        assert lost["error"]["retryable"] is True
        assert lost["document_metadata"]["doc_id"] == generate_doc_id("f-lost")
        DocumentResult(**lost)  # проходит контракт
        # Пометка сохраняется в состоянии запроса, а не только в ответе.
        assert fake_redis.hexists("req:req-lost:docs", "f-lost")

    def test_single_lost_file_makes_request_error(self, fake_redis):
        init_request_state("req-lost1", ["f1"])
        mark_started("req-lost1", "f1")
        self._age_start(
            fake_redis, "req-lost1", "f1", result_aggregator.parse_stale_after_seconds() + 1
        )
        assert get_request_state("req-lost1")["status"] == "error"

    def test_running_task_within_limit_stays_processing(self, fake_redis):
        init_request_state("req-run", ["f1"])
        mark_started("req-run", "f1")
        self._age_start(fake_redis, "req-run", "f1", settings.PARSE_TASK_TIME_LIMIT - 60)
        state = get_request_state("req-run")
        assert state["status"] == "processing"
        assert state["documents"] == []

    def test_queued_file_without_start_is_not_expired(self, fake_redis):
        """Очередь может стоять долго — время считается со старта задачи."""
        init_request_state("req-q", ["f1"])
        assert get_request_state("req-q")["status"] == "queued"

    def test_late_real_result_is_not_overwritten_by_timeout(self, fake_redis):
        init_request_state("req-late", ["f1"])
        mark_started("req-late", "f1")
        self._age_start(
            fake_redis, "req-late", "f1", result_aggregator.parse_stale_after_seconds() + 1
        )
        put_document_result("req-late", "f1", document("f1"))
        assert get_request_state("req-late")["status"] == "completed"


    def test_real_result_after_timeout_wins(self, fake_redis):
        """Опоздавший настоящий результат заменяет синтетическую ошибку."""
        init_request_state("req-after", ["f1"])
        mark_started("req-after", "f1")
        self._age_start(
            fake_redis, "req-after", "f1", result_aggregator.parse_stale_after_seconds() + 1
        )
        assert get_request_state("req-after")["status"] == "error"
        put_document_result("req-after", "f1", document("f1"))
        assert get_request_state("req-after")["status"] == "completed"


# ===========================================================================
# Повтор POST с request_id готового запроса
# ===========================================================================

@pytest.fixture
def client(fake_redis, mocker):
    dispatched = mocker.patch("ingest.api._dispatch")
    from ingest.api import app

    with TestClient(app) as test_client:
        test_client.dispatched = dispatched
        yield test_client


class TestRepost:
    def test_background_repost_of_completed_request_keeps_result(self, client):
        payload = {"request_id": "req-done", "event_id": "ev-1", "s3_fileid": ["f1"]}
        assert client.post("/internal/v1/parse/background", json=payload).status_code == 202
        put_document_result("req-done", "f1", document("f1"))
        client.dispatched.reset_mock()

        response = client.post("/internal/v1/parse/background", json=payload)

        assert response.status_code == 202
        assert response.json()["accepted"] is True
        assert response.json()["status"] == "queued"
        client.dispatched.assert_not_called()
        results = client.get("/internal/v1/parse/results/req-done").json()
        assert results["status"] == "completed"
        assert [d["s3_fileid"] for d in results["documents"]] == ["f1"]

    def test_sync_repost_returns_finished_result(self, client):
        init_request_state("req-sdone", ["f1"], "create", dialog_id="dlg")
        put_document_result("req-sdone", "f1", document("f1"))

        response = client.post("/internal/v1/parse", json={
            "request_id": "req-sdone", "dialog_id": "dlg", "s3_fileid": ["f1"],
        })

        assert response.status_code == 200
        assert [d["s3_fileid"] for d in response.json()["documents"]] == ["f1"]
        client.dispatched.assert_not_called()

    def test_repost_after_error_runs_again(self, client):
        """Ошибка — повод повторить: RAG ретраит с тем же request_id."""
        payload = {"request_id": "req-err", "event_id": "ev-1", "s3_fileid": ["f1"]}
        client.post("/internal/v1/parse/background", json=payload)
        put_document_result("req-err", "f1", document("f1", error={
            "reason_code": "PARSER_TIMEOUT", "message": "x", "retryable": True,
        }))
        client.dispatched.reset_mock()

        assert client.post("/internal/v1/parse/background", json=payload).status_code == 202
        client.dispatched.assert_called_once()
        assert get_request_state("req-err")["status"] == "queued"

    def test_repost_with_other_operation_conflicts(self, client):
        client.post("/internal/v1/parse/background", json={
            "request_id": "req-op", "event_id": "ev-1", "s3_fileid": ["f1"],
            "operation": "create",
        })
        put_document_result("req-op", "f1", document("f1"))

        response = client.post("/internal/v1/parse/background", json={
            "request_id": "req-op", "event_id": "ev-1", "s3_fileid": ["f1"],
            "operation": "update",
        })
        assert response.status_code == 409

    def test_repost_with_other_files_conflicts(self, client):
        client.post("/internal/v1/parse/background", json={
            "request_id": "req-x", "event_id": "ev-1", "s3_fileid": ["f1"],
        })
        put_document_result("req-x", "f1", document("f1"))

        response = client.post("/internal/v1/parse/background", json={
            "request_id": "req-x", "event_id": "ev-1", "s3_fileid": ["f2"],
        })
        assert response.status_code == 409
        assert get_request_state("req-x")["status"] == "completed"
