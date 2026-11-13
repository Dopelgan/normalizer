"""
Маршрут разбора: синхронный (вложение чата) — в отдельную очередь.

Под пакетной переобработкой (десятки чертежей, минуты на задачу) вложение
чата в той же очереди ml_gpu, что и пакет, не дождётся разбора: RAG ждёт
его ~30 с. Здесь проверяется, что синхронная ручка ставит задачу в
CELERY_INTERACTIVE_QUEUE, фоновая — в CELERY_ML_QUEUE, а при пустой
CELERY_INTERACTIVE_QUEUE всё идёт в CELERY_ML_QUEUE.
"""

import pytest
from fastapi.testclient import TestClient

from core.config import settings
from core.models.intake import IntakeFile
from ingest import api
from ingest.api import app, parse_queue


@pytest.fixture
def interactive_on(monkeypatch):
    monkeypatch.setattr(settings, "CELERY_INTERACTIVE_QUEUE", "interactive")
    monkeypatch.setattr(settings, "CELERY_ML_QUEUE", "ml_gpu")


@pytest.fixture
def interactive_off(monkeypatch):
    monkeypatch.setattr(settings, "CELERY_INTERACTIVE_QUEUE", "")
    monkeypatch.setattr(settings, "CELERY_ML_QUEUE", "ml_gpu")


class TestParseQueue:
    def test_sync_goes_to_interactive(self, interactive_on):
        assert parse_queue(interactive=True) == "interactive"

    def test_bulk_goes_to_ml_queue(self, interactive_on):
        assert parse_queue(interactive=False) == "ml_gpu"

    def test_disabled_keeps_old_route(self, interactive_off):
        assert parse_queue(interactive=True) == "ml_gpu"
        assert parse_queue(interactive=False) == "ml_gpu"


@pytest.fixture
def dispatched(mocker):
    return mocker.patch("ingest.api._dispatch")


@pytest.fixture
def client(fake_redis, dispatched, monkeypatch):
    # Синхронная ручка не ждёт результата, которого в тесте не будет.
    monkeypatch.setattr(settings, "PROCESSING_TIMEOUT_SECONDS", 0)
    with TestClient(app) as test_client:
        yield test_client


class TestEndpointsRoute:
    def test_sync_parse_is_dispatched_to_interactive(
        self, interactive_on, client, dispatched
    ):
        client.post("/internal/v1/parse", json={
            "request_id": "att-1", "dialog_id": "dlg-1", "s3_fileid": ["chat/a.pdf"],
        })
        dispatched.assert_called_once()
        assert dispatched.call_args.kwargs["queue"] == "interactive"

    def test_background_parse_is_dispatched_to_ml_queue(
        self, interactive_on, client, dispatched
    ):
        response = client.post("/internal/v1/parse/background", json={
            "request_id": "bulk-1", "event_id": "ev-1",
            "operation": "update", "s3_fileid": ["cluster/drawing.pdf"],
        })
        assert response.status_code == 202
        dispatched.assert_called_once()
        assert dispatched.call_args.kwargs["queue"] == "ml_gpu"

    def test_sync_parse_without_interactive_queue_uses_ml_queue(
        self, interactive_off, client, dispatched
    ):
        client.post("/internal/v1/parse", json={
            "request_id": "att-2", "dialog_id": "dlg-2", "s3_fileid": ["chat/b.pdf"],
        })
        assert dispatched.call_args.kwargs["queue"] == "ml_gpu"


class TestDispatchPublishes:
    def test_explicit_queue_reaches_apply_async(self, mocker):
        from ingest.tasks import process_document

        apply_async = mocker.patch.object(process_document, "apply_async")
        api._dispatch(
            "att-3",
            [IntakeFile(s3_fileid="chat/a.pdf"), IntakeFile(s3_fileid="chat/b.png")],
            "dlg-3", "", queue="interactive",
        )
        assert apply_async.call_count == 2
        for call in apply_async.call_args_list:
            assert call.kwargs["queue"] == "interactive"
            assert call.kwargs["kwargs"]["dialog_id"] == "dlg-3"
            assert call.kwargs["kwargs"]["event_id"] == ""
        assert [c.kwargs["kwargs"]["s3_fileid"] for c in apply_async.call_args_list] == [
            "chat/a.pdf", "chat/b.png",
        ]

    def test_queue_is_required(self):
        """Забыть очередь нельзя: молча уйти в ml_gpu — ровно исходная беда."""
        with pytest.raises(TypeError):
            api._dispatch("att-4", [IntakeFile(s3_fileid="x")], "dlg", "")


class TestCeleryRouting:
    def test_explicit_queue_beats_task_routes(self):
        """
        У process_document маршрут ml_gpu задан и в task_routes, и в
        декораторе. Явная очередь при постановке должна их перекрывать.
        """
        from core.celery_app import app as celery_app
        from ingest.tasks import process_document

        # Как в Task.apply_async: опции декоратора, поверх — явные.
        options = dict(process_document._get_exec_options(), queue="interactive")
        routed = celery_app.amqp.router.route(options, process_document.name)
        assert routed["queue"].name == "interactive"

    def test_default_route_is_still_ml_queue(self):
        from core.celery_app import app as celery_app

        routed = celery_app.amqp.router.route({}, "ingest.tasks.process_document")
        assert routed["queue"].name == settings.CELERY_ML_QUEUE

    def test_retry_stays_in_the_queue_it_came_from(self):
        """Повтор синхронной задачи не должен переехать в пакетную очередь."""
        from celery.app.task import Context

        from core.celery_app import app as celery_app
        from ingest.tasks import process_document

        # Redis публикует в анонимный обменник: exchange '' и routing_key,
        # равный имени очереди, — это и видит воркер в delivery_info.
        request = Context(
            id="t-1", args=(), kwargs={"request_id": "att-5", "s3_fileid": "a"},
            delivery_info={"exchange": "", "routing_key": "interactive"},
        )
        sig = process_document.signature_from_request(request=request)
        assert sig.options["queue"] == "interactive"
        # Итоговый маршрут повтора: поверх опций декоратора (queue=ml_gpu).
        options = dict(process_document._get_exec_options(), **sig.options)
        routed = celery_app.amqp.router.route(options, process_document.name)
        assert routed["queue"].name == "interactive"
