"""
Ручки приёма: Data Gateway и Quality Gate.

Здесь же регрессия на пятисотую Quality Gate: чтение известных хешей
открывало транзакцию, и следующий `with session.begin()` падал с
«A transaction is already begun on this Session» — то есть любой запрос,
в котором файл нашёлся, отвечал 500.
"""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import gateway.api as gateway_api
import quality.api as quality_api
from core.db.models import Base
from core.providers.storage import LocalStorageProvider

CONTENT = ("Технические требования к изделию. " * 40).encode("utf-8")


@pytest.fixture
def session_factory():
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, autocommit=False, autoflush=False)


@pytest.fixture
def storage(tmp_path, monkeypatch):
    from core.config import settings

    monkeypatch.setattr(settings, "SOURCE_PREFIX", "")
    monkeypatch.setattr(settings, "STORAGE_TYPE", "local")
    provider = LocalStorageProvider(base_path=str(tmp_path))
    (tmp_path / "отчёт.txt").write_bytes(CONTENT)
    (tmp_path / "смета.txt").write_bytes(CONTENT + b"\xd0\xb0")
    return provider


@pytest.fixture
def quality_client(session_factory, storage, monkeypatch, mocker):
    monkeypatch.setattr(quality_api, "SessionLocal", session_factory)
    mocker.patch.object(
        quality_api.StorageProviderFactory, "default", return_value=storage
    )
    forwarded = mocker.patch.object(quality_api.requests, "post")
    forwarded.return_value.raise_for_status.return_value = None
    with TestClient(quality_api.app) as client:
        yield client, forwarded


@pytest.fixture
def gateway_client(session_factory, storage, monkeypatch, mocker, fake_redis):
    """
    Data Gateway с очередью, выполняемой на месте.

    Celery в режиме `task_always_eager` выполняет задачу прямо в вызове
    `delay()`, поэтому к моменту ответа 202 вердикты уже лежат в состоянии
    приёма — ровно то, что нужно проверить одним запросом.
    """
    from core.celery_app import app as celery_app
    from core.intake import pipeline as intake_pipeline

    monkeypatch.setattr(gateway_api, "SessionLocal", session_factory)
    monkeypatch.setattr(intake_pipeline, "SessionLocal", session_factory)
    mocker.patch.object(
        intake_pipeline.StorageProviderFactory, "default", return_value=storage
    )
    monkeypatch.setattr(celery_app.conf, "task_always_eager", True)
    monkeypatch.setattr(celery_app.conf, "task_eager_propagates", False)
    forwarded = mocker.patch.object(intake_pipeline.requests, "post")
    forwarded.return_value.raise_for_status.return_value = None

    with TestClient(gateway_api.app) as client:
        yield client, forwarded


class TestQualityGate:
    def test_found_file_does_not_return_500(self, quality_client):
        """Регрессия: запись решения шла второй транзакцией и роняла ручку."""
        client, _ = quality_client
        response = client.post("/internal/v1/quality", json={
            "request_id": "req-1",
            "files": [{"s3_fileid": "отчёт.txt", "operation": "create"}],
        })
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["accepted"] == ["отчёт.txt"]
        assert body["verdicts"][0]["stage"] == "quality_gate"

    def test_decision_is_written(self, quality_client, session_factory):
        """Решение по файлу переживает запрос — иначе дубли не ловятся."""
        client, _ = quality_client
        client.post("/internal/v1/quality", json={
            "request_id": "req-2",
            "files": [{"s3_fileid": "отчёт.txt", "operation": "create"}],
        })

        from core.db.models import IntakeDecisionDB

        session = session_factory()
        try:
            rows = session.query(IntakeDecisionDB).all()
            assert [(r.s3_fileid, r.stage, r.outcome) for r in rows] == [
                ("отчёт.txt", "quality_gate", "accept")
            ]
            assert rows[0].file_hash
        finally:
            session.close()

    def test_exact_duplicate_inside_one_batch_rejected(self, quality_client, tmp_path):
        """Два одинаковых файла в одном пакете — второй дубликат."""
        (tmp_path / "копия.txt").write_bytes(CONTENT)
        client, _ = quality_client
        body = client.post("/internal/v1/quality", json={
            "request_id": "req-2b",
            "files": [
                {"s3_fileid": "отчёт.txt"},
                {"s3_fileid": "копия.txt"},
            ],
        }).json()
        assert body["accepted"] == ["отчёт.txt"]
        assert body["rejected"] == ["копия.txt"]

    def test_several_files_in_one_request(self, quality_client):
        client, _ = quality_client
        response = client.post("/internal/v1/quality", json={
            "request_id": "req-3",
            "files": [
                {"s3_fileid": "отчёт.txt", "operation": "create"},
                {"s3_fileid": "смета.txt", "operation": "update"},
                {"s3_fileid": "нет-такого", "operation": "create"},
            ],
        })
        assert response.status_code == 200, response.text
        body = response.json()
        assert set(body["accepted"]) == {"отчёт.txt", "смета.txt"}
        assert body["rejected"] == ["нет-такого"]

    def test_chain_stops_at_quality_gate(self, quality_client):
        """
        Передача в нормализатор закомментирована в `check()`: принятые файлы
        остаются на Quality Gate, `/internal/v1/parse/background` не
        вызывается, `forwarded` в ответе — `false`.
        """
        client, forwarded = quality_client
        body = client.post("/internal/v1/quality", json={
            "request_id": "req-4",
            "files": [
                {"s3_fileid": "отчёт.txt", "operation": "update"},
                {"s3_fileid": "смета.txt", "operation": "create"},
            ],
        }).json()
        assert set(body["accepted"]) == {"отчёт.txt", "смета.txt"}
        assert body["forwarded"] is False
        assert body["forward_error"] is None
        forwarded.assert_not_called()

    def test_forward_keeps_operation_per_file(self, quality_client):
        """
        Сама передача не удалена и остаётся по контракту: операция едет с
        файлом, `dialog_id` в теле нет. Проверяется прямым вызовом, потому
        что из ручки `_forward` сейчас не зовётся.
        """
        _client, forwarded = quality_client
        from core.models.intake import IntakeFile

        quality_api._forward("req-4", [
            IntakeFile(s3_fileid="отчёт.txt", operation="update"),
            IntakeFile(s3_fileid="смета.txt", operation="create"),
        ])
        payload = forwarded.call_args.kwargs["json"]
        assert payload["request_id"] == "req-4"
        assert {f["s3_fileid"]: f["operation"] for f in payload["files"]} == {
            "отчёт.txt": "update", "смета.txt": "create",
        }
        assert "dialog_id" not in payload

    def test_dialog_id_is_gone(self, quality_client):
        client, _ = quality_client
        body = client.post("/internal/v1/quality", json={
            "request_id": "req-5",
            "files": [{"s3_fileid": "отчёт.txt"}],
        }).json()
        assert "dialog_id" not in body


class TestDataGateway:
    """
    Приём поставлен на очередь: ручка отвечает 202 и идентификатором, по
    которому забираются вердикты.

    Повод — пятьдесят PDF без текстового слоя: синхронный приём не
    укладывался ни в таймаут между сервисами, ни в таймаут потребителя, и
    тот начинал слать запрос заново.
    """

    def test_intake_is_queued(self, gateway_client):
        client, _ = gateway_client
        response = client.post("/internal/v1/intake", json={
            "request_id": "req-10",
            "files": [
                {"s3_fileid": "отчёт.txt", "operation": "create"},
                {"s3_fileid": "смета.txt", "operation": "update"},
            ],
        })
        assert response.status_code == 202, response.text
        body = response.json()
        assert body["accepted"] is True
        assert body["total"] == 2
        assert body["poll_url"].endswith("/internal/v1/intake/results/req-10")
        assert "dialog_id" not in body

    def test_results_carry_verdicts_and_timings(self, gateway_client):
        client, _ = gateway_client
        client.post("/internal/v1/intake", json={
            "request_id": "req-10a",
            "files": [{"s3_fileid": "отчёт.txt", "operation": "create"}],
        })
        body = client.get("/internal/v1/intake/results/req-10a").json()
        assert body["status"] == "completed"
        assert body["processed"] == body["total"] == 1
        assert body["accepted"] == ["отчёт.txt"]
        stages = {v["stage"] for v in body["verdicts"]}
        assert stages == {"data_gateway", "quality_gate"}
        # Время операций едет вместе с вердиктом: без него на вопрос «где
        # файл провёл время» отвечать нечем.
        timings = body["files"][0]["timings_ms"]
        assert "intake.file" in timings
        assert any(k.startswith("quality_gate.") for k in timings)

    def test_unknown_request_is_404(self, gateway_client):
        client, _ = gateway_client
        assert client.get("/internal/v1/intake/results/нет-такого").status_code == 404

    def test_repeat_while_running_is_conflict(self, gateway_client, monkeypatch):
        """
        Потребитель, у которого истёк свой таймаут, повторяет запрос. Пока
        приём не закончен, повтор — конфликт, а не вторая обработка поверх
        первой.
        """
        from core.intake import state as intake_state

        client, _ = gateway_client
        monkeypatch.setattr(intake_state, "is_active", lambda request_id: True)
        response = client.post("/internal/v1/intake", json={
            "request_id": "req-10b",
            "files": [{"s3_fileid": "отчёт.txt"}],
        })
        assert response.status_code == 409

    def test_chain_stops_before_parser_by_default(self, gateway_client):
        client, forwarded = gateway_client
        client.post("/internal/v1/intake", json={
            "request_id": "req-10c",
            "files": [{"s3_fileid": "отчёт.txt"}],
        })
        body = client.get("/internal/v1/intake/results/req-10c").json()
        assert body["forwarded"] is False
        forwarded.assert_not_called()

    def test_forward_is_a_setting(self, gateway_client, monkeypatch):
        """Передача в нормализатор включается настройкой, а не правкой кода."""
        from core.config import settings

        monkeypatch.setattr(settings, "INTAKE_FORWARD_TO_PARSER", True)
        client, forwarded = gateway_client
        client.post("/internal/v1/intake", json={
            "request_id": "req-10d",
            "files": [{"s3_fileid": "отчёт.txt", "operation": "update"}],
        })
        body = client.get("/internal/v1/intake/results/req-10d").json()
        assert body["forwarded"] is True
        payload = forwarded.call_args.kwargs["json"]
        assert payload["files"] == [
            {"s3_fileid": "отчёт.txt", "operation": "update"}
        ]

    def test_sync_endpoint_keeps_old_shape(self, gateway_client):
        """Приём без очереди оставлен для ручной проверки."""
        client, _ = gateway_client
        body = client.post("/internal/v1/intake/sync", json={
            "request_id": "req-10e",
            "files": [
                {"s3_fileid": "отчёт.txt"},
                {"s3_fileid": "нет-такого"},
            ],
        }).json()
        assert body["accepted"] == ["отчёт.txt"]
        assert body["rejected"] == ["нет-такого"]
        assert "dialog_id" not in body

    def test_old_body_shape_rejected(self, gateway_client):
        client, _ = gateway_client
        response = client.post("/internal/v1/intake", json={
            "request_id": "req-12", "dialog_id": "d", "s3_fileid": ["отчёт.txt"],
        })
        assert response.status_code == 422

    def test_duplicate_fileid_rejected(self, gateway_client):
        client, _ = gateway_client
        response = client.post("/internal/v1/intake", json={
            "request_id": "req-13",
            "files": [
                {"s3_fileid": "отчёт.txt", "operation": "create"},
                {"s3_fileid": "отчёт.txt", "operation": "update"},
            ],
        })
        assert response.status_code == 422

    def test_batch_over_limit_rejected(self, gateway_client, monkeypatch):
        from core.config import settings

        monkeypatch.setattr(settings, "INTAKE_MAX_FILES", 2)
        client, _ = gateway_client
        response = client.post("/internal/v1/intake", json={
            "request_id": "req-14",
            "files": [{"s3_fileid": f"f{i}.txt"} for i in range(3)],
        })
        assert response.status_code == 422
