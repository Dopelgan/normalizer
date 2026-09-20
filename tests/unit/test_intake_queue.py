"""
Приём на очереди: состояние запроса, задача и цепочка этапов.

Синхронный приём упирался в два таймаута сразу — на запросе между Data
Gateway и Quality Gate (120 с) и на стороне внешнего потребителя (60 с).
Потребитель, не дождавшись ответа, слал запрос заново. Здесь проверяется,
что ни того, ни другого больше нет: работа идёт по файлу на задачу,
вердикты копятся в состоянии запроса, повтор не запускает вторую обработку.
"""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from core.db.models import Base, IntakeDecisionDB
from core.gateway.service import ACCEPT, QUARANTINE, REJECT
from core.intake import state
from core.intake.pipeline import IntakePipeline
from core.models.intake import IntakeFile
from core.providers.storage import LocalStorageProvider

CONTENT = ("Настоящий договор поставки оборудования. " * 30).encode("utf-8")


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
    (tmp_path / "договор.txt").write_bytes(CONTENT)
    (tmp_path / "копия.txt").write_bytes(CONTENT)
    (tmp_path / "фото.txt").write_bytes(
        ("Фото с отпуска, день рождения. " * 30).encode("utf-8")
    )
    return LocalStorageProvider(base_path=str(tmp_path))


@pytest.fixture
def pipeline(storage, session_factory):
    return IntakePipeline(storage=storage, session_factory=session_factory)


# ===========================================================================
# Состояние приёма
# ===========================================================================

class TestState:
    def test_status_goes_queued_processing_completed(self, fake_redis):
        state.init_request("r1", [{"s3_fileid": "a", "operation": "create"},
                                  {"s3_fileid": "b", "operation": "create"}])
        assert state.get_state("r1")["status"] == "queued"

        state.mark_started("r1")
        state.put_file_result("r1", "a", _result("a", ACCEPT))
        assert state.get_state("r1")["status"] == "processing"

        state.put_file_result("r1", "b", _result("b", REJECT))
        snapshot = state.get_state("r1")
        assert snapshot["status"] == "completed"
        assert snapshot["accepted"] == ["a"]
        assert snapshot["rejected"] == ["b"]

    def test_repeat_of_the_same_file_does_not_double_count(self, fake_redis):
        state.init_request("r2", [{"s3_fileid": "a", "operation": "create"}])
        state.put_file_result("r2", "a", _result("a", ACCEPT))
        state.put_file_result("r2", "a", _result("a", ACCEPT))
        assert state.get_state("r2")["processed"] == 1

    def test_unknown_request_has_no_state(self, fake_redis):
        assert state.get_state("нет-такого") is None
        assert state.is_active("нет-такого") is False

    def test_unfinished_request_is_active(self, fake_redis):
        state.init_request("r3", [{"s3_fileid": "a", "operation": "create"}])
        assert state.is_active("r3") is True
        state.put_file_result("r3", "a", _result("a", ACCEPT))
        assert state.is_active("r3") is False

    def test_content_claim_catches_a_twin(self, fake_redis):
        assert state.claim_content("abc", "первый") is None
        assert state.claim_content("abc", "второй") == "первый"
        # Своя же заявка повтором не считается: задача может выполниться дважды.
        assert state.claim_content("abc", "первый") is None

    def test_released_claim_frees_the_hash(self, fake_redis):
        state.claim_content("abc", "первый")
        state.release_content("abc", "первый")
        assert state.claim_content("abc", "второй") is None


def _result(s3_fileid, outcome):
    return {
        "s3_fileid": s3_fileid, "outcome": outcome, "verdicts": [],
        "forwarded": False, "forward_error": None, "timings_ms": {},
    }


# ===========================================================================
# Цепочка этапов
# ===========================================================================

class TestPipeline:
    def test_both_stages_run_without_http(self, pipeline, fake_redis):
        """
        Между Data Gateway и Quality Gate больше нет сетевого запроса —
        значит, нет и таймаута на нём.
        """
        result = pipeline.run("r10", IntakeFile(s3_fileid="договор.txt"))
        assert result["outcome"] == ACCEPT
        assert [v["stage"] for v in result["verdicts"]] == [
            "data_gateway", "quality_gate"
        ]

    def test_rejected_at_gateway_never_reaches_quality(self, pipeline, fake_redis):
        """Отсев идёт первым: личное до проверки качества не доходит."""
        result = pipeline.run("r11", IntakeFile(s3_fileid="фото.txt"))
        assert result["outcome"] == REJECT
        assert [v["stage"] for v in result["verdicts"]] == ["data_gateway"]

    def test_missing_file_is_rejected_with_a_reason(self, pipeline, fake_redis):
        result = pipeline.run("r12", IntakeFile(s3_fileid="нет-такого"))
        assert result["outcome"] == REJECT
        assert result["verdicts"][0]["reason"]

    def test_timings_cover_both_stages(self, pipeline, fake_redis):
        result = pipeline.run("r13", IntakeFile(s3_fileid="договор.txt"))
        timings = result["timings_ms"]
        assert "intake.file" in timings
        assert any(k.startswith("data_gateway.") for k in timings)
        assert any(k.startswith("quality_gate.") for k in timings)

    def test_decision_is_written_for_both_stages(
        self, pipeline, session_factory, fake_redis
    ):
        """Решение переживает запрос — иначе ни дублей, ни отчёта приёма."""
        pipeline.run("r14", IntakeFile(s3_fileid="договор.txt"))
        session = session_factory()
        try:
            rows = session.query(IntakeDecisionDB).all()
            assert {(r.stage, r.outcome) for r in rows} == {
                ("data_gateway", "accept"), ("quality_gate", "accept"),
            }
        finally:
            session.close()

    def test_twin_in_the_same_batch_is_rejected(self, pipeline, fake_redis):
        """
        Два одинаковых файла обрабатываются параллельно и записей друг
        друга в базе ещё не видят — гонку закрывает заявка на содержимое.
        """
        first = pipeline.run("r15", IntakeFile(s3_fileid="договор.txt"))
        second = pipeline.run("r15", IntakeFile(s3_fileid="копия.txt"))
        assert first["outcome"] == ACCEPT
        assert second["outcome"] == REJECT
        assert "дубликат" in second["verdicts"][-1]["reason"].lower()

    def test_failed_stage_does_not_raise(self, pipeline, fake_redis, monkeypatch):
        """Сбой на одном файле не уводит в повтор вердикты по остальным."""
        def boom(*_args, **_kwargs):
            raise RuntimeError("хранилище легло")

        monkeypatch.setattr(pipeline.locator, "locate", boom)
        result = pipeline.run("r16", IntakeFile(s3_fileid="договор.txt"))
        assert result["outcome"] == REJECT


# ===========================================================================
# Задача
# ===========================================================================

class TestTask:
    def test_task_writes_result_to_state(
        self, storage, session_factory, fake_redis, monkeypatch
    ):
        from core.intake import pipeline as pipeline_module
        from core.intake.tasks import process_intake_file

        monkeypatch.setattr(pipeline_module, "SessionLocal", session_factory)
        monkeypatch.setattr(
            pipeline_module.StorageProviderFactory, "default",
            staticmethod(lambda *a, **k: storage),
        )
        state.init_request("r20", [{"s3_fileid": "договор.txt", "operation": "create"}])
        process_intake_file.apply(kwargs={
            "request_id": "r20", "s3_fileid": "договор.txt", "operation": "create",
        })

        snapshot = state.get_state("r20")
        assert snapshot["status"] == "completed"
        assert snapshot["accepted"] == ["договор.txt"]

    def test_broken_file_goes_to_quarantine_not_rejection(
        self, storage, session_factory, fake_redis, monkeypatch
    ):
        """
        Ошибка обработки — не приговор документу. Отказ означал бы, что
        файл проверен и не годится; здесь он просто не проверен.
        """
        from core.intake import tasks as tasks_module
        from core.intake.tasks import process_intake_file

        class Broken:
            def run(self, *_args, **_kwargs):
                raise RuntimeError("модель не ответила")

        monkeypatch.setattr(tasks_module, "IntakePipeline", lambda *a, **k: Broken())
        state.init_request("r21", [{"s3_fileid": "договор.txt", "operation": "create"}])
        process_intake_file.apply(kwargs={
            "request_id": "r21", "s3_fileid": "договор.txt",
        })

        snapshot = state.get_state("r21")
        assert snapshot["quarantined"] == ["договор.txt"]
        assert "ошибкой" in snapshot["verdicts"][0]["reason"]

    def test_task_is_routed_to_the_intake_queue(self):
        """
        Приём и разбор — разные очереди: пачка сканов не должна ждать
        чертёж, который модель читает семь минут.
        """
        from core.celery_app import app
        from core.config import settings

        route = app.conf.task_routes["core.intake.tasks.process_intake_file"]
        assert route["queue"] == settings.CELERY_INTAKE_QUEUE
        assert settings.CELERY_INTAKE_QUEUE != settings.CELERY_ML_QUEUE
        assert settings.CELERY_INTAKE_QUEUE in {q.name for q in app.conf.task_queues}
