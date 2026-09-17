"""Celery-задача обработки документа: конвейер, идемпотентность, ошибки."""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from core.db.models import Base
from core.models.contract import Content, Fragment, Position, Provenance
from core.models.parse_result import ParsedBlock, ParseResult
from core.providers.file_locator import LocatedFile, SourceFileNotFound
from core.repositories import ChunkRepository, DocumentRepository
from core.result_aggregator import get_request_state, init_request_state
from ingest import tasks
from ingest.tasks import build_error_document, generate_doc_id, process_document


# ===========================================================================
# Окружение задачи
# ===========================================================================

@pytest.fixture
def task_db(monkeypatch):
    """Одна SQLite-база в памяти, общая для всех сессий задачи."""
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    monkeypatch.setattr(tasks, "SessionLocal", factory)
    return factory


class StubParser:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def parse(self, uri, file_type, metadata=None):
        self.calls.append((uri, file_type))
        return self.result


@pytest.fixture
def pipeline(monkeypatch, temp_storage, task_db, fake_redis):
    """Подменяет внешние зависимости задачи, оставляя реальными БД и агрегатор."""
    parse_result = ParseResult(
        blocks=[
            ParsedBlock(type="text", text="Настоящий документ описывает изделие. " * 3,
                        page=1, bbox=[0.1, 0.1, 0.9, 0.5], order=0),
        ],
        parser_name="mineru",
        source_kind="vector_pdf",
        page_count=1,
    )
    parser = StubParser(parse_result)

    monkeypatch.setattr(
        tasks.StorageProviderFactory, "default", staticmethod(lambda: temp_storage)
    )
    monkeypatch.setattr(
        tasks.FileLocator, "locate",
        lambda self, fileid: LocatedFile(fileid, f"documents/{fileid}.pdf", "pdf"),
    )
    monkeypatch.setattr(
        tasks.DocumentParserFactory, "get_parser", staticmethod(lambda storage=None: parser)
    )
    monkeypatch.setattr(
        tasks.DrawingProcessor, "enrich_fragments",
        lambda self, fragments, metadata: fragments,
    )
    return parser


# ===========================================================================
# Тесты
# ===========================================================================

class TestDocId:
    def test_deterministic(self):
        assert generate_doc_id("file-1") == generate_doc_id("file-1")
        assert generate_doc_id("file-1") != generate_doc_id("file-2")
        assert generate_doc_id("file-1").startswith("doc-")

    def test_fits_column(self):
        assert len(generate_doc_id("x" * 500)) <= 64


@pytest.mark.usefixtures("pipeline")
class TestHappyPath:
    def test_document_and_fragments_persisted(self, task_db):
        init_request_state("req-1", "dlg-1", ["file-1"])
        result = process_document("req-1", "dlg-1", "file-1")

        assert result["status"] == "success"
        assert result["fragments"] >= 1

        session = task_db()
        document = DocumentRepository(session).get(result["doc_id"])
        assert document.status == "indexed"
        assert document.s3_fileid == "file-1"
        assert document.extra_data["total_fragments"] == result["fragments"]
        assert ChunkRepository(session).count_by_doc_id(result["doc_id"]) == result["fragments"]
        session.close()

    def test_request_state_completed(self):
        init_request_state("req-1", "dlg-1", ["file-1"])
        process_document("req-1", "dlg-1", "file-1")

        state = get_request_state("req-1")
        assert state["status"] == "completed"
        assert state["documents"][0]["s3_fileid"] == "file-1"
        assert state["documents"][0]["document_metadata"]["status"] == "indexed"

    def test_raw_parse_saved(self, temp_storage):
        init_request_state("req-1", "dlg-1", ["file-1"])
        result = process_document("req-1", "dlg-1", "file-1")
        assert temp_storage.exists(f"raw_parse/{result['doc_id']}.json")

    def test_fragments_validate_against_contract(self):
        init_request_state("req-1", "dlg-1", ["file-1"])
        process_document("req-1", "dlg-1", "file-1")

        fragment = get_request_state("req-1")["documents"][0]["fragments"][0]
        assert set(fragment) == {
            "fragment_id", "type", "content", "position", "section_title",
            "confidence", "completeness", "provenance", "graph_nodes", "relations",
        }


class TestFlags:
    def test_flags_stay_with_their_fragment_when_drawing_inserted(
        self, monkeypatch, temp_storage, task_db, fake_redis
    ):
        """
        Регрессия: обработчик чертежей вставляет разбор листа целиком в
        начало списка, а пометки сопоставлялись по месту в списке — и
        уезжали на фрагмент вперёд. В `chunks.extra_data` попадал чужой
        `needs_review`, то есть признак, по которому отбирают на проверку.
        """
        parse_result = ParseResult(
            blocks=[
                ParsedBlock(type="text", text="Изделие описано здесь. " * 5,
                            page=1, bbox=[0.1, 0.1, 0.9, 0.4], order=0,
                            is_fallback=False),
                ParsedBlock(type="table", page=1, bbox=[0.1, 0.5, 0.9, 0.8],
                            order=1, is_fallback=True,
                            table_data={"headers": ["a"], "rows": [["1"]]}),
            ],
            parser_name="mineru", source_kind="vector_pdf", page_count=1,
        )
        monkeypatch.setattr(
            tasks.StorageProviderFactory, "default", staticmethod(lambda: temp_storage)
        )
        monkeypatch.setattr(
            tasks.FileLocator, "locate",
            lambda self, fileid: LocatedFile(fileid, f"documents/{fileid}.pdf", "pdf"),
        )
        monkeypatch.setattr(
            tasks.DocumentParserFactory, "get_parser",
            staticmethod(lambda storage=None: StubParser(parse_result)),
        )

        def insert_sheet(self, fragments, metadata):
            sheet = Fragment(
                fragment_id=f"{metadata['doc_id']}-frag-000",
                type="drawing",
                content=Content(image_ref="documents/f.pdf"),
                position=Position(page=1, bbox=[0.0, 0.0, 1.0, 1.0], order=0),
                confidence=0.9, completeness=0.9,
                provenance=Provenance(
                    method="vision_plus_vlm", strategy_level=7, source="drawing"
                ),
            )
            return [sheet] + list(fragments)

        monkeypatch.setattr(tasks.DrawingProcessor, "enrich_fragments", insert_sheet)

        init_request_state("req-flags", "dlg", ["file-flags"])
        process_document(
            request_id="req-flags", dialog_id="dlg", s3_fileid="file-flags"
        )

        session = task_db()
        try:
            rows = {
                row.id: (row.chunk_type, dict(row.extra_data or {}))
                for row in ChunkRepository(session).get_by_doc_id(
                    generate_doc_id("file-flags")
                )
            }
        finally:
            session.close()

        by_type = {}
        for chunk_type, extra in rows.values():
            by_type.setdefault(chunk_type, []).append(extra)

        # Вставленный лист не наследует чужих пометок.
        assert by_type["drawing"] == [{}]
        # А текст и таблица сохраняют свои.
        assert by_type["text"][0]["is_fallback"] is False
        assert by_type["table"][0]["is_fallback"] is True


@pytest.mark.usefixtures("pipeline")
class TestIdempotency:
    def test_second_create_reuses_stored_result(self, pipeline):
        init_request_state("req-1", "dlg-1", ["file-1"])
        process_document("req-1", "dlg-1", "file-1")
        assert len(pipeline.calls) == 1

        init_request_state("req-2", "dlg-1", ["file-1"])
        result = process_document("req-2", "dlg-1", "file-1")

        assert result["status"] == "already_processed"
        assert len(pipeline.calls) == 1        # повторного парсинга не было
        assert get_request_state("req-2")["status"] == "completed"
        assert get_request_state("req-2")["documents"][0]["fragments"]

    def test_update_forces_reprocessing(self, pipeline):
        init_request_state("req-1", "dlg-1", ["file-1"])
        process_document("req-1", "dlg-1", "file-1")

        init_request_state("req-2", "dlg-1", ["file-1"])
        result = process_document("req-2", "dlg-1", "file-1", operation="update")

        assert result["status"] == "success"
        assert len(pipeline.calls) == 2

    def test_reprocessing_does_not_duplicate_fragments(self, task_db):
        init_request_state("req-1", "dlg-1", ["file-1"])
        first = process_document("req-1", "dlg-1", "file-1")

        init_request_state("req-2", "dlg-1", ["file-1"])
        process_document("req-2", "dlg-1", "file-1", operation="update")

        session = task_db()
        assert ChunkRepository(session).count_by_doc_id(first["doc_id"]) == first["fragments"]
        session.close()


class TestFailures:
    def test_missing_source_file(self, monkeypatch, temp_storage, task_db, fake_redis):
        monkeypatch.setattr(
            tasks.StorageProviderFactory, "default", staticmethod(lambda: temp_storage)
        )
        monkeypatch.setattr(
            tasks.FileLocator, "locate",
            lambda self, fileid: (_ for _ in ()).throw(SourceFileNotFound("файл не найден")),
        )

        init_request_state("req-1", "dlg-1", ["file-1"])
        result = process_document("req-1", "dlg-1", "file-1")

        assert result["status"] == "error"
        state = get_request_state("req-1")
        assert state["status"] == "error"
        assert "не найден" in state["documents"][0]["error"]
        assert state["documents"][0]["document_metadata"]["status"] == "error"

    def test_parser_failure_marks_document_error(self, monkeypatch, temp_storage, task_db, fake_redis):
        class Boom:
            def parse(self, *args, **kwargs):
                raise RuntimeError("парсер лёг")

        monkeypatch.setattr(
            tasks.StorageProviderFactory, "default", staticmethod(lambda: temp_storage)
        )
        monkeypatch.setattr(
            tasks.FileLocator, "locate",
            lambda self, fileid: LocatedFile(fileid, f"documents/{fileid}.pdf", "pdf"),
        )
        monkeypatch.setattr(
            tasks.DocumentParserFactory, "get_parser", staticmethod(lambda storage=None: Boom())
        )
        # Без повторов — проверяем финальную ветку.
        monkeypatch.setattr(process_document, "max_retries", 0)

        init_request_state("req-1", "dlg-1", ["file-1"])
        result = process_document("req-1", "dlg-1", "file-1")

        assert result["status"] == "error"
        session = task_db()
        assert DocumentRepository(session).get(result["doc_id"]).status == "error"
        session.close()
        assert get_request_state("req-1")["status"] == "error"


class TestHelpers:
    def test_build_error_document_is_valid_contract(self):
        from core.models.contract import DocumentResult

        payload = build_error_document("file-1", "doc-1", "documents/a.pdf", "боль")
        DocumentResult(**payload)
        assert payload["document_metadata"]["status"] == "error"
        assert payload["error"] == "боль"

    def test_finalize_publishes_once_batch_is_complete(self, fake_redis, mocker):
        published = mocker.patch("ingest.tasks.publish_result")
        init_request_state("req-1", "dlg-1", ["f1", "f2"])

        tasks.finalize("req-1", "dlg-1", "f1", build_error_document("f1", "d1", "", "x"))
        published.assert_not_called()

        tasks.finalize("req-1", "dlg-1", "f2", build_error_document("f2", "d2", "", "x"))
        published.assert_called_once()

    def test_finalize_publishes_only_once_on_repeat(self, fake_redis, mocker):
        """acks_late допускает повтор задачи — подписчик не должен получить дубль."""
        published = mocker.patch("ingest.tasks.publish_result")
        init_request_state("req-2", "dlg-1", ["f1"])

        document = build_error_document("f1", "d1", "", "x")
        tasks.finalize("req-2", "dlg-1", "f1", document)
        tasks.finalize("req-2", "dlg-1", "f1", document)
        tasks.finalize("req-2", "dlg-1", "f1", document)

        published.assert_called_once()
