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
        source_page_count=1,
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
        init_request_state("req-1", ["file-1"])
        result = process_document("req-1", "file-1")

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
        init_request_state("req-1", ["file-1"])
        process_document("req-1", "file-1")

        state = get_request_state("req-1")
        assert state["status"] == "completed"
        assert state["documents"][0]["s3_fileid"] == "file-1"
        assert state["documents"][0]["document_metadata"]["status"] == "indexed"

    def test_raw_parse_saved(self, temp_storage):
        init_request_state("req-1", ["file-1"])
        result = process_document("req-1", "file-1")
        assert temp_storage.exists(f"raw_parse/{result['doc_id']}.json")

    def test_fragments_validate_against_contract(self):
        init_request_state("req-1", ["file-1"])
        process_document("req-1", "file-1")

        document = get_request_state("req-1")["documents"][0]
        fragment = document["fragments"][0]
        assert set(fragment) == {
            "fragment_id", "type", "content", "position", "section_title",
            "confidence", "completeness", "provenance", "extracted_facts",
            "graph_nodes", "relations",
            "section_path", "review_required", "warnings", "missing_regions",
            "completeness_mechanisms", "structure_score",
        }
        # Поля контракта уровня документа приезжают вместе с фрагментами.
        assert document["status"] == "completed"
        assert document["mime_type"]
        assert document["normalized_content_hash"].startswith("sha256:")
        assert document["processed_pages"]
        assert document["failed_pages"] == []
        assert fragment["provenance"]["source_file_id"] == "file-1"
        assert fragment["provenance"]["source_page"] == fragment["position"]["page"]


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

        init_request_state("req-flags", ["file-flags"])
        process_document(request_id="req-flags", s3_fileid="file-flags")

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
        assert "is_fallback" not in by_type["drawing"][0]
        assert by_type["drawing"][0]["contract_quality"]["completeness_unknown"] is True
        # А текст и таблица сохраняют свои.
        assert by_type["text"][0]["is_fallback"] is False
        assert by_type["table"][0]["is_fallback"] is True


@pytest.mark.usefixtures("pipeline")
class TestIdempotency:
    def test_second_create_reuses_stored_result(self, pipeline):
        init_request_state("req-1", ["file-1"])
        process_document("req-1", "file-1")
        assert len(pipeline.calls) == 1

        init_request_state("req-2", ["file-1"])
        result = process_document("req-2", "file-1")

        assert result["status"] == "already_processed"
        assert len(pipeline.calls) == 1        # повторного парсинга не было
        assert get_request_state("req-2")["status"] == "completed"
        assert get_request_state("req-2")["documents"][0]["fragments"]

    def test_update_forces_reprocessing(self, pipeline):
        init_request_state("req-1", ["file-1"])
        process_document("req-1", "file-1")

        init_request_state("req-2", ["file-1"])
        result = process_document("req-2", "file-1", operation="update")

        assert result["status"] == "success"
        assert len(pipeline.calls) == 2

    def test_reprocessing_does_not_duplicate_fragments(self, task_db):
        init_request_state("req-1", ["file-1"])
        first = process_document("req-1", "file-1")

        init_request_state("req-2", ["file-1"])
        process_document("req-2", "file-1", operation="update")

        session = task_db()
        assert ChunkRepository(session).count_by_doc_id(first["doc_id"]) == first["fragments"]
        session.close()


@pytest.mark.usefixtures("pipeline")
class TestStructuralProfile:
    @pytest.mark.parametrize('page_count,unknown', [(3, [2, 3]), (0, [])])
    def test_unknown_pages_survive_fresh_cached_and_request_results(self, pipeline, page_count, unknown):
        pipeline.result.page_count = page_count
        pipeline.result.source_page_count = page_count or None
        init_request_state("coverage-first", ["file-coverage"])
        process_document("coverage-first", "file-coverage")
        first = get_request_state("coverage-first")
        init_request_state("coverage-again", ["file-coverage"])
        result = process_document("coverage-again", "file-coverage")
        again = get_request_state("coverage-again")
        assert result["status"] == "already_processed"
        assert len(pipeline.calls) == 1
        assert first["status"] == again["status"] == "partial"
        assert first["documents"] == again["documents"]
        document = first["documents"][0]
        assert document["processed_pages"] == [1]
        assert document["failed_pages"] == []
        assert document["unknown_pages"] == unknown
        if page_count:
            assert "PAGE_COVERAGE_UNKNOWN: page 2" in document["warnings"]
        else:
            assert document["pages"] is None
            assert "PAGE_COUNT_UNKNOWN" in document["warnings"]

    def test_legacy_coverage_report_is_reprocessed(self, pipeline, task_db):
        init_request_state("coverage-old", ["file-old-coverage"])
        result = process_document("coverage-old", "file-old-coverage")
        session = task_db()
        with session.begin():
            DocumentRepository(session).update_extra(result["doc_id"], {"page_coverage_version": None})
        session.close()
        init_request_state("coverage-new", ["file-old-coverage"])
        assert process_document("coverage-new", "file-old-coverage")["status"] == "success"
        assert len(pipeline.calls) == 2

    def test_current_profile_and_source_coordinates_survive_cached_response(self):
        from core.processing_contract import PROCESSING_PROFILE_VERSION
        init_request_state("profile-first", ["file-profile"])
        process_document("profile-first", "file-profile")
        first = get_request_state("profile-first")["documents"][0]
        init_request_state("profile-again", ["file-profile"])
        result = process_document("profile-again", "file-profile")
        again = get_request_state("profile-again")["documents"][0]
        assert result["status"] == "already_processed"
        assert again == first
        assert first["processing_profile_version"] == PROCESSING_PROFILE_VERSION
        assert first["schema_version"] == "2.0"
        assert first["fragments"][0]["position"]["text_offsets_exact"]

    def test_old_processing_profile_is_reprocessed(self, pipeline, task_db):
        init_request_state("old-first", ["file-old"])
        result = process_document("old-first", "file-old")
        session = task_db()
        with session.begin():
            DocumentRepository(session).update_extra(result["doc_id"], {
                "processing_profile_version": "legacy-chunking",
            })
        session.close()
        init_request_state("old-again", ["file-old"])
        assert process_document("old-again", "file-old")["status"] == "success"
        assert len(pipeline.calls) == 2

    def test_partial_drawing_quality_survives_cached_response(self, monkeypatch):
        from core.models.contract import DrawingField

        def add_drawing(self, fragments, metadata):
            return fragments + [Fragment(
                fragment_id="", type="drawing",
                content=Content(structured_drawing_fields=[DrawingField(
                    category="size", value="20", confidence=0.95,
                )]),
                position=Position(page=1), confidence=0.95, completeness=None,
                provenance=Provenance(method="vision_plus_vlm", strategy_level=7, source="drawing"),
            )]

        monkeypatch.setattr(tasks.DrawingProcessor, "enrich_fragments", add_drawing)
        init_request_state("partial-first", ["file-partial"])
        process_document("partial-first", "file-partial")
        first = get_request_state("partial-first")["documents"][0]
        init_request_state("partial-again", ["file-partial"])
        process_document("partial-again", "file-partial")
        again = get_request_state("partial-again")["documents"][0]
        assert first == again
        assert first["status"] == "partial" and first["review_required"]
        assert first["fragments"][-1]["completeness"] is None
        assert first["fragments"][-1]["extracted_facts"][0]["confirmed"]

    @pytest.mark.parametrize("missing,status", [([], "completed"),
                                                ([{"kind": "table", "bbox": [0, 0, 1, 1], "reason": "region_not_read"}], "partial")])
    def test_unmeasured_drawing_is_not_partial_under_flag(self, monkeypatch, missing, status):
        from core.models.contract import DrawingField

        def add_drawing(self, fragments, metadata):
            return fragments + [Fragment(
                fragment_id="", type="drawing",
                content=Content(structured_drawing_fields=[DrawingField(
                    category="size", value="20", confidence=0.5,
                )]),
                position=Position(page=1), confidence=0.5, completeness=None,
                warnings=["DRAWING_COMPLETENESS_UNKNOWN"], missing_regions=missing,
                provenance=Provenance(method="vision_plus_vlm", strategy_level=7, source="drawing"),
            )]

        monkeypatch.setattr(tasks.settings, "DRAWING_UNMEASURED_NOT_PARTIAL", True)
        monkeypatch.setattr(tasks.DrawingProcessor, "enrich_fragments", add_drawing)
        name = f"unmeasured-{status}"
        init_request_state(name, [f"file-{name}"])
        process_document(name, f"file-{name}")
        document = get_request_state(name)["documents"][0]
        assert document["status"] == status
        # Фрагмент остаётся на проверке: флаг меняет только статус документа.
        assert document["fragments"][-1]["review_required"]


class TestFailures:
    def test_missing_source_file(self, monkeypatch, temp_storage, task_db, fake_redis):
        monkeypatch.setattr(
            tasks.StorageProviderFactory, "default", staticmethod(lambda: temp_storage)
        )
        monkeypatch.setattr(
            tasks.FileLocator, "locate",
            lambda self, fileid: (_ for _ in ()).throw(SourceFileNotFound("файл не найден")),
        )

        init_request_state("req-1", ["file-1"])
        result = process_document("req-1", "file-1")

        assert result["status"] == "error"
        state = get_request_state("req-1")
        assert state["status"] == "error"
        error = state["documents"][0]["error"]
        assert "не найден" in error["message"]
        assert error["reason_code"] == "SOURCE_NOT_FOUND"
        assert error["retryable"] is False
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

        init_request_state("req-1", ["file-1"])
        result = process_document("req-1", "file-1")

        assert result["status"] == "error"
        session = task_db()
        assert DocumentRepository(session).get(result["doc_id"]).status == "error"
        session.close()
        assert get_request_state("req-1")["status"] == "error"


class TestOriginalNameInLog:
    """
    Имя загруженного файла в логе: по снимку с кластером человек не скажет,
    какой из файлов сейчас разбирается.
    """

    @pytest.mark.parametrize("fileid, expected", [
        ("cluster-a/Чертёж вилки АБВГ.123456.001.pdf", "Чертёж вилки АБВГ.123456.001.pdf"),
        ("Договор 1507-2-26.docx", "Договор 1507-2-26.docx"),
        ("cluster-a/%D0%A2%D0%97.pdf", "ТЗ.pdf"),            # имя пришло URL-encoded
        ("0b3f8c1e-1111-2222-3333-444455556666", ""),         # вложение чата: имени нет
    ])
    def test_readable_name_comes_from_the_fileid(self, fileid, expected):
        assert tasks.original_name(fileid) == expected

    def test_log_line_names_the_uploaded_file(self, pipeline, fake_redis, caplog):
        import logging

        fileid = "cluster-a/Чертёж вилки АБВГ.123456.001.pdf"
        init_request_state("req-name", [fileid])
        with caplog.at_level(logging.INFO, logger="ingest.tasks"):
            process_document("req-name", fileid)
        line = next(m for m in (r.getMessage() for r in caplog.records)
                    if m.startswith("Документ ") and "снимок" in m)

        assert f"снимок {fileid}" in line
        assert "исходное имя «Чертёж вилки АБВГ.123456.001.pdf»" in line
        assert "тип .pdf" in line


class TestSnapshotNameFromMetadata:
    """
    Снимок под разбор RAG кладёт с транслитерированным именем в ключе, а
    настоящее — в метаданные объекта (`filename`, percent-encoded). Имя
    берётся оттуда, но отсутствие метаданных и сбой хранилища не должны
    ни ломать разбор, ни превращать имя в «неизвестно».
    """

    # Ключ снимка ровно такой, какой приходит с прода.
    SNAPSHOT = (
        "rag-parser-snapshots/4f1c/_____421451.003-01__4__.1_____._34_.jpg"
    )
    FALLBACK = "_____421451.003-01__4__.1_____._34_.jpg"
    URI = "s3://bucket/rag-parser-snapshots/4f1c/_____421451.003-01__4__.1_____._34_.jpg"

    class Storage:
        """Хранилище, отдающее заданные метаданные и считающее запросы."""

        def __init__(self, meta=None, boom=None):
            self._meta = meta
            self._boom = boom
            self.calls = []

        def metadata(self, uri):
            self.calls.append(uri)
            if self._boom is not None:
                raise self._boom
            return self._meta

    def test_metadata_name_wins_over_the_key(self):
        storage = self.Storage({
            # quote("Сборочный чертёж 421451.003-01.jpg", safe="")
            "filename": "%D0%A1%D0%B1%D0%BE%D1%80%D0%BE%D1%87%D0%BD%D1%8B%D0%B9"
                        "%20%D1%87%D0%B5%D1%80%D1%82%D1%91%D0%B6%20421451.003-01.jpg",
        })

        name = tasks.original_name(self.SNAPSHOT, storage, self.URI)

        assert name == "Сборочный чертёж 421451.003-01.jpg"
        assert storage.calls == [self.URI]

    def test_metadata_key_case_is_ignored(self):
        # boto3 нормализует x-amz-meta-* в нижний регистр, но полагаться на
        # регистр чужого ответа незачем.
        storage = self.Storage({"Filename": "%D0%A2%D0%97.pdf"})
        assert tasks.original_name(self.SNAPSHOT, storage, self.URI) == "ТЗ.pdf"

    @pytest.mark.parametrize("meta", [None, {}, {"filename": ""}, {"filename": "   "}])
    def test_no_metadata_keeps_the_old_name(self, meta):
        # Старые снимки метаданных не несут, а пустое и слишком длинное имя
        # RAG не записывает сознательно. Откат — имя из ключа, не «неизвестно».
        storage = self.Storage(meta)

        name = tasks.original_name(self.SNAPSHOT, storage, self.URI)

        assert name == self.FALLBACK
        assert name and name != "неизвестно"

    def test_storage_error_keeps_the_old_name_without_raising(self):
        storage = self.Storage(boom=RuntimeError("S3 недоступен"))

        assert tasks.original_name(self.SNAPSHOT, storage, self.URI) == self.FALLBACK

    def test_storage_error_is_logged_at_debug_not_error(self, caplog):
        import logging

        storage = self.Storage(boom=RuntimeError("S3 недоступен"))
        with caplog.at_level(logging.DEBUG, logger="ingest.tasks"):
            tasks.original_name(self.SNAPSHOT, storage, self.URI)

        records = [r for r in caplog.records if "Метаданные снимка" in r.getMessage()]
        assert records and all(r.levelno == logging.DEBUG for r in records)

    def test_real_path_document_does_not_ask_for_metadata(self):
        # 193 документа с реальным путём уже пишутся в лог верно; лишний
        # head_object им не нужен.
        storage = self.Storage({"filename": "%D0%A2%D0%97.pdf"})
        fileid = "cluster-a/Чертёж вилки АБВГ.123456.001.pdf"

        name = tasks.original_name(fileid, storage, f"s3://bucket/{fileid}")

        assert name == "Чертёж вилки АБВГ.123456.001.pdf"
        assert storage.calls == []

    def test_log_line_shows_the_decoded_snapshot_name(
        self, pipeline, temp_storage, fake_redis, monkeypatch, caplog
    ):
        import logging

        monkeypatch.setattr(
            tasks.FileLocator, "locate",
            lambda self, fileid: LocatedFile(fileid, f"s3://bucket/{fileid}", "jpg"),
        )
        monkeypatch.setattr(
            type(temp_storage), "metadata",
            lambda self, uri: {"filename": "%D0%A2%D0%97%20%E2%84%961.jpg"},
            raising=False,
        )

        init_request_state("req-snapshot", [self.SNAPSHOT])
        with caplog.at_level(logging.INFO, logger="ingest.tasks"):
            process_document("req-snapshot", self.SNAPSHOT)
        line = next(m for m in (r.getMessage() for r in caplog.records)
                    if m.startswith("Документ ") and "снимок" in m)

        assert f"снимок {self.SNAPSHOT}" in line
        assert "исходное имя «ТЗ №1.jpg»" in line
        # Транслитерация осталась только в самом ключе снимка, не в имени.
        assert f"«{self.FALLBACK}»" not in line


class TestHelpers:
    def test_build_error_document_is_valid_contract(self):
        from core.models.contract import DocumentResult

        payload = build_error_document("file-1", "doc-1", "documents/a.pdf", "боль")
        DocumentResult(**payload)
        assert payload["document_metadata"]["status"] == "error"
        assert payload["status"] == "error"
        assert payload["error"]["message"] == "боль"
        assert payload["error"]["reason_code"] == "INTERNAL_ERROR"

    def test_error_document_from_exception_carries_reason_code(self):
        payload = build_error_document(
            "file-1", "doc-1", "", SourceFileNotFound("файл не найден"), attempt=2
        )
        assert payload["error"] == {
            "reason_code": "SOURCE_NOT_FOUND",
            "message": "файл не найден",
            "retryable": False,
            "attempt": 2,
        }

    def test_finalize_publishes_once_batch_is_complete(self, fake_redis, mocker):
        published = mocker.patch("ingest.tasks.publish_result")
        init_request_state("req-1", ["f1", "f2"])

        tasks.finalize("req-1", "f1", build_error_document("f1", "d1", "", "x"))
        published.assert_not_called()

        tasks.finalize("req-1", "f2", build_error_document("f2", "d2", "", "x"))
        published.assert_called_once()

    def test_finalize_publishes_only_once_on_repeat(self, fake_redis, mocker):
        """acks_late допускает повтор задачи — подписчик не должен получить дубль."""
        published = mocker.patch("ingest.tasks.publish_result")
        init_request_state("req-2", ["f1"])

        document = build_error_document("f1", "d1", "", "x")
        tasks.finalize("req-2", "f1", document)
        tasks.finalize("req-2", "f1", document)
        tasks.finalize("req-2", "f1", document)

        published.assert_called_once()
