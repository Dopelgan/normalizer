"""Кэш результата лестницы по содержимому файла (LADDER_RESULT_CACHE)."""

import json
import logging
import os
import re
import threading
import time

import pytest

from core import telemetry
from core.config import settings
from core.ladder import result_cache
from core.models.parse_result import ParsedBlock, ParseResult
from core.providers.file_locator import LocatedFile
from core.repositories import ChunkRepository, DocumentRepository
from core.result_aggregator import init_request_state
from ingest import tasks
from ingest.tasks import process_document

SHA = "a" * 64


def _result(prefix="assets/doc-a/", name="scan.pdf", **extra):
    reserve = ParseResult(
        blocks=[ParsedBlock(type="text", text="запас распознавания", page=1)],
        is_fallback=True, parser_name="tesseract_full", source_kind="scanned_pdf",
        page_count=1,
    )
    data = dict(
        blocks=[
            ParsedBlock(type="text", text="Корпус Ø12 ±0,1", page=1,
                        bbox=[0.1, 0.2, 0.3, 0.4], confidence=0.873, method="ocr"),
            ParsedBlock(type="image", image_ref=f"{prefix}fig1.jpg", page=2),
        ],
        parser_name="mineru", source_kind="scanned_pdf", page_count=2,
        source_page_count=2,
        raw_output={"results": {name: {"md_content": "текст"}}},
        text_layer_chars={1: 0, 2: 5},
        text_layer_stats={"recovered": 3},
        ladder={"chosen_level": 5, "attempts": [{"level": 4, "error": None},
                                                {"level": 5, "error": None}]},
        fallback_reserve=reserve,
    )
    data.update(extra)
    return ParseResult(**data)


class CountingParser:
    def __init__(self, make):
        self.make = make
        self.calls = 0

    def parse(self, uri, file_type, metadata=None):
        self.calls += 1
        return self.make(metadata or {}, uri)


def _meta(doc_id="doc-a"):
    return {"doc_id": doc_id, "asset_prefix": f"assets/{doc_id}/"}


@pytest.fixture
def cache_on(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "LADDER_RESULT_CACHE", True)
    monkeypatch.setattr(settings, "LADDER_RESULT_CACHE_DIR", str(tmp_path / "ladder-cache"))
    monkeypatch.setattr(settings, "LADDER_RESULT_CACHE_DAYS", 30)
    monkeypatch.setattr(result_cache, "_last_prune", time.time())
    return tmp_path / "ladder-cache"


def _mineru_like(storage):
    """Как MinerU: картинка пишется под префикс документа, имя — из uri."""
    def make(metadata, uri):
        prefix = metadata["asset_prefix"]
        storage.write_file(f"{prefix}fig1.jpg", b"\xff\xd8 picture")
        return _result(prefix=prefix, name=uri.rsplit("/", 1)[-1])
    return make


def _dump(result):
    reserve = result.fallback_reserve
    return (json.dumps(result.model_dump(mode="json"), sort_keys=True, ensure_ascii=False),
            json.dumps(reserve.model_dump(mode="json"), sort_keys=True, ensure_ascii=False)
            if reserve else None)


def _parse(parser, storage, meta=None, uri="documents/scan.pdf", file_type="pdf", sha=SHA):
    return result_cache.parse(parser, storage, uri, file_type, meta or _meta(), sha)


class TestHitEqualsMiss:
    def test_hit_skips_ladder_and_is_identical(self, cache_on, temp_storage):
        parser = CountingParser(_mineru_like(temp_storage))
        with telemetry.collect() as first_timings:
            miss = _parse(parser, temp_storage)
        with telemetry.collect() as timings:
            hit = _parse(parser, temp_storage)
        assert parser.calls == 1
        assert _dump(hit) == _dump(miss)
        assert hit.fallback_reserve is not None
        assert hit.text_layer_chars == {1: 0, 2: 5}
        assert timings.get(result_cache.HIT_STAGE) == 1
        assert result_cache.HIT_STAGE not in first_timings

    def test_flag_off_runs_ladder_every_time(self, cache_on, temp_storage, monkeypatch):
        monkeypatch.setattr(settings, "LADDER_RESULT_CACHE", False)
        parser = CountingParser(_mineru_like(temp_storage))
        _parse(parser, temp_storage)
        _parse(parser, temp_storage)
        assert parser.calls == 2
        assert not cache_on.exists()

    def test_no_hash_means_no_cache(self, cache_on, temp_storage):
        parser = CountingParser(_mineru_like(temp_storage))
        _parse(parser, temp_storage, sha=None)
        _parse(parser, temp_storage, sha=None)
        assert parser.calls == 2


class TestKey:
    def test_changed_ladder_setting_is_a_miss(self, cache_on, temp_storage, monkeypatch):
        parser = CountingParser(_mineru_like(temp_storage))
        _parse(parser, temp_storage)
        monkeypatch.setattr(settings, "LADDER_SCORE_THRESHOLD", 0.5)
        _parse(parser, temp_storage)
        assert parser.calls == 2

    def test_other_bytes_or_type_is_a_miss(self, cache_on, temp_storage):
        parser = CountingParser(_mineru_like(temp_storage))
        _parse(parser, temp_storage)
        _parse(parser, temp_storage, sha="b" * 64)
        _parse(parser, temp_storage, file_type="jpg")
        assert parser.calls == 3

    def test_code_change_changes_key(self, monkeypatch):
        before = result_cache.cache_key(SHA, "pdf", {})
        monkeypatch.setattr(result_cache, "_code_digest", "другой код")
        assert result_cache.cache_key(SHA, "pdf", {}) != before

    def test_every_setting_the_ladder_reads_is_classified(self):
        # Новая настройка в коде лестницы должна попасть либо в ключ, либо
        # в явный список «результат не меняет» — иначе кэш отдаст результат,
        # посчитанный при другом её значении.
        read = set()
        for path in result_cache.ladder_sources():
            read |= set(re.findall(r"settings\.([A-Z][A-Z0-9_]+)",
                                   path.read_text(encoding="utf-8")))
        key, aside = set(result_cache.KEY_SETTINGS), set(result_cache.NOT_IN_KEY)
        assert not key & aside
        assert read - key - aside == set()
        assert all(hasattr(settings, name) for name in key | aside)

    def test_sources_cover_the_ladder_providers(self):
        names = {p.name for p in result_cache.ladder_sources()}
        assert {"router.py", "recognition.py", "mineru_parser.py",
                "tesseract_fallback.py", "text_layer.py", "config.py"} <= names

    def test_secrets_are_not_in_the_key(self):
        assert not {"QWEN_API_KEY", "S3_ACCESS_KEY", "S3_SECRET_KEY"} & set(
            result_cache.KEY_SETTINGS)


class TestNotStored:
    @pytest.mark.parametrize("extra", [
        {"degraded": ["LAYOUT_UNAVAILABLE"]},
        {"failed_pages": [2]},
        {"ladder": {"attempts": [{"level": 5, "error": "409 нехватка памяти"}]}},
    ])
    def test_unclean_result_is_not_cached(self, cache_on, temp_storage, extra):
        parser = CountingParser(lambda meta, uri: _result(**extra))
        _parse(parser, temp_storage)
        _parse(parser, temp_storage)
        assert parser.calls == 2
        assert not list(cache_on.rglob("*.json"))


class TestStore:
    def test_concurrent_writers_leave_one_valid_file(self, cache_on, temp_storage):
        parser = CountingParser(_mineru_like(temp_storage))
        barrier = threading.Barrier(8)

        def worker():
            barrier.wait()
            _parse(parser, temp_storage)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        files = list(cache_on.rglob("*"))
        entries = [f for f in files if f.is_file()]
        assert len(entries) == 1 and entries[0].suffix == ".json"
        json.loads(entries[0].read_text(encoding="utf-8"))
        before = parser.calls
        _parse(parser, temp_storage)
        assert parser.calls == before

    def test_corrupted_entry_is_a_miss_with_one_log_line(self, cache_on, temp_storage, caplog):
        parser = CountingParser(_mineru_like(temp_storage))
        _parse(parser, temp_storage)
        (entry,) = list(cache_on.rglob("*.json"))
        entry.write_text('{"format": "ladder-cache-v1", "key": ', encoding="utf-8")
        with caplog.at_level(logging.WARNING, logger=result_cache.__name__):
            _parse(parser, temp_storage)
        assert parser.calls == 2
        assert len([r for r in caplog.records if r.name == result_cache.__name__]) == 1

    def test_expired_entry_is_a_miss_and_pruned(self, cache_on, temp_storage, monkeypatch):
        parser = CountingParser(_mineru_like(temp_storage))
        _parse(parser, temp_storage)
        (entry,) = list(cache_on.rglob("*.json"))
        old = time.time() - 31 * 86400
        os.utime(entry, (old, old))
        assert result_cache.prune() == 1
        assert not entry.exists()

    def test_entry_older_than_retention_is_not_served(self, cache_on, temp_storage, monkeypatch):
        parser = CountingParser(_mineru_like(temp_storage))
        _parse(parser, temp_storage)
        monkeypatch.setattr(settings, "LADDER_RESULT_CACHE_DAYS", 0)
        _parse(parser, temp_storage)
        assert parser.calls == 2


class TestOtherDocument:
    def test_duplicate_gets_its_own_assets_and_name(self, cache_on, temp_storage):
        parser = CountingParser(_mineru_like(temp_storage))
        _parse(parser, temp_storage, meta=_meta("doc-a"), uri="documents/a/scan.pdf")
        hit = _parse(parser, temp_storage, meta=_meta("doc-b"), uri="documents/b/copy.pdf")
        assert parser.calls == 1
        # То, что дал бы свежий разбор второго файла.
        fresh = _mineru_like(temp_storage)(_meta("doc-b"), "documents/b/copy.pdf")
        assert _dump(hit) == _dump(fresh)
        assert temp_storage.exists("assets/doc-b/fig1.jpg")
        assert "doc-a" not in _dump(hit)[0]

    def test_copied_asset_has_same_bytes(self, cache_on, temp_storage):
        parser = CountingParser(_mineru_like(temp_storage))
        _parse(parser, temp_storage, meta=_meta("doc-a"))
        temp_storage.delete_file("assets/doc-b/fig1.jpg") if temp_storage.exists(
            "assets/doc-b/fig1.jpg") else None
        _parse(parser, temp_storage, meta=_meta("doc-b"))
        with temp_storage.get_stream("assets/doc-b/fig1.jpg") as stream:
            assert stream.read() == b"\xff\xd8 picture"

    def test_missing_source_asset_is_a_miss(self, cache_on, temp_storage):
        parser = CountingParser(_mineru_like(temp_storage))
        _parse(parser, temp_storage, meta=_meta("doc-a"))
        temp_storage.delete_file("assets/doc-a/fig1.jpg")
        _parse(parser, temp_storage, meta=_meta("doc-b"))
        assert parser.calls == 2

    def test_trace_of_other_document_is_a_miss(self, cache_on, temp_storage):
        def make(metadata, uri):
            block = ParsedBlock(type="text", text=f"источник {uri}", page=1)
            return ParseResult(blocks=[block], page_count=1)
        parser = CountingParser(make)
        _parse(parser, temp_storage, meta=_meta("doc-a"), uri="documents/a/scan.pdf")
        _parse(parser, temp_storage, meta=_meta("doc-b"), uri="documents/b/scan.pdf")
        assert parser.calls == 2


# ---------------------------------------------------------------- задача
class StubParser:
    def __init__(self, result):
        self.result = result
        self.calls = 0

    def parse(self, uri, file_type, metadata=None):
        self.calls += 1
        return self.result.model_copy(deep=True)


@pytest.fixture
def task_pipeline(monkeypatch, temp_storage, task_db, fake_redis, cache_on):
    parse_result = ParseResult(
        blocks=[ParsedBlock(type="text", text="Настоящий документ описывает изделие. " * 3,
                            page=1, bbox=[0.1, 0.1, 0.9, 0.5], order=0)],
        parser_name="mineru", source_kind="vector_pdf", page_count=1, source_page_count=1,
    )
    parser = StubParser(parse_result)
    for name in ("file-1", "file-2", "file-3"):
        payload = b"%PDF-1.4 same bytes" if name != "file-3" else b"%PDF-1.4 other"
        temp_storage.write_file(f"documents/{name}.pdf", payload)
    monkeypatch.setattr(tasks.StorageProviderFactory, "default", staticmethod(lambda: temp_storage))
    monkeypatch.setattr(
        tasks.FileLocator, "locate",
        lambda self, fileid: LocatedFile(fileid, f"documents/{fileid}.pdf", "pdf"),
    )
    monkeypatch.setattr(tasks.DocumentParserFactory, "get_parser",
                        staticmethod(lambda storage=None: parser))
    monkeypatch.setattr(tasks.DrawingProcessor, "enrich_fragments",
                        lambda self, fragments, metadata: fragments)
    return parser


@pytest.fixture
def task_db(monkeypatch):
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.pool import StaticPool

    from core.db.models import Base

    engine = create_engine("sqlite://", connect_args={"check_same_thread": False},
                           poolclass=StaticPool)
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    monkeypatch.setattr(tasks, "SessionLocal", factory)
    return factory


def _chunks(task_db, doc_id):
    session = task_db()
    try:
        rows = ChunkRepository(session).get_by_doc_id(doc_id)
        return [(r.id, r.chunk_type, r.content, r.position, r.provenance, r.confidence,
                 r.completeness, r.extracted_facts, r.order_index, r.extra_data)
                for r in sorted(rows, key=lambda row: row.id)]
    finally:
        session.close()


def _extra(task_db, doc_id):
    session = task_db()
    try:
        document = DocumentRepository(session).get(doc_id)
        return document.s3_fileid, document.file_hash, dict(document.extra_data)
    finally:
        session.close()


class TestTask:
    def test_exact_duplicate_does_not_run_ladder_again(self, task_pipeline, task_db):
        init_request_state("req-1", ["file-1", "file-2", "file-3"])
        first = process_document("req-1", "file-1")
        second = process_document("req-1", "file-2")
        process_document("req-1", "file-3")
        assert task_pipeline.calls == 2          # file-2 — из кэша, file-3 — другие байты
        assert first["doc_id"] != second["doc_id"]
        s3_a, hash_a, extra_a = _extra(task_db, first["doc_id"])
        s3_b, hash_b, extra_b = _extra(task_db, second["doc_id"])
        assert (s3_a, s3_b) == ("file-1", "file-2")
        assert hash_a == hash_b and hash_a
        assert extra_b["timings_ms"].get(result_cache.HIT_STAGE) == 1
        assert result_cache.HIT_STAGE not in extra_a["timings_ms"]
        texts = lambda rows: sorted(json.dumps(r[2], sort_keys=True) for r in rows)  # noqa: E731
        assert texts(_chunks(task_db, first["doc_id"])) == texts(_chunks(task_db, second["doc_id"]))

    def test_reprocess_hit_is_identical_to_miss(self, task_pipeline, task_db):
        init_request_state("req-1", ["file-1"])
        first = process_document("req-1", "file-1")
        before = _chunks(task_db, first["doc_id"])
        _, _, extra_before = _extra(task_db, first["doc_id"])
        init_request_state("req-2", ["file-1"])
        process_document("req-2", "file-1", operation="update")
        assert task_pipeline.calls == 1
        assert _chunks(task_db, first["doc_id"]) == before
        _, _, extra_after = _extra(task_db, first["doc_id"])
        for extra in (extra_before, extra_after):
            extra.pop("timings_ms")
            extra.pop("last_operation")
        assert extra_after == extra_before

    def test_flag_off_keeps_old_behaviour(self, task_pipeline, task_db, monkeypatch):
        monkeypatch.setattr(settings, "LADDER_RESULT_CACHE", False)
        init_request_state("req-1", ["file-1", "file-2"])
        process_document("req-1", "file-1")
        process_document("req-1", "file-2")
        assert task_pipeline.calls == 2
