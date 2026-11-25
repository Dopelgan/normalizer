"""Удаление документа отовсюду: ручка, хранилища, идемпотентность (ACL-02)."""

import json

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from core.config import settings
from core.db.models import Base, ChunkDB, DocumentDB, IntakeDecisionDB
from core.document_delete import delete_document, doc_id_for
from core.ladder import result_cache
from core.providers.storage import LocalStorageProvider
from core.result_aggregator import init_request_state, put_document_result
from ingest import api as ingest_api

FAKE_TOKEN = "test-token-not-a-secret"
FILEID = "documents/protocol-17.pdf"
OTHER = "documents/protocol-17-copy.pdf"
HASH = "a" * 64


@pytest.fixture
def session_factory(monkeypatch):
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    monkeypatch.setattr("core.db.session.SessionLocal", factory)
    return factory


@pytest.fixture
def storage(tmp_path, monkeypatch):
    provider = LocalStorageProvider(base_path=str(tmp_path / "artifacts"))
    monkeypatch.setattr(settings, "ASSETS_PREFIX", "assets/")
    monkeypatch.setattr(
        "core.providers.storage.StorageProviderFactory.default", lambda: provider,
    )
    return provider


@pytest.fixture
def cache_root(tmp_path, monkeypatch):
    root = tmp_path / "ladder-cache"
    monkeypatch.setattr(settings, "LADDER_RESULT_CACHE_DIR", str(root))
    return root


def _cache_entry(root, key, doc_id, file_sha):
    path = root / key[:2] / f"{key}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "format": "ladder-cache-v1", "key": key, "created": 0.0,
        "doc_id": doc_id, "file": file_sha, "uri": "s3://kb/x.pdf",
        "result": {"text": "секретный текст"},
    }), encoding="utf-8")
    return path


def _seed(session_factory, storage, fake_redis, cache_root, *, fileid=FILEID, file_hash=HASH):
    doc_id = doc_id_for(fileid)
    session = session_factory()
    session.add(DocumentDB(
        id=doc_id, s3_fileid=fileid, source_path=f"s3://kb/{fileid}",
        file_hash=file_hash, status="indexed", extra_data={},
    ))
    session.add(ChunkDB(
        id=f"{doc_id}-f1", doc_id=doc_id, chunk_type="text",
        content={"text": "секретный текст"}, provenance={"method": "text_layer"},
    ))
    session.add(IntakeDecisionDB(
        id=f"dec-{doc_id}", s3_fileid=fileid, stage="data_gateway",
        outcome="accept", reason="ok",
    ))
    session.commit()
    session.close()
    storage.write_file(f"assets/{doc_id}/page1.jpg", b"jpeg")
    init_request_state("req-1", [fileid], "create", event_id="ev-1")
    put_document_result("req-1", fileid, {"s3_fileid": fileid, "fragments": ["секрет"]})
    entry = _cache_entry(cache_root, doc_id[4:].ljust(64, "0"), doc_id, file_hash)
    return doc_id, entry


@pytest.fixture
def enabled(monkeypatch):
    monkeypatch.setattr(settings, "NORMALIZER_DELETE_ENABLED", True)


@pytest.fixture
def client(fake_redis, session_factory, storage, cache_root):
    with TestClient(ingest_api.app) as test_client:
        yield test_client


def _counts(session_factory):
    session = session_factory()
    try:
        return (
            session.query(DocumentDB).count(),
            session.query(ChunkDB).count(),
            session.query(IntakeDecisionDB).count(),
        )
    finally:
        session.close()


def test_doc_id_matches_ingest():
    from ingest.tasks import generate_doc_id

    assert doc_id_for(FILEID) == generate_doc_id(FILEID)


def test_disabled_flag_answers_501(client, session_factory, storage, fake_redis, cache_root):
    _seed(session_factory, storage, fake_redis, cache_root)
    response = client.delete(f"/internal/v1/documents/{FILEID}")
    assert response.status_code == 501
    assert _counts(session_factory) == (1, 1, 1)


def test_delete_removes_every_store(enabled, client, session_factory, storage, fake_redis, cache_root):
    doc_id, entry = _seed(session_factory, storage, fake_redis, cache_root)

    response = client.delete(f"/internal/v1/documents/{FILEID}")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "deleted"
    assert body["doc_id"] == doc_id
    assert _counts(session_factory) == (0, 0, 0)
    assert not storage.exists(f"assets/{doc_id}/page1.jpg")
    assert not entry.exists()
    # Поле файла заменено пустым документом; мета и счётчики запроса на месте.
    assert fake_redis.exists("req:req-1:meta")
    stored = json.loads(fake_redis.hget("req:req-1:docs", FILEID))
    assert stored["fragments"] == [] and "секрет" not in json.dumps(stored, ensure_ascii=False)
    assert fake_redis.exists(f"deleted:{doc_id}")
    assert (body["documents"], body["chunks"], body["intake_decisions"]) == (1, 1, 1)
    assert (body["assets"], body["cache_entries"], body["redis_results"]) == (1, 1, 1)


def test_second_delete_is_idempotent(enabled, client, session_factory, storage, fake_redis, cache_root):
    _seed(session_factory, storage, fake_redis, cache_root)
    assert client.delete(f"/internal/v1/documents/{FILEID}").status_code == 200

    again = client.delete(f"/internal/v1/documents/{FILEID}")

    assert again.status_code == 200
    assert again.json()["status"] == "absent"


def test_cache_entry_kept_when_content_shared(enabled, client, session_factory, storage,
                                             fake_redis, cache_root):
    _, entry = _seed(session_factory, storage, fake_redis, cache_root)
    _, other_entry = _seed(session_factory, storage, fake_redis, cache_root, fileid=OTHER)

    response = client.delete(f"/internal/v1/documents/{FILEID}")

    body = response.json()
    assert body["cache_kept_shared"] is True
    # Своя запись удаляемого документа уходит, общая запись тех же байтов остаётся.
    assert body["cache_entries"] == 1
    assert not entry.exists() and other_entry.exists()
    assert _counts(session_factory) == (1, 1, 1)  # остался только второй документ


def test_other_request_documents_survive_in_redis(enabled, session_factory, storage,
                                                  fake_redis, cache_root):
    _seed(session_factory, storage, fake_redis, cache_root)
    init_request_state("req-2", [FILEID, OTHER], "create", event_id="ev-2")
    put_document_result("req-2", FILEID, {"s3_fileid": FILEID})
    put_document_result("req-2", OTHER, {"s3_fileid": OTHER})

    delete_document(FILEID, session_factory=session_factory,
                    redis_client=fake_redis, storage=storage)

    assert sorted(fake_redis.hkeys("req:req-2:docs")) == sorted([FILEID, OTHER])
    assert json.loads(fake_redis.hget("req:req-2:docs", FILEID))["status"] == "error"
    assert json.loads(fake_redis.hget("req:req-2:docs", OTHER)) == {"s3_fileid": OTHER}
    assert fake_redis.exists("req:req-2:meta")


def test_assets_of_other_rows_with_same_fileid_removed(enabled, session_factory, storage,
                                                       fake_redis, cache_root):
    _seed(session_factory, storage, fake_redis, cache_root)
    session = session_factory()
    session.add(DocumentDB(id="doc-legacy", s3_fileid=FILEID, source_path="x",
                           status="indexed", extra_data={}))
    session.commit()
    session.close()
    storage.write_file("assets/doc-legacy/img.jpg", b"jpeg")

    report = delete_document(FILEID, session_factory=session_factory,
                             redis_client=fake_redis, storage=storage)

    assert report.documents == 2
    assert not storage.exists("assets/doc-legacy/img.jpg")


class _FakeS3:
    def __init__(self, errors):
        self.errors = errors

    def get_paginator(self, name):
        class _P:
            def paginate(self, **kwargs):
                return [{"Contents": [{"Key": "assets/doc-x/a.jpg"}, {"Key": "assets/doc-x/b.jpg"}]}]
        return _P()

    def delete_objects(self, **kwargs):
        return {"Errors": [{"Key": key, "Code": "AccessDenied"} for key in self.errors]}


def test_s3_per_key_errors_make_delete_partial(enabled, client, session_factory, fake_redis,
                                               cache_root, monkeypatch):
    from core.providers.storage import PartialDeleteError, S3StorageProvider

    monkeypatch.setattr(settings, "S3_READONLY", False)
    s3 = object.__new__(S3StorageProvider)
    s3.bucket, s3.s3_client = "kb", _FakeS3(["assets/doc-x/b.jpg"])
    with pytest.raises(PartialDeleteError) as caught:
        s3.delete_prefix("assets/doc-x/")
    assert caught.value.failed_keys == ["assets/doc-x/b.jpg"]

    s3.s3_client = _FakeS3([])
    assert s3.delete_prefix("assets/doc-x/") == 2

    s3.s3_client = _FakeS3(["assets/doc-x/b.jpg"])
    monkeypatch.setattr(
        "core.providers.storage.StorageProviderFactory.default", lambda: s3,
    )
    response = client.delete(f"/internal/v1/documents/{FILEID}")
    assert response.status_code == 500
    assert response.json()["status"] == "partial"
    assert "assets/doc-x/b.jpg" in response.json()["failed_keys"]


def test_token_required_when_flag_on(enabled, client, session_factory, storage, fake_redis,
                                     cache_root, monkeypatch):
    _seed(session_factory, storage, fake_redis, cache_root)
    monkeypatch.setattr(settings, "NORMALIZER_REQUIRE_TOKEN", True)
    monkeypatch.setattr(settings, "NORMALIZER_API_TOKEN", FAKE_TOKEN)

    assert client.delete(f"/internal/v1/documents/{FILEID}").status_code == 401
    assert _counts(session_factory) == (1, 1, 1)
    ok = client.delete(f"/internal/v1/documents/{FILEID}",
                       headers={"Authorization": f"Bearer {FAKE_TOKEN}"})
    assert ok.status_code == 200


def test_purge_orphans_lists_only_confirmed_missing():
    from tools.purge_orphans import find_orphans

    presence = {"a": True, "b": False, "c": None}
    rows = [("doc-a", "a"), ("doc-b", "b"), ("doc-c", "c"), ("doc-d", None)]
    assert find_orphans(rows, presence.get) == [{"doc_id": "doc-b", "s3_fileid": "b"}]


def test_cache_put_records_content_hash(cache_root, monkeypatch):
    from core.models.parse_result import ParseResult

    monkeypatch.setattr(result_cache, "cacheable", lambda result: True)
    result = ParseResult(blocks=[], parser_name="test")
    assert result_cache.put("cd" + "0" * 62, result, "s3://kb/x.pdf", "pdf",
                            {"doc_id": "doc-x"}, HASH)
    assert result_cache.drop_for("", HASH) == 1
