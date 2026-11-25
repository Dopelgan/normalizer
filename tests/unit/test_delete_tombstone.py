"""Разбор, начатый или поставленный до удаления документа, результат не сохраняет (ACL-02)."""

import time

from core import document_delete
from core.models.contract import DocumentResult
from core.repositories import ChunkRepository, DocumentRepository
from core.result_aggregator import get_request_state, init_request_state
from ingest.tasks import generate_doc_id, process_document
from tests.unit import test_tasks

# Фикстуры задачи (SQLite, хранилище, заглушка парсера) — из test_tasks.
task_db = test_tasks.task_db
pipeline = test_tasks.pipeline

FILEID = "file-1"


def _rows(task_db, doc_id):
    session = task_db()
    try:
        return (DocumentRepository(session).get(doc_id),
                ChunkRepository(session).count_by_doc_id(doc_id))
    finally:
        session.close()


def test_delete_during_parse_drops_result(pipeline, task_db, fake_redis):
    doc_id = generate_doc_id(FILEID)
    inner = pipeline.parse

    def parse_then_delete(uri, file_type, metadata=None):
        result = inner(uri, file_type, metadata)
        document_delete.write_tombstone(fake_redis, doc_id)  # удаление посреди разбора
        return result

    pipeline.parse = parse_then_delete
    init_request_state("req-1", [FILEID])

    outcome = process_document("req-1", FILEID)

    assert outcome["status"] == "discarded_deleted"
    assert _rows(task_db, doc_id) == (None, 0)
    document = get_request_state("req-1")["documents"][0]
    assert document["fragments"] == [] and document["status"] == "error"


def test_queued_request_from_before_delete_is_dropped(pipeline, task_db, fake_redis):
    doc_id = generate_doc_id(FILEID)
    init_request_state("req-1", [FILEID])
    time.sleep(0.01)
    document_delete.write_tombstone(fake_redis, doc_id)

    assert process_document("req-1", FILEID)["status"] == "discarded_deleted"
    assert _rows(task_db, doc_id) == (None, 0)


def test_request_after_delete_is_saved(pipeline, task_db, fake_redis):
    doc_id = generate_doc_id(FILEID)
    document_delete.write_tombstone(fake_redis, doc_id, now=time.time() - 60)
    init_request_state("req-2", [FILEID])

    assert process_document("req-2", FILEID)["status"] == "success"
    document, chunks = _rows(task_db, doc_id)
    assert document is not None and chunks >= 1


def test_deleted_stub_matches_contract():
    DocumentResult(**document_delete.deleted_document(FILEID, generate_doc_id(FILEID)))
