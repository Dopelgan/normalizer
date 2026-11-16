"""Срок хранения производных данных разбора (SC9): что удаляется и что остаётся."""

import json
import os
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from core.db.models import Base, ChunkDB, DocumentDB
from ingest import retention

NOW = datetime(2026, 11, 16, 12, tzinfo=timezone.utc)
OLD = (NOW - timedelta(days=30)).replace(tzinfo=None)
YOUNG = (NOW - timedelta(days=1)).replace(tzinfo=None)

OLD_SNAP = "rag-parser-snapshots/1/a.pdf"
NEW_SNAP = "rag-parser-snapshots/2/a.pdf"


def keep_list(keep=(NEW_SNAP,), known=(OLD_SNAP, NEW_SNAP)):
    return retention.KeepList(keep=set(keep), known=set(known))


@pytest.fixture()
def world(tmp_path):
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    store = retention.Store(tmp_path, raw_prefix="raw_parse", assets_prefix="assets")

    def add(doc_id, s3_fileid, *, status="indexed", updated=OLD):
        with factory() as session, session.begin():
            session.add(DocumentDB(id=doc_id, s3_fileid=s3_fileid, source_path="x", status=status,
                                   created_at=updated, updated_at=updated, extra_data={}))
            session.add(ChunkDB(id=f"{doc_id}-f1", doc_id=doc_id, chunk_type="text", content={"text": "t"},
                                provenance={}))
        write_files(doc_id)

    def write_files(doc_id, mtime=None):
        raw = tmp_path / "raw_parse" / f"{doc_id}.json"
        raw.parent.mkdir(parents=True, exist_ok=True)
        raw.write_text("{}" * 50)
        image = tmp_path / "assets" / doc_id / "p1.jpg"
        image.parent.mkdir(parents=True, exist_ok=True)
        image.write_bytes(b"x" * 1000)
        if mtime is not None:
            for path in (raw, image, image.parent):
                os.utime(path, (mtime, mtime))

    class World:
        pass

    w = World()
    w.factory, w.store, w.add, w.write_files, w.root = factory, store, add, write_files, tmp_path

    def ids():
        with factory() as session:
            return {d.id for d in session.query(DocumentDB).all()}

    def chunk_owners():
        with factory() as session:
            return {c.doc_id for c in session.query(ChunkDB).all()}

    w.ids, w.chunk_owners = ids, chunk_owners
    return w


def run(world, *, apply, entries=None, **kwargs):
    return retention.run(world.factory, world.store, kwargs.pop("keep", keep_list()), days=14, apply=apply,
                         now=NOW, pause=0, manifest=(entries.append if entries is not None else (lambda _: None)),
                         **kwargs)


def test_two_parses_of_one_document_only_the_older_is_removed(world) -> None:
    world.add("doc-old", OLD_SNAP)
    world.add("doc-new", NEW_SNAP)
    entries = []
    totals = run(world, apply=True, entries=entries)
    assert world.ids() == {"doc-new"} and world.chunk_owners() == {"doc-new"}
    assert not (world.root / "raw_parse" / "doc-old.json").exists()
    assert not (world.root / "assets" / "doc-old").exists()
    assert (world.root / "assets" / "doc-new" / "p1.jpg").exists()
    assert totals[retention.REMOVE_HISTORY]["removed"] == 1
    assert [e["doc_id"] for e in entries if e["action"] == "deleted"] == ["doc-old"]


def test_dry_run_removes_nothing_and_reports_the_candidate(world) -> None:
    world.add("doc-old", OLD_SNAP)
    world.add("doc-new", NEW_SNAP)
    entries = []
    totals = run(world, apply=False, entries=entries)
    assert world.ids() == {"doc-old", "doc-new"}
    assert (world.root / "raw_parse" / "doc-old.json").exists()
    history = totals[retention.REMOVE_HISTORY]
    assert (history["documents"], history["files"], history["chunks"], history["removed"]) == (1, 2, 1, 0)
    assert history["bytes"] == 100 + 1000
    assert not any(e["action"] == "deleted" for e in entries)


def test_age_is_respected(world) -> None:
    world.add("doc-old", OLD_SNAP, updated=YOUNG)
    totals = run(world, apply=True)
    assert world.ids() == {"doc-old"} and retention.KEEP_YOUNG in totals


def test_running_parse_is_never_touched(world) -> None:
    world.add("doc-old", OLD_SNAP, status="pending")
    run(world, apply=True)
    assert world.ids() == {"doc-old"} and (world.root / "assets" / "doc-old").exists()


def test_parse_rag_still_needs_is_kept(world) -> None:
    world.add("doc-old", OLD_SNAP)
    run(world, apply=True, keep=keep_list(keep=(OLD_SNAP, NEW_SNAP)))
    assert world.ids() == {"doc-old"}


def test_original_key_parse_is_the_latest_parse_of_that_file_and_is_kept(world) -> None:
    world.add("doc-orig", "6834222b-cc03-4676-8540-910c0713c445/act.pdf")
    run(world, apply=True, keep=keep_list(known=("6834222b-cc03-4676-8540-910c0713c445/act.pdf",)))
    assert world.ids() == {"doc-orig"}


def test_snapshot_unknown_to_rag_is_kept_unless_asked(world) -> None:
    world.add("doc-x", "rag-parser-snapshots/9/x.pdf")
    totals = run(world, apply=True)
    assert world.ids() == {"doc-x"} and retention.KEEP_UNKNOWN in totals
    run(world, apply=True, include_unknown=True)
    assert world.ids() == set()


def test_orphan_files_are_removed_only_when_old(world) -> None:
    old_ts = (NOW - timedelta(days=30)).timestamp()
    world.write_files("doc-orphan-old", mtime=old_ts)
    world.write_files("doc-orphan-new")
    retention.run(world.factory, world.store, keep_list(), days=14, apply=True, now=datetime.now(timezone.utc),
                  pause=0)
    assert not (world.root / "assets" / "doc-orphan-old").exists()
    assert (world.root / "assets" / "doc-orphan-new").exists()


def test_row_that_became_young_between_report_and_delete_is_kept(world) -> None:
    world.add("doc-old", OLD_SNAP)
    assert retention._remove_document(world.factory, world.store, "doc-old",
                                      cutoff=(OLD - timedelta(days=1))) is False
    assert world.ids() == {"doc-old"}


def _keep_json(**overrides):
    document = {"format": retention.KEEP_LIST_FORMAT, "generated_at": NOW.isoformat(),
                "keep": [NEW_SNAP], "known": [OLD_SNAP]}
    document.update(overrides)
    return json.dumps(document)


def test_keep_list_is_validated() -> None:
    parsed = retention.load_keep_list(_keep_json(), now=NOW)
    assert parsed.keep == {NEW_SNAP} and parsed.known == {OLD_SNAP, NEW_SNAP}
    for bad in (_keep_json(keep=[]), _keep_json(format="other"),
                _keep_json(generated_at=(NOW - timedelta(days=1)).isoformat()), "not json"):
        with pytest.raises(retention.KeepListError):
            retention.load_keep_list(bad, now=NOW)


def test_threshold_below_one_day_is_refused(world) -> None:
    with pytest.raises(ValueError):
        retention.run(world.factory, world.store, keep_list(), days=0)


def test_failed_file_removal_keeps_the_row_and_the_run_continues(world, monkeypatch) -> None:
    world.add("doc-a", "rag-parser-snapshots/1/a.pdf")
    world.add("doc-b", "rag-parser-snapshots/3/b.pdf")
    real_remove = world.store.remove

    def flaky(doc_id):
        if doc_id == "doc-a":
            raise OSError("disk")
        real_remove(doc_id)

    monkeypatch.setattr(world.store, "remove", flaky)
    entries = []
    run(world, apply=True, entries=entries,
        keep=keep_list(keep=(NEW_SNAP,), known=("rag-parser-snapshots/1/a.pdf", "rag-parser-snapshots/3/b.pdf")))
    assert world.ids() == {"doc-a"}
    assert [(e["action"], e["doc_id"]) for e in entries if e["action"] in ("failed", "deleted")] == [
        ("failed", "doc-a"), ("deleted", "doc-b")]
