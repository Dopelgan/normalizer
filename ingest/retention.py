"""
Срок хранения производных данных разбора (карточка SC9). По умолчанию — только отчёт.

Каждый разбор пишет `raw_parse/<doc_id>.json` (сырой результат, читает только
диагностика `tools/measure_repeated_drops.py`), каталог `assets/<doc_id>/`
(картинки страниц; их читает ветка чертежей **во время этого же разбора**) и строки
`chunks` (из них отдаётся повторный запрос того же файла). `doc_id` — хеш `s3_fileid`,
а RAG каждый разбор шлёт под новым снимком `rag-parser-snapshots/<id>/<имя>`, поэтому
каждая переобработка оставляет ещё один полный набор, который никто больше не читает.

Разбор удаляется целиком (файлы, затем строка документа вместе с фрагментами), только если:
* ключ — снимок RAG (`rag-parser-snapshots/`): разбор под исходным ключом один на файл
  и перезаписывается на месте, это и есть последний разбор этого файла;
* RAG этот снимок знает и он ему больше не нужен — список выдаёт
  `scripts/db_retention.py parser-keep-list` в репозитории RAG (последний успешный
  разбор каждого файла и каждой его версии, текущее событие документа, всё
  незавершённое и всё моложе порога — в списке «нужно»);
* статус `indexed` или `error` (`pending` — разбор идёт) и строка старше порога.

Снимок, которого RAG не знает, по умолчанию остаётся (`--include-unknown` — удалить).
Файлы без строки документа (`orphan`) удаляются, если они старше порога.
Ответ `GET /parse/results/{request_id}` берётся из Redis и от этих файлов не зависит.

Запуск — в контейнере воркера, где смонтирован том `ARTIFACTS_MOUNT`; список RAG
приходит через stdin (`--keep-file -`), команды — README, раздел «Срок хранения разборов».
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Set

from sqlalchemy import bindparam, text

from core.config import settings

SNAPSHOT_PREFIX = "rag-parser-snapshots/"
REMOVABLE_STATUSES = ("indexed", "error")
KEEP_LIST_FORMAT = "rag-parser-keep-list.v1"
# Список старше этого не принимается: за сутки RAG мог взять снимок в работу.
KEEP_LIST_MAX_AGE = timedelta(hours=6)

# Категории отчёта.
REMOVE_HISTORY = "remove:history"
REMOVE_UNKNOWN = "remove:unknown_to_rag"
REMOVE_ORPHAN = "remove:orphan_files"
KEEP_RAG = "keep:needed_by_rag"
KEEP_ORIGINAL = "keep:original_key"
KEEP_YOUNG = "keep:younger_than_limit"
KEEP_RUNNING = "keep:not_finished"
KEEP_UNKNOWN = "keep:unknown_to_rag"


class KeepListError(ValueError):
    pass


@dataclass(frozen=True)
class KeepList:
    keep: Set[str]
    known: Set[str]


def load_keep_list(raw: str, *, now: Optional[datetime] = None) -> KeepList:
    """Проверенный список RAG: формат, свежесть, непустой «нужно»."""
    try:
        document = json.loads(raw)
    except ValueError as exc:
        raise KeepListError(f"список RAG не JSON: {exc}") from exc
    if not isinstance(document, dict) or document.get("format") != KEEP_LIST_FORMAT:
        raise KeepListError("список RAG в неизвестном формате")
    try:
        generated = datetime.fromisoformat(document["generated_at"])
    except (KeyError, TypeError, ValueError) as exc:
        raise KeepListError("в списке RAG нет времени создания") from exc
    now = now or datetime.now(timezone.utc)
    if generated.tzinfo is None or not (now - KEEP_LIST_MAX_AGE <= generated <= now + timedelta(minutes=5)):
        raise KeepListError("список RAG устарел: нужен свежий (не старше 6 часов)")
    keep, known = document.get("keep"), document.get("known")
    if not isinstance(keep, list) or not isinstance(known, list) or not keep:
        raise KeepListError("список RAG пуст — удалять по нему нельзя")
    return KeepList(keep={str(k) for k in keep}, known={str(k) for k in known} | {str(k) for k in keep})


def classify(doc: Dict[str, Any], keep_list: KeepList, cutoff: datetime,
             *, include_unknown: bool = False) -> str:
    """Категория одного разбора. `cutoff` — наивное UTC, как `documents.updated_at`."""
    key = doc.get("s3_fileid") or ""
    if doc.get("status") not in REMOVABLE_STATUSES:
        return KEEP_RUNNING
    if not key.startswith(SNAPSHOT_PREFIX):
        return KEEP_ORIGINAL
    if doc["updated_at"] >= cutoff:
        return KEEP_YOUNG
    if key in keep_list.keep:
        return KEEP_RAG
    if key not in keep_list.known:
        return REMOVE_UNKNOWN if include_unknown else KEEP_UNKNOWN
    return REMOVE_HISTORY


def _tree_size(path: Path) -> tuple[int, int]:
    """(файлов, байт) файла или каталога; отсутствующий — (0, 0)."""
    if path.is_file():
        return 1, path.stat().st_size
    files = size = 0
    if path.is_dir():
        for root, _dirs, names in os.walk(path):
            for name in names:
                try:
                    size += (Path(root) / name).stat().st_size
                    files += 1
                except OSError:
                    pass
    return files, size


@dataclass
class Store:
    """Производные файлы разбора на томе `ARTIFACTS_MOUNT`."""

    root: Path
    raw_prefix: str = field(default_factory=lambda: settings.RAW_PARSE_S3_PREFIX.strip("/"))
    assets_prefix: str = field(default_factory=lambda: settings.ASSETS_PREFIX.strip("/"))

    def paths(self, doc_id: str) -> List[Path]:
        return [self.root / self.raw_prefix / f"{doc_id}.json", self.root / self.assets_prefix / doc_id]

    def size(self, doc_id: str) -> tuple[int, int]:
        files = size = 0
        for path in self.paths(doc_id):
            f, s = _tree_size(path)
            files, size = files + f, size + s
        return files, size

    def remove(self, doc_id: str) -> None:
        raw, assets = self.paths(doc_id)
        if raw.is_file():
            raw.unlink()
        if assets.is_dir():
            shutil.rmtree(assets)

    def doc_ids_on_disk_one(self, doc_id: str) -> float:
        """Самое позднее время изменения файлов одного разбора (0 — файлов нет)."""
        latest = 0.0
        for path in self.paths(doc_id):
            if path.exists():
                latest = max(latest, path.stat().st_mtime)
            if path.is_dir():
                for root, _dirs, names in os.walk(path):
                    for name in names:
                        try:
                            latest = max(latest, (Path(root) / name).stat().st_mtime)
                        except OSError:
                            pass
        return latest

    def doc_ids_on_disk(self) -> Dict[str, float]:
        """doc_id → самое позднее время изменения его файлов."""
        found: Dict[str, float] = {}
        raw_dir, assets_dir = self.root / self.raw_prefix, self.root / self.assets_prefix
        if raw_dir.is_dir():
            for entry in os.scandir(raw_dir):
                if entry.is_file() and entry.name.startswith("doc-") and entry.name.endswith(".json"):
                    doc_id = entry.name[:-len(".json")]
                    found[doc_id] = max(found.get(doc_id, 0.0), entry.stat().st_mtime)
        if assets_dir.is_dir():
            for entry in os.scandir(assets_dir):
                if entry.is_dir() and entry.name.startswith("doc-"):
                    found[entry.name] = max(found.get(entry.name, 0.0), entry.stat().st_mtime)
        return found


@dataclass
class Totals:
    documents: int = 0
    files: int = 0
    bytes: int = 0
    chunks: int = 0
    chunk_bytes: int = 0
    removed: int = 0
    removed_bytes: int = 0

    def as_dict(self) -> Dict[str, int]:
        return dict(self.__dict__)


def _chunk_stats(session, doc_ids: List[str]) -> Dict[str, tuple[int, int]]:
    if not doc_ids:
        return {}
    postgres = session.bind.dialect.name == "postgresql"
    size = "sum(pg_column_size(c.*))" if postgres else "0"
    rows = session.execute(text(
        f"SELECT c.doc_id, count(*), {size} FROM chunks c WHERE c.doc_id IN :ids GROUP BY c.doc_id"
    ).bindparams(_expanding("ids")), {"ids": doc_ids}).all()
    return {row[0]: (int(row[1]), int(row[2] or 0)) for row in rows}


def _expanding(name: str):
    return bindparam(name, expanding=True)


def _remove_document(session_factory, store: Store, doc_id: str, cutoff: datetime) -> bool:
    """Файлы и строку документа — под блокировкой строки, с повторной проверкой.

    Задача разбора берёт ту же строку `FOR UPDATE`, поэтому новый разбор этого файла
    ждёт конца транзакции и не застанет полуудалённые файлы.
    """
    session = session_factory()
    try:
        with session.begin():
            lock = " FOR UPDATE SKIP LOCKED" if session.bind.dialect.name == "postgresql" else ""
            row = session.execute(text(
                f"SELECT status, updated_at FROM documents WHERE id = :id{lock}"), {"id": doc_id}).first()
            if row is None or row[0] not in REMOVABLE_STATUSES or _naive(row[1]) >= cutoff:
                return False
            store.remove(doc_id)
            session.execute(text("DELETE FROM chunks WHERE doc_id = :id"), {"id": doc_id})
            session.execute(text("DELETE FROM documents WHERE id = :id"), {"id": doc_id})
        return True
    finally:
        session.close()


def _naive(value: Any) -> datetime:
    if isinstance(value, str):
        value = datetime.fromisoformat(value)
    return value.replace(tzinfo=None) if value.tzinfo else value


def run(
    session_factory, store: Store, keep_list: KeepList, *, days: int, apply: bool = False,
    include_unknown: bool = False, batch_size: int = 200, pause: float = 0.5,
    now: Optional[datetime] = None, manifest: Callable[[Dict[str, Any]], None] = lambda _: None,
    log: Callable[[str], None] = lambda _: None,
) -> Dict[str, Dict[str, int]]:
    """Отбор по правилам и (при `apply`) удаление пачками. Возвращает итоги по категориям."""
    if days < 1:
        raise ValueError("порог хранения — не меньше суток")
    now = now or datetime.now(timezone.utc)
    cutoff = _naive(now - timedelta(days=days))
    totals: Dict[str, Totals] = {}

    session = session_factory()
    try:
        docs = [
            {"id": r[0], "s3_fileid": r[1], "status": r[2], "updated_at": _naive(r[3])}
            for r in session.execute(text("SELECT id, s3_fileid, status, updated_at FROM documents")).all()
        ]
    finally:
        session.close()

    plan: List[Dict[str, Any]] = []
    for doc in docs:
        category = classify(doc, keep_list, cutoff, include_unknown=include_unknown)
        entry = {"category": category, "doc_id": doc["id"], "s3_fileid": doc["s3_fileid"],
                 "status": doc["status"], "updated_at": doc["updated_at"].isoformat()}
        plan.append(entry)
    known_ids = {doc["id"] for doc in docs}
    cutoff_ts = cutoff.replace(tzinfo=timezone.utc).timestamp()
    for doc_id, mtime in sorted(store.doc_ids_on_disk().items()):
        if doc_id not in known_ids and mtime < cutoff_ts:
            plan.append({"category": REMOVE_ORPHAN, "doc_id": doc_id, "s3_fileid": None, "status": None,
                         "updated_at": datetime.fromtimestamp(mtime, timezone.utc).replace(tzinfo=None).isoformat()})

    for start in range(0, len(plan), batch_size):
        batch = plan[start:start + batch_size]
        session = session_factory()
        try:
            stats = _chunk_stats(session, [e["doc_id"] for e in batch if e["category"] != REMOVE_ORPHAN])
        finally:
            session.close()
        for entry in batch:
            files, size = store.size(entry["doc_id"])
            chunks, chunk_bytes = stats.get(entry["doc_id"], (0, 0))
            entry.update(files=files, bytes=size, chunks=chunks, chunk_bytes=chunk_bytes)
            bucket = totals.setdefault(entry["category"], Totals())
            bucket.documents += 1
            bucket.files += files
            bucket.bytes += size
            bucket.chunks += chunks
            bucket.chunk_bytes += chunk_bytes
            manifest({"action": "candidate" if entry["category"].startswith("remove:") else "keep", **entry})
            if not apply or not entry["category"].startswith("remove:"):
                continue
            try:
                if entry["category"] == REMOVE_ORPHAN:
                    # Файлы могли начать писаться заново: новая строка или свежее время — мимо.
                    removed = (entry["doc_id"] not in _ids_in_db(session_factory, [entry["doc_id"]])
                               and store.doc_ids_on_disk_one(entry["doc_id"]) < cutoff_ts)
                    if removed:
                        store.remove(entry["doc_id"])
                else:
                    removed = _remove_document(session_factory, store, entry["doc_id"], cutoff)
            except Exception as exc:  # noqa: BLE001 — один документ не останавливает прогон
                # Строка откатывается, часть файлов могла уйти: их читают только разбор
                # этого же файла и диагностика, поэтому это безвредно, но в манифесте видно.
                manifest({"action": "failed", "error": f"{type(exc).__name__}: {exc}", **entry})
                log(f"не удалось удалить {entry['doc_id']}: {type(exc).__name__}")
                continue
            if removed:
                bucket.removed += 1
                bucket.removed_bytes += size
                manifest({"action": "deleted", **entry})
        removable = sum(1 for e in batch if e["category"].startswith("remove:"))
        log(f"пачка {start // batch_size + 1}: документов {len(batch)}, к удалению {removable}"
            + (f", удалено всего {sum(t.removed for t in totals.values())}" if apply else ""))
        if apply and removable and pause:
            time.sleep(pause)
    return {category: bucket.as_dict() for category, bucket in sorted(totals.items())}


def _ids_in_db(session_factory, doc_ids: Iterable[str]) -> Set[str]:
    session = session_factory()
    try:
        rows = session.execute(text("SELECT id FROM documents WHERE id IN :ids").bindparams(
            _expanding("ids")), {"ids": list(doc_ids)}).all()
        return {row[0] for row in rows}
    finally:
        session.close()


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Срок хранения разборов Normalizer (SC9); без --apply — отчёт.")
    parser.add_argument("--keep-file", required=True,
                        help="список RAG (`scripts/db_retention.py parser-keep-list`), `-` — stdin")
    parser.add_argument("--manifest", required=True, type=Path, help="файл JSONL: кандидаты и удалённое")
    parser.add_argument("--days", type=int, default=settings.ARTIFACT_RETENTION_DAYS,
                        help="возраст разбора, суток (ARTIFACT_RETENTION_DAYS)")
    parser.add_argument("--apply", action="store_true", help="удалить (без флага — только отчёт)")
    parser.add_argument("--include-unknown", action="store_true",
                        help="удалять и снимки, которых RAG не знает")
    parser.add_argument("--batch", type=int, default=200)
    parser.add_argument("--pause", type=float, default=0.5)
    args = parser.parse_args(argv)

    if settings.STORAGE_TYPE != "s3" or not settings.S3_READONLY:
        # Только при этой паре производные файлы лежат в ARTIFACTS_MOUNT
        # (`ReadThroughS3Provider`); иначе строки удалились бы, а файлы — нет.
        print("Инструмент работает только при STORAGE_TYPE=s3 и S3_READONLY=true "
              "(производные файлы в ARTIFACTS_MOUNT).", file=sys.stderr)
        return 2
    raw = sys.stdin.read() if args.keep_file == "-" else Path(args.keep_file).read_text(encoding="utf-8")
    try:
        keep_list = load_keep_list(raw)
    except KeepListError as exc:
        print(f"Отказ: {exc}", file=sys.stderr)
        return 2

    from core.db.session import SessionLocal

    with args.manifest.open("w", encoding="utf-8") as out:
        totals = run(
            SessionLocal, Store(Path(settings.ARTIFACTS_MOUNT)), keep_list, days=args.days, apply=args.apply,
            include_unknown=args.include_unknown, batch_size=args.batch, pause=args.pause,
            manifest=lambda entry: out.write(json.dumps(entry, ensure_ascii=False) + "\n"),
            log=lambda line: print(line, file=sys.stderr, flush=True),
        )
    print(f"Режим: {'удаление' if args.apply else 'отчёт (dry-run)'}; порог {args.days} сут.")
    for category, bucket in totals.items():
        print(f"{category}: документов {bucket['documents']}, файлов {bucket['files']}, "
              f"{bucket['bytes'] / 2**30:.2f} ГБ на диске, фрагментов {bucket['chunks']} "
              f"({bucket['chunk_bytes'] / 2**30:.2f} ГБ в БД), удалено {bucket['removed']} "
              f"({bucket['removed_bytes'] / 2**30:.2f} ГБ)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
