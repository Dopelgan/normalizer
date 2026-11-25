"""
Уборка документов, исходника которых больше нет в хранилище (ACL-02).

Документ удалён в Backend, а RAG по какой-то причине не позвал удаление
(`DELETE /internal/v1/documents/{s3_fileid}`): его текст остаётся в БД Parser.
Скрипт проходит таблицу `documents` и для каждого `s3_fileid` спрашивает
хранилище (`FileLocator.is_present`): исходника нет — документ сирота.

По умолчанию только список (dry-run), ничего не удаляется. `--apply`
удаляет каждую сироту тем же кодом, что и ручка (`core/document_delete.py`).
Ответ хранилища «не знаю» (сбой S3) сиротой не считается: удалять по сбою
связи нельзя.

Запуск в контейнере ingest_api (те же настройки, что у сервиса):

    python tools/purge_orphans.py              # список сирот
    python tools/purge_orphans.py --apply      # удалить сирот
    python tools/purge_orphans.py --limit 100  # первые 100 документов
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Callable, Iterable, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def find_orphans(
    rows: Iterable[tuple], is_present: Callable[[str], Optional[bool]],
) -> List[dict]:
    """Сироты из пар (doc_id, s3_fileid): только твёрдое «нет» хранилища."""
    orphans: List[dict] = []
    for doc_id, s3_fileid in rows:
        if not s3_fileid:
            continue
        if is_present(s3_fileid) is False:
            orphans.append({"doc_id": doc_id, "s3_fileid": s3_fileid})
    return orphans


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--apply", action="store_true",
                        help="удалить найденных сирот (по умолчанию только список)")
    parser.add_argument("--limit", type=int, default=0,
                        help="проверить не больше N документов (0 — все)")
    args = parser.parse_args(argv)

    from core.db.models import DocumentDB
    from core.db.session import SessionLocal
    from core.document_delete import delete_document
    from core.providers.file_locator import FileLocator
    from core.providers.storage import StorageProviderFactory
    from core.result_aggregator import get_client

    session = SessionLocal()
    try:
        query = session.query(DocumentDB.id, DocumentDB.s3_fileid).order_by(DocumentDB.id)
        if args.limit > 0:
            query = query.limit(args.limit)
        rows = query.all()
    finally:
        session.close()

    storage = StorageProviderFactory.default()
    orphans = find_orphans(rows, FileLocator(storage).is_present)
    for item in orphans:
        print(json.dumps(item, ensure_ascii=False))
    print(f"Проверено документов: {len(rows)}, сирот: {len(orphans)}", file=sys.stderr)
    if not args.apply:
        if orphans:
            print("Dry-run: ничего не удалено, для удаления — --apply", file=sys.stderr)
        return 0

    failed = 0
    redis_client = get_client()
    for item in orphans:
        report = delete_document(
            item["s3_fileid"], session_factory=SessionLocal,
            redis_client=redis_client, storage=storage,
        )
        failed += bool(report.errors)
        print(json.dumps(report.as_dict(), ensure_ascii=False))
    print(f"Удалено: {len(orphans) - failed}, с ошибками: {failed}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
