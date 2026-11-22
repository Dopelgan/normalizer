"""
Ответы прохода «лист целиком» в кэше модели, которые ограждённый проход
(DRAWING_SHEET_GUARDED, карточка D2) отбраковывает: оборванные потолком
токенов, зациклившиеся, пустые. Только чтение: кэш не меняется.

    python tools/vlm_cache_guard_report.py /data/shared/vlm-cache [--map keys.tsv] [--list]

Ключ кэша — хеш запроса, документа в нём нет. Счёт по документам — по файлу
`--map` (строки «документ<TAB>ключ», например из лога разбора «Ответ модели
взят из кэша (<ключ>)» с полным ключом или из стенда повтора); без него —
счёт по причинам и, с `--list`, сами ключи.

Пустой ответ ограждённый проход отбраковывает, только если на листе есть
текст (строки прохода «весь текст листа» или слоя распознавания), а в кэш не
пишет никогда; здесь он считается отдельно — текста листа в кэше нет.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.providers.qwen_client import extract_json, salvage_list, sheet_verdict  # noqa: E402


def classify(content: str, finish):
    """Причина отбраковки ответа листа, «ok», или None — это не ответ прохода листа."""
    if '"annotations"' not in content:
        return None
    data = extract_json(content)
    annotations = data.get("annotations") if isinstance(data, dict) else None
    if not isinstance(annotations, list):
        annotations = salvage_list(content, "annotations")
    verdict = sheet_verdict(finish, annotations)
    if verdict:
        return verdict
    return "empty" if not annotations else "ok"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("cache_dir")
    parser.add_argument("--map", help="TSV: документ<TAB>ключ кэша")
    parser.add_argument("--list", action="store_true", help="напечатать ключи отбракованных ответов")
    args = parser.parse_args()

    reasons: dict = {}
    for path in Path(args.cache_dir).rglob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        content = data.get("content")
        if not isinstance(content, str):
            continue
        reason = classify(content, data.get("finish"))
        if reason is not None:
            reasons[path.stem] = reason

    counts = collections.Counter(reasons.values())
    print("ответов листа:", sum(counts.values()))
    for reason in ("ok", "cut", "loop", "empty"):
        print(f"  {reason:6s} {counts.get(reason, 0)}")
    if args.map:
        by_doc: dict = collections.defaultdict(collections.Counter)
        for line in open(args.map, encoding="utf-8"):
            doc, _, key = line.rstrip("\n").partition("\t")
            reason = reasons.get(key.strip())
            if reason and reason != "ok":
                by_doc[doc][reason] += 1
        print("документов с отбракованными ответами:", len(by_doc))
        for doc, counter in sorted(by_doc.items()):
            print(f"  {doc}\t" + ", ".join(f"{k} {v}" for k, v in sorted(counter.items())))
    if args.list:
        for key, reason in sorted(reasons.items()):
            if reason != "ok":
                print(f"{reason}\t{key}")


if __name__ == "__main__":
    main()
