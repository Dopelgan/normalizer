"""
Сбор списка подписантов по разобранным документам (для DRAWING_LEXICON_FIX).

Список нужен `core/providers/lexicon_fix.py`: фамилия в графах «Разраб.»,
«Пров.», «Т.контр.», «Н.контр.», «Утв.» ставится к фамилии из него. Список
собирается здесь, а не пишется руками: подписантов знает сам корпус, и
при новых документах его можно пересобрать.

Что берётся. Результаты Normalizer (JSON: документ с `fragments`, список
документов или ответ с `documents`/`results` — ищутся все объекты с
`fragments` на любой глубине). Из чертёжных фрагментов — фамилии печатных
граф подписантов: `structured_payload.title_block` (ключи `developed`,
`checked`, `tech_control`, `norm_control`, `approved`) и поля «Разраб.: …»
и т. п. Рукописное и графы подписи не голосуют — именно их и правит список.

Как решается. Написание считается по числу разных документов, где оно
встречено. Написания, близкие по тому же расстоянию, что у правки
(`lexicon_fix.distance`, допуск `SIGNER_RATIO`), — одна группа; в список
идёт написание, которое в группе в большинстве. Группа, встреченная меньше
чем в `--min-docs` документах, в список не идёт: одиночное прочтение само
может быть порчей. Спор поровну — ни одного написания (печатается в
stderr). Написание группы, которое встречено не реже `--separate-share` от
главного и не меньше `--min-docs` раз, остаётся отдельной записью: это,
скорее всего, другой человек с похожей фамилией, и правка между ними
откажется сама (две близкие записи — спор).

Запуск (без сети и без пакета `core`, только стандартная библиотека):

    python3 tools/build_signer_lexicon.py <каталог результатов> \
        --output config/lexicon_signers.txt --min-docs 3

В Git список не кладётся: в нём фамилии людей заказчика.
"""

from __future__ import annotations

import argparse
import collections
import datetime
import importlib.util
import json
import os
import sys
from typing import Any, Dict, Iterable, Iterator, List, Optional, Set, Tuple

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_lexicon_fix():
    """`core/providers/lexicon_fix.py` по пути: без `core/__init__` и его зависимостей."""
    path = os.path.join(_ROOT, "core", "providers", "lexicon_fix.py")
    spec = importlib.util.spec_from_file_location("lexicon_fix_standalone", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


lexicon_fix = _load_lexicon_fix()


def _documents(data: Any) -> Iterator[Dict[str, Any]]:
    """Все объекты с `fragments` в JSON, на любой глубине."""
    if isinstance(data, dict):
        if isinstance(data.get("fragments"), list):
            yield data
            return
        for value in data.values():
            yield from _documents(value)
    elif isinstance(data, list):
        for item in data:
            yield from _documents(item)


def _rows(path: str) -> Iterator[Dict[str, Any]]:
    """
    Выгрузка чанков в JSONL: строка — `{doc_id, structured_drawing_fields}`
    (так поля чертежа лежат в таблице `chunks`). Строка становится
    документом с одним чертёжным фрагментом.
    """
    with open(path, encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except ValueError as error:
                print(f"пропущена строка {path}:{number}: {error}", file=sys.stderr)
                continue
            if not isinstance(row, dict):
                continue
            fields = row.get("structured_drawing_fields")
            if isinstance(fields, str):
                try:
                    fields = json.loads(fields)
                except ValueError:
                    fields = None
            yield {
                "document_metadata": {"doc_id": str(row.get("doc_id") or f"{path}#{number}")},
                "fragments": [{"type": "drawing",
                               "content": {"structured_drawing_fields": fields or []}}],
            }


def _doc_id(document: Dict[str, Any], fallback: str) -> str:
    meta = document.get("document_metadata")
    if isinstance(meta, dict) and meta.get("doc_id"):
        return str(meta["doc_id"])
    return str(document.get("s3_fileid") or document.get("doc_id") or fallback)


def names_of(document: Dict[str, Any]) -> Set[str]:
    """Фамилии печатных граф подписантов документа."""
    found: Set[str] = set()
    for fragment in document.get("fragments") or []:
        if not isinstance(fragment, dict) or fragment.get("type") != "drawing":
            continue
        content = fragment.get("content") or {}
        payload = content.get("structured_payload") or {}
        block = payload.get("title_block") if isinstance(payload, dict) else None
        if isinstance(block, dict):
            for key in lexicon_fix.ROLE_KEYS:
                value = block.get(key)
                if isinstance(value, str):
                    found.update(word for _, _, word in lexicon_fix.names_in(value))
        for field in content.get("structured_drawing_fields") or []:
            if not isinstance(field, dict):
                continue
            head, body, tail = lexicon_fix._split(str(field.get("value") or ""))
            if tail or not head or lexicon_fix._kind(head) != "role":
                continue
            found.update(word for _, _, word in lexicon_fix.names_in(body))
    # Одно написание — один голос документа, в каком бы регистре оно ни было.
    return {_display(word) for word in found}


# Графа штампа, из которой берутся слова для списка терминов (и материал).
_TERM_LABELS = ("наименование",)


def terms_of(document: Dict[str, Any]) -> Set[str]:
    """Слова наименования (графа штампа) и материала документа: кириллица, от 5 букв."""
    found: Set[str] = set()
    for fragment in document.get("fragments") or []:
        if not isinstance(fragment, dict) or fragment.get("type") != "drawing":
            continue
        content = fragment.get("content") or {}
        payload = content.get("structured_payload") or {}
        block = payload.get("title_block") if isinstance(payload, dict) else None
        texts: List[str] = []
        if isinstance(block, dict):
            texts += [str(block[k]) for k in ("name", "material") if isinstance(block.get(k), str)]
        for field in content.get("structured_drawing_fields") or []:
            if not isinstance(field, dict):
                continue
            value = str(field.get("value") or "")
            head, body, _ = lexicon_fix._split(value)
            if field.get("category") == "material":
                texts.append(body or value)
            elif head and lexicon_fix._label_fold(head) in _TERM_LABELS:
                texts.append(body)
        for text in texts:
            for match in lexicon_fix._WORD.finditer(text):
                word = match.group(0)
                if (len(lexicon_fix.fold(word)) >= lexicon_fix._TERM_MIN_CHARS
                        and lexicon_fix._CYRILLIC.search(word)
                        and not any(c.isdigit() for c in word)):
                    found.add(word.lower())
    return found


def _display(word: str) -> str:
    """«ИВАНОВ» и «Иванов» — одно написание; в список — с прописной."""
    if word.isupper():
        return "-".join(part[:1] + part[1:].lower() for part in word.split("-"))
    return word


def collect(paths: Iterable[str], kind: str = "signers") -> Tuple[collections.Counter, int]:
    """Написание -> число документов; второе — сколько документов прочитано."""
    pick = terms_of if kind == "terms" else names_of
    counts: collections.Counter = collections.Counter()
    seen: Dict[str, Set[str]] = {}
    for path in paths:
        if path.lower().endswith(".jsonl"):
            documents: Iterable[Dict[str, Any]] = _rows(path)
        else:
            try:
                with open(path, encoding="utf-8") as handle:
                    data = json.load(handle)
            except (OSError, ValueError) as error:
                print(f"пропущен {path}: {error}", file=sys.stderr)
                continue
            documents = _documents(data)
        for index, document in enumerate(documents):
            doc = _doc_id(document, f"{path}#{index}")
            seen.setdefault(doc, set()).update(pick(document))
    for names in seen.values():
        counts.update(names)
    return counts, len(seen)


def cluster(counts: collections.Counter, min_docs: int,
            separate_share: float) -> Tuple[List[str], List[str]]:
    """(список фамилий, спорные группы для отчёта)."""
    order = sorted(counts, key=lambda w: (-counts[w], w))
    groups: List[List[str]] = []
    for word in order:
        for group in groups:
            head = group[0]
            if lexicon_fix.distance(word, head) <= lexicon_fix.limit(head, lexicon_fix.SIGNER_RATIO):
                group.append(word)
                break
        else:
            groups.append([word])
    result: List[str] = []
    disputes: List[str] = []
    for group in groups:
        total = sum(counts[w] for w in group)
        if total < min_docs:
            continue
        head = group[0]
        if len(group) > 1 and counts[group[1]] == counts[head]:
            disputes.append(", ".join(f"{w} ({counts[w]})" for w in group))
            continue
        result.append(head)
        for other in group[1:]:
            if counts[other] >= min_docs and counts[other] >= separate_share * counts[head]:
                result.append(other)
    return sorted(set(result)), disputes


def _json_files(root: str) -> List[str]:
    if os.path.isfile(root):
        return [root]
    files = []
    for folder, _, names in os.walk(root):
        files.extend(os.path.join(folder, n) for n in names
                     if n.lower().endswith((".json", ".jsonl")))
    return sorted(files)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("source", help="каталог (или файл) результатов Normalizer в JSON или "
                        "выгрузка чанков JSONL ({doc_id, structured_drawing_fields} в строке)")
    parser.add_argument("--kind", choices=("signers", "terms"), default="signers",
                        help="подписанты (DRAWING_LEXICON_SIGNERS_PATH) или слова наименований "
                        "и материалов (DRAWING_LEXICON_TERMS_PATH)")
    parser.add_argument("--output", "-o", help="файл списка; без него — в stdout")
    parser.add_argument("--min-docs", type=int, default=3,
                        help="в скольких разных документах встречена группа (по умолчанию 3)")
    parser.add_argument("--separate-share", type=float, default=0.5,
                        help="доля от главного написания, с которой близкое остаётся отдельным")
    args = parser.parse_args(argv)

    counts, documents = collect(_json_files(args.source), args.kind)
    names, disputes = cluster(counts, args.min_docs, args.separate_share)
    for dispute in disputes:
        print(f"спор поровну, не взято: {dispute}", file=sys.stderr)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%MZ")
    lines = [
        (f"# Подписанты для DRAWING_LEXICON_SIGNERS_PATH, {stamp}" if args.kind == "signers"
         else f"# Термины для DRAWING_LEXICON_TERMS_PATH, {stamp}"),
        f"# документов: {documents}, написаний: {len(counts)}, в списке: {len(names)}, "
        f"--min-docs {args.min_docs}",
    ] + names
    text = "\n".join(lines) + "\n"
    if args.output:
        with open(args.output, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
        print(f"{args.output}: {len(names)} {'фамилий' if args.kind == 'signers' else 'терминов'} из {documents} документов", file=sys.stderr)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
