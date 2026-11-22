"""
Замер рамки бланка и переносов по сохранённому индексу 50 документов.

Что это за замер и чего он не делает. Вход — выгрузка индексированного
текста 50 документов с прода (по файлу на документ, чанки в порядке, с
длиной и метаданными). Правила применяются к **тексту чанков**, а не к
заново разобранным фрагментам: переразбор требует OCR и модели на сервере
заказчика. Для текстового документа чанк и фрагмент совпадают один к
одному (чанкер режет по фрагментам), поэтому замер честен для счёта
«сколько чанков снимется» и для текста, который увидит заказчик, — и
**не** предсказывает, как чанкер упакует то, что останется. Упаковка
соседних обрывков — его работа, и она в другом репозитории.

Что считается:

* доля чанков короче 120 символов до и после;
* медиана длины чанка до и после;
* сколько чанков снято как рамка бланка и из чего они состояли;
* сколько граф вырезано из середины связного текста и сколько переносов
  сращено — то, чего не видит ни один счётчик размера;
* потери: ни одно слово содержания не должно исчезнуть. Проверяется
  вычитанием словаря: все слова из трёх и более букв, которые были в
  документе до, обязаны быть в нём после — кроме подписей граф.

Запуск: `python tools/replay_document_furniture.py <каталог выгрузки>`
"""

from __future__ import annotations

import argparse
import collections
import io
import json
import os
import pathlib
import re
import statistics
import sys
from typing import Any, Dict, List

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
os.environ.setdefault("QWEN_ENDPOINT", "http://localhost:1")

_HEAD = re.compile(r"^-{5} чанк (\d+) \((\d+) символов\) -{5}$")
_WORD = re.compile(r"[А-Яа-яЁёA-Za-z]{3,}")
# Слово-подпись графы рамки: по нему «пропавшее» слово признаётся снятой
# графой, а не потерей.
_LABEL_WORD = re.compile(
    r"подпись|подп|дата|инв|подл|дубл|взам|перв|примен|справ|формат|лист|листов"
    r"|изм|и3м|докум|документа|зам", re.IGNORECASE)
SHORT = 120


def chunks_of(path: pathlib.Path) -> List[Dict[str, Any]]:
    """Чанки одного файла выгрузки: длина, метаданные, текст."""
    out: List[Dict[str, Any]] = []
    current: Dict[str, Any] | None = None
    for line in io.open(path, encoding="utf-8").read().split("\n"):
        match = _HEAD.match(line.strip())
        if match:
            current = {"chars": int(match.group(2)), "meta": {}, "lines": []}
            out.append(current)
            continue
        if current is None:
            continue
        if line.startswith("[мета] "):
            try:
                current["meta"] = json.loads(line[len("[мета] "):])
            except Exception:  # noqa: BLE001 — метаданные замеру не нужны
                pass
            continue
        current["lines"].append(line)
    for chunk in out:
        chunk["text"] = "\n".join(chunk["lines"]).strip()
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dump", type=pathlib.Path)
    parser.add_argument("--short", type=int, default=SHORT)
    args = parser.parse_args()

    from core.config import settings
    from core import document_furniture as furniture

    settings.SCAN_DROP_FRAME_FURNITURE = True
    settings.SCAN_JOIN_HYPHENATED_WORDS = True

    before: List[int] = []
    after: List[int] = []
    dropped: List[str] = []
    spliced = joined = 0
    lost_words: collections.Counter = collections.Counter()
    kinds: collections.Counter = collections.Counter()

    for path in sorted(args.dump.glob("*.txt")):
        words_before: set = set()
        words_after: set = set()
        for chunk in chunks_of(path):
            text = chunk["text"]
            before.append(chunk["chars"])
            words_before |= set(_WORD.findall(text.lower()))
            if furniture.is_furniture(text):
                dropped.append(text)
                continue
            text, cut = furniture.without_splices(text)
            spliced += cut
            text, count = furniture.without_hyphen_breaks(text)
            joined += count
            after.append(len(text))
            words_after |= set(_WORD.findall(text.lower()))
        # Слова, пропавшие из документа целиком, с разбором почему.
        # Подпись графы пропасть обязана — её и снимали. Половина слова,
        # разорванного переносом, тоже пропадает по делу: после сращивания
        # её нет, а целое слово есть, и считать это потерей — значит
        # считать потерей починку. Остаётся третий случай, и только он —
        # настоящая потеря.
        for word in words_before - words_after:
            if _LABEL_WORD.fullmatch(word):
                kinds["подпись графы"] += 1
                continue
            if any(word in whole for whole in words_after if len(whole) > len(word)):
                kinds["половина сращённого слова"] += 1
                continue
            kinds["НЕ ОБЪЯСНЕНО"] += 1
            lost_words[word] += 1

    def share(values: List[int]) -> float:
        return 100.0 * sum(1 for v in values if v < args.short) / max(1, len(values))

    print("чанков до:", len(before), " после:", len(after))
    print("короче %d: до %d (%.1f%%), после %d (%.1f%%)" % (
        args.short, sum(1 for v in before if v < args.short), share(before),
        sum(1 for v in after if v < args.short), share(after)))
    print("медиана: до %d, после %d" % (statistics.median(before),
                                        statistics.median(after)))
    print("снято как рамка бланка:", len(dropped))
    print("вырезано граф из середины текста:", spliced)
    print("сращено переносов:", joined)
    print()
    print("--- что снято, по повторяемости:")
    for text, count in collections.Counter(
            " ".join(t.split())[:60] for t in dropped).most_common(12):
        print("  x%-3d %s" % (count, text))
    print()
    print("--- пропавшие слова, по причине:")
    for kind, count in kinds.most_common():
        print("  %-28s %d" % (kind, count))
    print()
    print("--- НЕ ОБЪЯСНЕНО (настоящая потеря, если это не графа):")
    for word, count in lost_words.most_common(30):
        print("  x%-3d %s" % (count, word))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
