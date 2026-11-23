"""
Сбор словаря ошибок распознавания по корпусу (`OCR_CORRECTION_DICTIONARY`).

Словарь нужен `core/providers/ocr_correction.py`. Собирается здесь, по
корпусу самого заказчика, а не пишется руками: эталон — текст, набранный,
а не прочитанный (`--clean-methods`: текстовый слой PDF, Word, векторные
надписи), ошибки — слова распознавания и модели (`--noisy-methods`).

Пара «ошибка -> слово» берётся, только если ошибочного написания нет ни в
эталоне, ни в словарях моделей Tesseract (`--wordlist`, по одному слову в
строке), а исправление — частое слово эталона. Классы пар:

* `mixed_script`, `latin_for_cyr`, `digit_in_word` — прочтение по двойникам
  латиницы и цифр («3» — «З», «0» — «О»); латинское слово, которое есть в
  английском словаре, не трогается;
* `multi_glyph` — несколько знаков вместо одного («ьі» — «ы», «rn» — «т»);
* `shape_swap` — одна курсивная подмена из короткого списка пар формы
  (п/л, е/с, д/б, б/в, о/с, у/и, ш/щ …). Подмена гласной на гласную в
  окончании — это другая форма слова, а не ошибка, и в список не входит;
* `glued_word` — слипшийся однобуквенный предлог («идата» — «и дата»).

Отсеивается: обрывок слова у переноса (написание — начало или конец
частого слова эталона), обозначения с цифрами, кроме цифр-двойников,
единицы измерения как исправление, неоднозначный выбор (второе слово
встречается чаще трети первого), короткие (до трёх знаков) написания с
редким исправлением. В словарь идут пары, встреченные не меньше
`--min-count` раз. Список слов эталона (`words`) — для сращения переносов и
слов из двух алфавитов.

Выгрузка — JSON Lines, по фрагменту в строке:
`{"m": способ разбора, "t": текст, "td": {"headers": [...], "rows": [[...]]}}`.

    python tools/build_ocr_corrections.py фрагменты.jsonl \\
        --wordlist rus.words --wordlist eng.words \\
        --output ocr_corrections.json --pairs-tsv pairs.tsv --sample 100

Файл словаря содержит слова заказчика: он лежит у заказчика и в Git не
попадает. Решение о включении — по выборочной проверке пар в контексте
(`--sample`), а не по числу пар.
"""

from __future__ import annotations

import argparse
import collections
import itertools
import json
import random
import re
import sys
from typing import Counter, Dict, Iterable, List, Optional, Set, Tuple

CLEAN_METHODS = ("text_layer_extraction", "office_extraction", "plain_text_extraction",
                 "vector_text_layer")
NOISY_METHODS = ("ocr_layer", "mineru_ocr", "tesseract", "mineru_table",
                 "vision_plus_vlm", "vlm_full_page", "vlm_escalation")

_STRIP = re.compile(r"^[^\w±⌀Ø°]+|[^\w±⌀Ø°]+$")
_CYR = set("абвгдеёжзийклмнопрстуфхцчшщъыьэюя")
_LAT = set("abcdefghijklmnopqrstuvwxyz")
_ALPHABET = "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
# Двойники: латиница и цифры -> кириллица, первое прочтение — самое вероятное.
_LOOK = {
    "a": "ая", "b": "вбь", "c": "с", "d": "дб", "e": "е", "g": "д", "h": "н", "i": "йі",
    "k": "к", "m": "мт", "n": "пли", "o": "о", "p": "р", "r": "г", "s": "з", "t": "т",
    "u": "ицч", "x": "хк", "y": "уч", "z": "гз", "f": "г", "l": "і", "j": "з",
    "0": "о", "3": "з", "6": "б", "4": "ч", "8": "в",
}
_MULTI = (("rn", "т"), ("ьі", "ы"), ("ьl", "ы"), ("bl", "ы"), ("bi", "ы"), ("ъі", "ы"),
          ("io", "ю"), ("ii", "и"), ("hi", "ш"))
_SHAPE: Set[Tuple[str, str]] = set()
for _a, _b in ("пл", "ес", "дб", "бв", "ос", "лд", "тг", "пн", "пи", "ин", "цу", "шщ", "зэ"):
    _SHAPE |= {(_a, _b), (_b, _a)}
_SHAPE |= {("д", "в"), ("у", "и")}
_UNITS = frozenset({
    "мм", "см", "м", "км", "кг", "г", "мг", "т", "ма", "а", "ка", "в", "кв", "мв", "мкм",
    "мпа", "кпа", "па", "вт", "квт", "мвт", "гц", "кгц", "мгц", "ом", "ком", "мом", "с",
    "мин", "ч", "л", "мл", "шт", "нм", "мкс", "мс",
})
_PREPOSITIONS = set("свкиуао")
_MAX_VARIANTS = 4096


def fragment_text(record: Dict) -> str:
    text = record.get("t") or ""
    if text.strip():
        return text
    table = record.get("td") or {}
    cells = [str(h) for h in (table.get("headers") or [])]
    for row in table.get("rows") or []:
        cells.extend(str(c) for c in row)
    return "\n".join(cells)


def read_corpus(path: str, clean: Iterable[str], noisy: Iterable[str]
                ) -> Tuple[Counter[str], Counter[str], Counter[Tuple[str, str]]]:
    """Счётчики слов эталона, слов распознавания и пар «голова- хвост» у переноса."""
    clean_set, noisy_set = set(clean), set(noisy)
    clean_words: Counter[str] = collections.Counter()
    noisy_words: Counter[str] = collections.Counter()
    breaks: Counter[Tuple[str, str]] = collections.Counter()
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            method = record.get("m")
            if method in clean_set:
                target = clean_words
            elif method in noisy_set:
                target = noisy_words
            else:
                continue
            head = None
            for raw in fragment_text(record).split():
                core = _STRIP.sub("", raw)
                if not core:
                    head = None
                    continue
                target[core] += 1
                if target is noisy_words and head is not None:
                    breaks[(head, core)] += 1
                head = core if raw.endswith("-") and len(core) >= 2 else None
    return clean_words, noisy_words, breaks


class Miner:
    def __init__(self, clean_words: Counter[str], legit: Set[str], english: Set[str]):
        self.clean: Counter[str] = collections.Counter()
        for word, count in clean_words.items():
            self.clean[word.lower()] += count
        self.legit = legit
        self.english = english
        self.prefixes: Counter[str] = collections.Counter()
        self.suffixes: Counter[str] = collections.Counter()
        for word, count in self.clean.items():
            if len(word) >= 6 and count >= 3 and word.isalpha():
                for i in range(3, len(word) - 2):
                    self.prefixes[word[:i]] += count
                    self.suffixes[word[i:]] += count
        self.rejected: Counter[str] = collections.Counter()

    # -------------------------------------------------------------- проверки
    def known(self, word: str) -> bool:
        return self.clean[word] > 0 or word in self.legit or word in self.english

    def fragment(self, word: str) -> bool:
        return self.prefixes[word] >= 3 or self.suffixes[word] >= 3

    @staticmethod
    def script(word: str) -> Set[str]:
        kinds = set()
        for ch in word:
            if ch in _CYR:
                kinds.add("c")
            elif ch in _LAT:
                kinds.add("l")
            elif ch.isdigit():
                kinds.add("d")
            else:
                kinds.add("o")
        return kinds

    def choose(self, candidates: Iterable[str], min_count: int) -> Optional[str]:
        ranked = sorted(((self.clean[c], c) for c in candidates if self.clean[c] >= min_count),
                        reverse=True)
        if not ranked or (len(ranked) > 1 and ranked[0][0] < 3 * ranked[1][0]):
            return None
        return ranked[0][1]

    # -------------------------------------------------------------- прочтения
    @staticmethod
    def lookalikes(word: str) -> Set[str]:
        bases = [word] + [word.replace(a, b) for a, b in _MULTI if a in word]
        out: Set[str] = set()
        for base in bases:
            options = [_LOOK.get(ch, ch) for ch in base]
            total = 1
            for option in options:
                total *= len(option)
            if total > _MAX_VARIANTS:
                continue
            out.update("".join(combo) for combo in itertools.product(*options))
        out.discard(word)
        return out

    @staticmethod
    def edits(word: str) -> Set[str]:
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        out = {a + b[1:] for a, b in splits if b}
        out |= {a + c + b[1:] for a, b in splits if b for c in _ALPHABET}
        out |= {a + c + b for a, b in splits for c in _ALPHABET}
        return out - {word}

    def glued(self, word: str) -> Optional[str]:
        if self.fragment(word):
            return None
        head = word[0]
        if head in _PREPOSITIONS | {"c"} and len(word) >= 5 and word[0] != word[1]:
            rest = word[1:]
            if self.clean[rest] >= 20 and len(rest) >= 4 and not self.fragment(rest):
                return ("с" if head == "c" else head) + " " + rest
        tail = word[-1]
        if tail in "сcвк" and len(word) >= 6:
            rest = word[:-1]
            if self.clean[rest] >= 20 and len(rest) >= 5 and not self.fragment(rest):
                return rest + " " + ("с" if tail == "c" else tail)
        return None

    @staticmethod
    def shape_swap(word: str, fix: str) -> bool:
        if len(word) != len(fix):
            return False
        diff = [(a, b) for a, b in zip(word, fix) if a != b]
        return len(diff) == 1 and diff[0] in _SHAPE

    # ------------------------------------------------------------------ пара
    def correction(self, written: str, count: int) -> Optional[Tuple[str, str]]:
        """(исправление, класс) для написания из распознавания или None."""
        low = written.lower()
        kinds = self.script(low)
        if "o" in kinds or kinds == {"d"} or sum(ch.isalpha() for ch in low) < 2:
            return None
        if self.known(low):
            return None
        if len(set(low)) <= 2 and len(low) >= 4:
            self.rejected["repetitive"] += count
            return None
        digits = [ch for ch in low if ch.isdigit()]
        if digits and (not all(ch in "0368" for ch in digits) or re.search(r"[68]$|^[68]", low)
                       or (re.fullmatch(r"[a-zа-я]+[0-9]+", low) and written.isupper())):
            self.rejected["designation"] += count
            return None
        fix = kind = None
        if "l" in kinds or "d" in kinds:
            if kinds <= {"l"} and low in self.english:
                return None
            variants = self.lookalikes(low)
            if kinds <= {"l"} and len(low) <= 3:
                variants = {"".join(_LOOK.get(ch, ch)[0] for ch in low)}
            fix = self.choose(variants, 5 if "c" not in kinds else 2)
            if fix is None:
                for variant in variants:
                    glued = None if self.known(variant) else self.glued(variant)
                    if glued:
                        fix, kind = glued, "glued_word"
                        break
            else:
                kind = ("mixed_script" if "c" in kinds and "l" in kinds
                        else "digit_in_word" if "d" in kinds else "latin_for_cyr")
                if kinds <= {"l"} and self.clean[fix] < 20:
                    self.rejected["latin_rare_target"] += count
                    return None
        elif kinds == {"c"}:
            if (written[0].islower() or written[1:].islower()) and self.fragment(low):
                self.rejected["fragment"] += count
                return None
            glued = self.glued(low)
            if glued:
                fix, kind = glued, "glued_word"
            else:
                fix = self.choose({low.replace(a, b) for a, b in _MULTI if a in low}, 2)
                kind = "multi_glyph" if fix else None
                if not fix and len(low) >= 5:
                    found = self.choose(self.edits(low), 5)
                    if found and self.shape_swap(low, found):
                        fix, kind = found, "shape_swap"
                    elif found:
                        self.rejected["edit_not_shape"] += count
        if not fix:
            return None
        if len(low) <= 3 and self.clean[fix] < 200:
            self.rejected["short_rare_target"] += count
            return None
        if fix in _UNITS:
            self.rejected["unit_target"] += count
            return None
        return fix, kind


def build(clean_words: Counter[str], noisy_words: Counter[str], legit: Set[str],
          english: Set[str], min_count: int, min_word_count: int):
    miner = Miner(clean_words, legit, english)
    entries: Dict[str, Tuple[Optional[str], str, int]] = {}
    for written, count in noisy_words.items():
        found = miner.correction(written, count)
        if not found:
            continue
        fix, kind = found
        low = written.lower()
        previous = entries.get(low)
        if previous and previous[0] != fix:
            entries[low] = (None, kind, 0)  # два разных исправления — не берём
        else:
            entries[low] = (fix, kind, (previous[2] if previous else 0) + count)
    entries = {k: v for k, v in entries.items() if v[0] and v[2] >= min_count}
    words = sorted(w for w, c in miner.clean.items()
                   if c >= min_word_count and w.isalpha() and len(w) >= 2)
    return entries, words, miner


def contexts(path: str, wanted: Iterable[str], noisy: Iterable[str], per_word: int = 2
             ) -> Dict[str, List[str]]:
    want = {w: [] for w in wanted}
    noisy_set = set(noisy)
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("m") not in noisy_set:
                continue
            text = fragment_text(record)
            for match in re.finditer(r"\S+", text):
                core = _STRIP.sub("", match.group()).lower()
                if core in want and len(want[core]) < per_word:
                    snippet = text[max(0, match.start() - 40):match.end() + 40]
                    want[core].append(snippet.replace("\n", " / "))
    return want


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("corpus", help="выгрузка фрагментов, JSON Lines")
    parser.add_argument("--wordlist", action="append", default=[],
                        help="словарь Tesseract (rus.words, eng.words); имя с eng — английский")
    parser.add_argument("--clean-methods", default=",".join(CLEAN_METHODS))
    parser.add_argument("--noisy-methods", default=",".join(NOISY_METHODS))
    parser.add_argument("--min-count", type=int, default=2)
    parser.add_argument("--min-word-count", type=int, default=3)
    parser.add_argument("--output", required=True)
    parser.add_argument("--pairs-tsv", default="")
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args(argv)

    clean_methods = args.clean_methods.split(",")
    noisy_methods = args.noisy_methods.split(",")
    legit: Set[str] = set()
    english: Set[str] = set()
    for path in args.wordlist:
        with open(path, encoding="utf-8", errors="ignore") as handle:
            words = {line.strip().lower() for line in handle if line.strip()}
        (english if "eng" in path else legit).update(words)

    clean_words, noisy_words, _ = read_corpus(args.corpus, clean_methods, noisy_methods)
    entries, words, miner = build(clean_words, noisy_words, legit, english,
                                  args.min_count, args.min_word_count)

    total = sum(noisy_words.values())
    unknown = sum(c for w, c in noisy_words.items()
                  if w.isalpha() and len(w) >= 2 and not miner.known(w.lower()))
    by_kind: Counter[str] = collections.Counter()
    by_kind_count: Counter[str] = collections.Counter()
    for fix, kind, count in entries.values():
        by_kind[kind] += 1
        by_kind_count[kind] += count
    print(f"слов распознавания {total}, несловарных буквенных {unknown}", file=sys.stderr)
    print(f"пар {len(entries)}, вхождений {sum(v[2] for v in entries.values())}, "
          f"слов эталона {len(words)}", file=sys.stderr)
    for kind in sorted(by_kind, key=lambda k: -by_kind_count[k]):
        print(f"  {kind}: пар {by_kind[kind]}, вхождений {by_kind_count[kind]}", file=sys.stderr)
    print(f"отсеяно: {dict(miner.rejected)}", file=sys.stderr)

    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump({"version": 1, "kind": "ocr-correction-dictionary",
                   "pairs": {k: v[0] for k, v in sorted(entries.items())},
                   "words": words}, handle, ensure_ascii=False)
    if args.pairs_tsv:
        with open(args.pairs_tsv, "w", encoding="utf-8") as handle:
            for key, (fix, kind, count) in sorted(entries.items(), key=lambda x: -x[1][2]):
                handle.write(f"{key}\t{fix}\t{kind}\t{count}\n")
    if args.sample:
        rng = random.Random(args.seed)
        sample = rng.sample(sorted(entries), min(args.sample, len(entries)))
        found = contexts(args.corpus, sample, noisy_methods)
        for index, key in enumerate(sample, start=1):
            print(f"{index}. {key} -> {entries[key][0]} [{entries[key][1]}] || "
                  + " ## ".join(found[key]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
