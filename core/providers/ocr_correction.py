"""
Выправка ошибок распознавания по словарю корпуса (`OCR_CORRECTION_DICTIONARY`).

Зачем. Выправка слоя распознавания (`core/providers/ocr_cleanup.py`,
`OCR_WORD_FIX_ENABLED`) сверяет слово со словарём моделей Tesseract — общим
словарём языка. В нём нет слов и сокращений предприятия, а неоднозначное
прочтение там не правится вовсе. После неё в тексте остаются повторяющиеся
ошибки: латинские и цифровые двойники букв («3am», «FOCT»), курсивные
подмены одной буквы (п/л, д/б, е/с: «дапее», «усповия»), слипшийся предлог
(«идата»), разорванное переносом слово, которое уже стоит в строке
(«термо- метр»). Такое слово не находит ни поиск, ни судья ответа.

Откуда словарь. Файл собирается по корпусу самого заказчика
(`tools/build_ocr_corrections.py`): эталон — текст, набранный, а не
прочитанный (текстовый слой PDF, Word), ошибки — несловарные слова
распознавания, у которых прочтение по двойникам или одна курсивная подмена
даёт частое слово эталона. Слова заказчика лежат только в файле на его
сервере (`OCR_CORRECTION_DICTIONARY_PATH`), в коде их нет.

Что правится, и только в тексте распознавания (способ разбора фрагмента из
`OCR_METHODS`; текстовый слой, Word и таблицы не трогаются никогда):

* слово из пары словаря — заменой целиком, регистр по исходному слову;
* слово из двух алфавитов без цифр — прочтением по двойникам, если ровно
  одно прочтение есть в списке слов словаря;
* перенос внутри строки («термо- метр») — сращением, если
  сращённое слово есть в списке слов, а одна из половин словом не является;
* число с буквой вместо цифры («1Омм», «О,4», «2О мм») — только при флаге
  `OCR_CORRECTION_NUMERIC_GLYPHS` и только там, где это заведомо величина:
  есть десятичная запятая или следом единица измерения. «18О 9001» (ISO),
  «1О» (номер раздела) и обозначения не трогаются.

Знак «±» не восстанавливается: «+0,1» — законное верхнее отклонение, и по
тексту его от потерянного «±» не отличить.

Исходник не теряется: каждая замена — в `structured_payload["ocr_corrections"]`
как `{"rule", "from", "to", "offset"}`, где `offset` — позиция в тексте до
выправки (у ячейки таблицы вместо него `cell`).

Связи. Вызывается из `core/contract_finalize.py` рядом с прочими починками
текста, до идентификаторов и фактов.
"""

from __future__ import annotations

import functools
import itertools
import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Mapping, NamedTuple, Optional, Tuple

logger = logging.getLogger(__name__)

PAYLOAD_KEY = "ocr_corrections"

# Способы разбора, текст которых прочитан распознаванием или моделью.
# Неизвестный способ не правится: лучше оставить ошибку, чем испортить
# набранный текст.
OCR_METHODS: FrozenSet[str] = frozenset({
    "ocr_layer", "mineru_ocr", "tesseract", "mineru_table",
    "vlm_escalation", "vision_plus_vlm", "vlm_full_page",
})
# Чертёжный фрагмент не правится: его текст пересобирается из полей
# (`core/drawing_record.py`), а размеры стережёт `dimension_fidelity`.
FRAGMENT_TYPES: Tuple[str, ...] = ("text", "table")

# Двойники латиницы и цифр -> кириллические прочтения, первое — самое
# вероятное. Таблица шире строгих двойников: курсив ГОСТ даёт «u» вместо
# «и», «m» вместо «т», «n» вместо «п»; лишние прочтения проходят только
# через список слов словаря.
_LOOKALIKE: Dict[str, str] = {
    "a": "ая", "b": "вбь", "c": "с", "d": "дб", "e": "е", "g": "д", "h": "н",
    "i": "йі", "k": "к", "m": "мт", "n": "пли", "o": "о", "p": "р", "r": "г",
    "s": "з", "t": "т", "u": "ицч", "x": "хк", "y": "уч", "z": "гз", "f": "г",
}
_MAX_VARIANTS = 4096

_CYR = re.compile(r"[А-Яа-яЁё]")
_LAT = re.compile(r"[A-Za-z]")
# Слово: буквы и цифры подряд. Знак, точка и дефис его заканчивают.
_TOKEN = re.compile(r"[^\W_]+")
# Перенос внутри строки: голова, дефис, пробел (или перевод строки), хвост
# со строчной буквы.
_HYPHEN_BREAK = re.compile(
    r"(?<![^\W\d_])([A-Za-zА-Яа-яЁё]{2,})-(?:[ \t]*\r?\n[ \t]*|[ \t]+)([a-zа-яё]{2,})(?![^\W\d_])"
)

# Числа: буква-двойник цифры. «з» строчная не берётся: «1з» не встречается,
# а «з» — частая буква слов.
_DIGIT_GLYPH = {"О": "0", "O": "0", "о": "0", "o": "0", "l": "1", "I": "1", "З": "3"}
_UNITS = (
    "мкм", "мм", "см", "км", "кг", "кВт", "МВт", "Вт", "МПа", "кПа", "Па", "кВ", "мВ",
    "мА", "кГц", "МГц", "Гц", "кОм", "МОм", "Ом", "mm", "cm", "kg", "м", "В", "А",
)
_UNIT_ALT = "|".join(sorted(_UNITS, key=len, reverse=True))
# Слитно с числом — только единица из двух букв и больше: «9ОА» скорее
# обрывок обозначения, чем «90 А».
_GLUED_UNIT_ALT = "|".join(sorted((u for u in _UNITS if len(u) > 1), key=len, reverse=True))
_NUMBER = re.compile(
    r"(?<![\w.,/-])(?P<num>[0-9OОoоlIЗ]+(?:[.,][0-9OОoоlIЗ]+)?)"
    r"(?P<unit>" + _GLUED_UNIT_ALT + r")?(?![\w])"
)
_CODE_GOES_ON = re.compile(r"[.,/-][0-9OОoоlIЗ]")
_UNIT_AFTER = re.compile(r"[ \t]?(?:(?:" + _UNIT_ALT + r")(?![\w])|°|%)")


class Correction(NamedTuple):
    rule: str
    before: str
    after: str
    offset: int


@dataclass(frozen=True)
class Dictionary:
    """Пары «ошибка -> слово» (ключ в нижнем регистре) и список слов эталона."""

    pairs: Mapping[str, str] = field(default_factory=dict)
    words: FrozenSet[str] = frozenset()

    @property
    def empty(self) -> bool:
        return not self.pairs and not self.words


@functools.lru_cache(maxsize=4)
def load_dictionary(path: str) -> Dictionary:
    """
    Словарь из файла JSON `{"pairs": {ошибка: слово}, "words": [...]}`.
    Нет пути, нет файла, файл битый — пустой словарь и ничего не правится:
    выправка не обязательна, разбор не должен из-за неё падать.
    """
    if not path:
        return Dictionary()
    if not os.path.isfile(path):
        logger.warning("Словарь выправки распознавания не найден: %s", path)
        return Dictionary()
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError) as exc:
        logger.warning("Словарь выправки распознавания не прочитан: %s", exc)
        return Dictionary()
    return build_dictionary(data.get("pairs") or {}, data.get("words") or ())


def build_dictionary(pairs: Mapping[str, Any], words: Any) -> Dictionary:
    """
    Словарь с проверкой: пара, чьё «слово» (или часть разрезанного слова)
    само есть ключ (цепочка), и пара без изменения отбрасываются — иначе
    повторная выправка того же текста дала бы другой результат.
    """
    clean: Dict[str, str] = {}
    for key, value in pairs.items():
        if not isinstance(key, str) or not isinstance(value, str):
            continue
        low, target = key.strip().lower(), value.strip()
        if low and target and low != target.lower():
            clean[low] = target.lower()
    clean = {k: v for k, v in clean.items() if not any(part in clean for part in v.split(" "))}
    vocabulary = frozenset(w.strip().lower() for w in words if isinstance(w, str) and w.strip())
    return Dictionary(pairs=clean, words=vocabulary)


# ------------------------------------------------------------------ регистр
def _case_of(original: str) -> str:
    """upper | title | lower; цифра-двойник считается заглавной («3am» — «Зам»)."""
    letters = [("A" if ch in "0368" else ch) for ch in original if ch.isalpha() or ch in "0368"]
    # Заглавными — если строчных почти нет: «OCHOBHbIE» («ы» прочитано
    # строчной «b») — это «ОСНОВНЫЕ», а «MuHyc» — «Минус».
    if len(letters) > 1 and sum(ch.isupper() for ch in letters) >= 0.75 * len(letters):
        return "upper"
    if letters and letters[0].isupper():
        return "title"
    return "lower"


def _apply_case(word: str, case: str) -> str:
    if case == "upper":
        return word.upper()
    if case == "title":
        return word[:1].upper() + word[1:]
    return word


def case_like(target: str, original: str) -> str:
    """
    `target` (нижний регистр) в регистре `original`. Слипшееся слово
    («сТехническим» -> «с техническим») режется по той же букве, и каждая
    часть берёт регистр своего куска исходника.
    """
    parts = target.split(" ")
    if len(parts) == 1:
        return _apply_case(target, _case_of(original))
    out, position = [], 0
    for part in parts:
        piece = original[position:position + len(part)]
        position += len(part)
        out.append(_apply_case(part, _case_of(piece) if piece else "lower"))
    return " ".join(out)


# -------------------------------------------------------------------- слово
def _variants(word: str) -> Optional[List[str]]:
    options: List[str] = []
    count = 1
    for ch in word:
        if _CYR.match(ch):
            options.append(ch)
            continue
        mapped = _LOOKALIKE.get(ch)
        if not mapped:
            return None
        options.append(mapped)
        count *= len(mapped)
        if count > _MAX_VARIANTS:
            return None
    return ["".join(combo) for combo in itertools.product(*options)]


def resolve_mixed(token: str, words: FrozenSet[str]) -> Optional[str]:
    """Слово из двух алфавитов без цифр -> единственное прочтение из списка слов."""
    if not words or not (_CYR.search(token) and _LAT.search(token)):
        return None
    if any(ch.isdigit() for ch in token):
        return None
    variants = _variants(token.lower())
    if not variants:
        return None
    found = {v for v in variants if v in words}
    if len(found) != 1:
        return None
    return found.pop()


def _hyphen_neighbour(text: str, start: int, end: int) -> bool:
    """Половина слова у переноса: справа «-» и пробел, или слева «-» после слова."""
    if text[end:end + 1] == "-" and (end + 1 >= len(text) or text[end + 1].isspace()):
        return True
    before = text[:start].rstrip(" \t\r\n")
    return (len(before) < start and before.endswith("-")
            and len(before) >= 2 and before[-2].isalpha())


# --------------------------------------------------------------- величины
def _number_fix(text: str, match: "re.Match[str]") -> Optional[str]:
    num, unit = match.group("num"), match.group("unit") or ""
    glyphs = [ch for ch in num if ch in _DIGIT_GLYPH]
    if not glyphs or not any(ch.isdigit() for ch in num):
        return None
    # Обозначение, а не величина: длинная целая часть («42145100З.04»),
    # продолжение кода через точку или дефис («О5.01.35»). Обозначения
    # правит `designation_repair`, не здесь.
    if len(re.split(r"[.,]", num)[0]) > 4 or _CODE_GOES_ON.match(text, match.end()):
        return None
    # «l»/«I» — единица только рядом с настоящей цифрой: «II» — римское число.
    for index, ch in enumerate(num):
        if ch in "lI" and not any(n.isdigit() for n in num[max(0, index - 1):index + 2]):
            return None
    # Десятичная запятая — признак величины. Точка — только с двумя знаками
    # после неё: «О.7», «I.4» — чаще номер пункта или обрывок даты.
    has_decimal = "," in num or ("." in num and len(num.split(".", 1)[1]) >= 2)
    has_unit = bool(unit or _UNIT_AFTER.match(text, match.end()))
    if not (has_decimal or has_unit):
        return None
    # Буква в начале без запятой — скорее обозначение («О1»); исключение —
    # «l»/«I» перед единицей: «l5 мА» — это 15, а число с нуля не пишется.
    if num[0] in _DIGIT_GLYPH and not has_decimal and not (num[0] in "lI" and has_unit):
        return None
    fixed = "".join(_DIGIT_GLYPH.get(ch, ch) for ch in num)
    return fixed + unit


# ------------------------------------------------------------------ текст
def correct_text(
    text: str, dictionary: Dictionary, numeric: bool = False,
) -> Tuple[str, List[Correction]]:
    """
    Выправленный текст и список замен. Смещения — по исходному тексту.
    Пробелы и переводы строк сохраняются (кроме сращённого переноса).
    """
    if not text or (dictionary.empty and not numeric):
        return text, []
    edits: List[Correction] = []
    taken: List[Tuple[int, int]] = []

    def free(start: int, end: int) -> bool:
        return all(end <= s or start >= e for s, e in taken)

    def add(rule: str, start: int, end: int, after: str) -> None:
        edits.append(Correction(rule, text[start:end], after, start))
        taken.append((start, end))

    if dictionary.words:
        for match in _HYPHEN_BREAK.finditer(text):
            head, tail = match.group(1), match.group(2)
            joined = (head + tail).lower()
            if joined not in dictionary.words:
                continue
            if head.lower() in dictionary.words and tail.lower() in dictionary.words:
                continue
            add("hyphen_join", match.start(), match.end(), head + tail)

    for match in _TOKEN.finditer(text):
        start, end = match.start(), match.end()
        token = match.group(0)
        if not free(start, end) or _hyphen_neighbour(text, start, end):
            continue
        target = dictionary.pairs.get(token.lower())
        rule = "pair"
        if target is None:
            target = resolve_mixed(token, dictionary.words)
            rule = "mixed_script"
        if target is None:
            continue
        after = case_like(target, token)
        if after != token:
            add(rule, start, end, after)

    if numeric:
        for match in _NUMBER.finditer(text):
            if not free(match.start(), match.end()):
                continue
            fixed = _number_fix(text, match)
            if fixed is not None and fixed != match.group(0):
                add("number_glyph", match.start(), match.end(), fixed)

    if not edits:
        return text, []
    edits.sort(key=lambda edit: edit.offset)
    pieces, position = [], 0
    for edit in edits:
        pieces.append(text[position:edit.offset])
        pieces.append(edit.after)
        position = edit.offset + len(edit.before)
    pieces.append(text[position:])
    return "".join(pieces), edits


# --------------------------------------------------------------- фрагмент
def applies_to(fragment: Any) -> bool:
    provenance = getattr(fragment, "provenance", None)
    method = getattr(provenance, "method", None)
    return fragment.type in FRAGMENT_TYPES and method in OCR_METHODS


def correct_fragment(
    fragment: Any, dictionary: Dictionary, numeric: bool = False,
) -> List[Dict[str, Any]]:
    """
    Текст и ячейки таблицы фрагмента распознавания; замены — в payload.
    Фрагмент набранного текста не меняется (`applies_to`).
    """
    if not applies_to(fragment):
        return []
    records: List[Dict[str, Any]] = []
    content = fragment.content
    if content.text:
        content.text, edits = correct_text(content.text, dictionary, numeric)
        records.extend(
            {"rule": e.rule, "from": e.before, "to": e.after, "offset": e.offset} for e in edits
        )
    table = content.table_data
    if table is not None:
        def fix(value: Any, cell: List[Any]) -> Any:
            if not isinstance(value, str) or not value.strip():
                return value
            fixed, edits = correct_text(value, dictionary, numeric)
            records.extend(
                {"rule": e.rule, "from": e.before, "to": e.after, "offset": e.offset, "cell": cell}
                for e in edits
            )
            return fixed

        table.headers = [fix(h, ["header", i]) for i, h in enumerate(table.headers or [])]
        table.rows = [
            [fix(value, [r, c]) for c, value in enumerate(row)]
            for r, row in enumerate(table.rows or [])
        ]
    if records:
        payload = content.structured_payload or {}
        payload.setdefault(PAYLOAD_KEY, []).extend(records)
        content.structured_payload = payload
    return records
