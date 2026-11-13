"""
Слово не бывает из двух алфавитов (`SCAN_NO_MIXED_SCRIPT_WORDS`).

Зачем. Порча распознавания происходит внутри слова: «ТAДУ» — кириллическое
обозначение с латинской «A», «HТЛ» — с латинской «H», «ККS» — латинская
аббревиатура с кириллическими «К». Настоящий английский текст на этих
листах идёт целыми строками и блоками («Switching device with cold junction
compensation», «For NPP»), а не одной буквой внутри русского слова. Поэтому
запрет ставится на слово, а язык выбирается по строке
(`OCR_PER_LINE_LANGUAGE`, `core/providers/tesseract_fallback.py`), а не по
документу: полностью английский лист остаётся английским.

Что делается. У слова из двух алфавитов перебираются прочтения по
неразличимым знакам (`core/providers/designation_repair.py`). Слово
правится, если ровно одно прочтение — целиком в одном алфавите и при этом
есть в справочнике: код организации ЕСКД, код документа, префикс стандарта,
марка стали, аббревиатура из списка ниже. Справочник собран по корпусу
прода (`tools/harvest_designations.py`) — это список настоящих сокращений,
а не список ошибок.

Чего здесь нет. Слово, у которого ни одно прочтение не опознано, остаётся
как было. Именно поэтому не трогаются обозначения, смешанные по самой своей
записи: «Rизм», «Pисп», «Аind» (символ латиницей, индекс по-русски) и
химические формулы вида «KMnO» — ни одно их прочтение в справочнике не
значится. Не трогаются и слова, испорченные не алфавитом, а похожестью
букв («надпucu», «Cdeлaно»): их правит выправка по словарю
(`core/providers/ocr_cleanup.py`, `OCR_WORD_FIX_ENABLED`), где словарь —
модели самого Tesseract.

Единицы, знаки диаметра и градуса, размеры и обозначения с цифрами словом
не считаются: слово здесь — это только буквы.

Связи. Вызывается из `core/contract_finalize.py` до правки обозначений.
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

from core.providers.designation_repair import (
    ESKD_CODES, ESKD_KINDS, GRADES, STD_PREFIXES, Change, readings,
)

# Пометка фрагмента, в котором слово сведено к одному алфавиту.
WARNING = "MIXED_SCRIPT_WORD_RESOLVED"

# Аббревиатуры, которые встречаются в корпусе написанными одним алфавитом.
# Список настоящих сокращений: прочтения, которого здесь нет, не бывает.
_ABBREVIATIONS: Tuple[str, ...] = (
    # русские
    "НТЛ", "ОТК", "КПП", "ИНН", "АО", "ООО", "ОАО", "ЗАО", "НПО", "АЭС",
    "ГОСТ", "ОСТ", "СТО", "ТУ", "КД", "ЭД", "ПС", "РЭ", "СБ", "ВО", "ТБ",
    "ПМ", "ФО", "УКС", "РОСАТОМ", "ВАЭС", "ЛАЭС", "КАЭС", "БАЭС", "НВАЭС",
    "ОКП", "ОКПД", "ОКВЭД", "ЕСКД", "НП", "РД", "ТЗ", "ПК", "МЭК", "ИСО",
    # латинские
    "KKS", "QA", "IP", "ISO", "NPP", "OEM", "TM", "RU", "EN", "DIN", "IEC",
    "NTL", "PRIBOR", "RPR", "QNS", "WxHxD", "MAX", "MIN",
)
_VOCABULARY = frozenset(
    tuple(ESKD_CODES) + tuple(GRADES) + tuple(ESKD_KINDS)
    + tuple(STD_PREFIXES) + _ABBREVIATIONS
)
_FOLDED = {entry.casefold(): entry for entry in _VOCABULARY}

# Слово: только буквы, от двух знаков. С цифрой — обозначение, его правит
# `designation_repair`; знаки и единицы словом не считаются.
_WORD = re.compile(r"[A-Za-zА-Яа-яЁё]{2,}")
_CYRILLIC = re.compile(r"[А-Яа-яЁё]")
_LATIN = re.compile(r"[A-Za-z]")


def _pure(word: str) -> bool:
    """Слово целиком в одном алфавите."""
    return not (_CYRILLIC.search(word) and _LATIN.search(word))


def resolve_word(word: str) -> Optional[str]:
    """Единственное опознанное прочтение слова одним алфавитом, или None."""
    if _pure(word):
        return None
    found = {
        _FOLDED[variant.casefold()] for variant in readings(word)
        if _pure(variant) and variant.casefold() in _FOLDED
    }
    if len(found) != 1:
        return None
    resolved = found.pop()
    return resolved if resolved != word else None


def repair_text(text: str) -> Tuple[str, List[Change]]:
    """Текст, в котором слова из двух алфавитов сведены к одному."""
    if not text:
        return text, []
    changes: List[Change] = []
    pieces: List[str] = []
    position = 0
    for match in _WORD.finditer(text):
        resolved = resolve_word(match.group(0))
        if resolved is None:
            continue
        pieces.append(text[position:match.start()])
        pieces.append(resolved)
        position = match.end()
        changes.append(Change("mixed_script", match.group(0), resolved))
    if not changes:
        return text, []
    pieces.append(text[position:])
    return "".join(pieces), changes


def repair_fragment(fragment) -> List[Change]:
    """Слова текстового фрагмента; фрагмент получает пометку правки."""
    text = fragment.content.text
    if not text:
        return []
    repaired, changes = repair_text(text)
    if not changes:
        return []
    fragment.content.text = repaired
    fragment.warnings = list(dict.fromkeys(fragment.warnings + [WARNING]))
    return changes
