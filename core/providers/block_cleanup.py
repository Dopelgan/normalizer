"""
Уборка мусора листа чертежа (флаг DRAWING_BLOCK_CLEANUP).

Три вида мусора с одного скана листа детали, и у каждого одно место, где
он снимается.

1. **Таблица из бессмыслицы во весь лист.** MinerU принял рамку формата с
   повёрнутыми графами поля подшивки за таблицу на 0,9 листа, а ячейки —
   за текст: «nad a 0000001006800006 oN godu … upo vohP oN HM При НИ WD».
   Защита рамок (`FRAME_TABLE_GUARD_ENABLED`) её не сняла: при
   `OCR_KEEP_STRUCTURED_TABLES` «таблица» с короткими ячейками считается
   спецификацией. Здесь таблица на половину листа и больше снимается, если
   осмысленных слов в ней меньше 40 % (`gibberish_share` ≥ 0,6). Снятие —
   в `text_layer.reconcile_with_text_layer` до защиты рамок.
2. **Поле с рамкой на полстраницы.** Рамка области, где поле прочли (весь
   штамп, половина листа), рисуется на просмотре как рамка поля. Поле с
   рамкой больше 0,2 страницы теряет рамку (`placeable: false`, рамка —
   [0, 0, 1, 1], `bbox_exact: false`); значение остаётся. Технические
   требования (`note`) занимают пол-листа законно и не трогаются. Место —
   конец `field_placement.apply`, где рамки уже в долях страницы.
3. **Подпись графы вместо значения.** Модель отдаёт подписи граф штампа
   отдельными полями: «Лит.», «Масса», «№ докум.». Такое поле штампа не
   выходит (`margin_columns.is_caption`). Место — сразу после сведения
   полей листа в `DrawingProcessor._process`.

Вызовов модели нет; модуль — чистые функции.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Sequence

# Таблица на такую долю листа и больше проверяется на бессмыслицу (порог
# тот же, что у защиты рамок `text_layer._FRAME_TABLE_AREA`).
PAGE_TABLE_AREA = 0.5
# Доля бессмысленных слов, с которой таблица — мусор.
GIBBERISH_SHARE = 0.6
# Меньше слов — судить не по чему, таблица остаётся.
MIN_TOKENS = 8
# Рамка поля больше этой доли страницы — область, а не место надписи.
FIELD_AREA = 0.2
# Категории полей, которые законно занимают большую часть листа.
WHOLE_SHEET_CATEGORIES = frozenset({"note"})

_TAG = re.compile(r"<[^>]+>")
_EDGE = re.compile(r"^[\s.,;:!?()\[\]{}«»\"'*_|]+|[\s.,;:!?()\[\]{}«»\"'*_|]+$")
# Число: «4», «3,74», «1:2», «0,5*», «±0,1». Длинный ряд цифр без
# разделителей («0000001006800006») числом листа не бывает.
_NUMBER = re.compile(r"^[±+\-−]?\d{1,6}(?:[.,:/]\d{1,6})*%?$")
# Обозначение и марка: «АБВГ.123456.001», «1012-05», «М6х20», «7798-70»,
# «Ra6,3», «12Х18Н10Т», «Ст3».
_DESIGNATION = re.compile(
    r"^(?=.*\d)[A-Za-zА-Яа-яЁё]{0,5}[.\s]?\d{1,8}(?:[.,\-/xх×][A-Za-zА-Яа-яЁё]{0,3}\d{1,8})*"
    r"[A-Za-zА-Яа-яЁё]{0,3}\d{0,3}$"
)
_STEEL = re.compile(r"^(?:\d{1,2}[A-ZА-ЯЁ]{1,2}\d{0,2}){2,}[A-ZА-ЯЁ]?$")
_CYR = re.compile(r"^[А-Яа-яЁё]+(?:-[А-Яа-яЁё]+)*$")
_LAT = re.compile(r"^[A-Za-z]+$")
_CYR_VOWELS = set("аеёиоуыэюяАЕЁИОУЫЭЮЯ")
_LAT_VOWELS = set("aeiouyAEIOUY")
# Короткие слова, которые встречаются в спецификации и примечаниях.
_SHORT_WORDS = frozenset({
    "и", "в", "с", "к", "у", "о", "а", "на", "по", "из", "до", "от", "не", "за", "то",
    "мм", "шт", "кг", "м", "см", "г", "№", "ед", "кол", "поз", "зона",
})
# Латиница в русском чертеже — почти всегда обозначения стандартов и знаки.
_LATIN_TERMS = frozenset({
    "gost", "iso", "din", "en", "astm", "aisi", "ra", "rz", "max", "min", "hrc", "hb",
    "it", "dn", "pn", "sw", "ost",
})


def _tokens(text: str) -> List[str]:
    tokens = []
    for raw in _TAG.sub(" ", text or "").split():
        token = _EDGE.sub("", raw)
        if token:
            tokens.append(token)
    return tokens


def _case_ok(word: str) -> bool:
    """Слово целиком строчными, с большой буквы или целиком прописными."""
    return word.islower() or word.isupper() or (word[0].isupper() and word[1:].islower())


def meaningful(token: str) -> bool:
    """
    Токен похож на смысл: число, обозначение или марка, подпись графы,
    русское слово (от трёх букв, с гласной, без смеси регистров), короткое
    служебное слово или латинский термин стандарта. Латиница вообще
    считается словом только с четырёх букв и с гласной: в русском чертеже
    короткая латиница — это остатки повёрнутого текста («oN», «WD», «HM»).
    """
    from core.providers import margin_columns

    if _NUMBER.match(token) or _DESIGNATION.match(token) or _STEEL.match(token):
        return True
    if margin_columns.is_caption(token):
        return True
    lower = token.lower()
    if lower in _SHORT_WORDS or lower in _LATIN_TERMS:
        return True
    if _CYR.match(token):
        letters = token.replace("-", "")
        return len(letters) >= 3 and _case_ok(letters) and bool(set(letters) & _CYR_VOWELS)
    if _LAT.match(token):
        return len(token) >= 4 and _case_ok(token) and bool(set(token) & _LAT_VOWELS)
    return False


def gibberish_share(text: str) -> float:
    """Доля бессмысленных токенов; 0, если токенов меньше `MIN_TOKENS`."""
    tokens = _tokens(text)
    if len(tokens) < MIN_TOKENS:
        return 0.0
    bad = sum(1 for token in tokens if not meaningful(token))
    return bad / len(tokens)


def table_text(block: Any) -> str:
    """Текст ячеек таблицы блока MinerU (строки `table_data`, иначе `text`)."""
    rows = ((getattr(block, "table_data", None) or {}).get("rows")) or []
    cells: List[str] = []
    for row in rows:
        if isinstance(row, (list, tuple)):
            cells.extend(str(cell) for cell in row if cell is not None)
        elif isinstance(row, dict):
            cells.extend(str(cell) for cell in row.values() if cell is not None)
    if cells:
        return " ".join(cells)
    return str(getattr(block, "text", "") or "")


def is_gibberish_page_table(block: Any) -> bool:
    """Таблица на полстраницы и больше, ячейки которой — бессмыслица."""
    if getattr(block, "type", None) != "table":
        return False
    try:
        area = float(block.area())
    except Exception:  # noqa: BLE001 — без рамки не судим
        return False
    return area >= PAGE_TABLE_AREA and gibberish_share(table_text(block)) >= GIBBERISH_SHARE


def drop_gibberish_tables(blocks: Sequence[Any]) -> List[Any]:
    """Блоки без таблиц-бессмыслиц во весь лист."""
    return [block for block in blocks if not is_gibberish_page_table(block)]


def _area(bbox: Sequence[float]) -> float:
    if not bbox or len(bbox) != 4:
        return 0.0
    x1, y1, x2, y2 = (float(c) for c in bbox)
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


WHOLE = [0.0, 0.0, 1.0, 1.0]


def unplace_oversized(fields: Iterable[Any]) -> int:
    """
    Поля с рамкой больше `FIELD_AREA` страницы теряют рамку: `placeable`
    false, рамка — весь лист (значение контракта «место неизвестно»).
    Рамки в долях страницы. Возвращает число таких полей.
    """
    changed = 0
    for field in fields:
        if field.category in WHOLE_SHEET_CATEGORIES:
            continue
        if field.placeable is False or _area(field.bbox) <= FIELD_AREA:
            continue
        field.bbox = list(WHOLE)
        field.bbox_exact = False
        field.placeable = False
        changed += 1
    return changed


def drop_caption_fields(fields: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Поля штампа, у которых значение — одна подпись графы, не выходят."""
    from core.providers import margin_columns

    kept = []
    for item in fields:
        if item.get("category") == "title_block":
            _, value = margin_columns.split_value(str(item.get("value") or ""))
            if margin_columns.is_caption(value):
                continue
        kept.append(item)
    return kept
