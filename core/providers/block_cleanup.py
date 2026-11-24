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
   на странице чертежа в ней есть сильный признак мусора (ряд цифр длиннее
   10 или смесь регистров «oN», «vohP») и мусорных токенов (они же и
   латиница в одну-две буквы) не меньше 40 % (`junk_share`). Сигнал — по
   признакам мусора, а не по словарю: английская таблица и русская из
   сокращений остаются. Снятие — в `text_layer.reconcile_with_text_layer`
   до защиты рамок.
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
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

# Таблица на такую долю листа и больше проверяется на мусор (порог тот же,
# что у защиты рамок `text_layer._FRAME_TABLE_AREA`).
PAGE_TABLE_AREA = 0.5
# Доля мусорных токенов, с которой таблица — мусор.
JUNK_SHARE = 0.4
# Меньше токенов — судить не по чему, таблица остаётся.
MIN_TOKENS = 8
# Рамка поля больше этой доли страницы — область, а не место надписи.
FIELD_AREA = 0.2
# Категории полей, которые законно занимают большую часть листа.
WHOLE_SHEET_CATEGORIES = frozenset({"note"})

_TAG = re.compile(r"<[^>]+>")
_EDGE = re.compile(r"^[\s.,;:!?()\[\]{}«»\"'*_|]+|[\s.,;:!?()\[\]{}«»\"'*_|]+$")
# Признаки мусора повёрнутого текста. Сигнал строится на них, а не на «нет
# такого слова»: английская таблица из обычных слов и русская из сокращений
# («Кол.», «Прим.», «НИ») словарю не обязаны, а мусором не являются.
# * ряд цифр длиннее 10 без разделителей: «0000001006800006»;
_LONG_DIGITS = re.compile(r"\d{11,}")
# * смесь регистров внутри латинского слова: «oN», «vohP», «uN»;
_MIXED_CASE = re.compile(r"^[A-Za-z]*[a-z][A-Z][A-Za-z]*$")
# * латиница в одну-две буквы: «WD», «HM», «xq».
_SHORT_LATIN = re.compile(r"^[A-Za-z]{1,2}$")
# Короткая латиница, которая бывает и в настоящей таблице: служебные слова
# английского текста, единицы, шероховатость, обозначения.
_SHORT_LATIN_OK = frozenset({
    "a", "an", "of", "in", "to", "on", "at", "by", "or", "is", "it", "as", "be", "no",
    "we", "up", "do", "if", "so", "mm", "cm", "kg", "km", "ra", "rz", "id", "ok", "pc",
})


def _tokens(text: str) -> List[str]:
    tokens = []
    for raw in _TAG.sub(" ", text or "").split():
        token = _EDGE.sub("", raw)
        if token:
            tokens.append(token)
    return tokens


def junk(token: str) -> bool:
    """Токен с признаком мусора повёрнутого текста (см. выше)."""
    if _LONG_DIGITS.search(token):
        return True
    if _MIXED_CASE.match(token):
        return True
    return bool(_SHORT_LATIN.match(token)) and token.lower() not in _SHORT_LATIN_OK


def _strong(token: str) -> bool:
    """Признак, которого в настоящей таблице не бывает: длинный ряд цифр или смесь регистров."""
    return bool(_LONG_DIGITS.search(token) or _MIXED_CASE.match(token))


def junk_share(text: str) -> float:
    """
    Доля мусорных токенов; 0, если токенов меньше `MIN_TOKENS` или нет ни
    одного сильного признака (длинный ряд цифр, смесь регистров): одни
    короткие латинские сокращения таблицу мусором не делают.
    """
    tokens = _tokens(text)
    if len(tokens) < MIN_TOKENS or not any(_strong(t) for t in tokens):
        return 0.0
    return sum(1 for token in tokens if junk(token)) / len(tokens)


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


def is_junk_page_table(block: Any) -> bool:
    """Таблица на полстраницы и больше, ячейки которой — мусор повёрнутого текста."""
    if getattr(block, "type", None) != "table":
        return False
    try:
        area = float(block.area())
    except Exception:  # noqa: BLE001 — без рамки не судим
        return False
    return area >= PAGE_TABLE_AREA and junk_share(table_text(block)) >= JUNK_SHARE


def drop_gibberish_tables(
    blocks: Sequence[Any], drawing_page: Optional[Callable[[int], bool]] = None,
) -> List[Any]:
    """
    Блоки без таблиц-мусора во весь лист. Снимаются только на страницах
    чертежей (`drawing_page(page)`; без него — на любой): мусор этот — из
    повёрнутых граф рамки ЕСКД, а в тексте договора такой таблице неоткуда
    взяться. Проверка страницы идёт только для таблицы-кандидата: она
    может стоить рендера страницы.
    """
    kept = []
    for block in blocks:
        if is_junk_page_table(block) and (drawing_page is None or drawing_page(block.page)):
            continue
        kept.append(block)
    return kept


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
