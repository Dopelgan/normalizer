"""
Уборка надписей без места (DRAWING_UNPLACED_CLEANUP).

Проход «содержание листа» (DRAWING_SHEET_CONTENT_PASS) и поля из текста
листа (DRAWING_FIELDS_FROM_TEXT) дают надписи без рамки: место им берётся у
прохода с рамками, а без пары остаётся «весь лист». Проверить такую надпись
по месту нельзя, поэтому до полей не доходят те, что заведомо не надписи
листа:

* голая подпись графы штампа или поля подшивки без значения («Копировал»,
  «Справ. №», «Изм. Лист № докум.»): на 22 эталонных листах — около 120
  полей на прогон, у прохода с рамками их было 20;
* слово, смешавшее буквы, которых нет в другом алфавите («Гervasov»): это
  сбой чтения, а не надпись. Двойники вроде «Х», «Н», «Т» в марке стали
  смешением не считаются.

Ключ `loose` — сравнение записей без пробелов, знаков и регистра, латиница-
двойник — кириллицей: по нему надпись содержания находит пару у прохода с
рамками, когда записи различаются только точкой или похожей буквой.
"""

from __future__ import annotations

import re
from typing import Optional

from core.providers import margin_columns

_LOOKALIKE = str.maketrans("aceopxykmhtb", "асеорхукмнтв")
_LOOSE = re.compile(r"[^0-9a-zа-яё⌀±°№]")
# Буквы, у которых нет двойника в другом алфавите.
_LATIN_ONLY = frozenset("bdfgijlnqrsuvwzDFGIJLNQRSUVWZ")
_CYRILLIC_ONLY = frozenset("бгдёжзийлпфцчшщъыьэюяБГДЁЖЗИЙЛПФЦЧШЩЪЫЬЭЮЯ")
_WORD = re.compile(r"[A-Za-zА-Яа-яЁё]+")


def loose(text: str) -> str:
    """Запись для сравнения: без пробелов и знаков, латиница-двойник — кириллицей."""
    return _LOOSE.sub("", str(text or "").lower().translate(_LOOKALIKE))


def mixed_script(text: str) -> bool:
    """Есть ли слово с буквами обоих алфавитов, не являющимися двойниками."""
    for word in _WORD.findall(str(text or "")):
        if any(c in _LATIN_ONLY for c in word) and any(c in _CYRILLIC_ONLY for c in word):
            return True
    return False


def bare_caption(text: str) -> bool:
    """Подпись графы без значения: «Копировал», «Справ. №», «Т.контр.:»."""
    value = str(text or "").strip()
    if not value:
        return False
    if margin_columns.is_caption(value):
        return True
    label, rest = margin_columns.split_value(value)
    return bool(label) and (not rest or margin_columns.is_caption(rest)) and margin_columns.is_caption(label)


def noise(text: str) -> Optional[str]:
    """Причина, по которой надпись без места не становится полем, или None."""
    if bare_caption(text):
        return "подпись графы без значения"
    if mixed_script(text):
        return "смешение алфавитов в слове"
    return None
