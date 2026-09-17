"""
Приведение прочитанного текста чертежа к записи по ГОСТ.

Зачем это нужно отдельным модулем. Любая модель и любой OCR спотыкаются на
трёх местах: знак диаметра (`⌀110` читается как `0110`, `Ф110`, `ed`), знак
допуска (`±` как `+-`) и гомоглифы — кириллические `Н`, `М`, `К`, `а` вместо
латинских в обозначениях посадок, резьбы и шероховатости. Дальше по
конвейеру эти строки разбираются правилами (`vector_drawing.classify`), и
`20Н7` с кириллической «Н» правилу посадки не соответствует: размер теряет
допуск, а поле — категорию.

Главное ограничение, ради которого правила здесь такие узкие. Сплошная
свёртка гомоглифов делает хуже: она ломает верно прочитанное (`Гайка M4`
превращается в `Гайка МЧ`). Поэтому каждая замена ограничена своим
контекстом — буква меняется только там, где рядом стоит цифра нужного вида,
и только в обозначении, а не в произвольном слове.
"""

from __future__ import annotations

import re
from typing import Tuple

# Знак диаметра во всех начертаниях, которые встречаются в текстовом слое и
# в ответах моделей. Внутри системы он всегда один.
DIAMETER = "⌀"

_DIAMETER_GLYPHS = re.compile(r"[ØøΦϕφ⌀]")
# Кириллическая «Ф» и латинская «O» как знак диаметра — только вплотную к
# числу: «Ф20» это диаметр, «Фаска» — слово.
_DIAMETER_CYRILLIC = re.compile(r"(?<![А-Яа-яЁёA-Za-z])[Фф](?=\s?\d)")

_PLUS_MINUS = re.compile(r"\+\s*/\s*-|\+-|-\+")

# Посадка: буква квалитета после числа. «20Н7» -> «20H7», но «20 Норм» — нет.
_FIT_AFTER_SIZE = re.compile(r"(?<=\d)\s?([НhКкМмРрЕеСсХх])(?=\d{1,2}(?!\d))")
_FIT_CYRILLIC = {"Н": "H", "К": "K", "М": "M", "Р": "P", "Е": "E", "С": "C", "Х": "X",
                 "h": "h", "к": "k", "м": "m", "р": "p", "е": "e", "с": "c", "х": "x"}

# Резьба: «М12», «М12х1,5». Кириллическая «М» встречается почти всегда.
_THREAD_CYRILLIC = re.compile(r"(?<![А-Яа-яЁёA-Za-z])[Мм](?=\s?\d+(?:[.,]\d+)?)")

# Шероховатость: «Ка 3.2», «Ва 3,2», «Rа» с кириллической «а».
_ROUGHNESS = re.compile(r"(?<![А-Яа-яЁёA-Za-z])([RКВВK])\s?([aаzз])(?=\s?\d)")

# Поле допуска резьбы пишется через дефис: «M12-6H».
_THREAD_FIELD = re.compile(r"(?<=\d)-(\d)([НhКк])(?![А-Яа-яЁё\w])")

_SPACES = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Строка как её прочитала модель -> строка в записи по ГОСТ."""
    if not text:
        return ""
    result = _SPACES.sub(" ", str(text)).strip()
    result = _DIAMETER_GLYPHS.sub(DIAMETER, result)
    result = _DIAMETER_CYRILLIC.sub(DIAMETER, result)
    result = _PLUS_MINUS.sub("±", result)
    result = _THREAD_CYRILLIC.sub("M", result)
    result = _ROUGHNESS.sub(lambda m: ("R" if m.group(1) in "RКВВK" else m.group(1))
                            + ("a" if m.group(2) in "aа" else "z"), result)
    result = _FIT_AFTER_SIZE.sub(lambda m: _FIT_CYRILLIC.get(m.group(1), m.group(1)), result)
    result = _THREAD_FIELD.sub(lambda m: "-" + m.group(1) + ("H" if m.group(2) in "Нh" else "K"),
                               result)
    return result.strip()


def looks_like_text(text: str) -> Tuple[bool, str]:
    """
    Похоже ли это на надпись, а не на обрывок графики. Второй элемент —
    причина отказа, чтобы её было видно в логе, а не гадать по пустому месту.
    """
    if not text:
        return False, "пустая строка"
    letters = sum(1 for character in text if character.isalnum())
    if letters < 1:
        return False, "нет ни одной буквы или цифры"
    if letters / len(text) < 0.4:
        return False, "больше половины символов — не буквы и не цифры"
    return True, ""
