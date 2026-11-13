"""
Геометрия трёх настоящих листов для проверки связи величины с видом.

Это не синтетика: числа сняты с листов ТАДУ 687000.001.001 (крышка),
ТАДУ 302182.001 ГЧ (расширитель) и ТАДУ 305179.001.03 СБ (пробка ЭК150) —
тех самых, по которым 13.11 задавались 27 бытовых вопросов. Лист открывался
так же, как в бою: рендер в `DRAWING_RENDER_DPI`, выравнивание, маска
тёмного, стирание штампа, граф, длинных прямых и связного текста.

Почему в тестах лежит сетка клеток, а не лист. Проверяется связывание, и в
него уходит ровно это: какие клетки крупной сетки заняты чернилами
(`_CELL` — 1,2 % большей стороны, клетка 12 px на 1000) и где лежат рамки
надписей. Самих чертежей заказчика в репозитории нет, а по двоичной сетке
12-пиксельных клеток чертёж не восстановить.

В каждом листе:

* ``grid`` — строки сетки, «1» — в клетке есть тёмное. Это уже результат
  стирания штампа и прозы, то есть вход `_components`.
* ``annotations`` — надписи листа: текст, рамка в долях листа и эталон. В
  эталоне стоит имя вида, которому надпись принадлежит на самом деле; ``""``
  — не принадлежит никакому (общая шероховатость, требования, главный вид
  без заголовка), ``"title"`` — сама надпись и есть заголовок вида,
  ``"pointer"`` — буква-указание у стрелки или выноски. Рамки сняты с
  растра по сетке 0,05; в бою их даёт модель.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any, Dict, List, Tuple

_PATH = pathlib.Path(__file__).with_name("real_sheet_views.json")

SHEETS: Tuple[str, ...] = ("kryshka_p1", "rasshir_p1", "probka_p1")


def sheet(name: str) -> Dict[str, Any]:
    """Снятая с листа сетка, надписи и эталон привязки."""
    with _PATH.open(encoding="utf-8") as handle:
        return json.load(handle)[name]


def grid(data: Dict[str, Any]):
    """Сетка клеток в том виде, в каком её ждёт `_components`."""
    import numpy

    rows = data["grid"]
    return numpy.array([[char == "1" for char in row] for row in rows], dtype=bool)


def step(data: Dict[str, Any]) -> Tuple[float, float]:
    return (data["step"][0], data["step"][1])


def fields(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Надписи -> поля контракта той же формы, что даёт ветка чертежей."""
    from core.providers import vector_drawing

    out: List[Dict[str, Any]] = []
    for value, bbox, _role in data["annotations"]:
        category, _confidence = vector_drawing.classify(value)
        out.append({"category": category, "value": value, "bbox": list(bbox)})
    return out


def expected(data: Dict[str, Any]) -> Dict[str, str]:
    """Эталон: надпись -> имя вида (``""`` — вида нет)."""
    return {
        value: role for value, _bbox, role in data["annotations"]
        if role not in ("title", "pointer")
    }


def title_names(data: Dict[str, Any]) -> Dict[str, str]:
    """Рамка заголовка -> имя вида в эталоне (с дефисом вместо тире)."""
    out: Dict[str, str] = {}
    for value, bbox, role in data["annotations"]:
        if role != "title":
            continue
        name = value.split("(")[0].replace("–", "-").replace("—", "-").strip()
        out[json.dumps([round(c, 4) for c in bbox])] = name or value
    return out
