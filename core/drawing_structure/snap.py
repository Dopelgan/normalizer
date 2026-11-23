"""
Структура листа, этап S3: рамка элемента — по строке текстового слоя или
распознавания.

Рамки модели грубые (доли картинки, которую она видела, с ошибкой в
несколько процентов листа). Если рядом с элементом есть строка текстового
слоя векторного PDF или строка распознавания (`position.lines`, флаг
RECOGNITION_LINES) с той же надписью, элемент берёт её рамку и
`bbox_exact = true`.

Значение элемента здесь не меняется никогда: строка распознавания даёт
только геометрию. Мусорное прочтение («Каарёинаты отверстий 88.5») с
надписью не совпадёт и рамку не даст, а если совпало — совпала и надпись.

Строки берутся у всех фрагментов страницы — текста, таблиц и распознавания
скана (`candidates._lines_of`); строки другой страницы и строки повёрнутого
неразмещённого листа отсекаются там же. Без RECOGNITION_LINES привязка идёт
только по текстовому слою векторного PDF.
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional, Sequence

from core.drawing_structure.candidates import centre, is_place
from core.drawing_structure.merge import NUMERIC, Element
from core.providers import reading_reconcile

# Поиск строки: центр не дальше этого от центра элемента (доли листа) …
REACH = 0.06
# … или рамки пересекаются с рамкой элемента, расширенной на столько.
PAD = 0.02
# Строка не длиннее надписи во столько раз — иначе берётся её часть по символам.
SPAN_RATIO = 1.3
_TOKEN = re.compile(r"\S+")


def _expanded(bbox: Sequence[float], pad: float) -> List[float]:
    return [bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad]


def _intersects(a: Sequence[float], b: Sequence[float]) -> bool:
    return min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1])


def _span(line_text: str, line_bbox: Sequence[float], start: int, end: int) -> List[float]:
    """Часть горизонтальной строки по символам: моноширинная оценка."""
    x1, y1, x2, y2 = (float(v) for v in line_bbox)
    if (y2 - y1) > (x2 - x1) or not line_text:
        return [x1, y1, x2, y2]     # вертикальная строка: часть не оценить
    total = len(line_text)
    left = x1 + (x2 - x1) * start / total
    right = x1 + (x2 - x1) * end / total
    return [round(left, 4), round(y1, 4), round(right, 4), round(y2, 4)]


def _find(value: str, text: str, numeric: bool) -> Optional[tuple]:
    """Где в строке надпись элемента: (начало, конец) или None."""
    if not value or not text:
        return None
    if numeric:
        core = reading_reconcile.core(value) or value
        if len(reading_reconcile.digits(core)) < 1:
            return None
        for match in reading_reconcile._EXPRESSION.finditer(text):
            if reading_reconcile.same_inscription(match.group(0).strip(), core) and (
                    reading_reconcile.digits(match.group(0)) == reading_reconcile.digits(core)):
                return match.start(), match.end()
        return None
    want = reading_reconcile.notation(value)
    if len(want) < 3:
        return None
    if reading_reconcile.notation(text) == want:
        return 0, len(text)
    lowered = text.casefold()
    index = lowered.find(value.casefold().strip())
    if index >= 0:
        return index, index + len(value.strip())
    return None


def snap(elements: Sequence[Element], lines: Sequence[Dict[str, Any]]) -> int:
    """Привязывает рамки элементов к строкам; возвращает число привязанных."""
    if not lines:
        return 0
    snapped = 0
    for element in elements:
        if element.exact:
            continue
        numeric = element.category in NUMERIC
        best = None
        for line in lines:
            box = line["bbox"]
            if element.placed:
                (ex, ey), (lx, ly) = centre(element.bbox), centre(box)
                near = math.hypot(ex - lx, ey - ly) <= REACH or _intersects(
                    _expanded(element.bbox, PAD), box)
                if not near:
                    continue
            found = _find(element.value, line["text"], numeric)
            if found is None:
                continue
            distance = (math.hypot(*(a - b for a, b in zip(centre(element.bbox), centre(box))))
                        if element.placed else 0.0)
            if best is None or distance < best[0]:
                best = (distance, line, found, 1)
            elif not element.placed:
                best = (best[0], best[1], best[2], best[3] + 1)
        if best is None or (not element.placed and best[3] > 1):
            # Без места надпись привязывается только к единственной строке.
            continue
        _, line, (start, end), _ = best
        text = line["text"]
        box = list(line["bbox"])
        width = box[2] - box[0]
        if len(text) > SPAN_RATIO * max(1, end - start) and width > 0:
            box = _span(text, box, start, end)
        if not is_place(box):
            continue
        element.bbox = [round(float(v), 4) for v in box]
        element.exact = True
        snapped += 1
    return snapped
