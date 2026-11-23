"""
Структура листа, этап S2: сведение прочтений одной надписи в один элемент.

Одна надпись приходит несколькими прочтениями: с листа целиком, с вырезки
вида, из текстового слоя, дочитыванием D4. Здесь кандидаты собираются в
группы по месту и по записи, и из группы выходит один элемент, остальные
прочтения — в его `alternatives`.

Правила (риск §9 плана: два «R1» на одном виде — разные надписи):

* значение одно не сводит никогда — нужно и место: центры ближе высоты
  надписи (у одного прохода) или пересечение рамок; разные проходы читают
  разные картинки, их рамки грубее — до `CROSS_PASS` листа;
* величины сравниваются по числовому скелету (`reading_reconcile.same_inscription`),
  прочие надписи — по записи с допуском на одну-две буквы («Постанов»/«Постнов»);
* кандидат без места (рамка — весь лист или область) сводится только с
  единственным элементом той же записи, иначе остаётся отдельным;
* победитель — по порядку D3: текстовый слой > согласованное прочтение >
  дочитывание в родном разрешении > вырезка вида > лист целиком.

Вызовов модели нет.
"""

from __future__ import annotations

import difflib
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from core.drawing_structure.candidates import (
    PASS_REREAD,
    PASS_SHEET,
    PASS_VIEW_CROP,
    SOURCE_VECTOR,
    Candidate,
    centre,
    is_place,
)
from core.providers import reading_reconcile

NUMERIC = ("size", "tolerance", "thread", "radius", "roughness")
# Разные проходы: рамки модели на разных картинках расходятся сильнее.
CROSS_PASS = 0.05
MIN_NEAR = 0.012
SIMILAR = 0.8
# Рамки накрывают друг друга на столько — одно место, даже при расхождении цифр.
SAME_SPOT = 0.5
# Две надписи одного вида с разными значениями, но одним местом: одно — ошибка чтения.
_RANK_PASS = {PASS_REREAD: 2.5, PASS_VIEW_CROP: 2.0, PASS_SHEET: 1.0}


@dataclass
class Element:
    """Сведённая надпись: значение победителя и все прочтения."""

    category: str
    value: str
    bbox: List[float]
    exact: bool
    source: str
    passes: List[str]
    confidence: float
    agreement: Optional[str] = None
    confirmed: bool = False
    tolerance: Optional[str] = None
    nature: str = "executive"
    link: Optional[Dict[str, Any]] = None
    handwritten: bool = False
    label: str = ""
    alternatives: List[Dict[str, Any]] = field(default_factory=list)
    field_refs: List[Tuple[int, int]] = field(default_factory=list)
    members: List[Candidate] = field(default_factory=list)
    # Сколько прочтений-повторов слилось в элемент (без него самого).
    merged: int = 0

    @property
    def placed(self) -> bool:
        return is_place(self.bbox)

    def needs_check(self) -> bool:
        """Прочтения расходятся, а ни текстового слоя, ни согласия нет."""
        if self.confirmed:
            return False
        if self.agreement == "conflict":
            return True
        if self.source == SOURCE_VECTOR or self.agreement in ("vector", "agreed"):
            return False
        return bool(self.alternatives)


def family(category: str) -> str:
    """Группа категорий, внутри которой прочтения одной надписи сравнимы."""
    if category in NUMERIC or category == "position":
        return "value"
    if category in ("title_block", "material"):
        return "title"
    return category or "other"


def rank(candidate: Candidate) -> Tuple[float, float]:
    score = _RANK_PASS.get(candidate.pass_, 1.0)
    if candidate.agreement in ("agreed",):
        score = 3.0
    if candidate.source == SOURCE_VECTOR or candidate.agreement == "vector":
        score = 4.0
    if candidate.confirmed:
        score = 5.0
    return score, candidate.confidence


def height(bbox: Sequence[float]) -> float:
    return max(0.0, float(bbox[3]) - float(bbox[1]))


def overlap(a: Sequence[float], b: Sequence[float]) -> float:
    """Доля меньшей рамки, накрытая другой."""
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    smaller = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return (w * h) / smaller if smaller > 0 else 0.0


def same_place(a: Candidate, b: Element) -> bool:
    if not a.placed or not b.placed:
        return False
    if overlap(a.bbox, b.bbox) >= 0.3:
        return True
    (ax, ay), (bx, by) = centre(a.bbox), centre(b.bbox)
    distance = math.hypot(ax - bx, ay - by)
    near = max(height(a.bbox), height(b.bbox), MIN_NEAR)
    if a.pass_ in b.passes and a.source == b.source:
        return distance <= near
    return distance <= max(near, CROSS_PASS)


def same_text(a: str, b: str, numeric: bool) -> bool:
    if numeric:
        if reading_reconcile.notation(a) == reading_reconcile.notation(b):
            return True
        return reading_reconcile.same_inscription(a, b) and bool(reading_reconcile.digits(a))
    fa, fb = reading_reconcile.notation(a), reading_reconcile.notation(b)
    if not fa or not fb:
        return False
    if fa == fb:
        return True
    if reading_reconcile.digits(fa) != reading_reconcile.digits(fb):
        return False
    return difflib.SequenceMatcher(None, fa, fb).ratio() >= SIMILAR


def _matches(candidate: Candidate, element: Element) -> bool:
    if family(candidate.category) != family(element.category) and (
            reading_reconcile.notation(candidate.value) != reading_reconcile.notation(element.value)):
        return False
    numeric = candidate.category in NUMERIC or element.category in NUMERIC
    same_kind = (reading_reconcile.kind(candidate.value) == reading_reconcile.kind(element.value)
                 and reading_reconcile.groups(candidate.value) == reading_reconcile.groups(element.value))
    if (numeric and same_kind and candidate.placed and element.placed
            and overlap(candidate.bbox, element.bbox) >= SAME_SPOT
            and reading_reconcile.digits(candidate.value)):
        # Одно место, тот же вид и число чисел — одна надпись, прочитанная
        # с ошибкой в цифре («⌀8,8» против «⌀8,5» текстового слоя).
        return True
    if not same_text(candidate.value, element.value, numeric) and not any(
            same_text(candidate.value, member.value, numeric) for member in element.members):
        return False
    return True


def merge(candidates: Sequence[Candidate]) -> List[Element]:
    """Кандидаты страницы -> элементы; порядок — по силе прочтения, затем по месту."""
    ordered = sorted(candidates, key=rank, reverse=True)
    elements: List[Element] = []
    unplaced: List[Candidate] = []
    for candidate in ordered:
        if not candidate.placed:
            unplaced.append(candidate)
            continue
        target = next((e for e in elements if same_place(candidate, e) and _matches(candidate, e)), None)
        if target is None:
            elements.append(_new(candidate))
        else:
            _absorb(target, candidate)
    for candidate in unplaced:
        key = reading_reconcile.notation(candidate.value)
        same = [e for e in elements if reading_reconcile.notation(e.value) == key
                and family(e.category) == family(candidate.category)]
        if len(same) == 1:
            _absorb(same[0], candidate)
            continue
        # Без места и без единственного двойника — отдельный элемент с той
        # рамкой, что есть (область прочтения); `bbox_exact` у него ложь.
        twin = next((e for e in elements if not e.placed and _matches(candidate, e)), None)
        if twin is not None:
            _absorb(twin, candidate)
        else:
            elements.append(_new(candidate))
    for element in elements:
        _finish(element)
    return elements


def _new(candidate: Candidate) -> Element:
    element = Element(
        category=candidate.category, value=candidate.value, bbox=list(candidate.bbox),
        exact=candidate.exact, source=candidate.source, passes=[candidate.pass_],
        confidence=candidate.confidence, agreement=candidate.agreement,
        confirmed=candidate.confirmed, tolerance=candidate.tolerance, nature=candidate.nature,
        link=candidate.link, handwritten=candidate.handwritten, label=candidate.label,
    )
    element.members.append(candidate)
    if candidate.field_ref is not None:
        element.field_refs.append(candidate.field_ref)
    return element


def _absorb(element: Element, candidate: Candidate) -> None:
    element.members.append(candidate)
    element.merged += 1
    if candidate.field_ref is not None and candidate.field_ref not in element.field_refs:
        element.field_refs.append(candidate.field_ref)
    if candidate.pass_ not in element.passes:
        element.passes.append(candidate.pass_)
    if candidate.placed and (not element.placed or (candidate.exact and not element.exact)):
        element.bbox, element.exact = list(candidate.bbox), candidate.exact
    if element.link is None and candidate.link is not None:
        element.link = candidate.link
    element.handwritten = element.handwritten or candidate.handwritten
    element.confidence = max(element.confidence, candidate.confidence)
    element.confirmed = element.confirmed or candidate.confirmed


def _finish(element: Element) -> None:
    """Прочтения, отличные от значения, — в `alternatives`, без повторов записи."""
    seen = {reading_reconcile.notation(element.value)}
    alternatives: List[Dict[str, Any]] = []
    sources = set()
    for member in element.members:
        sources.add((member.source, member.pass_))
        readings = [{"value": member.value, "source": member.source, "pass": member.pass_,
                     "confidence": member.confidence}] + list(member.readings)
        for reading in readings:
            key = reading_reconcile.notation(reading["value"])
            if not key or key in seen:
                continue
            seen.add(key)
            alternatives.append({
                "value": reading["value"], "source": reading["source"], "pass": reading["pass"],
                "confidence": round(float(reading["confidence"] if reading["confidence"] is not None
                                          else member.confidence * 0.8), 3),
            })
    element.alternatives = alternatives
    if element.agreement is None and len(sources) >= 2 and not alternatives:
        # Два прохода прочли одно и то же: это согласие, как у D3.
        element.agreement = "agreed"
    for member in element.members:
        # Проходы, прочитавшие надпись по сведению D3 (вырезка, дочитывание).
        for reading in member.readings:
            if reading["pass"] not in element.passes:
                element.passes.append(reading["pass"])
