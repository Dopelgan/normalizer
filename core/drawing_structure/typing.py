"""
Структура листа, этап S4: тип надписи и разбор величины.

Категорию поля дала модель или правила (`vector_drawing.classify`), но
структуре нужна не категория, а тип элемента: размер с видом, номиналом,
допуском и посадкой; ссылка на пункт требований («п4», «п.8»); номер
позиции; заголовок вида; шероховатость; пункт требований; прочая надпись.
Прежде «п4» уходила размером, а «4 отв.» — выноской; здесь это ссылка на
пункт и число отверстий при размере.

Разбор детерминированный, по записи ЕСКД; вызовов модели нет. Что не
распознано как тип, остаётся надписью (`texts`) — ничто не пропадает.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from core.providers import drawing_views

# Тип элемента структуры.
DIMENSION = "dimension"
NOTE_REF = "note_ref"
COUNT = "count"
POSITION = "position"
VIEW_TITLE = "view_title"
SYMBOL = "symbol"
NOTE = "note"
SURFACE = "surface"
TEXT = "text"
TITLE = "title"

_NUMBER = r"\d+(?:[.,]\d+)?"
_NUM = re.compile(_NUMBER)

# «п4», «п.5», «п. 8», «n.10» (латинская «n» — частое прочтение «п»), «пп.3, 4».
_NOTE_REF = re.compile(r"(?<![A-Za-zА-Яа-яЁё])[пПnN]{1,2}\s?\.?\s?(\d{1,2})(?:\s*[,и]\s*(\d{1,2}))?(?!\d|[.,]\d)")
_NOTE_REF_ONLY = re.compile(
    r"^\s*(?:см\.?\s*)?[пПnN]{1,2}\s?\.?\s?\d{1,2}(?:\s*[,и]\s*\d{1,2})?\s*\.?\s*$")
# «4 отв.», «N отв.», «8 отверстий», «4 места», «2 паза» — число элементов.
_COUNT = re.compile(
    r"(?P<count>\d{1,3}|[NnКк]|[A-Za-z])\s*(?P<what>отв(?:ерсти[йея])?\.?|мест(?:а|о)?\.?|паз(?:а|ов)?\.?"
    r"|шт\.?|зуб(?:а|ьев)?\.?|ребр(?:а|о)?\.?)",
    re.IGNORECASE,
)
# Прочтения «4 отв.» моделью: «4N01», «4, Nот», «4 Nот» — число и обломок «отв».
_COUNT_GARBLED = re.compile(r"^\s*(\d{1,3})\s*,?\s*(?:N\s?0?1|N\s?от|Nот|отв?)\s*\.?\s*$", re.IGNORECASE)
_POSITION = re.compile(r"^\s*(?:поз\.?\s*)?(\d{1,3})(?:\s*[.,]\s*(\d{1,3}))?\s*\*?\s*$", re.IGNORECASE)
_ROUGHNESS = re.compile(r"(R[az])\s*(" + _NUMBER + r")", re.IGNORECASE)
_ROUGH_SIGN = re.compile(r"[√∇]")
_GENERAL_ROUGHNESS = re.compile(r"\(\s*[√∇]\s*\)|остальн", re.IGNORECASE)
_NOTE_NUMBER = re.compile(r"^\s*(\d{1,2})\s*[.)]\s*(.+)$", re.S)
_MARKER_DEFINITION = re.compile(r"^\s*(\*{1,3})\s*(?:[-–—]\s*)?(?:размер|разм\.)", re.IGNORECASE)
_MARKER_ANYWHERE = re.compile(r"(?<![\d\w])(\*{1,3})\s*(?:[-–—]\s*)?размер[ыа]?\s+для\s+справ", re.IGNORECASE)
# Слово в пункте требований: «1. 15» — не пункт, «1. Размеры …» — пункт.
_WORD = re.compile(r"[А-Яа-яЁё]{3}")
_SURFACE_WORDS = re.compile(r"(сторон|поверхн|плоскост|кромк|торец|торц)", re.IGNORECASE)

# Величина: знак вида, номинал или выражение, допуск.
_LEAD = re.compile(r"^\s*(?P<lead>⌀|Ø|ø|R|M|М|Sф|S⌀|SR|Tr|G|K|□)?\s*", re.IGNORECASE)
_FORMULA = re.compile(r"\(([^()]*[A-Za-zА-Яа-я][^()]*)\)")
_VARIABLE = re.compile(r"(?<![A-Za-zА-Яа-яЁё])([A-ZА-ЯЁ]|[a-z])(?![A-Za-zА-Яа-яЁё])")
_PLUS_MINUS = re.compile(r"±\s*(" + _NUMBER + r")")
_SIGNED = re.compile(r"([+\-−])\s*(" + _NUMBER + r")")
_FIT = re.compile(r"(?<![A-Za-zА-Яа-яЁё\d])(?P<fit>[A-Za-z]{1,2}\d{1,2}(?:/[A-Za-z]{1,2}\d{1,2})?)(?![\d])")
_THREAD_FIT = re.compile(r"-\s*(?P<fit>\d{1,2}[A-Za-z]{1,2}(?:\d{1,2}[A-Za-z])?)")
_CHAMFER = re.compile(r"^\s*(" + _NUMBER + r")\s*[x×хХ]\s*(" + _NUMBER + r")\s*°")
_ANGLE = re.compile(r"^\s*(" + _NUMBER + r")\s*°")
_THREAD = re.compile(r"^\s*[MМ]\s*(" + _NUMBER + r")(?:\s*[x×хХ]\s*(" + _NUMBER + r"))?", re.IGNORECASE)
_PLAIN = re.compile(r"^\s*(" + _NUMBER + r")")


@dataclass
class Typed:
    """Итог разбора одной надписи."""

    kind: str
    # Разбор размера (`DIMENSION`): поля элемента `dimensions[]`.
    dimension: Dict[str, Any] = field(default_factory=dict)
    note_refs: List[int] = field(default_factory=list)
    count: Optional[str] = None
    number: Optional[str] = None
    view: Optional[Tuple[str, str, str]] = None
    symbol: Dict[str, Any] = field(default_factory=dict)
    note_number: Optional[int] = None
    note_text: str = ""
    marker: Optional[str] = None
    label: str = ""


def _float(text: str) -> Optional[float]:
    try:
        return float(text.replace(",", ".").replace("−", "-"))
    except (TypeError, ValueError):
        return None


def note_refs(text: str) -> List[int]:
    """Номера пунктов требований, на которые ссылается надпись: «п.8», «пп.3, 4»."""
    found: List[int] = []
    for match in _NOTE_REF.finditer(text or ""):
        for group in match.groups():
            if group and int(group) not in found:
                found.append(int(group))
    return found


def strip_note_refs(text: str) -> str:
    return " ".join(_NOTE_REF.sub(" ", text or "").split()).strip(" ,;")


def count_of(text: str) -> Optional[str]:
    """Число элементов из «4 отв.», «N отв.», «4 места», «4N01»; иначе None."""
    garbled = _COUNT_GARBLED.match(text or "")
    if garbled:
        return garbled.group(1)
    match = _COUNT.search(text or "")
    if not match:
        return None
    count = match.group("count")
    return count.upper() if not count.isdigit() else count


def _strip_count(text: str) -> str:
    if _COUNT_GARBLED.match(text or ""):
        return ""
    return " ".join(_COUNT.sub(" ", text or "").split()).strip(" ,;")


def tolerance(rest: str, thread: bool = False) -> Optional[Dict[str, Any]]:
    """Допуск после номинала: «±0,2», «-0,20+0,05», «H7», «h14», «-6g»."""
    rest = rest or ""
    out: Dict[str, Any] = {}
    plus_minus = _PLUS_MINUS.search(rest)
    if plus_minus:
        value = _float(plus_minus.group(1))
        out.update(upper=value, lower=-value if value is not None else None, source="explicit")
    else:
        signed = [(sign, _float(number)) for sign, number in _SIGNED.findall(rest)]
        signed = [(sign, number) for sign, number in signed if number is not None]
        if signed:
            values = [number if sign == "+" else -number for sign, number in signed[:2]]
            if len(values) == 1:
                values.append(0.0)
            out.update(upper=max(values), lower=min(values), source="explicit")
    fit = (_THREAD_FIT.search(rest) if thread else None) or _FIT.search(rest)
    if fit:
        out["fit"] = fit.group("fit")
        out.setdefault("source", "fit_notation")
    return out or None


def parse_dimension(value: str, category: str = "size") -> Dict[str, Any]:
    """
    Размер -> {kind, nominal, unit, tolerance, reference, count, formula}.

    Виды: linear, diameter, radius, angle, thread, chamfer. Параметрический
    размер «(X+38)±0,2» номинала не имеет: выражение идёт в `formula`.
    """
    text = " ".join(str(value or "").split())
    count = count_of(text)
    body = _strip_count(text) if count else text
    reference = bool(re.search(r"\d\s*\*|\)\s*\*", body))
    body_clean = body.replace("*", "").strip()
    out: Dict[str, Any] = {"value_raw": text, "kind": "linear", "nominal": None, "unit": "мм",
                           "tolerance": None, "reference": reference}
    if count:
        out["count"] = count
    formula = _FORMULA.search(body_clean)
    if formula:
        expr = formula.group(1).replace(" ", "")
        variables = sorted(set(_VARIABLE.findall(expr)))
        if variables:
            out["formula"] = {"expr": expr, "variables": variables}
            lead = body_clean[:formula.start()]
            if "⌀" in lead or "Ø" in lead:
                out["kind"] = "diameter"
            elif lead.strip().upper().startswith("R"):
                out["kind"] = "radius"
            out["tolerance"] = tolerance(body_clean[formula.end():])
            return out
    chamfer = _CHAMFER.match(body_clean)
    if chamfer:
        out.update(kind="chamfer", nominal=_float(chamfer.group(1)))
        out["angle"] = _float(chamfer.group(2))
        return out
    thread = _THREAD.match(body_clean)
    if thread and (category == "thread" or thread.group(2) or re.match(r"^\s*[MМ]\s?\d", body_clean)):
        out.update(kind="thread", nominal=_float(thread.group(1)))
        if thread.group(2):
            out["pitch"] = _float(thread.group(2))
        out["tolerance"] = tolerance(body_clean[thread.end():], thread=True)
        return out
    lead = _LEAD.match(body_clean)
    sign = (lead.group("lead") or "") if lead else ""
    rest = body_clean[lead.end():] if lead else body_clean
    if sign in ("⌀", "Ø", "ø"):
        out["kind"] = "diameter"
    elif sign.upper() in ("R", "SR"):
        out["kind"] = "radius"
    angle = _ANGLE.match(rest)
    if angle and out["kind"] == "linear":
        out.update(kind="angle", nominal=_float(angle.group(1)), unit="°")
        out["tolerance"] = tolerance(rest[angle.end():])
        return out
    plain = _PLAIN.match(rest)
    if plain:
        out["nominal"] = _float(plain.group(1))
        out["tolerance"] = tolerance(rest[plain.end():])
    return out


def _is_dimension_text(text: str) -> bool:
    """Похоже на размер: число в начале (со знаком вида) или выражение."""
    body = strip_note_refs(text)
    if not body or len(body) > 40:
        return False
    if _FORMULA.search(body) and _NUM.search(body):
        return True
    lead = _LEAD.match(body)
    rest = body[lead.end():] if lead else body
    return bool(_PLAIN.match(rest) or _THREAD.match(body) or _CHAMFER.match(body))


def classify(value: str, category: str) -> Typed:
    """Тип надписи по её записи и категории прохода."""
    text = " ".join(str(value or "").split())
    if category in ("title_block", "material"):
        return Typed(kind=TITLE)
    if category == "note":
        numbered = _NOTE_NUMBER.match(text)
        marker = _MARKER_ANYWHERE.search(text) or (
            _MARKER_DEFINITION.match(numbered.group(2)) if numbered else None)
        if numbered and _WORD.search(numbered.group(2)):
            return Typed(kind=NOTE, note_number=int(numbered.group(1)), note_text=numbered.group(2).strip(),
                         marker=marker.group(1) if marker else None, note_refs=note_refs(numbered.group(2)))
    title = drawing_views.parse_title(text)
    if title is not None and category not in ("size", "radius", "thread", "tolerance") or (
            title is not None and title[2]):
        return Typed(kind=VIEW_TITLE, view=title)
    if _NOTE_REF_ONLY.match(text):
        return Typed(kind=NOTE_REF, note_refs=note_refs(text))
    if category == "roughness" or (_ROUGHNESS.search(text) and len(text) <= 24):
        rough = _ROUGHNESS.search(text)
        symbol = {"kind": "roughness", "value": f"{rough.group(1)} {rough.group(2)}" if rough else text,
                  "scope": "general" if _GENERAL_ROUGHNESS.search(text) else None}
        return Typed(kind=SYMBOL, symbol=symbol, note_refs=note_refs(text))
    count = count_of(text)
    if count and not _is_dimension_text(_strip_count(text)):
        return Typed(kind=COUNT, count=count, note_refs=note_refs(text))
    position = _POSITION.match(text)
    if category == "position" and position:
        number = position.group(1) + (f".{position.group(2)}" if position.group(2) else "")
        return Typed(kind=POSITION, number=number)
    if category in ("size", "radius", "thread", "tolerance", "position", "callout") and _is_dimension_text(text):
        if category == "callout" and _POSITION.match(text):
            # Число без знака на выноске — номер позиции, а не размер.
            return Typed(kind=POSITION, number=_POSITION.match(text).group(1))
        refs = note_refs(text)
        dimension = parse_dimension(strip_note_refs(text) if refs else text, category)
        return Typed(kind=DIMENSION, dimension=dimension, note_refs=refs, count=dimension.get("count"))
    if category == "callout":
        refs = note_refs(text)
        rest = strip_note_refs(text)
        if _SURFACE_WORDS.search(rest):
            return Typed(kind=SURFACE, label=rest, note_refs=refs)
        if refs and not rest:
            return Typed(kind=NOTE_REF, note_refs=refs)
        return Typed(kind=TEXT, note_refs=refs)
    if category == "note":
        refs = note_refs(text)
        if _SURFACE_WORDS.search(text) and len(text) <= 40 and refs:
            return Typed(kind=SURFACE, label=strip_note_refs(text), note_refs=refs)
    return Typed(kind=TEXT, note_refs=note_refs(text) if len(text) <= 40 else [])


def transcript_notes(lines: List[str]) -> List[Tuple[int, str]]:
    """
    Пункты технических требований из строк прохода «весь текст листа»:
    «1. Размеры для справок.» и продолжения пункта на следующих строках.
    """
    notes: List[Tuple[int, str]] = []
    started = False
    for line in lines:
        text = line.strip()
        if re.match(r"^технические\s+требования", text, re.IGNORECASE):
            started = True
            continue
        numbered = _NOTE_NUMBER.match(text)
        if numbered and len(numbered.group(2)) >= 8 and _WORD.search(numbered.group(2)):
            number = int(numbered.group(1))
            if notes and number <= notes[-1][0]:
                started = False
                continue
            notes.append((number, numbered.group(2).strip()))
            started = True
            continue
        if started and notes and text and not _NOTE_NUMBER.match(text) and re.match(r"^[а-яё(]", text):
            number, body = notes[-1]
            notes[-1] = (number, f"{body} {text}")
    return notes


def marker_of_note(text: str) -> Optional[str]:
    """Звёздочка, которую определяет пункт: «* Размеры для справок» -> «*»."""
    match = _MARKER_DEFINITION.match(text or "") or _MARKER_ANYWHERE.search(text or "")
    return match.group(1) if match else None


def position_number(text: str) -> Optional[str]:
    match = _POSITION.match(text or "")
    return match.group(1) if match else None
