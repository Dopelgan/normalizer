"""
Извлечение фактов и значений из фрагментов (контракт, раздел 4).

Задача — отдать RAG не только текстовый chunk, но и явные значения: числа,
диапазоны, даты, сроки, единицы измерения и идентификаторы. Отбор фактов
под конкретный вопрос пользователя здесь не делается: Parser извлекает,
решает RAG.

Правила, которые здесь закодированы:

* значения и единицы не выдумываются — только то, что есть в тексте;
* `value.raw` сохраняет исходную запись целиком, вместе с запятой как
  десятичным разделителем и с исходной единицей;
* факт из таблицы помнит свои координаты: строка, столбец и заголовок
  лежат в `provenance`;
* `fact_id` стабилен, пока стабилен `fragment_id`.
"""

import re
from typing import Any, Dict, List, Optional, Tuple

from core.facts.units import (
    DURATION_RE,
    UNIT_RE,
    canonical,
    canonical_duration,
)
from core.models.contract import ExtractedFact, FactValue, Fragment

# Максимум фактов на фрагмент: страница прайс-листа иначе даёт тысячи
# значений, и полезность их падает до нуля.
MAX_FACTS_PER_FRAGMENT = 60

_SPACE = "[    ]"
_NUMBER = rf"\d{{1,3}}(?:{_SPACE}\d{{3}})+(?:[.,]\d+)?|\d+(?:[.,]\d+)?"
_SIGNED = rf"[-+−]?(?:{_NUMBER})"
_DASH = r"(?:\.\.\.|\.\.|—|–|-|÷)"

# Порядок важен: диапазон ищется раньше одиночного числа, иначе «0,4-0,6»
# распадётся на два несвязанных значения.
RANGE_RE = re.compile(
    rf"(?P<min>{_SIGNED}){_SPACE}*{_DASH}{_SPACE}*(?P<max>{_SIGNED})"
    rf"(?:{_SPACE}*(?P<unit>[^\s,;.]+))?"
)
PLUSMINUS_RE = re.compile(
    rf"(?P<base>{_SIGNED})(?:{_SPACE}*(?P<base_unit>[^\s,;.±]+))?"
    rf"{_SPACE}*(?:±|\+/-){_SPACE}*(?P<delta>{_NUMBER})"
    rf"(?:{_SPACE}*(?P<unit>[^\s,;.]+))?"
)
# Оборот «от X до Y» — тот же диапазон, только словами. Без него он
# распадался на два независимых значения с «>=» и «<=».
FROM_TO_RE = re.compile(
    rf"\bот{_SPACE}+(?P<min>{_SIGNED})(?:{_SPACE}*(?P<min_unit>[^\s,;.]+?))?"
    rf"{_SPACE}+до{_SPACE}+(?P<max>{_SIGNED})(?:{_SPACE}*(?P<unit>[^\s,;.]+))?",
    re.IGNORECASE,
)
NUMBER_RE = re.compile(rf"(?P<number>{_SIGNED})(?:{_SPACE}*(?P<unit>[^\s,;.)]+))?")
DATE_RE = re.compile(
    r"\b(?:\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2})?Z?)?"
    r"|\d{1,2}\.\d{1,2}\.\d{2,4})\b"
)
IDENTIFIER_RE = re.compile(
    r"\b(?:ГОСТ(?:\s+Р)?|ОСТ|СТО|ТУ|СНиП|ISO|DIN|EN|ASTM|ANSI|IEC)"
    r"\s+[\dA-Za-zА-Яа-я][\w.\-–/]*",
    re.IGNORECASE,
)
# Децимальный номер конструкторского документа: ТАДУ 405220.002, АБВГ.123.456
DECIMAL_ID_RE = re.compile(r"\b[А-ЯA-Z]{2,6}[ .]\d{3,}[.\d]*\b")

# Оператор сравнения задаётся словами или знаком. Список намеренно короткий:
# сомнительные обороты лучше оставить равенством, чем переврать смысл.
_OPERATORS: List[Tuple[str, str]] = [
    ("не более", "<="), ("не выше", "<="), ("не превышает", "<="),
    ("не должна превышать", "<="), ("не должен превышать", "<="),
    ("максимум", "<="), ("максимальн", "<="), ("до", "<="),
    ("не менее", ">="), ("не ниже", ">="), ("минимум", ">="),
    ("минимальн", ">="), ("от", ">="),
    ("≤", "<="), ("<=", "<="), ("≥", ">="), (">=", ">="),
]
_SIGN_ONLY = [("<", "<"), (">", ">")]

_CLAUSE_SPLIT = re.compile(r"[;\n\r]+|(?<=[а-яa-z0-9)])\.\s+(?=[А-ЯA-Z])")
_LABEL_SPLIT = re.compile(r"\s[—–-]\s|:\s|\s—|\s–")


# ===========================================================================
# Разбор значений
# ===========================================================================

def to_number(raw: str) -> Optional[float]:
    """«1 200,5» -> 1200.5. Возвращает None, когда это не число."""
    cleaned = (
        raw.replace(" ", "").replace(" ", "").replace(" ", "")
        .replace(" ", "").replace("−", "-").replace(",", ".")
    )
    try:
        return float(cleaned)
    except ValueError:
        return None


def _unit_of(candidate: Optional[str]) -> Optional[str]:
    """Единица, если хвост после числа ею и является."""
    if not candidate:
        return None
    match = UNIT_RE.match(candidate.strip())
    if match and match.end() == len(candidate.strip()):
        return canonical(match.group("unit"))
    return None


def _duration_of(candidate: Optional[str]) -> Optional[str]:
    if not candidate:
        return None
    match = DURATION_RE.match(candidate.strip())
    if match and match.end() == len(candidate.strip()):
        return canonical_duration(match.group("unit"))
    return None


def _operator(prefix: str) -> str:
    """Оператор из слов перед числом. По умолчанию — равенство."""
    lowered = prefix.lower()
    best: Tuple[int, str] = (-1, "=")
    for word, operator in _OPERATORS:
        position = lowered.rfind(word)
        if position > best[0]:
            best = (position, operator)
    if best[0] >= 0:
        return best[1]
    for sign, operator in _SIGN_ONLY:
        if lowered.rstrip().endswith(sign):
            return operator
    return "="


# Предлоги и союзы сами по себе названием параметра не являются.
_STOP_TAIL = {
    "по", "в", "во", "на", "с", "со", "из", "для", "при", "за", "к", "и",
    "а", "но", "или", "же", "что", "как",
}


def _strip_operator(text: str) -> str:
    """Хвост вроде «не более» относится к значению, а не к названию."""
    cleaned = text.strip(" \t.,:;—–-()")
    changed = True
    while changed and cleaned:
        changed = False
        lowered = cleaned.lower()
        for word, _operator in _OPERATORS:
            if lowered.endswith(word):
                cleaned = cleaned[: len(cleaned) - len(word)].strip(" \t.,:;—–-()")
                changed = True
                break
    return cleaned


def _label(clause: str, start: int, after: int = 0) -> str:
    """
    Название параметра так, как оно записано в документе.

    Сначала пробуем разделитель «название — значение»; если его нет, берём
    несколько слов перед самим значением. Отсчёт идёт от конца предыдущего
    значения (`after`): иначе название второго факта в клаузе вбирает
    в себя первый — «не более 30 суток по ГОСТ ...».
    """
    head = clause[max(after, 0):start].strip(" \t\u00a0")
    if not head:
        return ""
    parts = [part for part in _LABEL_SPLIT.split(head) if part and part.strip()]
    candidate = _strip_operator(parts[-1] if parts else head)
    # «Срок хранения — не более 30 суток»: после разделителя стоит один
    # оператор, и название осталось слева от него.
    if not candidate and len(parts) > 1:
        candidate = _strip_operator(parts[-2])
    words = candidate.split()
    # Служебное слово — свойство фразы, а не название параметра.
    while words and words[-1].lower().strip(".,;:") in _STOP_TAIL:
        words.pop()
    words = words[-8:]
    label = " ".join(words).strip(" .,:;—–-")
    if label and label[:1].isupper() and not label.isupper():
        label = label[0].lower() + label[1:]
    return label


# ===========================================================================
# Текст
# ===========================================================================

def _spans(clause: str) -> List[Tuple[int, int, FactValue]]:
    """Непересекающиеся значения клаузы в порядке появления."""
    found: List[Tuple[int, int, FactValue]] = []
    taken: List[Tuple[int, int]] = []

    def free(start: int, end: int) -> bool:
        return all(end <= s or start >= e for s, e in taken)

    def take(start: int, end: int, value: FactValue) -> None:
        taken.append((start, end))
        found.append((start, end, value))

    for match in DATE_RE.finditer(clause):
        take(match.start(), match.end(),
             FactValue(kind="date", raw=match.group(0), operator="="))

    for regex in (IDENTIFIER_RE, DECIMAL_ID_RE):
        for match in regex.finditer(clause):
            # Точка в конце предложения к обозначению не относится, а вот
            # точка внутри («405220.002») — относится.
            raw = match.group(0).strip().rstrip(".,;:")
            if raw and free(match.start(), match.start() + len(raw)):
                take(match.start(), match.start() + len(raw),
                     FactValue(kind="identifier", raw=raw, operator="="))

    for match in PLUSMINUS_RE.finditer(clause):
        if not free(match.start(), match.end()):
            continue
        base, delta = to_number(match.group("base")), to_number(match.group("delta"))
        if base is None or delta is None:
            continue
        unit = _unit_of(match.group("unit")) or _unit_of(match.group("base_unit"))
        end = match.end() if _unit_of(match.group("unit")) else match.end("delta")
        take(match.start(), end, FactValue(
            kind="range", raw=clause[match.start():end].strip(),
            min=round(base - delta, 10), max=round(base + delta, 10),
            unit=unit, operator="range",
        ))

    for match in FROM_TO_RE.finditer(clause):
        if not free(match.start(), match.end()):
            continue
        low, high = to_number(match.group("min")), to_number(match.group("max"))
        if low is None or high is None or high < low:
            continue
        unit = _unit_of(match.group("unit")) or _unit_of(match.group("min_unit"))
        duration = None if unit else _duration_of(match.group("unit"))
        end = match.end() if (_unit_of(match.group("unit")) or duration) else match.end("max")
        take(match.start(), end, FactValue(
            kind="duration" if duration else "range",
            raw=clause[match.start():end].strip(),
            min=low, max=high, unit=unit or duration, operator="range",
        ))

    for match in RANGE_RE.finditer(clause):
        if not free(match.start(), match.end()):
            continue
        low, high = to_number(match.group("min")), to_number(match.group("max"))
        if low is None or high is None or high < low:
            continue
        unit = _unit_of(match.group("unit"))
        duration = None if unit else _duration_of(match.group("unit"))
        end = match.end() if (unit or duration) else match.end("max")
        take(match.start(), end, FactValue(
            kind="duration" if duration else "range",
            raw=clause[match.start():end].strip(),
            min=low, max=high, unit=unit or duration, operator="range",
        ))

    for match in NUMBER_RE.finditer(clause):
        if not free(match.start(), match.end("number")):
            continue
        number = to_number(match.group("number"))
        if number is None:
            continue
        unit = _unit_of(match.group("unit"))
        duration = None if unit else _duration_of(match.group("unit"))
        end = match.end() if (unit or duration) else match.end("number")
        take(match.start(), end, FactValue(
            kind="duration" if duration else "number",
            raw=clause[match.start():end].strip(),
            number=number, unit=unit or duration,
            operator=_operator(clause[:match.start()]),
        ))

    found.sort(key=lambda item: item[0])
    return found


def facts_from_text(
    text: str, fragment_id: str, key: str = "document.parameter", start_index: int = 1
) -> List[ExtractedFact]:
    """Факты из связного текста: числа, диапазоны, даты, сроки, обозначения."""
    facts: List[ExtractedFact] = []
    index = start_index
    for clause in _CLAUSE_SPLIT.split(text or ""):
        clause = clause.strip()
        if not clause:
            continue
        previous_end = 0
        for start, end, value in _spans(clause):
            label = _label(clause, start, previous_end)
            previous_end = end
            facts.append(ExtractedFact(
                fact_id=f"{fragment_id}:fact-{index}",
                fragment_id=fragment_id,
                key=key,
                label=label,
                value=value,
                confidence=_confidence(value, label),
            ))
            index += 1
            if len(facts) >= MAX_FACTS_PER_FRAGMENT:
                return facts
    return facts


def _confidence(value: FactValue, label: str) -> float:
    """
    Уверенность в факте, а не в тексте: единица и название параметра рядом
    со значением — это и есть признаки того, что значение прочитано верно.
    """
    score = 0.85
    if value.unit:
        score += 0.08
    if label:
        score += 0.05
    if value.kind in ("identifier", "date"):
        score = 0.95 if not label else 0.97
    return round(min(score, 0.98), 3)


# ===========================================================================
# Таблицы
# ===========================================================================

def facts_from_table(
    table: Dict[str, Any], fragment_id: str, key: str = "document.parameter"
) -> List[ExtractedFact]:
    """
    Факты из таблицы. Каждый факт помнит строку, столбец и заголовок: без
    координат значение из таблицы невозможно показать в исходной ячейке.
    """
    headers = [str(h) for h in (table.get("headers") or [])]
    facts: List[ExtractedFact] = []
    index = 1

    for row_number, row in enumerate(table.get("rows") or []):
        if not isinstance(row, (list, tuple)) or not row:
            continue
        row_label = str(row[0]).strip() if row[0] is not None else ""
        row_label_is_value = to_number(row_label) is not None
        for column, cell in enumerate(row):
            if cell is None:
                continue
            raw = str(cell).strip()
            if not raw or (column == 0 and not row_label_is_value):
                continue
            spans = _spans(raw)
            if not spans:
                continue
            header = headers[column] if column < len(headers) else ""
            label = row_label if not row_label_is_value else header
            for _start, _end, value in spans:
                facts.append(ExtractedFact(
                    fact_id=f"{fragment_id}:fact-{index}",
                    fragment_id=fragment_id,
                    key=key,
                    label=(label or header).strip(),
                    value=value,
                    confidence=_confidence(value, label or header),
                    provenance={"row": row_number, "column": column, "header": header},
                ))
                index += 1
                if len(facts) >= MAX_FACTS_PER_FRAGMENT:
                    return facts
    return facts


# ===========================================================================
# Чертежи
# ===========================================================================

def facts_from_drawing_fields(fields, fragment_id: str) -> List[ExtractedFact]:
    """
    Факты из разобранных полей чертежа: размер, допуск, шероховатость,
    материал. Категория поля становится ключом факта — так RAG отличает
    исполнительный размер от справочного, не разбирая текст заново.
    """
    facts: List[ExtractedFact] = []
    for index, field in enumerate(fields or [], start=1):
        raw = str(getattr(field, "value", "") or "").strip()
        if not raw:
            continue
        category = str(getattr(field, "category", "") or "unknown")
        unit = canonical(getattr(field, "unit", "") or None)
        number = to_number(raw)
        tolerance = getattr(field, "tolerance", None)
        delta = to_number(str(tolerance).lstrip("±+") ) if tolerance else None

        if number is not None and delta is not None:
            value = FactValue(
                kind="range", raw=f"{raw} {tolerance}".strip(),
                min=round(number - delta, 10), max=round(number + delta, 10),
                unit=unit, operator="range",
            )
        elif number is not None:
            value = FactValue(kind="number", raw=raw, number=number, unit=unit, operator="=")
        else:
            value = FactValue(kind="text", raw=raw, unit=unit, operator="=")

        facts.append(ExtractedFact(
            fact_id=f"{fragment_id}:fact-{index}",
            fragment_id=fragment_id,
            key=f"drawing.{category}",
            label=category,
            value=value,
            confidence=round(float(getattr(field, "confidence", 0.0) or 0.0), 3),
            provenance={
                "nature": getattr(field, "nature", None),
                "source_of_tolerance": getattr(field, "source_of_tolerance", None),
                "detected_by": getattr(field, "provenance", None),
            },
        ))
        if len(facts) >= MAX_FACTS_PER_FRAGMENT:
            break
    return facts


# ===========================================================================
# Точка входа
# ===========================================================================

def extract_facts(fragment: Fragment) -> List[ExtractedFact]:
    """Факты одного фрагмента — по его типу."""
    content = fragment.content
    if fragment.type == "table" and content.table_data is not None:
        return facts_from_table(content.table_data.model_dump(), fragment.fragment_id)
    if fragment.type == "drawing" and content.structured_drawing_fields:
        return facts_from_drawing_fields(
            content.structured_drawing_fields, fragment.fragment_id
        )
    if content.text:
        key = "document.formula" if fragment.type == "formula" else "document.parameter"
        return facts_from_text(content.text, fragment.fragment_id, key=key)
    return []
