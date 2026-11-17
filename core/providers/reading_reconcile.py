"""
Сведение прочтений полей чертежа (карточка D3, DRAWING_READINGS_RECONCILED).

Одна надпись листа читается несколькими независимыми проходами: проход
«лист целиком» (поля фрагмента листа), вырезки видов MinerU (поля своих
фрагментов той же страницы), проход «весь текст листа» (строки фрагмента
листа), слой распознавания страницы (текстовые фрагменты) и, у векторного
PDF, текстовый слой. Прежде каждое прочтение становилось своим полем, и в
записи стояли две противоречащие строки — «⌀6,6+0,05» с листа и
«⌀6,6-0,05» с вырезки — без пометки, какая из них спорная.

Здесь для полей размеров, допусков, резьб, радиусов и шероховатости:

1. Прочтения одной надписи находятся по числовому скелету (цифры), виду
   величины (⌀, R, M, звёздочка справочного размера) и месту на листе.
2. Поле фрагмента листа получает `readings` — все прочтения с источником —
   и `agreement`: «vector» — значение из текстового слоя; «agreed» —
   прочтение поддержано двумя источниками и сильнее остальных; «single» —
   прочтение одно; «conflict» — прочтения расходятся и большинства нет.
3. Значение: текстовый слой; иначе прочтение с наибольшим числом
   источников; при равенстве — правдоподобное по записи ЕСКД (звёздочка
   внутри числа и «№» в размере — следы неверного чтения), затем — прочтение
   прохода «лист целиком».
4. Уверенность — по согласию; подтверждённым поле не становится никогда:
   статус «подтверждено» ставит только человек.
5. Поле вырезки, сведённое с полем листа, уходит из вырезки: величина
   остаётся одна, со своими прочтениями.

Разница одной записи — минус U+2212 против дефиса, запятая против точки,
пробелы, «x» против «×», латиница против похожей кириллицы — расхождением
не считается.

Вызовов модели нет: всё, что сводится, уже прочитано.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from core.models.contract import DrawingField, Fragment

NUMERIC = ("size", "tolerance", "thread", "radius", "roughness")

# Источники прочтения.
SHEET = "sheet"            # проход «лист целиком» или поле фрагмента листа
CROP = "crop"              # вырезка вида MinerU той же страницы
TRANSCRIPT = "transcript"  # проход «весь текст листа»
OCR = "ocr"                # слой распознавания страницы
VECTOR = "vector"          # текстовый слой векторного PDF

AGREED, SINGLE, CONFLICT = "agreed", "single", "conflict"

# Уверенность по согласию. Текстовый слой — 0.94, ниже порога подтверждения
# 0.95 (core/drawing_quality.py): подтверждает только человек.
CONFIDENCE = {VECTOR: 0.94, AGREED: 0.9, SINGLE: 0.7, CONFLICT: 0.4}

# Два прочтения — одна надпись, если центры рамок ближе этой доли листа.
NEAR = 0.05

_LOOKALIKE = str.maketrans({
    "А": "A", "В": "B", "Е": "E", "К": "K", "М": "M", "Н": "H", "О": "O", "Р": "P",
    "С": "C", "Т": "T", "Х": "X", "У": "Y", "а": "a", "е": "e", "о": "o", "р": "p",
    "с": "c", "х": "x", "у": "y", "к": "k", "м": "m", "т": "t", "н": "h", "в": "b",
    "−": "-", "–": "-", "—": "-", "‐": "-", "‑": "-",
    "Ø": "⌀", "ø": "⌀", "∅": "⌀", "×": "x", " ": " ",
})
_DECIMAL_COMMA = re.compile(r"(?<=\d),(?=\d)")
_SPACE = re.compile(r"\s+")
_NOT_DIGIT = re.compile(r"\D")
# Число с тем, что при нём пишется: знак диаметра, радиуса, резьбы, «x»,
# отклонения, справочная звёздочка, градусы и минуты.
_EXPRESSION = re.compile(
    r"(?:[⌀ØRMМ]\s?)?\d+(?:[.,]\d+)?"
    r"(?:\s?[°'′\"]\d*)*"
    r"(?:\s?[xх×*]\s?\d+(?:[.,]\d+)?)?"
    r"(?:\s?[±+\-−]\s?\d+(?:[.,]\d+)?(?:\s?[°'′])?)*"
    r"(?:\*)?"
)
# Звёздочка между цифрами и «№» в размере в записи ЕСКД не встречаются: это
# прочитанные «+» и «x». Отклонение без знака после отклонения со знаком
# («⌀18+0,25 0,10») — потерянный знак второго отклонения.
_IMPLAUSIBLE = re.compile(r"\d\*\d|[№Nº]|[±+\-−]\s?\d+[.,]\d+\s+\d+[.,]\d+")


def notation(value: Any) -> str:
    """Запись значения без различий записи: по ней два прочтения равны."""
    text = str(value or "").translate(_LOOKALIKE).casefold().translate(_LOOKALIKE)
    text = _DECIMAL_COMMA.sub(".", text)
    return _SPACE.sub("", text)


def digits(value: Any) -> str:
    """Числовой скелет: цифры подряд. «15*» и «1,5*», «70*2» и «70+2» — одна надпись."""
    return _NOT_DIGIT.sub("", str(value or ""))


def kind(value: Any) -> Tuple[str, bool]:
    """Вид величины: знак впереди (⌀, R, M или ничего) и справочная звёздочка в конце."""
    text = notation(value)
    lead = text[:1] if text[:1] in ("⌀", "r", "m") else ""
    return lead, text.endswith("*")


def groups(value: Any) -> int:
    """
    Сколько чисел в записи: «M1, M2» — два, «M1.2» — одно, «134,9» и «1349» —
    по одному. Разное число чисел — разные надписи, даже при общих цифрах.
    """
    return len(_GROUP.findall(str(value or "")))


_GROUP = re.compile(r"\d+(?:[.,]\d+)?")


def core(value: Any) -> str:
    """
    Числовое выражение внутри значения поля — та его часть, которую видно и
    в тексте без мест: «M18x1,5-8g» -> «M18x1,5-8», «12 пазов» -> «12».
    """
    found = [m.group(0).strip() for m in _EXPRESSION.finditer(str(value or ""))]
    found = [text for text in found if digits(text)]
    return max(found, key=len) if found else ""


def same_inscription(a: Any, b: Any) -> bool:
    """
    Два прочтения могут быть одной надписью: тот же вид и число чисел, и те
    же цифры — или то же главное число, а расходятся отклонения («10°±2°» и
    «10°±1°»: модель путает мелкие цифры отклонения, а не сам размер).
    """
    if kind(a) != kind(b) or groups(a) != groups(b):
        return False
    if digits(a) == digits(b):
        return True
    first_a, first_b = _GROUP.search(str(a or "")), _GROUP.search(str(b or ""))
    return bool(first_a and first_b and groups(a) > 1
                and first_a.group(0).replace(",", ".") == first_b.group(0).replace(",", "."))


def plausible(value: Any) -> bool:
    return not _IMPLAUSIBLE.search(str(value or ""))


def _centre(bbox: Sequence[float]) -> Optional[Tuple[float, float]]:
    if not bbox or len(bbox) != 4:
        return None
    x1, y1, x2, y2 = (float(v) for v in bbox)
    if [x1, y1, x2, y2] == [0.0, 0.0, 1.0, 1.0] or x2 - x1 > 0.5 or y2 - y1 > 0.5:
        return None     # рамка по умолчанию или область — не место надписи
    return (x1 + x2) / 2, (y1 + y2) / 2


def _to_page(bbox: Sequence[float], frame: Sequence[float]) -> List[float]:
    """Рамка в долях вырезки -> в долях страницы."""
    fx1, fy1, fx2, fy2 = (float(v) for v in frame)
    w, h = fx2 - fx1, fy2 - fy1
    x1, y1, x2, y2 = (float(v) for v in bbox)
    return [fx1 + x1 * w, fy1 + y1 * h, fx1 + x2 * w, fy1 + y2 * h]


def expressions(lines: Iterable[str]) -> List[str]:
    """Числовые выражения строк текста: кандидаты в прочтения величин."""
    found: List[str] = []
    for line in lines:
        for match in _EXPRESSION.finditer(str(line or "")):
            text = match.group(0).strip()
            if digits(text):
                found.append(text)
    return found


def _text_reading(field_value: str, candidates: Sequence[str]) -> Optional[str]:
    """
    Прочтение той же надписи в тексте без мест (строки листа, слой
    распознавания). Сравнивается числовое выражение поля (`core`): буквы
    вокруг него текст отдаёт отдельными словами. Совпало с полем — прочтение
    и есть значение поля; иначе — значение поля с выражением из текста, и
    только если вариант в тексте один: из двух разных неизвестно, какой из
    них эта надпись.
    """
    own = core(field_value)
    if len(digits(own)) < 2 or ", " in field_value or ";" in field_value:
        return None     # одна цифра есть на листе везде; список значений — не одна надпись
    same = [c for c in candidates if same_inscription(c, own)]
    keys = {notation(c) for c in same}
    if notation(own) in keys:
        return field_value
    if len(keys) == 1:
        return field_value.replace(own, same[0].strip(), 1)
    return None


def choose(readings: Sequence[Dict[str, str]], primary: str) -> Tuple[str, str]:
    """
    Значение и согласие по прочтениям. `primary` — источник, чьё прочтение
    берётся при равенстве (проход «лист целиком»).
    """
    vector = [r for r in readings if r["source"] == VECTOR]
    if vector:
        return vector[0]["value"], VECTOR
    support: Dict[str, set] = {}
    first: Dict[str, str] = {}
    for reading in readings:
        key = notation(reading["value"])
        support.setdefault(key, set()).add(reading["source"])
        first.setdefault(key, reading["value"])
    if len(support) == 1:
        key = next(iter(support))
        return first[key], (AGREED if len(support[key]) >= 2 else SINGLE)
    primary_keys = [notation(r["value"]) for r in readings if r["source"] == primary]

    def rank(key: str) -> tuple:
        return (len(support[key]), plausible(first[key]), key in primary_keys)

    # Текст без мест (строки листа, слой распознавания) сам значение не
    # меняет: «15» в тексте может быть другой надписью, чем «1,5» на листе.
    # Чужое прочтение побеждает, только если его видел проход с местом.
    placed = {SHEET, CROP, VECTOR}
    contenders = [key for key in support if support[key] & placed]
    ordered = sorted(contenders, key=rank, reverse=True)
    best = ordered[0]
    runner = max(len(support[key]) for key in support if key != best)
    if len(support[best]) > runner and len(support[best]) >= 2:
        return first[best], AGREED
    return first[best], CONFLICT


def _is_whole_sheet(fragment: Fragment) -> bool:
    bbox = list(fragment.position.bbox or []) if fragment.position else []
    return bbox == [0.0, 0.0, 1.0, 1.0] or bool(
        (fragment.content.structured_payload or {}).get("transcript_line_count")
    )


def _page(fragment: Fragment) -> Optional[int]:
    if fragment.position is None or fragment.position.page is None:
        return None
    return int(fragment.position.page)


# Ссылка на пункт технических требований «п.10», прочитанная латинской «n»:
# в русском листе «n.10» не пишется, а поиск по «п.10» её не находит. На
# пилоте 17.11 так «пропали» девять ссылок одного листа.
_ITEM_REF = re.compile(r"(?<![A-Za-zА-Яа-яЁё0-9])n\.\s?(?=\d)")


def item_refs(text: str) -> str:
    """«n.10» -> «п.10»; прочее не трогается."""
    return _ITEM_REF.sub(lambda m: "п" + m.group(0)[1:], text) if text and "n." in text else text


def apply(fragments: List[Fragment]) -> List[Fragment]:
    """Сводит прочтения полей чертёжных фрагментов по страницам; список тот же."""
    for fragment in fragments:
        if fragment.type != "drawing":
            continue
        if fragment.content.text:
            fragment.content.text = item_refs(fragment.content.text)
        for field in fragment.content.structured_drawing_fields or []:
            field.value = item_refs(field.value)
    pages: Dict[int, List[Fragment]] = {}
    texts: Dict[int, List[str]] = {}
    for fragment in fragments:
        page = _page(fragment)
        if page is None:
            continue
        if fragment.type == "drawing":
            pages.setdefault(page, []).append(fragment)
        elif fragment.type in ("text", "table") and fragment.content.text:
            texts.setdefault(page, []).extend(fragment.content.text.split("\n"))
    for page, drawings in pages.items():
        reconcile_page(drawings, texts.get(page, []))
    return fragments


def reconcile_page(drawings: List[Fragment], layer_lines: Sequence[str]) -> None:
    """Поля чертёжных фрагментов одной страницы."""
    vector = [f for f in drawings if f.provenance.method == "vector_text_layer"]
    vector_ids = {id(f) for f in vector}
    model = [f for f in drawings if id(f) not in vector_ids]
    if not model:
        return
    whole = [f for f in model if _is_whole_sheet(f)]
    if not whole:
        return      # без листа целиком рамки вырезок сравнивать не с чем
    primary = max(whole, key=lambda f: len(f.content.structured_drawing_fields or []))
    crops = [f for f in model if f is not primary]
    transcript = expressions((primary.content.text or "").split("\n"))
    layer = expressions(layer_lines)
    vector_fields = [field for f in vector for field in (f.content.structured_drawing_fields or [])]

    # Поля вырезок в координатах страницы: (фрагмент, поле, рамка на странице).
    crop_fields = []
    for crop in crops:
        frame = list(crop.position.bbox or []) if crop.position else []
        for field in crop.content.structured_drawing_fields or []:
            placed = _to_page(field.bbox, frame) if len(frame) == 4 and _centre(field.bbox) else []
            crop_fields.append((crop, field, placed))
    taken: set = set()
    twins: Dict[int, DrawingField] = {}

    for field in primary.content.structured_drawing_fields or []:
        numeric = field.category in NUMERIC
        readings = [{"source": SHEET, "value": field.value}]
        # Вырезка: ближайшее поле той же надписи. Величина — по цифрам, виду
        # и числу чисел (запись могла прочитаться по-разному); прочая надпись
        # — той же категории и той же записи.
        twin = _nearest(field, crop_fields, taken, numeric)
        if twin is not None:
            taken.add(twin)
            twins[id(crop_fields[twin][1])] = field
            readings.append({"source": CROP, "value": crop_fields[twin][1].value})
        if not numeric:
            if twin is not None:
                _set(field, field.value, readings, AGREED)
            continue
        for source, candidates in ((TRANSCRIPT, transcript), (OCR, layer)):
            value = _text_reading(field.value, candidates)
            if value is not None:
                readings.append({"source": source, "value": value})
        layer_twin = _vector_twin(field, vector_fields)
        if layer_twin is not None:
            readings.append({"source": VECTOR, "value": layer_twin.value})
        value, agreement = choose(readings, SHEET)
        _set(field, value, readings, agreement)

    # Сведённые поля уходят из вырезок; несведённые остаются там, где были,
    # с одним прочтением.
    gone = {id(crop_fields[index][1]) for index in taken}
    for crop in crops:
        own = crop.content.structured_drawing_fields or []
        if (all(id(field) in gone for field in own) and not (crop.content.text or "").strip()
                and not (crop.content.structured_payload or {}).get("tables")):
            # Вырезка, все надписи которой есть на листе, без текста и таблиц:
            # пустой чертёжный фрагмент искался бы по служебным ключам payload.
            # Её поля остаются на месте — повтор лучше пустого фрагмента.
            gone -= {id(field) for field in own}
        kept = []
        for field in own:
            if id(field) in gone:
                continue
            twin_field = twins.get(id(field))
            if twin_field is not None:
                # Оставленный повтор говорит то же, что поле листа.
                _set(field, twin_field.value, list(twin_field.readings or []),
                     twin_field.agreement or SINGLE)
            if field.category in NUMERIC and field.agreement is None:
                _set(field, field.value, [{"source": CROP, "value": field.value}], SINGLE)
            kept.append(field)
        crop.content.structured_drawing_fields = kept


def _numbers(value: Any) -> List[str]:
    """Числа записи с десятичной точкой: «1,5» -> ["1.5"], «15» -> ["15"]."""
    return [number.replace(",", ".") for number in _GROUP.findall(str(value or ""))]


def _vector_twin(field: DrawingField, vector_fields: Sequence[DrawingField]) -> Optional[DrawingField]:
    """
    Поле текстового слоя той же надписи: те же числа с той же десятичной
    записью («1,5» не «15»), тот же вид, ближайшее к полю и не дальше
    размера его рамки (не больше NEAR). Текстовый слой побеждает в выборе
    значения, поэтому совпадения только по цифрам здесь мало.
    """
    here = _centre(field.bbox)
    if here is None:
        return None
    x1, y1, x2, y2 = (float(v) for v in field.bbox)
    radius = min(NEAR, max(x2 - x1, y2 - y1))
    numbers, shape = _numbers(field.value), kind(field.value)
    best, best_distance = None, None
    for other in vector_fields:
        there = _centre(other.bbox)
        if there is None or kind(other.value) != shape or _numbers(other.value) != numbers:
            continue
        distance = math.dist(here, there)
        if distance <= radius and (best_distance is None or distance < best_distance):
            best, best_distance = other, distance
    return best


def _nearest(field: DrawingField, crop_fields: Sequence[tuple], taken: set,
             numeric: bool) -> Optional[int]:
    """Номер ближайшего поля вырезки, прочитавшего ту же надпись, или None."""
    here = _centre(field.bbox)
    if here is None:
        return None
    own = notation(field.value)
    best, best_distance = None, None
    for index, (_, other, placed) in enumerate(crop_fields):
        if index in taken:
            continue
        if numeric:
            if other.category not in NUMERIC or not same_inscription(other.value, field.value):
                continue
        elif other.category != field.category or notation(other.value) != own:
            continue
        there = _centre(placed) if placed else None
        if there is None:
            continue
        distance = math.dist(here, there)
        if distance <= NEAR and (best_distance is None or distance < best_distance):
            best, best_distance = index, distance
    return best


def _set(field: DrawingField, value: str, readings: List[Dict[str, str]], agreement: str) -> None:
    from core.providers import vector_drawing

    if value != field.value:
        field.value = value
        field.tolerance = vector_drawing.extract_tolerance(value)
        field.nature = vector_drawing.determine_nature(value)
        field.source_of_tolerance = vector_drawing.tolerance_source(value)
    field.readings = readings
    field.agreement = agreement
    # Уверенность по согласию; спор категории (модель против правил,
    # `drawing_processor._reconcile`: 0.6 и ниже) понижает её, как и прежде, —
    # в той же доле 0.6 / 0.9.
    disputed = field.confidence <= _CATEGORY_DISPUTED
    field.confidence = round(CONFIDENCE[agreement] * (2 / 3 if disputed else 1), 3)


# Уверенность поля, у которого модель и правила разошлись в категории
# (`drawing_processor._DISAGREED`).
_CATEGORY_DISPUTED = 0.6


def conflict_note(field: Dict[str, Any]) -> str:
    """Хвост записи у спорного поля: «(прочтения расходятся: ±0,3 | +0,3)»."""
    if field.get("agreement") != CONFLICT:
        return ""
    seen: List[str] = []
    for reading in field.get("readings") or []:
        value = str(reading.get("value") or "")
        if value and notation(value) not in {notation(v) for v in seen}:
            seen.append(value)
    if len(seen) < 2:
        return ""
    return " (прочтения расходятся: " + " | ".join(seen) + ")"


def share(fragments: Iterable[Fragment]) -> Counter:
    """Счёт полей по согласию — для отчёта замера."""
    counts: Counter = Counter()
    for fragment in fragments:
        for field in fragment.content.structured_drawing_fields or []:
            if field.agreement:
                counts[field.agreement] += 1
    return counts
