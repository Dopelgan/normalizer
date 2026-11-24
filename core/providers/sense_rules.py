"""
Правила здравого смысла для полей чертежа (флаг DRAWING_SENSE_RULES).

Модель читает надпись листа по-разному в разных проходах, а сведение
прочтений (D3) оставляет самое уверенное — даже когда оно бессмысленно для
этого листа. Пример с листа детали: обведённая толщина листа «S0,5*»
прочитана как «50,5*» (уверенность 0.9, со связью «радиус скругления»),
как «S0,5*» (0.4, спорное и спрятанное) и на вырезке вида — как
шероховатость «Ra 0,5*». Штамп при этом говорит «Лист 0,5 ГОСТ …»: толщина
известна, и «50,5*» — это «S0,5*» с буквой S, прочитанной пятёркой.

Правила здесь детерминированные и без вызовов модели; применяются к
фрагментам страницы после сведения прочтений:

1. Толщина листа. Из графы материала берётся толщина t («Лист 0,5 ГОСТ …»,
   «Лента 0,5», «S0,5»). Размер «5<t>*» (только со звёздочкой) и поле с
   записью «5<t>» или цифрами t, среди прочтений которого есть «S<t>»,
   становятся «S<t>»; прежнее прочтение остаётся в `readings`, поле
   помечается на сверку. Повторы «S<t>» в одном месте листа сводятся в одно.
2. Шероховатость со справочной звёздочкой или с прочтением «S<число>»:
   поверх размера или толщины (то же место: IoU ≥ 0,3 или центр меньшей
   рамки в большей) она снимается, а её прочтения уходят в оставшееся поле;
   вдали от размеров становится размером только при прочтении «S…» или
   совпадении с t, иначе остаётся шероховатостью с `needs_review`.
3. Связь размера проверяется по виду величины после количества и скобки:
   «радиус» — у «R…»/«SR…» («Ra», «Rz» — не радиус), «диаметр» — у
   «⌀/Ø/Ф…» или «M…», «отверстие» — у них же и у резьб «G», «Rc», «Tr»,
   «К». Несовпадение — связь снимается, значение остаётся.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from core.models.contract import DrawingField, Fragment
from core.providers import reading_reconcile
from core.providers.reading_reconcile import CONFIDENCE, CONFLICT, CROP, SHEET, SINGLE

logger = logging.getLogger(__name__)

# Источник прочтения, выведенного правилом (в `readings`).
SENSE = "sense_rule"

# Категории, которые правило толщины может переписать. Шероховатость — своим
# правилом (2), остальные категории (штамп, выноски) числами листа не бывают.
_SIZE_LIKE = ("size", "tolerance", "radius")

# Две рамки — одно место, если пересечение к объединению не меньше этого
# или центр одной лежит внутри другой.
OVERLAP_IOU = 0.3

# Толщина: «Лист 0,5 ГОСТ 19904-90», «Лист Б-ПН-0,5×1000×2000 ГОСТ 19903»,
# «Лента 0,5-Н ГОСТ 503». После числа (и, может быть, марки через дефис)
# обязательно «×», «ГОСТ» или «ТУ»: в штампе «Лист 2 Листов 5» и «Лист 1/2» —
# номер листа, а не толщина. Конец записи сразу после числа («Лист 0,5»)
# годится только в графе материала (`_STOCK_AT_END`).
_NUMBER = r"(\d{1,2}(?:[.,]\d{1,2})?)"
_STOCK_HEAD = r"(?<![А-Яа-яЁёA-Za-z])(?:Лист|Лента|Полоса)\s+(?:[A-ZА-ЯЁ]{1,4}-)*" + _NUMBER
_SHEET_STOCK = re.compile(
    _STOCK_HEAD + r"(?=(?:-[A-ZА-ЯЁ]{1,3})?\s*(?:[xх×]|ГОСТ|ТУ))",
    re.IGNORECASE,
)
_STOCK_AT_END = re.compile(_STOCK_HEAD + r"\s*$", re.IGNORECASE)
# «S0,5» в графе материала — толщина, записанная знаком. Только латинская S:
# кириллическая «С 20» — марка или сталь, а не толщина.
_S_MARK = re.compile(r"(?<![A-Za-zА-Яа-яЁё])S\s?" + _NUMBER + r"(?![\d.,/])")
# Прочтение-толщина: «S0,5», «S0,5*» (латинская S или похожая кириллическая С).
_S_READING = re.compile(r"^[SС]\s?(\d+(?:[.,]\d+)?)(\*?)$")
# Префикс шероховатости, который снимается при переводе в размер.
_ROUGHNESS_PREFIX = re.compile(r"^\s*(?:√\s*)?R[az]\s*", re.IGNORECASE)

# Количество или приставка перед величиной: «4×R1», «2 отв. ⌀6»,
# «4 отверстия ⌀6», «2 места ⌀8»; затем, может быть, скобка: «(R5)».
_COUNT = (r"^\s*(?:\d+\s*(?:[xх×*]|отв(?:ерстия|ерстий|ерстие)?\.?|мест[аo]?|паз(?:а|ов)?)?\s*)?"
          r"\(?\s*")
# Радиус: «R5», «SR10», «R 2,5»; «Ra 6,3» и «Rz 20» — шероховатость, не радиус
# (после R обязательна цифра).
_RADIUS_VALUE = re.compile(_COUNT + r"S?R\s?\d")
# Знак диаметра: ⌀, Ø, ∅ и буква «Ф», которой его пишут на скане.
_DIAMETER_SIGN = r"[⌀Øø∅ФфΦ]"
_DIAMETER_VALUE = re.compile(_COUNT + r"(?:" + _DIAMETER_SIGN + r"|[MМ])\s?\d")
# Отверстие: диаметр или резьба — M, G, Rc, Tr, коническая «К».
_HOLE_VALUE = re.compile(_COUNT + r"(?:" + _DIAMETER_SIGN + r"|[MМ]|G|Rc|Tr|К)\s?\d")


# ===========================================================================
# Мелкие чистые функции
# ===========================================================================

def _compact(value: Any) -> str:
    """Запись без пробелов, с десятичной запятой: «50.5 *» -> «50,5*»."""
    return re.sub(r"\s+", "", str(value or "")).replace(".", ",")


def _plausible_thickness(text: str) -> bool:
    try:
        number = float(text.replace(",", "."))
    except ValueError:
        return False
    return 0.05 <= number <= 30.0


def thickness(texts: Iterable[Any]) -> Optional[str]:
    """
    Толщина листа из записей графы материала: «Лист 0,5 ГОСТ 19904-90» ->
    «0,5». Нет толщины или записи расходятся — None: гадать правило не
    должно, а два разных листа на одной странице — не один материал.

    Запись — строка (графа материала) или пара (строка, графа ли это
    материала). «S<t>» и «Лист <t>» в конце записи берутся только из графы
    материала: в прочих графах штампа это номер листа или чужая надпись.
    """
    found: List[str] = []
    for entry in texts:
        text, material = entry if isinstance(entry, tuple) else (entry, True)
        text = str(text or "")
        patterns = (_SHEET_STOCK, _STOCK_AT_END, _S_MARK) if material else (_SHEET_STOCK,)
        for pattern in patterns:
            for match in pattern.finditer(text):
                value = match.group(1).replace(".", ",")
                if _plausible_thickness(value):
                    found.append(value)
    unique = list(dict.fromkeys(found))
    return unique[0] if len(unique) == 1 else None


def s_reading(value: Any, t: Optional[str] = None) -> Optional[Tuple[str, bool]]:
    """
    «S0,5*» -> («0,5», True). С `t` — только толщина `t`. Иначе None.
    """
    match = _S_READING.match(_compact(value))
    if not match:
        return None
    number = match.group(1)
    if t is not None and number != t:
        return None
    return number, bool(match.group(2))


def is_thickness(value: Any, t: Optional[str] = None) -> bool:
    return s_reading(value, t) is not None


def misread_thickness(value: Any, t: str) -> Optional[str]:
    """
    Размер, в котором «S» толщины прочитана пятёркой: «50,5*» при t=0,5 ->
    «S0,5*». Только со справочной звёздочкой: голый размер «50,5» (или «55»
    при t=5) на таком листе правдоподобен, а обведённая толщина почти всегда
    со звёздочкой. Голый «5<t>» переписывается только при прочтении «S<t>»
    среди прочтений поля (`_thickness_rule`).
    """
    compact = _compact(value)
    star = compact.endswith("*")
    body = compact[:-1] if star else compact
    if body != "5" + t or not star:
        return None
    return "S" + t + "*"


def _looks_like_thickness(value: Any, t: str) -> bool:
    """Запись поля — «5<t>» (со звёздочкой или без) или цифры самой толщины."""
    compact = _compact(value).rstrip("*")
    return compact == "5" + t or reading_reconcile.digits(compact) == reading_reconcile.digits(t)


def link_fits(value: Any, label: Any) -> bool:
    """
    Подходит ли вид величины к тому, что она, по связи, измеряет: радиус —
    только у «R…», диаметр и отверстие — только у «⌀…»/«Ø…»/«M…». Прочие
    подписи правило не проверяет.
    """
    text = str(label or "").casefold()
    shown = str(value or "").strip()
    if "радиус" in text:
        return bool(_RADIUS_VALUE.match(shown))
    if "отверст" in text:
        return bool(_HOLE_VALUE.match(shown))
    if "диаметр" in text:
        return bool(_DIAMETER_VALUE.match(shown))
    return True


def _about_thickness(label: Any) -> bool:
    return "толщ" in str(label or "").casefold()


def _area(box: Sequence[float]) -> float:
    return max(0.0, float(box[2]) - float(box[0])) * max(0.0, float(box[3]) - float(box[1]))


def _real_box(box: Optional[Sequence[float]]) -> bool:
    """Рамка надписи, а не «весь лист» по умолчанию и не область."""
    return reading_reconcile._centre(box or []) is not None


def overlaps(a: Optional[Sequence[float]], b: Optional[Sequence[float]], strict: bool = False) -> bool:
    """
    Одно место: пересечение к объединению ≥ OVERLAP_IOU или центр одной
    внутри другой. `strict` — центр только меньшей рамки внутри большей:
    крупная рамка (область вида) своим центром накрывает любую мелкую
    надпись, и снимать по такому совпадению нельзя.
    """
    if not (_real_box(a) and _real_box(b)):
        return False
    ax1, ay1, ax2, ay2 = (float(v) for v in a)
    bx1, by1, bx2, by2 = (float(v) for v in b)
    pairs = (
        (reading_reconcile._centre(a), (bx1, by1, bx2, by2), _area(a) <= _area(b)),
        (reading_reconcile._centre(b), (ax1, ay1, ax2, ay2), _area(b) <= _area(a)),
    )
    for (cx, cy), (x1, y1, x2, y2), smaller in pairs:
        if strict and not smaller:
            continue
        if x1 <= cx <= x2 and y1 <= cy <= y2:
            return True
    inter = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    union = _area(a) + _area(b) - inter
    return union > 0 and inter / union >= OVERLAP_IOU


# ===========================================================================
# Поля страницы
# ===========================================================================

class _Item:
    """Поле страницы с фрагментом и рамкой в общей для сравнения системе."""

    def __init__(self, fragment: Fragment, field: DrawingField, whole: bool):
        self.fragment = fragment
        self.field = field
        self.whole = whole
        self.box = _page_box(fragment, field, whole)
        self.gone = False


def _page_box(fragment: Fragment, field: DrawingField, whole: bool) -> Optional[List[float]]:
    """
    Рамка поля в долях страницы. Лист целиком — как есть; вырезка — переносом
    из долей вырезки (`reading_reconcile._to_page`). При повороте на 90°
    перенос неверен (как и у переноса связи в D3) — места не сравниваются.
    """
    if not _real_box(field.bbox):
        return None
    if whole:
        return [float(v) for v in field.bbox]
    frame = list(fragment.position.bbox or []) if fragment.position else []
    if len(frame) != 4 or reading_reconcile._turn(fragment):
        return None
    return reading_reconcile._to_page(field.bbox, frame)


def _comparable(a: _Item, b: _Item, strict: bool = False) -> bool:
    if a.fragment is b.fragment:
        return overlaps(a.field.bbox, b.field.bbox, strict)
    if reading_reconcile._turn(a.fragment) or reading_reconcile._turn(b.fragment):
        return False
    return overlaps(a.box, b.box, strict)


def _source(item: _Item) -> str:
    return SHEET if item.whole else CROP


def _rewrite(item: _Item, value: str, category: Optional[str] = None) -> None:
    """
    Новое значение поля с прежним прочтением в `readings`: в записи остаётся
    «прочтения расходятся: S0,5* | 50,5*», поле помечено на сверку.
    Уверенность — как у одного прочтения (0.7): значение подтверждено
    штампом, но не человеком; ниже порога просмотра (0.6) оно не прячется.
    """
    from core.providers import vector_drawing

    field = item.field
    readings = list(field.readings or [{"source": _source(item), "value": field.value}])
    readings.append({"source": SENSE, "value": value})
    recategorised = bool(category) and category != field.category
    if category:
        field.category = category
    differs = reading_reconcile.notation(field.value) != reading_reconcile.notation(value)
    agreement = CONFLICT if differs or field.agreement == CONFLICT else (field.agreement or SINGLE)
    changed = value != field.value
    reading_reconcile._set(field, value, readings, agreement, disputed=False)
    if changed or recategorised:
        # Производные от записи (допуск, вид, единица) — от новой записи и
        # категории: «Ra 0,5*» с единицей шероховатости, ставшее размером
        # «S0,5*», единицу шероховатости не сохраняет.
        field.tolerance = vector_drawing.extract_tolerance(value)
        field.nature = vector_drawing.determine_nature(value)
        field.source_of_tolerance = vector_drawing.tolerance_source(value)
        field.unit = vector_drawing.extract_unit(value, field.category)
    field.confidence = max(field.confidence, CONFIDENCE[SINGLE]) if differs else field.confidence
    if differs:
        field.needs_review = True
    if field.link is not None and not _about_thickness(field.link.target_label):
        # «50,5* -> радиус скругления»: связь была у неверного прочтения.
        field.link = None


def _removable(item: _Item) -> bool:
    """
    Снимать поле можно, если фрагмент не останется пустым: пустой чертёжный
    фрагмент без текста и таблиц искался бы по служебным ключам (как в D3).
    """
    content = item.fragment.content
    left = [f for f in content.structured_drawing_fields or [] if f is not item.field]
    return bool(left) or bool((content.text or "").strip()) or bool(
        (content.structured_payload or {}).get("tables"))


def _drop(item: _Item) -> None:
    item.gone = True
    content = item.fragment.content
    content.structured_drawing_fields = [
        f for f in content.structured_drawing_fields or [] if f is not item.field
    ]


def _readings_values(field: DrawingField) -> List[str]:
    return [str(r.get("value") or "") for r in field.readings or []]


def _s_alternative(field: DrawingField, t: Optional[str]) -> Optional[str]:
    """Прочтение-толщина среди прочтений поля: «S0,5*» или None."""
    for value in _readings_values(field):
        found = s_reading(value, t)
        if found is not None:
            return "S" + found[0] + ("*" if found[1] else "")
    return None


def _roughness_rule(items: List[_Item], t: Optional[str], stats: Dict[str, int]) -> None:
    for item in items:
        field = item.field
        if item.gone or field.category != "roughness":
            continue
        alternative = _s_alternative(field, None)
        if "*" not in (field.value or "") and alternative is None:
            continue
        # Справочная звёздочка у шероховатости в ЕСКД не ставится, «S…» —
        # толщина: это размер, прочитанный как шероховатость.
        survivor = next((
            other for other in items
            if other is not item and not other.gone and _comparable(item, other, strict=True)
            and (other.field.category in _SIZE_LIKE or is_thickness(other.field.value))
        ), None)
        if survivor is not None:
            if _removable(item):
                # Прочтения снятой шероховатости — прочтения того же места:
                # они уходят в поле, которое осталось.
                _merge_readings(survivor, item)
                logger.debug("Шероховатость %r снята поверх %r", field.value, survivor.field.value)
                _drop(item)
                stats["roughness_dropped"] += 1
            else:
                # Снять нельзя (фрагмент опустеет), а размер на этом месте
                # уже есть: второй размер был бы повтором.
                field.needs_review = True
                stats["roughness_review"] += 1
            continue
        value = alternative
        if value is None and t is not None:
            stripped = _ROUGHNESS_PREFIX.sub("", field.value or "")
            if reading_reconcile.digits(stripped) == reading_reconcile.digits(t):
                value = "S" + t + ("*" if "*" in stripped else "")
        if value is None:
            # Одна звёздочка без «S» и без толщины — повод для сверки, а не
            # для смены категории: шероховатость остаётся.
            field.needs_review = True
            stats["roughness_review"] += 1
            continue
        _rewrite(item, value, category="size")
        stats["roughness_to_size"] += 1


def _merge_readings(keep: _Item, other: _Item) -> None:
    """Прочтения `other` дописываются в `keep` (без повторов); расхождение — на сверку."""
    merged = list(keep.field.readings or [{"source": _source(keep), "value": keep.field.value}])
    seen = {(r.get("source"), r.get("value")) for r in merged}
    extra = other.field.readings or [{"source": _source(other), "value": other.field.value}]
    for reading in extra:
        pair = (reading.get("source"), reading.get("value"))
        if pair not in seen:
            merged.append(dict(reading))
            seen.add(pair)
    keep.field.readings = merged
    if other.field.needs_review or len({reading_reconcile.notation(v)
                                        for v in _readings_values(keep.field)}) > 1:
        keep.field.needs_review = True


def _thickness_rule(items: List[_Item], t: str, stats: Dict[str, int]) -> None:
    for item in items:
        field = item.field
        if item.gone or field.category not in _SIZE_LIKE or is_thickness(field.value, t):
            continue
        value = misread_thickness(field.value, t)
        if value is None and _looks_like_thickness(field.value, t):
            # Прочтение «S<t>» правит только запись, похожую на толщину:
            # у постороннего размера случайное «S…» среди прочтений — шум.
            alternative = _s_alternative(field, t)
            if alternative is not None:
                star = alternative.endswith("*") or "*" in (field.value or "")
                value = "S" + t + ("*" if star else "")
        if value is None:
            continue
        _rewrite(item, value, category="size")
        stats["thickness"] += 1


def _deduplicate_thickness(items: List[_Item], t: str, stats: Dict[str, int]) -> None:
    """Повторы «S<t>» в одном месте листа: остаётся одно поле со всеми прочтениями."""
    marks = [item for item in items if not item.gone and is_thickness(item.field.value, t)]
    # Сначала лист целиком, затем самое уверенное.
    marks.sort(key=lambda i: (not i.whole, -i.field.confidence))
    for n, keep in enumerate(marks):
        if keep.gone:
            continue
        for other in marks[n + 1:]:
            if other.gone or not _comparable(keep, other) or not _removable(other):
                continue
            _merge_readings(keep, other)
            if "*" in (other.field.value or "") and "*" not in (keep.field.value or ""):
                keep.field.value += "*"
            _drop(other)
            stats["thickness_merged"] += 1


def check_links(fields: Iterable[Any]) -> int:
    """
    Снимает связи, вид которых не подходит величине (правило 3). Поля —
    модели `DrawingField` или словари прохода листа. Возвращает число
    снятых связей.
    """
    removed = 0
    for field in fields:
        if isinstance(field, dict):
            link = field.get("link")
            label = link.get("target_label") if isinstance(link, dict) else getattr(link, "target_label", None)
            if link is not None and not link_fits(field.get("value"), label):
                field.pop("link", None)
                removed += 1
        elif field.link is not None and not link_fits(field.value, field.link.target_label):
            field.link = None
            removed += 1
    return removed


def _material_texts(drawings: Sequence[Fragment]) -> List[str]:
    """Записи графы материала страницы: поле «material» или графа штампа с «Материал»."""
    texts = []
    for fragment in drawings:
        for field in fragment.content.structured_drawing_fields or []:
            value = field.value or ""
            if field.category == "material":
                texts.append((value, True))
            elif field.category == "title_block" and "материал" in value.casefold():
                # Графа «Материал: …» штампа — та же графа материала; прочие
                # графы со словом «материал» — нет.
                label = value.split(":", 1)[0].strip().casefold() if ":" in value else ""
                texts.append((value, label == "материал"))
    return texts


def apply_page(drawings: List[Fragment]) -> Dict[str, int]:
    """Правила для чертёжных фрагментов одной страницы; правит их на месте."""
    stats = {"thickness": 0, "thickness_merged": 0, "roughness_dropped": 0,
             "roughness_to_size": 0, "roughness_review": 0, "links_dropped": 0}
    t = thickness(_material_texts(drawings))
    items = [
        _Item(fragment, field, reading_reconcile._is_whole_sheet(fragment))
        for fragment in drawings
        for field in list(fragment.content.structured_drawing_fields or [])
    ]
    _roughness_rule(items, t, stats)
    if t is not None:
        _thickness_rule(items, t, stats)
        _deduplicate_thickness(items, t, stats)
    for fragment in drawings:
        stats["links_dropped"] += check_links(fragment.content.structured_drawing_fields or [])
    return stats


def apply(fragments: List[Fragment]) -> Dict[str, int]:
    """Правила по страницам документа; фрагменты правятся на месте."""
    pages: Dict[Optional[int], List[Fragment]] = {}
    for fragment in fragments:
        if fragment.type == "drawing" and fragment.content.structured_drawing_fields:
            pages.setdefault(reading_reconcile._page(fragment), []).append(fragment)
    total: Dict[str, int] = {}
    for drawings in pages.values():
        for key, value in apply_page(drawings).items():
            total[key] = total.get(key, 0) + value
    return total


# ===========================================================================
# Общая шероховатость в правом верхнем углу (DRAWING_CORNER_ROUGHNESS)
# ===========================================================================

# Угол рамки, где по ЕСКД (ГОСТ 2.309) стоит знак общей шероховатости: доли
# рамки формата (или листа, если рамка не найдена).
CORNER = (0.70, 0.00, 1.00, 0.12)
CORNER_LABEL = "Шероховатость (общая)"
_ROUGHNESS_VALUE = re.compile(r"^(R[az])\s*(\d+(?:[.,]\d+)?)$", re.IGNORECASE)


def corner_box(frame: Optional[Sequence[float]]) -> List[float]:
    """Рамка угла в долях листа: от рамки формата, если она найдена."""
    fx1, fy1, fx2, fy2 = (float(v) for v in (frame if frame and len(frame) == 4 else (0, 0, 1, 1)))
    w, h = fx2 - fx1, fy2 - fy1
    return [round(fx1 + CORNER[0] * w, 4), round(fy1 + CORNER[1] * h, 4),
            round(fx1 + CORNER[2] * w, 4), round(fy1 + CORNER[3] * h, 4)]


def corner_roughness(data: Any) -> Optional[str]:
    """
    Ответ модели про знак в углу -> «Ra 6,3 (√)» / «Ra 6,3» или None.
    Пустой, неуверенный или не похожий на шероховатость ответ — None:
    додумывать знак нельзя.
    """
    if not isinstance(data, dict) or data.get("unsure"):
        return None
    raw = re.sub(r"\s+", " ", str(data.get("value") or "")).strip().lstrip("√").strip()
    match = _ROUGHNESS_VALUE.match(raw)
    if not match:
        return None
    letters = "R" + match.group(1)[1].lower()
    value = f"{letters} {match.group(2).replace('.', ',')}"
    return value + (" (√)" if data.get("check_mark") is True else "")


def roughness_known(value: str, fields: Iterable[Dict[str, Any]]) -> bool:
    """Та же шероховатость уже прочитана на листе (без знака «(√)»)."""
    wanted = reading_reconcile.notation(value.replace("(√)", ""))
    for item in fields:
        if item.get("category") != "roughness":
            continue
        own = str(item.get("value") or "").replace("(√)", "").replace("√", "")
        if reading_reconcile.notation(own) == wanted:
            return True
    return False
