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

Под своими флагами (по умолчанию выключены) — классы полей (C11), позиции
по спецификации сборки (C12), ряд шероховатости ГОСТ 2789 (S1) и общий знак
шероховатости в углу листа (C14); см. раздел в конце модуля.
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
# «S⌀20» — диаметр сферы.
_DIAMETER_VALUE = re.compile(_COUNT + r"S?(?:" + _DIAMETER_SIGN + r"|[MМ])\s?\d")
# Отверстие: диаметр или резьба — M, G, Rc, Tr, коническая «К».
_HOLE_VALUE = re.compile(_COUNT + r"(?:S?" + _DIAMETER_SIGN + r"|[MМ]|G|Rc|Tr|К)\s?\d")


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


def _thickness_rule(items: List[_Item], t: str, stats: Dict[str, int], need_s: bool = False) -> None:
    """
    `need_s` — толщина взята не из графы материала, а из текста листа или
    примечаний (DRAWING_THICKNESS_EVIDENCE): там «Лист 2 ГОСТ 19903-74» может
    быть строкой спецификации, и настоящий размер «52*» стал бы «S2*». Тогда
    «5<t>*» переписывается только при прочтении «S<t>» того же места.
    """
    for item in items:
        field = item.field
        if item.gone or field.category not in _SIZE_LIKE or is_thickness(field.value, t):
            continue
        value = misread_thickness(field.value, t)
        if value is not None and need_s and _s_alternative(field, t) is None:
            value = None
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


def _page_texts(fragment: Fragment, evidence: bool = False) -> List[str]:
    """
    Текст страницы для толщины: весь текст листа и текст прочих фрагментов.

    `build_fragment` убирает `transcript_lines` из payload и кладёт весь
    текст листа в `content.text` чертёжного фрагмента: к правилам строки
    листа приходят только оттуда. С `evidence` (DRAWING_THICKNESS_EVIDENCE)
    текст чертёжного фрагмента берётся построчно; без него — как раньше.
    """
    content = fragment.content
    if fragment.type == "drawing":
        lines = [str(line) for line in (content.structured_payload or {}).get("transcript_lines") or [] if line]
        if evidence and content.text:
            lines.extend(line for line in content.text.splitlines() if line.strip())
        return lines
    return [content.text] if content.text else []


def _notes(drawings: Sequence[Fragment]) -> List[str]:
    return [field.value or "" for fragment in drawings
            for field in fragment.content.structured_drawing_fields or [] if field.category == "note"]


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


# Листовой прокат без толщины в графе материала или в примечании:
# «Лист Б-ПН-НО ГОСТ 19903-74», «Лента ГОСТ 503-81». Слово проката
# и ГОСТ/ТУ в той же строке (регистр любой): «листа» и «лист 2» без ГОСТ —
# не материал.
_SHEET_METAL = re.compile(r"(?<![А-Яа-яЁёA-Za-z])(?:Лист|Лента|Полоса)(?![А-Яа-яЁё])[^\n]{0,60}?(?:ГОСТ|ТУ)",
                          re.IGNORECASE)


def _sheet_metal(entries: Iterable[Tuple[str, bool]], notes: Iterable[str]) -> bool:
    """Графа материала (или примечание) называет листовой прокат."""
    texts = [text for text, material in entries if material] + list(notes)
    return any(_SHEET_METAL.search(str(text or "")) for text in texts)


# Метка сварного шва ГОСТ 2.312 «2№3» (два места, шов №3), прочитанная
# резьбой: «2M3», «2M», «2N*1». Резьба так не пишется: количество отверстий
# стоит отдельно («2 отв. M3»), а знак «№» модель отдаёт буквой M или N.
_WELD_AS_THREAD = re.compile(r"^\d+[MМNН№]\*?\d{0,2}$")
# Позиционное обозначение элемента на схеме ГОСТ 2.710: «M1» — двигатель,
# «M11», «M1(2)», «M12-15» (элементы с 12 по 15). У резьбы на схеме места нет:
# схема изделия не чертёж, а «M12-6g» (с полем допуска) остаётся резьбой.
_SCHEME_DESIGNATOR = re.compile(r"^[MМ]\d{1,3}(?:\(\d+\)|-\d{1,3})?$")


# Метка шва ГОСТ 2.312: «№1», «2№3», «4N°1» (знак номера — и «N°», «Nº»).
_WELD_MARK = re.compile(r"^\d*\s?(?:№|N[°ºo]?)\s?\d{1,2}$")
# Номер позиции на полке выноски сборочного чертежа (ГОСТ 2.109): голое
# число без единиц и знаков.
_POSITION_NUMBER = re.compile(r"^\d{1,3}$")


def _position_rule(items: List[_Item], stats: Dict[str, int]) -> None:
    """
    Позиции и метки швов на сборочном чертеже (DRAWING_SENSE_POSITIONS).
    Демо-проверка 24.11, пробка ЭК150: номера позиций 2–7 и 11 пришли
    выносками, а метки швов «№1»…«2№3» — позициями. На листе сборки голое
    число на выноске — номер позиции, а «№» с цифрой — метка шва (выноска).
    Только `sheet_type` = `assembly`: на листе детали голое число на выноске
    — ссылка на пункт или размер.
    """
    for item in items:
        field = item.field
        if item.gone or field.category not in ("position", "callout"):
            continue
        payload = item.fragment.content.structured_payload or {}
        if payload.get("sheet_type") != "assembly":
            continue
        value = " ".join(str(field.value or "").split())
        if field.category == "position" and _WELD_MARK.match(value):
            field.category = "callout"
            stats["weld_marks"] += 1
        elif field.category == "callout" and _POSITION_NUMBER.match(value) and int(value) > 0:
            field.category = "position"
            stats["positions"] += 1


def _thread_rule(items: List[_Item], stats: Dict[str, int]) -> None:
    """
    Резьба, которой на листе нет: метка шва «2№3» на сборке пробки ЭК150 и
    обозначения элементов «M1», «M11» на электрических схемах Э4 приходили
    резьбой и попадали в индекс как «резьбы листа». Значение не меняется,
    меняется только род: метка шва — выноска, обозначение — позиция.
    """
    for item in items:
        field = item.field
        if item.gone or field.category != "thread":
            continue
        value = " ".join(str(field.value or "").split())
        payload = item.fragment.content.structured_payload or {}
        if _WELD_AS_THREAD.match(value):
            field.category = "callout"
            field.needs_review = True
            field.link = None
            stats["thread_weld"] += 1
        elif payload.get("sheet_type") == "scheme" and _SCHEME_DESIGNATOR.match(value):
            field.category = "position"
            field.link = None
            stats["thread_designator"] += 1


# Шероховатость или толщина, прочитанная поверх размера того же места:
# приставка «S», «Ra», «Rz» или «5» на месте «S» снимается, цифры сравниваются.
_TWIN_PREFIX = re.compile(r"^(?:√)?(?:R[az]|[SС]|5)", re.IGNORECASE)
# Примечание «* Размеры для справок»: звёздочка у размера — справочная.
_REFERENCE_NOTE = re.compile(r"размер\w*\s+для\s+справ", re.IGNORECASE)


def _twin_key(value: Any) -> str:
    return reading_reconcile.digits(_TWIN_PREFIX.sub("", _compact(value).rstrip("*")))


def _twin_rank(field: DrawingField) -> int:
    """Кто остаётся из двойников одного места: толщина, размер, шероховатость."""
    if is_thickness(field.value):
        return 0
    return 1 if field.category in _SIZE_LIKE else 2


def _twin_rule(items: List[_Item], stats: Dict[str, int]) -> None:
    """
    Двойники одного места (рамки с IoU от `OVERLAP_IOU`): «S0,5*», «50,5*» и
    «Ra 0,5*» — одно прочтение толщины тремя способами. Сводятся в одно
    поле, только если цифры без приставки совпадают и хотя бы одно из них
    — толщина или шероховатость: два размера на одном месте этим правилом
    не трогаются. Остаётся толщина, затем размер; прочтения сводятся.
    """
    twins = [item for item in items if not item.gone
             and item.field.category in _SIZE_LIKE + ("roughness",)]
    twins.sort(key=lambda i: (_twin_rank(i.field), not i.whole, -i.field.confidence))
    for n, keep in enumerate(twins):
        if keep.gone:
            continue
        key = _twin_key(keep.field.value)
        if not key:
            continue
        for other in twins[n + 1:]:
            if other.gone or _twin_key(other.field.value) != key:
                continue
            if _twin_rank(keep.field) == _twin_rank(other.field) == 1:
                continue
            if not _comparable(keep, other, strict=True) or not _removable(other):
                continue
            _merge_readings(keep, other)
            _drop(other)
            stats["twins_merged"] += 1


def _reference_thickness_rule(items: List[_Item], notes: str, stats: Dict[str, int],
                              sheet_metal: bool = False) -> None:
    """
    Толщина без графы материала: размер «5<x>*», среди прочтений которого
    есть «S<x>», на листе с примечанием «* Размеры для справок» становится
    «S<x>*». Звёздочка справочная, и прочтение «S» того же места — два
    свидетельства вместо графы материала. `sheet_metal`
    (DRAWING_THICKNESS_EVIDENCE) — второе свидетельство вместо примечания:
    графа материала называет листовой прокат, но без толщины.
    """
    if not (_REFERENCE_NOTE.search(notes) or sheet_metal):
        return
    for item in items:
        field = item.field
        if item.gone or field.category not in _SIZE_LIKE or is_thickness(field.value):
            continue
        compact = _compact(field.value)
        if not compact.endswith("*") or not compact.startswith("5"):
            continue
        alternative = _s_alternative(field, compact[1:-1])
        if alternative is None:
            continue
        _rewrite(item, "S" + compact[1:-1] + "*", category="size")
        stats["thickness"] += 1


def apply_page(drawings: List[Fragment], texts: Sequence[str] = (),
               positions: Optional[set] = None) -> Dict[str, int]:
    """
    Правила для чертёжных фрагментов одной страницы; правит их на месте.

    `texts` — прочий текст той же страницы (весь текст листа, строки OCR):
    толщина из него берётся только по строгому образцу «Лист 0,5 ГОСТ …»
    и только если графы материала нет. `positions` — номера графы «Поз.»
    спецификации документа (DRAWING_SENSE_SPEC_POSITIONS).
    """
    stats = {"thickness": 0, "thickness_merged": 0, "roughness_dropped": 0,
             "roughness_to_size": 0, "roughness_review": 0, "links_dropped": 0,
             "thread_weld": 0, "thread_designator": 0, "twins_merged": 0, "weld_marks": 0, "positions": 0}
    from core.config import settings

    evidence = settings.DRAWING_THICKNESS_EVIDENCE
    material = _material_texts(drawings)
    t = thickness(material)
    need_s = False
    # На сборке «Лист 2 ГОСТ 19903-74» в тексте листа — строка спецификации,
    # а не толщина листа.
    assembly = any((f.content.structured_payload or {}).get("sheet_type") == "assembly" for f in drawings)
    if t is None and evidence and not assembly:
        # Примечание «Материал заготовки: Лист 0,5 ГОСТ 19904-90» — та же
        # толщина; только строгий образец: «лист 2» в примечании — ссылка.
        t = thickness([(note, False) for note in _notes(drawings)])
        need_s = t is not None
    if t is None and evidence and not assembly:
        t = thickness([(text, False) for text in texts])
        need_s = t is not None
    elif t is None and texts and not evidence:
        t = thickness([(text, False) for text in texts])
    items = [
        _Item(fragment, field, reading_reconcile._is_whole_sheet(fragment))
        for fragment in drawings
        for field in list(fragment.content.structured_drawing_fields or [])
    ]
    # Толщина из текста или примечаний шероховатость в размер не переводит:
    # без прочтения «S» это могла быть строка спецификации.
    _roughness_rule(items, None if need_s else t, stats)
    _thread_rule(items, stats)
    if settings.DRAWING_SENSE_POSITIONS:
        _position_rule(items, stats)
    if t is not None:
        _thickness_rule(items, t, stats, need_s)
        _deduplicate_thickness(items, t, stats)
    else:
        _reference_thickness_rule(items, "\n".join(_notes(drawings)) + "\n" + "\n".join(texts), stats,
                                  sheet_metal=evidence and _sheet_metal(material, _notes(drawings)))
    _twin_rule(items, stats)
    if settings.DRAWING_SENSE_CLASS_RULES:
        _class_rule(items, stats)
    if settings.DRAWING_SENSE_SPEC_POSITIONS and positions:
        _spec_position_rule(items, positions, stats)
    if settings.DRAWING_SENSE_RA_SERIES:
        _ra_series_rule(items, stats)
    if settings.DRAWING_CORNER_ROUGHNESS_V2:
        # После ряда: исправленная запятая («Ra 08» -> «Ra 0,8») уже стоит.
        _corner_rule(drawings, items, stats)
    for fragment in drawings:
        stats["links_dropped"] += check_links(fragment.content.structured_drawing_fields or [])
    return stats


def apply(fragments: List[Fragment]) -> Dict[str, int]:
    """Правила по страницам документа; фрагменты правятся на месте."""
    from core.config import settings

    evidence = settings.DRAWING_THICKNESS_EVIDENCE
    pages: Dict[Optional[int], List[Fragment]] = {}
    texts: Dict[Optional[int], List[str]] = {}
    for fragment in fragments:
        page = reading_reconcile._page(fragment)
        if fragment.type == "drawing" and fragment.content.structured_drawing_fields:
            pages.setdefault(page, []).append(fragment)
        texts.setdefault(page, []).extend(_page_texts(fragment, evidence))
    # Спецификация — документа целиком: она бывает на своём листе.
    positions = spec_positions(fragments) if settings.DRAWING_SENSE_SPEC_POSITIONS else None
    total: Dict[str, int] = {}
    for page, drawings in pages.items():
        for key, value in apply_page(drawings, texts.get(page, []), positions).items():
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


# Где в вырезке угла стоит сам знак: у правого края рамки, в верхней части
# угла (доли вырезки). Рамкой поля была вся вырезка угла — на 0,28 ширины
# листа, и поле на листе стояло левее знака на 0,08.
CORNER_SIGN = (0.55, 0.0, 1.0, 0.65)


def corner_sign_box(box: Sequence[float]) -> List[float]:
    """Рамка поля общего знака шероховатости внутри вырезки угла `box`."""
    x1, y1, x2, y2 = (float(v) for v in box)
    w, h = x2 - x1, y2 - y1
    return [round(x1 + CORNER_SIGN[0] * w, 4), round(y1 + CORNER_SIGN[1] * h, 4),
            round(x1 + CORNER_SIGN[2] * w, 4), round(y1 + CORNER_SIGN[3] * h, 4)]


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


# ===========================================================================
# Классы полей, позиции по спецификации, ряд шероховатости, общий знак в углу
# (DRAWING_SENSE_CLASS_RULES, DRAWING_SENSE_SPEC_POSITIONS,
# DRAWING_SENSE_RA_SERIES, DRAWING_CORNER_ROUGHNESS_V2)
# ===========================================================================

def _count(stats: Dict[str, int], key: str) -> None:
    # Ключ появляется только при срабатывании под своим флагом: без флагов
    # учёт `apply` тот же, что и раньше.
    stats[key] = stats.get(key, 0) + 1


def _note_value(item: _Item, value: str, review: bool) -> None:
    """
    Новая запись поля без спора прочтений: прежняя остаётся первым
    прочтением в `readings`, выведенная — с источником `sense_rule`.
    Связь не снимается: правило меняет запись, а не место величины.
    """
    from core.providers import vector_drawing

    field = item.field
    readings = list(field.readings or [{"source": _source(item), "value": field.value}])
    readings.append({"source": SENSE, "value": value})
    field.readings = readings
    field.value = value
    field.tolerance = vector_drawing.extract_tolerance(value)
    field.nature = vector_drawing.determine_nature(value)
    field.source_of_tolerance = vector_drawing.tolerance_source(value)
    field.unit = vector_drawing.extract_unit(value, field.category)
    if review:
        field.needs_review = True


def _recategorise(field: DrawingField, category: str) -> None:
    from core.providers import vector_drawing

    field.category = category
    field.unit = vector_drawing.extract_unit(field.value or "", category)


# --- C11: смешение категорий ------------------------------------------------

# Количество мест или отверстий без величины: «2 места», «4 отв.», «3 отверстия».
# Это пояснение к размеру (примечание), а не выноска с номером.
_COUNT_NOTE = re.compile(
    r"^\d{1,3}\s*(?:мест[аo]?|отв(?:ерсти[яей]|ерстие)?)\.?$", re.IGNORECASE)
# Пункт технических требований: номер, затем текст с заглавной буквы и
# хотя бы три слова. Графа «Материал» номером пункта не начинается.
# Сортамент («2 Лист Б-ПН…», «Круг…») пунктом не считается.
_NUMBERED_NOTE = re.compile(
    r"^\d{1,2}\s*[.)]?\s+(?!(?:Лист|Круг|Труба|Шестигранник|Полоса|Пруток|Проволока|Квадрат|Уголок"
    r"|Швеллер|Лента|Сталь)\b)[А-ЯЁ][а-яё]+(?:\s+\S+){3,}")
# Величина с номиналом и симметричным отклонением: «30°±2°», «12±0,1»,
# «⌀20±0,05», «45°30′±1°». Номинал обязателен: «±0,1» без него — допуск.
_ANGLE_OR_NUMBER = r"\d+(?:[.,]\d+)?\s*(?:°(?:\s*\d+\s*['′])?|['′])?"
_SIZE_WITH_PM = re.compile(
    r"^(?:" + _DIAMETER_SIGN + r"\s?)?" + _ANGLE_OR_NUMBER + r"\s*±\s*" + _ANGLE_OR_NUMBER + r"$")


def _class_rule(items: List[_Item], stats: Dict[str, int]) -> None:
    """
    Однозначные смешения категорий (DRAWING_SENSE_CLASS_RULES):
    «2 места», «4 отв.» в выноске или позиции — примечание `note`;
    пункт технических требований «3 Допускается…» в материале — `note`;
    «30°±2°» в допуске — размер с допуском `size` (допуск остаётся в
    `tolerance`), повтор того же места размером снимается; метка шва «№1»,
    «2№3» в позиции — выноска на листе любого вида. Значение не меняется.
    """
    for item in items:
        field = item.field
        if item.gone:
            continue
        value = " ".join(str(field.value or "").split())
        if field.category in ("callout", "position") and _COUNT_NOTE.match(value):
            _recategorise(field, "note")
            _count(stats, "class_count_note")
        elif field.category == "material" and _NUMBERED_NOTE.match(value):
            # «3 Допускается применять любой другой металлопрокат…» —
            # пункт технических требований, а не графа «Материал».
            _recategorise(field, "note")
            _count(stats, "class_numbered_note")
        elif field.category == "position" and _WELD_MARK.match(value):
            _recategorise(field, "callout")
            _count(stats, "class_weld_mark")
        elif field.category == "tolerance" and _SIZE_WITH_PM.match(value):
            twin = next((other for other in items
                         if other is not item and not other.gone and other.field.category == "size"
                         and reading_reconcile.notation(other.field.value) == reading_reconcile.notation(value)
                         and _comparable(item, other)), None)
            if twin is not None and _removable(item):
                _merge_readings(twin, item)
                _drop(item)
                _count(stats, "class_pm_merged")
            else:
                _recategorise(field, "size")
                _count(stats, "class_pm_size")


# --- C12: позиции по спецификации ------------------------------------------

_SPEC_POSITION = re.compile(r"^\d{1,3}$")
# Доля номеров 1…max, при которой спецификация считается прочитанной целиком.
_SPEC_COMPLETE = 0.8


def _position_column(headers: Sequence[Any]) -> Optional[int]:
    for n, header in enumerate(headers or []):
        if str(header or "").strip().casefold().startswith("поз"):
            return n
    return None


def _rows_positions(headers: Sequence[Any], rows: Iterable[Any]) -> List[int]:
    """Номера графы «Поз.» в строках с обозначением или наименованием."""
    rows = list(rows or [])
    column = _position_column(headers)
    if column is None and rows and isinstance(rows[0], (list, tuple)):
        # Шапка прочитана первой строкой таблицы.
        column = _position_column(rows[0])
        rows = rows[1:]
    if column is None:
        return []
    found = []
    for row in rows:
        if isinstance(row, dict):
            row = list(row.values())
        elif isinstance(row, str):
            row = row.split(" | ")
        cells = [str(c if c is not None else "").strip() for c in row or []]
        if column >= len(cells) or not _SPEC_POSITION.match(cells[column]):
            continue
        if any(any(ch.isalpha() for ch in cell) for n, cell in enumerate(cells) if n != column):
            number = int(cells[column])
            if number > 0:
                found.append(number)
    return found


def spec_positions(fragments: Iterable[Fragment]) -> Optional[set]:
    """
    Номера позиций спецификации документа: графа «Поз.» таблиц листа
    (`structured_payload.tables`) и табличных фрагментов. Спецификации нет —
    None: без неё правило позиций не делает ничего.
    """
    from core.providers.drawing_vision import KIND_TABLE

    found: List[int] = []
    for fragment in fragments:
        content = fragment.content
        if fragment.type == "drawing":
            for table in (content.structured_payload or {}).get("tables") or []:
                if isinstance(table, dict) and table.get("kind") == KIND_TABLE:
                    found.extend(_rows_positions(table.get("headers") or [], table.get("rows")))
        elif content.table_data is not None:
            found.extend(_rows_positions(content.table_data.headers, content.table_data.rows))
    # Одна строка — ещё не спецификация: по ней нельзя судить о прочих номерах.
    unique = set(found)
    return unique if len(unique) >= 2 else None


def _on_leader(field: DrawingField) -> bool:
    """Голое число стоит на полке выноски: категория «выноска»."""
    return field.category == "callout"


def _spec_position_rule(items: List[_Item], positions: set, stats: Dict[str, int]) -> None:
    """
    Позиция сверяется со спецификацией той же сборки
    (DRAWING_SENSE_SPEC_POSITIONS). «Позиция» с номером, которого в графе
    «Поз.» нет, — размер, если у неё есть связь с элементом, иначе выноска
    (с `needs_review`). Голое число на выноске с номером из спецификации —
    позиция. Метки швов «№N» не трогаются. Только лист сборки.

    Спецификация, прочитанная не целиком (из номеров 1…max есть меньше
    `_SPEC_COMPLETE` — пропуски строк, оборванная таблица), судить о
    позициях не может: «нет в спецификации» значило бы «не дочитали».
    Тогда правило не делает ничего.
    """
    top = max(positions)
    if len({number for number in positions if 1 <= number <= top}) < _SPEC_COMPLETE * top:
        return
    for item in items:
        field = item.field
        if item.gone:
            continue
        payload = item.fragment.content.structured_payload or {}
        if payload.get("sheet_type") != "assembly":
            continue
        value = " ".join(str(field.value or "").split())
        if not _SPEC_POSITION.match(value) or int(value) == 0:
            continue
        number = int(value)
        if field.category == "position" and number not in positions:
            _recategorise(field, "size" if field.link is not None else "callout")
            field.needs_review = True
            _count(stats, "spec_position_dropped")
        elif number in positions and _on_leader(field):
            _recategorise(field, "position")
            _count(stats, "spec_position_added")


# --- S1: ряд шероховатости ГОСТ 2789 ---------------------------------------

# Предпочтительные значения Ra и Rz, мкм (ГОСТ 2789-73, выделенные в рядах).
_RA_PREFERRED = (0.012, 0.025, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.3, 12.5, 25, 50, 100)
_RZ_PREFERRED = (0.025, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.3, 12.5, 25, 50, 100, 200, 400, 800, 1600)


def _full_series(low: float, high: float) -> Tuple[float, ...]:
    """Полный ряд ГОСТ 2789: 1; 1,25; 1,6; 2; 2,5; 3,2; 4; 5; 6,3; 8 по декадам."""
    return tuple(sorted({
        round(base * scale, 4)
        for base in (1.0, 1.25, 1.6, 2.0, 2.5, 3.2, 4.0, 5.0, 6.3, 8.0)
        for scale in (0.001, 0.01, 0.1, 1, 10, 100, 1000)
        if low <= round(base * scale, 4) <= high
    }))


# Полные ряды: Ra 0,008…400 (в нём и 10, 12,5, 16, 20, 25, 32, 40, 50, 63,
# 80, 100, 125, 160, 200, 250, 320, 400), Rz 0,025…1600.
_RA_FULL = _full_series(0.008, 400)
_RZ_FULL = _full_series(0.025, 1600)
# Шероховатость «тонкого» листа: Ra/Rz не грубее этого значения.
_FINE_LIMIT = 12.5
_RA_RECORD = re.compile(r"^(√?\s*)(R[az])(\s*)(\d{2,3})(\s*\*?\s*(?:\(\s*[√Vv✓]\s*\))?)$")
_ROUGHNESS_NUMBER = re.compile(r"R[az]\s*(\d+(?:[.,]\d+)?)")


def _in(series: Sequence[float], number: float) -> bool:
    return any(abs(number - v) < 1e-9 for v in series)


def ra_series_fix(value: Any, fine_sheet: bool = False) -> Optional[str]:
    """
    Потерянная запятая шероховатости: «Ra 08» -> «Ra 0,8». Иначе None.

    Правило (ГОСТ 2789):
    1. Запись с запятой не трогается; правится только целое число из двух-трёх
       цифр.
    2. Исправление — вставка запятой ровно в одном месте, после которой
       получается предпочтительное значение ряда; дробная часть не кончается
       нулём («160» никогда не становится «1,60»: так значения ряда не пишут).
    3. Если само прочтение — значение полного ряда («Ra 16», «Ra 63»,
       «Ra 125», «Rz 40»), оно может быть верным: правится, только если на
       том же листе есть другая шероховатость не грубее 12,5 (`fine_sheet`) —
       для Ra. Полный ряд Rz без такой оговорки не правится вовсе.
    4. Число не из полного ряда («Ra 08», «Ra 04») правится всегда.
    """
    text = str(value or "").strip()
    match = _RA_RECORD.match(text)
    if not match:
        return None
    head, letters, gap, number, tail = match.groups()
    kind = "R" + letters[1].lower()
    preferred = _RA_PREFERRED if kind == "Ra" else _RZ_PREFERRED
    full = _RA_FULL if kind == "Ra" else _RZ_FULL
    in_full = not number.startswith("0") and _in(full, float(number))
    if in_full and not (kind == "Ra" and fine_sheet):
        return None
    found = []
    for cut in range(1, len(number)):
        whole, part = number[:cut], number[cut:]
        if (len(whole) > 1 and whole.startswith("0")) or part.endswith("0"):
            continue
        if _in(preferred, float(whole + "." + part)):
            found.append(whole + "," + part)
    if len(found) != 1:
        return None
    return head + letters + gap + found[0] + tail


def _fine_elsewhere(items: List[_Item], item: _Item) -> bool:
    """Есть ли на листе другая шероховатость не грубее 12,5."""
    for other in items:
        if other is item or other.gone or other.field.category != "roughness":
            continue
        match = _ROUGHNESS_NUMBER.search(str(other.field.value or ""))
        if match and float(match.group(1).replace(",", ".")) <= _FINE_LIMIT:
            return True
    return False


def _ra_series_rule(items: List[_Item], stats: Dict[str, int]) -> None:
    """Потерянная запятая шероховатости (DRAWING_SENSE_RA_SERIES); сырое прочтение — в `readings`."""
    fixes = []
    for item in items:
        if item.gone or item.field.category != "roughness":
            continue
        fixed = ra_series_fix(item.field.value, fine_sheet=_fine_elsewhere(items, item))
        if fixed is not None:
            fixes.append((item, fixed))
    # Сначала решение по всем, потом правка: исправленная запись не должна
    # сама становиться «другой тонкой шероховатостью» для соседней.
    for item, fixed in fixes:
        _note_value(item, fixed, review=True)
        _count(stats, "ra_series")


# --- C14: общая шероховатость в углу листа ---------------------------------

# Знак общей шероховатости записью: «Ra 6,3 (√)», «√Rz 20 (√)», «√ (√)»
# (без параметров). Галочку в скобках распознавание отдаёт и буквой V.
_GENERAL_ROUGHNESS = re.compile(
    r"^√?\s*(?:(R[az])\s*(\d+(?:[.,]\d+)?))?\s*\(\s*[√Vv✓/]\s*\)$", re.IGNORECASE)
_PLAIN_ROUGHNESS = re.compile(r"^√?\s*(R[az])\s*(\d+(?:[.,]\d+)?)$", re.IGNORECASE)


def general_roughness(value: Any) -> Optional[str]:
    """Запись общего знака -> «Ra 6,3 (√)» или «√ (√)»; иначе None."""
    text = " ".join(str(value or "").split())
    match = _GENERAL_ROUGHNESS.match(text)
    if not match:
        return None
    if match.group(1) is None:
        # «√ (√)» — знак без параметров; «(V)» без знака — не шероховатость.
        return "√ (√)" if text.startswith("√") else None
    return "R%s %s (√)" % (match.group(1)[1].lower(), match.group(2).replace(".", ","))


def _unplaced(box: Optional[Sequence[float]]) -> bool:
    """Рамки нет или она — почти весь лист: место надписи не прочитано."""
    if not box or len(box) != 4:
        return True
    x1, y1, x2, y2 = (float(v) for v in box)
    return (x2 - x1) * (y2 - y1) >= 0.8


def _inside(box: Optional[Sequence[float]], zone: Sequence[float]) -> bool:
    centre = reading_reconcile._centre(box or [])
    if centre is None:
        return False
    return float(zone[0]) <= centre[0] <= float(zone[2]) and float(zone[1]) <= centre[1] <= float(zone[3])


def _plain_general(value: Any) -> Optional[str]:
    match = _PLAIN_ROUGHNESS.match(" ".join(str(value or "").split()))
    if not match:
        return None
    return "R%s %s (√)" % (match.group(1)[1].lower(), match.group(2).replace(".", ","))


def _corner_rule(drawings: Sequence[Fragment], items: List[_Item], stats: Dict[str, int]) -> None:
    """
    Общий знак шероховатости (DRAWING_CORNER_ROUGHNESS_V2), без вызовов модели:
    1) поле с записью «Ra x (√)» / «√ (√)» — шероховатость с этой записью;
       без места (рамка всего листа) — в правом верхнем углу рамки;
    2) иначе единственная шероховатость «Ra x» листа целиком в правом
       верхнем углу рамки (`CORNER`), вне вырезок видов, на листе не сборки
       и не схемы, получает «(√)»;
    3) иначе строка текста листа ровно «Ra x (√)» даёт поле в углу.
    Шероховатость внутри вида общей не становится никогда.
    """
    general = False
    for item in items:
        field = item.field
        if item.gone or field.category in ("title_block", "material"):
            continue
        canonical = general_roughness(field.value)
        if canonical is None:
            continue
        general = True
        if field.category != "roughness":
            _recategorise(field, "roughness")
            _count(stats, "corner_recategorised")
        if canonical != field.value:
            _note_value(item, canonical, review=False)
            _count(stats, "corner_canonical")
        if item.whole and _unplaced(field.bbox):
            # Знак «Ra x (√)» без места (содержание листа без рамок): по
            # ГОСТ 2.309 общий знак стоит в правом верхнем углу рамки, и
            # рамка всего листа была бы враньём о месте.
            frame = ((item.fragment.content.structured_payload or {}).get("layout") or {}).get("frame")
            field.bbox = corner_sign_box(corner_box(frame))
            item.box = _page_box(item.fragment, field, item.whole)
            _count(stats, "corner_placed")
    if general:
        return
    sheets = [f for f in drawings if reading_reconcile._is_whole_sheet(f)]
    if len(sheets) != 1:
        return
    sheet = sheets[0]
    payload = sheet.content.structured_payload or {}
    if payload.get("sheet_type") in ("assembly", "scheme"):
        return
    zone = corner_box((payload.get("layout") or {}).get("frame"))
    views = [list(f.position.bbox) for f in drawings
             if f is not sheet and f.position and f.position.bbox and len(f.position.bbox) == 4]
    candidates = [
        item for item in items
        if not item.gone and item.fragment is sheet and item.field.category == "roughness"
        and _plain_general(item.field.value) is not None
        and _inside(item.field.bbox, zone)
        and not any(_inside(item.field.bbox, view) for view in views)
    ]
    values = {_plain_general(item.field.value) for item in candidates}
    if len(values) == 1:
        value = values.pop()
        for item in candidates:
            _note_value(item, value, review=False)
        _count(stats, "corner_marked")
        return
    if candidates:
        # Разные шероховатости в углу — какая из них общая, правило не решает.
        return
    lines = [line for line in (sheet.content.text or "").splitlines() if line.strip()]
    found = {general_roughness(line) for line in lines} - {None}
    if len(found) != 1:
        return
    from core.providers import vector_drawing

    value = found.pop()
    sheet.content.structured_drawing_fields = list(sheet.content.structured_drawing_fields or []) + [
        DrawingField(category="roughness", value=value, bbox=corner_sign_box(zone), confidence=0.7,
                     provenance="full_page_ocr", unit=vector_drawing.extract_unit(value, "roughness"))]
    _count(stats, "corner_from_text")
