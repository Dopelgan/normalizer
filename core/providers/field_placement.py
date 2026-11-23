"""
Размещение полей чертежа на странице (DRAWING_FIELD_PLACEMENT_V2).

Чем было плохо. Рамка поля (`DrawingField.bbox`) — доли той картинки,
которую видела модель, после её выправления (`drawing_vision.analyse`:
наклон `angle` без расширения, затем поворот `turn` кратно 90° с
расширением). Для прохода «лист целиком» это почти страница, а для вырезки
вида MinerU — сама вырезка: поле «Б» с рамкой [0.07, 0.02, …] стоит в углу
вырезки, а просмотрщик рисовал его в углу страницы. Кроме того, одна
надпись приходила несколькими полями — с листа, с вырезки, из клетки
штампа и из сводной записи штампа (у всех полей сводной записи одна рамка —
весь штамп), а ссылки на пункты требований («п4» у размера) читались как
размеры.

Что здесь делается, по порядку, после всех проходов документа:

1. Рамки полей (и связи размера `link`) каждого чертёжного фрагмента
   переводятся в доли страницы: обратный поворот, обратный наклон, затем
   рамка вырезки на странице (`position.bbox` фрагмента). У фрагмента
   ставится `coordinate_reference = "page"`. Рамка по умолчанию или
   область (общая рамка нескольких полей, больше половины картинки) —
   не место надписи: поле получает рамку своей картинки и `bbox_exact=False`.
2. Ссылка на пункт требований в поле величины («п4», «n.5») становится
   выноской «п.4».
3. Повторы одной надписи на странице сводятся в одно поле: значение — из
   самого уверенного прочтения (подписанное «Метка: значение» — главнее,
   по нему собирается запись штампа), рамка — самая точная. Расходящиеся
   прочтения («Постнов»/«Постанов») остаются у поля в `alternatives`,
   поле помечается `needs_review`.
4. Сомнительные прочтения (уверенность ниже 0.6, слово из смеси латиницы и
   кириллицы, «4N01» в выноске) помечаются `low_quality` и `needs_review`.

Вызовов модели нет. Векторные чертежи (`vector_text_layer`) не трогаются:
их рамки уже в долях страницы.
"""

from __future__ import annotations

import math
import re
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Sequence, Tuple

from core.models.contract import DrawingField, Fragment
from core.providers.reading_reconcile import NUMERIC, digits, notation

PAGE = "page"
WHOLE = [0.0, 0.0, 1.0, 1.0]

# Рамка больше этой доли картинки — область, а не место надписи.
REGION_SHARE = 0.5
# Столько и больше полей фрагмента с одной и той же рамкой — сводная запись
# области (штамп целиком), а не места надписей.
SHARED_BOX = 3
# Две рамки — одно место, если они пересекаются с таким запасом (доля страницы).
MARGIN = 0.02
# Похожесть записей, при которой два прочтения — одна надпись с вариантами.
SIMILAR = 0.8
LOW_CONFIDENCE = 0.6

# Категории, которые могут оказаться одной надписью.
_KIN = {
    frozenset({"title_block", "material"}),
    frozenset({"callout", "note"}),
    frozenset({"callout", "position"}),
}
_LABELLED = re.compile(r"^\s*([A-Za-zА-Яа-яЁё][^:]{0,40}?)\s*:\s*(.+)$", re.S)
_HANDWRITTEN = re.compile(r"\s*\((?:рукописно|от руки)\)\s*$", re.I)
_ITEM_REF = re.compile(r"^\s*[пПnN]\s*\.?\s*(\d{1,2})\s*$")
_MIXED_WORD = re.compile(r"[A-Za-z][А-Яа-яЁё]|[А-Яа-яЁё][A-Za-z]")
_DIGIT_LATIN_DIGIT = re.compile(r"^\d+[A-Za-z]\d+$")
_NOT_WORD = re.compile(r"[^\w]+")


def enabled() -> bool:
    try:
        from core.config import settings
    except Exception:  # noqa: BLE001 — без настроек размещение выключено
        return False
    return bool(getattr(settings, "DRAWING_FIELD_PLACEMENT_V2", False))


# ===========================================================================
# Координаты: выправленная картинка -> страница
# ===========================================================================

def _unturn(x: float, y: float, turn: int, w: float, h: float) -> Tuple[float, float]:
    """Пиксель картинки после поворота `turn` (против часовой) -> до поворота."""
    if turn == 90:
        return w - y, x
    if turn == 180:
        return w - x, h - y
    if turn == 270:
        return y, h - x
    return x, y


def _unskew(x: float, y: float, angle: float, w: float, h: float) -> Tuple[float, float]:
    """Пиксель после наклона `angle` вокруг центра без расширения -> до наклона."""
    if not angle:
        return x, y
    cx, cy = w / 2.0, h / 2.0
    t = math.radians(angle)
    dx, dy = x - cx, y - cy
    return cx + dx * math.cos(t) - dy * math.sin(t), cy + dx * math.sin(t) + dy * math.cos(t)


def point_to_page(x: float, y: float, frame: Sequence[float], angle: float = 0.0,
                  turn: int = 0, size: Optional[Sequence[float]] = None) -> Tuple[float, float]:
    """
    Точка в долях выправленной картинки -> в долях страницы. `frame` — рамка
    картинки на странице, `size` — её размер до выправления (без него наклон
    не учитывается: при долях градуса сдвиг меньше пикселя на надпись).
    """
    turn = int(turn or 0) % 360
    w, h = (float(size[0]), float(size[1])) if size else (1.0, 1.0)
    tw, th = (h, w) if turn in (90, 270) else (w, h)
    px, py = _unturn(x * tw, y * th, turn, w, h)
    if size:
        px, py = _unskew(px, py, float(angle or 0.0), w, h)
    u, v = px / w, py / h
    fx1, fy1, fx2, fy2 = (float(c) for c in frame)
    return (_clamp(fx1 + u * (fx2 - fx1)), _clamp(fy1 + v * (fy2 - fy1)))


def box_to_page(bbox: Sequence[float], frame: Sequence[float], angle: float = 0.0,
                turn: int = 0, size: Optional[Sequence[float]] = None) -> List[float]:
    """Рамка в долях выправленной картинки -> охватывающая рамка в долях страницы."""
    x1, y1, x2, y2 = (float(c) for c in bbox)
    points = [point_to_page(x, y, frame, angle, turn, size)
              for x, y in ((x1, y1), (x2, y1), (x1, y2), (x2, y2))]
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    return [round(min(xs), 4), round(min(ys), 4), round(max(xs), 4), round(max(ys), 4)]


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _is_region(bbox: Sequence[float]) -> bool:
    if not bbox or len(bbox) != 4:
        return True
    x1, y1, x2, y2 = (float(c) for c in bbox)
    return [x1, y1, x2, y2] == WHOLE or (x2 - x1) * (y2 - y1) > REGION_SHARE


def _placed_by_processor(fragment: Fragment) -> bool:
    payload = fragment.content.structured_payload or {}
    return (fragment.type == "drawing" and fragment.provenance.method != "vector_text_layer"
            and payload.get("coordinate_reference") == "deskewed_page"
            and isinstance(payload.get("layout"), dict))


def place_fragment(fragment: Fragment) -> None:
    """Рамки полей и связей фрагмента — в доли страницы."""
    payload = fragment.content.structured_payload or {}
    layout = payload.get("layout") or {}
    frame = list(fragment.position.bbox) if fragment.position and fragment.position.bbox else WHOLE
    angle = float(layout.get("angle") or 0.0)
    turn = int(layout.get("turn") or 0)
    size = layout.get("source_size")
    size = size if isinstance(size, (list, tuple)) and len(size) == 2 and min(size) > 0 else None
    fields = fragment.content.structured_drawing_fields or []
    shared: Dict[Tuple[float, ...], int] = {}
    for field in fields:
        key = tuple(round(float(c), 4) for c in field.bbox)
        shared[key] = shared.get(key, 0) + 1
    for field in fields:
        key = tuple(round(float(c), 4) for c in field.bbox)
        if _is_region(field.bbox) or shared[key] >= SHARED_BOX:
            # Место надписи неизвестно: только область, где её прочли.
            field.bbox = (box_to_page(field.bbox, frame, angle, turn, size)
                          if not _is_whole(field.bbox) else [round(float(c), 4) for c in frame])
            field.bbox_exact = False
        else:
            field.bbox = box_to_page(field.bbox, frame, angle, turn, size)
            field.bbox_exact = True
        if field.link is not None:
            field.link.target_bbox = box_to_page(field.link.target_bbox, frame, angle, turn, size)
            field.link.leader = [list(map(lambda c: round(c, 4),
                                          point_to_page(p[0], p[1], frame, angle, turn, size)))
                                 for p in field.link.leader if len(p) == 2]
    payload["coordinate_reference"] = PAGE
    fragment.content.structured_payload = payload


def _is_whole(bbox: Sequence[float]) -> bool:
    return [float(c) for c in bbox] == WHOLE


# ===========================================================================
# Классы надписей
# ===========================================================================

def reclassify(field: DrawingField) -> None:
    """«п4» в поле величины — ссылка на пункт требований, а не размер."""
    if field.category in NUMERIC:
        match = _ITEM_REF.match(field.value or "")
        if match:
            field.category = "callout"
            field.value = f"п.{match.group(1)}"
            field.tolerance = None
            field.unit = ""
            # Связь «п4 -> габаритная ширина» (крышка 687000.001.001) —
            # связь ссылки на пункт требований, а не величины: её нет.
            field.link = None


def mark_quality(field: DrawingField) -> None:
    """Сомнительное прочтение: прятать в просмотре и отдать на сверку."""
    value = field.value or ""
    words = value.split()
    garbled = any(_MIXED_WORD.search(word) for word in words) or (
        field.category in ("callout", "position") and any(_DIGIT_LATIN_DIGIT.match(w) for w in words)
    )
    if garbled or field.confidence < LOW_CONFIDENCE:
        field.low_quality = True
        field.needs_review = True


# ===========================================================================
# Повторы одной надписи
# ===========================================================================

def _split_label(value: str) -> Tuple[str, str]:
    text = _HANDWRITTEN.sub("", value or "")
    match = _LABELLED.match(text)
    if match and not any(ch.isdigit() for ch in match.group(1)):
        return match.group(1).strip().casefold(), match.group(2).strip()
    return "", text.strip()


def _folded(value: str) -> str:
    return _NOT_WORD.sub("", notation(value))


def _kin(a: DrawingField, b: DrawingField) -> bool:
    return a.category == b.category or frozenset({a.category, b.category}) in _KIN


def _overlap(a: Sequence[float], b: Sequence[float]) -> bool:
    return (a[0] - MARGIN <= b[2] and b[0] - MARGIN <= a[2]
            and a[1] - MARGIN <= b[3] and b[1] - MARGIN <= a[3])


def _area(bbox: Sequence[float]) -> float:
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def _relation(a: DrawingField, b: DrawingField) -> Optional[str]:
    """
    Как связаны два прочтения: «same» — одна запись; «variant» — та же
    надпись, прочитанная иначе; «part»/«part_variant» — одно прочтение —
    часть другого («Лицевая сторона» и «Лицевая сторона п.8», клетка штампа
    и сводная запись «Материал: …»); None — разные надписи.
    """
    if not _kin(a, b):
        return None
    label_a, value_a = _split_label(a.value)
    label_b, value_b = _split_label(b.value)
    if label_a and label_b and label_a != label_b:
        return None     # «Лист: 1» и «Листов: 1» — разные графы
    fa, fb = _folded(value_a), _folded(value_b)
    if not fa or not fb:
        return None
    if fa == fb:
        if min(len(fa), len(fb)) < 2 and (a.bbox_exact is False or b.bbox_exact is False):
            return None     # «1» в штампе — и лист, и листов: не угадываем
        return "same"
    if a.category in NUMERIC or b.category in NUMERIC:
        return None     # у величин повтором считается только та же запись
    short, long_ = (fa, fb) if len(fa) <= len(fb) else (fb, fa)
    if len(short) >= 2 and short in long_:
        return "part"
    if len(short) < 4 or digits(short) not in digits(long_):
        return None
    if digits(fa) == digits(fb) and SequenceMatcher(None, fa, fb).ratio() >= SIMILAR:
        return "variant"
    # Частью считается только заметно более короткое прочтение: «Белянинова»
    # против «Белянчинова» — одна фамилия в двух прочтениях, а не часть.
    if len(long_) - len(short) >= 3 and _window_ratio(short, long_) >= SIMILAR:
        return "part_variant"
    return None


def _window_ratio(short: str, long_: str) -> float:
    """Похожесть короткой записи на лучший кусок длинной той же длины."""
    size = len(short)
    return max((SequenceMatcher(None, short, long_[i:i + size]).ratio()
                for i in range(0, len(long_) - size + 1)), default=0.0)


def _precision(field: DrawingField) -> Tuple[int, float]:
    return (0 if field.bbox_exact is False else 1, -_area(field.bbox))


def _union(a: Sequence[float], b: Sequence[float]) -> List[float]:
    return [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]


def _merge(keep: DrawingField, other: DrawingField, relation: str, from_parts: bool) -> DrawingField:
    """Одно поле из двух прочтений одной надписи."""
    variants = list(keep.alternatives or []) + list(other.alternatives or [])
    if relation in ("part", "part_variant"):
        # Остаётся целое прочтение; рамка — его собственная, а у сводной
        # записи области — рамка найденных частей.
        whole, part = ((keep, other) if len(_folded(_split_label(keep.value)[1]))
                       >= len(_folded(_split_label(other.value)[1])) else (other, keep))
        merged = whole.model_copy(deep=True)
        if part.bbox_exact and (whole.bbox_exact is False or from_parts):
            merged.bbox = _union(whole.bbox, part.bbox) if from_parts else list(part.bbox)
            merged.bbox_exact = True
        if relation == "part_variant":
            variants.append(_split_label(part.value)[1])
    else:
        by_value = sorted((keep, other), key=lambda f: (bool(_split_label(f.value)[0]), f.confidence),
                          reverse=True)[0]
        by_box = max((keep, other), key=_precision)
        merged = by_value.model_copy(deep=True)
        merged.bbox = list(by_box.bbox)
        merged.bbox_exact = by_box.bbox_exact
        if relation == "variant":
            variants.append(_split_label((other if by_value is keep else keep).value)[1])
        else:
            merged.confidence = max(keep.confidence, other.confidence)
    for source in (keep, other):
        if merged.link is None and source.link is not None:
            merged.link = source.link.model_copy(deep=True)
    chosen = _folded(_split_label(merged.value)[1])
    variants = list(dict.fromkeys(v for v in variants if _folded(v) != chosen))
    merged.alternatives = variants or None
    merged.needs_review = True if (variants or keep.needs_review or other.needs_review) else None
    merged.low_quality = True if (keep.low_quality and other.low_quality) else None
    return merged


def dedupe_page(drawings: List[Fragment]) -> int:
    """Повторы надписей среди полей чертёжных фрагментов одной страницы. Возвращает число снятых."""
    keys: List[Tuple[int, int]] = []
    fields: Dict[Tuple[int, int], DrawingField] = {}
    for fragment in drawings:
        for n, field in enumerate(fragment.content.structured_drawing_fields or []):
            keys.append((id(fragment), n))
            fields[(id(fragment), n)] = field
    removed: set = set()
    from_parts: set = set()
    # Сначала поля с точной рамкой: к ним сводятся прочтения областей.
    order = sorted(keys, key=lambda k: _precision(fields[k]), reverse=True)
    for key_a in order:
        if key_a in removed:
            continue
        for key_b in order:
            if key_b == key_a or key_b in removed:
                continue
            a, b = fields[key_a], fields[key_b]
            if not _overlap(a.bbox, b.bbox):
                continue
            relation = _relation(a, b)
            if relation is None:
                continue
            region = b if b.bbox_exact is False else (a if a.bbox_exact is False else None)
            if region is not None and relation in ("same", "variant") and _rivals(
                    region, fields, removed, {key_a, key_b}):
                continue    # прочтение области подходит нескольким надписям — не угадываем
            parts = key_a in from_parts or key_b in from_parts
            merged = _merge(a, b, relation, parts)
            if relation in ("part", "part_variant") and merged.bbox_exact and (
                    a.bbox_exact is False or b.bbox_exact is False or parts):
                from_parts.add(key_a)
            fields[key_a] = merged
            removed.add(key_b)
    for fragment in drawings:
        if fragment.content.structured_drawing_fields is None:
            continue
        fragment.content.structured_drawing_fields = [
            fields[(id(fragment), n)] for n in range(len(fragment.content.structured_drawing_fields))
            if (id(fragment), n) not in removed
        ]
    return len(removed)


def _rivals(region: DrawingField, fields: Dict[Any, DrawingField], removed: set, pair: set) -> bool:
    """Есть ли у прочтения области ещё одна точная надпись с той же записью."""
    for key, other in fields.items():
        if key in pair or key in removed or other.bbox_exact is False:
            continue
        if _overlap(region.bbox, other.bbox) and _relation(region, other) == "same":
            return True
    return False


# ===========================================================================
# Вход
# ===========================================================================

def apply(fragments: List[Fragment]) -> List[Fragment]:
    """Размещение, классы, повторы и качество полей чертежей документа; список тот же."""
    pages: Dict[int, List[Fragment]] = {}
    for fragment in fragments:
        if not _placed_by_processor(fragment):
            continue
        place_fragment(fragment)
        for field in fragment.content.structured_drawing_fields or []:
            reclassify(field)
            mark_quality(field)
        if fragment.position is not None:
            pages.setdefault(int(fragment.position.page), []).append(fragment)
    for drawings in pages.values():
        dedupe_page(drawings)
    return fragments
