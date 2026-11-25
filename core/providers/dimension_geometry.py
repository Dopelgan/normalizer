"""
Размерные линии на растре листа (`DRAWING_DIMENSION_GEOMETRY`, карточка N-G1).

Зачем. Все чертежи заказчика — сканы: геометрии векторного PDF у них нет, и
связь размера с элементом (`dimension_links`) держится только на ответе
модели, которая сама прослеживает линии на уменьшенном листе. Здесь та же
связь строится классическим зрением на выправленном листе в разрешении
рендера (200 dpi), без модели и без обучения.

Что ищется у надписи линейного размера (и диаметра, проставленного как
линейный размер по образующим).

1. **Размерная линия.** Длинный тонкий горизонтальный отрезок под надписью
   (у повёрнутой надписи — вертикальный справа от неё; надпись с другой
   стороны допускается с небольшим штрафом) не дальше 1,5 высоты текста,
   длиной от 2,5 высоты текста и толщиной до ~3 px. Отрезок должен проходить
   под серединой надписи. Толстый отрезок — контур детали, а не размерная
   линия.
2. **Выносные линии.** Поперёк размерной линии, ближайшие слева и справа от
   середины надписи: с одной стороны заходят за размерную линию на 1–5 мм
   (ЕСКД 2.307), с другой уходят к элементу. Отрезок, пересекающий линию
   далеко в обе стороны (осевая, чужая выносная), выносной не считается.
   Нет выносной у какого-то конца — связи нет. Так цепочка размеров на одной
   линии делится по выносным, а надпись берёт свой участок.
3. **Стрелки** у концов (утолщение линии у выносной) — только уверенность.
4. **Элемент.** Выносная прослеживается от размерной линии до конца; её
   дальний конец — точка на кромке, от которой идёт размер (`edges`).
5. **Роль по виду.** Вид — связная область толстых линий (контура) после
   удаления линий не толще размерной; рамка формата и штамп в виды не идут
   (слишком велики). Оба конца на краях вида вдоль оси — `overall`
   (габаритный), один — `edge` (от края), ни одного — `inner`.
6. **Согласие масштаба.** У линейных размеров одного вида отношение
   «значение / длина в пикселях» одно и то же. При четырёх и больше размерах
   вида выпадающее больше чем на 12 % от медианы получает `scale_ok: false`,
   а поле — `needs_review`: так ловится и ошибка прочтения цифры, и «50,5»,
   которое на деле «S0,5».

У радиуса (и диаметра по окружности) ищется выноска: прямой отрезок от
надписи наружу со стрелкой на дальнем конце; без стрелки связи нет.

Связь ставится только при подтверждённых концах. Векторная связь не
переписывается; у связи модели подпись остаётся её, а уверенность не
поднимается. Ничего не нашлось или сбой — поля как были.

Модуль работает на numpy и PIL — новых зависимостей нет.
"""

from __future__ import annotations

import logging
import math
import re
import time
from dataclasses import dataclass, field as dataclass_field
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

METHOD_RASTER = "raster"
_CATEGORIES = ("size", "radius")

AXIS_H = "horizontal"
AXIS_V = "vertical"
ROLE_OVERALL = "overall"
ROLE_EDGE = "edge"
ROLE_INNER = "inner"
KIND_LINEAR = "linear"
KIND_DIAMETER = "diameter"
KIND_RADIUS = "radius"

# --- размерная линия (доли высоты надписи h) -------------------------------
_BAND_H = 1.5            # линия не дальше 1,5h от надписи
_OTHER_SIDE_H = 0.3      # линия не с той стороны надписи — штраф к расстоянию
_MIN_LINE_H = 2.5        # линия не короче 2,5h
_MIN_LINE_WIDTH = 0.8    # и не короче 0,8 ширины надписи
_MAX_THICK_PX = 3        # линия не толще ~3 px (на 200 dpi — 0,4 мм)
_MAX_THICK_H = 0.15      # или 0,15h на листе с крупным текстом
_GAP_PX = 3              # разрыв скана внутри линии
_UNIQUE_H = 0.25         # две подтверждённые линии ближе этой разницы — неразличимы
_TRIES = 3
# --- выносные линии -------------------------------------------------------
_OVER_MIN_H = 0.12       # заходит за размерную линию хотя бы на ~0,4 мм
_OVER_MIN_PX = 3
_OVER_MAX_H = 2.0        # и не больше ~5–7 мм
_EXT_MIN_H = 0.5         # к элементу — не короче 0,5h
_EXT_MIN_PX = 8
_MIN_SPAN_WIDTH = 0.5    # участок не уже половины надписи
_TRACE_SHARE = 0.5       # выносная прослеживается не дальше половины листа
# --- стрелки -----------------------------------------------------------------
_ARROW_REACH_H = 1.2     # стрелка не длиннее 1,2h
_ARROW_GAP_PX = 3        # у самой выносной утолщение — это её толщина
_ARROW_WIDE_H = 0.6      # стрелка не шире 0,6h; шире — пересекающая линия
_ARROW_MIN_H = 0.2       # утолщение тянется хотя бы на 0,2h
_ARROW_MIN_PX = 3
# --- выноска радиуса и диаметра ------------------------------------------
_RING_H = 0.5            # кольцо поиска вокруг надписи
_LEADER_REACH_D = 0.3    # выноска не длиннее 30 % листа
_LEADER_REACH_H = 40.0   # и 40h
_LEADER_MIN_H = 2.5
_LEADER_POINTS = 32
_LEADER_COARSE_DEG = 0.5
_LEADER_FINE_DEG = 0.1
_TIP_H = 0.25           # остриё упирается в штрих поперёк выноски шириной от 0,5h
# --- вид и роль ------------------------------------------------------------
_GRID_SIDE = 800         # сетка связности: столько клеток по длинной стороне
_VIEW_MAX_AREA = 0.7     # больше — рамка формата со штампом, а не вид
_ROLE_TOL_H = 0.5
_ROLE_TOL_D = 0.004
# --- масштаб ---------------------------------------------------------------
_SCALE_TOL = 0.12
_SCALE_MIN = 4

CONFIDENCE_LINEAR = (0.6, 0.7, 0.8)   # стрелок у концов: 0, 1, 2
CONFIDENCE_LEADER = 0.6

_DIAMETER_SIGNS = ("⌀", "Ø", "ø", "∅", "Φ", "φ", "Ф")
_RADIUS = re.compile(r"^R\s*\d")
_NOT_LINEAR = re.compile(r"°|\d\s*[×xхX*]\s*\d|^\s*[MSМС]\s*\d", re.IGNORECASE)
_NUMBER = re.compile(r"(\d+(?:[.,]\d+)?)")

Point = Tuple[float, float]


# ===========================================================================
# Флаг
# ===========================================================================

def enabled() -> bool:
    """Геометрия размерных линий включена флагом `DRAWING_DIMENSION_GEOMETRY`."""
    try:
        from core.config import settings
    except Exception:  # noqa: BLE001 — без настроек геометрии нет
        return False
    return bool(getattr(settings, "DRAWING_DIMENSION_GEOMETRY", False))


def _numpy():
    try:
        import numpy
    except ImportError:  # pragma: no cover — numpy есть в образе
        return None
    return numpy


# ===========================================================================
# Лист
# ===========================================================================

class _Sheet:
    """Маска тёмного выправленного листа в разрешении рендера."""

    def __init__(self, image, np):
        from core.providers.raster_drawing import _downscale, _threshold

        self.np = np
        grey = np.asarray(image.convert("L"))
        small = np.asarray(_downscale(image, 1000).convert("L"), dtype=np.float32)
        self.ink = grey < _threshold(small, np)
        self.height, self.width = self.ink.shape
        self._transposed = None

    @property
    def long_side(self) -> int:
        return max(self.width, self.height)

    def canvas(self, axis: str):
        """Маска, где искомая размерная линия горизонтальна."""
        if axis == AXIS_H:
            return self.ink
        if self._transposed is None:
            self._transposed = self.np.ascontiguousarray(self.ink.T)
        return self._transposed


@dataclass
class _Found:
    kind: str
    ends: List[Point]                 # концы размерной линии (или выноски), px листа
    edges: List[Point]                # точки на кромках элемента, px листа
    confidence: float
    axis: Optional[str] = None        # у линейного размера
    span_px: Optional[int] = None
    thickness: int = 0
    distance: float = 0.0
    h: float = 0.0
    role: Optional[str] = None
    view: Optional[Tuple[int, int, int, int]] = None
    scale_ok: Optional[bool] = None
    extra: Dict[str, Any] = dataclass_field(default_factory=dict)


# ===========================================================================
# Вид величины по записи
# ===========================================================================

def kind_of(value: Any) -> Optional[str]:
    """
    Что за величина по записи: линейный размер, диаметр, радиус или None
    (угол, фаска «2×45°», резьба, толщина «S…», количество «4×…»).
    """
    text = str(value or "").strip()
    if not text:
        return None
    if _RADIUS.match(text):
        return KIND_RADIUS
    if any(sign in text for sign in _DIAMETER_SIGNS):
        return None if "°" in text else KIND_DIAMETER
    if _NOT_LINEAR.search(text):
        return None
    if re.match(r"^[(\[]?\s*\d", text):
        return KIND_LINEAR
    return None


def number_of(value: Any) -> Optional[float]:
    """Номинал размера (первое число после знака диаметра), мм; None — нет числа."""
    text = str(value or "").strip()
    for sign in _DIAMETER_SIGNS:
        if sign in text:
            text = text.split(sign, 1)[1]
            break
    match = _NUMBER.search(text)
    if not match:
        return None
    try:
        number = float(match.group(1).replace(",", "."))
    except ValueError:
        return None
    return number if number > 0 else None


# ===========================================================================
# Мелкие операции над масками
# ===========================================================================

def _contig(block, np, axis: int):
    """Длина сплошного тёмного от начала вдоль оси; всё тёмное — вся длина."""
    size = block.shape[axis]
    if size == 0:
        shape = list(block.shape)
        del shape[axis]
        return np.zeros(shape, dtype=np.int64)
    first = np.argmin(block, axis=axis)
    return np.where(block.all(axis=axis), size, first)


def _bridge(rows, np, gap: int):
    """Разрывы до `gap` px вдоль строки закрываются (скан рвёт тонкую линию)."""
    if gap <= 0 or rows.shape[1] < 3:
        return rows
    left = np.zeros_like(rows)
    right = np.zeros_like(rows)
    for d in range(1, gap + 1):
        left[:, d:] |= rows[:, :-d]
        right[:, :-d] |= rows[:, d:]
    return rows | (left & right)


def _widen(block, np):
    """Столбец плюс соседи: выносная на скане гуляет на пиксель."""
    out = block.copy()
    if block.shape[1] > 1:
        out[:, 1:] |= block[:, :-1]
        out[:, :-1] |= block[:, 1:]
    return out


def _reach(strip, np, gap: int) -> int:
    """Сколько тёмного подряд от начала полосы, с разрывами до `gap` px."""
    empty = np.flatnonzero(~strip)
    if empty.size == 0:
        return int(strip.size)
    breaks = np.flatnonzero(np.diff(empty) > 1)
    starts = np.concatenate(([empty[0]], empty[breaks + 1]))
    ends = np.concatenate((empty[breaks], [empty[-1]]))
    for start, end in zip(starts, ends):
        if end - start + 1 > gap or end == strip.size - 1:
            return int(start)
    return int(strip.size)


def _erode(mask, np, radius: int):
    """Эрозия квадратом (2r+1): исчезает всё не толще 2r px."""
    out = mask.copy()
    for d in range(1, radius + 1):
        out[:, d:] &= mask[:, :-d]
        out[:, :-d] &= mask[:, d:]
        out[:, :d] = False
        out[:, -d:] = False
    rows = out.copy()
    for d in range(1, radius + 1):
        out[d:, :] &= rows[:-d, :]
        out[:-d, :] &= rows[d:, :]
        out[:d, :] = False
        out[-d:, :] = False
    return out


def _dilate(mask, np, radius: int = 1):
    out = mask.copy()
    for d in range(1, radius + 1):
        out[:, d:] |= mask[:, :-d]
        out[:, :-d] |= mask[:, d:]
    rows = out.copy()
    for d in range(1, radius + 1):
        out[d:, :] |= rows[:-d, :]
        out[:-d, :] |= rows[d:, :]
    return out


def _groups(indices, np, gap: int = 2) -> List[Tuple[int, int]]:
    """Подряд идущие номера (с разрывом до `gap`) — группы (первый, последний)."""
    if len(indices) == 0:
        return []
    breaks = np.flatnonzero(np.diff(indices) > gap)
    starts = np.concatenate(([indices[0]], indices[breaks + 1]))
    ends = np.concatenate((indices[breaks], [indices[-1]]))
    return [(int(s), int(e)) for s, e in zip(starts, ends)]


# ===========================================================================
# Линейный размер: размерная линия и выносные
# ===========================================================================

def _thick_limit(h: float) -> int:
    return max(_MAX_THICK_PX, int(round(_MAX_THICK_H * h)))


def _linear(sheet: _Sheet, box: Sequence[float], axis: str) -> Tuple[Optional[_Found], str]:
    """
    Размерная линия вдоль `axis` у надписи `box` (px листа). В координатах
    `canvas` линия всегда горизонтальна: у вертикальной лист транспонирован.
    """
    np = sheet.np
    canvas = sheet.canvas(axis)
    x1, y1, x2, y2 = box if axis == AXIS_H else (box[1], box[0], box[3], box[2])
    h = max(6.0, y2 - y1)
    rows_total, cols_total = canvas.shape
    cx = int(round((x1 + x2) / 2))
    if not 0 <= cx < cols_total:
        return None, "outside"
    lo = max(0, int(y1 - _BAND_H * h))
    hi = min(rows_total, int(math.ceil(y2 + _BAND_H * h)) + 1)
    if hi - lo < 2:
        return None, "outside"
    band = _bridge(canvas[lo:hi], np, _GAP_PX)
    left = _contig(band[:, cx::-1], np, 1)
    right = _contig(band[:, cx:], np, 1)
    length = np.where(band[:, cx], left + right - 1, 0)
    need = max(_MIN_LINE_H * h, _MIN_LINE_WIDTH * (x2 - x1))
    rows = np.flatnonzero(length >= need)
    if rows.size == 0:
        return None, "no_line"
    limit = _thick_limit(h)
    lines = []
    for r0, r1 in _groups(rows, np, gap=1):
        thickness = r1 - r0 + 1
        if thickness > limit:
            continue          # толстая — контур, а не размерная линия
        best = r0 + int(np.argmax(length[r0:r1 + 1]))
        y = lo + (r0 + r1) / 2.0
        if y > y2:
            distance = y - y2
        elif y < y1:
            distance = y1 - y + _OTHER_SIDE_H * h
        else:
            distance = 0.0
        lines.append((distance, lo + r0, lo + r1, cx - int(left[best]) + 1, cx + int(right[best]) - 1))
    if not lines:
        return None, "thick_line"
    lines.sort()
    confirmed: List[_Found] = []
    why = "no_extension"
    for distance, top, bottom, a, b in lines[:_TRIES]:
        found, why = _extensions(sheet, canvas, top, bottom, a, b, cx, h, x2 - x1)
        if found is None:
            continue
        found.distance = distance
        confirmed.append(found)
        if len(confirmed) == 2:
            break
    if not confirmed:
        return None, why
    if (len(confirmed) == 2 and confirmed[1].distance - confirmed[0].distance < _UNIQUE_H * h
            and abs((confirmed[1].span_px or 0) - (confirmed[0].span_px or 0)) > _UNIQUE_H * h):
        return None, "ambiguous"
    found = confirmed[0]
    found.axis = axis
    if axis == AXIS_V:
        found.ends = [(p[1], p[0]) for p in found.ends]
        found.edges = [(p[1], p[0]) for p in found.edges]
    return found, "ok"


def _extensions(sheet: _Sheet, canvas, top: int, bottom: int, a: int, b: int, cx: int,
                h: float, width: float) -> Tuple[Optional[_Found], str]:
    """Выносные линии у концов участка размерной линии под надписью."""
    np = sheet.np
    rows_total, cols_total = canvas.shape
    over_min = max(_OVER_MIN_PX, _OVER_MIN_H * h)
    over_max = _OVER_MAX_H * h
    ext_min = max(_EXT_MIN_PX, _EXT_MIN_H * h)
    reach = int(math.ceil(over_max + max(ext_min, 0.5 * h))) + 2
    xa, xb = max(0, a - 3), min(cols_total, b + 4)
    up = _widen(canvas[max(0, top - reach):top, xa:xb][::-1], np)
    down = _widen(canvas[bottom + 1:bottom + 1 + reach, xa:xb], np)
    up_n = _contig(up, np, 0)
    down_n = _contig(down, np, 0)
    short = np.minimum(up_n, down_n)
    longer = np.maximum(up_n, down_n)
    crossing = (short >= over_min) & (short <= over_max) & (longer >= ext_min)
    groups = _groups(np.flatnonzero(crossing), np, gap=2)
    if not groups:
        return None, "no_extension"
    centres = []
    for g0, g1 in groups:
        mid = (g0 + g1) // 2
        centres.append((xa + (g0 + g1) / 2.0, 1 if down_n[mid] >= up_n[mid] else -1))
    lefts = [c for c in centres if c[0] <= cx]
    rights = [c for c in centres if c[0] >= cx]
    if not lefts or not rights:
        return None, "no_extension"
    (xl, dl), (xr, dr) = lefts[-1], rights[0]
    span = xr - xl
    if span < max(_EXT_MIN_PX, _MIN_SPAN_WIDTH * width):
        return None, "short_span"
    thickness = _stroke(canvas, np, top, bottom, int(round(xl)), int(round(xr)), h)
    y_mid = (top + bottom) / 2.0
    arrows = sum(_arrow_at(canvas, np, top, bottom, int(round(x)), h, thickness) for x in (xl, xr))
    limit = int(_TRACE_SHARE * rows_total)
    edges = []
    for x, direction in ((xl, dl), (xr, dr)):
        start = bottom + 1 if direction > 0 else top - 1
        edges.append((x, _trace(canvas, np, int(round(x)), start, direction, limit, h, thickness)))
    found = _Found(
        kind=KIND_LINEAR, ends=[(xl, y_mid), (xr, y_mid)], edges=edges,
        confidence=CONFIDENCE_LINEAR[arrows], span_px=int(round(span)), thickness=thickness, h=h,
    )
    found.extra["arrows"] = arrows
    return found, "ok"


def _stroke(canvas, np, top: int, bottom: int, a: int, b: int, h: float) -> int:
    """
    Толщина размерной линии: медиана поперечной толщины по её длине. Строки,
    где линия целиком сплошная, на скане её занижают (шум рвёт край).
    """
    rows_total = canvas.shape[0]
    cap = int(math.ceil(_ARROW_WIDE_H * h))
    r0, r1 = max(0, top - cap), min(rows_total, bottom + cap + 1)
    centre = (top + bottom) // 2 - r0
    block = canvas[r0:r1, a:b + 1:max(1, (b - a) // 200)]
    if block.size == 0:
        return bottom - top + 1
    up = _contig(block[centre::-1], np, 0)
    down = _contig(block[centre:], np, 0)
    width = np.where(block[centre], up + down - 1, 0)
    width = width[width > 0]
    return int(np.median(width)) if width.size else bottom - top + 1


def _arrow_at(canvas, np, top: int, bottom: int, x: int, h: float, thickness: int) -> int:
    """1 — у конца `x` линия утолщается стрелкой (с любой стороны выносной)."""
    rows_total, cols_total = canvas.shape
    cap = int(math.ceil(_ARROW_WIDE_H * h))
    r0, r1 = max(0, top - cap), min(rows_total, bottom + cap + 1)
    centre = (top + bottom) // 2 - r0
    reach = int(math.ceil(_ARROW_REACH_H * h))
    need = max(_ARROW_MIN_PX, _ARROW_MIN_H * h)
    wider = thickness + max(2, int(round(0.08 * h)))
    for c0, c1 in ((x + _ARROW_GAP_PX, x + reach), (x - reach, x - _ARROW_GAP_PX)):
        c0, c1 = max(0, c0), min(cols_total, c1 + 1)
        if c1 - c0 < 2:
            continue
        block = canvas[r0:r1, c0:c1]
        up = _contig(block[centre::-1], np, 0)
        down = _contig(block[centre:], np, 0)
        width = np.where(block[centre], up + down - 1, 0)
        if int(((width >= wider) & (width <= _ARROW_WIDE_H * h)).sum()) >= need:
            return 1
    return 0


def _trace(canvas, np, x: int, start: int, direction: int, limit: int, h: float, thickness: int) -> float:
    """
    Дальний конец выносной от строки `start` в сторону `direction`: там, где
    она встречает контур (поперечная линия или толстая кромка — штрих
    становится шире выносной), или где обрывается. Первые строки у размерной
    линии пропускаются: там рядом стрелка.
    """
    rows_total, cols_total = canvas.shape
    side = int(math.ceil(_ARROW_WIDE_H * h))
    c0, c1 = max(0, x - side), min(cols_total, x + side + 1)
    centre = x - c0
    if direction > 0:
        block = canvas[start:min(rows_total, start + limit), c0:c1]
    else:
        block = canvas[max(0, start - limit + 1):start + 1, c0:c1][::-1]
    if block.shape[0] == 0:
        return float(start)
    core = block[:, max(0, centre - 1):centre + 2].any(axis=1)
    length = _reach(core, np, _GAP_PX)
    left = _contig(block[:, centre::-1], np, 1)
    right = _contig(block[:, centre:], np, 1)
    width = np.where(block[:, centre], left + right - 1, 0)
    skip = int(math.ceil(0.3 * h)) + 2
    wide = np.flatnonzero(width[skip:length] >= thickness + 3)
    for r0, r1 in _groups(wide, np, gap=1):
        # Тонкая поперечная линия (чужая размерная, штриховка) — пересечение,
        # выносная идёт дальше; широкий штрих — контур, тут кромка элемента.
        if r1 - r0 + 1 > thickness + 1:
            length = skip + r0 + 1
            break
    return float(start + direction * max(0, length - 1))


# ===========================================================================
# Радиус и диаметр по окружности: выноска со стрелкой
# ===========================================================================

def _leader(sheet: _Sheet, box: Sequence[float], kind: str) -> Tuple[Optional[_Found], str]:
    """Прямая выноска от надписи наружу со стрелкой на дальнем конце."""
    np = sheet.np
    x1, y1, x2, y2 = box
    h = max(6.0, min(x2 - x1, y2 - y1))
    reach = min(_LEADER_REACH_D * sheet.long_side, _LEADER_REACH_H * h)
    wx0, wy0 = max(0, int(x1 - reach)), max(0, int(y1 - reach))
    wx1, wy1 = min(sheet.width, int(x2 + reach) + 1), min(sheet.height, int(y2 + reach) + 1)
    window = sheet.ink[wy0:wy1, wx0:wx1]
    if window.size == 0:
        return None, "outside"
    fat = _dilate(window, np, 1)
    rows, cols = window.shape
    margin = _RING_H * h
    rx0, ry0, rx1, ry1 = x1 - margin - wx0, y1 - margin - wy0, x2 + margin - wx0, y2 + margin - wy0
    ring = _ring(rx0, ry0, rx1, ry1)
    inside = [(x, y) for x, y in ring if 0 <= x < cols and 0 <= y < rows]
    if not inside:
        return None, "outside"
    dark = [fat[y, x] for x, y in inside]
    starts = _ring_groups(inside, dark)[:_LEADER_POINTS]
    if not starts:
        return None, "no_line"
    centre = ((x1 + x2) / 2 - wx0, (y1 + y2) / 2 - wy0)
    need = max(_LEADER_MIN_H * h, 2 * _EXT_MIN_PX)
    best: List[Tuple[float, Point, float]] = []
    for px, py in starts:
        out_x, out_y = px - centre[0], py - centre[1]
        angle, length = _best_ray(fat, np, px, py, out_x, out_y, 3.0 * h, reach)
        if angle is not None and length >= need:
            best.append((length, (px, py), angle))
    if not best:
        return None, "no_line"
    best.sort(reverse=True)
    for length, (px, py), angle in best[:_TRIES]:
        ux, uy = math.cos(angle), math.sin(angle)
        if not _ray_arrow(window, np, px, py, ux, uy, length, h):
            continue
        start = (px + wx0, py + wy0)
        end = (px + ux * length + wx0, py + uy * length + wy0)
        found = _Found(kind=kind, ends=[start, end], edges=[end], confidence=CONFIDENCE_LEADER, h=h)
        return found, "ok"
    return None, "no_arrow"


def _ring(x0: float, y0: float, x1: float, y1: float) -> List[Tuple[int, int]]:
    """Точки периметра рамки по часовой стрелке, через пиксель."""
    ix0, iy0, ix1, iy1 = int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1))
    points = [(x, iy0) for x in range(ix0, ix1)]
    points += [(ix1, y) for y in range(iy0, iy1)]
    points += [(x, iy1) for x in range(ix1, ix0, -1)]
    points += [(ix0, y) for y in range(iy1, iy0, -1)]
    return points


def _ring_groups(points: Sequence[Tuple[int, int]], dark: Sequence[bool]) -> List[Tuple[int, int]]:
    """Середины тёмных участков периметра — места, где линия выходит от надписи."""
    n = len(points)
    if not any(dark):
        return []
    if all(dark):
        return []
    shift = next(i for i in range(n) if not dark[i])
    out = []
    run: List[int] = []
    for k in range(1, n + 1):
        i = (shift + k) % n
        if dark[i]:
            run.append(i)
        elif run:
            out.append(points[run[len(run) // 2]])
            run = []
    if run:
        out.append(points[run[len(run) // 2]])
    return out


def _ray_lengths(fat, np, px: float, py: float, angles, steps):
    """Длина сплошного тёмного по лучам из (px, py) под углами `angles`."""
    rows, cols = fat.shape
    xs = np.rint(px + np.cos(angles)[:, None] * steps[None, :]).astype(np.int64)
    ys = np.rint(py + np.sin(angles)[:, None] * steps[None, :]).astype(np.int64)
    valid = (xs >= 0) & (xs < cols) & (ys >= 0) & (ys < rows)
    hit = np.zeros(xs.shape, dtype=bool)
    hit[valid] = fat[ys[valid], xs[valid]]
    return _contig(hit, np, 1) * float(steps[1] - steps[0] if len(steps) > 1 else 1.0)


def _best_ray(fat, np, px: float, py: float, out_x: float, out_y: float,
              probe: float, reach: float) -> Tuple[Optional[float], float]:
    """Направление самой длинной прямой от точки периметра наружу."""
    angles = np.radians(np.arange(0.0, 360.0, _LEADER_COARSE_DEG))
    outward = np.cos(angles) * out_x + np.sin(angles) * out_y > 0
    angles = angles[outward]
    if angles.size == 0:
        return None, 0.0
    near = _ray_lengths(fat, np, px, py, angles, np.arange(0.0, probe, 1.0))
    if near.size == 0 or near.max() < probe - 1.0:
        return None, float(near.max() if near.size else 0.0)
    top = np.flatnonzero(near >= near.max() - 1.0)
    best_angle, best_len = None, 0.0
    steps = np.arange(0.0, reach, 1.0)
    for index in _spread(top, np):
        fine = angles[index] + np.radians(np.arange(-_LEADER_COARSE_DEG, _LEADER_COARSE_DEG + 1e-9,
                                                    _LEADER_FINE_DEG))
        lengths = _ray_lengths(fat, np, px, py, fine, steps)
        k = int(np.argmax(lengths))
        if lengths[k] > best_len:
            best_angle, best_len = float(fine[k]), float(lengths[k])
    return best_angle, best_len


def _spread(indices, np) -> List[int]:
    """Из соседних углов с одинаковой длиной — середина каждой группы."""
    return [(g0 + g1) // 2 for g0, g1 in _groups(np.asarray(indices), np, gap=1)][:6]


def _ray_arrow(window, np, px: float, py: float, ux: float, uy: float, length: float, h: float) -> bool:
    """
    У дальнего конца луча — стрелка (утолщение против толщины середины
    выноски), а остриё упирается в толстый штрих (дугу контура). Стрелка
    размерной линии упирается в тонкую выносную — это не выноска радиуса.
    """
    rows, cols = window.shape
    nx, ny = -uy, ux

    def widths(distances, cap):
        offsets = np.arange(-cap, cap + 1)
        xs = np.rint(px + ux * distances[:, None] + nx * offsets[None, :]).astype(np.int64)
        ys = np.rint(py + uy * distances[:, None] + ny * offsets[None, :]).astype(np.int64)
        valid = (xs >= 0) & (xs < cols) & (ys >= 0) & (ys < rows)
        hit = np.zeros(xs.shape, dtype=bool)
        hit[valid] = window[ys[valid], xs[valid]]
        up = _contig(hit[:, cap::-1], np, 1)
        down = _contig(hit[:, cap:], np, 1)
        return np.where(hit[:, cap], up + down - 1, 0)

    cap = int(math.ceil(_ARROW_WIDE_H * h))
    middle = widths(np.arange(0.3 * length, 0.7 * length, 2.0), cap)
    middle = middle[middle > 0]
    if middle.size == 0:
        return False
    base = float(np.median(middle))
    if base > _thick_limit(h) + 1:
        return False      # толстый отрезок — контур, а не выноска
    near_end = widths(np.arange(max(_ARROW_GAP_PX, length - _ARROW_REACH_H * h), length - 2, 1.0), cap)
    wider = base + max(2, int(round(0.08 * h)))
    hits = int(((near_end >= wider) & (near_end <= _ARROW_WIDE_H * h)).sum())
    if hits < max(_ARROW_MIN_PX, _ARROW_MIN_H * h):
        return False
    tip_cap = max(3, int(round(_TIP_H * h)))
    tip = widths(np.arange(length - 0.5 * h, length + 3.0, 1.0), tip_cap)
    return int((tip >= 2 * tip_cap - 1).sum()) >= base + 1


# ===========================================================================
# Виды листа и роль размера
# ===========================================================================

class _Views:
    """
    Связные области толстых линий (контур детали). Тонкие линии — размерные,
    выносные, осевые, штриховка — снимаются эрозией не тоньше размерной
    линии этого листа; рамка формата со штампом отсекается по площади.
    """

    def __init__(self, sheet: _Sheet, thickness: int):
        np = sheet.np
        self.radius = max(1, (thickness + 1) // 2)
        core = _erode(sheet.ink, np, self.radius)
        f = max(1, int(math.ceil(sheet.long_side / _GRID_SIDE)))
        rows, cols = sheet.height // f, sheet.width // f
        grid = core[:rows * f, :cols * f].reshape(rows, f, cols, f).any(axis=(1, 3))
        grid = _dilate(grid, np, 1)
        self.cell = f
        self.boxes = _component_boxes(grid, np)
        self.area = float(rows * cols)

    def view_of(self, points: Sequence[Point], pad: float) -> Optional[Tuple[int, int, int, int]]:
        """Наибольший вид (рамка, px листа), в рамке которого все точки."""
        f = self.cell
        best, best_area = None, 0.0
        for gx0, gy0, gx1, gy1 in self.boxes:
            area = float((gx1 - gx0 + 1) * (gy1 - gy0 + 1))
            if area > _VIEW_MAX_AREA * self.area:
                continue
            # Клетка сетки — f px, и сетка расширена на клетку: край вида —
            # внутренняя граница крайних клеток.
            box = ((gx0 + 1) * f, (gy0 + 1) * f, gx1 * f - 1, gy1 * f - 1)
            if all(box[0] - pad <= x <= box[2] + pad and box[1] - pad <= y <= box[3] + pad
                   for x, y in points) and area > best_area:
                best, best_area = box, area
        return best


def _component_boxes(grid, np) -> List[Tuple[int, int, int, int]]:
    """Рамки связных областей сетки (8-связность), по отрезкам строк."""
    parent: List[int] = []
    boxes: List[List[int]] = []

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    previous: List[Tuple[int, int, int]] = []
    for y in range(grid.shape[0]):
        row = grid[y]
        if not row.any():
            previous = []
            continue
        edges = np.diff(np.concatenate(([0], row.view(np.int8), [0])))
        starts = np.flatnonzero(edges == 1)
        ends = np.flatnonzero(edges == -1) - 1
        current: List[Tuple[int, int, int]] = []
        j = 0
        for s, e in zip(starts.tolist(), ends.tolist()):
            index = len(parent)
            parent.append(index)
            boxes.append([s, y, e, y])
            while j < len(previous) and previous[j][1] < s - 1:
                j += 1
            k = j
            while k < len(previous) and previous[k][0] <= e + 1:
                a, b = find(previous[k][2]), find(index)
                if a != b:
                    parent[b] = a
                k += 1
            current.append((s, e, index))
        previous = current
    merged: Dict[int, List[int]] = {}
    for index, box in enumerate(boxes):
        root = find(index)
        have = merged.get(root)
        if have is None:
            merged[root] = list(box)
        else:
            have[0], have[1] = min(have[0], box[0]), min(have[1], box[1])
            have[2], have[3] = max(have[2], box[2]), max(have[3], box[3])
    return [tuple(b) for b in merged.values()]  # type: ignore[misc]


def _role(found: _Found, views: _Views, sheet: _Sheet) -> None:
    """Роль линейного размера по рамке его вида вдоль оси."""
    pad = max(_ROLE_TOL_H * found.h, 2 * views.cell)
    view = views.view_of(found.edges, pad)
    if view is None:
        return
    found.view = view
    tol = max(_ROLE_TOL_H * found.h, 2 * views.cell, _ROLE_TOL_D * sheet.long_side) + views.radius
    if found.axis == AXIS_H:
        a, b = sorted(p[0] for p in found.ends)
        lo, hi = view[0], view[2]
    else:
        a, b = sorted(p[1] for p in found.ends)
        lo, hi = view[1], view[3]
    on_edge = int(abs(a - lo) <= tol) + int(abs(b - hi) <= tol)
    found.role = (ROLE_INNER, ROLE_EDGE, ROLE_OVERALL)[on_edge]


def _scale(found: Dict[int, _Found], fields: Sequence[Dict[str, Any]]) -> int:
    """Согласие «значение / длина» внутри вида; возвращает число выпавших."""
    by_view: Dict[Tuple[int, int, int, int], List[Tuple[int, float]]] = {}
    for n, item in found.items():
        if item.axis is None or item.view is None or not item.span_px:
            continue
        number = number_of(fields[n].get("value"))
        if number is None:
            continue
        by_view.setdefault(item.view, []).append((n, number / float(item.span_px)))
    outliers = 0
    for rows in by_view.values():
        if len(rows) < _SCALE_MIN:
            continue
        ratios = sorted(r for _, r in rows)
        middle = ratios[len(ratios) // 2] if len(ratios) % 2 else (
            ratios[len(ratios) // 2 - 1] + ratios[len(ratios) // 2]) / 2
        for n, ratio in rows:
            ok = abs(ratio / middle - 1.0) <= _SCALE_TOL
            found[n].scale_ok = ok
            outliers += int(not ok)
    return outliers


# ===========================================================================
# Применение к полям листа
# ===========================================================================

def eligible(item: Dict[str, Any], forbidden: Sequence[Sequence[float]] = ()) -> bool:
    """Размер или радиус детали с настоящей рамкой, не в штампе и не в таблице."""
    from core.providers import dimension_links

    if str(item.get("category") or "") not in _CATEGORIES:
        return False
    if kind_of(item.get("value")) is None:
        return False
    return dimension_links.linkable(item, forbidden)


def measure(sheet: _Sheet, item: Dict[str, Any]) -> Tuple[Optional[_Found], str]:
    """Размерная линия (или выноска) одной надписи."""
    kind = kind_of(item.get("value"))
    raw = item.get("bbox") or ()
    box = (float(raw[0]) * sheet.width, float(raw[1]) * sheet.height,
           float(raw[2]) * sheet.width, float(raw[3]) * sheet.height)
    width, height = box[2] - box[0], box[3] - box[1]
    why = "no_line"
    if kind in (KIND_LINEAR, KIND_DIAMETER):
        axes = []
        if width >= 0.5 * height:
            axes.append(AXIS_H)
        if height >= 0.5 * width:
            axes.append(AXIS_V)
        hits = []
        for axis in axes:
            found, why = _linear(sheet, box, axis)
            if found is not None:
                found.kind = kind
                hits.append(found)
        if len(hits) == 2:
            hits.sort(key=lambda f: (-f.confidence, f.distance / max(f.h, 1.0)))
            if (hits[0].confidence == hits[1].confidence
                    and abs(hits[0].distance / hits[0].h - hits[1].distance / hits[1].h) < _UNIQUE_H):
                return None, "ambiguous"
        if hits:
            return hits[0], "ok"
    if kind in (KIND_RADIUS, KIND_DIAMETER):
        return _leader(sheet, box, kind)
    return None, why


def _label(found: _Found) -> str:
    if found.kind == KIND_RADIUS:
        return "радиус"
    if found.kind == KIND_DIAMETER:
        return "диаметр"
    along = "по горизонтали" if found.axis == AXIS_H else "по вертикали"
    if found.role == ROLE_OVERALL:
        return f"габаритный размер {along}"
    if found.role == ROLE_EDGE:
        return f"размер от края {along}"
    return f"размер {along}"


def _as_link(found: _Found, sheet: _Sheet, previous: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    w, h = float(sheet.width), float(sheet.height)

    def point(p: Point) -> List[float]:
        return [round(min(1.0, max(0.0, p[0] / w)), 4), round(min(1.0, max(0.0, p[1] / h)), 4)]

    points = list(found.ends) + list(found.edges)
    if found.axis is None:
        pad = 0.25 * found.h
        end = found.edges[0]
        target = [end[0] - pad, end[1] - pad, end[0] + pad, end[1] + pad]
    else:
        target = [min(p[0] for p in points), min(p[1] for p in points),
                  max(p[0] for p in points), max(p[1] for p in points)]
    link: Dict[str, Any] = {
        "target_label": _label(found),
        "target_bbox": [round(min(1.0, max(0.0, c / (w if i % 2 == 0 else h))), 4)
                        for i, c in enumerate(target)],
        "leader": [point(p) for p in found.ends],
        "method": METHOD_RASTER,
        "confidence": round(found.confidence, 2),
        "dimension_kind": found.kind,
        "edges": [point(p) for p in found.edges],
    }
    if found.axis is not None:
        link["axis"] = found.axis
        link["span_px"] = int(found.span_px or 0)
    if found.role is not None:
        link["extent_role"] = found.role
    if found.scale_ok is not None:
        link["scale_ok"] = found.scale_ok
    if isinstance(previous, dict) and previous.get("target_label"):
        # Подпись модели остаётся: «что измеряется» растр не скажет. Её
        # уверенность совпадением с геометрией не поднимается.
        link["target_label"] = previous["target_label"]
        link["confidence"] = round(min(link["confidence"], float(previous.get("confidence") or 0.0)), 2)
    return link


def apply(
    fields: List[Dict[str, Any]],
    image,
    *,
    forbidden: Sequence[Sequence[float]] = (),
) -> Optional[Dict[str, Any]]:
    """
    Ставит `link` с `method: "raster"` полям, у которых размерная линия
    подтверждена, и возвращает учёт листа. Флаг выключен — `None`, поля не
    трогаются. Векторная связь не переписывается.
    """
    if not enabled():
        return None
    np = _numpy()
    started = time.monotonic()
    picks = [n for n, item in enumerate(fields) if eligible(item, forbidden)]
    stats: Dict[str, Any] = {"dimensions": len(picks), "linked": 0, "linear": 0, "leader": 0,
                             "vector_kept": 0, "roles": {}, "scale_outliers": 0, "missed": {}}
    if not picks or image is None or np is None:
        return stats
    try:
        sheet = _Sheet(image, np)
    except Exception as exc:  # noqa: BLE001 — сбой маски не роняет лист
        logger.warning("Маска листа для размерных линий не построилась: %s", exc)
        stats["error"] = type(exc).__name__
        return stats
    found: Dict[int, _Found] = {}
    for n in picks:
        link = fields[n].get("link")
        if isinstance(link, dict) and link.get("method") == "vector":
            stats["vector_kept"] += 1
            continue
        try:
            hit, why = measure(sheet, fields[n])
        except Exception as exc:  # noqa: BLE001 — сбой одной надписи не роняет лист
            logger.warning("Размерная линия не разобрана: %s", exc)
            hit, why = None, "error"
        if hit is None:
            stats["missed"][why] = stats["missed"].get(why, 0) + 1
            continue
        found[n] = hit
    linear = [f for f in found.values() if f.axis is not None]
    if linear:
        try:
            thickness = sorted(f.thickness for f in linear)[len(linear) // 2]
            views = _Views(sheet, thickness)
            for item in linear:
                _role(item, views, sheet)
        except Exception as exc:  # noqa: BLE001 — без видов нет только ролей
            logger.warning("Виды листа для ролей размеров не найдены: %s", exc)
        stats["scale_outliers"] = _scale(found, fields)
    for n, item in found.items():
        fields[n]["link"] = _as_link(item, sheet, fields[n].get("link"))
        if item.scale_ok is False:
            fields[n]["needs_review"] = True
        stats["linked"] += 1
        stats["linear" if item.axis is not None else "leader"] += 1
        if item.role is not None:
            stats["roles"][item.role] = stats["roles"].get(item.role, 0) + 1
    stats["seconds"] = round(time.monotonic() - started, 3)
    return stats
