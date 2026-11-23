"""
Связь размера с тем, что он измеряет (`DRAWING_DIMENSION_LINKS`).

Зачем. У прочитанной величины есть рамка надписи (`bbox` поля), и модуль
`drawing_views` говорит, какому виду она принадлежит. Но «⌀20» на виде А —
это диаметр какого-то отверстия, а «505±10» — габаритная длина, и ни то ни
другое в данных не сказано: человек в просмотре распознавания видит число,
но не видит, к чему оно относится. Здесь у поля появляется ключ `link`:
что измеряется (короткая фраза по-русски), рамка измеряемого элемента и
линия от надписи к нему (концы размерной линии или выноски).

Два способа, по порядку.

1. **Геометрия векторного PDF.** Команды рисования страницы
   (`page.get_drawings()`) переводятся в координаты выправленного листа —
   те же доли, что у `bbox` полей. Рядом с надписью ищется размерная линия
   со стрелками на обоих концах (линейный размер: стрелки смотрят в разные
   стороны, выносные линии через острия дают рамку элемента) или линия-
   выноска с одной стрелкой или точкой на конце (цель — контур, которого
   касается остриё). Линия без стрелок связью не считается: на листе полно
   линий рядом с текстом, и ближняя из них — не обязательно размерная.
   Две равноценные размерные линии у одной надписи — тоже отказ.

2. **Модель на растре.** Для сканов и для величин, которые геометрия не
   связала, — один вызов Qwen-VL на лист: величины отмечены на картинке
   красными рамками с номерами, модель отвечает по каждому номеру фразой,
   рамкой элемента и концами линии. Ответ проверяется, а не принимается на
   веру: рамка внутри листа, не в штампе и не в таблице, не сама надпись и
   не дальше бюджета расстояния от неё; линия начинается у надписи и
   кончается у элемента. Не прошло проверку — связи нет: неверная связь
   утверждает, что размер относится к элементу, к которому он не относится,
   и это хуже её отсутствия.

Поля не переставляются и не удаляются; меняется только ключ `link`.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field as dataclass_field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

# Категории, у которых есть что связывать: размер, радиус, резьба,
# шероховатость и допуск формы указывают на поверхность, номер позиции —
# на деталь. Штамп, материал, примечания — величины листа, а не элемента.
LINKABLE = frozenset({"size", "radius", "thread", "roughness", "tolerance", "position"})

METHOD_VECTOR = "vector"
METHOD_VLM = "vlm"

# Области разметки, где размеров детали не бывает: надпись там — графа или
# ячейка таблицы, и цель внутри них — тоже не элемент детали.
_NOT_ON_PART = ("title_block", "table", "side_column")

# --- геометрия (доли высоты надписи h и длинной стороны листа D) ----------
_SEARCH_H = 2.0          # размерная линия не дальше 2h от рамки надписи
_SEARCH_D = 0.006        # и не меньше 0,6 % листа: рамка модели неточна
_MIN_SEGMENT_H = 0.6     # короче — штрих буквы или штриховки
_PARALLEL_COS = 0.98     # «параллельно надписи»: до ~11°
_ARROW_COS = 0.9         # стрелка вдоль линии: до ~25°
_PERPENDICULAR_COS = 0.2  # выносная линия поперёк размерной
_ARROW_MAX_D = 0.015     # стрелка не длиннее 1,5 % листа
_ARROW_MIN_D = 0.0008
_UNIQUE_H = 0.25         # две линии ближе этой разницы — неразличимы
_CHAIN_DEPTH = 3         # выноска: полка, наклонная, излом
_CHAIN_LIMIT = 12
_TARGET_MAX_D = 0.5      # контур крупнее половины листа — не элемент, а рамка
_CURVE_STEPS = 6

# --- проверка ответа модели (доли длинной стороны листа) ------------------
_VLM_TARGET_GAP = 0.2    # рамка элемента не дальше 20 % листа от надписи
_VLM_LEADER_START = 0.04  # линия начинается у надписи
_VLM_LEADER_END = 0.03   # и кончается у элемента
_VLM_TARGET_AREA = 0.25  # элемент не больше четверти листа
_VLM_OVERALL_AREA = 0.6  # габарит — деталь целиком, но не весь лист
_VLM_TARGET_MIN = 0.002
_VLM_SELF_SHARE = 0.8    # рамка «элемента» почти целиком на самой надписи
_LABEL_MAX = 60
_CYRILLIC = re.compile(r"[А-Яа-яЁё]")

VECTOR_CONFIDENCE_BOTH = 0.9
VECTOR_CONFIDENCE_ONE = 0.75
VECTOR_CONFIDENCE_LEADER = 0.7
VLM_CONFIDENCE_LEADER = 0.6
VLM_CONFIDENCE_BARE = 0.45

Point = Tuple[float, float]
Box = Tuple[float, float, float, float]


# ===========================================================================
# Флаг
# ===========================================================================

def enabled() -> bool:
    """Связывание включено флагом `DRAWING_DIMENSION_LINKS` (по умолчанию нет)."""
    try:
        from core.config import settings
    except Exception:  # noqa: BLE001 — без настроек связывание выключено
        return False
    return bool(getattr(settings, "DRAWING_DIMENSION_LINKS", False))


def _max_asked() -> int:
    try:
        from core.config import settings
    except Exception:  # noqa: BLE001
        return 60
    return int(getattr(settings, "DRAWING_DIMENSION_LINKS_MAX", 60) or 0)


# ===========================================================================
# Координаты: страница PDF -> выправленный лист
# ===========================================================================

@dataclass
class PageFrame:
    """
    Точка страницы PDF (неповёрнутые координаты `get_drawings`) -> пиксель
    выправленного листа, который видела модель.

    Цепочка та же, что у рендера и разметки: поворот страницы
    (`rotation_matrix`), масштаб рендера, наклон `angle` вокруг центра без
    расширения и поворот `turn` кратно 90° с расширением — оба против часовой,
    как у `PIL.Image.rotate` (`drawing_vision.analyse`, `reproduce`).
    """

    matrix: Tuple[float, float, float, float, float, float]
    page_size: Tuple[float, float]       # повёрнутая страница, пункты
    sheet_size: Tuple[int, int]          # выправленный лист, пиксели
    angle: float = 0.0
    turn: int = 0

    @classmethod
    def from_page(cls, page, sheet_size: Tuple[int, int], angle: float = 0.0, turn: int = 0) -> "PageFrame":
        m = page.rotation_matrix
        return cls(
            matrix=(m.a, m.b, m.c, m.d, m.e, m.f),
            page_size=(float(page.rect.width), float(page.rect.height)),
            sheet_size=(int(sheet_size[0]), int(sheet_size[1])),
            angle=float(angle or 0.0),
            turn=int(turn or 0) % 360,
        )

    def _rendered_size(self) -> Tuple[float, float]:
        width, height = self.sheet_size
        return (height, width) if self.turn in (90, 270) else (width, height)

    def __call__(self, x: float, y: float) -> Point:
        a, b, c, d, e, f = self.matrix
        px, py = a * x + c * y + e, b * x + d * y + f
        w0, h0 = self._rendered_size()
        px = px / self.page_size[0] * w0
        py = py / self.page_size[1] * h0
        if self.angle:
            cx, cy = w0 / 2.0, h0 / 2.0
            t = math.radians(self.angle)
            dx, dy = px - cx, py - cy
            px = cx + dx * math.cos(t) + dy * math.sin(t)
            py = cy - dx * math.sin(t) + dy * math.cos(t)
        if self.turn == 90:
            px, py = py, w0 - px
        elif self.turn == 180:
            px, py = w0 - px, h0 - py
        elif self.turn == 270:
            px, py = h0 - py, px
        return px, py


# ===========================================================================
# Геометрия листа
# ===========================================================================

@dataclass
class Arrow:
    """Остриё стрелки (или точка выноски: `direction` None)."""

    tip: Point
    direction: Optional[Point]
    path: int


@dataclass
class Geometry:
    """Отрезки и стрелки листа в пикселях выправленного листа."""

    size: Tuple[int, int]
    # (x1, y1, x2, y2, номер пути)
    segments: List[Tuple[float, float, float, float, int]] = dataclass_field(default_factory=list)
    arrows: List[Arrow] = dataclass_field(default_factory=list)
    path_boxes: Dict[int, Box] = dataclass_field(default_factory=dict)
    curved: set = dataclass_field(default_factory=set)
    _grid: Dict[Tuple[int, int], List[int]] = dataclass_field(default_factory=dict)
    _cell: float = 0.0

    @property
    def diagonal(self) -> float:
        return float(max(self.size))

    def index(self) -> None:
        """Сетка концов отрезков: соседей выноски ищем по ней, а не перебором."""
        self._cell = max(4.0, self.diagonal * 0.004)
        self._grid = {}
        for n, (x1, y1, x2, y2, _) in enumerate(self.segments):
            for x, y in ((x1, y1), (x2, y2)):
                self._grid.setdefault(self._key(x, y), []).append(n)

    def _key(self, x: float, y: float) -> Tuple[int, int]:
        return int(x // self._cell), int(y // self._cell)

    def ends_near(self, point: Point, eps: float) -> List[int]:
        """Отрезки, один из концов которых не дальше `eps` от точки."""
        if not self._grid:
            self.index()
        kx, ky = self._key(*point)
        reach = int(eps // self._cell) + 1
        found: List[int] = []
        for gx in range(kx - reach, kx + reach + 1):
            for gy in range(ky - reach, ky + reach + 1):
                for n in self._grid.get((gx, gy), ()):
                    x1, y1, x2, y2, _ = self.segments[n]
                    if min(_dist(point, (x1, y1)), _dist(point, (x2, y2))) <= eps:
                        found.append(n)
        return sorted(set(found))


def geometry_from_page(page, frame: PageFrame) -> Optional[Geometry]:
    """
    Команды рисования страницы -> отрезки и стрелки листа. `None` — на
    странице нет векторной графики (скан в PDF): связывать геометрией нечего.
    """
    try:
        drawings = page.get_drawings()
    except Exception as exc:  # noqa: BLE001 — битая страница не роняет лист
        logger.warning("Команды рисования страницы не прочитались: %s", exc)
        return None
    return geometry_from_paths(drawings, frame)


def geometry_from_paths(drawings: Sequence[Dict[str, Any]], frame: Callable[[float, float], Point],
                        size: Optional[Tuple[int, int]] = None) -> Optional[Geometry]:
    """Пути `get_drawings` (или такие же словари в тестах) -> `Geometry`."""
    size = size or getattr(frame, "sheet_size", None)
    if not drawings or not size:
        return None
    geometry = Geometry(size=(int(size[0]), int(size[1])))
    longest = geometry.diagonal
    for number, path in enumerate(drawings):
        points: List[Point] = []
        pieces: List[Tuple[Point, Point]] = []
        curved = False
        for item in path.get("items") or ():
            kind = item[0]
            if kind == "l":
                pieces.append((frame(*_xy(item[1])), frame(*_xy(item[2]))))
            elif kind == "c":
                # Дуга — несколькими хордами: одна хорда на четверть
                # окружности отстоит от дуги на 0,3 радиуса, и остриё
                # выноски на контуре отверстия до неё не дотянулось бы.
                curved = True
                bezier = [_xy(item[k]) for k in range(1, 5)]
                samples = [frame(*_bezier(bezier, step / _CURVE_STEPS)) for step in range(_CURVE_STEPS + 1)]
                pieces.extend(zip(samples, samples[1:]))
            elif kind == "re":
                x0, y0, x1, y1 = _rect(item[1])
                corners = [frame(x0, y0), frame(x1, y0), frame(x1, y1), frame(x0, y1)]
                pieces.extend(zip(corners, corners[1:] + corners[:1]))
            elif kind == "qu":
                quad = item[1]
                corners = [frame(*_xy(quad.ul)), frame(*_xy(quad.ur)),
                           frame(*_xy(quad.lr)), frame(*_xy(quad.ll))]
                pieces.extend(zip(corners, corners[1:] + corners[:1]))
        if not pieces:
            continue
        for start, end in pieces:
            points.extend((start, end))
        xs, ys = [p[0] for p in points], [p[1] for p in points]
        box = (min(xs), min(ys), max(xs), max(ys))
        geometry.path_boxes[number] = box
        if curved:
            geometry.curved.add(number)
        arrow = _arrow(path, pieces, box, curved, longest, number)
        if arrow is not None:
            geometry.arrows.append(arrow)
            continue
        for start, end in pieces:
            if _dist(start, end) > 0.0:
                geometry.segments.append((start[0], start[1], end[0], end[1], number))
    geometry.index()
    return geometry


def _arrow(path: Dict[str, Any], pieces: Sequence[Tuple[Point, Point]], box: Box,
           curved: bool, longest: float, number: int) -> Optional[Arrow]:
    """Стрелка: закрашенный узкий треугольник, открытая «галочка» или точка."""
    side = max(box[2] - box[0], box[3] - box[1])
    if not (_ARROW_MIN_D * longest <= side <= _ARROW_MAX_D * longest):
        return None
    filled = path.get("fill") is not None
    vertices = _distinct([p for piece in pieces for p in piece], max(0.5, side * 0.05))
    if filled and curved:
        # Точка на конце выноски (выноска внутрь контура, ГОСТ 2.316).
        width, height = box[2] - box[0], box[3] - box[1]
        if min(width, height) >= 0.6 * max(width, height):
            return Arrow(tip=((box[0] + box[2]) / 2, (box[1] + box[3]) / 2), direction=None, path=number)
        return None
    if filled and len(vertices) == 3:
        return _triangle_tip(vertices, number)
    if not filled and len(pieces) == 2:
        # Открытая стрелка — два коротких штриха из общей точки.
        (a1, a2), (b1, b2) = pieces
        for tip, one, two in ((a1, a2, b2), (a1, a2, b1), (a2, a1, b2), (a2, a1, b1)):
            other = b1 if two is b2 else b2
            if _dist(tip, other) > max(0.5, side * 0.05):
                continue
            u, v = _unit(one, tip), _unit(two, tip)
            if u is None or v is None:
                return None
            spread = math.degrees(math.acos(max(-1.0, min(1.0, u[0] * v[0] + u[1] * v[1]))))
            if 8.0 <= spread <= 50.0:
                direction = _normalized((u[0] + v[0], u[1] + v[1]))
                return Arrow(tip=tip, direction=direction, path=number) if direction else None
    return None


def _triangle_tip(vertices: Sequence[Point], number: int) -> Optional[Arrow]:
    """Остриё узкого треугольника — вершина против самой короткой стороны."""
    sides = []
    for n in range(3):
        a, b = vertices[(n + 1) % 3], vertices[(n + 2) % 3]
        sides.append((_dist(a, b), n))
    sides.sort()
    shortest, tip_index = sides[0]
    if shortest > 0.7 * sides[1][0]:
        return None  # равносторонний треугольник — значок, а не стрелка
    tip = vertices[tip_index]
    base = [vertices[n] for n in range(3) if n != tip_index]
    middle = ((base[0][0] + base[1][0]) / 2, (base[0][1] + base[1][1]) / 2)
    direction = _unit(middle, tip)
    return Arrow(tip=tip, direction=direction, path=number) if direction else None


# ===========================================================================
# Связь по геометрии
# ===========================================================================

def vector_link(
    item: Dict[str, Any],
    geometry: Geometry,
    views: Sequence[Sequence[float]] = (),
) -> Tuple[Optional[Dict[str, Any]], str]:
    """Связь одной величины по геометрии листа и причина, если связи нет."""
    width, height = geometry.size
    frac = _field_box(item)
    if frac is None:
        return None, "no_coordinates"
    box = (frac[0] * width, frac[1] * height, frac[2] * width, frac[3] * height)
    text_w, text_h = box[2] - box[0], box[3] - box[1]
    h = max(2.0, min(text_w, text_h))
    axis = (1.0, 0.0) if text_w >= text_h else (0.0, 1.0)
    search = max(_SEARCH_H * h, _SEARCH_D * geometry.diagonal)
    eps = max(2.0, 0.12 * h)
    inner = _pad(box, 0.1 * h)

    near: List[Tuple[float, int]] = []
    for n, (x1, y1, x2, y2, _) in enumerate(geometry.segments):
        if _dist((x1, y1), (x2, y2)) < _MIN_SEGMENT_H * h:
            continue
        if _inside((x1, y1), inner) and _inside((x2, y2), inner):
            continue  # штрих самой надписи (текст отрезками)
        gap = _segment_box_gap((x1, y1), (x2, y2), box)
        if gap <= search:
            near.append((gap, n))
    if not near:
        return None, "no_lines"
    near.sort()

    centre = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
    linear: List[Tuple[float, int, Arrow, Arrow]] = []
    for gap, n in near:
        segment = geometry.segments[n]
        direction = _unit((segment[0], segment[1]), (segment[2], segment[3]))
        if direction is None or abs(direction[0] * axis[0] + direction[1] * axis[1]) < _PARALLEL_COS:
            continue
        pair = _arrow_pair(geometry, segment, direction, centre, eps)
        if pair is not None:
            linear.append((gap, n, pair[0], pair[1]))
    if linear:
        linear.sort(key=lambda row: row[0])
        best = linear[0]
        for other in linear[1:]:
            if other[0] - best[0] >= _UNIQUE_H * h:
                break
            if not _same_tips((best[2], best[3]), (other[2], other[3]), eps):
                return None, "ambiguous"
        return _linear_link(item, geometry, best, eps, views), ""

    leaders: List[Tuple[float, Arrow, Point, frozenset]] = []
    for gap, n in near[:_CHAIN_LIMIT]:
        found = _leader_tip(geometry, n, box, h, eps)
        if found is not None:
            leaders.append((gap, *found))
    if not leaders:
        return None, "no_arrows"
    leaders.sort(key=lambda row: row[0])
    best_leader = leaders[0]
    for other in leaders[1:]:
        if other[0] - best_leader[0] >= _UNIQUE_H * h:
            break
        if _dist(other[1].tip, best_leader[1].tip) > eps:
            return None, "ambiguous"
    return _leader_link(item, geometry, best_leader, h, eps), ""


def _arrow_pair(geometry: Geometry, segment, direction: Point, centre: Point,
                eps: float) -> Optional[Tuple[Arrow, Arrow]]:
    """Две стрелки размерной линии, между которыми (или ближе всех к которым) стоит надпись."""
    start = (segment[0], segment[1])
    on_line: List[Tuple[float, Arrow]] = []
    for arrow in geometry.arrows:
        if arrow.direction is None:
            continue
        if _point_segment_gap(arrow.tip, start, (segment[2], segment[3])) > eps:
            continue
        if abs(arrow.direction[0] * direction[0] + arrow.direction[1] * direction[1]) < _ARROW_COS:
            continue
        t = (arrow.tip[0] - start[0]) * direction[0] + (arrow.tip[1] - start[1]) * direction[1]
        on_line.append((t, arrow))
    if len(on_line) < 2:
        return None
    on_line.sort(key=lambda row: row[0])
    middle = (centre[0] - start[0]) * direction[0] + (centre[1] - start[1]) * direction[1]
    pairs = []
    for (t1, a1), (t2, a2) in zip(on_line, on_line[1:]):
        # Стрелки пары смотрят в разные стороны: внутрь (обычный размер)
        # или наружу (малый размер со стрелками снаружи).
        if a1.direction[0] * a2.direction[0] + a1.direction[1] * a2.direction[1] > -_ARROW_COS:
            continue
        if t2 - t1 <= eps:
            continue
        outside = 0.0 if t1 <= middle <= t2 else min(abs(middle - t1), abs(middle - t2))
        pairs.append((outside, a1, a2))
    if not pairs:
        return None
    pairs.sort(key=lambda row: row[0])
    if len(pairs) > 1 and pairs[0][0] == pairs[1][0]:
        return None
    return pairs[0][1], pairs[0][2]


def _linear_link(item, geometry: Geometry, best, eps: float, views) -> Dict[str, Any]:
    _, n, first, second = best
    segment = geometry.segments[n]
    direction = _unit((segment[0], segment[1]), (segment[2], segment[3])) or (1.0, 0.0)
    points = [first.tip, second.tip]
    found_sides = 0
    for tip in (first.tip, second.tip):
        touched = False
        for m, other in enumerate(geometry.segments):
            if m == n:
                continue
            other_dir = _unit((other[0], other[1]), (other[2], other[3]))
            if other_dir is None:
                continue
            if abs(other_dir[0] * direction[0] + other_dir[1] * direction[1]) > _PERPENDICULAR_COS:
                continue
            if _point_segment_gap(tip, (other[0], other[1]), (other[2], other[3])) <= eps:
                points.extend(((other[0], other[1]), (other[2], other[3])))
                touched = True
        found_sides += int(touched)
    target = _pad(_bounds(points), eps)
    width, height = geometry.size
    label = _linear_label(item, direction, target, width, height, views)
    return _link(
        label, _to_frac(target, geometry.size),
        [_point_frac(first.tip, geometry.size), _point_frac(second.tip, geometry.size)],
        METHOD_VECTOR,
        VECTOR_CONFIDENCE_BOTH if found_sides == 2 else VECTOR_CONFIDENCE_ONE,
    )


def _leader_tip(geometry: Geometry, start: int, box: Box, h: float,
                eps: float) -> Optional[Tuple[Arrow, Point, frozenset]]:
    """Выноска от линии у надписи: единственное остриё на конце цепочки отрезков."""
    chain = {start}
    frontier = [start]
    for _ in range(_CHAIN_DEPTH):
        grown: List[int] = []
        for n in frontier:
            x1, y1, x2, y2, _ = geometry.segments[n]
            for end in ((x1, y1), (x2, y2)):
                # Остриё кончает выноску: за ним — контур цели, а не выноска.
                if any(_dist(end, arrow.tip) <= eps for arrow in geometry.arrows):
                    continue
                for m in geometry.ends_near(end, eps):
                    if m in chain or len(chain) >= _CHAIN_LIMIT:
                        continue
                    sx1, sy1, sx2, sy2, path = geometry.segments[m]
                    # Выноска — прямые отрезки; дуга — уже контур.
                    if path in geometry.curved or _dist((sx1, sy1), (sx2, sy2)) < 0.3 * h:
                        continue
                    chain.add(m)
                    grown.append(m)
        frontier = grown
        if not frontier:
            break
    if len(chain) >= _CHAIN_LIMIT:
        return None  # цепочка разрослась — это контур, а не выноска
    tips: List[Tuple[Arrow, Point]] = []
    for arrow in geometry.arrows:
        if _gap_to_box(arrow.tip, box) < 0.5 * h:
            continue
        for n in chain:
            x1, y1, x2, y2, _ = geometry.segments[n]
            for end, other in (((x1, y1), (x2, y2)), ((x2, y2), (x1, y1))):
                if _dist(arrow.tip, end) > eps:
                    continue
                if arrow.direction is not None:
                    outward = _unit(other, end)
                    if outward is None or outward[0] * arrow.direction[0] + outward[1] * arrow.direction[1] < _ARROW_COS:
                        continue
                near_end = min(((x1, y1), (x2, y2)), key=lambda p: _gap_to_box(p, box))
                tips.append((arrow, near_end))
                break
            else:
                continue
            break
    distinct: List[Tuple[Arrow, Point]] = []
    for arrow, near_end in tips:
        if all(_dist(arrow.tip, kept.tip) > eps for kept, _ in distinct):
            distinct.append((arrow, near_end))
    if len(distinct) != 1:
        return None
    arrow, _ = distinct[0]
    start_point = _nearest_on_chain(geometry, chain, box)
    return arrow, start_point, frozenset(geometry.segments[n][4] for n in chain)


def _nearest_on_chain(geometry: Geometry, chain: Iterable[int], box: Box) -> Point:
    best: Optional[Tuple[float, Point]] = None
    for n in chain:
        x1, y1, x2, y2, _ = geometry.segments[n]
        for point in ((x1, y1), (x2, y2)):
            gap = _gap_to_box(point, box)
            if best is None or gap < best[0]:
                best = (gap, point)
    return best[1] if best else (box[0], box[3])


def _leader_link(item, geometry: Geometry, best, h: float, eps: float) -> Dict[str, Any]:
    _, arrow, start, chain_paths = best
    tip = arrow.tip
    chosen: Optional[Tuple[float, int]] = None
    for x1, y1, x2, y2, path in geometry.segments:
        # Сама выноска целью не бывает: цель — контур, которого касается остриё.
        if path == arrow.path or path in chain_paths:
            continue
        if _point_segment_gap(tip, (x1, y1), (x2, y2)) > eps:
            continue
        box = geometry.path_boxes.get(path)
        if box is None:
            continue
        side = max(box[2] - box[0], box[3] - box[1])
        if side < 0.5 * h or side > _TARGET_MAX_D * geometry.diagonal:
            continue
        area = (box[2] - box[0]) * (box[3] - box[1])
        if chosen is None or area < chosen[0]:
            chosen = (area, path)
    if chosen is not None:
        target = _pad(geometry.path_boxes[chosen[1]], eps)
        curved = chosen[1] in geometry.curved
    else:
        target = (tip[0] - h, tip[1] - h, tip[0] + h, tip[1] + h)
        curved = False
    label = _leader_label(item, curved)
    return _link(
        label, _to_frac(target, geometry.size),
        [_point_frac(start, geometry.size), _point_frac(tip, geometry.size)],
        METHOD_VECTOR, VECTOR_CONFIDENCE_LEADER,
    )


# ===========================================================================
# Подпись цели
# ===========================================================================

def _value(item: Dict[str, Any]) -> str:
    return str(item.get("value") or "").strip()


def _is_diameter(value: str) -> bool:
    return value[:1] in ("⌀", "Ø", "ø", "∅", "Φ", "φ")


def _linear_label(item, direction: Point, target: Box, width: int, height: int,
                  views: Sequence[Sequence[float]]) -> str:
    value = _value(item)
    category = str(item.get("category") or "")
    if _is_diameter(value):
        return "диаметр"
    if category == "radius" or value[:1] in ("R", "r"):
        return "радиус"
    if "°" in value:
        return "угол"
    horizontal = abs(direction[0]) >= abs(direction[1])
    span = (target[2] - target[0]) / width if horizontal else (target[3] - target[1]) / height
    for view in views:
        if len(view) != 4:
            continue
        # Размер почти во всю ширину (высоту) вида — габаритный.
        extent = (view[2] - view[0]) if horizontal else (view[3] - view[1])
        centre = ((target[0] + target[2]) / 2 / width, (target[1] + target[3]) / 2 / height)
        inside = view[0] - 0.05 <= centre[0] <= view[2] + 0.05 and view[1] - 0.05 <= centre[1] <= view[3] + 0.05
        if inside and extent > 0 and span >= 0.85 * extent:
            return "габаритная длина" if horizontal else "габаритная высота"
    if abs(direction[0]) > 0.98 or abs(direction[1]) > 0.98:
        return "размер по горизонтали" if horizontal else "размер по вертикали"
    return "наклонный размер"


def _leader_label(item, curved: bool) -> str:
    value = _value(item)
    category = str(item.get("category") or "")
    if _is_diameter(value):
        return "диаметр отверстия" if curved else "диаметр"
    if category == "radius" or value[:1] in ("R", "r"):
        return "радиус скругления" if curved else "радиус"
    if category == "thread":
        return "резьба"
    if category == "roughness":
        return "шероховатость поверхности"
    if category == "tolerance":
        return "поверхность с допуском"
    if category == "position":
        return "деталь позиции"
    return "элемент по выноске"


# ===========================================================================
# Связь по модели
# ===========================================================================

_PROMPT = """На чертеже красными рамками с номерами отмечены размеры и другие подписанные величины:
{items}

Для каждой величины найди на чертеже, к чему она относится: какое отверстие, поверхность, кромку,
длину или деталь она измеряет или указывает. Иди по размерной линии и выносным линиям до стрелок
или по линии-выноске до её острия.

Ответь только JSON:
{{"links": [{{"id": 1, "target": "диаметр отверстия", "target_bbox": [x1, y1, x2, y2], "leader": [[x, y], [x, y]]}}]}}

- target — коротко по-русски, что измеряется: «диаметр отверстия», «габаритная длина»,
  «толщина стенки», «радиус скругления», «шероховатость поверхности», «деталь позиции».
- target_bbox — рамка измеряемого элемента на листе: отверстия, кромки, участка контура или
  детали. Это НЕ рамка самой надписи и не красная рамка: такой ответ не принимается.
  Для длины, ширины или высоты — рамка измеряемого участка детали от выносной линии до выносной.
- leader — концы размерной линии или выноски: первая точка у надписи, последняя — у элемента
  (остриё стрелки). Для размера со стрелками на обоих концах — два острия.
- Координаты — целые тысячные доли стороны листа от 0 до 1000: x слева, y сверху; рамка — лево, верх, право, низ.
- Если не видно, к чему относится величина, пропусти её номер. Не угадывай."""


def vlm_prompt(asked: Sequence[Tuple[int, Dict[str, Any]]]) -> str:
    """
    Текст запроса: номер и надпись каждой величины.

    Координат надписи в запросе нет намеренно: на листе крышки модель,
    получив рамки надписей числами, вернула их же как рамки целей у всех
    12 величин (проверка отбросила все). Где надпись — показывает красная
    рамка с номером на картинке.
    """
    rows = [f"{number}: «{_value(item)}»" for number, item in asked]
    return _PROMPT.format(items="\n".join(rows))


def marked_image(image, asked: Sequence[Tuple[int, Dict[str, Any]]]):
    """Копия листа с красными рамками и номерами отмеченных величин."""
    from PIL import ImageDraw, ImageFont

    marked = image.convert("RGB").copy()
    draw = ImageDraw.Draw(marked)
    width, height = marked.size
    stroke = max(2, int(max(width, height) / 900))
    size = max(14, int(max(width, height) / 90))
    try:
        font = ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1: шрифт без размера
        font = ImageFont.load_default()
    for number, item in asked:
        box = _field_box(item)
        if box is None:
            continue
        x0, y0, x1, y1 = box[0] * width, box[1] * height, box[2] * width, box[3] * height
        draw.rectangle((x0 - stroke, y0 - stroke, x1 + stroke, y1 + stroke), outline=(230, 0, 0), width=stroke)
        draw.text((x0, max(0, y0 - size - stroke)), str(number), fill=(230, 0, 0), font=font)
    return marked


def validate_vlm(
    data: Any,
    asked: Sequence[Tuple[int, Dict[str, Any]]],
    size: Tuple[int, int],
    forbidden: Sequence[Sequence[float]] = (),
) -> Tuple[Dict[int, Dict[str, Any]], Dict[str, int]]:
    """
    Ответ модели -> {номер: связь}, только прошедшие проверку, и счёт
    отказов по причинам. Любое сомнение — отказ: связь не угадывается.
    """
    dropped: Dict[str, int] = {}

    def drop(reason: str) -> None:
        dropped[reason] = dropped.get(reason, 0) + 1

    by_number = {number: item for number, item in asked}
    rows = data.get("links") if isinstance(data, dict) else None
    if not isinstance(rows, list):
        return {}, ({"no_answer": 1} if asked else {})
    width, height = size
    longest = float(max(width, height)) or 1.0
    out: Dict[int, Dict[str, Any]] = {}
    seen: set = set()
    for row in rows:
        if not isinstance(row, dict):
            drop("malformed")
            continue
        try:
            number = int(row.get("id"))
        except (TypeError, ValueError):
            drop("malformed")
            continue
        if number not in by_number:
            drop("unknown_id")
            continue
        if number in seen:
            # Два ответа на один номер: какой из них верный — не знаем.
            out.pop(number, None)
            drop("duplicate_id")
            continue
        seen.add(number)
        item = by_number[number]
        label = str(row.get("target") or "").strip()
        if not label or len(label) > _LABEL_MAX or not _CYRILLIC.search(label):
            drop("bad_label")
            continue
        target = _model_box(row.get("target_bbox"), span_ok=True)
        if target is None:
            drop("bad_bbox")
            continue
        area = (target[2] - target[0]) * (target[3] - target[1])
        # Габарит измеряет деталь целиком — его цель вправе быть крупнее.
        largest = _VLM_OVERALL_AREA if label.lower().startswith("габарит") else _VLM_TARGET_AREA
        if area > largest or max(target[2] - target[0], target[3] - target[1]) < _VLM_TARGET_MIN:
            drop("bad_bbox")
            continue
        own = _field_box(item)
        if own is None:
            drop("no_coordinates")
            continue
        own_px = _px(own, size)
        target_px = _px(target, size)
        if _overlap(target_px, own_px) >= _VLM_SELF_SHARE * _area(target_px):
            drop("target_is_label")
            continue
        if _box_gap(own_px, target_px) > _VLM_TARGET_GAP * longest:
            drop("too_far")
            continue
        centre = ((target[0] + target[2]) / 2, (target[1] + target[3]) / 2)
        if any(len(region) == 4 and region[0] <= centre[0] <= region[2] and region[1] <= centre[1] <= region[3]
               for region in forbidden):
            drop("target_in_title_block")
            continue
        leader_raw = row.get("leader")
        leader: List[List[float]] = []
        if leader_raw not in (None, [], ""):
            leader = _model_points(leader_raw)
            if len(leader) < 2:
                drop("bad_leader")
                continue
            first = (leader[0][0] * width, leader[0][1] * height)
            last = (leader[-1][0] * width, leader[-1][1] * height)
            # Острия двусторонней размерной линии оба у элемента, а надпись
            # между ними: тогда «у надписи» — ближний к ней конец.
            start_gap = min(_gap_to_box(first, own_px), _gap_to_box(last, own_px))
            end_gap = min(_gap_to_box(last, target_px), _gap_to_box(first, target_px))
            if start_gap > _VLM_LEADER_START * longest or end_gap > _VLM_LEADER_END * longest:
                drop("leader_mismatch")
                continue
        out[number] = _link(
            label, list(target), leader, METHOD_VLM,
            VLM_CONFIDENCE_LEADER if leader else VLM_CONFIDENCE_BARE,
        )
    return out, dropped


# ===========================================================================
# Применение к полям листа
# ===========================================================================

def linkable(item: Dict[str, Any], forbidden: Sequence[Sequence[float]] = ()) -> bool:
    """Величина детали с настоящими координатами, не из штампа и не из таблицы."""
    if str(item.get("category") or "") not in LINKABLE:
        return False
    box = _field_box(item)
    if box is None:
        return False
    centre = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
    return not any(len(region) == 4 and region[0] <= centre[0] <= region[2] and region[1] <= centre[1] <= region[3]
                   for region in forbidden)


def forbidden_regions(layout) -> List[List[float]]:
    """Рамки штампа, таблиц и боковых граф листа в долях."""
    out: List[List[float]] = []
    for region in getattr(layout, "regions", None) or ():
        if getattr(region, "kind", "") in _NOT_ON_PART and len(getattr(region, "bbox", ()) or ()) == 4:
            out.append([float(c) for c in region.bbox])
    return out


def apply(
    fields: List[Dict[str, Any]],
    *,
    image=None,
    forbidden: Sequence[Sequence[float]] = (),
    geometry: Optional[Geometry] = None,
    ask: Optional[Callable[[Any, str, int], Any]] = None,
    views: Sequence[Sequence[float]] = (),
    has_time: Callable[[], bool] = lambda: True,
) -> Dict[str, Any]:
    """
    Ставит `link` полям листа, у которых связь нашлась, и возвращает учёт
    для `structured_payload["dimension_links"]`.

    `ask(image, prompt, count)` — вызов модели (ответ — разобранный JSON или
    None); без него и без картинки модельного шага нет. `has_time` — хватает
    ли бюджета листа на вызов.
    """
    for item in fields:
        item.pop("link", None)
    picks = [n for n, item in enumerate(fields) if linkable(item, forbidden)]
    stats: Dict[str, Any] = {
        "dimensions": len(picks), "linked_vector": 0, "linked_vlm": 0,
        "asked_vlm": 0, "not_asked": 0, "dropped": {},
        "geometry": geometry is not None,
    }

    def drop(reason: str, count: int = 1) -> None:
        stats["dropped"][reason] = stats["dropped"].get(reason, 0) + count

    rest: List[int] = []
    for n in picks:
        if geometry is None:
            rest.append(n)
            continue
        try:
            link, why = vector_link(fields[n], geometry, views)
        except Exception as exc:  # noqa: BLE001 — сбой геометрии не роняет лист
            logger.warning("Связь размера по геометрии не построилась: %s", exc)
            link, why = None, "vector_error"
        if link is not None:
            fields[n]["link"] = link
            stats["linked_vector"] += 1
        else:
            stats.setdefault("vector_missed", {})
            stats["vector_missed"][why] = stats["vector_missed"].get(why, 0) + 1
            rest.append(n)

    if rest and ask is not None and image is not None:
        limit = _max_asked()
        asked_rows = rest[:limit] if limit > 0 else []
        stats["not_asked"] = len(rest) - len(asked_rows)
        if asked_rows and has_time():
            asked = [(number + 1, fields[n]) for number, n in enumerate(asked_rows)]
            stats["asked_vlm"] = len(asked)
            data = None
            try:
                data = ask(marked_image(image, asked), vlm_prompt(asked), len(asked))
            except Exception as exc:  # noqa: BLE001 — сбой вызова не роняет лист
                if _stops_the_task(exc):
                    raise
                logger.warning("Связь размеров моделью не получена: %s", exc)
            found, dropped = validate_vlm(data, asked, tuple(image.size), forbidden)
            for reason, count in dropped.items():
                drop(reason, count)
            for number, n in enumerate(asked_rows, start=1):
                if number in found:
                    fields[n]["link"] = found[number]
                    stats["linked_vlm"] += 1
            stats["vlm_unlinked"] = len(asked) - len(found)
        elif asked_rows:
            stats["not_asked"] = len(rest)
            drop("no_time", len(asked_rows))
    elif rest:
        stats["not_asked"] = len(rest)
    stats["linked"] = stats["linked_vector"] + stats["linked_vlm"]
    return stats


# ===========================================================================
# Мелочи
# ===========================================================================

def _stops_the_task(exc: BaseException) -> bool:
    """Обрыв задачи или отмена листа — не сбой связи: их не глотаем."""
    try:
        from celery.exceptions import SoftTimeLimitExceeded
    except ImportError:  # pragma: no cover
        SoftTimeLimitExceeded = ()  # type: ignore[assignment]
    from core import deadline

    return isinstance(exc, (deadline.Cancelled,) + ((SoftTimeLimitExceeded,) if SoftTimeLimitExceeded else ()))


def _link(label: str, target: Sequence[float], leader: Sequence[Sequence[float]],
          method: str, confidence: float) -> Dict[str, Any]:
    return {
        "target_label": label,
        "target_bbox": [round(max(0.0, min(1.0, float(c))), 4) for c in target],
        "leader": [[round(max(0.0, min(1.0, float(p[0]))), 4), round(max(0.0, min(1.0, float(p[1]))), 4)]
                   for p in leader],
        "method": method,
        "confidence": round(float(confidence), 2),
    }


def _field_box(item: Dict[str, Any]) -> Optional[List[float]]:
    """Рамка поля в долях; рамка «весь лист» (нет координат) — None."""
    raw = item.get("bbox")
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        return None
    try:
        box = [float(c) for c in raw]
    except (TypeError, ValueError):
        return None
    if box == [0.0, 0.0, 1.0, 1.0] or max(box) > 1.5 or box[2] <= box[0] or box[3] <= box[1]:
        return None
    return box


def _model_box(raw: Any, span_ok: bool = False) -> Optional[List[float]]:
    """
    Рамка от модели: тысячные (или доли), строго внутри листа. `span_ok` —
    рамка нулевой толщины (длина отрезком «от выносной до выносной»)
    принимается и получает толщину `_VLM_TARGET_MIN`.
    """
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        return None
    try:
        box = [float(c) for c in raw]
    except (TypeError, ValueError):
        return None
    if max(box) > 1.5:
        box = [c / 1000.0 for c in box]
    if min(box) < -0.005 or max(box) > 1.005:
        return None
    box = [max(0.0, min(1.0, c)) for c in box]
    if span_ok and box[2] >= box[0] and box[3] >= box[1] and (box[2] > box[0]) != (box[3] > box[1]):
        half = _VLM_TARGET_MIN / 2
        if box[2] == box[0]:
            box[0], box[2] = max(0.0, box[0] - half), min(1.0, box[2] + half)
        else:
            box[1], box[3] = max(0.0, box[1] - half), min(1.0, box[3] + half)
    if box[2] <= box[0] or box[3] <= box[1]:
        return None
    return box


def _model_points(raw: Any) -> List[List[float]]:
    if not isinstance(raw, (list, tuple)) or not 2 <= len(raw) <= 8:
        return []
    points: List[List[float]] = []
    flat: List[float] = []
    for point in raw:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            return []
        try:
            points.append([float(point[0]), float(point[1])])
        except (TypeError, ValueError):
            return []
        flat.extend(points[-1])
    if max(flat) > 1.5:
        points = [[x / 1000.0, y / 1000.0] for x, y in points]
    if any(not (-0.005 <= c <= 1.005) for point in points for c in point):
        return []
    return [[max(0.0, min(1.0, x)), max(0.0, min(1.0, y))] for x, y in points]


def _xy(point: Any) -> Point:
    return float(point[0] if isinstance(point, (list, tuple)) else point.x), \
        float(point[1] if isinstance(point, (list, tuple)) else point.y)


def _bezier(points: Sequence[Point], t: float) -> Point:
    (x0, y0), (x1, y1), (x2, y2), (x3, y3) = points
    u = 1.0 - t
    return (u ** 3 * x0 + 3 * u * u * t * x1 + 3 * u * t * t * x2 + t ** 3 * x3,
            u ** 3 * y0 + 3 * u * u * t * y1 + 3 * u * t * t * y2 + t ** 3 * y3)


def _rect(rect: Any) -> Box:
    if isinstance(rect, (list, tuple)):
        return float(rect[0]), float(rect[1]), float(rect[2]), float(rect[3])
    return float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1)


def _distinct(points: Sequence[Point], eps: float) -> List[Point]:
    out: List[Point] = []
    for point in points:
        if all(_dist(point, kept) > eps for kept in out):
            out.append(point)
    return out


def _dist(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _unit(a: Point, b: Point) -> Optional[Point]:
    """Единичный вектор из a в b."""
    return _normalized((b[0] - a[0], b[1] - a[1]))


def _normalized(v: Point) -> Optional[Point]:
    length = math.hypot(v[0], v[1])
    if length <= 1e-9:
        return None
    return v[0] / length, v[1] / length


def _pad(box: Box, pad: float) -> Box:
    return box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad


def _bounds(points: Sequence[Point]) -> Box:
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def _inside(point: Point, box: Box) -> bool:
    return box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3]


def _gap_to_box(point: Point, box: Sequence[float]) -> float:
    dx = max(box[0] - point[0], 0.0, point[0] - box[2])
    dy = max(box[1] - point[1], 0.0, point[1] - box[3])
    return math.hypot(dx, dy)


def _box_gap(a: Sequence[float], b: Sequence[float]) -> float:
    dx = max(b[0] - a[2], 0.0, a[0] - b[2])
    dy = max(b[1] - a[3], 0.0, a[1] - b[3])
    return math.hypot(dx, dy)


def _area(box: Sequence[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _overlap(a: Sequence[float], b: Sequence[float]) -> float:
    return _area((max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])))


def _px(box: Sequence[float], size: Tuple[int, int]) -> Box:
    return box[0] * size[0], box[1] * size[1], box[2] * size[0], box[3] * size[1]


def _to_frac(box: Box, size: Tuple[int, int]) -> List[float]:
    return [box[0] / size[0], box[1] / size[1], box[2] / size[0], box[3] / size[1]]


def _point_frac(point: Point, size: Tuple[int, int]) -> List[float]:
    return [point[0] / size[0], point[1] / size[1]]


def _point_segment_gap(point: Point, a: Point, b: Point) -> float:
    dx, dy = b[0] - a[0], b[1] - a[1]
    length = dx * dx + dy * dy
    if length <= 1e-12:
        return _dist(point, a)
    t = max(0.0, min(1.0, ((point[0] - a[0]) * dx + (point[1] - a[1]) * dy) / length))
    return _dist(point, (a[0] + t * dx, a[1] + t * dy))


def _segment_box_gap(a: Point, b: Point, box: Box) -> float:
    """Расстояние от отрезка до прямоугольника (0 — пересекает или внутри)."""
    if _inside(a, box) or _inside(b, box):
        return 0.0
    corners = [(box[0], box[1]), (box[2], box[1]), (box[2], box[3]), (box[0], box[3])]
    for c1, c2 in zip(corners, corners[1:] + corners[:1]):
        if _segments_cross(a, b, c1, c2):
            return 0.0
    gaps = [_gap_to_box(a, box), _gap_to_box(b, box)]
    gaps.extend(_point_segment_gap(corner, a, b) for corner in corners)
    return min(gaps)


def _segments_cross(p1: Point, p2: Point, q1: Point, q2: Point) -> bool:
    def orient(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    d1, d2 = orient(q1, q2, p1), orient(q1, q2, p2)
    d3, d4 = orient(p1, p2, q1), orient(p1, p2, q2)
    return (d1 * d2 < 0) and (d3 * d4 < 0)


def _same_tips(a: Tuple[Arrow, Arrow], b: Tuple[Arrow, Arrow], eps: float) -> bool:
    first = sorted((x.tip for x in a))
    second = sorted((x.tip for x in b))
    return all(_dist(p, q) <= eps for p, q in zip(first, second))
