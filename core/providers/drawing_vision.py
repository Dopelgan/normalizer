"""
Разметка листа чертежа геометрией: выравнивание, рамка формата, сетки таблиц.

Разделение труда в ветке чертежей такое: зрение находит, *где* что лежит, а
мультимодальная модель читает, *что* там написано. Это не эстетика, а вывод
стенда: сплошное чтение листа теряет таблицы целиком (спецификация — ноль
строк из пяти), а та же таблица, нарезанная по найденной сетке и прочитанная
областью, восстанавливается полностью. Геометрия при этом ничего не
выдумывает и не требует ни GPU, ни обучения.

Что здесь есть и чего нет.

* **Выравнивание.** На 185 мм ширины наклон 0,7° уводит линию на 2,3 мм, и
  сетка таблицы перестаёт находиться. Угол ищется перебором в пределах
  `DRAWING_DESKEW_LIMIT`: у выправленного листа профиль тёмного по строкам
  резче, чем у наклонённого. Поворот применяется только если выигрыш
  заметен — крутить ровный лист ради нуля значит портить его пересэмплингом.
* **Рамка формата** — самая длинная линия у каждого края. Она же граница
  области, внутри которой имеет смысл искать таблицы.
* **Таблицы.** Длинные отрезки объединяются в связные группы по точкам
  пересечения. Группа, где есть хотя бы три горизонтали и две вертикали, —
  таблица; её ячейки получаются из сетки. Штамп — это таблица, лежащая в
  правом нижнем углу листа.
* **Чего нет:** поиска видов, выносок, кружков позиций и знаков
  шероховатости. Стенд показал, что классическое зрение даёт здесь десятки
  ложных срабатываний (116 «полок выносок» на одном листе), а модель читает
  эти надписи и так.

Модуль работает на numpy и PIL — новых зависимостей ветка не приносит.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from core.config import settings
from core.providers.raster_drawing import WORK_SIDE, _downscale, _threshold

logger = logging.getLogger(__name__)

KIND_TITLE_BLOCK = "title_block"
KIND_TABLE = "table"

# Отрезок считается линией сетки с этой доли стороны листа.
_MIN_LINE = 0.05
# Линии ближе этого друг к другу — одна и та же линия разной толщины.
_MERGE_GAP = 0.004
# Штамп по ГОСТ 2.104 — в правом нижнем углу.
_TITLE_BLOCK_CORNER = (0.45, 0.55)
# Таблица меньше этого по площади листа — скорее рамка допуска, чем таблица.
_MIN_TABLE_AREA = 0.004
# Линия длиннее этой доли стороны принадлежит рамке формата, а не таблице.
_FRAME_LINE = 0.7
# Перебор угла при выравнивании.
_DESKEW_STEP = 0.2
_DESKEW_GAIN = 1.03


@dataclass
class Region:
    """Найденная область листа: что это и где лежит."""

    kind: str
    bbox: List[float]                       # доли листа, [x1, y1, x2, y2]
    rows: int = 0
    cols: int = 0
    cells: List[List[float]] = field(default_factory=list)

    @property
    def area(self) -> float:
        return max(0.0, self.bbox[2] - self.bbox[0]) * max(0.0, self.bbox[3] - self.bbox[1])

    def as_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "bbox": [round(c, 4) for c in self.bbox],
            "rows": self.rows,
            "cols": self.cols,
            "cells": len(self.cells),
        }


@dataclass
class SheetLayout:
    """Разметка листа: выправленная картинка и найденные на ней области."""

    image: Any
    angle: float = 0.0
    frame: Optional[List[float]] = None
    regions: List[Region] = field(default_factory=list)
    signals: Dict[str, Any] = field(default_factory=dict)

    @property
    def title_block(self) -> Optional[Region]:
        for region in self.regions:
            if region.kind == KIND_TITLE_BLOCK:
                return region
        return None

    def hint(self) -> str:
        """Короткая подсказка модели: что на листе нашла геометрия."""
        parts: List[str] = []
        if self.frame:
            parts.append("рамка формата найдена")
        if self.title_block:
            parts.append("штамп в правом нижнем углу")
        tables = sum(1 for r in self.regions if r.kind == KIND_TABLE)
        if tables:
            parts.append(f"таблиц на листе: {tables}")
        if abs(self.angle) >= 0.1:
            parts.append(f"лист выправлен на {self.angle:+.1f}°")
        return "; ".join(parts)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "angle": round(self.angle, 2),
            "frame": [round(c, 4) for c in self.frame] if self.frame else None,
            "regions": [r.as_dict() for r in self.regions],
            **self.signals,
        }


# ===========================================================================
# Разметка
# ===========================================================================

def analyse(image, deskew_limit: Optional[float] = None) -> SheetLayout:
    """Картинка листа -> выправленная картинка и области на ней."""
    numpy = _numpy()
    if numpy is None:
        return SheetLayout(image=image, signals={"vision": "numpy недоступен"})

    limit = settings.DRAWING_DESKEW_LIMIT if deskew_limit is None else deskew_limit
    straight, angle = deskew(image, limit)

    mask = ink_mask(straight)
    if mask is None:
        return SheetLayout(image=straight, angle=angle, signals={"vision": "маска не построилась"})

    height, width = mask.shape
    horizontals = _lines(mask, axis=0)
    verticals = _lines(mask, axis=1)
    frame = _frame(horizontals, verticals, width, height)
    # Линии рамки формата в сетку таблиц не входят: штамп примыкает к рамке,
    # и без этого он склеивается с ней в одну группу во весь лист.
    inner_h = [h for h in horizontals if h[2] - h[1] < _FRAME_LINE * width]
    inner_v = [v for v in verticals if v[2] - v[1] < _FRAME_LINE * height]
    regions = _tables(inner_h, inner_v, width, height)
    regions = _mark_title_block(regions, frame)

    layout = SheetLayout(
        image=straight,
        angle=angle,
        frame=frame,
        regions=regions[: max(1, settings.DRAWING_MAX_REGIONS)],
        signals={
            "horizontal_lines": len(horizontals),
            "vertical_lines": len(verticals),
            "tables": len(regions),
        },
    )
    logger.info(
        "Разметка листа: наклон %.1f°, рамка %s, таблиц %d (штамп %s)",
        angle, "есть" if frame else "нет", len(regions),
        "найден" if layout.title_block else "не найден",
    )
    return layout


def plain(image) -> SheetLayout:
    """Разметка выключена: лист идёт в модель как есть."""
    return SheetLayout(image=image, signals={"vision": "выключено"})


def deskew(image, limit: float) -> Tuple[Any, float]:
    """
    Выправление наклона перебором угла. Возвращает картинку и угол; при
    нулевом угле возвращается исходная картинка без пересэмплинга.
    """
    numpy = _numpy()
    if numpy is None or limit <= 0:
        return image, 0.0

    mask = ink_mask(image)
    if mask is None:
        return image, 0.0

    base = _sharpness(mask, numpy)
    best_angle, best_score = 0.0, base
    steps = int(limit / _DESKEW_STEP)
    for step in range(1, steps + 1):
        for angle in (step * _DESKEW_STEP, -step * _DESKEW_STEP):
            rotated = _rotate_mask(mask, angle, numpy)
            score = _sharpness(rotated, numpy)
            if score > best_score:
                best_angle, best_score = angle, score

    if not best_angle or best_score < base * _DESKEW_GAIN:
        return image, 0.0
    try:
        from PIL import Image
        straight = image.rotate(
            best_angle, resample=Image.BICUBIC, expand=False, fillcolor="white"
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Поворот листа не удался: %s", exc)
        return image, 0.0
    return straight, best_angle


def ink_mask(image, work_side: int = WORK_SIDE):
    """Булева маска тёмного на уменьшенной копии листа. `None` — не вышло."""
    numpy = _numpy()
    if numpy is None:
        return None
    try:
        work = _downscale(image, work_side)
        grey = numpy.asarray(work.convert("L"), dtype=numpy.float32)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Маска листа не построилась: %s", exc)
        return None
    if grey.size == 0:
        return None
    return grey < _threshold(grey, numpy)


def crop(image, bbox: Sequence[float], upscale: Optional[float] = None, padding: float = 0.01):
    """
    Вырезает область по долям листа и увеличивает её. Увеличение — не
    украшение: мелкий текст штампа читается на увеличенном фрагменте
    заметно лучше, а лист целиком в таком разрешении не отправить.
    """
    width, height = image.size
    x1 = max(0.0, float(bbox[0]) - padding)
    y1 = max(0.0, float(bbox[1]) - padding)
    x2 = min(1.0, float(bbox[2]) + padding)
    y2 = min(1.0, float(bbox[3]) + padding)
    box = (int(x1 * width), int(y1 * height), max(1, int(x2 * width)), max(1, int(y2 * height)))
    piece = image.crop(box)

    factor = settings.QWEN_REGION_UPSCALE if upscale is None else upscale
    if factor and factor > 1 and min(piece.size) > 0:
        try:
            from PIL import Image
            target = (int(piece.width * factor), int(piece.height * factor))
            limit = settings.QWEN_REGION_MAX_SIDE
            if max(target) > limit:
                scale = limit / max(target)
                target = (max(1, int(target[0] * scale)), max(1, int(target[1] * scale)))
            piece = piece.resize(target, Image.LANCZOS)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Увеличение области не удалось: %s", exc)
    return piece


# ===========================================================================
# Линии, рамка, таблицы
# ===========================================================================

def _lines(mask, axis: int) -> List[Tuple[float, float, float]]:
    """
    Длинные отрезки. `axis=0` — горизонтали, `axis=1` — вертикали.
    Отрезок отдаётся как (положение, начало, конец) в пикселях маски.
    """
    numpy = _numpy()
    profile = mask if axis == 0 else mask.T
    length = profile.shape[1]
    minimum = max(8, int(_MIN_LINE * length))

    found: List[Tuple[float, float, float]] = []
    for index in range(profile.shape[0]):
        row = profile[index]
        if row.sum() < minimum:
            continue
        for start, end in _runs(row, numpy):
            if end - start >= minimum:
                found.append((float(index), float(start), float(end)))
    return _merge(found, gap=max(2.0, _MERGE_GAP * profile.shape[0]))


def _runs(row, numpy) -> List[Tuple[int, int]]:
    """Непрерывные участки тёмного в строке маски, с допуском в один пиксель."""
    filled = numpy.flatnonzero(row)
    if filled.size == 0:
        return []
    breaks = numpy.flatnonzero(numpy.diff(filled) > 2)
    starts = numpy.concatenate(([filled[0]], filled[breaks + 1]))
    ends = numpy.concatenate((filled[breaks], [filled[-1]]))
    return [(int(s), int(e)) for s, e in zip(starts, ends)]


def _merge(lines, gap: float):
    """Линия толщиной в несколько пикселей — одна линия, а не пять."""
    if not lines:
        return []
    lines = sorted(lines)
    merged = [list(lines[0])]
    for position, start, end in lines[1:]:
        last = merged[-1]
        overlap = min(last[2], end) - max(last[1], start)
        if position - last[0] <= gap and overlap > 0:
            last[0] = (last[0] + position) / 2
            last[1] = min(last[1], start)
            last[2] = max(last[2], end)
        else:
            merged.append([position, start, end])
    return [tuple(line) for line in merged]


def _frame(horizontals, verticals, width: int, height: int) -> Optional[List[float]]:
    """Рамка формата: длинные линии у каждого края, не меньше трёх сторон."""
    long_h = [h for h in horizontals if h[2] - h[1] >= 0.7 * width]
    long_v = [v for v in verticals if v[2] - v[1] >= 0.7 * height]
    if len(long_h) < 2 or len(long_v) < 2:
        return None
    top = min(h[0] for h in long_h) / height
    bottom = max(h[0] for h in long_h) / height
    left = min(v[0] for v in long_v) / width
    right = max(v[0] for v in long_v) / width
    if bottom - top < 0.5 or right - left < 0.5:
        return None
    return [round(left, 4), round(top, 4), round(right, 4), round(bottom, 4)]


def _tables(horizontals, verticals, width: int, height: int) -> List[Region]:
    """
    Группы пересекающихся линий -> таблицы. Связность считается по точкам
    пересечения: линии, которые нигде не встречаются, в одну таблицу не
    попадают, сколько бы их ни было рядом.
    """
    nodes = [("h", line) for line in horizontals] + [("v", line) for line in verticals]
    parent = list(range(len(nodes)))

    def root(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for i, (kind_i, line_i) in enumerate(nodes):
        if kind_i != "h":
            continue
        y, x_start, x_end = line_i
        for j, (kind_j, line_j) in enumerate(nodes):
            if kind_j != "v":
                continue
            x, y_start, y_end = line_j
            if x_start - 2 <= x <= x_end + 2 and y_start - 2 <= y <= y_end + 2:
                a, b = root(i), root(j)
                if a != b:
                    parent[a] = b

    groups: Dict[int, List[Tuple[str, Tuple[float, float, float]]]] = {}
    for index, node in enumerate(nodes):
        groups.setdefault(root(index), []).append(node)

    regions: List[Region] = []
    for group in groups.values():
        rows = sorted(line[0] for kind, line in group if kind == "h")
        columns = sorted(line[0] for kind, line in group if kind == "v")
        if len(rows) < 3 or len(columns) < 2:
            continue
        bbox = [
            columns[0] / width, rows[0] / height,
            columns[-1] / width, rows[-1] / height,
        ]
        region = Region(
            kind=KIND_TABLE,
            bbox=[round(max(0.0, min(1.0, c)), 4) for c in bbox],
            rows=len(rows) - 1,
            cols=len(columns) - 1,
            cells=_cells(rows, columns, width, height),
        )
        # Рамка формата тоже группа пересекающихся линий — таблицей она
        # становиться не должна.
        if region.area >= _MIN_TABLE_AREA and region.area <= 0.6:
            regions.append(region)

    regions.sort(key=lambda r: r.area, reverse=True)
    return regions


def _cells(rows, columns, width: int, height: int) -> List[List[float]]:
    """Сетка ячеек таблицы. Объединённые графы здесь разбиты — это оценка."""
    cells: List[List[float]] = []
    for top, bottom in zip(rows, rows[1:]):
        for left, right in zip(columns, columns[1:]):
            if bottom - top < 3 or right - left < 3:
                continue
            cells.append([
                round(left / width, 4), round(top / height, 4),
                round(right / width, 4), round(bottom / height, 4),
            ])
            if len(cells) >= 400:
                return cells
    return cells


def _mark_title_block(regions: List[Region], frame: Optional[List[float]]) -> List[Region]:
    """Таблица в правом нижнем углу — основная надпись."""
    corner_x, corner_y = _TITLE_BLOCK_CORNER
    candidates = [
        region for region in regions
        if (region.bbox[0] + region.bbox[2]) / 2 >= corner_x
        and (region.bbox[1] + region.bbox[3]) / 2 >= corner_y
    ]
    if not candidates:
        return regions

    # Ближайшая к правому нижнему углу, а не самая большая: рядом со штампом
    # часто лежит таблица изменений.
    best = min(candidates, key=lambda r: (1 - r.bbox[2]) ** 2 + (1 - r.bbox[3]) ** 2)
    best.kind = KIND_TITLE_BLOCK
    # Внешние стороны штампа — это сама рамка формата, и в группу линий они
    # не попали. Дотягиваем область до рамки, иначе крайние графы срежутся.
    if frame:
        if 0 < frame[2] - best.bbox[2] < 0.1:
            best.bbox[2] = frame[2]
        if 0 < frame[3] - best.bbox[3] < 0.1:
            best.bbox[3] = frame[3]
    return regions


# ===========================================================================
# Помощники
# ===========================================================================

def _numpy():
    try:
        import numpy
    except ImportError:  # pragma: no cover — numpy есть в рантайме
        logger.warning("numpy недоступен: разметка листа пропущена")
        return None
    return numpy


def _sharpness(mask, numpy) -> float:
    """
    Насколько резок профиль тёмного по строкам. У выправленного листа линии
    сетки собираются в узкие пики, у наклонённого — размазаны.
    """
    if mask is None or mask.size == 0:
        return 0.0
    profile = mask.sum(axis=1).astype("float32")
    return float((numpy.diff(profile) ** 2).mean())


def _rotate_mask(mask, angle: float, numpy):
    """Поворот маски через PIL: дешевле, чем крутить исходную картинку."""
    try:
        from PIL import Image
    except ImportError:  # pragma: no cover
        return mask
    picture = Image.fromarray((mask * 255).astype("uint8"))
    rotated = picture.rotate(angle, resample=Image.BILINEAR, expand=False, fillcolor=0)
    return numpy.asarray(rotated) > 127
