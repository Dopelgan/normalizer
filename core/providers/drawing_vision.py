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
# Поворот листа по штампу: штамп не дальше этого от правого и нижнего края
# рамки, длинная сторона не короче полутора коротких, а без подтверждения
# OSD — короткая не меньше 0,2 длинной (55 из 185 мм у формы 1, 40 у формы 2).
_TURN_CORNER_GAP = 0.08
_TURN_MIN_ASPECT = 1.5
_TURN_STRICT_RATIO = 0.2
_OSD_SIDE = 2400
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
    # Поворот кратно 90° против часовой (как у PIL.rotate), которым штамп
    # приведён в правый нижний угол; координаты областей — после него.
    turn: int = 0
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
        if self.turn:
            parts.append(f"лист повёрнут на {self.turn}° против часовой, штамп внизу справа")
        if abs(self.angle) >= 0.1:
            parts.append(f"лист выправлен на {self.angle:+.1f}°")
        return "; ".join(parts)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "angle": round(self.angle, 2),
            "turn": self.turn,
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

    found = _find_regions(straight)
    if found is None:
        return SheetLayout(image=straight, angle=angle, signals={"vision": "маска не построилась"})
    frame, regions, signals = found

    # Лист, повёрнутый на 90° (альбомный A3, отсканированный книжным) или
    # вверх ногами, без поворота теряет штамп: в правом нижнем углу его нет,
    # и модель читает надписи боком. Поворот выбирается по штампу и
    # проверяется: если на повёрнутом листе штамп не нашёлся, лист остаётся
    # как был.
    turn, source = _orientation_turn(straight, frame, regions)
    if turn:
        turned = straight.rotate(turn, expand=True, fillcolor="white")
        refound = _find_regions(turned)
        marked = _mark_title_block(refound[1], refound[0]) if refound else []
        if any(r.kind == KIND_TITLE_BLOCK for r in marked):
            straight = turned
            frame, regions, signals = refound
            signals["turn_source"] = source
        else:
            signals["turn_rejected"] = source
            turn = 0
    regions = _mark_title_block(regions, frame)

    layout = SheetLayout(
        image=straight,
        angle=angle,
        turn=turn,
        frame=frame,
        regions=regions[: max(1, settings.DRAWING_MAX_REGIONS)],
        signals=signals,
    )
    logger.info(
        "Разметка листа: поворот %d°, наклон %.1f°, рамка %s, таблиц %d (штамп %s)",
        turn, angle, "есть" if frame else "нет", len(regions),
        "найден" if layout.title_block else "не найден",
    )
    return layout


def _find_regions(image) -> Optional[Tuple[Optional[List[float]], List[Region], Dict[str, Any]]]:
    """Рамка формата и таблицы листа, штамп ещё не отмечен. `None` — нет маски."""
    mask = ink_mask(image)
    if mask is None:
        return None
    height, width = mask.shape
    horizontals = _lines(mask, axis=0)
    verticals = _lines(mask, axis=1)
    frame = _frame(horizontals, verticals, width, height)
    # Линии рамки формата в сетку таблиц не входят: штамп примыкает к рамке,
    # и без этого он склеивается с ней в одну группу во весь лист.
    inner_h = [h for h in horizontals if h[2] - h[1] < _FRAME_LINE * width]
    inner_v = [v for v in verticals if v[2] - v[1] < _FRAME_LINE * height]
    regions = _tables(inner_h, inner_v, width, height)
    signals: Dict[str, Any] = {
        "horizontal_lines": len(horizontals),
        "vertical_lines": len(verticals),
        "tables": len(regions),
    }
    return frame, regions, signals


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
# Ориентация листа
# ===========================================================================

def _turn_bbox(bbox: Sequence[float], turn: int) -> List[float]:
    """
    Рамка в долях листа после `PIL.rotate(turn, expand=True)` (против
    часовой). При 90° верхний правый угол уходит в верхний левый.
    """
    x1, y1, x2, y2 = (float(c) for c in bbox)
    if turn == 90:
        return [y1, 1 - x2, y2, 1 - x1]
    if turn == 180:
        return [1 - x2, 1 - y2, 1 - x1, 1 - y1]
    if turn == 270:
        return [1 - y2, x1, 1 - y1, x2]
    return [x1, y1, x2, y2]


def _stamp_like(
    region: Region,
    frame: Optional[List[float]],
    size: Tuple[int, int],
    turn: int,
    strict: bool,
) -> bool:
    """
    Похожа ли таблица на штамп, если лист повернуть на `turn`: лежит в
    правом нижнем углу у рамки и вытянута вдоль нижнего края (185×55 мм по
    ГОСТ 2.104, у последующих листов 185×15 мм).

    `strict` — для поворота без подтверждения OSD: штамп формы 1/2 (короткая
    сторона не меньше 0,2 длинной) и не меньше трёх граф вдоль длинной
    стороны. Узкая колонка дополнительных граф у левого поля (12×145 мм)
    так за штамп не сойдёт.
    """
    width, height = size
    if turn in (90, 270):
        width, height = height, width
    box = _turn_bbox(region.bbox, turn)
    edge = _turn_bbox(frame, turn) if frame else [0.0, 0.0, 1.0, 1.0]
    corner_x, corner_y = _TITLE_BLOCK_CORNER
    if (box[0] + box[2]) / 2 < corner_x or (box[1] + box[3]) / 2 < corner_y:
        return False
    if edge[2] - box[2] > _TURN_CORNER_GAP or edge[3] - box[3] > _TURN_CORNER_GAP:
        return False
    long_side = (box[2] - box[0]) * width
    short_side = (box[3] - box[1]) * height
    if long_side < _TURN_MIN_ASPECT * short_side:
        return False
    if not strict:
        return True
    # Графы вдоль длинной стороны: при повороте на 90° строки и столбцы
    # сетки меняются местами.
    columns = region.rows if turn in (90, 270) else region.cols
    return short_side >= _TURN_STRICT_RATIO * long_side and columns >= 3


def _orientation_turn(
    image, frame: Optional[List[float]], regions: Sequence[Region]
) -> Tuple[int, str]:
    """
    Поворот кратно 90° против часовой, после которого штамп окажется в
    правом нижнем углу, и чем он подтверждён ("osd" или "geometry").

    Лист, на котором штамп уже стоит где положено, не поворачивается никогда,
    и OSD для него не спрашивается: ошибка OSD на разреженном тексте листа
    не должна переворачивать верный лист. Иначе спрашивается OSD; уверенный
    ответ принимается, только если после такого поворота штамп встаёт на
    место. Без уверенного OSD лист поворачивается, только если штампу
    отвечает ровно один поворот по строгим признакам.
    """
    if not settings.DRAWING_AUTO_ROTATE or not regions:
        return 0, ""
    size = image.size
    if any(_stamp_like(r, frame, size, 0, strict=False) for r in regions):
        return 0, ""
    candidates = [
        turn for turn in (90, 180, 270)
        if any(_stamp_like(r, frame, size, turn, strict=False) for r in regions)
    ]
    if not candidates:
        return 0, ""

    from core.providers.ocr_prepare import osd_rotation

    try:
        gray = image.convert("L")
        # Лист A1 в 200 dpi OSD читает долго, а ориентацию текста видно и так.
        gray.thumbnail((_OSD_SIDE, _OSD_SIDE))
    except Exception:  # noqa: BLE001
        gray = image
    answer = osd_rotation(gray)
    if answer is not None and answer[1] >= settings.OCR_OSD_MIN_CONFIDENCE:
        # OSD говорит, на сколько повернуть по часовой; PIL крутит против.
        turn = (-answer[0]) % 360
        return (turn, "osd") if turn in candidates else (0, "")

    strict = [
        turn for turn in candidates
        if any(_stamp_like(r, frame, size, turn, strict=True) for r in regions)
    ]
    if len(strict) == 1:
        return strict[0], "geometry"
    return 0, ""


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
