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

# Колонка дополнительных граф по ГОСТ 2.104 — 12 мм у левого поля подшивки,
# то есть слева от внутренней рамки. На A4 (210 мм) это 0,057 ширины листа,
# на A1 (841 мм) — 0,014; берётся с запасом, лишняя полоса бумаги чтению не
# мешает, а срезанная графа теряется целиком.
_SIDE_COLUMN_WIDTH = 0.09
# Полоса шире этого — не графы, а часть чертежа: рамка найдена неверно.
_SIDE_COLUMN_MAX = 0.2

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
# Допуск «внутри рамки» для кандидатов в штамп, доли листа.
_FRAME_SLACK = 0.01
# Полоса у края листа, где ищутся линии внутренней рамки.
_EDGE_BAND = 0.15
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
    turn, source = _orientation_turn(straight, frame, regions, signals.get("inner_frame"))
    if turn:
        turned = straight.rotate(turn, expand=True, fillcolor="white")
        refound = _find_regions(turned)
        # Проверка независимая от выбора: на повёрнутом листе заново
        # найденная сетка должна дать штамп на месте.
        if refound and any(
            _stamp_like(r, refound[0], turned.size, 0, strict=False) for r in refound[1]
        ):
            straight = turned
            frame, regions, signals = refound
            signals["turn_source"] = source
        else:
            signals["turn_rejected"] = source
            turn = 0
    regions = _mark_title_block(regions, frame)
    if frame is None and _recovery_enabled():
        frame = _recover_frame(straight, regions, signals)
        if frame is not None:
            # Штамп дотягивается до найденной рамки так же, как до обычной.
            #
            # Осторожно: `_mark_title_block` правит области на месте, и здесь
            # он вызывается по ним второй раз. Это безопасно ровно потому,
            # что первый вызов прошёл с `frame=None` и ничего не дотянул.
            # Если эти строки когда-нибудь переставить местами или вызвать
            # первый раз с найденной рамкой, область штампа растянется дважды.
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
    inner = _inner_frame(horizontals, verticals, width, height) if frame else None
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
    if inner:
        signals["inner_frame"] = inner
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


def side_column(layout: "SheetLayout") -> Optional[List[float]]:
    """
    Полоса с колонкой дополнительных граф: от внутренней рамки влево на
    `_SIDE_COLUMN_WIDTH`, во всю высоту рамки.

    Там лежат «Перв. примен.», «Справ. №», «Инв. № подл.», «Подп. и дата»,
    «Взам. инв. №», «Инв. № дубл.» — то, чего в вырезке штампа нет никогда:
    штамп примыкает к рамке справа, а эти графы стоят слева от неё, в поле
    подшивки. `None` — внутренней рамки нет или полоса не похожа на поле
    подшивки.

    Берётся только внутренняя рамка. Внешняя — это обрез листа, и слева от
    неё лежит не поле подшивки, а пустая бумага: один вызов модели на
    чистую полоску.
    """
    frame = layout.signals.get("inner_frame")
    if not frame:
        return None
    left, top, _, bottom = (float(value) for value in frame)
    if left <= 0.004 or left > _SIDE_COLUMN_MAX:
        # Рамка у самого края: поля подшивки на листе нет (или скан обрезан).
        return None
    start = max(0.0, left - _SIDE_COLUMN_WIDTH)
    return [round(start, 4), round(max(0.0, top), 4),
            round(left, 4), round(min(1.0, bottom), 4)]


# Внутренняя рамка по ГОСТ 2.301 для форматов, которые встречаются на
# чертежах: (ширина, высота) в мм. Нужна только пропорция — по ней формат
# узнаётся без знания разрешения, в котором лист отрисован.
_FRAMES_MM = (
    (185, 287), (277, 195),      # A4 и A4 альбомный
    (395, 287), (272, 410),      # A3 и A3 книжный
    (569, 407), (395, 584),      # A2 и A2 книжный
    (816, 574), (569, 831),      # A1 и A1 книжный
    (1159, 818),                 # A0
)
# Основная надпись формы 1 по ГОСТ 2.104 — 185×55 мм в правом нижнем углу
# внутренней рамки.
_STAMP_MM = (185.0, 55.0)
# Пропорция рамки отличается от табличной больше чем на столько — формат не
# узнан, и подгонять область по ГОСТ нельзя.
_FORMAT_TOLERANCE = 0.06


def stamp_bbox(layout: "SheetLayout") -> Optional[List[float]]:
    """
    Область основной надписи по ГОСТ: 185×55 мм в правом нижнем углу
    внутренней рамки, в долях листа. `None` — рамки нет или её пропорция
    не отвечает ни одному формату.

    Зачем это поверх найденной сетки. Длинные линии штампа примыкают к
    рамке формата и в группу линий не попадают, поэтому сетка находит не
    весь штамп, а его кусок: на листе АЕГН…0015 СБ это оказался правый
    блок «Лит. | Масса | Масштаб» — без обозначения, наименования и строк
    подписей. Вырезка по ГОСТ не зависит от того, какие линии уцелели.
    """
    if layout.image is None:
        return None
    # Внутренняя рамка точнее, но её находит не каждый лист: на листе
    # АЕГН…0015 СБ нижняя линия ушла на 0,85 высоты, и пропорция не сошлась
    # ни с одним форматом. Тогда берётся рамка формата — у неё та же
    # пропорция с точностью до поля подшивки.
    for candidate in (layout.signals.get("inner_frame"), layout.frame):
        found = _stamp_in(candidate, layout.image.size)
        if found is not None:
            return found
    return None


def _stamp_in(frame: Any, size: Tuple[int, int]) -> Optional[List[float]]:
    """Область штампа внутри рамки `frame`, если её пропорция — формат ГОСТ."""
    if not frame:
        return None
    left, top, right, bottom = (float(value) for value in frame)
    width, height = right - left, bottom - top
    if width <= 0 or height <= 0:
        return None
    # Рамка во весь лист — это не рамка, а обрез: её пропорция равна
    # пропорции бумаги (0,707) и случайно сходится с форматом A2 книжным.
    # Настоящая рамка всегда отступает от края.
    if left <= 0.004 and top <= 0.004 and right >= 0.996 and bottom >= 0.996:
        return None
    # Пропорция рамки на листе: доли листа домножаются на его стороны.
    aspect = (width * size[0]) / (height * size[1])
    best = min(_FRAMES_MM, key=lambda mm: abs(mm[0] / mm[1] - aspect))
    if abs(best[0] / best[1] - aspect) > _FORMAT_TOLERANCE * best[0] / best[1]:
        return None
    stamp = [
        right - width * _STAMP_MM[0] / best[0],
        bottom - height * _STAMP_MM[1] / best[1],
        right, bottom,
    ]
    return [round(max(0.0, min(1.0, value)), 4) for value in stamp]


def title_block_crop(layout: "SheetLayout", region: Region) -> List[float]:
    """
    Что вырезать под чтение штампа: область по ГОСТ, если формат листа
    узнан, иначе — то, что нашла сетка.

    Область по ГОСТ именно заменяет найденную, а не дополняет её. Сетка
    ошибается в обе стороны: на одном листе она дала правый блок штампа
    без обозначения и фамилий, на другом — половину листа, где сам штамп
    занимает седьмую часть вырезки. Вырезать половину листа ради штампа
    значит вернуться к тому разрешению, на котором путаются фамилии.
    """
    return stamp_bbox(layout) or list(region.bbox)


def upright(image, turn: int = -90):
    """
    Колонка дополнительных граф, повёрнутая под чтение. По ГОСТ 2.104
    надписи в ней читаются снизу вверх, поэтому полоса поворачивается по
    часовой: нижняя графа уходит налево и становится первой в строке.
    """
    try:
        return image.rotate(turn, expand=True, fillcolor="white")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Поворот колонки граф не удался: %s", exc)
        return image


def crop(
    image,
    bbox: Sequence[float],
    upscale: Optional[float] = None,
    padding: float = 0.01,
    max_side: Optional[int] = None,
):
    """
    Вырезает область по долям листа и увеличивает её. Увеличение — не
    украшение: мелкий текст штампа читается на увеличенном фрагменте
    заметно лучше, а лист целиком в таком разрешении не отправить.

    `max_side` — потолок длинной стороны после увеличения. Он обязан
    совпадать с тем, с которым вырезку отправят модели: ужать её здесь до
    1600, а потом просить 2600 значит не увеличить ничего.
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
            limit = settings.QWEN_REGION_MAX_SIDE if max_side is None else max_side
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


def _inner_frame(horizontals, verticals, width: int, height: int) -> Optional[List[float]]:
    """
    Внутренняя рамка: ближайшие к середине длинные линии с каждой стороны.
    `_frame` берёт крайние — у скана это часто обрез листа, а между ним и
    рамкой лежит поле подшивки с колонкой дополнительных граф.
    """
    long_h = [h[0] / height for h in horizontals if h[2] - h[1] >= 0.7 * width]
    long_v = [v[0] / width for v in verticals if v[2] - v[1] >= 0.7 * height]
    # Только у края: верхняя линия штампа A4 во всю ширину рамки тоже длинная.
    top = [y for y in long_h if y <= _EDGE_BAND]
    bottom = [y for y in long_h if y >= 1 - _EDGE_BAND]
    left = [x for x in long_v if x <= _EDGE_BAND]
    right = [x for x in long_v if x >= 1 - _EDGE_BAND]
    if not (top and bottom and left and right):
        return None
    return [round(max(left), 4), round(max(top), 4), round(min(right), 4), round(min(bottom), 4)]


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
# Восстановление рамки
# ===========================================================================
#
# `_frame` требует по две сплошные линии длиной 0,7 стороны. На скане линия
# рамки распадается: при остаточном наклоне она переходит со строки на
# строку, и куски короче `_MIN_LINE` детектор не видит вовсе. Таблицы и штамп
# при этом находятся — линий на листе хватает, не проходит порог.
#
# Три способа ниже включаются каждый своим флагом и работают только там, где
# обычная рамка не нашлась. Общее правило: неверная рамка хуже отсутствующей —
# по ней размеры уйдут к чужому виду. Поэтому каждый способ либо получает
# прямоугольник с пропорцией формата ГОСТ 2.301, либо отказывается и пишет
# причину в `frame_recovery.refused`.
#
# Все числа здесь — из геометрии листа, а не из подгонки под примеры:
#
# * допуск по строке — тот же `_MERGE_GAP`, которым детектор уже склеивает
#   линию разной толщины;
# * разрыв, через который линия сшивается, — `_MIN_LINE`: кусок линии короче
#   этого детектор отбрасывает, значит дыра такой длины может быть его же
#   слепым пятном, а не концом линии;
# * длина сшитой линии считается по чернилам, без разрывов, и сравнивается с
#   тем же `_FRAME_LINE`;
# * поля 20 и 5 мм и размеры листов — ГОСТ 2.301, штамп 185 × 55/40/15 мм — ГОСТ 2.104
#   (формы 1, 2 и 2а);
# * `_SIDE_MAJORITY` — сторона считается подтверждённой, когда прочерчена
#   больше чем наполовину: меньше половины — стороны скорее нет, чем есть;
# * доля листа форматы не отсеивает, а только упорядочивает — почему, см.
#   `_fills`.

# Поля листа по ГОСТ 2.301: правое, верхнее и нижнее по 5 мм, левое (поле
# подшивки) — 20 мм. Разницу между ними проверяет `_rejected`.
_MARGIN_MM = 5.0
# Колонка дополнительных граф по ГОСТ 2.104: 12 мм влево от рамки. Только её
# линии продолжаются за рамку — нижняя линия рамки уходит под колонку.
_SIDE_COLUMN_MM = 12.0
# Листы ГОСТ 2.301, короткая и длинная стороны в мм.
_SHEETS_MM = ((210, 297), (297, 420), (420, 594), (594, 841), (841, 1189))
# Шаг ряда форматов ГОСТ 2.301 по стороне: соседний формат той же ориентации
# вдвое больше по площади, то есть в корень из двух по каждой стороне.
_SHEET_STEP = 2 ** 0.5
_STAMP_HEIGHTS_MM = (55.0, 40.0, 15.0)
_SIDE_MAJORITY = 0.5
# Сколько линий с каждой стороны листа перебирается при сборке прямоугольника.
_SIDE_CANDIDATES = 4

SOURCE_STITCH = "stitch"
SOURCE_FOUR_SIDES = "four_sides"
SOURCE_TITLE_BLOCK = "title_block"

# Линия для восстановления: (положение, начало, конец, длина по чернилам).
_Line = Tuple[float, float, float, float]


def _recovery_enabled() -> bool:
    return bool(
        settings.DRAWING_FRAME_STITCH
        or settings.DRAWING_FRAME_FOUR_SIDES
        or settings.DRAWING_FRAME_FROM_TITLE_BLOCK
    )


def _recover_frame(image, regions: Sequence[Region], signals: Dict[str, Any]) -> Optional[List[float]]:
    """
    Рамка листа, на котором `_frame` её не нашёл. Найденная рамка — внутренняя
    (по ГОСТ), она же пишется в `inner_frame`. Итог и причины отказов —
    в `signals["frame_recovery"]`.
    """
    report: Dict[str, Any] = {"source": None, "refused": {}}
    signals["frame_recovery"] = report
    mask = _thin_line_mask(image) if settings.DRAWING_FRAME_STITCH else ink_mask(image)
    if mask is None:
        # Отказ без маски — тоже отказ: без него лист выпал бы из счёта.
        report["refused"]["mask"] = "маска листа не построилась"
        return None

    if settings.DRAWING_FRAME_STITCH:
        horizontals, verticals = _stitched(mask, axis=0), _stitched(mask, axis=1)
    else:
        horizontals = [(p, a, b, b - a) for p, a, b in _lines(mask, axis=0)]
        verticals = [(p, a, b, b - a) for p, a, b in _lines(mask, axis=1)]

    attempts = []
    if settings.DRAWING_FRAME_STITCH:
        attempts.append((SOURCE_STITCH, lambda: _frame_by_sides(
            horizontals, verticals, mask, whole_sheet=True)))
    if settings.DRAWING_FRAME_FOUR_SIDES:
        attempts.append((SOURCE_FOUR_SIDES, lambda: _frame_by_sides(
            horizontals, verticals, mask, whole_sheet=False)))
    if settings.DRAWING_FRAME_FROM_TITLE_BLOCK:
        block = next((r for r in regions if r.kind == KIND_TITLE_BLOCK), None)
        attempts.append((SOURCE_TITLE_BLOCK, lambda: _frame_by_title_block(
            block, horizontals, verticals, mask)))

    for source, attempt in attempts:
        frame, detail = attempt()
        if frame is None:
            report["refused"][source] = detail
            continue
        report["source"] = source
        report["format_mm"] = detail
        report["format_certain"] = _format_certain(frame, mask.shape)
        signals["inner_frame"] = frame
        logger.info(
            "Рамка листа восстановлена (%s): %s, формат рамки %s мм%s",
            source, frame, detail,
            "" if report["format_certain"] else " (формат не подтверждён: лист обрезан по рамке)",
        )
        return frame
    return None


def _format_certain(frame: Sequence[float], shape: Tuple[int, int]) -> bool:
    """
    Можно ли верить ярлыку формата у найденной рамки.

    Пропорция у горизонтальных форматов почти вырождена, и разделяет их
    только занятая доля картинки. Когда рамка упирается в край картинки,
    собственного поля листа (5 мм по ГОСТ 2.301, 20 мм слева) в кадре нет:
    скан обрезан по рамке, доля ничего не говорит, и формат может оказаться
    соседним по ряду. Сама рамка при этом верна — ярлык нет.

    На корпусе это 3 листа из 22 восстановленных: рамка у всех трёх легла
    точно на линию, а формат у двух оказался на шаг больше настоящего.
    """
    height, width = shape
    edge = (2 * _reach(min(height, width)) + 1) / min(height, width)
    left, top, right, bottom = (float(value) for value in frame)
    return min(left, top, 1.0 - right, 1.0 - bottom) > edge


def _thin_line_mask(image):
    """
    Маска тёмного, на которой тонкая линия остаётся линией. `ink_mask`
    уменьшает лист усреднением: линия тоньше пикселя маски (рамка в 0,3 мм
    на A3 при 1000 пикселях на лист) сереет, порог режет её на точки, и ни
    один кусок не дотягивает до `_MIN_LINE`. Здесь лист сначала уменьшается
    по минимуму — в каждом блоке берётся самый тёмный пиксель, — и только
    остаток до рабочего размера добирается усреднением.
    """
    numpy = _numpy()
    if numpy is None:
        return None
    try:
        from PIL import Image
        grey = numpy.asarray(image.convert("L"))
        factor = max(grey.shape) // WORK_SIDE
        if factor >= 2:
            rows, columns = grey.shape[0] // factor, grey.shape[1] // factor
            grey = grey[: rows * factor, : columns * factor]
            grey = grey.reshape(rows, factor, columns, factor).min(axis=(1, 3))
        work = _downscale(Image.fromarray(grey), WORK_SIDE)
        grey = numpy.asarray(work, dtype=numpy.float32)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Маска тонких линий не построилась: %s", exc)
        return None
    if grey.size == 0:
        return None
    return grey < _threshold(grey, numpy)


def _reach(rows: int) -> int:
    """Допуск по строке в пикселях маски — как у склейки толщины в `_lines`."""
    return int(round(max(2.0, _MERGE_GAP * rows)))


def _stitched(mask, axis: int) -> List[_Line]:
    """
    Линии, сшитые из кусков. К одной линии относятся кусок, съехавший на
    соседнюю строку (остаточный наклон после выравнивания), и кусок за
    разрывом не длиннее `_MIN_LINE`. Положение — центр тяжести чернил,
    длина — по чернилам, без разрывов.
    """
    numpy = _numpy()
    profile = mask if axis == 0 else mask.T
    rows, length = profile.shape
    reach = _reach(rows)
    minimum = max(8, int(_MIN_LINE * length))

    # Полоса: строка вместе с соседними в пределах допуска. Линия, идущая
    # лесенкой, в полосе непрерывна.
    band = profile.copy()
    for shift in range(1, reach + 1):
        band[shift:] |= profile[:-shift]
        band[:-shift] |= profile[shift:]

    pieces: List[Tuple[int, int, int]] = []                   # строка, начало, конец
    for index in numpy.flatnonzero(band.sum(axis=1) >= minimum):
        for start, end in _runs(band[index], numpy):
            if end - start >= minimum:
                pieces.append((int(index), start, end))
    pieces.sort(key=lambda piece: piece[1] - piece[2])        # длинные первыми

    lines: List[_Line] = []
    used = [False] * len(pieces)
    for seed, (row, start, end) in enumerate(pieces):
        if used[seed]:
            continue
        used[seed] = True
        spans = [(start, end)]
        near = [i for i, piece in enumerate(pieces) if abs(piece[0] - row) <= reach]
        grown = True
        while grown:
            grown = False
            for i in near:
                _, a, b = pieces[i]
                # Разрыв до `minimum` — слепое пятно детектора, а не конец линии.
                if not used[i] and a <= end + minimum and b >= start - minimum:
                    start, end = min(start, a), max(end, b)
                    spans.append((a, b))
                    used[i] = True
                    grown = True
        # Полоса толще линии: её крайние строки дают ту же линию ещё раз.
        for i, piece in enumerate(pieces):
            if abs(piece[0] - row) <= 2 * reach and piece[1] >= start and piece[2] <= end:
                used[i] = True
        low, high = max(0, row - reach), min(rows, row + reach + 1)
        weight = profile[low:high, start:end + 1].sum(axis=1)
        total = float(weight.sum())
        if total <= 0:
            continue
        position = float((weight * numpy.arange(low, high)).sum() / total)
        lines.append((position, float(start), float(end), _union(spans)))
    return lines


def _union(spans: Sequence[Tuple[int, int]]) -> float:
    """Суммарная длина отрезков без двойного счёта перекрытий."""
    total, reached = 0, None
    for start, end in sorted(spans):
        if reached is None or start > reached:
            total += end - start
            reached = end
        elif end > reached:
            total += end - reached
            reached = end
    return float(total)


def _sheet_of(paper: Tuple[int, int]) -> Tuple[int, int]:
    """Лист ГОСТ 2.301, на котором лежит рамка `paper`, в её ориентации."""
    sheets = [pair for short, long in _SHEETS_MM for pair in ((short, long), (long, short))]
    fitting = [sheet for sheet in sheets if sheet[0] >= paper[0] and sheet[1] >= paper[1]]
    return min(fitting, key=lambda sheet: sheet[0] * sheet[1])


def _share(paper: Tuple[int, int]) -> Tuple[float, float]:
    """Какую долю своего листа рамка `paper` занимает по ГОСТ 2.301."""
    sheet = _sheet_of(paper)
    return paper[0] / sheet[0], paper[1] / sheet[1]


def _fills(share_w: float, share_h: float, paper: Tuple[int, int]) -> bool:
    """
    Может ли рамка `paper` занимать на картинке долю `share_w`×`share_h`.

    Отсеять формат по доле листа нельзя, и это главное, что тут нужно знать.
    Доля считается от картинки, а картинка — не лист: сканер добавляет вокруг
    листа поле, ширина которого заранее не известна. На A3 достаточно 8 мм
    поля с каждой стороны, чтобы рамка заняла 0,917 высоты вместо
    номинальных 0,966, и прежняя нижняя граница (номинал минус непечатаемое
    поле принтера) отвергала верную рамку. Хуже того, прямоугольник, у
    которого правая сторона — не линия рамки, а край бумаги, при этом границу
    проходил и получал ярлык «A4 альбомный» на листе A3: рамка смещена, а
    формат не тот, и размеры ушли бы к чужому масштабу.

    Поэтому здесь остался только тот предел, который от поля скана не зависит:
    больше целой картинки рамка быть не может, а меньше — не меньше, чем если
    бы вокруг листа добавили ещё целый шаг формата ГОСТ. Лишнего белого
    больше, чем на формат вокруг, — это уже не скан листа.

    Разделять форматы (A4 альбомный, A1 и A0 отличаются по пропорции на доли
    процента) доля листа всё-таки помогает, но как упорядочивание, а не как
    отказ: см. `_format_of`.
    """
    nominal_w, nominal_h = _share(paper)
    return (
        nominal_w / _SHEET_STEP <= share_w <= 1.0
        and nominal_h / _SHEET_STEP <= share_h <= 1.0
    )


def _format_of(
    box_w: float, box_h: float, width: int, height: int, tolerance: float,
) -> Tuple[Optional[Tuple[int, int]], str]:
    """
    Формат внутренней рамки ГОСТ 2.301 для прямоугольника в пикселях маски
    или причина отказа.

    Пропорция отсеивает, доля листа — упорядочивает. Горизонтальные форматы
    по пропорции почти вырождены (A0 1,4169, A4 альбомный 1,4205, A1 1,4216 —
    разница в доли процента), и выбрать между ними по одной пропорции нельзя.
    Выбирается тот, чья доля листа ближе к наблюдаемой. Поле скана наблюдаемую
    долю только занижает, поэтому при прочих равных побеждает формат поменьше
    — у большего рамка обязана прилегать к краю листа теснее, чем видно.

    На саму рамку выбор формата не влияет: прямоугольник собран из найденных
    линий, формат — ярлык к нему (и масштаб мм/пиксель для проверок и для
    вырезки штампа).
    """
    aspect = box_w / box_h
    close = [mm for mm in _FRAMES_MM if abs(mm[0] / mm[1] - aspect) <= tolerance * mm[0] / mm[1]]
    if not close:
        return None, "пропорция не отвечает формату ГОСТ 2.301"
    fitting = [mm for mm in close if _fills(box_w / width, box_h / height, mm)]
    if not fitting:
        return None, "рамка больше картинки или вокруг неё лишний формат белого"
    share_w, share_h = box_w / width, box_h / height

    def mismatch(mm: Tuple[int, int]) -> Tuple[float, float]:
        nominal_w, nominal_h = _share(mm)
        return (
            abs(nominal_w - share_w) + abs(nominal_h - share_h),
            abs(mm[0] / mm[1] - aspect),
        )

    return min(fitting, key=mismatch), ""


def _rejected(
    box: Sequence[float],
    paper: Tuple[int, int],
    mask,
    horizontals: Sequence[_Line],
    verticals: Sequence[_Line],
) -> str:
    """
    Почему прямоугольник `box` (пиксели маски) с пропорцией формата `paper` —
    не рамка листа; пустая строка — возражений нет.

    Поле подшивки. По ГОСТ 2.301 левое поле 20 мм, правое 5 мм: у рамки
    левое поле шире правого больше, чем на погрешность измерения. У обводки
    таблицы и у листа, лежащего боком, поля равны.

    Пересечения. Рамку формата не пересекает ничего: чертёж лежит внутри, за
    рамкой только поле. Линия, идущая сквозь «сторону» или продолжающая её
    за угол, означает, что это графа штампа или край таблицы, а не рамка.
    Исключение одно — левая сторона: под колонку дополнительных граф
    (12 мм по ГОСТ 2.104) уходят её собственные линии и нижняя линия рамки.
    """
    left, top, right, bottom = box
    width = mask.shape[1]
    slack = 2 * _reach(min(mask.shape)) + 1
    scale = (right - left) / paper[0]                         # пикселей маски на мм

    if left - (width - right) <= slack:
        return "нет поля подшивки слева"

    column = _SIDE_COLUMN_MM * scale + slack
    # Сама сторона может продолжаться за угол на поле в 5 мм — там край
    # бумаги и тень сканера попадают в ту же полосу; дальше поля — уже нет.
    margin = _MARGIN_MM * scale + slack
    for position, start, end, _ in horizontals:
        if not top - slack <= position <= bottom + slack:
            continue
        inside = top + slack < position < bottom - slack
        if start < right - slack and end > right + (slack if inside else margin):
            return "линия листа пересекает сторону: право"
        if inside and end > left + slack and start < left - column:
            return "линия листа пересекает сторону: лево"
    for position, start, end, _ in verticals:
        if not left - slack <= position <= right + slack:
            continue
        beyond = slack if left + slack < position < right - slack else margin
        if start < top - beyond and end > top + slack:
            return "линия листа пересекает сторону: верх"
        if start < bottom - slack and end > bottom + beyond:
            return "линия листа пересекает сторону: низ"
    return ""


class _Refusals:
    """
    Причины, по которым отпали кандидаты в рамку.

    Прямоугольников перебирается много, и причина у каждого своя. Держать
    одну переменную нельзя: в поле попадала бы причина последнего по счёту
    кандидата, а не та, из-за которой не прошёл самый похожий на рамку. Здесь
    причины копятся с числом кандидатов, и в отчёт идут от частой к редкой —
    по этому полю потом видно, какая проверка режет листы на самом деле.
    """

    def __init__(self) -> None:
        self._seen: Dict[str, int] = {}

    def add(self, why: str) -> None:
        if why:
            self._seen[why] = self._seen.get(why, 0) + 1

    def summary(self, default: str) -> str:
        if not self._seen:
            return default
        ordered = sorted(self._seen.items(), key=lambda item: (-item[1], item[0]))
        return "; ".join("%s (кандидатов %d)" % (why, count) for why, count in ordered)


def _covered(line: _Line, start: float, end: float) -> float:
    """Какая доля отрезка [start, end] прочерчена линией."""
    if end <= start:
        return 0.0
    span = min(line[2], end) - max(line[1], start)
    # Чернил в пересечении не больше, чем чернил в линии всего.
    return max(0.0, min(span, line[3])) / (end - start)


def _frame_by_sides(
    horizontals: Sequence[_Line],
    verticals: Sequence[_Line],
    mask,
    whole_sheet: bool,
) -> Tuple[Optional[List[float]], Any]:
    """
    Рамка из четырёх линий у краёв листа.

    `whole_sheet=True` — каждая линия длиной `_FRAME_LINE` стороны листа, как
    требует `_frame`, только длина считается у сшитой линии. `False` —
    согласие четырёх сторон: каждая прочерчена больше чем на половину *своей*
    стороны прямоугольника. Порог ниже, но он применяется ко всем четырём
    сторонам сразу; рамки по одной, двум или трём сторонам не бывает.

    В обоих случаях прямоугольник обязан иметь пропорцию формата ГОСТ 2.301,
    занимать на листе долю этого формата (`_fills`), иметь поле подшивки и не
    пересекаться линиями листа (`_rejected`). Допуск пропорции — погрешность измерения: каждая из
    двух сторон известна с точностью до полосы сшивки. Из подходящих берётся
    внутренний: рамка по ГОСТ лежит внутри обреза листа, а тень сканера и
    край бумаги — снаружи.
    """
    height, width = mask.shape
    need = _FRAME_LINE if whole_sheet else _SIDE_MAJORITY
    # Прямоугольник не меньше половины листа, сторона прочерчена на `need`
    # своей длины — значит, линия короче этого стороной быть не может.
    floor = need if whole_sheet else need * 0.5
    long_h = [line for line in horizontals if line[3] >= floor * width]
    long_v = [line for line in verticals if line[3] >= floor * height]

    def side(lines: Sequence[_Line], size: int, near_start: bool) -> List[_Line]:
        if near_start:
            found = [line for line in lines if line[0] <= _EDGE_BAND * size]
        else:
            found = [line for line in lines if line[0] >= (1 - _EDGE_BAND) * size]
        return sorted(found, key=lambda line: -line[3])[:_SIDE_CANDIDATES]

    sides = {
        "верх": side(long_h, height, True), "низ": side(long_h, height, False),
        "лево": side(long_v, width, True), "право": side(long_v, width, False),
    }
    missing = [name for name, lines in sides.items() if not lines]
    if missing:
        return None, "нет линии у края: " + ", ".join(missing)

    tolerance_px = 2.0 * (2 * _reach(min(height, width)) + 1)
    best: Optional[Tuple[float, List[float], Tuple[int, int]]] = None
    refused = _Refusals()
    for top in sides["верх"]:
        for bottom in sides["низ"]:
            for left in sides["лево"]:
                for right in sides["право"]:
                    box_w, box_h = right[0] - left[0], bottom[0] - top[0]
                    if box_w < 0.5 * width or box_h < 0.5 * height:
                        refused.add("прямоугольник меньше половины листа")
                        continue
                    if not whole_sheet and min(
                        _covered(top, left[0], right[0]), _covered(bottom, left[0], right[0]),
                        _covered(left, top[0], bottom[0]), _covered(right, top[0], bottom[0]),
                    ) < need:
                        refused.add("сторона прочерчена меньше чем наполовину")
                        continue
                    paper, why = _format_of(
                        box_w, box_h, width, height, tolerance_px / min(box_w, box_h))
                    if paper is None:
                        refused.add(why)
                        continue
                    why = _rejected([left[0], top[0], right[0], bottom[0]], paper,
                                    mask, horizontals, verticals)
                    if why:
                        refused.add(why)
                        continue
                    area = box_w * box_h
                    if best is None or area < best[0]:
                        best = (area, [left[0] / width, top[0] / height,
                                       right[0] / width, bottom[0] / height], paper)
    if best is None:
        return None, refused.summary("стороны не складываются в прямоугольник")
    return [round(c, 4) for c in best[1]], "%dx%d" % best[2]


def _frame_by_title_block(
    block: Optional[Region],
    horizontals: Sequence[_Line],
    verticals: Sequence[_Line],
    mask,
) -> Tuple[Optional[List[float]], Any]:
    """
    Рамка по основной надписи. Штамп по ГОСТ 2.104 стоит в правом нижнем углу
    внутренней рамки и имеет ширину 185 мм — это масштаб листа. Правая и
    нижняя стороны рамки — линии, которыми штамп замыкается; левая и верхняя
    откладываются от них на размер формата.

    Формат принимается только тогда, когда на предсказанных местах левой и
    верхней сторон действительно лежат линии, прочерченные больше чем
    наполовину, и подходит ровно один формат. Долей листа формат здесь не
    подтверждается: картинка — не лист, и доля зависит от поля, которое
    добавил сканер (см. `_fills`), так что подтверждением она быть не может.
    """
    if block is None:
        return None, "штамп не найден"
    height, width = mask.shape
    reach = _reach(min(height, width))
    x_left, y_top = block.bbox[0] * width, block.bbox[1] * height
    x_end, y_end = block.bbox[2] * width, block.bbox[3] * height

    # Правая и нижняя стороны: линии не левее и не выше края найденной сетки,
    # идущие вдоль штампа больше чем на половину его стороны. Сетка часто
    # кончается на внутренней графе, поэтому размер штампа проверяется ниже.
    rights = [v for v in verticals if v[0] >= x_end - reach
              and _covered(v, y_top, y_end) >= _SIDE_MAJORITY]
    bottoms = [h for h in horizontals if h[0] >= y_end - reach
               and _covered(h, x_left, x_end) >= _SIDE_MAJORITY]
    if not rights or not bottoms:
        return None, "штамп не замкнут линиями справа и снизу"

    found: List[Tuple[List[float], Tuple[int, int]]] = []
    stamp_seen, rejected = False, _Refusals()
    for right in rights:
        scale = (right[0] - x_left) / _STAMP_MM[0]            # пикселей маски на мм
        if scale <= 0:
            continue
        # Высота штампа — одна из трёх по ГОСТ, с точностью до полосы. Формы
        # перебираются от высокой к низкой: внутри штампа формы 1 есть линия
        # в 40 мм от верха, а под штампом формы 2 линии в 55 мм нет.
        bottom = None
        for mm in _STAMP_HEIGHTS_MM:
            fitting = [h for h in bottoms if abs(h[0] - y_top - mm * scale) <= 2 * reach + 1]
            if fitting:
                bottom = min(fitting, key=lambda h: h[0])
                break
        if bottom is None:
            continue
        stamp_seen = True
        slack = _MARGIN_MM * scale
        for paper in _FRAMES_MM:
            left, top = right[0] - paper[0] * scale, bottom[0] - paper[1] * scale
            if left < -slack or top < -slack:
                continue
            left, top = max(0.0, left), max(0.0, top)
            # Линии на предсказанных местах: не дальше половины поля.
            near = max(float(reach), slack / 2)
            left_line = _nearest(verticals, left, near, top, bottom[0])
            top_line = _nearest(horizontals, top, near, left, right[0])
            if left_line is None or top_line is None:
                rejected.add("на предсказанных местах левой и верхней сторон линий нет")
                continue
            left, top = left_line[0], top_line[0]
            why = _rejected([left, top, right[0], bottom[0]], paper, mask, horizontals, verticals)
            if why:
                rejected.add(why)
                continue
            box = [left / width, top / height, right[0] / width, bottom[0] / height]
            found.append(([round(c, 4) for c in box], paper))
    if not stamp_seen:
        return None, "размер штампа не отвечает ГОСТ 2.104"
    if not found:
        return None, rejected.summary("формат по штампу не подтверждён линиями")
    if len({paper for _, paper in found}) > 1:
        return None, "по штампу подходят несколько форматов"
    # Один формат, несколько пар линий (рамка и обрез рядом): внутренняя.
    box, paper = min(found, key=lambda item: (item[0][2] - item[0][0]) * (item[0][3] - item[0][1]))
    return box, "%dx%d" % paper


def _nearest(
    lines: Sequence[_Line], position: float, near: float, start: float, end: float,
) -> Optional[_Line]:
    """Линия у `position`, прочерченная больше чем на половину [start, end]."""
    close = [line for line in lines if abs(line[0] - position) <= near
             and _covered(line, start, end) >= _SIDE_MAJORITY]
    return min(close, key=lambda line: abs(line[0] - position)) if close else None


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


def _within(bbox: Sequence[float], frame: Sequence[float]) -> bool:
    return (
        bbox[0] >= frame[0] - _FRAME_SLACK and bbox[1] >= frame[1] - _FRAME_SLACK
        and bbox[2] <= frame[2] + _FRAME_SLACK and bbox[3] <= frame[3] + _FRAME_SLACK
    )


def _orientation_turn(
    image,
    frame: Optional[List[float]],
    regions: Sequence[Region],
    inner: Optional[List[float]] = None,
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
    # Лист считается стоящим верно, если в правом нижнем углу есть хоть
    # какая-то таблица — тот же признак, по которому _mark_title_block
    # отмечает штамп. Строже нельзя: штамп во всю ширину рамки A4 теряет
    # длинные линии (они уходят в рамку) и распадается на куски, ни один
    # из которых не вытянут, и тогда верный лист поворачивался бы.
    corner_x, corner_y = _TITLE_BLOCK_CORNER
    if any(
        (r.bbox[0] + r.bbox[2]) / 2 >= corner_x and (r.bbox[1] + r.bbox[3]) / 2 >= corner_y
        for r in regions
    ):
        return 0, ""
    # Таблица, выходящая за рамку, — колонка дополнительных граф на поле
    # подшивки (или что-то, слитое с ней), а не штамп. Поле подшивки на
    # повёрнутом листе может оказаться с любой стороны.
    bounds = inner or frame
    inside = [r for r in regions if not bounds or _within(r.bbox, bounds)]
    size = image.size
    candidates = [
        turn for turn in (90, 180, 270)
        if any(_stamp_like(r, frame, size, turn, strict=False) for r in inside)
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
        if any(_stamp_like(r, frame, size, turn, strict=True) for r in inside)
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
