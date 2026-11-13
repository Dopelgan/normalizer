"""
Связь надписи листа с видом: где лежит вид и какому виду принадлежит величина.

Зачем. У каждой величины, прочитанной моделью, есть координаты (`bbox` поля),
а у вида — нет: заголовок «Б (4:1)» приходит такой же надписью, как и всё
остальное, и ничто не говорит, какая часть листа ему принадлежит. Поэтому
«⌀8,5» и «вид Б» лежат в записи рядом, но врозь, и вопрос «какой диаметр
отверстий на виде Б» по данным не отвечается.

Что делает модуль. По ГОСТ 2.305 вид занимает область листа, его заголовок
стоит рядом с ней (сверху), а размеры внутри области относятся к этому виду.
Отсюда три шага:

1. **Области.** Маска тёмного без длинных прямых (рамка, сетки таблиц) и без
   известных областей разметки (штамп, таблицы, боковые графы) распадается на
   связные группы — это изображения на листе. Группы считаются по крупной
   сетке клеток, а не по пикселям: иначе один вид распадается на сотни
   штрихов.
2. **Заголовок к области.** Надпись, похожая на обозначение вида по ГОСТ
   2.305 («А», «Б (4:1)», «А-А», «Разрез В-В»), привязывается к ближайшей
   группе. Группа без заголовка видом не объявляется: блок технических
   требований — тоже связная группа, и видом он не является.
3. **Величина к виду.** Поле, попавшее в клетки ровно одного вида, этому виду
   и принадлежит. Поле между двумя видами не принадлежит никому и остаётся
   без привязки: неверная привязка хуже её отсутствия, потому что утверждает,
   что диаметр относится к детали, к которой он не относится.

Чего модуль не делает — и почему. Он не ищет выноски и не отдаёт координату
точки, на которую выноска показывает. Выносок он не ищет потому, что прежний
стенд дал на одном листе 116 ложных «полок выносок». Координату цели здесь
пробовали считать прогулкой по чернилам от рамки надписи: на трёх настоящих
листах такая прогулка дала 20 «целей», и ни одна из них не была остриём
выноски — штрих обрывался на ближайшей размерной линии в двух-трёх пикселях
от самой надписи. Поэтому поля `target` в выдаче нет: неверная координата —
такое же ложное утверждение, как неверная привязка.

Выноска при этом работает, но иначе — не как находка, а как связность:
величина, соединённая выноской со штрихами вида, попадает с ними в одну
связную группу сама, и отдельного детектора для этого не нужно. Именно так
«⌀8,5 N отв.», стоящее за пределами развёртки на листе ТАДУ 687000.001.001,
получает свой вид.

Модуль ничего не пишет в текст записи: он отдаёт структуру, текст собирает
потребитель.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field as dataclass_field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from core.providers import drawing_vision

logger = logging.getLogger(__name__)

# Клетка крупной сетки — доля большей стороны листа. Меньше — вид распадается
# на штрихи, больше — соседние виды слипаются.
_CELL = 0.012
# Группа мельче этого (в клетках) — не изображение, а обрывок штриха.
_MIN_CELLS = 8
# Отрезок длиннее этой доли стороны — рамка или сетка, а не штрих вида.
_LONG_LINE = 0.30
# Заголовок дальше этого от группы (доли листа) к ней не относится.
_TITLE_REACH = 0.09
# Группа без заголовка присоединяется к виду, если она ближе этого и вдвое
# ближе, чем к любому другому виду.
_ABSORB = 0.035
_ABSORB_MARGIN = 2.0
# ...и только если она заметно меньше его самого: соседний вид по площади
# сравним с видом, а размерная цепочка — нет. Без этого предела вид с
# заголовком забирал соседний вид без заголовка и выдавал его размеры за
# свои — четыре неверные привязки на первом же листе.
_ABSORB_SIZE = 0.5
# Штамп по ГОСТ 2.104 — 185x55 мм, на A4 это 0,27 площади листа. Область,
# названная штампом, но большая этого, штампом не считается.
_STAMP_AREA = 0.3
# Надпись длиннее этого и с пробелом — фраза, а не величина: её чернила
# из маски убираются.
_PROSE_CHARS = 25
# Запас, на который величина считается «у вида»: ноль — внутри клеток вида.
_NEAR = 0.02
# Насколько второй вид должен быть дальше первого, чтобы выбор был однозначным.
_UNIQUE_MARGIN = 0.02

KIND_VIEW = "view"
KIND_SECTION = "section"
KIND_DETAIL = "detail"

BASIS_INSIDE = "inside_view"
BASIS_NEAR = "near_view"
BASIS_NONE = "unattributed"

# Обозначение вида по ГОСТ 2.305: одна прописная кириллическая буква или пара
# через дефис, при желании со словом («Вид А», «Разрез Б-Б», «Сечение В-В»)
# и с масштабом в скобках. Буквы, которыми ГОСТ 2.316 обозначать виды не
# велит (З, Й, О, Х, Ъ, Ы, Ь), сюда не попадают: «О» — это ноль, «Х» —
# знак умножения в «М24х1,5», и без этого запрета каждый второй размер
# объявлялся бы видом.
_LETTER = "[АБВГДЕЖИКЛМНПРСТУФЦЧШЩЭЮЯ]"
_WORD = r"(?:вид|разрез|сечение)"
_SCALE = r"(?:\s*\(\s*\d+\s*[:.]\s*\d+(?:,\d+)?\s*\)\s*)"
_TITLE_RE = re.compile(
    r"^\s*(?:(?P<word>%s)\s+)?(?P<name>%s(?:\s*[-–—]\s*%s)?)\s*(?P<scale>%s)?\s*$"
    % (_WORD, _LETTER, _LETTER, _SCALE),
    re.IGNORECASE,
)
# Знак «развёрнуто» и прочие служебные глифы модель читает как попало; такой
# заголовок признаётся только вместе с масштабом в скобках.
_UNNAMED_RE = re.compile(r"^\s*\S{1,2}\s*\(\s*\d+\s*[:.]\s*\d+(?:,\d+)?\s*\)\s*$")

# Категории, которые видом не бывают и в привязке не участвуют.
_NOT_A_VALUE = frozenset({"title_block", "material"})
# Рамка «весь лист» — признак того, что модель координат не дала.
_WHOLE_SHEET = (0.0, 0.0, 1.0, 1.0)


@dataclass
class ViewTitle:
    """Надпись, похожая на обозначение вида."""

    name: str
    kind: str
    scale: str
    bbox: List[float]
    index: int


@dataclass
class View:
    """Вид листа: имя, заголовок и занятая им область."""

    view_id: str
    name: str
    kind: str
    scale: str
    title_bbox: List[float]
    bbox: List[float]
    cells: set = dataclass_field(default_factory=set)
    # Вид, с которого снят выносной элемент: там стоит буква-указание с той
    # же буквой, что и в заголовке (ГОСТ 2.305 п. 7).
    taken_from: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "id": self.view_id,
            "name": self.name,
            "kind": self.kind,
            "bbox": [round(c, 4) for c in self.bbox],
            "title_bbox": [round(c, 4) for c in self.title_bbox],
        }
        if self.scale:
            out["scale"] = self.scale
        if self.taken_from:
            out["taken_from"] = self.taken_from
        return out


# ===========================================================================
# Разбор заголовка
# ===========================================================================

def parse_title(text: str) -> Optional[Tuple[str, str, str]]:
    """«Б (4:1)» -> ('Б', 'detail', '4:1'). `None` — это не заголовок вида."""
    raw = (text or "").strip()
    if not raw or len(raw) > 24:
        return None
    match = _TITLE_RE.match(raw)
    if match is None:
        if _UNNAMED_RE.match(raw):
            scale = re.sub(r"[^\d:.,]", "", raw.split("(")[-1]).strip(".")
            return "", KIND_VIEW, scale.replace(".", ":")
        return None
    name = re.sub(r"\s*[-–—]\s*", "-", match.group("name").strip().upper())
    word = (match.group("word") or "").lower()
    scale = match.group("scale") or ""
    scale = re.sub(r"[^\d:.,]", "", scale).replace(".", ":").strip(":") if scale else ""
    if "-" in name:
        kind = KIND_SECTION
    elif word in ("разрез", "сечение"):
        kind = KIND_SECTION
    else:
        # Одна буква с масштабом — выносной элемент (ГОСТ 2.305 п. 7), без
        # масштаба — вид по стрелке. Для привязки разница только в названии.
        kind = KIND_DETAIL if scale else KIND_VIEW
    return name, kind, scale


def titles(fields: Sequence[Dict[str, Any]]) -> List[ViewTitle]:
    """Заголовки видов среди прочитанных надписей, с их координатами."""
    found: List[ViewTitle] = []
    for index, item in enumerate(fields):
        bbox = _bbox_of(item)
        if bbox is None:
            continue
        parsed = parse_title(str(item.get("value") or ""))
        if parsed is None:
            continue
        name, kind, scale = parsed
        found.append(ViewTitle(name=name, kind=kind, scale=scale, bbox=bbox, index=index))
    return found


def _bbox_of(item: Dict[str, Any]) -> Optional[List[float]]:
    """Координаты надписи; `None` — координат нет (модель дала весь лист)."""
    raw = item.get("bbox")
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        return None
    try:
        box = [float(c) for c in raw]
    except (TypeError, ValueError):
        return None
    if tuple(round(c, 4) for c in box) == _WHOLE_SHEET:
        return None
    if box[2] <= box[0] or box[3] <= box[1]:
        return None
    return box


# ===========================================================================
# Области изображений
# ===========================================================================

def _prose(fields: Sequence[Dict[str, Any]]) -> List[List[float]]:
    """
    Рамки связного текста: технические требования и пояснения.

    Прозу видно по самой надписи — это фраза, а не величина, — и её чернила
    надо убрать из маски: на листе пробки блок требований примыкает к
    выносному элементу «Б (2:1)», и без этого вид Б и требования оказывались
    одной связной группой, то есть все пункты требований получали привязку к
    виду Б.
    """
    out: List[List[float]] = []
    for item in fields:
        value = str(item.get("value") or "")
        if len(value) < _PROSE_CHARS or " " not in value.strip():
            continue
        box = _bbox_of(item)
        if box is not None:
            out.append(box)
    return out


def _blanked(mask, layout: "drawing_vision.SheetLayout",
             fields: Sequence[Dict[str, Any]] = ()):
    """Маска без рамки, граф, штампа и связного текста."""
    numpy = _numpy()
    if numpy is None:
        return None
    work = mask.copy()
    height, width = work.shape

    def rect(box: Sequence[float], pad: float = 0.0) -> None:
        x1 = max(0, int((box[0] - pad) * width))
        y1 = max(0, int((box[1] - pad) * height))
        x2 = min(width, int((box[2] + pad) * width) + 1)
        y2 = min(height, int((box[3] + pad) * height) + 1)
        if x2 > x1 and y2 > y1:
            work[y1:y2, x1:x2] = False

    # Стирается только штамп, и только когда он похож на штамп по размеру.
    # Прочие области разметки не стираются: «таблицей» геометрия называет и
    # сетку, которую она увидела в контуре детали, а слишком большой «штамп»
    # на листе расширителя занимал нижнюю половину листа и вместе с ней
    # уносил вид В. Текст и сетки без заголовка видом всё равно не станут.
    for region in layout.regions:
        if region.kind == drawing_vision.KIND_TITLE_BLOCK and region.area <= _STAMP_AREA:
            rect(region.bbox, pad=0.004)
    stamp = drawing_vision.stamp_bbox(layout)
    if stamp:
        rect(stamp, pad=0.004)
    for box in _prose(fields):
        rect(box, pad=0.002)
    side = drawing_vision.side_column(layout)
    if side:
        rect(side, pad=0.004)
    if layout.frame:
        x1, y1, x2, y2 = layout.frame
        rect([0.0, 0.0, 1.0, y1 + 0.004])
        rect([0.0, y2 - 0.004, 1.0, 1.0])
        rect([0.0, 0.0, x1 + 0.004, 1.0])
        rect([x2 - 0.004, 0.0, 1.0, 1.0])

    # Длинные прямые — рамка и остатки сеток: они соединяют всё со всем.
    rows = work.sum(axis=1)
    cols = work.sum(axis=0)
    for y in numpy.nonzero(rows > _LONG_LINE * width)[0]:
        work[y, :] = False
    for x in numpy.nonzero(cols > _LONG_LINE * height)[0]:
        work[:, x] = False
    return work


def _grid(mask, cell: int):
    """Крупная сетка: клетка занята, если в ней есть хоть один тёмный пиксель."""
    numpy = _numpy()
    height, width = mask.shape
    rows = (height + cell - 1) // cell
    cols = (width + cell - 1) // cell
    pad = numpy.zeros((rows * cell, cols * cell), dtype=bool)
    pad[:height, :width] = mask
    return pad.reshape(rows, cell, cols, cell).any(axis=(1, 3))


def _components(grid) -> List[set]:
    """Связные группы занятых клеток (4-связность)."""
    rows, cols = grid.shape
    seen = [[False] * cols for _ in range(rows)]
    out: List[set] = []
    for sy in range(rows):
        for sx in range(cols):
            if not grid[sy, sx] or seen[sy][sx]:
                continue
            stack = [(sy, sx)]
            seen[sy][sx] = True
            cells = set()
            while stack:
                y, x = stack.pop()
                cells.add((y, x))
                for ny, nx in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
                    if 0 <= ny < rows and 0 <= nx < cols and grid[ny, nx] and not seen[ny][nx]:
                        seen[ny][nx] = True
                        stack.append((ny, nx))
            if len(cells) >= _MIN_CELLS:
                out.append(cells)
    return out


def _cells_bbox(cells: set, step: Tuple[float, float]) -> List[float]:
    ys = [y for y, _ in cells]
    xs = [x for _, x in cells]
    return [min(xs) * step[0], min(ys) * step[1],
            (max(xs) + 1) * step[0], (max(ys) + 1) * step[1]]


def _inside(box: Sequence[float], outer: Sequence[float], pad: float = 0.0) -> bool:
    """Рамка целиком лежит внутри другой, расширенной на `pad`."""
    return (box[0] >= outer[0] - pad and box[1] >= outer[1] - pad
            and box[2] <= outer[2] + pad and box[3] <= outer[3] + pad)


def _distance(cells: set, box: Sequence[float], step: Tuple[float, float]) -> float:
    """Расстояние от рамки надписи до группы клеток, в долях листа."""
    best = 1.0
    for y, x in cells:
        cx1, cy1 = x * step[0], y * step[1]
        cx2, cy2 = cx1 + step[0], cy1 + step[1]
        dx = max(box[0] - cx2, cx1 - box[2], 0.0)
        dy = max(box[1] - cy2, cy1 - box[3], 0.0)
        gap = (dx * dx + dy * dy) ** 0.5
        if gap < best:
            best = gap
            if best == 0.0:
                break
    return best


# ===========================================================================
# Виды листа
# ===========================================================================

def areas(
    layout: "drawing_vision.SheetLayout", fields: Sequence[Dict[str, Any]] = (),
) -> Tuple[List[set], Tuple[float, float], Dict[str, Any]]:
    """Связные группы клеток листа: изображения, блоки текста, цепочки размеров."""
    signals: Dict[str, Any] = {}
    mask = drawing_vision.ink_mask(layout.image)
    if mask is None:
        signals["why"] = "маска листа не построилась"
        return [], (0.0, 0.0), signals
    work = _blanked(mask, layout, fields)
    if work is None:
        signals["why"] = "numpy недоступен"
        return [], (0.0, 0.0), signals

    height, width = work.shape
    cell = max(4, int(_CELL * max(height, width)))
    grid = _grid(work, cell)
    step = (cell / width, cell / height)
    groups = _components(grid)
    signals["groups"] = len(groups)
    if not groups:
        signals["why"] = "изображений на листе не нашлось"
    return groups, step, signals


def _chosen(found_titles: Sequence[ViewTitle]) -> Tuple[List[ViewTitle], List[ViewTitle], int]:
    """
    Какие надписи — заголовки видов, а какие — буквы-указания.

    Одна и та же буква на листе стоит дважды: в заголовке выносного элемента
    («Б (4:1)») и у стрелки или выноски, которая показывает, откуда он снят
    (просто «Б»). Различает их масштаб: у заголовка он есть, у указания —
    нет. Если масштаба нет ни у одной, различить нельзя, и **обе** надписи
    заголовком не признаются: лучше оставить вид без имени, чем назвать
    видом стрелку.
    """
    by_name: Dict[str, List[ViewTitle]] = {}
    for title in found_titles:
        by_name.setdefault(title.name, []).append(title)

    chosen: List[ViewTitle] = []
    pointers: List[ViewTitle] = []
    dropped = 0
    for name, group in by_name.items():
        if len(group) == 1:
            chosen.append(group[0])
            continue
        scaled = [title for title in group if title.scale]
        if len(scaled) == 1:
            chosen.append(scaled[0])
            pointers.extend(title for title in group if title is not scaled[0])
            continue
        dropped += len(group)
    chosen.sort(key=lambda title: (not title.scale, title.name))
    return chosen, pointers, dropped


def views_from(
    groups: Sequence[set], step: Tuple[float, float], found_titles: Sequence[ViewTitle],
) -> Tuple[List[View], Dict[str, Any]]:
    """Группы клеток + заголовки -> виды листа."""
    signals: Dict[str, Any] = {}
    chosen, pointers, dropped = _chosen(found_titles)
    if dropped:
        signals["titles_ambiguous"] = dropped
    if not chosen or not groups:
        return [], signals

    # Заголовок забирает ближайшую группу; одна группа — один заголовок.
    claimed: Dict[int, int] = {}
    homeless = 0
    for title_index, title in enumerate(chosen):
        best, best_gap = None, _TITLE_REACH
        for group_index, cells in enumerate(groups):
            if group_index in claimed:
                continue
            # Чернила самого заголовка — не область вида: иначе заголовок,
            # отрезанный от своего изображения, объявляет видом себя.
            if _inside(_cells_bbox(cells, step), title.bbox, pad=0.012):
                continue
            gap = _distance(cells, title.bbox, step)
            if gap < best_gap:
                best, best_gap = group_index, gap
        if best is None:
            homeless += 1
            continue
        claimed[best] = title_index
    if homeless:
        signals["titles_without_area"] = homeless

    result: List[View] = []
    for group_index, title_index in sorted(claimed.items(), key=lambda kv: kv[1]):
        title = chosen[title_index]
        cells = set(groups[group_index])
        result.append(View(
            view_id="v%d" % (len(result) + 1),
            name=title.name,
            kind=title.kind,
            scale=title.scale,
            title_bbox=[round(c, 4) for c in title.bbox],
            bbox=_cells_bbox(cells, step),
            cells=cells,
        ))
    if not result:
        return [], signals

    # Группы без заголовка — размерные цепочки и выноски сбоку от вида.
    # Присоединяются только к явно ближайшему виду.
    absorbed = 0
    for group_index, cells in enumerate(groups):
        if group_index in claimed:
            continue
        box = _cells_bbox(cells, step)
        gaps = sorted(
            (_distance(view.cells, box, step), order)
            for order, view in enumerate(result)
        )
        if not gaps or gaps[0][0] > _ABSORB:
            continue
        if len(gaps) > 1 and gaps[1][0] < max(gaps[0][0] * _ABSORB_MARGIN, _ABSORB):
            continue
        view = result[gaps[0][1]]
        if len(cells) > _ABSORB_SIZE * len(view.cells):
            continue
        view.cells |= cells
        view.bbox = _cells_bbox(view.cells, step)
        absorbed += 1
    if absorbed:
        signals["groups_absorbed"] = absorbed

    # Буква-указание говорит, с какого вида снят выносной элемент.
    for pointer in pointers:
        target = next((view for view in result if view.name == pointer.name), None)
        if target is None:
            continue
        gaps = sorted(
            (_distance(view.cells, pointer.bbox, step), order)
            for order, view in enumerate(result) if view is not target
        )
        if gaps and gaps[0][0] <= _NEAR:
            target.taken_from = result[gaps[0][1]].view_id
    return result, signals


def views(
    layout: "drawing_vision.SheetLayout", fields: Sequence[Dict[str, Any]]
) -> Tuple[List[View], Tuple[float, float], Dict[str, Any]]:
    """Виды листа: заголовок из надписей + область из геометрии."""
    found_titles = titles(fields)
    signals: Dict[str, Any] = {"titles": len(found_titles)}
    if not found_titles:
        return [], (0.0, 0.0), signals
    groups, step, area_signals = areas(layout, fields)
    signals.update(area_signals)
    if not groups:
        return [], step, signals
    result, view_signals = views_from(groups, step, found_titles)
    signals.update(view_signals)
    return result, step, signals


# ===========================================================================
# Привязка величин
# ===========================================================================

def links(
    found: Sequence[View],
    fields: Sequence[Dict[str, Any]],
    step: Tuple[float, float],
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Какому виду принадлежит каждая прочитанная величина."""
    title_indexes = {title.index for title in titles(fields)}
    # Связный текст — требования и пояснения — относится к листу, а не к виду,
    # даже когда лежит вплотную к нему.
    prose_boxes = {tuple(box) for box in _prose(fields)}
    out: List[Dict[str, Any]] = []
    counts = {"linked": 0, "unlinked": 0, "ambiguous": 0, "no_coordinates": 0}
    for index, item in enumerate(fields):
        if index in title_indexes:
            continue
        if str(item.get("category") or "") in _NOT_A_VALUE:
            continue
        box = _bbox_of(item)
        if box is not None and tuple(box) in prose_boxes:
            continue
        record: Dict[str, Any] = {
            "field_index": index,
            "value": str(item.get("value") or ""),
            "category": str(item.get("category") or ""),
        }
        if box is None:
            counts["no_coordinates"] += 1
            record.update({"view": None, "basis": BASIS_NONE, "why": "no_coordinates"})
            out.append(record)
            continue
        record["bbox"] = [round(c, 4) for c in box]

        gaps = sorted(
            (_distance(view.cells, box, step), order) for order, view in enumerate(found)
        )
        first = gaps[0] if gaps else None
        second = gaps[1] if len(gaps) > 1 else None
        if first is None or first[0] > _NEAR:
            counts["unlinked"] += 1
            record.update({"view": None, "basis": BASIS_NONE, "why": "outside_views"})
            out.append(record)
            continue
        if second is not None and second[0] - first[0] < _UNIQUE_MARGIN:
            counts["ambiguous"] += 1
            record.update({
                "view": None, "basis": BASIS_NONE, "why": "between_views",
                "between": [found[first[1]].name or found[first[1]].view_id,
                            found[second[1]].name or found[second[1]].view_id],
            })
            out.append(record)
            continue

        view = found[first[1]]
        counts["linked"] += 1
        record.update({
            "view": view.view_id,
            "view_name": view.name,
            "basis": BASIS_INSIDE if first[0] == 0.0 else BASIS_NEAR,
        })
        out.append(record)
    return out, counts


# ===========================================================================
# Итог для `structured_payload`
# ===========================================================================

def attach(
    layout: "drawing_vision.SheetLayout",
    fields: Sequence[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """
    Блок `structured_payload["value_links"]`: виды листа и привязка величин.

    `None` — связывать нечего: флаг `DRAWING_VALUE_VIEW_LINKS` выключен, на
    листе нет ни одного заголовка вида или нет геометрии. Поля не меняются:
    привязка ссылается на них по `field_index`, значению и координатам, чтобы
    потребитель мог сверить соответствие.
    """
    if not enabled():
        return None
    found, step, signals = views(layout, fields)
    if not found or not step[0]:
        return None
    records, counts = links(found, fields, step)
    out = {
        "coordinate_reference": "deskewed_page",
        "views": [view.as_dict() for view in found],
        "fields": records,
        "counts": counts,
        "signals": dict(signals),
    }
    logger.info(
        "Виды листа: %d, величин привязано %d, без привязки %d, между видами %d",
        len(found), counts["linked"], counts["unlinked"], counts["ambiguous"],
    )
    return out


def enabled() -> bool:
    """Связывание включено флагом `DRAWING_VALUE_VIEW_LINKS` (по умолчанию нет)."""
    try:
        from core.config import settings
    except Exception:  # noqa: BLE001 — без настроек связывание выключено
        return False
    return bool(getattr(settings, "DRAWING_VALUE_VIEW_LINKS", False))


def _numpy():
    try:
        import numpy
    except Exception:  # noqa: BLE001 — без numpy связывание просто не работает
        return None
    return numpy
