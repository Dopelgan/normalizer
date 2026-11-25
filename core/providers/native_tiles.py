"""
Крупный лист плитками в родном разрешении (DRAWING_NATIVE_TILES).

Проход «лист целиком» ужимает лист до QWEN_SHEET_MAX_SIDE (2000 px). Высота
прописной буквы наименьшего шрифта ГОСТ 2.304 (2,5 мм) после ужатия: A4 —
около 17 px, A3 — 12, A2 — 8, A1 — 6. На A3 и крупнее читается 15–45 %
надписей, а та же модель по вырезке в родном разрешении читает их верно.

Что здесь:
- `wanted` — нужен ли листу проход плитками: прописная после ужатия ниже
  DRAWING_NATIVE_TILES_MIN_CAP_PX или ответ листа оборвался (плотный лист);
- `NativeSheet` — тот же выправленный лист в родном разрешении: PDF
  рендерится заново в DRAWING_NATIVE_TILES_DPI, растр берётся как есть;
- `plan` — сетка плиток: наименьшее число плиток, при котором каждая уходит
  модели не грубее DRAWING_NATIVE_TILES_MIN_DPI и не больше
  DRAWING_NATIVE_TILES_MAX_SIDE по стороне;
- `read` — плитки читаются коротким запросом «[код, надпись]» без рамок,
  разом не больше DRAWING_NATIVE_TILES_PARALLEL, в пределах бюджета листа;
- `merge` — надписи плиток одним списком: одна и та же запись из
  перекрывающихся плиток — одна надпись, место — пересечение плиток;
- `unpaired` — правило пары: надпись прохода с рамками побеждает, плитка
  добавляет только то, чему пары нет (`qwen_client.with_tiles`: без рамки,
  как надписи содержания листа).

Отказ плитки не трогает прочтение листа: остаётся то, что дал проход с
рамками, и лист не помечается недочитанным.
"""

from __future__ import annotations

import contextvars
import logging
import math
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from celery.exceptions import SoftTimeLimitExceeded

from core import deadline
from core.config import settings

logger = logging.getLogger(__name__)

# Наименьший шрифт чертежа по ГОСТ 2.304 — 2,5 мм (1,8 мм стандарт не
# рекомендует). По нему и считается, читается ли лист после ужатия.
CAP_MM = 2.5
_MM_PER_INCH = 25.4
# Разумные пределы dpi из сведений растра; вне их сведениям не верим.
_DPI_RANGE = (50.0, 1200.0)


# ---------------------------------------------------------------- порог
def cap_height_px(size: Sequence[int], dpi: float, max_side: Optional[int]) -> float:
    """Высота прописной CAP_MM в пикселях картинки, ужатой до `max_side` по стороне."""
    side = max(size) if size else 0
    scale = 1.0 if not side or not max_side or side <= max_side else max_side / side
    return CAP_MM / _MM_PER_INCH * float(dpi) * scale


def wanted(size: Sequence[int], dpi: float, located: Any) -> str:
    """
    Причина читать лист плитками или пустая строка. `size` и `dpi` — рабочий
    растр листа (тот, что уходит проходу листа), `located` — прочтение
    прохода с рамками (`SheetReading` или None).
    """
    if cap_height_px(size, dpi, settings.QWEN_SHEET_MAX_SIDE) < float(settings.DRAWING_NATIVE_TILES_MIN_CAP_PX):
        return "small_print"
    if located is not None and (getattr(located, "cut", False) or getattr(located, "rejected", "")):
        return "dense"
    return ""


def raster_dpi(image) -> Optional[float]:
    """dpi из сведений растра или None, если их нет или они неправдоподобны."""
    try:
        value = (getattr(image, "info", None) or {}).get("dpi")
        dpi = float(value[0] if isinstance(value, (tuple, list)) else value)
    except (TypeError, ValueError, IndexError):
        return None
    return dpi if _DPI_RANGE[0] <= dpi <= _DPI_RANGE[1] else None


def render_dpi(size: Sequence[int], dpi: float) -> Optional[float]:
    """
    dpi нового рендера PDF: DRAWING_NATIVE_TILES_DPI, но не больше
    DRAWING_NATIVE_TILES_RENDER_MAX_MP мегапикселей на лист (A1 в 300 dpi —
    70 Мп, 210 МБ в памяти). None — новый рендер не точнее рабочего.
    """
    width_in, height_in = size[0] / float(dpi), size[1] / float(dpi)
    target = float(settings.DRAWING_NATIVE_TILES_DPI)
    limit = float(settings.DRAWING_NATIVE_TILES_RENDER_MAX_MP) * 1e6
    if width_in > 0 and height_in > 0 and limit > 0:
        target = min(target, math.sqrt(limit / (width_in * height_in)))
    return target if target > float(dpi) + 1 else None


# ---------------------------------------------------------------- лист
class NativeSheet:
    """
    Выправленный лист в родном разрешении, лениво и один раз на лист.

    `base` — рабочий выправленный растр (`layout.image`), `base_dpi` — его
    dpi; `render(dpi)` — тот же лист, выправленный так же, в другом dpi, или
    None (растр: родное разрешение и есть рабочее). Потокобезопасно: проход
    листа и проход штампа могут идти разом (DRAWING_PASSES_PARALLEL).
    """

    def __init__(self, base, base_dpi: float, render: Optional[Callable[[float], Any]] = None):
        self.base = base
        self.base_dpi = float(base_dpi)
        self._render = render
        self._lock = threading.Lock()
        self._loaded: Optional[Tuple[Any, float]] = None

    @property
    def size(self) -> Tuple[int, int]:
        return tuple(self.base.size)  # type: ignore[return-value]

    def image(self) -> Tuple[Any, float]:
        """(картинка, её dpi). При отказе рендера — рабочий растр."""
        with self._lock:
            if self._loaded is None:
                self._loaded = self._load()
            return self._loaded

    def _load(self) -> Tuple[Any, float]:
        dpi = render_dpi(self.size, self.base_dpi) if self._render is not None else None
        if dpi is None:
            return self.base, self.base_dpi
        try:
            image = self._render(dpi)
        except (SoftTimeLimitExceeded, deadline.Cancelled):
            # Срок задачи и отмена — не отказ рендера: их глотать нельзя.
            raise
        except Exception as exc:  # noqa: BLE001 — рендер не роняет лист
            logger.warning("Лист в родном разрешении не отрисован (%.0f dpi): %s", dpi, exc)
            image = None
        if image is None:
            return self.base, self.base_dpi
        return image, dpi

    def crop(self, bbox: Sequence[float], upscale: float, max_side: int, padding: float = 0.01):
        """
        Вырезка по долям листа из родного разрешения. Увеличение делится на
        то, во сколько раз родной растр крупнее рабочего: размер вырезки тот
        же, что и прежде, но детали в ней настоящие, а не дорисованные.
        """
        from core.providers import drawing_vision

        image, _ = self.image()
        ratio = image.size[0] / float(self.base.size[0] or 1)
        factor = max(1.0, float(upscale) / ratio) if ratio > 0 else float(upscale)
        return drawing_vision.crop(image, bbox, upscale=factor, padding=padding, max_side=max_side)


# ---------------------------------------------------------------- сетка
@dataclass
class TilePlan:
    """Сетка плиток: столбцы, строки, dpi, в котором плитка уходит модели, рамки, сторона."""

    columns: int
    rows: int
    dpi: float
    boxes: List[List[float]] = field(default_factory=list)
    max_side: int = 0


def grid_boxes(columns: int, rows: int, overlap: float) -> List[List[float]]:
    """
    Плитки columns x rows в долях листа, в порядке чтения (сверху вниз,
    слева направо). Соседние плитки перекрываются на `overlap` шага сетки:
    строка на стыке целиком попадает хотя бы в одну.
    """
    columns, rows = max(1, int(columns)), max(1, int(rows))
    half = max(0.0, float(overlap)) / 2
    boxes = []
    for row in range(rows):
        for column in range(columns):
            step_x, step_y = 1.0 / columns, 1.0 / rows
            boxes.append([
                round(max(0.0, column * step_x - half * step_x), 4),
                round(max(0.0, row * step_y - half * step_y), 4),
                round(min(1.0, (column + 1) * step_x + half * step_x), 4),
                round(min(1.0, (row + 1) * step_y + half * step_y), 4),
            ])
    return boxes


def plan(size: Sequence[int], dpi: float) -> TilePlan:
    """
    Сетка для листа `size` пикселей в `dpi`. Перебираются сетки не больше
    DRAWING_NATIVE_TILES_MAX плиток; у каждой — dpi, в котором её плитка
    помещается в DRAWING_NATIVE_TILES_MAX_SIDE (и не выше
    DRAWING_NATIVE_TILES_DPI: растр 600 dpi ужимается). Берётся сетка с
    наименьшим числом плиток, которая даёт не меньше
    DRAWING_NATIVE_TILES_MIN_DPI; если такой нет — с наибольшим dpi.
    """
    width, height = float(size[0]), float(size[1])
    dpi = float(dpi)
    overlap = max(0.0, float(settings.DRAWING_NATIVE_TILES_OVERLAP))
    limit = float(settings.DRAWING_NATIVE_TILES_MAX_SIDE)
    target = min(float(settings.DRAWING_NATIVE_TILES_DPI), dpi)
    floor = min(float(settings.DRAWING_NATIVE_TILES_MIN_DPI), target)
    most = max(1, int(settings.DRAWING_NATIVE_TILES_MAX))
    options = []
    for columns in range(1, most + 1):
        for rows in range(1, most // columns + 1):
            tile_w = width * (1 + overlap) / columns if columns > 1 else width
            tile_h = height * (1 + overlap) / rows if rows > 1 else height
            scale = min(target / dpi, limit / tile_w, limit / tile_h)
            options.append((columns * rows, -(dpi * scale), columns, rows, tile_w, tile_h, scale))
    options.sort()
    good = [o for o in options if -o[1] >= floor - 0.5]
    chosen = good[0] if good else min(options, key=lambda o: (o[1], o[0]))
    _, _, columns, rows, tile_w, tile_h, scale = chosen
    return TilePlan(
        columns=columns, rows=rows, dpi=round(dpi * scale, 1),
        boxes=grid_boxes(columns, rows, overlap),
        max_side=int(min(limit, math.ceil(max(tile_w, tile_h) * scale))),
    )


# ---------------------------------------------------------------- чтение
@dataclass
class TileReading:
    """Надписи всех плиток листа одним списком и учёт прохода."""

    sheet_type: str = "unknown"
    items: List[Dict[str, Any]] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)


def read(client, native: NativeSheet, located: Any, until: Optional[float]) -> Optional[TileReading]:
    """
    Плитки листа, если лист в них нуждается (`wanted`), иначе None. None и
    тогда, когда не прочитана ни одна плитка: прочтение листа остаётся
    прежним. Плитка не начинается, если вызов не успеет до `until` (часы
    `gpu_gate.budget_now()`) или до срока задачи.
    """
    from core.providers.qwen_client import _time_for_call

    reason = wanted(native.size, native.base_dpi, located)
    if not reason:
        return None
    if not _time_for_call(until):
        logger.info("Плитки листа пропущены: бюджет листа исчерпан")
        return None
    image, dpi = native.image()
    layout = plan(image.size, dpi)
    limit = threading.Semaphore(max(1, int(settings.DRAWING_NATIVE_TILES_PARALLEL)))

    def one(box):
        # Срок — после ожидания очереди: дождавшаяся места плитка не
        # начинает вызов, на который времени уже нет.
        with limit:
            if not _time_for_call(until):
                return None
            from core.providers.qwen_client import crop_box

            try:
                return client.read_sheet_tile(crop_box(image, box), layout.max_side)
            except (SoftTimeLimitExceeded, deadline.Cancelled):
                raise
            except Exception as exc:  # noqa: BLE001 — плитка не роняет лист
                logger.warning("Плитка листа не прочитана: %s", exc)
                return None

    workers = min(len(layout.boxes), max(1, int(settings.DRAWING_NATIVE_TILES_PARALLEL)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(contextvars.copy_context().run, one, box) for box in layout.boxes]
        answers = [future.result() for future in futures]

    parts = [(box, answer.annotations) for box, answer in zip(layout.boxes, answers) if answer is not None]
    read_count = len(parts)
    stats = {
        "reason": reason, "grid": [layout.columns, layout.rows], "dpi": layout.dpi,
        "tiles": len(layout.boxes), "read": read_count,
        "cut": sum(1 for answer in answers if answer is not None and answer.cut),
    }
    if not read_count:
        logger.warning("Плитки листа не прочитаны (%d): остаётся прочтение листа", len(layout.boxes))
        return None
    kinds = [answer.sheet_type for answer in answers if answer is not None and answer.sheet_type != "unknown"]
    items = merge(parts)
    stats["items"] = len(items)
    logger.info(
        "Лист плитками в родном разрешении (%s): сетка %dx%d, %.0f dpi, прочитано %d из %d, надписей %d",
        reason, layout.columns, layout.rows, layout.dpi, read_count, len(layout.boxes), len(items),
    )
    return TileReading(
        sheet_type=max(set(kinds), key=kinds.count) if kinds else "unknown",
        items=items, stats=stats,
    )


# ---------------------------------------------------------------- сведение
def key(value: Any) -> str:
    """Запись надписи для сравнения: без пробелов, регистра и разницы «,»/«.»."""
    from core.providers import drawing_text

    text = drawing_text.normalize(str(value or ""))
    return re.sub(r"\s+", "", text).casefold().replace(",", ".")


def _intersection(a: Sequence[float], b: Sequence[float]) -> Optional[List[float]]:
    box = [max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])]
    return box if box[2] > box[0] and box[3] > box[1] else None


def merge(parts: Sequence[Tuple[Sequence[float], Sequence[Dict[str, Any]]]]) -> List[Dict[str, Any]]:
    """
    Надписи плиток `[(рамка плитки, надписи)]` одним списком.

    Надпись плитки совпадает с уже взятой надписью, если запись та же и
    место той надписи (пересечение её плиток) перекрывается с этой плиткой:
    это одна надпись в зоне перекрытия, прочитанная дважды. Каждая взятая
    надпись совпадает с одной надписью плитки, поэтому повторы внутри плитки
    («⌀8» у каждого отверстия) остаются, а та же запись в непересекающихся
    плитках — две надписи. Место надписи (`area`, доли листа) — грубое:
    плитка или пересечение плиток, где её прочли.

    Повторы — по правилу листа: без DRAWING_KEEP_REPEATS лист пишет каждую
    запись один раз, и плитки тоже дают её один раз, где бы она ни стояла;
    иначе плитки добавили бы повторы, которых проход листа не даёт.
    """
    keep_repeats = bool(settings.DRAWING_KEEP_REPEATS)
    taken: List[Dict[str, Any]] = []
    for tile, (box, items) in enumerate(parts):
        for item in items:
            if not isinstance(item, dict) or not str(item.get("value") or "").strip():
                continue
            value_key = key(item.get("value"))
            if not keep_repeats:
                if any(entry["key"] == value_key for entry in taken):
                    continue
                taken.append({"key": value_key, "tiles": {tile}, "area": list(box), "item": item})
                continue
            for entry in taken:
                if entry["key"] != value_key or tile in entry["tiles"]:
                    continue
                common = _intersection(entry["area"], box)
                if common is not None:
                    entry["tiles"].add(tile)
                    entry["area"] = common
                    break
            else:
                taken.append({"key": value_key, "tiles": {tile}, "area": list(box), "item": item})
    return [{**entry["item"], "area": [round(v, 4) for v in entry["area"]]} for entry in taken]


def unpaired(items: Sequence[Dict[str, Any]], located: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Надписи плиток, которым нет пары среди надписей прохода с рамками. Пара —
    та же запись, а при разнице в точке, пробеле или похожей букве — запись
    без знаков (`unplaced_cleanup.loose`); каждая надпись листа — пара одной
    надписи плитки. Надпись с парой не выходит: прочитанное на месте
    побеждает. Голые подписи граф и смешение алфавитов без пары отброшены.
    """
    from core.providers import unplaced_cleanup

    exact: Dict[str, List[int]] = {}
    loose: Dict[str, List[int]] = {}
    for n, item in enumerate(located or []):
        if not isinstance(item, dict):
            continue
        exact.setdefault(key(item.get("value")), []).append(n)
        loose_key = unplaced_cleanup.loose(item.get("value"))
        if loose_key:
            loose.setdefault(loose_key, []).append(n)
    used: set = set()

    def take(indexes: Sequence[int]) -> bool:
        for n in indexes:
            if n not in used:
                used.add(n)
                return True
        return False

    extra: List[Dict[str, Any]] = []
    for item in items:
        if take(exact.get(key(item.get("value")), ())):
            continue
        loose_key = unplaced_cleanup.loose(item.get("value"))
        if loose_key and take(loose.get(loose_key, ())):
            continue
        if unplaced_cleanup.noise(item.get("value")):
            continue
        extra.append(dict(item))
    return extra
