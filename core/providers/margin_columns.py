"""
Графы поля подшивки и графа 26 листа ЕСКД (флаг DRAWING_MARGIN_COLUMNS).

Зачем отдельный проход. Слева от внутренней рамки, в поле подшивки, стоят
графы 19–25 по ГОСТ 2.104: «Инв. № подл.», «Подп. и дата», «Взам. инв. №»,
«Инв. № дубл.», «Подп. и дата» (снизу вверх) и «Перв. примен.», «Справ. №»
(сверху). Надписи в них повёрнуты на 90°, а в левом верхнем углу рамки
стоит графа 26 — обозначение документа, повёрнутое на 180°. Проход листа
целиком повёрнутый текст не читает: на скане листа детали из всех граф
пришло одно поле «Инв. № подл.: 05» вместо «1012-05», да ещё с рамкой на
половину листа — это рамка области штампа, а не графы.

Здесь графы вырезаются по геометрии ГОСТ от внутренней рамки, ставятся
под чтение (полоса — поворотом по часовой, графа 26 — на 180°) и уходят в
модель одной картинкой и одним вызовом. Рамка каждой графы известна из
той же геометрии, поэтому поле получает рамку своей клетки, а не области.

Модуль — чистые функции над геометрией и ответом модели; сеть — в
`QwenClient.read_margin_columns`, вызов — `DrawingProcessor._margin_columns`.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from core.providers import title_block

# Источник прочтения графы поля подшивки в `readings`.
SOURCE = "margin_column"
# Источник прежнего прочтения, которое заменило чтение графы.
SHEET_SOURCE = "sheet"

# Ширина колонки граф 19–23 и 24–25 по ГОСТ 2.104: 5 мм подпись + 7 мм значение.
COLUMN_MM = 12.0
# Графы по ГОСТ 2.104, высота в мм: нижняя колонка — от нижней линии рамки
# вверх, верхняя — от верхней линии вниз. Ключ — внутреннее имя графы;
# подпись берётся из `title_block.LABELS` (у второй «Подп. и дата» — та же).
LOWER: Tuple[Tuple[str, float], ...] = (
    ("inventory", 25.0),
    ("signed_date", 35.0),
    ("replaced_inventory", 25.0),
    ("inventory_dup", 25.0),
    ("signed_date_dup", 35.0),
)
UPPER: Tuple[Tuple[str, float], ...] = (
    ("first_use", 60.0),
    ("reference", 60.0),
)
# Графа 26 — 70×14 мм в левом верхнем углу внутренней рамки.
TOP_BOX_MM = (70.0, 14.0)

# Рамка у самого края или дальше этого от края — поля подшивки на листе нет
# или рамка найдена неверно (те же пороги, что у `drawing_vision.side_column`).
_MIN_LEFT = 0.004
_MAX_LEFT = 0.2

_LABEL_KEY = {"signed_date_dup": "signed_date"}


def label(key: str) -> str:
    """Подпись графы — каноническая из `title_block.LABELS`."""
    return title_block.label(_LABEL_KEY.get(key, key))


def keys() -> List[str]:
    return [key for key, _ in LOWER] + [key for key, _ in UPPER]


# ===========================================================================
# Геометрия
# ===========================================================================

def inner_frame(layout: Any) -> Optional[List[float]]:
    """
    Внутренняя рамка листа или None. Рамка формата годится, только если она
    отступает от края: на скане `_frame` часто находит обрез листа, и слева
    от него не поле подшивки, а пустота.
    """
    signals = getattr(layout, "signals", None) or {}
    for candidate in (signals.get("inner_frame"), getattr(layout, "frame", None)):
        if candidate and len(candidate) == 4:
            frame = [float(v) for v in candidate]
            if _MIN_LEFT < frame[0] <= _MAX_LEFT and frame[2] > frame[0] and frame[3] > frame[1]:
                return frame
    return None


def frame_mm(frame: Sequence[float], size: Tuple[int, int]) -> Optional[Tuple[float, float]]:
    """Размер внутренней рамки в мм по пропорции (формат ГОСТ 2.301) или None."""
    from core.providers import drawing_vision

    left, top, right, bottom = (float(v) for v in frame)
    width, height = right - left, bottom - top
    if width <= 0 or height <= 0 or not size or min(size) <= 0:
        return None
    aspect = (width * size[0]) / (height * size[1])
    best = min(drawing_vision._FRAMES_MM, key=lambda mm: abs(mm[0] / mm[1] - aspect))
    ratio = best[0] / best[1]
    if abs(ratio - aspect) > drawing_vision._FORMAT_TOLERANCE * ratio:
        return None
    return float(best[0]), float(best[1])


def geometry(layout: Any) -> Optional[Dict[str, Any]]:
    """
    Вырезки и клетки граф в долях листа: {'strip', 'top_box', 'cells'} или None.

    `strip` — поле подшивки от края листа до рамки во всю высоту рамки (то,
    что уходит в модель); `cells` — рамка каждой графы по ГОСТ: 12 мм у
    рамки, высота графы. Клетка никогда не шире и не выше полосы.
    """
    image = getattr(layout, "image", None)
    frame = inner_frame(layout)
    if image is None or frame is None:
        return None
    mm = frame_mm(frame, image.size)
    if mm is None:
        return None
    left, top, right, bottom = frame
    per_x = (right - left) / mm[0]
    per_y = (bottom - top) / mm[1]
    strip = [0.0, round(top, 4), round(left, 4), round(bottom, 4)]
    x1 = max(0.0, left - COLUMN_MM * per_x)
    cells: Dict[str, List[float]] = {}
    y = bottom
    for key, height in LOWER:
        upper = max(top, y - height * per_y)
        cells[key] = [round(x1, 4), round(upper, 4), round(left, 4), round(y, 4)]
        y = upper
    y = top
    for key, height in UPPER:
        lower = min(bottom, y + height * per_y)
        cells[key] = [round(x1, 4), round(y, 4), round(left, 4), round(lower, 4)]
        y = lower
    top_box = [round(left, 4), round(top, 4),
               round(min(right, left + TOP_BOX_MM[0] * per_x), 4),
               round(min(bottom, top + TOP_BOX_MM[1] * per_y), 4)]
    return {"strip": strip, "top_box": top_box, "cells": cells, "frame": frame}


def compose(strip_image, box_image):
    """
    Одна картинка на один вызов: сверху полоса граф, повёрнутая по часовой
    (нижняя графа уходит влево, надписи читаются слева направо), снизу —
    графа 26, повёрнутая на 180°.
    """
    from PIL import Image

    strip = strip_image.rotate(-90, expand=True, fillcolor="white")
    box = box_image.rotate(180, expand=True, fillcolor="white")
    gap = max(8, strip.height // 4)
    width = max(strip.width, box.width)
    canvas = Image.new("RGB", (width, strip.height + gap + box.height), "white")
    canvas.paste(strip.convert("RGB"), (0, 0))
    canvas.paste(box.convert("RGB"), (0, strip.height + gap))
    return canvas


# ===========================================================================
# Ответ модели
# ===========================================================================

_EMPTY = {"", "-", "—", "–", "нет", "пусто", "none", "null", "n/a"}


def _clean(value: Any) -> Tuple[str, bool]:
    """Значение графы и рукописное ли оно: строкой или {'value', 'handwritten'}."""
    handwritten = False
    if isinstance(value, dict):
        handwritten = bool(value.get("handwritten"))
        value = value.get("value")
    if value is None or isinstance(value, (list, tuple, dict, bool)):
        return "", False
    text = " ".join(str(value).split())
    if text.lower() in _EMPTY:
        return "", False
    return text, handwritten


def parse(data: Any) -> Dict[str, Any]:
    """
    Ответ модели -> {'columns': {ключ: (значение, рукописное)}, 'designation'}.

    Пустая графа, неизвестный ключ и значение, которое повторяет подпись
    графы («Инв. № подл.» в графе «Инв. № подл.»), отбрасываются:
    додумывать графу нельзя.
    """
    result: Dict[str, Any] = {"columns": {}, "designation": ""}
    if not isinstance(data, dict):
        return result
    columns = data.get("columns")
    columns = columns if isinstance(columns, dict) else {}
    known = set(keys())
    for key, raw in columns.items():
        key = str(key).strip()
        if key not in known:
            continue
        text, handwritten = _clean(raw)
        if not text or is_caption(text):
            continue
        if key in ("first_use",):
            text = title_block.fix_designation(text)
        # Рукописной может быть только графа, которая заполняется от руки.
        by_hand = handwritten and _LABEL_KEY.get(key, key) in title_block._HAND_FILLED
        result["columns"][key] = (text, by_hand)
    designation, _ = _clean(data.get("top_designation"))
    if designation and not is_caption(designation):
        result["designation"] = title_block.fix_designation(designation)
    return result


# ===========================================================================
# Подписи граф штампа (и для DRAWING_BLOCK_CLEANUP)
# ===========================================================================

# Подписи граф основной надписи и поля подшивки, которые модель отдаёт как
# значения: «Лит.», «Масса», «№ докум.», «Подп.». Сравнение — по `_fold`.
_CAPTIONS = (
    "Лит.", "Литера", "Масса", "Масштаб", "Лист", "Листов", "Формат", "Разраб.",
    "Пров.", "Т.контр.", "Н.контр.", "Утв.", "Изм.", "№ докум.", "Докум.",
    "Подп.", "Дата", "Подп. и дата", "Инв. № подл.", "Инв. № дубл.",
    "Взам. инв. №", "Перв. примен.", "Справ. №", "Копировал", "Кол. уч.",
    "Кол.уч", "№ подл.", "Обозначение", "Наименование", "Материал",
)
CAPTIONS = frozenset(title_block._fold(c) for c in _CAPTIONS + tuple(title_block.LABELS.values()))
_SPLIT = re.compile(r"[\s|;/]+")
# Связка внутри подписи «Подп. и дата».
_JOINERS = frozenset({"и"})


def is_caption(value: str) -> bool:
    """
    Значение — только подпись графы (или ряд подписей «Изм. Лист № докум.»),
    без регистра и знаков препинания.
    """
    folded = title_block._fold(value)
    if not folded:
        return False
    if folded in CAPTIONS:
        return True
    parts = [title_block._fold(p) for p in _SPLIT.split(str(value))]
    parts = [p for p in parts if p]
    return bool(parts) and all(p in CAPTIONS or p in _JOINERS for p in parts) \
        and any(p in CAPTIONS for p in parts)


# ===========================================================================
# Сведение с полями листа
# ===========================================================================

_HAND_MARK = re.compile(r"\s*\((?:рукописно|от руки)\)\s*$", re.IGNORECASE)


def split_value(value: str) -> Tuple[str, str]:
    """«Инв. № подл.: 05 (рукописно)» -> («Инв. № подл.», «05»)."""
    text = str(value or "")
    if ":" not in text:
        return "", _HAND_MARK.sub("", text).strip()
    name, _, rest = text.partition(":")
    return name.strip(), _HAND_MARK.sub("", rest).strip()


def _same(a: str, b: str) -> bool:
    from core.providers.reading_reconcile import notation

    return notation(a) == notation(b)


def _contains(longer: str, shorter: str) -> bool:
    from core.providers.reading_reconcile import notation

    a, b = notation(longer), notation(shorter)
    return bool(b) and len(a) > len(b) and b in a


def _with_reading(field: Dict[str, Any], source: str, value: str) -> List[Dict[str, str]]:
    readings = list(field.get("readings") or [{"source": SHEET_SOURCE, "value": str(field.get("value") or "")}])
    if not any(r.get("value") == value for r in readings):
        readings.append({"source": source, "value": value})
    return readings


def merge(fields: List[Dict[str, Any]], reads: List[Dict[str, Any]]) -> Dict[str, int]:
    """
    Поля граф (`reads`, с рамкой своей клетки) сводятся с полями листа на месте.
    Место графы известно из геометрии, поэтому поле этой графы в любом
    случае получает рамку клетки (прежде у него была рамка всей области).

    * То же значение — поле листа остаётся (`confirmed`).
    * Одно значение содержит другое («05» и «1012-05»: лист прочитал хвост) —
      остаётся более длинное, другое — прочтением в `readings` (`extended`).
    * Значения расходятся иначе — остаётся значение листа: какое из двух
      верно, правилом не решить. Чтение графы — второе прочтение, поле на
      сверку (`conflict`).
    * Графы на листе не было — поле добавляется (`added`).
    """
    stats = {"added": 0, "confirmed": 0, "extended": 0, "conflict": 0}
    used: set = set()
    for read in reads:
        name, value = split_value(read["value"])
        wanted = title_block._fold(name)
        same_label = [
            n for n, item in enumerate(fields)
            if n not in used and item.get("category") in ("title_block", "material")
            and title_block._fold(split_value(item.get("value", ""))[0]) == wanted
        ]
        match = next((n for n in same_label if _same(split_value(fields[n]["value"])[1], value)), None)
        if match is not None:
            fields[match]["bbox"] = list(read["bbox"])
            used.add(match)
            stats["confirmed"] += 1
            continue
        if not same_label:
            fields.append(read)
            used.add(len(fields) - 1)
            stats["added"] += 1
            continue
        index = same_label[0]
        old = fields[index]
        old_value = split_value(old.get("value", ""))[1]
        used.add(index)
        if _contains(value, old_value):
            replaced = dict(read)
            replaced["readings"] = [
                {"source": SHEET_SOURCE, "value": str(old.get("value") or "")},
                {"source": SOURCE, "value": read["value"]},
            ]
            fields[index] = replaced
            stats["extended"] += 1
            continue
        old["readings"] = _with_reading(old, SOURCE, read["value"])
        old["bbox"] = list(read["bbox"])
        if not _contains(old_value, value):
            old["needs_review"] = True
            stats["conflict"] += 1
        else:
            stats["extended"] += 1
    return stats


def check_designation(fields: List[Dict[str, Any]], designation: str) -> str:
    """
    Графа 26 — повтор обозначения: только подтверждает основное, значение
    основного не меняет никогда и второго поля не создаёт.

    `confirmed` — то же обозначение; `conflict` — расходятся: графа 26 —
    второе прочтение в `readings`, поле на сверку; `no_main` — основного
    обозначения нет; `none` — графа 26 пуста.
    """
    if not designation:
        return "none"
    wanted = title_block._fold(title_block.label("designation"))
    index = next((n for n, item in enumerate(fields)
                  if item.get("category") == "title_block"
                  and title_block._fold(split_value(item.get("value", ""))[0]) == wanted), None)
    if index is None:
        return "no_main"
    field = fields[index]
    name, main = split_value(field["value"])
    if _same(main, designation):
        return "confirmed"
    field["readings"] = _with_reading(field, SOURCE, f"{name}: {designation}")
    field["needs_review"] = True
    return "conflict"
