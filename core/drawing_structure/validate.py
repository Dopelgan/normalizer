"""
Структура листа, этап S7: проверка, учёт и предел размера.

Проверяется то, что можно проверить без картинки: рамки внутри листа и
рамки формата, уникальность `id`, пропуски в нумерации пунктов требований,
позиции без строки спецификации, звёздочка без пункта, который её
определяет, читаемость масштаба, «лист ≤ листов». Нарушение не роняет
лист, а даёт предупреждение `DRAWING_STRUCTURE_*` (для `fragment.warnings`)
и, где это про значение, — `review.state = needs_check`.

Итог проходит модель схемы `drawing-sheet-v1` и предел 48 КБ
(`cap_sheet_size`: сначала линии выносок, затем надписи). Не уложившаяся
или не прошедшая схему структура не отдаётся вовсе.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from pydantic import ValidationError

from core.drawing_structure.links import Item
from core.drawing_structure.title import sheet_numbers
from core.models.drawing_sheet import DrawingSheet, cap_sheet_size

logger = logging.getLogger(__name__)

W_OUTSIDE = "DRAWING_STRUCTURE_OUTSIDE_FRAME"
W_DUPLICATE_ID = "DRAWING_STRUCTURE_DUPLICATE_ID"
W_NOTE_GAP = "DRAWING_STRUCTURE_NOTE_GAP"
W_POSITION = "DRAWING_STRUCTURE_POSITION_UNMATCHED"
W_MARKER = "DRAWING_STRUCTURE_MARKER_UNDEFINED"
W_SCALE = "DRAWING_STRUCTURE_SCALE_UNREADABLE"
W_SHEETS = "DRAWING_STRUCTURE_SHEET_COUNT"
W_INVALID = "DRAWING_STRUCTURE_INVALID"
W_TRUNCATED = "DRAWING_STRUCTURE_TRUNCATED"
W_TOO_LARGE = "DRAWING_STRUCTURE_TOO_LARGE"

_SCALE = re.compile(r"^\s*\d+(?:,\d+)?\s*:\s*\d+(?:,\d+)?\s*$")
# Рамка элемента может чуть выходить за рамку формата: надпись на линии рамки.
_FRAME_MARGIN = 0.02


def check(sections: Dict[str, List[Item]], title_block: Dict[str, Any],
          frame: Dict[str, Any], sheet_type: str) -> List[str]:
    """Проверки S7 над собранными разделами; меняет `review.state` там, где спорно значение."""
    warnings: List[str] = []
    ids = [item.data["id"] for items in sections.values() for item in items]
    ids += [typed["id"] for typed in (title_block.get("fields") or {}).values()]
    if len(ids) != len(set(ids)):
        warnings.append(W_DUPLICATE_ID)

    inner = frame.get("inner_frame")
    if isinstance(inner, list) and len(inner) == 4:
        x1, y1, x2, y2 = (float(v) for v in inner)
        outside = 0
        for items in sections.values():
            for item in items:
                bx1, by1, bx2, by2 = item.data["bbox"]
                if (bx1 < x1 - _FRAME_MARGIN or by1 < y1 - _FRAME_MARGIN
                        or bx2 > x2 + _FRAME_MARGIN or by2 > y2 + _FRAME_MARGIN):
                    outside += 1
        if outside:
            warnings.append(W_OUTSIDE)

    # Сноски «1)» нумеруются отдельно и пропуском в пунктах не считаются.
    numbers = sorted({item.data["number"] for item in sections["notes"]
                      if item.data.get("number") and item.data.get("kind") != "footnote"})
    if numbers and numbers != list(range(1, numbers[-1] + 1)):
        warnings.append(W_NOTE_GAP)

    specs = [t for t in sections["tables"] if t.data["kind"] == "spec"]
    if specs and any(not item.data.get("table_row") for item in sections["positions"]):
        warnings.append(W_POSITION)

    defined = {item.data.get("defines_marker") for item in sections["notes"]}
    for item in sections["dimensions"]:
        if item.data.get("reference") and "*" in item.data["value_raw"]:
            marker = "**" if "**" in item.data["value_raw"] else "*"
            if marker not in defined:
                warnings.append(W_MARKER)
                break

    fields = title_block.get("fields") or {}
    scale = fields.get("scale")
    if scale is not None and not _SCALE.match(str(scale.get("value") or "")):
        warnings.append(W_SCALE)
        scale.setdefault("review", {})["state"] = "needs_check"
    sheet, sheets = sheet_numbers(fields)
    if sheet is not None and sheets is not None and sheet > sheets:
        warnings.append(W_SHEETS)
        for key in ("sheet", "sheets"):
            fields[key].setdefault("review", {})["state"] = "needs_check"
    return list(dict.fromkeys(warnings))


def _clean(data: Any) -> Any:
    """Служебные ключи сборки (с «_») из выдачи — прочь."""
    if isinstance(data, dict):
        return {k: _clean(v) for k, v in data.items() if not str(k).startswith("_")}
    if isinstance(data, list):
        return [_clean(v) for v in data]
    return data


def stats(sections: Dict[str, List[Item]], title_block: Dict[str, Any], merged: int) -> Dict[str, Any]:
    elements = [item.data for items in sections.values() for item in items]
    elements += list((title_block.get("fields") or {}).values())
    return {
        "elements": len(elements),
        "merged_duplicates": merged,
        "alternatives": sum(len(e.get("alternatives") or []) for e in elements),
        "needs_check": sum(1 for e in elements if (e.get("review") or {}).get("state") == "needs_check"),
        "unlinked_dimensions": sum(1 for item in sections["dimensions"]
                                   if not item.data.get("feature_id") and not item.data.get("link")),
        "truncated": False,
    }


def finish(page: int, sheet_type: str, frame: Dict[str, Any], title_block: Dict[str, Any],
           sections: Dict[str, List[Item]], merged: int,
           warnings: List[str]) -> Tuple[Optional[DrawingSheet], List[str]]:
    """Схема и предел размера. `None` — структура не прошла и не отдаётся."""
    warnings = list(warnings) + check(sections, title_block, frame, sheet_type)
    raw = {
        "schema": "drawing-sheet-v1",
        "page": page,
        "sheet_type": sheet_type,
        "frame": frame,
        "title_block": title_block if (title_block.get("fields") or title_block.get("roles")
                                       or title_block.get("changes")) else None,
        **{name: [item.data for item in items] for name, items in sections.items()},
        "stats": stats(sections, title_block, merged),
    }
    try:
        sheet = DrawingSheet.model_validate(_clean(raw))
    except ValidationError as exc:
        logger.warning("Структура листа %s не прошла схему: %s", page, exc.errors()[:3])
        return None, list(dict.fromkeys(warnings + [W_INVALID]))
    capped, fits = cap_sheet_size(sheet)
    if not fits:
        return None, list(dict.fromkeys(warnings + [W_TOO_LARGE]))
    if capped.stats.truncated:
        warnings.append(W_TRUNCATED)
    return capped, list(dict.fromkeys(warnings))
