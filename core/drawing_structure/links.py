"""
Структура листа, этапы S5–S6: виды, разнесение элементов и связи.

S5. Виды листа: из `value_links` (DRAWING_VALUE_VIEW_LINKS), иначе из
заголовков видов («Б (4:1)», «А-А») и рамок вырезок видов MinerU той же
страницы; вырезка без заголовка — главный вид. Элемент относится к самому
тесному виду, в котором лежит центр его рамки.

S6. Связи, всё по уже прочитанному:
* размер -> элемент конструкции: связь `link` прохода DRAWING_DIMENSION_LINKS
  (вектор или модель), иначе элемент выводится из вида размера: «⌀8,5 N отв.»
  -> отверстие, «М20×1,5» -> резьба, R -> скругление, «1×45°» -> фаска;
* «4 отв.», «N отв.», «4 места» рядом с размером -> число у размера;
* выноска «п.8» -> пункт 8 требований; «Лицевая сторона п.8» -> поверхность
  и пункт 8; звёздочка у размера -> пункт, который её определяет;
* номер позиции -> строка спецификации с тем же номером;
* шероховатость в правом верхнем углу или «(√)» — общая, иначе — поверхности.

Связь, которой нет в прочтениях, не придумывается: размер без связи
остаётся без `feature_id`, а в учёте — `unlinked_dimensions`.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from core.drawing_structure import typing as typing_
from core.drawing_structure.candidates import PASS_LINKS, PagePool, area, centre, is_place
from core.drawing_structure.merge import Element
from core.models.drawing_sheet import element_anchor

# Привязка соседей: «4 отв.» к размеру, «п.8» к элементу.
NEAR = 0.06
# Заголовок вида не дальше этого от рамки вырезки.
TITLE_REACH = 0.05
# Угол общей шероховатости: правый верхний угол листа.
_GENERAL_CORNER = (0.6, 0.2)
_SPEC_HEADER = re.compile(r"поз|обозначен|наименован|кол", re.IGNORECASE)
_CHANGE_HEADER = re.compile(r"изм|№\s*докум|извещ", re.IGNORECASE)
_MATERIAL_HEADER = re.compile(r"материал|марка|сортамент", re.IGNORECASE)


@dataclass
class Item:
    """Элемент структуры до выдачи: раздел, словарь схемы и исходный элемент."""

    section: str          # views | features | dimensions | positions | notes | tables | symbols | texts
    kind: str             # вид для `id`: view, feature, dimension, …
    data: Dict[str, Any]
    element: Optional[Element] = None
    value: str = ""       # значение для отпечатка
    note_numbers: List[int] = field(default_factory=list)


def distance(a: Sequence[float], b: Sequence[float]) -> float:
    (ax, ay), (bx, by) = centre(a), centre(b)
    return math.hypot(ax - bx, ay - by)


def box_gap(point_box: Sequence[float], frame: Sequence[float]) -> float:
    """Расстояние от центра рамки до области (0 — внутри)."""
    x, y = centre(point_box)
    dx = max(frame[0] - x, 0.0, x - frame[2])
    dy = max(frame[1] - y, 0.0, y - frame[3])
    return math.hypot(dx, dy)


def base(element: Element, pass_extra: Sequence[str] = ()) -> Dict[str, Any]:
    """Общие ключи элемента схемы из сведённого элемента."""
    passes = list(dict.fromkeys(list(element.passes) + list(pass_extra)))
    state = "confirmed" if element.confirmed else ("needs_check" if element.needs_check() else "auto")
    return {
        "bbox": [round(float(v), 4) for v in element.bbox],
        "bbox_exact": bool(element.exact),
        "provenance": {"source": element.source, "passes": passes},
        "confidence": round(min(1.0, max(0.0, element.confidence)), 3),
        "review": {"state": state},
        "alternatives": [dict(a) for a in element.alternatives],
        "field_refs": [{"fragment": f, "field": i} for f, i in element.field_refs],
    }


# ===========================================================================
# S5. Виды
# ===========================================================================

def build_views(pool: PagePool, typed: Sequence[Tuple[Element, typing_.Typed]]) -> List[Item]:
    views: List[Item] = []
    title_bbox = pool.title_bbox
    if pool.value_views:
        for view in pool.value_views:
            bbox = view.get("bbox")
            if not isinstance(bbox, list) or len(bbox) != 4:
                continue
            title_box = view.get("title_bbox") or bbox
            data = {
                "bbox": bbox, "bbox_exact": False,
                "provenance": {"source": "vlm", "passes": ["sheet"]}, "confidence": 0.7,
                "review": {"state": "auto"}, "alternatives": [], "field_refs": [],
                "label": view.get("name") or None, "kind": view.get("kind") or "view",
                "scale": view.get("scale") or None, "_title_box": title_box,
                "title_bbox": title_box if title_box != bbox else None,
            }
            views.append(Item("views", "view", data, value=str(view.get("name") or "")))
    else:
        frames = [f for f in pool.crop_frames
                  if not (title_bbox and _overlap_share(f, title_bbox) > 0.5)]
        used: set = set()
        for element, kind in typed:
            if kind.kind != typing_.VIEW_TITLE or kind.view is None:
                continue
            name, view_kind, scale = kind.view
            bbox, frame_index = list(element.bbox), None
            if is_place(element.bbox):
                gaps = sorted((_title_gap(element.bbox, frame), i) for i, frame in enumerate(frames)
                              if i not in used)
                if gaps and gaps[0][0] <= TITLE_REACH:
                    frame_index = gaps[0][1]
                    used.add(frame_index)
                    bbox = list(frames[frame_index])
            data = base(element)
            data.update({"bbox": bbox, "label": name or None, "title_raw": element.value,
                         "kind": view_kind, "scale": scale or None,
                         "_title_box": list(element.bbox), "_has_area": frame_index is not None,
                         # Рамка вида стала областью вырезки — место заголовка отдельно.
                         "title_bbox": list(element.bbox) if frame_index is not None else None})
            views.append(Item("views", "view", data, element=element, value=element.value))
        rest = [frame for i, frame in enumerate(frames) if i not in used]
        if rest:
            main = max(rest, key=area)
            data = {
                "bbox": main, "bbox_exact": False,
                "provenance": {"source": "vlm", "passes": ["view_crop"]}, "confidence": 0.6,
                "review": {"state": "auto"}, "alternatives": [], "field_refs": [],
                "kind": "main", "_title_box": None, "_has_area": True,
            }
            views.append(Item("views", "view", data, value="main"))
    return views


def _title_gap(title: Sequence[float], frame: Sequence[float]) -> float:
    """Заголовок вида — внутри рамки вырезки или прямо над ней."""
    inside = box_gap(title, frame)
    if inside == 0.0:
        return 0.0
    x, _ = centre(title)
    above = frame[1] - title[3]
    if frame[0] - 0.02 <= x <= frame[2] + 0.02 and -0.01 <= above <= TITLE_REACH:
        return max(0.0, above)
    return inside


def _overlap_share(a: Sequence[float], b: Sequence[float]) -> float:
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    return (w * h) / max(area(a), 1e-9)


def view_of(bbox: Sequence[float], views: Sequence[Item]) -> Optional[Item]:
    """Самый тесный вид с областью, в котором лежит центр рамки."""
    if not is_place(bbox):
        return None
    inside = [v for v in views if v.data.get("_has_area", True) and box_gap(bbox, v.data["bbox"]) == 0.0]
    return min(inside, key=lambda v: area(v.data["bbox"])) if inside else None


# ===========================================================================
# S6. Элементы и связи
# ===========================================================================

def build(pool: PagePool, typed: Sequence[Tuple[Element, typing_.Typed]],
          title_block: Dict[str, Any]) -> Dict[str, List[Item]]:
    """Разделы структуры из типизированных элементов (без `id`, их ставит `identify`)."""
    sections: Dict[str, List[Item]] = {name: [] for name in (
        "views", "features", "dimensions", "positions", "notes", "tables", "symbols", "texts")}
    sections["views"] = build_views(pool, typed)
    counts: List[Tuple[Element, typing_.Typed]] = []
    refs: List[Tuple[Element, typing_.Typed]] = []
    surfaces: List[Tuple[Element, typing_.Typed]] = []
    for element, kind in typed:
        if kind.kind == typing_.DIMENSION:
            sections["dimensions"].append(_dimension(element, kind))
        elif kind.kind == typing_.POSITION:
            data = base(element)
            data.update({"number": kind.number or element.value.strip()})
            if element.link:
                data["leader"] = element.link.get("leader") or []
                data["target_bbox"] = element.link.get("target_bbox")
            sections["positions"].append(Item("positions", "position", data, element, element.value))
        elif kind.kind == typing_.NOTE:
            data = base(element)
            data.update({"number": kind.note_number, "text": kind.note_text,
                         "defines_marker": kind.marker if kind.marker in ("*", "**") else None})
            sections["notes"].append(Item("notes", "note", data, element, element.value,
                                          note_numbers=list(kind.note_refs)))
        elif kind.kind == typing_.SYMBOL:
            data = base(element)
            data.update({"kind": kind.symbol.get("kind", "roughness"), "value": kind.symbol.get("value"),
                         "scope": kind.symbol.get("scope")})
            sections["symbols"].append(Item("symbols", "symbol", data, element, element.value,
                                            note_numbers=list(kind.note_refs)))
        elif kind.kind == typing_.COUNT:
            counts.append((element, kind))
        elif kind.kind == typing_.NOTE_REF:
            refs.append((element, kind))
        elif kind.kind == typing_.SURFACE:
            surfaces.append((element, kind))
        elif kind.kind == typing_.VIEW_TITLE:
            continue    # стал видом
        else:
            data = base(element)
            data["text"] = element.value
            sections["texts"].append(Item("texts", "text", data, element, element.value,
                                          note_numbers=list(kind.note_refs)))

    _transcript_notes(pool, sections["notes"])
    _counts(counts, sections)
    for element, kind in surfaces:
        data = base(element)
        data.update({"kind": "surface", "label": kind.label or element.value, "derived_from": []})
        sections["features"].append(Item("features", "feature", data, element, element.value,
                                         note_numbers=list(kind.note_refs)))
    _derived_features(sections)
    _note_refs(refs, sections)
    _tables(pool, sections)
    _views_of(sections)
    _roughness_scope(sections)
    return sections


def _dimension(element: Element, kind: typing_.Typed) -> Item:
    parsed = dict(kind.dimension)
    data = base(element)
    count = parsed.pop("count", None)
    data.update({
        "value_raw": parsed.get("value_raw") or element.value,
        "kind": parsed.get("kind", "linear"),
        "nominal": parsed.get("nominal"),
        "unit": parsed.get("unit"),
        "tolerance": parsed.get("tolerance"),
        "reference": bool(parsed.get("reference")) or element.nature == "reference",
        "tool_provided": element.nature == "tool_provided",
        "count": _count_value(count),
        "formula": parsed.get("formula"),
    })
    if element.link:
        data["leader"] = element.link.get("leader") or []
        method = element.link.get("method")
        data["link"] = {"method": method if method in ("vector", "vlm") else "rule",
                        "confidence": round(float(element.link.get("confidence") or 0.0), 3)}
        if PASS_LINKS not in data["provenance"]["passes"]:
            data["provenance"]["passes"].append(PASS_LINKS)
    item = Item("dimensions", "dimension", data, element, element.value, note_numbers=list(kind.note_refs))
    item.data["_count_raw"] = count
    return item


def _count_value(count: Any) -> Any:
    """Число повторов для схемы: целое >= 1, иначе запись листа («N»), иначе None."""
    text = " ".join(str(count or "").split())
    if text.isdigit():
        return int(text) if int(text) >= 1 else None
    return text if text and len(text) <= 16 else None


def _counts(counts: Sequence[Tuple[Element, typing_.Typed]], sections: Dict[str, List[Item]]) -> None:
    """«4 отв.», «N отв.» рядом с размером -> число у размера; иначе — надпись."""
    for element, kind in counts:
        target = None
        if is_place(element.bbox):
            near = [(distance(element.bbox, d.data["bbox"]), d) for d in sections["dimensions"]
                    if is_place(d.data["bbox"]) and d.data.get("count") is None
                    and not d.data.get("_count_raw")]
            near = [pair for pair in near if pair[0] <= NEAR]
            preferred = [p for p in near if p[1].data["kind"] in ("diameter", "thread")]
            pool = preferred or near
            if pool:
                target = min(pool, key=lambda p: p[0])[1]
        if target is None:
            data = base(element)
            data["text"] = element.value
            sections["texts"].append(Item("texts", "text", data, element, element.value,
                                          note_numbers=list(kind.note_refs)))
            continue
        count = kind.count or ""
        # «N отв.»: число задано буквой — `count` хранит запись листа ("N").
        target.data["count"] = _count_value(count)
        target.data["_count_raw"] = count
        target.data["_count_text"] = element.value
        target.data["field_refs"].extend({"fragment": f, "field": i} for f, i in element.field_refs)
        target.note_numbers.extend(n for n in kind.note_refs if n not in target.note_numbers)


def _derived_features(sections: Dict[str, List[Item]]) -> None:
    """Элемент конструкции из вида размера и его связи (сканы: геометрии нет)."""
    for item in sections["dimensions"]:
        data = item.data
        kind = None
        label = ((item.element.link or {}).get("target_label") if item.element else "") or ""
        if data["kind"] == "thread":
            kind = "thread"
        elif data["kind"] == "diameter" and (data.get("_count_raw") or "отв" in label.lower()):
            kind = "hole"
        elif data["kind"] == "radius":
            kind = "fillet"
        elif data["kind"] == "chamfer":
            kind = "chamfer"
        if kind is None:
            continue
        target = (item.element.link or {}).get("target_bbox") if item.element else None
        bbox = target if target and len(target) == 4 else data["bbox"]
        same = next((f for f in sections["features"] if f.data["kind"] == kind
                     and f.data.get("_from_link") and target
                     and _overlap_share(f.data["bbox"], bbox) > 0.5), None)
        if same is not None:
            same.data["_derived"].append(item)
            item.data["_feature"] = same
            continue
        feature = {
            "bbox": list(bbox), "bbox_exact": False,
            "provenance": {"source": data["provenance"]["source"], "passes": list(data["provenance"]["passes"])},
            "confidence": data["confidence"], "review": {"state": "auto"}, "alternatives": [],
            "field_refs": [], "kind": kind, "label": label or None,
            "count": data.get("count"), "_derived": [item], "_from_link": bool(target),
        }
        if data.get("_count_text"):
            # Запись числа элементов как на листе: «8 отв.», «N отв.».
            feature["label"] = data["_count_text"]
        feat = Item("features", "feature", feature, value=f"{kind}:{data['value_raw']}")
        sections["features"].append(feat)
        item.data["_feature"] = feat


def _note_refs(refs: Sequence[Tuple[Element, typing_.Typed]], sections: Dict[str, List[Item]]) -> None:
    """Выноска «п.8» -> ближайший размер, поверхность или знак; иначе — надпись."""
    for element, kind in refs:
        target = None
        if is_place(element.bbox):
            pool = [i for name in ("features", "dimensions", "symbols") for i in sections[name]
                    if is_place(i.data["bbox"])]
            near = [(distance(element.bbox, i.data["bbox"]), i) for i in pool]
            near = [pair for pair in near if pair[0] <= NEAR]
            if near:
                target = min(near, key=lambda p: (p[1].section != "features", p[0]))[1]
        if target is None:
            data = base(element)
            data["text"] = element.value
            sections["texts"].append(Item("texts", "text", data, element, element.value,
                                          note_numbers=list(kind.note_refs)))
            continue
        target.note_numbers.extend(n for n in kind.note_refs if n not in target.note_numbers)
        target.data["field_refs"].extend({"fragment": f, "field": i} for f, i in element.field_refs)


def _transcript_notes(pool: PagePool, notes: List[Item]) -> None:
    """Пункты требований из строк листа, которых нет среди полей."""
    have = {item.data.get("number") for item in notes}
    for number, text in typing_.transcript_notes(pool.transcript):
        if number in have:
            continue
        have.add(number)
        marker = typing_.marker_of_note(text)
        data = {
            "bbox": [0.0, 0.0, 1.0, 1.0], "bbox_exact": False,
            "provenance": {"source": "vlm", "passes": ["transcribe"]}, "confidence": 0.6,
            "review": {"state": "auto"}, "alternatives": [], "field_refs": [],
            "number": number, "text": text, "defines_marker": marker if marker in ("*", "**") else None,
        }
        notes.append(Item("notes", "note", data, value=f"{number}. {text}",
                          note_numbers=typing_.note_refs(text)))


def _tables(pool: PagePool, sections: Dict[str, List[Item]]) -> None:
    for table in pool.tables:
        headers = list(table["headers"])
        rows = [list(r) for r in table["rows"]]
        if not headers and rows and any(_SPEC_HEADER.search(c) for c in rows[0]):
            headers = rows.pop(0)
        joined = " ".join(headers) + " " + table["title"]
        if _CHANGE_HEADER.search(joined):
            kind = "change"
        elif _SPEC_HEADER.search(joined):
            kind = "spec"
        elif _MATERIAL_HEADER.search(joined):
            kind = "material"
        else:
            kind = "other"
        bbox = table["bbox"] or [0.0, 0.0, 1.0, 1.0]
        data = {
            "bbox": bbox, "bbox_exact": False,
            "provenance": {"source": "vlm", "passes": ["table"]}, "confidence": 0.7,
            "review": {"state": "auto"}, "alternatives": [], "field_refs": [],
            "kind": kind, "headers": headers,
            "rows": [{"cells": [{"text": c} for c in row]} for row in rows],
        }
        sections["tables"].append(Item("tables", "table", data, value=table["title"] or joined.strip()))


def _views_of(sections: Dict[str, List[Item]]) -> None:
    views = sections["views"]
    if not views:
        return
    for name in ("dimensions", "positions", "features"):
        for item in sections[name]:
            view = view_of(item.data["bbox"], views)
            item.data["_view"] = view


def _roughness_scope(sections: Dict[str, List[Item]]) -> None:
    for item in sections["symbols"]:
        if item.data.get("kind") != "roughness" or item.data.get("scope"):
            continue
        x1, _, _, y2 = item.data["bbox"]
        corner = is_place(item.data["bbox"]) and x1 >= _GENERAL_CORNER[0] and y2 <= _GENERAL_CORNER[1]
        item.data["scope"] = "general" if corner else "surface"
        if not corner:
            near = [(distance(item.data["bbox"], f.data["bbox"]), f) for f in sections["features"]
                    if f.data["kind"] == "surface" and is_place(f.data["bbox"])]
            near = [pair for pair in near if pair[0] <= NEAR * 1.5]
            if near:
                item.data["_feature"] = min(near, key=lambda p: p[0])[1]


# ===========================================================================
# Отпечатки, `id` и ссылки по `id`
# ===========================================================================

def identify(sections: Dict[str, List[Item]], title_block: Dict[str, Any]) -> None:
    """`id` в порядке чтения (сверху вниз, слева направо) и `anchor` у каждого элемента."""
    for name, items in sections.items():
        items.sort(key=lambda i: (int(i.data["bbox"][1] * 64), i.data["bbox"][0]))
        for n, item in enumerate(items, start=1):
            item.data["id"] = f"{item.kind}-{n}"
            item.data["anchor"] = element_anchor(item.kind, item.value or "", item.data["bbox"])
    fields = (title_block or {}).get("fields") or {}
    for n, (key, typed) in enumerate(fields.items(), start=1):
        typed["id"] = f"title-{n}"
        box = typed.get("bbox") or title_block.get("bbox") or [0.0, 0.0, 1.0, 1.0]
        typed["anchor"] = element_anchor("title", f"{key}={typed.get('value') or ''}", box)


def resolve(sections: Dict[str, List[Item]]) -> List[str]:
    """Ссылки между элементами по `id`; возвращает предупреждения о несвязанном."""
    warnings: List[str] = []
    notes_by_number: Dict[int, Item] = {}
    for item in sections["notes"]:
        number = item.data.get("number")
        if number is not None and number not in notes_by_number:
            notes_by_number[number] = item
        item.data["refs"] = {"notes": [], "dimensions": [], "features": [], "symbols": []}
    markers = {item.data.get("defines_marker"): item for item in sections["notes"] if item.data.get("defines_marker")}

    for item in sections["notes"]:
        for number in item.note_numbers:
            target = notes_by_number.get(number)
            if target is not None and target is not item:
                item.data["refs"]["notes"].append(target.data["id"])
    missing = 0
    for name in ("dimensions", "features", "symbols", "positions", "texts"):
        for item in sections[name]:
            ids: List[str] = []
            for number in item.note_numbers:
                target = notes_by_number.get(number)
                if target is None:
                    missing += 1
                    continue
                ids.append(target.data["id"])
                bucket = {"dimensions": "dimensions", "features": "features", "symbols": "symbols"}.get(name)
                if bucket:
                    target.data["refs"][bucket].append(item.data["id"])
            if name == "dimensions":
                raw = item.data["value_raw"]
                marker = "**" if "**" in raw else ("*" if item.data.get("reference") and "*" in raw else None)
                if marker and marker in markers:
                    ids.append(markers[marker].data["id"])
                    markers[marker].data["refs"]["dimensions"].append(item.data["id"])
                item.data["note_refs"] = list(dict.fromkeys(ids))
    if missing:
        warnings.append("DRAWING_STRUCTURE_NOTE_MISSING")

    for name in ("dimensions", "positions", "features"):
        for item in sections[name]:
            view = item.data.pop("_view", None)
            item.data["view_id"] = view.data["id"] if view is not None else None
    for item in sections["dimensions"]:
        feature = item.data.pop("_feature", None)
        item.data["feature_id"] = feature.data["id"] if feature is not None else None
    for feature in sections["features"]:
        derived = feature.data.pop("_derived", [])
        feature.data["derived_from"] = [d.data["id"] for d in derived]
        if derived and feature.data.get("view_id") is None:
            feature.data["view_id"] = derived[0].data.get("view_id")
    for item in sections["symbols"]:
        feature = item.data.pop("_feature", None)
        item.data["feature_id"] = feature.data["id"] if feature is not None else None
    _spec_rows(sections)
    return warnings


def _spec_rows(sections: Dict[str, List[Item]]) -> None:
    """Номер позиции -> строка спецификации с тем же номером в графе «Поз.»."""
    specs = [t for t in sections["tables"] if t.data["kind"] == "spec"]
    for item in sections["positions"]:
        number = str(item.data.get("number") or "").strip()
        for table in specs:
            column = next((i for i, h in enumerate(table.data["headers"]) if re.search(r"поз", h, re.I)), None)
            for row_index, row in enumerate(table.data["rows"]):
                cells = [c["text"].strip() for c in row["cells"]]
                candidates = [cells[column]] if column is not None and column < len(cells) else cells[:2]
                if number and number in candidates:
                    item.data["table_row"] = {"table_id": table.data["id"], "row": row_index}
                    break
            if item.data.get("table_row"):
                break
