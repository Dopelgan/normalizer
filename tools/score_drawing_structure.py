"""
Оценка структуры чертёжного листа против эталона структуры.

Зачем. Плоский список полей чертежа (`structured_drawing_fields`) не несёт
связей и мест: размер из вырезки вида лежит в координатах вырезки, штамп
читается дважды, ссылка «п.4» выдаётся размером. Модель листа
`drawing-sheet-v1` это исправляет, и её нужно мерить одинаково на плоском
выходе (база) и на структуре (флаг включён).

Две команды, только стандартная библиотека, данных заказчика в коде нет:

    build  --reference reference-sheets.jsonl --out reference-structure.jsonl
        эталон структуры по проверенным надписям листа: типизированный штамп,
        виды, примечания, ссылки «п.N», маркеры «*», до 10 размеров со связями,
        позиции. Всё, что правилами не выводится однозначно, помечено
        `needs_human` и перечислено в `human_labelling` строки листа.

    score  --reference reference-sheets.jsonl --structure reference-structure.jsonl
           --readings readings.jsonl [--json итог.json]
        чтения: строка на (лист, прогон) с `items` (плоский выход) или с
        `sheet` (объект `drawing-sheet-v1`, проецируется в плоские надписи).
        Метрики по плечу и по виду листа:
          recall/precision численных надписей (как score_sheets.py, 0,05 листа);
          placement — доля сопоставленных надписей, центр которых внутри
            эталонной рамки с полем 2 % листа;
          dup_place — надписи с тем же значением и местом, что у предыдущей;
          dup_excess — копии значения сверх числа в эталоне (вырезки видов);
          title_dup — повтор значения штампа («Обозначение: X» и «X»);
          variants — пары близких разных прочтений на одном месте;
          noteref_as_size, balloon_as_size — ссылки «п.N» и номера позиций,
            выданные размером; position_recall;
          title_typed — доля ключей штампа с верным значением;
          link_precision / link_coverage — связи структуры против эталона.
"""

from __future__ import annotations

import argparse
import collections
import difflib
import json
import math
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

NUMERIC = {"size", "tolerance", "thread", "radius", "roughness"}
PAD = 0.02  # поле вокруг эталонной рамки при проверке места, доля листа
SAME_PLACE = 0.02  # центры ближе — одно место
MATCH_DISTANCE = 0.05  # как score_sheets.py
NEAR = 0.05  # «рядом» для связей по близости
MAX_DIMENSIONS = 10

_LOOKALIKE = str.maketrans({
    "А": "A", "В": "B", "Е": "E", "К": "K", "М": "M", "Н": "H", "О": "O", "Р": "P", "С": "C",
    "Т": "T", "Х": "X", "У": "Y", "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "х": "x",
    "у": "y", "к": "k", "м": "m", "т": "t", "н": "h", "в": "b",
    "−": "-", "–": "-", "—": "-", "‐": "-", "‑": "-",
    "Ø": "⌀", "ø": "⌀", "∅": "⌀", "×": "x", " ": " ",
})


def norm(value: object) -> str:
    """Нормализация как в score_sheets.py: похожие буквы, минусы, запятая, пробелы, регистр."""
    text = str(value).translate(_LOOKALIKE).casefold().translate(_LOOKALIKE)
    text = re.sub(r"(?<=\d),(?=\d)", ".", text)
    return re.sub(r"\s+", "", text)


def centre(bbox: object) -> Optional[Tuple[float, float]]:
    """Центр рамки; None — нет рамки, рамка всего листа или область больше половины листа."""
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        x1, y1, x2, y2 = (float(v) for v in bbox)
    except (TypeError, ValueError):
        return None
    if not all(0.0 <= v <= 1.0 for v in (x1, y1, x2, y2)) or x2 - x1 > 0.5 or y2 - y1 > 0.5:
        return None
    return (x1 + x2) / 2, (y1 + y2) / 2


def inside(point: Optional[Tuple[float, float]], bbox: object, pad: float = PAD) -> bool:
    if point is None or not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return False
    x1, y1, x2, y2 = (float(v) for v in bbox)
    return x1 - pad <= point[0] <= x2 + pad and y1 - pad <= point[1] <= y2 + pad


def dist(a: Optional[Tuple[float, float]], b: Optional[Tuple[float, float]]) -> float:
    return math.dist(a, b) if a is not None and b is not None else math.inf


def area(bbox: object) -> float:
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return math.inf
    return abs((float(bbox[2]) - float(bbox[0])) * (float(bbox[3]) - float(bbox[1])))


# ---------------------------------------------------------------- разбор надписей

# Ссылка на пункт технических требований: «п.8», «п4», «Лицевая сторона п.8».
NOTE_REF = re.compile(r"(?:^|[\s(])п\.?\s*(\d{1,2})(?![\d,.])", re.IGNORECASE)
NOTE_REF_ONLY = re.compile(r"^\s*п\.?\s*\d{1,2}\s*$", re.IGNORECASE)
# Пункт требований: «5. Текст», «1) Текст», «1 * Размеры…».
NOTE_ITEM = re.compile(r"^\s*(\d{1,2})\s*[.)]?\s+(\S.*)$", re.DOTALL)
MARKER = re.compile(r"^\s*(\*{1,3}\d?)\s")
# Заголовок вида: «Вид А», «Разрез Б-Б», «Б (4:1) 4 места», «(1:2,5)».
VIEW_TITLE = re.compile(
    r"^\s*(?:(Вид|Разрез|Сечение)\s+)?([A-ZА-ЯЁ](?:\s*-\s*[A-ZА-ЯЁ])?)?\s*"
    r"(?:\(\s*(\d+(?:[.,]\d+)?\s*:\s*\d+(?:[.,]\d+)?)\s*\))?\s*(.*)$",
    re.DOTALL,
)
PLACES = re.compile(r"(\d+)\s*мест", re.IGNORECASE)
HOLES = re.compile(r"(?:^|\s)(\d+|N)\s*отв\.?", re.IGNORECASE)
NUMBER = r"\d+(?:[.,]\d+)?"
FORMULA = re.compile(r"\(\s*([A-ZА-Я])\s*([+-])\s*(" + NUMBER + r")\s*\)")

# Ключи штампа: подпись поля → ключ структуры.
TITLE_KEYS = {
    "обозначение": "designation", "наименование": "name", "материал": "material",
    "масса": "mass", "масштаб": "scale", "лист": "sheet", "листов": "sheets",
    "формат": "format", "организация": "organisation", "инв. № подл.": "inventory_no",
    "изм.": "change", "литера": "letter",
}
ROLE_KEYS = {
    "разраб.": "developer", "разработал": "developer", "пров.": "checker", "проверил": "checker",
    "т.контр.": "tech_control", "н.контр.": "norm_control", "утв.": "approver",
}
TYPED_KEYS = ("designation", "name", "material", "mass", "scale", "sheet", "sheets", "format")
# Подпись начинается с буквы: «1:1» — масштаб, а не подпись «1».
_LABEL = re.compile(r"^\s*([^\W\d_][^:]{0,19}):\s*(.+)$", re.DOTALL)


def split_label(value: str) -> Tuple[Optional[str], str]:
    """«Обозначение: ТАДУ 1» → ("обозначение", "ТАДУ 1"); без подписи → (None, value)."""
    found = _LABEL.match(value)
    if not found:
        return None, value.strip()
    return found.group(1).strip().casefold(), found.group(2).strip()


def parse_dimension(value: str) -> Dict[str, Any]:
    """Вид размера, номинал, допуск, маркер справочного размера и формула."""
    text = value.strip()
    flat = text.replace(" ", "")
    kind = "linear"
    if re.match(r"^[⌀ØøΦ∅]", flat):
        kind = "diameter"
    elif re.match(r"^R\d", flat, re.IGNORECASE):
        kind = "radius"
    elif re.match(r"^[MМ]\d", flat):
        kind = "thread"
    elif re.search(r"[x×х]\s*45\s*°", text):
        kind = "chamfer"
    elif "°" in text:
        kind = "angle"
    marker = re.search(r"(\*{1,3}\d?)\s*$", flat)
    formula = FORMULA.search(text)
    nominal = None
    if formula is None:
        number = re.search(NUMBER, flat)
        nominal = float(number.group(0).replace(",", ".")) if number else None
    tolerance: Dict[str, Any] = {}
    if found := re.search(r"±\s*(" + NUMBER + ")", text):
        size = float(found.group(1).replace(",", "."))
        tolerance = {"upper": size, "lower": -size}
    elif found := re.search(r"([+-]" + NUMBER + r")\s*/?\s*([+-]" + NUMBER + ")", flat):
        values = sorted(float(v.replace(",", ".")) for v in found.groups())
        tolerance = {"upper": values[1], "lower": values[0]}
    # Поле допуска идёт сразу за номиналом: «⌀20H7», «15h14»; «x» — знак резьбы, не поле.
    letters = "A-HJKMNP-VYZa-hjkmnp-vyz"
    if fit := re.search(rf"(?<=\d)([{letters}]{{1,2}}\d{{1,2}})(?![\d.])", flat):
        tolerance["fit"] = fit.group(1)
    return {
        "kind": kind, "nominal": nominal, "tolerance": tolerance or None,
        "marker": marker.group(1) if marker else None,
        "formula": f"{formula.group(1)}{formula.group(2)}{formula.group(3)}" if formula else None,
    }


# ---------------------------------------------------------------- эталон структуры


def _item(item: Dict[str, Any]) -> Dict[str, Any]:
    return {"value": item["value"], "bbox": item.get("bbox")}


def _title_block(items: List[Dict[str, Any]],
                 sheet_format: Optional[str]) -> Tuple[Dict, List, List]:
    """Типизированный штамп: значение из сводного поля, рамка из ячейки с тем же значением."""
    cells = [i for i in items if i["category"] in ("title_block", "material")
             and centre(i.get("bbox")) is not None]
    by_key: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
    roles: List[Dict[str, Any]] = []
    for item in items:
        if item["category"] not in ("title_block", "material"):
            continue
        label, value = split_label(item["value"])
        if label is None:
            continue
        if label in ROLE_KEYS:
            roles.append({"role": ROLE_KEYS[label], "name": value, "personal": True})
        elif label in TITLE_KEYS:
            by_key[TITLE_KEYS[label]].append({"value": value, "bbox": item.get("bbox")})
    # Ячейки без сводного поля: «Лист 1», «Листов 12», «Формат А3».
    for item in cells:
        for key, pattern in (("sheets", r"^Листов\s*(\d+)$"), ("sheet", r"^Лист\s*(\d+)$"),
                             ("format", r"^Формат\s*(А\d(?:х\d)?)$")):
            if (found := re.match(pattern, item["value"].strip())) and not by_key.get(key):
                by_key[key].append({"value": found.group(1), "bbox": item["bbox"]})
    if sheet_format and not by_key.get("format"):
        by_key["format"].append({"value": sheet_format, "bbox": None})
    fields: Dict[str, Any] = {}
    human: List[str] = []
    for key in TYPED_KEYS:
        variants = by_key.get(key) or []
        values = list(dict.fromkeys(v["value"] for v in variants))
        if not values:
            fields[key] = {"value": None, "bbox": None, "needs_human": True}
            human.append(f"title_block.{key}")
            continue
        value = values[0]
        cell = next((c for c in cells if norm(c["value"]) == norm(value)), None)
        if cell is None and key == "material":
            parts = [c for c in cells if norm(c["value"]) and norm(c["value"]) in norm(value)
                     and c["category"] == "title_block"]
            cell = min(parts, key=lambda c: area(c["bbox"])) if parts else None
        field = {"value": value, "bbox": cell["bbox"] if cell else None,
                 "needs_human": len(values) > 1}
        if len(values) > 1:
            field["alternatives"] = values[1:]
            human.append(f"title_block.{key}")
        fields[key] = field
    return fields, roles, human


def _views(items: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Виды по заголовкам с масштабом или словом «Вид/Разрез/Сечение»; буква — указатель."""
    views: Dict[str, Dict[str, Any]] = {}
    pointers: List[Dict[str, Any]] = []
    for item in items:
        if item["category"] not in ("callout", "note"):
            continue
        found = VIEW_TITLE.match(item["value"])
        if not found:
            continue
        word, label, scale, rest = found.groups()
        if not (word or scale):
            if label and not rest.strip():
                pointers.append({"label": label.replace(" ", ""), "bbox": item.get("bbox")})
            continue
        if not label and not scale:
            continue
        label = (label or "").replace(" ", "")
        kind = "view"
        if word in ("Разрез", "Сечение"):
            kind = "section"
        elif scale:
            left, right = (float(v.replace(",", ".")) for v in scale.split(":"))
            if left > right:
                kind = "detail"
        places = PLACES.search(rest or "")
        # Буква вида читается то латиницей, то кириллицей — один вид.
        key = norm(label) if label else f"scale:{norm(scale)}"
        view = views.get(key)
        entry = {"label": label or None, "title_raw": item["value"], "kind": kind,
                 "scale": scale.replace(" ", "") if scale else None,
                 "count": int(places.group(1)) if places else None,
                 "title_bbox": item.get("bbox"), "bbox": None, "needs_human": ["bbox"]}
        if view is None:
            views[key] = entry
            continue
        # Один вид прочитан дважды: оставить короткий заголовок, число мест — из любого.
        view["count"] = view["count"] or entry["count"]
        if area(entry["title_bbox"]) < area(view["title_bbox"]):
            view["title_bbox"], view["title_raw"] = entry["title_bbox"], entry["title_raw"]
    result = list(views.values())
    if result:
        result.insert(0, {"label": None, "title_raw": None, "kind": "main", "scale": None,
                          "count": None, "title_bbox": None, "bbox": None,
                          "needs_human": ["bbox"]})
    return result, pointers


def _notes(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Пункты требований по номеру; из нескольких прочтений — самое узкое (один пункт)."""
    found: Dict[int, List[Dict[str, Any]]] = collections.defaultdict(list)
    for item in items:
        if item["category"] != "note":
            continue
        match = NOTE_ITEM.match(item["value"])
        if not match:
            continue
        number, text = int(match.group(1)), match.group(2).strip()
        if number == 0 or (len(text) < 8 and not text.startswith("*")):
            continue  # «2 блок», «12» — не пункт требований
        found[number].append({"text": text, "bbox": item.get("bbox"), "raw": item["value"]})
    notes = []
    for number in sorted(found):
        best = min(found[number], key=lambda n: area(n["bbox"]))
        marker = MARKER.match(best["text"] + " ")
        notes.append({"number": number, "text": best["text"], "bbox": best["bbox"],
                      "defines_marker": marker.group(1) if marker else None})
    return notes


def build_sheet(row: Dict[str, Any]) -> Dict[str, Any]:
    """Строка эталона структуры для одного листа."""
    items = [i for i in row.get("items", []) if i.get("check") != "unreadable"]
    human: List[str] = []
    title, roles, title_human = _title_block(items, row.get("format"))
    human += title_human
    views, pointers = _views(items)
    if views:
        human.append("views[*].bbox")
    notes = _notes(items)
    note_numbers = {n["number"] for n in notes}
    markers = {n["defines_marker"]: n["number"] for n in notes if n["defines_marker"]}

    links: List[Dict[str, Any]] = []
    note_refs: List[Dict[str, Any]] = []
    for item in items:
        if item["category"] not in ("callout", "size", "note", "tolerance"):
            continue
        if item["category"] == "note" and NOTE_ITEM.match(item["value"]):
            continue
        for number in {int(n) for n in NOTE_REF.findall(" " + item["value"])}:
            ref = {**_item(item), "note": number, "note_exists": number in note_numbers}
            note_refs.append(ref)
            links.append({"type": "note_ref", "from": _item(item), "to": number,
                          "status": "derived" if ref["note_exists"] else "needs_human"})

    holes = [i for i in items if i["category"] in ("callout", "note", "size")
             and HOLES.search(" " + i["value"])]
    candidates = []
    for item in items:
        if item["category"] not in NUMERIC - {"roughness"} or NOTE_REF_ONLY.match(item["value"]):
            continue
        parsed = parse_dimension(item["value"])
        here = centre(item.get("bbox"))
        refs = sorted({r["note"] for r in note_refs if dist(here, centre(r["bbox"])) <= NEAR})
        hole = min(holes, key=lambda h: dist(here, centre(h.get("bbox"))), default=None)
        if hole is not None and dist(here, centre(hole.get("bbox"))) > NEAR:
            hole = None
        rank = (bool(parsed["tolerance"]) + bool(parsed["marker"]) + bool(parsed["formula"])
                + (parsed["kind"] != "linear") + bool(refs) + bool(hole))
        candidates.append((rank, item, parsed, refs, hole))
    candidates.sort(key=lambda c: -c[0])
    dimensions = []
    for n, (_, item, parsed, refs, hole) in enumerate(candidates[:MAX_DIMENSIONS], 1):
        dim_id = f"dimension-{n}"
        feature = {"thread": "thread", "radius": "fillet", "chamfer": "chamfer"}.get(parsed["kind"])
        need = ["view"]
        if parsed["kind"] == "diameter" and hole is not None:
            feature = "hole"
            need.append("feature")
        count = None
        if hole is not None:
            count_raw = HOLES.search(" " + hole["value"]).group(1)
            count = int(count_raw) if count_raw.isdigit() else count_raw
            need.append("count")
        if refs:
            need.append("note_refs")
        dimensions.append({"id": dim_id, "value_raw": item["value"], "bbox": item.get("bbox"),
                           **parsed, "count": count, "feature": feature, "view": None,
                           "note_refs": refs, "needs_human": need})
        source = {"value": item["value"], "bbox": item.get("bbox")}
        links.append({"type": "view", "from": source, "to": None, "status": "needs_human"})
        if feature:
            links.append({"type": "feature", "from": source, "to": feature,
                          "status": "needs_human" if feature == "hole" else "derived"})
        for number in refs:
            links.append({"type": "note_ref", "from": source, "to": number,
                          "status": "needs_human"})
        if parsed["marker"]:
            target = markers.get(parsed["marker"])
            links.append({"type": "marker", "from": source, "to": target,
                          "status": "derived" if target else "needs_human"})
    if dimensions:
        human.append("dimensions[*].view")

    positions = []
    for item in items:
        if item["category"] != "position":
            continue
        balloon = re.fullmatch(r"\s*(\d{1,3})\s*", item["value"])
        positions.append({"number": int(balloon.group(1)) if balloon else None,
                          "value_raw": item["value"], "bbox": item.get("bbox"),
                          "balloon": bool(balloon), "table_row": None,
                          "needs_human": ["table_row"] if balloon else []})
        if balloon:
            links.append({"type": "spec_row", "from": _item(item), "to": None,
                          "status": "needs_human"})
    if any(p["balloon"] for p in positions):
        human.append("positions[*].table_row")
    undefined = sorted({d["marker"] for d in dimensions if d["marker"]} - set(markers))
    return {
        "schema": "reference-structure-v1", "sheet_id": row["sheet_id"], "kind": row.get("kind"),
        "page": row.get("page"), "status": "derived-unverified",
        "title_block": {"fields": title, "roles": roles},
        "views": views, "view_pointers": pointers, "notes": notes,
        "markers": markers, "undefined_markers": undefined, "note_refs": note_refs,
        "not_sizes": sorted({norm(r["value"]) for r in note_refs
                             if NOTE_REF_ONLY.match(r["value"])}),
        "dimensions": dimensions, "positions": positions, "links": links,
        "human_labelling": sorted(set(human)),
    }


# ---------------------------------------------------------------- проекция структуры


def project(sheet: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """`drawing-sheet-v1` → плоские надписи (как structured_drawing_fields) и связи."""
    items: List[Dict[str, Any]] = []
    links: List[Dict[str, Any]] = []
    category = {"diameter": "size", "linear": "size", "angle": "size", "chamfer": "size",
                "radius": "radius", "thread": "thread"}
    views = {v.get("id"): v.get("label") for v in sheet.get("views") or []}
    features = {f.get("id"): f.get("kind") for f in sheet.get("features") or []}
    title = (sheet.get("title_block") or {}).get("fields") or {}
    for key, field in title.items():
        if isinstance(field, dict) and field.get("value") not in (None, ""):
            items.append({"category": "material" if key == "material" else "title_block",
                          "value": str(field["value"]), "bbox": field.get("bbox"),
                          "title_key": key})
    for view in sheet.get("views") or []:
        if view.get("title_raw"):
            items.append({"category": "callout", "value": view["title_raw"],
                          "bbox": view.get("title_bbox") or view.get("bbox")})
    for dim in sheet.get("dimensions") or []:
        source = {"value": dim.get("value_raw", ""), "bbox": dim.get("bbox")}
        items.append({"category": category.get(dim.get("kind"), "size"), **source})
        if dim.get("view_id") in views:
            links.append({"type": "view", "from": source, "to": views[dim["view_id"]]})
        if dim.get("feature_id") in features:
            links.append({"type": "feature", "from": source, "to": features[dim["feature_id"]]})
        for number in dim.get("note_refs") or []:
            links.append({"type": "note_ref", "from": source, "to": int(number)})
    for note in sheet.get("notes") or []:
        items.append({"category": "note", "value": f"{note.get('number')}. {note.get('text', '')}",
                      "bbox": note.get("bbox")})
        if note.get("defines_marker"):
            links.append({"type": "marker_def", "marker": note["defines_marker"],
                          "to": note.get("number")})
    for position in sheet.get("positions") or []:
        source = {"value": str(position.get("number", "")), "bbox": position.get("bbox")}
        items.append({"category": "position", **source})
        row = position.get("table_row")
        if isinstance(row, dict) and row.get("row") is not None:
            links.append({"type": "spec_row", "from": source, "to": row["row"]})
    for symbol in sheet.get("symbols") or []:
        if symbol.get("kind") == "roughness":
            items.append({"category": "roughness", "value": symbol.get("value", ""),
                          "bbox": symbol.get("bbox")})
    for text in sheet.get("texts") or []:
        source = {"value": text.get("value", ""), "bbox": text.get("bbox")}
        items.append({"category": text.get("category", "callout"), **source})
        for number in text.get("note_refs") or []:
            links.append({"type": "note_ref", "from": source, "to": int(number)})
    return items, links


# ---------------------------------------------------------------- метрики


def pair(reference: List[Dict[str, Any]], reading: List[Dict[str, Any]],
         max_distance: Optional[float]) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    """Пары (эталон, чтение) по категории и значению, ближайшие центры первыми; каждая — раз."""
    pool: Dict[Tuple[str, str], List[int]] = collections.defaultdict(list)
    for n, item in enumerate(reading):
        pool[(item["category"], norm(item["value"]))].append(n)
    options = []
    for r, item in enumerate(reference):
        here = centre(item.get("bbox"))
        for n in pool.get((item["category"], norm(item["value"])), []):
            there = centre(reading[n].get("bbox"))
            d = dist(here, there) if here is not None else 0.0
            if math.isinf(d):
                d = 0.0  # у чтения нет места — сопоставляется, как в score_sheets.py
            if max_distance is None or here is None or d <= max_distance:
                options.append((d, r, n))
    used_r, used_n, result = set(), set(), []
    for _, r, n in sorted(options):
        if r not in used_r and n not in used_n:
            used_r.add(r)
            used_n.add(n)
            result.append((reference[r], reading[n]))
    return result


def duplicates(items: List[Dict[str, Any]]) -> Tuple[int, int]:
    """(повторы на том же месте, повторы штампа: одно значение сводным полем и ячейкой)."""
    seen: List[Tuple[str, Optional[Tuple[float, float]], object]] = []
    same_place = 0
    for item in items:
        key, here = norm(item["value"]), centre(item.get("bbox"))
        if any(k == key and (dist(here, c) <= SAME_PLACE or (c is None and here is None
                                                            and b == item.get("bbox")))
               for k, c, b in seen):
            same_place += 1
        seen.append((key, here, item.get("bbox")))
    title_seen: set = set()
    title_dup = 0
    for item in items:
        if item["category"] != "title_block":
            continue
        _, value = split_label(item["value"])
        key = norm(value)
        if key in title_seen:
            title_dup += 1
        title_seen.add(key)
    return same_place, title_dup


def excess(reference: List[Dict[str, Any]], items: List[Dict[str, Any]]) -> int:
    """Копии значения сверх эталона: вырезка вида повторяет надписи листа."""
    want = collections.Counter((i["category"], norm(i["value"])) for i in reference)
    have = collections.Counter((i["category"], norm(i["value"])) for i in items)
    return sum(max(0, n - max(1, want[k])) for k, n in have.items())


def variants(items: List[Dict[str, Any]]) -> int:
    """Пары близких, но разных прочтений одной категории на одном месте («Постнов»/«Постанов»)."""
    count = 0
    for a_n, a in enumerate(items):
        ca = centre(a.get("bbox"))
        if ca is None:
            continue
        for b in items[a_n + 1:]:
            if b["category"] != a["category"] or dist(ca, centre(b.get("bbox"))) > SAME_PLACE:
                continue
            va, vb = norm(split_label(a["value"])[1]), norm(split_label(b["value"])[1])
            if va != vb and difflib.SequenceMatcher(None, va, vb).ratio() >= 0.7:
                count += 1
    return count


def title_typed(reference: Dict[str, Any], items: List[Dict[str, Any]]) -> Tuple[int, int]:
    fields = (reference.get("title_block") or {}).get("fields") or {}
    got: Dict[str, set] = collections.defaultdict(set)
    for item in items:
        if item.get("title_key"):
            got[item["title_key"]].add(norm(item["value"]))
            continue
        if item["category"] not in ("title_block", "material"):
            continue
        label, value = split_label(item["value"])
        if label in TITLE_KEYS:
            got[TITLE_KEYS[label]].add(norm(value))
    total = right = 0
    for key, field in fields.items():
        if not isinstance(field, dict) or field.get("value") in (None, "") or field.get(
                "needs_human"):
            continue
        total += 1
        # Ключ считается верным, только если выдано одно значение и оно совпадает.
        right += got.get(key) == {norm(field["value"])}
    return right, total


def link_scores(reference: Dict[str, Any], links: List[Dict[str, Any]]) -> Dict[str, int]:
    """Связи структуры против эталона: только эталонные связи с известной целью."""
    known = [link for link in reference.get("links") or []
             if link.get("to") is not None and link.get("status") in ("derived", "confirmed")]
    result = collections.Counter(reference=len(known))
    markers = {link["marker"]: link["to"] for link in links if link["type"] == "marker_def"}
    used: set = set()
    for link in known:
        # Связь маркера утверждается структурой, только если она выдала пункт с этим маркером.
        marker = parse_dimension(link["from"]["value"])["marker"]
        if link["type"] == "marker" and marker in markers:
            result["judged"] += 1
            result["correct"] += markers.get(marker) == link["to"]
    for link in links:
        if link["type"] == "marker_def":
            continue
        here = centre(link["from"].get("bbox"))
        best, best_d = None, math.inf
        for n, ref in enumerate(known):
            if ref["type"] != link["type"] or norm(ref["from"]["value"]) != norm(
                    link["from"]["value"]):
                continue
            d = dist(here, centre(ref["from"].get("bbox")))
            d = 0.0 if math.isinf(d) and here is None else d
            if d <= MATCH_DISTANCE and d < best_d and (n, link["to"]) not in used:
                best, best_d = n, d
        if best is None:
            result["unjudged"] += 1
            continue
        result["judged"] += 1
        if str(known[best]["to"]) == str(link["to"]):
            result["correct"] += 1
            used.add((best, link["to"]))
    return dict(result)


def score_reading(ref_items: List[Dict[str, Any]], ref_struct: Dict[str, Any],
                  row: Dict[str, Any]) -> collections.Counter:
    if isinstance(row.get("sheet"), dict):
        items, links = project(row["sheet"])
    else:
        items, links = list(row.get("items") or []), []
    c: collections.Counter = collections.Counter()
    ref_num = [i for i in ref_items if i["category"] in NUMERIC]
    num = [i for i in items if i["category"] in NUMERIC]
    c.update(num_ref=len(ref_num), num_read=len(num),
             num_hit=len(pair(ref_num, num, MATCH_DISTANCE)))
    placed = [(r, m) for r, m in pair(ref_items, items, None) if centre(r.get("bbox"))]
    c.update(matched=len(placed),
             placed=sum(inside(centre(m.get("bbox")), r["bbox"]) for r, m in placed))
    same_place, title_dup = duplicates(items)
    c.update(emitted=len(items), dup_place=same_place, title_dup=title_dup,
             dup_excess=excess(ref_items, items), variants=variants(items))
    not_sizes = set(ref_struct.get("not_sizes") or [])
    c["noteref_as_size"] = sum(1 for i in num if norm(i["value"]) in not_sizes
                               or NOTE_REF_ONLY.match(i["value"]))
    balloons = [p for p in ref_struct.get("positions") or [] if p.get("balloon")]
    c["balloon_as_size"] = sum(
        1 for i in num for p in balloons
        if norm(i["value"]) == norm(p["value_raw"])
        and dist(centre(i.get("bbox")), centre(p.get("bbox"))) <= SAME_PLACE)
    ref_pos = [i for i in ref_items if i["category"] == "position"]
    c.update(pos_ref=len(ref_pos), pos_hit=len(pair(
        ref_pos, [i for i in items if i["category"] == "position"], MATCH_DISTANCE)))
    right, total = title_typed(ref_struct, items)
    c.update(title_right=right, title_total=total)
    for key, value in link_scores(ref_struct, links).items():
        c["link_" + key] += value
    c["links_emitted"] += len([link for link in links if link["type"] != "marker_def"])
    c["reads"] += 1
    return c


def ratio(a: float, b: float) -> Optional[float]:
    return round(a / b, 3) if b else None


def summarise(c: collections.Counter) -> Dict[str, Any]:
    return {
        "reads": c["reads"],
        "recall": ratio(c["num_hit"], c["num_ref"]),
        "precision": ratio(c["num_hit"], c["num_read"]),
        "placement": ratio(c["placed"], c["matched"]),
        "dup_place": ratio(c["dup_place"], c["emitted"]),
        "dup_excess": ratio(c["dup_excess"], c["emitted"]),
        "title_dup": c["title_dup"], "variants": c["variants"],
        "noteref_as_size": c["noteref_as_size"], "balloon_as_size": c["balloon_as_size"],
        "position_recall": ratio(c["pos_hit"], c["pos_ref"]),
        "title_typed": ratio(c["title_right"], c["title_total"]),
        "links_emitted": c["links_emitted"],
        "link_precision": ratio(c["link_correct"], c["link_judged"]),
        "link_coverage": ratio(c["link_correct"], c["link_reference"]),
    }


def score(reference: Dict[str, Dict[str, Any]], structure: Dict[str, Dict[str, Any]],
          readings: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Итог по плечу: «*» — все листы, далее по виду листа."""
    totals: Dict[str, Dict[str, collections.Counter]] = collections.defaultdict(
        lambda: collections.defaultdict(collections.Counter))
    for row in readings:
        ref = reference.get(row["sheet_id"])
        if ref is None:
            continue
        c = score_reading(ref.get("items") or [], structure.get(row["sheet_id"]) or {}, row)
        arm = row.get("arm", "prod")
        totals[arm]["*"].update(c)
        totals[arm][ref.get("kind") or "?"].update(c)
    return {arm: {kind: summarise(c) for kind, c in sorted(kinds.items())}
            for arm, kinds in sorted(totals.items())}


COLUMNS = ("reads", "recall", "precision", "placement", "dup_place", "dup_excess", "title_dup",
           "variants", "noteref_as_size", "balloon_as_size", "position_recall", "title_typed",
           "links_emitted", "link_precision", "link_coverage")


def _load(path: str) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument("--reference", required=True)
    build.add_argument("--out", required=True)
    run = sub.add_parser("score")
    run.add_argument("--reference", required=True)
    run.add_argument("--structure", required=True)
    run.add_argument("--readings", required=True, nargs="+")
    run.add_argument("--json")
    args = parser.parse_args(argv)
    if args.command == "build":
        rows = [build_sheet(row) for row in _load(args.reference)]
        with open(args.out, "w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        for row in rows:
            print(f"{row['sheet_id']} {row['kind']}: views={len(row['views'])} "
                  f"notes={len(row['notes'])} dims={len(row['dimensions'])} "
                  f"positions={len(row['positions'])} links={len(row['links'])} "
                  f"human={','.join(row['human_labelling'])}")
        return
    reference = {row["sheet_id"]: row for row in _load(args.reference)}
    structure = {row["sheet_id"]: row for row in _load(args.structure)}
    readings = [row for path in args.readings for row in _load(path)]
    result = score(reference, structure, readings)
    fmt = lambda v: "-" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v))  # noqa: E731
    for arm, kinds in result.items():
        print(f"== {arm}")
        print("   " + " ".join(f"{name[:10]:>10s}" for name in ("kind",) + COLUMNS))
        for kind, row in kinds.items():
            cells = [kind] + [row[c] for c in COLUMNS]
            print("   " + " ".join(f"{fmt(v)[:10]:>10s}" for v in cells))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as stream:
            json.dump(result, stream, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
