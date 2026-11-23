"""
Структура листа: основная надпись (штамп), часть этапов S2/S4.

Штамп приходит дважды: клетками прохода «лист целиком» (у каждой — своя
рамка, подпись графы чаще отдельной клеткой: «Масса» и под ней «3,74») и
сводной записью отдельного прохода по штампу («Масса: 3,74», у всех граф
одна рамка — весь штамп). Здесь обе половины сводятся по каноническому
ключу графы (`title_block.canonical`): значение — из сводного прохода
(он читает штамп в своём, высоком разрешении), рамка — из клетки. Разные
прочтения одной графы («Глинка»/«Глинко») остаются в `alternatives`.

Подписанты — `roles[]` с `personal: true`: фамилии в граф связей не идут.
"""

from __future__ import annotations

import difflib
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from core.drawing_structure.candidates import (
    PASS_SHEET,
    PASS_TITLE_BLOCK,
    Candidate,
    PagePool,
    centre,
    is_place,
)
from core.providers import reading_reconcile, title_block

# Канонический ключ графы -> имя поля схемы drawing-sheet-v1.
SCHEMA_KEY = {"organization": "organisation", "inventory": "inventory_no"}
ROLES = ("developed", "checked", "tech_control", "norm_control", "approved")
_DATE = re.compile(r"\d{1,2}[.\-/]\d{1,2}[.\-/]\d{2,4}|\d{1,2}[.\-/]\d{4}")
_NOTICE = re.compile(r"(?:[А-ЯЁA-Z]{2,4}[. ])?\d+(?:/\d+)?-\d{2}")
_SIMILAR = 0.75


def fold(text: str) -> str:
    return reading_reconcile.notation(text)


def similar(a: str, b: str) -> bool:
    """Одна надпись в двух прочтениях: та же запись или почти та же."""
    fa, fb = fold(a), fold(b)
    if not fa or not fb:
        return False
    if fa == fb:
        return True
    if reading_reconcile.digits(fa) != reading_reconcile.digits(fb):
        return False
    return difflib.SequenceMatcher(None, fa, fb).ratio() >= _SIMILAR


def _label_key(text: str) -> Optional[str]:
    """Клетка — подпись графы («Масса», «Лит.», «Листов»), а не значение."""
    key = title_block.canonical(text)
    if key is None:
        return None
    folded = title_block._fold(text)
    if folded in title_block._SYNONYMS or len(folded) <= 12:
        return key
    return None


def _role_line(text: str) -> Optional[Tuple[str, str, str]]:
    """«Разраб. Бабанин 10.02.2022» -> ('developed', 'Бабанин', '10.02.2022')."""
    words = text.split()
    if len(words) < 2:
        return None
    key = title_block._signer(words[0])
    if key is None:
        return None
    rest = words[1:]
    date = next((w for w in rest if _DATE.fullmatch(w)), "")
    names = [w for w in rest if w != date and title_block._fold(w) not in title_block._NOT_A_NAME]
    return key, " ".join(names), date


def in_title(candidate: Candidate, pool: PagePool) -> bool:
    """Надпись относится к штампу: категория штампа или место в нём."""
    if candidate.category in ("title_block",):
        return True
    if candidate.category == "material":
        return True
    if candidate.label:
        return True
    return False


def build(candidates: Sequence[Candidate], pool: PagePool) -> Tuple[Dict[str, Any], List[Candidate]]:
    """
    Основная надпись из кандидатов штампа и сводной записи прохода по
    штампу. Возвращает блок `title_block` (поля — словари элементов без
    `id`/`anchor`, их ставит сборка) и кандидатов, не ставших графами
    (уходят в `texts`).
    """
    fields: Dict[str, Dict[str, Any]] = {}
    roles: Dict[str, Dict[str, Any]] = {}
    leftovers: List[Candidate] = []
    label_cells: List[Tuple[str, Candidate]] = []
    value_cells: List[Candidate] = []

    # Сводная запись: «Метка: значение» (поля) и payload прохода по штампу.
    # Рамка, общая для нескольких подписанных граф, — весь штамп, а не клетка.
    shared: Dict[Tuple[float, ...], int] = {}
    for candidate in candidates:
        if candidate.label:
            box = tuple(round(float(v), 3) for v in candidate.bbox)
            shared[box] = shared.get(box, 0) + 1
    for candidate in candidates:
        if candidate.label:
            key = title_block.canonical(candidate.label)
            if key is None:
                key = "other:" + candidate.label
            own_cell = shared.get(tuple(round(float(v), 3) for v in candidate.bbox), 0) < 2
            _put(fields, key, candidate, own_cell and is_place(candidate.bbox))
        elif not is_place(candidate.bbox):
            # Значение без метки с рамкой всего штампа: место неизвестно.
            value_cells.append(candidate)
        else:
            label_key = _label_key(candidate.value)
            if label_key is not None:
                label_cells.append((label_key, candidate))
                continue
            role = _role_line(candidate.value)
            if role is not None:
                _role(roles, role, candidate)
                continue
            value_cells.append(candidate)
    for key, value in pool.title_fields.items():
        if key == title_block.AS_READ:
            continue
        if key not in fields:
            fields[key] = _typed(value, pool.title_bbox, PASS_TITLE_BLOCK, None, 0.8, False)
    for key, value in pool.title_handwritten.items():
        if key not in fields:
            fields[key] = _typed(value, pool.title_bbox, PASS_TITLE_BLOCK, None, 0.6, True)
        else:
            fields[key]["handwritten"] = True

    # Клетки-значения: графа по подписи над/слева, иначе по совпадению значения.
    for cell in value_cells:
        key = _key_by_label(cell, label_cells) if is_place(cell.bbox) else None
        if key is None:
            key = _key_by_value(cell, fields)
        if key is None and cell.category == "material":
            key = "material"
        if key is None:
            leftovers.append(cell)
            continue
        _cell_into(fields, key, cell)

    # Подписанты из сводной записи: графы «Разраб.» и т. д. -> roles.
    for key in ROLES:
        typed = fields.pop(key, None)
        if typed is None:
            continue
        entry = roles.setdefault(key, {"role": key, "name": "", "date": "", "handwritten": False,
                                       "personal": True, "_refs": [], "bbox": None})
        if not entry["name"]:
            entry["name"] = typed["value"]
        elif not similar(entry["name"], typed["value"]):
            entry.setdefault("alternatives", []).append(typed["value"])
        entry["_refs"].extend(typed["field_refs"])
        entry["handwritten"] = entry["handwritten"] or typed["handwritten"]
    signed = fields.pop("signed_date", None)
    if signed is not None:
        for part in re.split(r"\s*;\s*", signed["value"]):
            words = part.split()
            key = title_block._signer(words[0]) if words else None
            date = next((w for w in words if _DATE.fullmatch(w)), "")
            if key in roles and date and not roles[key]["date"]:
                roles[key]["date"] = date
        fields["signed_date"] = signed
    for label_key, cell in label_cells:
        # Подпись графы — не значение, но и не теряется: её поле ссылается на клетку.
        typed = fields.get(label_key)
        if typed is not None and cell.field_ref is not None:
            typed["field_refs"].append(cell.field_ref)

    changes = _changes(fields.pop("changes", None), pool.title_rows)
    placed = [c for c in candidates if is_place(c.bbox) and not c.label]
    title_box = pool.title_bbox or _union([c.bbox for c in placed]) or [0.0, 0.0, 1.0, 1.0]
    for typed in fields.values():
        # Расходящиеся прочтения графы («Глинка»/«Глинко») — на проверку.
        typed["review"] = {"state": "needs_check" if typed["alternatives"] else "auto"}
        if not typed["bbox"]:
            # Клетки графы не нашлось по подписи: клетка с тем же значением
            # (одна «1» на «Лист» и «Листов» одноклеточного листа), иначе — весь штамп.
            twin = next((c for c in placed if similar(c.value, typed["value"])), None)
            typed["bbox"] = list(twin.bbox) if twin is not None else list(title_box)
            typed["bbox_exact"] = False
        typed["field_refs"] = [{"fragment": f, "field": i} for f, i in dict.fromkeys(typed["field_refs"])]
    out_fields = {SCHEMA_KEY.get(key, key): value for key, value in fields.items()}
    block = {
        "bbox": pool.title_bbox or (title_box if placed else None),
        "fields": out_fields,
        "roles": [_role_out(entry) for entry in roles.values()],
        "changes": changes,
    }
    return block, leftovers


def _union(boxes: Sequence[Sequence[float]]) -> Optional[List[float]]:
    if not boxes:
        return None
    return [min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)]


def _typed(value: str, bbox, pass_: str, ref: Optional[Tuple[int, int]], confidence: float,
           handwritten: bool, source: str = "vlm") -> Dict[str, Any]:
    return {
        "value": value, "bbox": list(bbox) if bbox else None, "handwritten": handwritten,
        "bbox_exact": False, "provenance": {"source": source, "passes": [pass_]},
        "confidence": round(confidence, 3), "alternatives": [],
        "field_refs": [ref] if ref is not None else [],
    }


def _put(fields: Dict[str, Dict[str, Any]], key: str, candidate: Candidate, own_cell: bool) -> None:
    current = fields.get(key)
    if current is None:
        # Рамка сводной записи — весь штамп (или боковая колонка), не клетка:
        # место графе даёт клетка листа (`_cell_into`); своя рамка — только у
        # подписанной клетки листа и у текстового слоя.
        own = candidate.bbox if (candidate.exact or own_cell) and is_place(candidate.bbox) else None
        fields[key] = _typed(candidate.value, own, candidate.pass_, candidate.field_ref,
                             candidate.confidence, candidate.handwritten, candidate.source)
        fields[key]["bbox_exact"] = own is not None
        return
    if own_cell and is_place(candidate.bbox) and not current["bbox"]:
        current["bbox"] = list(candidate.bbox)
    _add_reading(current, candidate)


def _add_reading(current: Dict[str, Any], candidate: Candidate) -> None:
    if candidate.field_ref is not None and candidate.field_ref not in current["field_refs"]:
        current["field_refs"].append(candidate.field_ref)
    if candidate.pass_ not in current["provenance"]["passes"]:
        current["provenance"]["passes"].append(candidate.pass_)
    if fold(candidate.value) != fold(current["value"]) and all(
            fold(a["value"]) != fold(candidate.value) for a in current["alternatives"]):
        current["alternatives"].append({"value": candidate.value, "source": candidate.source,
                                        "pass": candidate.pass_, "confidence": candidate.confidence})
    current["handwritten"] = current["handwritten"] or candidate.handwritten


def _cell_into(fields: Dict[str, Dict[str, Any]], key: str, cell: Candidate) -> None:
    current = fields.get(key)
    if current is None:
        fields[key] = _typed(cell.value, cell.bbox if is_place(cell.bbox) else None, cell.pass_,
                             cell.field_ref, cell.confidence, cell.handwritten, cell.source)
        if is_place(cell.bbox):
            fields[key]["bbox_exact"] = cell.exact
        return
    # Значение сводного прохода остаётся; рамка — клетки.
    if is_place(cell.bbox) and (current["bbox"] is None or not is_place(current["bbox"])):
        current["bbox"] = list(cell.bbox)
        current["bbox_exact"] = cell.exact
    _add_reading(current, cell)


def _key_by_label(cell: Candidate, labels: Sequence[Tuple[str, Candidate]]) -> Optional[str]:
    """Графа клетки по подписи прямо над ней (или слева в той же строке)."""
    cx, cy = centre(cell.bbox)
    best: Optional[Tuple[float, str]] = None
    for key, label in labels:
        lx1, ly1, lx2, ly2 = label.bbox
        x1, y1, x2, y2 = cell.bbox
        overlap_x = min(x2, lx2) - max(x1, lx1)
        width = min(x2 - x1, lx2 - lx1)
        above = overlap_x > 0.5 * width and 0 <= y1 - ly2 + 0.005 and y1 - ly2 < 0.03
        lcx, lcy = centre(label.bbox)
        left = abs(lcy - cy) < 0.6 * (y2 - y1 + 0.01) and 0 <= x1 - lx2 + 0.005 and x1 - lx2 < 0.03
        if above or left:
            distance = abs(lcx - cx) + abs(lcy - cy)
            if best is None or distance < best[0]:
                best = (distance, key)
    return best[1] if best else None


def _key_by_value(cell: Candidate, fields: Dict[str, Dict[str, Any]]) -> Optional[str]:
    """Графа клетки по совпадению с единственной графой сводной записи."""
    if len(fold(cell.value)) < 2:
        return None     # «1» есть и в «Лист», и в «Листов»
    matches = [key for key, typed in fields.items() if similar(cell.value, typed["value"])]
    return matches[0] if len(matches) == 1 else None


def _role(roles: Dict[str, Dict[str, Any]], role: Tuple[str, str, str], cell: Candidate) -> None:
    key, name, date = role
    entry = roles.setdefault(key, {"role": key, "name": "", "date": "", "handwritten": False,
                                   "personal": True, "_refs": [], "bbox": None})
    if name and not entry["name"]:
        entry["name"] = name
    elif name and not similar(entry["name"], name):
        entry.setdefault("alternatives", []).append(name)
    if date and not entry["date"]:
        entry["date"] = date
    if cell.field_ref is not None:
        entry["_refs"].append(cell.field_ref)
    if is_place(cell.bbox):
        entry["bbox"] = list(cell.bbox)


def _role_out(entry: Dict[str, Any]) -> Dict[str, Any]:
    out = {"role": entry["role"], "name": entry["name"], "date": entry["date"],
           "handwritten": entry["handwritten"], "personal": True}
    if entry.get("alternatives"):
        out["alternatives"] = list(dict.fromkeys(entry["alternatives"]))
    if entry.get("bbox"):
        out["bbox"] = entry["bbox"]
    if entry["_refs"]:
        out["field_refs"] = [{"fragment": f, "field": i} for f, i in sorted(set(entry["_refs"]))]
    return out


def _changes(aggregated: Optional[Dict[str, Any]], rows: Sequence[Sequence[str]]) -> List[Dict[str, Any]]:
    """Графа изменений: строки таблицы штампа, иначе сводная запись «2 Нов. ТАДУ.409-23; …»."""
    out: List[Dict[str, Any]] = []
    seen = set()
    for row in rows:
        cells = [c for c in row if c]
        notice = next((c for c in cells if _NOTICE.fullmatch(c)), "")
        if not notice or notice in seen:
            continue
        seen.add(notice)
        rev = next((c for c in cells if c.isdigit() and len(c) <= 2), "")
        date = next((c for c in cells if _DATE.fullmatch(c)), "")
        sign = next((c for c in cells if c not in (notice, rev, date)
                     and title_block._fold(c) not in title_block._NOT_A_NAME
                     and not c.isdigit()), "")
        out.append({"rev": rev, "doc": notice, "sign": sign, "date": date})
    if aggregated is not None:
        for part in re.split(r"\s*;\s*", aggregated["value"]):
            notice = _NOTICE.search(part)
            if not notice or notice.group(0) in seen:
                continue
            seen.add(notice.group(0))
            words = part.replace(notice.group(0), " ").split()
            rev = next((w for w in words if w.isdigit() and len(w) <= 2), "")
            sign = " ".join(w for w in words if w != rev)
            out.append({"rev": rev, "doc": notice.group(0), "sign": sign, "date": ""})
    return out


def sheet_numbers(fields: Dict[str, Dict[str, Any]]) -> Tuple[Optional[int], Optional[int]]:
    def number(key: str) -> Optional[int]:
        typed = fields.get(key)
        digits = re.sub(r"\D", "", str((typed or {}).get("value") or ""))
        return int(digits) if digits and len(digits) <= 3 else None
    return number("sheet"), number("sheets")


__all__ = ["build", "in_title", "similar", "sheet_numbers", "SCHEMA_KEY", "PASS_SHEET"]
