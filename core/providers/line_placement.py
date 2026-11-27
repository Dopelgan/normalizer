"""
Место надписи без места — по строкам распознавания страницы
(DRAWING_PLACE_FROM_LINES).

Зачем. Содержание листа без рамок (DRAWING_SHEET_CONTENT_PASS), поля из
текста листа и плитки дают надписи, которые прочитаны, но места не имеют:
у них рамка всего листа или области (`bbox_exact=False`). Просмотрщик
рисует такую надпись на весь лист, а сверка с листом считает её прочтением
не на месте. Между тем у страницы есть строки распознавания с рамками
(`position.lines`, RECOGNITION_LINES): слой PDF, строки MinerU и Tesseract.

Что делается. Для каждого поля без места ищется строка страницы с той же
записью (без пробелов, регистра и разницы «,»/«.»). Ровно одна такая
строка — поле получает её рамку; запись внутри строки — часть рамки строки
по доле знаков. Две и больше строк — место неизвестно, поле не трогается.
Подпись графы («Масса: 1,26») ищется без подписи. Вызовов модели нет.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence, Tuple

from core.models.contract import DrawingField, Fragment

# Рамка больше этой доли страницы — не место надписи.
REGION_SHARE = 0.3
# Запись короче — внутри строки не ищется: «1» есть в половине строк листа.
MIN_INSIDE = 4

_LABEL = re.compile(r"^\s*[A-Za-zА-Яа-яЁё][^:]{0,40}?:\s*(.+)$", re.S)
_HANDWRITTEN = re.compile(r"\s*\((?:рукописно|от руки)\)\s*$", re.I)


def key(text: str) -> str:
    """Запись для сравнения: без пробелов и регистра, «,» как «.», «Ø» как «⌀»."""
    value = re.sub(r"\s+", "", str(text or "")).casefold()
    return value.replace(",", ".").replace("ø", "⌀").replace("∅", "⌀")


def _body(value: str) -> str:
    value = _HANDWRITTEN.sub("", str(value or ""))
    match = _LABEL.match(value)
    return match.group(1) if match else value


def unplaced(field: DrawingField) -> bool:
    box = field.bbox
    if not box or len(box) != 4 or not all(isinstance(v, (int, float)) for v in box):
        return True
    x1, y1, x2, y2 = (float(v) for v in box)
    return field.bbox_exact is False or (x2 - x1) * (y2 - y1) > REGION_SHARE


def _page_lines(fragments: Sequence[Fragment], page: int) -> List[Tuple[str, List[float]]]:
    """Строки страницы `page` у всех фрагментов документа, без повторов."""
    seen = set()
    lines: List[Tuple[str, List[float]]] = []
    for fragment in fragments:
        position = fragment.position
        if position is None or not position.lines:
            continue
        for line in position.lines:
            line_page = line.page if line.page is not None else position.page
            if line_page != page or not line.text or not line.bbox or len(line.bbox) != 4:
                continue
            if line.rotation:
                # Повёрнутая строка: доля знаков вдоль x не даёт места.
                continue
            item = (key(line.text), tuple(round(float(c), 4) for c in line.bbox))
            if item in seen:
                continue
            seen.add(item)
            lines.append((item[0], [float(c) for c in line.bbox]))
    return lines


def _box_inside(line_key: str, box: List[float], wanted: str) -> Optional[List[float]]:
    """Часть рамки строки под записью `wanted` по доле знаков; None — нет или не одна."""
    start = line_key.find(wanted)
    if start < 0 or line_key.find(wanted, start + 1) >= 0:
        return None
    x1, y1, x2, y2 = box
    width = (x2 - x1) / max(1, len(line_key))
    return [round(x1 + start * width, 4), round(y1, 4),
            round(x1 + (start + len(wanted)) * width, 4), round(y2, 4)]


def find_box(value: str, lines: Sequence[Tuple[str, List[float]]]) -> Optional[List[float]]:
    """Рамка единственной строки (или части строки) с записью `value`."""
    wanted = key(_body(value))
    if not wanted:
        return None
    whole = [box for line_key, box in lines if line_key == wanted]
    if len(whole) == 1:
        return [round(c, 4) for c in whole[0]]
    if whole or len(wanted) < MIN_INSIDE:
        return None
    inside = [(line_key, box) for line_key, box in lines if wanted in line_key]
    if len(inside) != 1:
        return None
    return _box_inside(inside[0][0], inside[0][1], wanted)


def apply(fragments: List[Fragment]) -> Dict[str, int]:
    """Поля без места у чертёжных фрагментов документа; правка на месте."""
    stats = {"placed": 0, "left": 0}
    cache: Dict[int, List[Tuple[str, List[float]]]] = {}
    for fragment in fragments:
        if fragment.type != "drawing" or fragment.position is None:
            continue
        payload = fragment.content.structured_payload or {}
        if payload.get("coordinate_reference") != "page":
            # Рамки полей ещё в долях вырезки: строки страницы с ними несравнимы.
            continue
        page = int(fragment.position.page)
        if page not in cache:
            cache[page] = _page_lines(fragments, page)
        lines = cache[page]
        if not lines:
            continue
        for field in fragment.content.structured_drawing_fields or []:
            if not unplaced(field):
                continue
            box = find_box(field.value, lines)
            if box is None:
                stats["left"] += 1
                continue
            field.bbox = box
            field.bbox_exact = True
            stats["placed"] += 1
    return stats


# ===========================================================================
# Пункт требований целиком и его строки (DRAWING_NOTE_PIECES_MERGE)
# ===========================================================================

# Короче — не строка пункта, а надпись: склеивать нечего.
_PIECE_MIN = 8
_NOTE_KINDS = ("note", "callout")


def _union(boxes: Sequence[Sequence[float]]) -> List[float]:
    return [round(min(b[0] for b in boxes), 4), round(min(b[1] for b in boxes), 4),
            round(max(b[2] for b in boxes), 4), round(max(b[3] for b in boxes), 4)]


def _cover(whole: str, pieces: Sequence[Tuple[str, int]]) -> Optional[List[int]]:
    """
    Номера строк, которые подряд и без остатка составляют `whole`; None — не
    составляют или строка одна. Строки идут по порядку чтения: каждая
    следующая начинается там, где кончилась предыдущая.
    """
    at, used = 0, []
    for piece, number in pieces:
        if not whole.startswith(piece, at):
            continue
        at += len(piece)
        used.append(number)
        if at == len(whole):
            return used if len(used) > 1 else None
    return None


def merge_note_pieces(fragments: List[Fragment]) -> Dict[str, int]:
    """
    Пункт технических требований пришёл дважды: целиком без места (содержание
    листа без рамок) и строками с местами (проход с рамками режет пункт по
    строкам листа). На листе штанги АЕГН строка «Присадочная проволока…»
    выходила в требованиях два раза. Пункт целиком остаётся одним полем с
    рамкой — объединением рамок его строк; строки снимаются. Не сложилось
    без остатка — ничего не меняется.
    """
    stats = {"merged": 0, "pieces": 0}
    for fragment in fragments:
        if fragment.type != "drawing":
            continue
        payload = fragment.content.structured_payload or {}
        if payload.get("coordinate_reference") != "page":
            continue
        fields = list(fragment.content.structured_drawing_fields or [])
        placed = sorted(
            (n for n, f in enumerate(fields) if f.category in _NOTE_KINDS and not unplaced(f)),
            key=lambda n: (round(float(fields[n].bbox[1]), 3), float(fields[n].bbox[0])))
        gone: set = set()
        for whole in fields:
            if whole.category not in _NOTE_KINDS or not unplaced(whole):
                continue
            wanted = key(whole.value)
            if len(wanted) < 2 * _PIECE_MIN:
                continue
            pieces = [(key(fields[m].value), m) for m in placed
                      if m not in gone and len(key(fields[m].value)) >= _PIECE_MIN
                      and key(fields[m].value) in wanted]
            used = _cover(wanted, pieces)
            if used is None:
                continue
            whole.bbox = _union([fields[m].bbox for m in used])
            whole.bbox_exact = True
            gone.update(used)
            stats["merged"] += 1
            stats["pieces"] += len(used)
        if gone:
            fragment.content.structured_drawing_fields = [
                f for m, f in enumerate(fields) if m not in gone]
    return stats
