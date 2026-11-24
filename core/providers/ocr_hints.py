"""
OCR-подсказки для чтения листа чертежа (флаг DRAWING_OCR_HINTS).

Строки PP-OCRv5 (детектор + кириллический распознаватель) по вырезкам ГОСТ
без поиска рамки — штамп 200×75 мм справа снизу, левая графа 32×170 мм,
повёрнутая на 90°, и поле листа плитками 2000 px — приходят с рамками в
долях страницы. Модель зрения остаётся единственной, кто раскладывает
значения по полям; OCR только свидетельствует:

* значение поля, которое OCR прочитал в том же месте так же, получает второе
  прочтение `{"source": "ocr"}` (подтверждение);
* расхождение в одном знаке из пар, которые путает модель (S/5, №/М, Б/6,
  З/3), — поле на сверку; «№» у OCR там, где модель прочитала резьбу «М»,
  значит метку шва, а не резьбу;
* строка OCR, которой нет у модели, в поля напрямую не попадает: её вырезку
  дочитывает модель, и поле добавляется, только если модель прочитала то же;
* пустую графу «Инв. № подл.» и пустое «Обозначение» OCR заполняет по
  образцу записи, с пометкой «из OCR, требует сверки».

Источник строк — файлы `DRAWING_OCR_HINTS_DIR` (подготовленные заранее, по
имени файла документа и странице) или сервис `DRAWING_OCR_HINTS_ENDPOINT`
(`tools/ocr_hints_server.py`: GPU, если хватает памяти, иначе CPU).
Модуль — чистые функции над полями и строками; вызов модели — через
переданного клиента.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from core.config import settings

logger = logging.getLogger(__name__)

SOURCE = "ocr"
OCR_MARK = "из OCR, требует сверки"

# Вырезки по ГОСТ 2.104 от краёв листа, мм: штамп справа снизу, левая графа
# у левого края снизу. Совпадают с замером OCR (Отчёт-12.11/OCR-замер-24.11).
STAMP_MM = (200.0, 75.0)
MARGIN_MM = (32.0, 170.0)
TILE_PX = 2000

# Похожие знаки (после `norm`): модель и OCR путают их на скане 150–300 dpi.
_CONFUSABLE = {frozenset(pair) for pair in (("s", "5"), ("№", "m"), ("n", "m"), ("б", "6"), ("з", "3"),
                                            ("b", "6"))}
_NUMBER_SIGN = re.compile(r"№|N[°ºo]\s?\d|N\s?\d", re.IGNORECASE)
_LOOKALIKE = str.maketrans({
    "А": "A", "В": "B", "Е": "E", "К": "K", "М": "M", "Н": "H", "О": "O", "Р": "P", "С": "C", "Т": "T",
    "Х": "X", "У": "Y", "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "х": "x", "у": "y",
    "−": "-", "–": "-", "—": "-", "Ø": "⌀", "ø": "⌀", "∅": "⌀", "×": "x", "°": "", "º": "",
})
_LABELLED = re.compile(r"^([^:\d]{2,30}):\s*(.+)$")
_HAND = re.compile(r"\s*\((?:рукописно|из OCR[^)]*)\)\s*$", re.IGNORECASE)
# Что дочитывать из OCR: короткая запись величины или знака, а не слово.
_MEASURE = re.compile(r"^(?:\d+\s*(?:отв\.?|x)\s*)?(?:[⌀RМMG]|Ra\s?|Rz\s?|S)?\s?\d+(?:[.,]\d+)?"
                      r"(?:\s?[±+\-]\s?\d+(?:[.,]\d+)?)*\*?$", re.IGNORECASE)
_INVENTORY = re.compile(r"^\d{3,5}\s?-\s?\d{2,3}$")
_DESIGNATION = re.compile(r"^[А-ЯЁA-Z]{2,5}\s?\d{6}\.\d{3}(?:\.\d{2,3})*(?:\s?-\d{2,3})?(?:\s?[А-ЯЁA-Z]{1,3}\d?)?$")
# Поле с рамкой больше этого (доли листа) — область, а не место надписи.
_REGION_AREA = 0.02
_PAD = 0.006
MIN_SCORE = 0.9
MAX_REREAD = 12


# ===========================================================================
# Строки
# ===========================================================================

def norm(value: Any) -> str:
    """Запись для сравнения: похожие буквы, регистр, десятичная запятая, без пробелов."""
    text = str(value or "").translate(_LOOKALIKE).casefold().translate(_LOOKALIKE)
    text = re.sub(r"(?<=\d),(?=\d)", ".", text)
    return re.sub(r"\s+", "", text)


def value_of(field_value: Any) -> str:
    """«Инв. № подл.: 1012-05 (рукописно)» -> «1012-05»."""
    text = _HAND.sub("", str(field_value or ""))
    match = _LABELLED.match(text)
    return match.group(2).strip() if match else text.strip()


def one_char(a: str, b: str) -> bool:
    """Две записи (после `norm`) различаются ровно одним знаком из похожих пар."""
    if len(a) != len(b) or a == b:
        return False
    diff = [(x, y) for x, y in zip(a, b) if x != y]
    return len(diff) == 1 and frozenset(diff[0]) in _CONFUSABLE


def _overlap(a: Sequence[float], b: Sequence[float], pad: float = _PAD) -> bool:
    return (a[0] - pad <= b[2] and b[0] - pad <= a[2] and a[1] - pad <= b[3] and b[1] - pad <= a[3])


def _area(box: Sequence[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _placed(field: Dict[str, Any]) -> bool:
    box = field.get("bbox") or []
    return len(box) == 4 and 0 < _area(box) <= _REGION_AREA


# ===========================================================================
# Источник строк
# ===========================================================================

_INDEX: Dict[str, Dict[Tuple[str, int], List[Dict[str, Any]]]] = {}


def _document_name(metadata: Dict[str, Any]) -> str:
    for key in ("source_path", "s3_fileid", "file_name"):
        value = str(metadata.get(key) or "")
        if value:
            return value.rsplit("/", 1)[-1]
    return ""


def from_dir(directory: str, metadata: Dict[str, Any], page: int) -> Optional[List[Dict[str, Any]]]:
    """Строки из файлов каталога: {"file", "page", "lines": [{text, score, box, region}]}."""
    index = _INDEX.get(directory)
    if index is None:
        index = {}
        for name in sorted(os.listdir(directory)) if os.path.isdir(directory) else []:
            if not name.endswith(".json"):
                continue
            try:
                data = json.load(open(os.path.join(directory, name), encoding="utf-8"))
            except (OSError, ValueError):
                continue
            index[(str(data.get("file") or ""), int(data.get("page") or 1))] = list(data.get("lines") or [])
        _INDEX[directory] = index
    return index.get((_document_name(metadata), int(page)))


def gost_pieces(size: Tuple[int, int], dpi: float) -> List[Tuple[str, List[float], bool]]:
    """
    Вырезки листа: (область, рамка в долях листа, повёрнута ли). Мм — по
    разрешению рендера; штамп и графа — от краёв листа, без поиска рамки.
    """
    width, height = size
    px_per_mm = dpi / 25.4
    stamp = [max(0.0, 1 - STAMP_MM[0] * px_per_mm / width), max(0.0, 1 - STAMP_MM[1] * px_per_mm / height),
             1.0, 1.0]
    margin = [0.0, max(0.0, 1 - MARGIN_MM[1] * px_per_mm / height),
              min(1.0, MARGIN_MM[0] * px_per_mm / width), 1.0]
    pieces = [("stamp", [round(v, 5) for v in stamp], False), ("margin", [round(v, 5) for v in margin], True)]
    cols = max(1, -(-width // TILE_PX))
    rows = max(1, -(-height // TILE_PX))
    for r in range(rows):
        for c in range(cols):
            x1, x2 = c / cols, (c + 1) / cols
            y1, y2 = r / rows, (r + 1) / rows
            pad_x, pad_y = 0.05 / cols, 0.05 / rows
            pieces.append(("field", [max(0.0, x1 - pad_x), max(0.0, y1 - pad_y),
                                     min(1.0, x2 + pad_x), min(1.0, y2 + pad_y)], False))
    return pieces


def to_page(box_px: Sequence[float], piece_box: Sequence[float], piece_size: Tuple[int, int],
            rotated: bool) -> List[float]:
    """Рамка строки в пикселях вырезки -> доли листа (вырезка графы повёрнута по часовой)."""
    x1, y1, x2, y2 = (float(v) for v in box_px)
    pw, ph = piece_size
    if rotated:
        u1, u2, v1, v2 = y1 / ph, y2 / ph, 1 - x2 / pw, 1 - x1 / pw
    else:
        u1, u2, v1, v2 = x1 / pw, x2 / pw, y1 / ph, y2 / ph
    bx1, by1, bx2, by2 = piece_box
    w, h = bx2 - bx1, by2 - by1
    return [round(bx1 + u1 * w, 5), round(by1 + v1 * h, 5), round(bx1 + u2 * w, 5), round(by1 + v2 * h, 5)]


def from_endpoint(image, dpi: float) -> Optional[List[Dict[str, Any]]]:
    """
    Строки от сервиса OCR по вырезкам ГОСТ. Ответ сервиса на картинку —
    {"lines": [{"text", "score", "box": [x1, y1, x2, y2] px}]}. Сбой — None.
    """
    import requests

    lines: List[Dict[str, Any]] = []
    for region, box, rotated in gost_pieces(image.size, dpi):
        w, h = image.size
        piece = image.crop((int(box[0] * w), int(box[1] * h), int(box[2] * w), int(box[3] * h))).convert("RGB")
        if rotated:
            piece = piece.rotate(-90, expand=True)
        buffer = io.BytesIO()
        piece.save(buffer, format="PNG")
        try:
            response = requests.post(settings.DRAWING_OCR_HINTS_ENDPOINT,
                                     json={"image": base64.b64encode(buffer.getvalue()).decode("ascii")},
                                     timeout=settings.DRAWING_OCR_HINTS_TIMEOUT)
            response.raise_for_status()
            found = response.json().get("lines") or []
        except Exception as exc:  # noqa: BLE001 — подсказка не роняет лист
            logger.warning("OCR-подсказки недоступны: %s", exc)
            return None
        for line in found:
            text = str(line.get("text") or "").strip()
            if text:
                lines.append({"text": text, "score": float(line.get("score") or 0.0), "region": region,
                              "box": to_page(line["box"], box, piece.size, rotated)})
    return lines


# ===========================================================================
# Сведение с полями
# ===========================================================================

def _reading(field: Dict[str, Any], text: str) -> None:
    readings = list(field.get("readings") or [{"source": "sheet", "value": str(field.get("value") or "")}])
    if not any(r.get("source") == SOURCE and r.get("value") == text for r in readings):
        readings.append({"source": SOURCE, "value": text})
    field["readings"] = readings


def match(fields: List[Dict[str, Any]], lines: Sequence[Dict[str, Any]], stats: Dict[str, int]) -> None:
    """Подтверждение и расхождение в одном знаке — для полей с местом на листе."""
    for field in fields:
        if field.get("category") in ("note", "table") or not _placed(field):
            continue
        value = norm(value_of(field.get("value")))
        if not value:
            continue
        near = [line for line in lines if _overlap(line["box"], field["bbox"])]
        same = next((line for line in near if norm(line["text"]) == value
                     or (len(value) >= 4 and value in norm(line["text"]))), None)
        if same is not None:
            _reading(field, same["text"])
            stats["confirmed"] += 1
            continue
        close = next((line for line in near if one_char(norm(line["text"]), value)), None)
        if close is None:
            continue
        _reading(field, close["text"])
        field["needs_review"] = True
        stats["one_char"] += 1
        if field.get("category") == "thread" and _NUMBER_SIGN.search(close["text"]):
            # «2№3» у OCR, «2M3» у модели: метка шва, а не резьба.
            field["category"] = "callout"
            stats["not_thread"] += 1


def candidates(fields: Sequence[Dict[str, Any]], lines: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Строки поля листа, которых нет у модели: короткая величина, уверенное OCR, рядом нет поля."""
    placed = [f for f in fields if _placed(f)]
    known = {norm(value_of(f.get("value"))) for f in fields}
    found: List[Dict[str, Any]] = []
    for line in lines:
        text = " ".join(str(line["text"]).split())
        if line.get("region") != "field" or float(line.get("score") or 0) < MIN_SCORE:
            continue
        if not _MEASURE.match(text) or norm(text) in known:
            continue
        if any(_overlap(line["box"], f["bbox"]) for f in placed):
            continue
        if any(norm(text) == norm(other["text"]) and _overlap(line["box"], other["box"], 0.0) for other in found):
            continue
        found.append(line)
        if len(found) >= MAX_REREAD:
            break
    return found


def reread(qwen, image, lines: Sequence[Dict[str, Any]], stats: Dict[str, int]) -> List[Dict[str, Any]]:
    """
    Строки OCR — вырезками модели одной картинкой. Поле добавляется, только
    если модель прочитала ту же запись: OCR своё в поля не вносит.
    """
    from core.providers import disputed_reread, vector_drawing

    if not lines:
        return []
    pieces = [disputed_reread.crop(image, line["box"]) for line in lines]
    answers = qwen.read_values(disputed_reread.grid(pieces), len(pieces)) or {}
    stats["reread"] += len(lines)
    added: List[Dict[str, Any]] = []
    for n, line in enumerate(lines, start=1):
        answer = " ".join(str(answers.get(n) or "").split())
        if not answer or norm(answer) != norm(line["text"]):
            continue
        category, _ = vector_drawing.classify(answer)
        if category == "note":
            # Голое число со звёздочкой или допуском — размер: кандидат уже
            # прошёл образец величины (`_MEASURE`).
            category = "size"
        added.append({
            "category": category, "value": answer, "bbox": list(line["box"]), "confidence": 0.6,
            "tolerance": vector_drawing.extract_tolerance(answer),
            "nature": vector_drawing.determine_nature(answer),
            "unit": vector_drawing.extract_unit(answer, category),
            "source_of_tolerance": vector_drawing.tolerance_source(answer),
            "readings": [{"source": SOURCE, "value": line["text"]}, {"source": "crop", "value": answer}],
        })
    stats["added"] += len(added)
    return added


def fill_empty(fields: List[Dict[str, Any]], lines: Sequence[Dict[str, Any]], stats: Dict[str, int]) -> None:
    """
    Пустые «Инв. № подл.» и «Обозначение» — из OCR по образцу записи, на сверку.
    Графа подшивки — только из полосы графы, обозначение — из штампа.
    """
    from core.providers import vector_drawing

    labels = {str(f.get("value") or "").split(":", 1)[0].strip() for f in fields if ":" in str(f.get("value") or "")}
    wanted = (("Инв. № подл.", "margin", _INVENTORY), ("Обозначение", "stamp", _DESIGNATION))
    for label, region, pattern in wanted:
        if label in labels:
            continue
        found = [line for line in lines if line.get("region") == region
                 and float(line.get("score") or 0) >= MIN_SCORE and pattern.match(" ".join(line["text"].split()))]
        if not found:
            continue
        best = max(found, key=lambda line: (line["box"][2] - line["box"][0]) * (line["box"][3] - line["box"][1]))
        text = " ".join(best["text"].split())
        fields.append({
            "category": "title_block", "value": f"{label}: {text} ({OCR_MARK})", "bbox": list(best["box"]),
            "confidence": 0.5, "needs_review": True, "tolerance": None,
            "nature": vector_drawing.determine_nature(text), "unit": "",
            "source_of_tolerance": vector_drawing.tolerance_source(text),
            "readings": [{"source": SOURCE, "value": text}],
        })
        stats["filled"] += 1


def new_stats() -> Dict[str, int]:
    return {"lines": 0, "confirmed": 0, "one_char": 0, "not_thread": 0, "reread": 0, "added": 0, "filled": 0}
