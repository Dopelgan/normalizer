"""
Векторные чертежи: определение листа-чертежа и разбор полей из текстового слоя.

Три вещи, из-за которых ветка чертежей раньше не работала, и что тут сделано.

1. Чертёж вообще не доезжал до обработчика. Он включался только на фрагменте
   `type="image"`, то есть когда MinerU выделил на странице рисунок. Лист из
   CAD — это векторный PDF во весь формат, картинкой он не становится.
   Поэтому здесь есть `detect_drawing_pages`: страница классифицируется по
   собственной геометрии, до всякого ML.
2. Размеры на чертеже подписаны повёрнутым текстом, а Ø, R, Ra и рамки
   допусков общий VLM читает плохо и охотно досочиняет. Но у векторного
   чертежа всё это лежит в текстовом слое — вместе с углом поворота.
   Распознавание тут просто не нужно.
3. Поля дозаполнялись выдумкой (единица «мм» по умолчанию, «справочный» от
   любой звёздочки). Здесь неизвестное остаётся пустым.

Для сканированных чертежей всё это неприменимо — там по-прежнему нужен
детектор и OCR; модуль честно ничего не вернёт.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from core.providers.text_layer import PYMUPDF_AVAILABLE, TextLayer, TextLine, open_pdf

logger = logging.getLogger(__name__)

try:
    import pymupdf as _pymupdf
except ImportError:  # pragma: no cover
    try:
        import fitz as _pymupdf
    except ImportError:
        _pymupdf = None

PROVENANCE_VECTOR = "vector_text_layer"

# --------------------------------------------------------------- признаки
_MANY_PATHS = 120          # столько векторных путей на листе уже не текст
_FRAME_COVERAGE = 0.65     # рамка формата занимает почти весь лист
_DRAWING_SCORE = 0.5       # порог отнесения страницы к чертежам

# Штамп по ГОСТ 2.104 — в правом нижнем углу листа.
_TITLE_BLOCK_REGION = (0.55, 0.78, 1.0, 1.0)

# --------------------------------------------------------------- шаблоны
_NUMBER = r"\d+(?:[.,]\d+)?"
_DIAMETER = re.compile(r"[ØøΦ⌀]\s*" + _NUMBER)
_RADIUS = re.compile(r"^R\s*" + _NUMBER, re.IGNORECASE)
_THREAD = re.compile(r"(?:^|\s)M\s*" + _NUMBER + r"(?:\s*[x×]\s*" + _NUMBER + r")?", re.IGNORECASE)
_ROUGHNESS = re.compile(r"(?:R[az]\s*" + _NUMBER + r"|√|∇)", re.IGNORECASE)
_ANGLE = re.compile(_NUMBER + r"\s*°")
_PLAIN_SIZE = re.compile(r"^" + _NUMBER + r"$")
_CALLOUT = re.compile(r"^[А-ЯA-Z](?:\s*[-–—]\s*[А-ЯA-Z])?$")
_FIT = re.compile(r"(?<![A-Za-zА-Яа-яЁё])[HhJjKkNnPpSsEeFfGg][sS]?\d{1,2}(?!\d)")
# «±» — всегда допуск, хоть слитно с размером (20±0,1). А вот «+» и «-»
# только когда стоят отдельно: в «ИМГ-001» и «2026-12-31» дефис принадлежит
# обозначению, и раньше такие строки объявлялись допусками.
_EXPLICIT_TOL = re.compile(
    r"(?:±\s*" + _NUMBER + r"|(?<![\w.,])[+\-]\s*" + _NUMBER + r")"
)
_RANGE_TOL = re.compile(_NUMBER + r"\s*\.{2,3}\s*" + _NUMBER)
_REFERENCE_MARK = re.compile(r"(?:" + _NUMBER + r"\s*\*|\bсправ)", re.IGNORECASE)

_MATERIAL_WORDS = ("сталь", "чугун", "алюмин", "латунь", "бронза", "полиамид", "ГОСТ 380")
_UNIT_PATTERNS: Tuple[Tuple[str, re.Pattern], ...] = (
    ("мкм", re.compile(r"мкм|µm|μm", re.IGNORECASE)),
    ("мм", re.compile(r"\bмм\b|\bmm\b", re.IGNORECASE)),
    ("см", re.compile(r"\bсм\b|\bcm\b", re.IGNORECASE)),
    ("°", re.compile(r"°|град", re.IGNORECASE)),
    ("HRC", re.compile(r"HRC")),
    ("HB", re.compile(r"\bHB\b")),
)


# ===========================================================================
# Классификация страницы
# ===========================================================================

@dataclass
class DrawingPage:
    """Вердикт по одной странице и признаки, по которым он вынесен."""

    page: int
    score: float
    signals: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_drawing(self) -> bool:
        return self.score >= _DRAWING_SCORE


def detect_drawing_pages(data: bytes) -> Dict[int, DrawingPage]:
    """
    Байты PDF -> вердикт по каждой странице. Пусто, если PyMuPDF недоступен
    или файл не открылся: тогда конвейер просто работает как раньше.
    """
    document = open_pdf(data)
    if document is None:
        return {}
    try:
        return drawing_pages_from_document(document)
    finally:
        document.close()


def drawing_pages_from_document(document: Any) -> Dict[int, DrawingPage]:
    """
    То же по уже открытому документу: файл за проход открывается один раз,
    и вердикт по чертежам снимается с того же документа, что и текстовый
    слой. Документ не закрывается — его закрывает тот, кто открыл.
    """
    verdicts: Dict[int, DrawingPage] = {}
    for index in range(document.page_count):
        page_no = index + 1
        try:
            verdicts[page_no] = _score_page(document[index], page_no)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Не удалось оценить стр. %d на чертёж: %s", page_no, exc)

    found = sorted(p for p, v in verdicts.items() if v.is_drawing)
    if found:
        logger.info("Страницы, распознанные как чертежи: %s", found)
    return verdicts


def _score_page(page: Any, page_no: int) -> DrawingPage:
    width, height = float(page.rect.width), float(page.rect.height)
    area = max(1.0, width * height)

    try:
        drawings = page.get_drawings()
    except Exception:  # noqa: BLE001
        drawings = []

    path_count = len(drawings)
    frame = _has_format_frame(drawings, area)
    title_block = _title_block_lines(drawings, width, height)
    text = page.get_text("text") or ""
    chars = len("".join(text.split()))

    signals = {
        "vector_paths": path_count,
        "format_frame": frame,
        "title_block_lines": title_block,
        "chars": chars,
        "landscape": width > height,
    }

    score = 0.0
    if path_count >= _MANY_PATHS:
        score += 0.35
    elif path_count >= _MANY_PATHS // 3:
        score += 0.15
    if frame:
        score += 0.25
    if title_block >= 6:
        score += 0.25
    elif title_block >= 3:
        score += 0.1
    # Много графики и мало сплошного текста — это чертёж, а не иллюстрация
    # внутри статьи.
    if path_count >= _MANY_PATHS and chars < 2500:
        score += 0.15

    return DrawingPage(page=page_no, score=round(min(1.0, score), 3), signals=signals)


def _has_format_frame(drawings: Sequence[Dict[str, Any]], page_area: float) -> bool:
    """Рамка формата — прямоугольник почти во весь лист."""
    for item in drawings:
        rect = item.get("rect")
        if rect is None:
            continue
        try:
            covered = abs(float(rect.width) * float(rect.height))
        except Exception:  # noqa: BLE001
            continue
        if covered / page_area >= _FRAME_COVERAGE:
            return True
    return False


def _title_block_lines(drawings: Sequence[Dict[str, Any]], width: float, height: float) -> int:
    """Число отрезков в зоне штампа: его сетка граф даёт характерный пучок."""
    if width <= 0 or height <= 0:
        return 0
    x1, y1, x2, y2 = _TITLE_BLOCK_REGION
    count = 0
    for item in drawings:
        rect = item.get("rect")
        if rect is None:
            continue
        try:
            cx = (float(rect.x0) + float(rect.x1)) / 2 / width
            cy = (float(rect.y0) + float(rect.y1)) / 2 / height
        except Exception:  # noqa: BLE001
            continue
        if x1 <= cx <= x2 and y1 <= cy <= y2:
            count += 1
    return count


# ===========================================================================
# Разбор полей чертежа
# ===========================================================================

def extract_drawing_fields(layer: TextLayer, page: int) -> List[Dict[str, Any]]:
    """
    Строки текстового слоя листа -> структурированные поля чертежа.

    Значения берутся из файла как есть, без распознавания, поэтому
    `confidence` отражает уверенность в *категории*, а не в тексте.
    """
    fields: List[Dict[str, Any]] = []
    for line in layer.lines(page):
        text = " ".join(line.text.split())
        if not text:
            continue
        category, category_confidence = classify(text, line)
        fields.append({
            "category": category,
            "value": text,
            "tolerance": extract_tolerance(text),
            "nature": determine_nature(text),
            "unit": extract_unit(text, category),
            "confidence": category_confidence,
            "bbox": list(line.bbox),
            "source_of_tolerance": tolerance_source(text),
            "provenance": PROVENANCE_VECTOR,
        })
    return fields


def classify(text: str, line: Optional[TextLine] = None) -> Tuple[str, float]:
    """Категория поля и уверенность в ней. Порядок проверок — от частного."""
    if line is not None and _in_title_block(line):
        if any(word in text.lower() for word in _MATERIAL_WORDS):
            return "material", 0.9
        return "title_block", 0.9

    if _THREAD.search(text):
        return "thread", 0.9
    if _RADIUS.search(text):
        return "radius", 0.95
    if _ROUGHNESS.search(text):
        return "roughness", 0.9
    if _DIAMETER.search(text):
        return "size", 0.95
    if any(word in text.lower() for word in _MATERIAL_WORDS):
        return "material", 0.85
    if _CALLOUT.match(text):
        return "callout", 0.8
    if _PLAIN_SIZE.match(text) or _ANGLE.search(text):
        return "size", 0.8
    if _EXPLICIT_TOL.search(text) or _RANGE_TOL.search(text) or _FIT.search(text):
        return "tolerance", 0.75
    # Всё остальное — технические требования и подписи. Это не разбор, а
    # честное «не классифицировано», и уверенность соответствующая.
    return "note", 0.4


def _in_title_block(line: TextLine) -> bool:
    x1, y1, x2, y2 = _TITLE_BLOCK_REGION
    cx, cy = line.center
    return x1 <= cx <= x2 and y1 <= cy <= y2


def extract_tolerance(text: str) -> Optional[str]:
    for pattern in (_EXPLICIT_TOL, _FIT, _RANGE_TOL):
        match = pattern.search(text)
        if match:
            return match.group().strip()
    return None


def tolerance_source(text: str) -> str:
    if _EXPLICIT_TOL.search(text) or _RANGE_TOL.search(text):
        return "explicit"
    if _FIT.search(text):
        return "fit_notation"
    return "unknown"


def extract_unit(text: str, category: str) -> str:
    """
    Единица — только если она названа или однозначно следует из категории.
    Пустая строка честнее подставленных «мм»: раньше так помечался любой
    текст, включая тот, у которого единицы нет вовсе.
    """
    for unit, pattern in _UNIT_PATTERNS:
        if pattern.search(text):
            return unit
    if category in ("size", "radius", "thread") and re.search(_NUMBER, text):
        return "мм"      # линейный размер на машиностроительном чертеже
    return ""


def determine_nature(text: str) -> str:
    """
    Справочный размер по ГОСТ 2.307 помечается звёздочкой после числа.
    Проверяется именно такая пометка, а не любая звёздочка в строке.
    """
    if _REFERENCE_MARK.search(text):
        return "reference"
    lowered = text.lower()
    if "инстр" in lowered or "tool" in lowered:
        return "tool_provided"
    return "executive"


def structure_ratio(fields: Sequence[Dict[str, Any]]) -> float:
    """
    Доля полей, для которых удалось определить конкретную категорию.
    Это и есть честная полнота разбора: `note` означает «не разобрано».
    """
    if not fields:
        return 0.0
    classified = sum(1 for f in fields if f.get("category") != "note")
    return round(classified / len(fields), 3)
