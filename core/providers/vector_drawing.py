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

from core.providers import dimension_fidelity, page_prose
from core.providers.text_layer import (
    PYMUPDF_AVAILABLE,
    TextLayer,
    TextLine,
    open_pdf,
    page_turn,
    upright_bbox,
)

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
_TEXT_IN_FRAME_SCORE = 0.4  # проза в рамке ЕСКД: ниже порога, но видно в signals
# Столько векторных путей у листа с текстом в рамке не бывает: рамка, штамп
# и пара таблиц — десятки-сотни. Сборочный чертёж с длинными техническими
# требованиями остаётся чертежом, сколько бы прозы на нём ни было.
_TEXT_IN_FRAME_MAX_PATHS = 400
# Лист, где вся векторная графика — горизонтальные и вертикальные отрезки
# (рамка, штамп, сетка таблиц), а картинок почти нет, — таблица или текст в
# рамке. У чертежа есть окружности и дуги (кривые), штриховка и выноски
# (наклонные отрезки) или вставленное изображение вида. На реальных PDF
# заказчика (ТЗ, ПМ1, ТУ, ПОК, СТО) такие листы — 90 % ложных «чертежей»,
# и prose-признак их не ловит: короткие строки ячеек — не предложения.
_GRID_MAX_CURVE_SHARE = 0.02
_GRID_MAX_DIAGONAL_SHARE = 0.02
_GRID_MAX_IMAGE_COVERAGE = 0.15
# Доли на листе с тысячами отрезков пропустят десятки кривых и наклонных —
# а это уже контакты и клеммы схемы Э3. Поэтому и абсолютный предел.
_GRID_MAX_NON_ORTHOGONAL = 40
# Классификатор листа (флаг DRAWING_PAGE_CLASSIFIER_ENABLED, поверх
# DRAWING_TEXT_IN_FRAME_ENABLED). Лист ТЗ, ПМ, ТУ со вставленным рисунком:
# вся векторная графика — та же сетка рамки, штампа и таблиц, а рисунок
# лежит картинкой на трети-половине листа. Порог картинок 15 % отправил бы такой
# лист в чертежи целиком, и весь его текст становился полями чертежа. Сам
# рисунок от этого не теряется: MinerU выделит его картинкой, и он пойдёт
# в Qwen-VL отдельным фрагментом.
_FIGURE_MAX_IMAGE_COVERAGE = 0.6
_FIGURE_MAX_DRAWING_MARKS = 5
# Штамп чертежа (ГОСТ 2.104, форма 1) отличается от штампа текстового
# документа (формы 2 и 2а) графами «Масса» и «Масштаб». Подписи граф —
# строка из одних меток штампа (MuPDF может склеить «Лит. Масса Масштаб»);
# в прозе эти слова стоят внутри предложения.
_DRAWING_FORM_LABEL = re.compile(
    r"^\s*(?:(?:Лит\.?|Масса|Масштаб|Mass|Weight|Scale)\s*)*"
    r"(?:Масса|Масштаб|Mass|Weight|Scale)(?:\s*(?:Лит\.?|Масса|Масштаб|Mass|Weight|Scale))*\s*$",
    re.IGNORECASE | re.MULTILINE,
)

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
    # Пути `get_drawings()` у страницы с /Rotate (альбомный лист, сохранённый
    # книжным) — в неповёрнутой системе, а `page.rect` — в видимой. Без
    # пересчёта сетка штампа искалась не в том углу, и повёрнутый лист
    # недобирал признаков чертежа.
    turn = int(getattr(page, "rotation", 0) or 0) % 360
    matrix = page.rotation_matrix if turn else None
    title_block = _title_block_lines(drawings, width, height, matrix)
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

    score = min(1.0, score)
    if score >= _DRAWING_SCORE and _text_in_frame_enabled():
        # Рамка и штамп есть и у ПОК, ТУ, ТЗ, писем. Если на листе проза или
        # одна сетка таблиц, это страница текста: её поля чертежа и Qwen-VL
        # только портят.
        grid = _grid_geometry(page, drawings, area)
        if grid is not None:
            signals["geometry"] = grid
        if grid is not None and grid["grid_only"]:
            signals["text_in_frame"] = True
            score = min(score, _TEXT_IN_FRAME_SCORE)
        elif grid is not None and _page_classifier_enabled() and _text_page_with_figure(grid, text):
            signals["text_in_frame"] = True
            signals["figure_in_text_page"] = True
            score = min(score, _TEXT_IN_FRAME_SCORE)
        elif path_count < _TEXT_IN_FRAME_MAX_PATHS:
            verdict = page_prose.assess(text.splitlines())
            signals["prose"] = verdict.signals
            if verdict.is_prose:
                signals["text_in_frame"] = True
                score = min(score, _TEXT_IN_FRAME_SCORE)

    return DrawingPage(page=page_no, score=round(score, 3), signals=signals)


def _grid_geometry(page: Any, drawings: Sequence[Dict[str, Any]],
                   area: float) -> Optional[Dict[str, Any]]:
    """
    Из чего состоит графика листа: доли кривых и наклонных отрезков, площадь
    картинок. None, если состав путей неизвестен (нет `items`) — тогда
    решать по геометрии нельзя.
    """
    total = curves = diagonal = 0
    for item in drawings:
        for part in item.get("items") or ():
            total += 1
            kind = part[0]
            if kind == "c":
                curves += 1
            elif kind == "l":
                a, b = part[1], part[2]
                if abs(float(a.x) - float(b.x)) > 1 and abs(float(a.y) - float(b.y)) > 1:
                    diagonal += 1
    if not total:
        return None
    covered = 0.0
    rect = page.rect
    try:
        for info in page.get_image_info():
            x0, y0, x1, y1 = (float(v) for v in info["bbox"])
            w = min(x1, float(rect.x1)) - max(x0, float(rect.x0))
            h = min(y1, float(rect.y1)) - max(y0, float(rect.y0))
            if w > 0 and h > 0:
                covered += w * h
    except Exception:  # noqa: BLE001
        # Картинки не прочитались: вставленный вид мог остаться незамеченным,
        # поэтому по геометрии не решаем.
        return None
    curve_share = curves / total
    diagonal_share = diagonal / total
    image_coverage = covered / area
    lines_only = (curves + diagonal <= _GRID_MAX_NON_ORTHOGONAL
                  and curve_share <= _GRID_MAX_CURVE_SHARE
                  and diagonal_share <= _GRID_MAX_DIAGONAL_SHARE)
    return {
        "curve_share": round(curve_share, 3),
        "diagonal_share": round(diagonal_share, 3),
        "image_coverage": round(image_coverage, 3),
        "lines_only": lines_only,
        "grid_only": lines_only and image_coverage <= _GRID_MAX_IMAGE_COVERAGE,
    }


def _text_page_with_figure(grid: Dict[str, Any], text: str) -> bool:
    """
    Лист текстового документа со вставленным рисунком: векторная графика —
    одна сетка, картинки не больше `_FIGURE_MAX_IMAGE_COVERAGE` листа, штамп
    не чертёжный. Чертёж, вставленный в ТЗ целиком со своим штампом формы 1,
    и чертёж с видом-картинкой в рамке остаются чертежами.
    """
    if not (grid["lines_only"]
            and grid["image_coverage"] <= _FIGURE_MAX_IMAGE_COVERAGE
            and not _DRAWING_FORM_LABEL.search(text)):
        return False
    # Размеры текстом поверх вида-картинки (Ø, R, резьба, шероховатость) —
    # это чертёж. Простые числа не в счёт: ими полны таблицы нагрузок в ТЗ.
    marks = sum(
        1 for line in text.splitlines()
        if _DIAMETER.search(line) or _RADIUS.search(line.strip())
        or _THREAD.search(line) or _ROUGHNESS.search(line)
    )
    return marks < _FIGURE_MAX_DRAWING_MARKS


def _text_in_frame_enabled() -> bool:
    from core.config import settings

    return bool(getattr(settings, "DRAWING_TEXT_IN_FRAME_ENABLED", False))


def _page_classifier_enabled() -> bool:
    from core.config import settings

    return bool(getattr(settings, "DRAWING_PAGE_CLASSIFIER_ENABLED", False))


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


def _title_block_lines(
    drawings: Sequence[Dict[str, Any]], width: float, height: float, matrix: Any = None
) -> int:
    """
    Число отрезков в зоне штампа: его сетка граф даёт характерный пучок.
    `matrix` — матрица поворота страницы (`page.rotation_matrix`): переводит
    пути в видимую систему, в которой заданы `width` и `height`.
    """
    if width <= 0 or height <= 0:
        return 0
    x1, y1, x2, y2 = _TITLE_BLOCK_REGION
    count = 0
    for item in drawings:
        rect = item.get("rect")
        if rect is None:
            continue
        try:
            if matrix is not None:
                rect = rect * matrix
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
    lines = layer.lines(page)
    # Лист, выведенный на страницу боком без /Rotate: штамп — в правом
    # нижнем углу прямо стоящего листа, а не страницы.
    turn = page_turn(lines)
    for line in lines:
        text = " ".join(line.text.split())
        if not text:
            continue
        category, category_confidence = classify(text, line, turn)
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


def classify(text: str, line: Optional[TextLine] = None, turn: int = 0) -> Tuple[str, float]:
    """
    Категория поля и уверенность в ней. Порядок проверок — от частного.
    `turn` — поворот текста страницы (`page_turn`) для зоны штампа.
    """
    if line is not None and _in_title_block(line, turn):
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
    # Марка стали («12Х18Н10Т») — материал, а не размер с посадкой: правило
    # посадки видит в ней «H10» и объявляло её допуском.
    if _fidelity_enabled() and dimension_fidelity.is_steel_grade(text):
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


def _in_title_block(line: TextLine, turn: int = 0) -> bool:
    x1, y1, x2, y2 = _TITLE_BLOCK_REGION
    left, top, right, bottom = upright_bbox(line.bbox, turn)
    cx, cy = (left + right) / 2, (top + bottom) / 2
    return x1 <= cx <= x2 and y1 <= cy <= y2


def _fidelity_enabled() -> bool:
    from core.config import settings

    return bool(getattr(settings, "DRAWING_DIMENSION_FIDELITY", False))


def _without_grades(text: str) -> str:
    """Текст, в котором марки стали закрыты точками (`DRAWING_DIMENSION_FIDELITY`)."""
    if not _fidelity_enabled():
        return text
    return dimension_fidelity.mask_steel_grades(text)


def extract_tolerance(text: str) -> Optional[str]:
    masked = _without_grades(text)
    for pattern in (_EXPLICIT_TOL, _FIT, _RANGE_TOL):
        match = pattern.search(masked)
        if match:
            # Марки стали закрыты: «12Х18Н10Т» допуска не даёт. Вне марок
            # закрытый текст совпадает с исходным посимвольно.
            return text[match.start():match.end()].strip()
    return None


def tolerance_source(text: str) -> str:
    masked = _without_grades(text)
    if _EXPLICIT_TOL.search(masked) or _RANGE_TOL.search(masked):
        return "explicit"
    if _FIT.search(masked):
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
