"""
Какие страницы векторного PDF нужны уровню 5 (`LADDER_LAYOUT_PAGE_SUBSET`).

На PDF с текстовым слоем текст уровня 5 всё равно берётся из слоя, а от
MinerU нужна только структура: таблицы, картинки, формулы. Страница, на
которой нет ни линий сетки, ни растровой вставки, ни векторной графики, ни
шрифта формул, MinerU ничего своего не даёт — её строки слоя сшивка вернёт
сама. Такие страницы в MinerU не отправляются.

Страница нужна уровню 5, если на ней есть:

* линии сетки (`_count_rules` из `context.py`, тот же порог, что у признака
  `ruled_pages`) — рамка таблицы; соседние страницы берутся тоже: таблица
  продолжается на следующую, а MinerU отдаёт её целиком в рамке первой
  страницы;
* растровая вставка от `_MIN_IMAGE_SHARE` площади листа;
* векторная графика, кроме прямых линий: схема, график, залитые ячейки;
* шрифт формул или символы математических блоков Юникода.

Сомнение — в пользу MinerU: признак лишний стоит времени, пропущенный —
таблицы или картинки. Документ, у которого слой неполон (скан, смешанный,
страница без текста), идёт целиком, как прежде: `None`.
"""

from __future__ import annotations

import logging
import re
from typing import Any, List, Optional

from core.ladder.context import DocumentContext, _count_rules
from core.ladder.text_layer_complete import complete_text_layer
from core.providers.text_layer import open_pdf

logger = logging.getLogger(__name__)

# Тот же порог, что у признака `ruled_pages` (`context._has_ruled_tables`).
_MIN_RULES = 6
# Растр меньше этой доли листа — значок, а не картинка. Порог низкий
# намеренно: MinerU отдаёт картинкой и логотип в 0,7 % листа (замер 17.11 на
# локальной выборке: вставку в 0,23 % листа он не отдал ни разу).
_MIN_IMAGE_SHARE = 0.003
# Векторная фигура (не прямая линия) от этой доли листа — графика: квадрат
# 50 × 50 pt на листе A4.
_MIN_SHAPE_SHARE = 0.005
# Столько мелких фигур на листе — тоже графика (схема из мелких деталей).
_MIN_SHAPES = 5

# Шрифты формул. «Symbol» сюда не входит: им и OpenSymbol набраны маркеры
# списков — им отдельная проверка `_SYMBOL_FONT`.
_MATH_FONT = re.compile(
    r"math|cmmi|cmsy|cmex|msam|msbm|euclid|mt\s*extra|stix|mathematicalpi",
    re.IGNORECASE,
)
_SYMBOL_FONT = re.compile(r"symbol", re.IGNORECASE)
# Математические операторы, дополнительные стрелки и знаки, буквенно-цифровые
# математические символы.
_MATH_CHARS = re.compile(
    "[∀-⋿⟀-⟯⦀-⫿𝐀-𝟿]"
)


def pages_for_layout(context: DocumentContext) -> Optional[List[int]]:
    """
    Номера страниц (с 1), которые нужны уровню 5; пустой список — ни одна.
    `None` — решать не по страницам: документ идёт в MinerU целиком.
    """
    if complete_text_layer(context) is None or not context.data:
        return None
    layer = context.text_layer
    document = open_pdf(context.data)
    if document is None:
        return None
    try:
        total = document.page_count
        if layer is None or total != layer.page_count:
            return None
        needed = set()
        tables = set()
        for index in range(total):
            page_no = index + 1
            try:
                reason = _page_reason(document[index], layer, page_no)
            except Exception as exc:  # noqa: BLE001 — страница не прочиталась: решать за неё нельзя
                logger.warning("Страница %d: признаки для MinerU не сняты (%s) — документ целиком",
                               page_no, exc)
                return None
            if reason is None:
                continue
            needed.add(page_no)
            if reason == "rules":
                tables.add(page_no)
    finally:
        document.close()
    for page in tables:
        needed.update(p for p in (page - 1, page + 1) if 1 <= p <= total)
    return sorted(needed)


def _page_reason(page: Any, layer, page_no: int) -> Optional[str]:
    """Почему странице нужен MinerU; `None` — не нужен."""
    area = abs(page.rect)
    if not area:
        return "size"
    drawings = page.get_drawings()
    if _count_rules(drawings) >= _MIN_RULES:
        return "rules"
    if _has_shapes(drawings, area):
        return "graphics"
    for info in page.get_image_info():
        bbox = info.get("bbox")
        if bbox is None:
            continue
        x0, y0, x1, y1 = bbox
        if abs((x1 - x0) * (y1 - y0)) / area >= _MIN_IMAGE_SHARE:
            return "image"
    if any(_MATH_CHARS.search(line.text or "") for line in layer.lines(page_no)):
        return "formula"
    if _formula_spans(page):
        return "formula"
    return None


def _formula_spans(page: Any) -> bool:
    """
    Текст шрифтом формул. Шрифт Symbol — формула, только если это не
    маркер списка: один знак в начале строки.
    """
    fonts = [str(font[3] or "") for font in page.get_fonts()]
    if not any(_MATH_FONT.search(name) or _SYMBOL_FONT.search(name) for name in fonts):
        return False
    for block in page.get_text("dict").get("blocks", []):
        for line in block.get("lines", []):
            for index, span in enumerate(line.get("spans", [])):
                font = str(span.get("font") or "")
                text = (span.get("text") or "").strip()
                if not text:
                    continue
                if _MATH_FONT.search(font):
                    return True
                if _SYMBOL_FONT.search(font) and not (index == 0 and len(text) == 1):
                    return True
    return False


def _has_shapes(drawings, area: float) -> bool:
    """Векторная графика, кроме прямых линий (их считает `_count_rules`)."""
    shapes = 0
    for item in drawings:
        rect = item.get("rect")
        if rect is None:
            continue
        width, height = abs(float(rect.width)), abs(float(rect.height))
        if min(width, height) <= 2.0:
            # Линия (или точка): рамки таблиц и подчёркивания.
            continue
        if _invisible(item):
            continue
        if width * height / area >= _MIN_SHAPE_SHARE:
            return True
        shapes += 1
        if shapes >= _MIN_SHAPES:
            return True
    return False


def _invisible(item) -> bool:
    """Белая заливка без обводки: фон строки из текстового редактора, не графика."""
    if item.get("color") is not None or item.get("type") not in ("f", None):
        return False
    fill = item.get("fill")
    return fill is not None and all(component >= 0.95 for component in fill)
