"""
Починка таблицы по строкам, прочитанным отдельно от неё.

Зачем это есть. Модель таблиц MinerU отдаёт ячейки в LaTeX. Для колонки с
формулами это верно, для колонки с русской прозой — каша (см.
`latex_text`). Структура таблицы при этом обычно правильная: колонок
столько, сколько на листе, и строк столько же. Значит, чинить надо не
структуру, а содержимое ячеек, и взять его есть откуда: OCR читает тот же
лист построчно и с координатами.

Сопоставление геометрическое и намеренно осторожное: колонки берутся по
разрыву в горизонтальном распределении строк, строки таблицы — по
вертикальным разрывам внутри колонки. Если разбиение не совпало с формой,
которую заявила таблица, функция честно возвращает False, и выше по
конвейеру таблица уходит на деградацию, а не заполняется наугад.

Модуль работает с любыми строками, у которых есть `text` и `bbox`, и
ничего не импортирует из `text_layer` — иначе получился бы цикл.
"""

from __future__ import annotations

import logging
import statistics
from typing import Any, Dict, List, Optional, Sequence

from core.models.parse_result import ParsedBlock
from core.providers import latex_text

logger = logging.getLogger(__name__)

# Во сколько раз вертикальный разрыв между строками должен превышать
# обычный межстрочный, чтобы считаться границей строки таблицы.
_ROW_GAP_FACTOR = 1.8

# Минимальный разрыв между колонками в долях ширины таблицы. Ниже этого
# «две колонки» — это одна колонка с неровным левым краем.
_MIN_COLUMN_GAP = 0.02


def grid_of(table_data: Optional[Dict[str, Any]]) -> List[List[str]]:
    """Таблица как список строк, заголовок первой строкой (если он есть)."""
    if not table_data:
        return []
    grid: List[List[str]] = []
    headers = list(table_data.get("headers") or [])
    if any(str(h).strip() for h in headers):
        grid.append([str(h) for h in headers])
    for row in table_data.get("rows") or []:
        grid.append([str(c) for c in row])
    return grid


def apply_grid(table_data: Dict[str, Any], grid: List[List[str]]) -> None:
    """Обратная операция к `grid_of`: раскладывает сетку в headers/rows."""
    headers = list(table_data.get("headers") or [])
    if any(str(h).strip() for h in headers) and grid:
        table_data["headers"] = grid[0]
        table_data["rows"] = [list(r) for r in grid[1:]]
    else:
        table_data["rows"] = [list(r) for r in grid]


def repair_cells(block: ParsedBlock, lines: Sequence[Any]) -> bool:
    """
    Заменяет ячейки с LaTeX-кашей текстом строк, попавших в ту же клетку.

    Возвращает True, только если заменены **все** испорченные ячейки. False
    означает «геометрия не сошлась» — таблицу надо деградировать, а не
    оставлять наполовину починенной.
    """
    table = block.table_data or {}
    grid = grid_of(table)
    if not grid or not lines:
        return False

    broken = [(r, c) for r, row in enumerate(grid) for c, cell in enumerate(row)
              if latex_text.is_mangled_text(cell)]
    if not broken:
        return True

    columns = _column_bands(lines, max(len(r) for r in grid))
    if columns is None:
        logger.info("Таблица на стр. %s: колонки по строкам не разложились", block.page)
        return False

    filled = 0
    for column_index, column_lines in enumerate(columns):
        wanted = [r for r, c in broken if c == column_index]
        if not wanted:
            continue
        bands = _row_bands(column_lines, len(grid))
        if bands is None:
            logger.info(
                "Таблица на стр. %s: в колонке %d строк не %d",
                block.page, column_index, len(grid),
            )
            return False
        for row_index in wanted:
            text = _text_of(bands[row_index])
            if not text:
                return False
            grid[row_index][column_index] = text
            filled += 1

    if filled != len(broken):
        return False

    apply_grid(table, grid)
    block.table_data = table
    logger.info("Таблица на стр. %s: ячеек восстановлено по OCR %d", block.page, filled)
    return True


def degrade(block: ParsedBlock, lines: Sequence[Any]) -> List[ParsedBlock]:
    """
    Таблица, которую не удалось починить, разбирается на части.

    Формульные ячейки остаются формулами — LaTeX в них настоящий. Проза
    берётся из строк OCR и уходит текстовыми блоками. Сама таблица остаётся
    блоком без разбора: у неё есть картинка области, а `completeness` 0
    честно отправит её на проверку человеку. Молчаливая выдача каши в
    индекс хуже любого из этих исходов.
    """
    grid = grid_of(block.table_data)
    replacements: List[ParsedBlock] = []

    for row in grid:
        for cell in row:
            if not cell.strip() or latex_text.is_mangled_text(cell):
                continue
            if "\\" not in cell:
                continue
            replacements.append(ParsedBlock(
                type="formula", text=cell, page=block.page, bbox=list(block.bbox),
                confidence=block.confidence, method="mineru_formula",
            ))

    for line in lines:
        text = " ".join((getattr(line, "text", "") or "").split())
        if not text:
            continue
        replacements.append(ParsedBlock(
            type="text", text=text, page=block.page, bbox=list(getattr(line, "bbox", block.bbox)),
            confidence=float(getattr(line, "confidence", 1.0) or 1.0),
            method="ocr_layer_recovered", is_fallback=True,
        ))

    block.table_data = None
    logger.info(
        "Таблица на стр. %s не разобрана: формул %d, строк текста %d",
        block.page,
        sum(1 for b in replacements if b.type == "formula"),
        sum(1 for b in replacements if b.type == "text"),
    )
    return replacements


# ===========================================================================
# Геометрия
# ===========================================================================

def _column_bands(lines: Sequence[Any], count: int) -> Optional[List[List[Any]]]:
    """Строки, разложенные по колонкам. None — разложить не удалось."""
    if count <= 0 or len(lines) < count:
        return None
    if count == 1:
        return [list(lines)]

    ordered = sorted(lines, key=lambda l: l.bbox[0])
    starts = [l.bbox[0] for l in ordered]
    gaps = sorted(
        ((starts[i + 1] - starts[i], i) for i in range(len(starts) - 1)),
        reverse=True,
    )[: count - 1]
    if len(gaps) < count - 1 or any(gap < _MIN_COLUMN_GAP for gap, _ in gaps):
        return None

    cuts = sorted(index for _gap, index in gaps)
    bands: List[List[Any]] = []
    previous = 0
    for cut in cuts:
        bands.append(ordered[previous:cut + 1])
        previous = cut + 1
    bands.append(ordered[previous:])
    return bands if all(bands) else None


def _row_bands(lines: Sequence[Any], count: int) -> Optional[List[List[Any]]]:
    """Строки колонки, разложенные по строкам таблицы. None — не сошлось."""
    if count <= 0 or not lines:
        return None
    ordered = sorted(lines, key=lambda l: l.bbox[1])
    if count == 1:
        return [ordered]
    if len(ordered) < count:
        return None

    heights = [l.bbox[3] - l.bbox[1] for l in ordered if l.bbox[3] > l.bbox[1]]
    if not heights:
        return None
    typical = statistics.median(heights)

    gaps: List[tuple] = []
    for index in range(len(ordered) - 1):
        gap = ordered[index + 1].bbox[1] - ordered[index].bbox[3]
        gaps.append((gap, index))

    wide = sorted(gaps, reverse=True)[: count - 1]
    if len(wide) < count - 1 or any(gap < typical * (_ROW_GAP_FACTOR - 1) for gap, _ in wide):
        return None

    cuts = sorted(index for _gap, index in wide)
    bands: List[List[Any]] = []
    previous = 0
    for cut in cuts:
        bands.append(ordered[previous:cut + 1])
        previous = cut + 1
    bands.append(ordered[previous:])
    return bands if all(bands) else None


def _text_of(lines: Sequence[Any]) -> str:
    """Текст полосы строк в порядке чтения."""
    ordered = sorted(lines, key=lambda l: (round(l.bbox[1], 3), l.bbox[0]))
    return "\n".join(
        " ".join((getattr(line, "text", "") or "").split()) for line in ordered
    ).strip()
