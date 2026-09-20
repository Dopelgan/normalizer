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
import re
import statistics
from typing import Any, Dict, List, Optional, Sequence

from core.models.parse_result import ParsedBlock
from core.providers import latex_text

logger = logging.getLogger(__name__)

# Во сколько раз вертикальный разрыв между строками должен превышать
# обычный межстрочный, чтобы считаться границей строки таблицы.
_ROW_GAP_FACTOR = 1.8

# Минимальный разрыв между колонками в долях ширины листа. Ниже этого
# «две колонки» — это одна колонка с неровным левым краем.
_MIN_COLUMN_GAP = 0.02

# Доля слов, которая обязана доехать из строк в ячейки, чтобы починку
# считать состоявшейся. Ниже — геометрия разложила строки не по тем
# клеткам, и часть листа пропала бы молча.
_KEEP_RATIO = 0.9

# Слово, по которому проверяется сохранность. Короче четырёх букв берутся
# обозначения величин («м», «с», «мм»), они есть и в формулах.
_WORD_RE = re.compile(r"[^\W\d_]{4,}")


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

    Возвращает True, только если заменены **все** испорченные ячейки и в
    ячейки доехало содержимое строк. False означает «геометрия не сошлась»
    — таблицу надо деградировать, а не оставлять наполовину починенной.

    Заполнить все испорченные ячейки мало: строка, отнесённая не к той
    колонке, попадает в клетку, которую никто не чинит, и пропадает совсем.
    Так `physical_formulas.png` потерял больше половины описаний при
    заявленной полноте 1.0. Поэтому после раскладки проверяется, что слова
    строк действительно лежат в сетке.
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

    total, kept = _word_recall(grid, lines)
    if total and kept < total * _KEEP_RATIO:
        logger.info(
            "Таблица на стр. %s: в ячейки доехало слов %d из %d — не чиним",
            block.page, kept, total,
        )
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
    """
    Строки, разложенные по колонкам. None — разложить не удалось.

    Режем по вертикальным коридорам пустоты: берём занятые строками отрезки
    по горизонтали, склеиваем пересекающиеся и смотрим, что осталось между
    ними. Раньше разрез искался по разрыву в левых краях — на таблице, где
    формулы в первой колонке выключены по центру, самый большой такой
    разрыв оказывался внутри колонки, и строки уезжали не туда.
    """
    if count <= 0 or not lines:
        return None
    if count == 1:
        return [list(lines)]
    if len(lines) < count:
        return None

    corridors = _corridors([(l.bbox[0], l.bbox[2]) for l in lines], _MIN_COLUMN_GAP)
    if len(corridors) < count - 1:
        return None
    # Коридоров бывает больше, чем колонок: числа в ячейках выключены по
    # правому краю и оставляют пустоту внутри колонки. Берём самые широкие.
    cuts = sorted(x for x, _width in sorted(corridors, key=lambda c: -c[1])[: count - 1])

    bands: List[List[Any]] = [[] for _ in range(count)]
    for line in lines:
        center = (line.bbox[0] + line.bbox[2]) / 2
        index = sum(1 for cut in cuts if center > cut)
        bands[index].append(line)
    return bands if all(bands) else None


def _corridors(spans: Sequence[tuple], min_width: float) -> List[tuple]:
    """Пустые вертикальные полосы между занятыми отрезками: (середина, ширина)."""
    merged: List[List[float]] = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])

    result: List[tuple] = []
    for left, right in zip(merged, merged[1:]):
        width = right[0] - left[1]
        if width >= min_width:
            result.append(((left[1] + right[0]) / 2, width))
    return result


def _row_bands(lines: Sequence[Any], count: int) -> Optional[List[List[Any]]]:
    """
    Строки колонки, разложенные по строкам таблицы. None — не сошлось.

    Разрывы берутся естественные: те, что заметно шире обычного
    межстрочного. Раньше бралось ровно `count - 1` самых больших разрывов,
    и если строк таблицы на листе оказывалось меньше, чем заявила модель,
    лишний разрез проходил посреди строки и сдвигал всё содержимое на
    клетку. Теперь несовпадение — повод деградировать, а не сдвигать.
    """
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
    threshold = statistics.median(heights) * (_ROW_GAP_FACTOR - 1)

    cuts = [
        index
        for index in range(len(ordered) - 1)
        if ordered[index + 1].bbox[1] - ordered[index].bbox[3] >= threshold
    ]
    if len(cuts) != count - 1:
        return None

    bands: List[List[Any]] = []
    previous = 0
    for cut in cuts:
        bands.append(ordered[previous:cut + 1])
        previous = cut + 1
    bands.append(ordered[previous:])
    return bands if all(bands) else None


def _word_recall(grid: List[List[str]], lines: Sequence[Any]) -> tuple:
    """
    Сколько слов из строк доехало до сетки: (всего, доехало).

    Сравниваются слова, а не строки целиком: ячейка, которую чинить не
    пришлось, несёт текст от модели, и он отличается от строки распознавания
    мелочами. Короткие обозначения величин не считаются — они встречаются
    и в формулах, где ничего чинить не нужно.
    """
    haystack = _fold(" ".join(cell for row in grid for cell in row))
    total = kept = 0
    for line in lines:
        for word in _WORD_RE.findall(getattr(line, "text", "") or ""):
            total += 1
            if _fold(word) in haystack:
                kept += 1
    return total, kept


def _fold(text: str) -> str:
    """Представление для сравнения: без пробелов, знаков и регистра."""
    return "".join(ch.lower() for ch in text if ch.isalnum())


def _text_of(lines: Sequence[Any]) -> str:
    """Текст полосы строк в порядке чтения."""
    ordered = sorted(lines, key=lambda l: (round(l.bbox[1], 3), l.bbox[0]))
    return "\n".join(
        " ".join((getattr(line, "text", "") or "").split()) for line in ordered
    ).strip()
