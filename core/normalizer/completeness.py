"""
Расчёт полноты (`completeness`) фрагментов.

Полнота — доля исходного содержимого, доехавшая до фрагмента. Считается
из наблюдаемых величин (площадь блоков на листе, заполненность ячеек),
а не из самих же нарезанных фрагментов.
"""

from typing import Dict, List, Optional, Sequence

from core.models.parse_result import ParsedBlock

FALLBACK_PENALTY = 0.8      # OCR-фоллбэк заведомо теряет часть содержимого
NO_BBOX_PENALTY = 0.9       # координат нет — покрытие страницы не проверить


class PageCoverage:
    """
    Покрытие страниц блоками: сколько площади листа занято распознанным.

    `exhaustive` — извлечение было исчерпывающим (DOCX, TXT, XLSX, DXF).
    У таких форматов нет ни листа, ни координат, и штраф за отсутствие bbox
    к ним неприменим: он ставился за «покрытие не проверить», а проверять
    нечего — потерять содержимое эти стратегии не могли. Полный разбор
    абзаца DOCX объявлялся неполным на 0.9 и попадал в очередь на проверку.
    """

    def __init__(self, blocks: List[ParsedBlock], exhaustive: bool = False):
        self._exhaustive = exhaustive
        self._coverage: Dict[int, float] = {}
        self._has_bbox: Dict[int, bool] = {}

        per_page: Dict[int, List[ParsedBlock]] = {}
        for block in blocks:
            per_page.setdefault(block.page, []).append(block)

        for page, page_blocks in per_page.items():
            area = sum(b.area() for b in page_blocks)
            # Признак «координаты реально есть»: хотя бы один блок не на весь лист.
            has_bbox = any(b.bbox != [0.0, 0.0, 1.0, 1.0] for b in page_blocks)
            self._has_bbox[page] = has_bbox
            self._coverage[page] = min(1.0, area) if has_bbox else 1.0

    def of(self, page: int) -> float:
        return self._coverage.get(page, 0.0)

    def has_bbox(self, page: int) -> bool:
        return self._has_bbox.get(page, False)

    def text_completeness(self, page: int, is_fallback: bool) -> float:
        value = self.of(page)
        if not self.has_bbox(page) and not self._exhaustive:
            value *= NO_BBOX_PENALTY
        if is_fallback:
            value *= FALLBACK_PENALTY
        return round(max(0.0, min(1.0, value)), 3)


class TextLayerRecall:
    """
    Полнота по факту: сколько символов текстового слоя PDF доехало до
    блоков. В отличие от покрытия площади это прямое измерение потери —
    именно оно показало бы 0.12 на первой странице `test.pdf` вместо
    невнятного 0.039 и 0.0 на страницах, потерянных целиком.

    Доступно только для векторных PDF; у скана эталона нет.
    """

    def __init__(self, blocks: Sequence[ParsedBlock], layer_chars: Dict[int, int]):
        produced: Dict[int, int] = {}
        for block in blocks:
            produced[block.page] = produced.get(block.page, 0) + _block_chars(block)

        self._recall: Dict[int, float] = {}
        for page, expected in (layer_chars or {}).items():
            page = int(page)
            if expected <= 0:
                continue
            self._recall[page] = round(min(1.0, produced.get(page, 0) / expected), 3)

    @property
    def available(self) -> bool:
        return bool(self._recall)

    def of(self, page: int) -> Optional[float]:
        return self._recall.get(page)

    def overall(self) -> Optional[float]:
        if not self._recall:
            return None
        return round(sum(self._recall.values()) / len(self._recall), 3)


def _block_chars(block: ParsedBlock) -> int:
    """Символы без пробелов, которые блок реально несёт."""
    if block.type == "table" and block.table_data:
        cells = [str(c) for row in (block.table_data.get("rows") or []) for c in row]
        cells += [str(h) for h in (block.table_data.get("headers") or [])]
        return len("".join("".join(c.split()) for c in cells))
    return len("".join((block.text or "").split()))


def table_completeness(table_data: Optional[dict]) -> float:
    """Доля заполненных ячеек таблицы."""
    if not table_data:
        return 0.0
    rows = table_data.get("rows") or []
    headers = table_data.get("headers") or []
    cells = [c for row in rows for c in row]
    if not cells:
        # Только заголовки — структура есть, данных нет.
        return 0.5 if any(str(h).strip() for h in headers) else 0.0
    filled = sum(1 for c in cells if str(c).strip())
    return round(filled / len(cells), 3)


def formula_completeness(
    text: Optional[str], mathml: Optional[str], parsed: Optional[dict]
) -> float:
    """Формула полна настолько, насколько есть машинно-читаемое представление."""
    if not text or not text.strip():
        return 0.0
    score = 0.5
    if mathml:
        score += 0.3
    if parsed and parsed.get("base_var"):
        score += 0.2
    return round(min(1.0, score), 3)
