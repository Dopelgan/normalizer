"""
Текстовый слой PDF: извлечение и сшивка с layout-блоками парсера.

Зачем это есть. MinerU в режиме `pipeline` рендерит страницу в растр и
читает только то, что layout-модель признала текстовой рамкой. Мелкий
светло-серый курсив по контрасту в рамку не попадает, код в серой заливке
размечается как рисунок — и содержимое исчезает молча. Плюс даже удачно
распознанный текст приходит с подменами визуально похожих букв
(`rps` -> `грs`, `expires_at` -> `exрires_at`), что для поиска фатально.

У векторного PDF всё это уже есть в самом файле: текст, координаты, кегль,
цвет и угол поворота. Поэтому для векторных документов слой становится
источником истины по тексту, а MinerU остаётся источником структуры —
рамок, таблиц, формул и картинок.

Модуль ничего не знает про MinerU и не ходит в сеть: на вход байты PDF,
на выход строки с координатами в долях [0, 1].
"""

from __future__ import annotations

import difflib
import logging
import math
import statistics
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from core.models.parse_result import ParsedBlock
from core.providers import latex_text, table_from_lines

logger = logging.getLogger(__name__)

try:  # PyMuPDF >= 1.24 экспортирует себя как `pymupdf`, раньше — как `fitz`
    import pymupdf as _pymupdf
except ImportError:  # pragma: no cover
    try:
        import fitz as _pymupdf
    except ImportError:
        _pymupdf = None

PYMUPDF_AVAILABLE = _pymupdf is not None

# Методы, которыми блок мог получить свой текст (уезжает в provenance).
METHOD_TEXT_LAYER = "text_layer"
METHOD_TEXT_LAYER_RECOVERED = "text_layer_recovered"
METHOD_PARSER_OCR = "mineru_ocr"
METHOD_OCR_LAYER = "ocr_layer"
METHOD_OCR_LAYER_RECOVERED = "ocr_layer_recovered"

_MONO_MARKERS = ("mono", "courier", "consol", "menlo", "inconsolata", "hack", "fira code")

# Строка считается попавшей в рамку, если её центр внутри рамки с этим допуском
# (в долях страницы) — layout-рамки MinerU обычно чуть у́же реального текста.
_CONTAINMENT_SLACK = 0.004

# Ниже этой доли символов восстановленный из слоя текст считается результатом
# кривой сшивки, и текст парсера сохраняется как есть.
_REBUILD_MIN_RATIO = 0.6

# Насколько похожи должны быть ячейка таблицы и строка слоя, чтобы считать их
# одним и тем же текстом. Сравнение идёт по «свёрнутым» буквам, поэтому
# сходство ниже 0.85 означает уже разные строки, а не ошибку распознавания.
_CELL_MATCH_RATIO = 0.85

# Пары визуально неразличимых букв латиницы и кириллицы. Распознавание
# постоянно их путает — `rps` превращается в `грs`, `БД` в `БD`, — и для
# сравнения обе стороны сводятся к одному представлению.
_HOMOGLYPHS = {
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "у": "y", "х": "x",
    "к": "k", "м": "m", "т": "t", "н": "h", "в": "b", "г": "r", "и": "u",
    "д": "d", "ь": "b", "з": "3", "ч": "4",
    "А": "a", "В": "b", "Е": "e", "К": "k", "М": "m", "Н": "h", "О": "o",
    "Р": "p", "С": "c", "Т": "t", "У": "y", "Х": "x", "Д": "d", "З": "3",
}


# ===========================================================================
# Модель слоя
# ===========================================================================

@dataclass
class TextLine:
    """Одна строка текстового слоя с координатами в долях страницы."""

    text: str
    bbox: List[float]
    page: int
    size: float = 0.0
    rotation: float = 0.0
    color: int = 0
    mono: bool = False
    # Уверенность чтения строки. У векторного слоя это единица по
    # построению — текст взят из файла. У слоя, собранного распознаванием,
    # сюда уезжает уверенность OCR, и выдавать её за единицу нельзя.
    confidence: float = 1.0

    @property
    def height(self) -> float:
        return max(0.0, self.bbox[3] - self.bbox[1])

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.bbox[0] + self.bbox[2]) / 2, (self.bbox[1] + self.bbox[3]) / 2)

    @property
    def is_rotated(self) -> bool:
        return abs(self.rotation) > 5.0

    def char_count(self) -> int:
        """Символы без пробелов — единица измерения полноты."""
        return len("".join(self.text.split()))


@dataclass
class TextLayer:
    """Текстовый слой всего документа, разложенный по страницам."""

    pages: Dict[int, List[TextLine]] = field(default_factory=dict)
    page_sizes: Dict[int, Tuple[float, float]] = field(default_factory=dict)
    page_count: int = 0

    def lines(self, page: int) -> List[TextLine]:
        return self.pages.get(page, [])

    def char_count(self, page: Optional[int] = None) -> int:
        if page is not None:
            return sum(line.char_count() for line in self.lines(page))
        return sum(line.char_count() for lines in self.pages.values() for line in lines)

    def char_counts_by_page(self) -> Dict[int, int]:
        return {page: self.char_count(page) for page in range(1, self.page_count + 1)}

    def pages_with_text(self) -> int:
        return sum(1 for lines in self.pages.values() if any(l.char_count() for l in lines))

    def is_usable(self, min_ratio: float = 0.5, min_chars: int = 40) -> bool:
        """
        Слой пригоден, если текст есть хотя бы на половине страниц. У скана
        слоя нет вовсе, у гибридного PDF он есть частично — там сшивка всё
        равно полезна для тех страниц, где текст есть.
        """
        if not self.page_count or self.char_count() < min_chars:
            return False
        return self.pages_with_text() / self.page_count >= min_ratio


# ===========================================================================
# Извлечение
# ===========================================================================

def open_pdf(data: bytes) -> Optional[Any]:
    """
    Байты PDF -> открытый документ PyMuPDF. `None`, если библиотеки нет или
    файл не открылся.

    Открытие вынесено наружу, чтобы один файл за проход открывался один раз:
    текстовый слой и поиск чертёжных листов спрашивают один и тот же
    документ, а не читают файл заново каждый на свой вопрос. Закрывает
    документ тот, кто его открыл.
    """
    if not PYMUPDF_AVAILABLE:
        logger.warning(
            "PyMuPDF не установлен — текстовый слой не читается, "
            "качество текста ограничено OCR. Поставьте pymupdf."
        )
        return None
    try:
        return _pymupdf.open(stream=data, filetype="pdf")
    except Exception as exc:  # noqa: BLE001 — битый или зашифрованный файл
        logger.warning("PyMuPDF не смог открыть PDF: %s", exc)
        return None


def extract_text_layer(data: bytes) -> Optional[TextLayer]:
    """
    Байты PDF -> текстовый слой. `None`, если PyMuPDF не установлен или файл
    не открылся: это не повод ронять разбор, конвейер просто останется на OCR.
    """
    document = open_pdf(data)
    if document is None:
        return None
    try:
        return layer_from_document(document)
    finally:
        document.close()


def layer_from_document(document: Any) -> TextLayer:
    """Текстовый слой уже открытого документа. Документ не закрывается."""
    layer = TextLayer()
    layer.page_count = document.page_count
    for index in range(document.page_count):
        page_no = index + 1
        try:
            page = document[index]
            width, height = float(page.rect.width), float(page.rect.height)
            layer.page_sizes[page_no] = (width, height)
            layer.pages[page_no] = _lines_from_page(page, page_no, width, height)
        except Exception as exc:  # noqa: BLE001 — одна битая страница не роняет документ
            logger.warning("Страница %d не прочиталась из текстового слоя: %s", page_no, exc)
            layer.pages[page_no] = []

    logger.info(
        "Текстовый слой: %d стр., %d стр. с текстом, %d символов",
        layer.page_count, layer.pages_with_text(), layer.char_count(),
    )
    return layer


def _lines_from_page(page: Any, page_no: int, width: float, height: float) -> List[TextLine]:
    """Строки страницы с нормализованными координатами."""
    if width <= 0 or height <= 0:
        return []

    raw = page.get_text("dict")
    lines: List[TextLine] = []

    for block in raw.get("blocks") or []:
        if block.get("type") != 0:      # 0 — текст, 1 — растровая картинка
            continue
        for line in block.get("lines") or []:
            spans = line.get("spans") or []
            text = "".join(span.get("text", "") for span in spans)
            if not text.strip():
                continue

            bbox = line.get("bbox") or [0, 0, 0, 0]
            sizes = [float(s.get("size", 0) or 0) for s in spans]
            fonts = " ".join(str(s.get("font", "")) for s in spans).lower()
            colors = [int(s.get("color", 0) or 0) for s in spans]

            lines.append(TextLine(
                text=text.rstrip(),
                bbox=[
                    max(0.0, min(1.0, float(bbox[0]) / width)),
                    max(0.0, min(1.0, float(bbox[1]) / height)),
                    max(0.0, min(1.0, float(bbox[2]) / width)),
                    max(0.0, min(1.0, float(bbox[3]) / height)),
                ],
                page=page_no,
                size=max(sizes) if sizes else 0.0,
                rotation=_direction_to_degrees(line.get("dir")),
                color=statistics.mode(colors) if colors else 0,
                mono=any(marker in fonts for marker in _MONO_MARKERS),
            ))

    return lines


def _direction_to_degrees(direction: Optional[Sequence[float]]) -> float:
    """
    Вектор направления письма PyMuPDF -> угол в градусах. Нужен для чертежей:
    размеры там подписаны повёрнутым текстом, и в растре его никакой OCR
    уверенно не берёт, а в слое угол известен точно.
    """
    if not direction or len(direction) < 2:
        return 0.0
    cos_a, sin_a = float(direction[0]), float(direction[1])
    if cos_a == 0.0 and sin_a == 0.0:
        return 0.0
    return round(math.degrees(math.atan2(-sin_a, cos_a)), 1)


# ===========================================================================
# Группировка строк в абзацы
# ===========================================================================

def group_lines(lines: Sequence[TextLine]) -> List[List[TextLine]]:
    """
    Соседние строки -> абзацы. Разрыв — там, где вертикальный зазор заметно
    больше обычного межстрочного, меняется угол поворота или меняется
    моноширинность (граница «проза / код»).
    """
    if not lines:
        return []

    ordered = sorted(lines, key=_reading_key)
    heights = [l.height for l in ordered if l.height > 0]
    typical = statistics.median(heights) if heights else 0.012
    gap_limit = max(typical * 0.85, 0.004)

    groups: List[List[TextLine]] = [[ordered[0]]]
    for line in ordered[1:]:
        previous = groups[-1][-1]
        gap = line.bbox[1] - previous.bbox[3]
        same_style = (line.mono == previous.mono) and (abs(line.rotation - previous.rotation) < 5)
        if gap > gap_limit or not same_style:
            groups.append([line])
        else:
            groups[-1].append(line)
    return groups


def lines_to_text(lines: Sequence[TextLine]) -> str:
    """
    Текст абзаца. Моноширинный блок — это код или таблица, там перевод
    строки несёт смысл, и склеивать строки пробелом нельзя.
    """
    ordered = sorted(lines, key=_reading_key)
    if any(line.mono for line in ordered):
        return "\n".join(line.text.rstrip() for line in ordered).strip()
    return " ".join(" ".join(line.text.split()) for line in ordered).strip()


def merge_line_bboxes(lines: Sequence[TextLine]) -> List[float]:
    if not lines:
        return [0.0, 0.0, 1.0, 1.0]
    return [
        min(l.bbox[0] for l in lines),
        min(l.bbox[1] for l in lines),
        max(l.bbox[2] for l in lines),
        max(l.bbox[3] for l in lines),
    ]


def _reading_key(line: TextLine) -> Tuple[float, float]:
    """Порядок чтения: сверху вниз, слева направо, с округлением строки."""
    return (round(line.bbox[1], 3), round(line.bbox[0], 3))


# ===========================================================================
# Сшивка слоя с блоками парсера
# ===========================================================================

@dataclass(frozen=True)
class LayerSource:
    """
    Чем добыт слой. От этого зависит и подпись метода у блока, и то, можно
    ли ставить блоку уверенность 1.0: у векторного слоя текст взят из файла
    и сомнений не вызывает, у слоя из распознавания — вызывает всегда.
    """

    method: str = METHOD_TEXT_LAYER
    recovered_method: str = METHOD_TEXT_LAYER_RECOVERED
    exact: bool = True

    def confidence_of(self, lines: Sequence["TextLine"]) -> float:
        if self.exact:
            return 1.0
        weights = [max(1, line.char_count()) for line in lines]
        total = sum(weights)
        if not total:
            return 0.0
        return round(
            sum(l.confidence * w for l, w in zip(lines, weights)) / total, 3
        )


#: Слой векторного PDF: текст из файла, точный.
VECTOR_LAYER = LayerSource()
#: Слой, собранный построчным распознаванием растра.
OCR_LAYER = LayerSource(
    method=METHOD_OCR_LAYER,
    recovered_method=METHOD_OCR_LAYER_RECOVERED,
    exact=False,
)


@dataclass
class ReconcileStats:
    """Что дала сшивка — уезжает в логи и в сырой результат парсера."""

    rebuilt_blocks: int = 0
    recovered_blocks: int = 0
    recovered_chars: int = 0
    repaired_cells: int = 0
    #: Таблицы, в ячейках которых нашлась проза под LaTeX-разметкой.
    mangled_tables: int = 0
    #: Из них починенные по строкам слоя.
    repaired_tables: int = 0
    #: Из них разобранные на текст и формулы, потому что починить не вышло.
    degraded_tables: int = 0
    layer_chars_by_page: Dict[int, int] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "rebuilt_blocks": self.rebuilt_blocks,
            "recovered_blocks": self.recovered_blocks,
            "recovered_chars": self.recovered_chars,
            "repaired_cells": self.repaired_cells,
            "mangled_tables": self.mangled_tables,
            "repaired_tables": self.repaired_tables,
            "degraded_tables": self.degraded_tables,
        }


def reconcile_with_text_layer(
    blocks: Sequence[ParsedBlock],
    layer: TextLayer,
    source: LayerSource = VECTOR_LAYER,
) -> Tuple[List[ParsedBlock], ReconcileStats]:
    """
    Текст блоков берётся из слоя, а не из OCR; всё, что не попало ни в одну
    рамку, возвращается отдельными блоками. Структура (таблицы, формулы,
    картинки) остаётся за парсером — слой про неё ничего не знает.

    `source` говорит, откуда слой: у векторного PDF это сам файл, у растра —
    построчное распознавание. Разбор от этого не меняется, меняются подпись
    метода и уверенность: выдавать распознанное за текст из файла нельзя.
    """
    stats = ReconcileStats(layer_chars_by_page=layer.char_counts_by_page())
    by_page: Dict[int, List[ParsedBlock]] = {}
    for block in blocks:
        by_page.setdefault(block.page, []).append(block)

    pages = sorted(set(by_page) | set(p for p in layer.pages if layer.char_count(p)))
    result: List[ParsedBlock] = []

    for page in pages:
        page_blocks = by_page.get(page, [])
        page_lines = [l for l in layer.lines(page) if l.char_count()]

        if not page_lines:
            result.extend(page_blocks)
            continue

        # Если MinerU отдал только `content_list`, координат у блоков нет и
        # bbox стоит на весь лист. Сопоставить с ними строки невозможно, и
        # без этой ветки текст удвоился бы: один раз распознанный, другой —
        # восстановленный из слоя. Слой в такой ситуации строго лучше: он и
        # точнее, и с координатами.
        if _all_text_is_coordinateless(page_blocks):
            page_blocks = [b for b in page_blocks if b.type != "text"]

        assignments = _assign_lines(page_lines, page_blocks)

        leftovers = list(assignments.get(None) or [])

        extra_blocks: List[ParsedBlock] = []
        for block in page_blocks:
            matched = assignments.get(id(block)) or []
            if block.type == "text":
                if matched:
                    _rebuild_text_block(block, matched, stats, source)
                else:
                    block.method = block.method or METHOD_PARSER_OCR
            elif block.type == "table" and matched and _carries_own_text(block):
                # Структуру таблицы строит MinerU — слой про неё ничего не
                # знает. Но текст ячеек всё равно распознан, поэтому его
                # значения выправляются по слою.
                extra_blocks.extend(_fix_table(block, matched, stats))
            elif matched and not _carries_own_text(block):
                # Рамка распознана как таблица или картинка, но содержимого
                # в ней не оказалось (у MinerU это обычное дело на сложной
                # вёрстке). Текст из слоя внутри такой рамки иначе пропал бы
                # молча — возвращаем его отдельным блоком.
                leftovers.extend(matched)

        page_blocks.extend(extra_blocks)

        for group in group_lines(leftovers):
            text = lines_to_text(group)
            if not text:
                continue
            stats.recovered_blocks += 1
            stats.recovered_chars += sum(l.char_count() for l in group)
            page_blocks.append(ParsedBlock(
                type="text",
                text=text,
                page=page,
                bbox=merge_line_bboxes(group),
                confidence=source.confidence_of(group),
                method=source.recovered_method,
            ))

        page_blocks.sort(key=_block_reading_key)
        result.extend(page_blocks)

    for order, block in enumerate(result):
        block.order = order

    logger.info(
        "Сшивка со слоем: перезаписано блоков %d, восстановлено блоков %d "
        "(%d символов), выправлено ячеек таблиц %d",
        stats.rebuilt_blocks, stats.recovered_blocks, stats.recovered_chars,
        stats.repaired_cells,
    )
    return result, stats


def _assign_lines(
    lines: Sequence[TextLine], blocks: Sequence[ParsedBlock]
) -> Dict[Any, List[TextLine]]:
    """
    Строка -> рамка, в которую она попала центром. Из нескольких подходящих
    берётся самая мелкая: вложенные рамки встречаются постоянно. Ключ `None`
    собирает всё, что не попало никуда.
    """
    candidates = [b for b in blocks if b.bbox != [0.0, 0.0, 1.0, 1.0]]
    assignments: Dict[Any, List[TextLine]] = {}

    for line in lines:
        cx, cy = line.center
        best: Optional[ParsedBlock] = None
        best_area = float("inf")
        for block in candidates:
            x1, y1, x2, y2 = block.bbox
            inside = (
                x1 - _CONTAINMENT_SLACK <= cx <= x2 + _CONTAINMENT_SLACK
                and y1 - _CONTAINMENT_SLACK <= cy <= y2 + _CONTAINMENT_SLACK
            )
            if not inside:
                continue
            area = block.area()
            if area < best_area:
                best, best_area = block, area
        assignments.setdefault(id(best) if best is not None else None, []).append(line)

    return assignments


def _rebuild_text_block(
    block: ParsedBlock,
    matched: Sequence[TextLine],
    stats: ReconcileStats,
    source: LayerSource = VECTOR_LAYER,
) -> None:
    """
    Текст блока — из слоя. Если из слоя вышло заметно меньше, чем распознал
    парсер, сшивка сработала неверно (например, рамка съехала) — тогда
    оставляем распознанное и честно помечаем это как OCR.
    """
    rebuilt = lines_to_text(matched)
    if not rebuilt:
        block.method = block.method or METHOD_PARSER_OCR
        return

    existing = len("".join((block.text or "").split()))
    fresh = len("".join(rebuilt.split()))
    if existing and fresh < existing * _REBUILD_MIN_RATIO:
        logger.debug(
            "Слой дал %d симв. против %d у парсера на стр. %d — оставляем распознанное",
            fresh, existing, block.page,
        )
        block.method = METHOD_PARSER_OCR
        return

    block.text = rebuilt
    block.bbox = merge_line_bboxes(matched)
    block.method = source.method
    block.confidence = source.confidence_of(matched)
    stats.rebuilt_blocks += 1


def fold_homoglyphs(text: str) -> str:
    """
    Строка -> представление, в котором визуально одинаковые буквы разных
    алфавитов неразличимы. Нужно только для сравнения: наружу всегда уходит
    текст из файла, а не свёрнутый.
    """
    return "".join(_HOMOGLYPHS.get(ch, ch) for ch in text.lower() if not ch.isspace())


def _fix_table(
    block: ParsedBlock, matched: Sequence[TextLine], stats: ReconcileStats
) -> List[ParsedBlock]:
    """
    Содержимое ячеек — по строкам слоя. Возвращает блоки, которыми таблицу
    пришлось заменить (пусто, если заменять не пришлось).

    Обычный случай — выправка значений: распознанный текст ячейки меняется
    на строку слоя, если это заведомо она же. Особый случай — ячейка, в
    которой лежит не текст, а проза под LaTeX-разметкой (модель таблиц
    MinerU заворачивает так русские описания). Сравнивать такую ячейку со
    строкой бессмысленно: её надо заменить целиком, и решает это геометрия.
    Не сошлась геометрия — таблица разбирается на текст и формулы, потому
    что выдать разметку побуквенно в индекс хуже любой потери структуры.
    """
    if not latex_text.mangled_cells(block.table_data):
        _repair_table_cells(block, matched, stats)
        return []

    stats.mangled_tables += 1
    if table_from_lines.repair_cells(block, matched):
        stats.repaired_tables += 1
        _repair_table_cells(block, matched, stats)
        return []

    stats.degraded_tables += 1
    return table_from_lines.degrade(block, matched)


def _repair_table_cells(
    block: ParsedBlock, matched: Sequence[TextLine], stats: ReconcileStats
) -> None:
    """
    Значения ячеек заменяются на текст из слоя, если это заведомо та же
    строка. Каждая строка слоя расходуется один раз, поэтому одинаковые
    ячейки не схлопываются в одну.
    """
    table = block.table_data or {}
    pool = [" ".join(line.text.split()) for line in matched if line.text.strip()]
    if not pool:
        return

    folded_pool = [fold_homoglyphs(text) for text in pool]
    used: set = set()

    def repair(value: Any) -> Any:
        text = str(value)
        if not text.strip():
            return value
        target = fold_homoglyphs(text)
        best_index, best_ratio = -1, 0.0
        for index, candidate in enumerate(folded_pool):
            if index in used or not candidate:
                continue
            ratio = difflib.SequenceMatcher(None, target, candidate).ratio()
            if ratio > best_ratio:
                best_index, best_ratio = index, ratio
        if best_index < 0 or best_ratio < _CELL_MATCH_RATIO:
            return value
        used.add(best_index)
        if pool[best_index] != text:
            stats.repaired_cells += 1
        return pool[best_index]

    table["headers"] = [repair(h) for h in (table.get("headers") or [])]
    table["rows"] = [[repair(c) for c in row] for row in (table.get("rows") or [])]
    block.table_data = table


def _all_text_is_coordinateless(blocks: Sequence[ParsedBlock]) -> bool:
    """Есть текстовые блоки, и ни у одного нет собственных координат."""
    text_blocks = [b for b in blocks if b.type == "text"]
    if not text_blocks:
        return False
    return all(b.bbox == [0.0, 0.0, 1.0, 1.0] for b in text_blocks)


def _carries_own_text(block: ParsedBlock) -> bool:
    """Есть ли у нетекстового блока собственное содержимое."""
    if block.type == "table":
        table = block.table_data or {}
        return bool(table.get("rows") or table.get("headers"))
    if block.type == "formula":
        return bool(block.text and block.text.strip())
    return block.type == "image"


def _block_reading_key(block: ParsedBlock) -> Tuple[float, float]:
    return (round(block.bbox[1], 3), round(block.bbox[0], 3))
