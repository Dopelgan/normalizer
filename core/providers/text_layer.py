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

import dataclasses
import difflib
import logging
import math
import re
import statistics
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

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
#: Разметку сняли, не имея чем заменить: читается глазами, но не более.
METHOD_LATEX_DECODED = "latex_decoded"
#: Таблица перечитана моделью зрения (уровень 7 лестницы стратегий).
METHOD_VLM_TABLE = "vlm_escalation"

_MONO_MARKERS = ("mono", "courier", "consol", "menlo", "inconsolata", "hack", "fira code")

# Скан с наложенной надписью: страницу занимает одна картинка, а в слое лежит
# только штамп («ЭКСПЕРТНОЕ ЗАКЛЮЧЕНИЕ № …», ~70 символов). Такая страница —
# растр, её текст есть только в картинке. Распознанный скан (невидимый слой
# OCR поверх картинки) несёт сотни символов и остаётся текстовым.
_SCAN_IMAGE_SHARE = 0.5
_OVERLAY_MAX_CHARS = 200

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

# Доля слов строки, которая должна найтись в разобранных таблицах, чтобы
# счесть строку повтором содержимого таблицы, и минимальная длина строки
# в словах: по двум словам повтор не опознаётся.
_DUPLICATE_RATIO = 0.8
_DUPLICATE_MIN_WORDS = 4

_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)

# Склейка строк ячейки. Модель таблиц MinerU собирает многострочную ячейку
# скана без разделителя («Креплениебиозащитына»), и слова пропадают для
# поиска. Режутся только токены не короче стольких букв и цифр и не длиннее
# предела: дальше перебор разбиений дорог, а склейки такой длины не бывает.
_GLUED_MIN_ALNUM = 6
_GLUED_MAX_LEN = 200
# Знаки, которые снимаются с краёв слова слоя при сборке словаря.
_WORD_EDGE = "|.,;:!?()[]{}<>«»\"'*_-–—/\\"
_MARKUP_RE = re.compile(r"(<[^>]*>)")
# Кусок склейки короче трёх букв берётся, только если это служебное слово:
# словарь слоя с мусором OCR («ре», «ный») иначе режет настоящие слова.
_SHORT_WORDS = frozenset(
    "в во на по с со к ко и о об у от до из за не ни для а но же ли "
    "a an of to in on at by or and the for".split()
)
# Склейка «известное слово + неизвестное»: длина известной части и остатка.
_KNOWN_PART_MIN = 7
_UNKNOWN_PART_MIN = 6
# Знаки, после которых склейка режется, если за ними идёт слово из букв
# («См.ТАДУ»). Перед цифрами и после «/» граница не ставится: это шифр.
_CUT_AFTER = ".,;:)»"
_VOWELS = set("аеёиоуыэюяaeiouy")

# Подписи основной надписи чертежа (ГОСТ 2.104). Чертёжный курсив модель
# MinerU и Tesseract читают с устойчивыми подменами: «Разрад.», «Проб.»,
# «Лucт», «Macwmad», «FOCT», «Дама». Словарь закрытый, поэтому сверка с ним
# идёт по «курсивной» свёртке, которая склеивает эти пары букв, — и только в
# таблице, которая сама похожа на штамп (см. `_looks_like_stamp`).
# Группы подписей: графы изменений, подписи, характеристики листа, поля
# формата и учёта. «ГОСТ» — не подпись: в штамп он попадает графой
# материала, но признаком штампа не считается.
_STAMP_GROUPS = (
    ("Изм", "Лист", "Листов", "докум", "Подп", "Дата"),
    ("Разраб", "Пров", "контр", "Утв"),
    ("Лит", "Масса", "Масштаб"),
    ("Копировал", "Формат", "Инв", "подл", "Взам", "дубл", "Перв", "примен", "Справ"),
)
# Подписи исполнителей; «контр» — из «Т.контр.»/«Н.контр.»: графа есть
# только в штампе и читается чаще, чем курсивные «Разраб.»/«Пров.».
_STAMP_SIGNATURES = ("Разраб", "Пров", "Утв", "контр")
_STAMP_LABELS = [label for group in _STAMP_GROUPS for label in group] + ["ГОСТ"]
# Штамп: не меньше стольких разных подписей буква в букву, из не меньше
# стольких групп, среди них подпись исполнителя, и таблица внизу листа
# (нижний край рамки ниже этой доли высоты). Таблица изменений или
# спецификация с «Изм. Лист № докум. Подп. Дата» штампом так не станет.
_STAMP_MIN_LABELS = 4
_STAMP_MIN_GROUPS = 2
_STAMP_MIN_BOTTOM = 0.75
# Лист повёрнут боком, если так повёрнуто не меньше этой доли символов
# страницы; строка считается повёрнутой кратно 90° с этим допуском, градусы.
_PAGE_TURN_SHARE = 0.6
_TURN_TOLERANCE = 10.0
# Строка участвует в решении о повороте листа, если в ней столько букв.
_TURN_MIN_LETTERS = 3
# Курсивные подмены поверх обычной свёртки: т/m, и/у/u, б/в/д/6, г/F, ш/w.
# Пары а/о здесь нет: «прав.» не должно стать «Пров.». В открытом тексте
# так сравнивать нельзя — только с закрытым словарём штампа.
_CURSIVE = {"m": "t", "y": "u", "b": "d", "б": "d", "6": "d", "f": "r",
            "ш": "w", "щ": "w"}
# Наименование в штампе с приклеенной цифрой соседней графы («Крышка2»).
_WORD_NUMBER_RE = re.compile(r"^([^\W\d_]{4,})(\d{1,3})$")
# Число внутри склейки: цифры со знаками, начинается и кончается цифрой.
_NUMBER_RE = re.compile(r"^\d(?:[\d.,:/\-]*\d)?$")
# Марка стали по ГОСТ латиницей («12X18H10T» вместо «12Х18Н10Т»): число,
# затем не меньше двух легирующих букв с числами до двух цифр. Только
# X — это размер («770X520X320»), а не марка.
_STEEL_GRADE_RE = re.compile(r"^\d{1,3}(?:[ABCEHKMPTXАВСЕНКМРТХ]\d{0,2}){2,}$")

# Строки слоя внутри разобранной таблицы, не пошедшие на починку ячеек.
# Строка возвращается текстом, только если несёт слова (длиннее трёх букв),
# которых в таблице нет: не меньше стольких новых слов и не меньше такой
# их доли среди слов строки длиннее двух букв. Иначе это
# повтор содержимого ячеек или мусор от линий сетки.
_TABLE_LINE_MIN_NEW_WORDS = 2
_TABLE_LINE_MIN_NEW_SHARE = 0.5

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
    # Куски строки распознавания между широкими пробелами; пусто — строка
    # не делится. Берутся вместо строки только на развороте с колонками
    # прозы (`column_lines`).
    parts: Tuple["TextLine", ...] = ()

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
    # Доля страницы под самой большой растровой картинкой.
    image_share: Dict[int, float] = field(default_factory=dict)

    def lines(self, page: int) -> List[TextLine]:
        return self.pages.get(page, [])

    def char_count(self, page: Optional[int] = None) -> int:
        if page is not None:
            return sum(line.char_count() for line in self.lines(page))
        return sum(line.char_count() for lines in self.pages.values() for line in lines)

    def char_counts_by_page(self) -> Dict[int, int]:
        return {page: self.char_count(page) for page in range(1, self.page_count + 1)}

    def overlay_only(self, page: int) -> bool:
        """Страница — скан, а слой несёт только наложенную на него надпись."""
        return (self.image_share.get(page, 0.0) >= _SCAN_IMAGE_SHARE
                and self.char_count(page) < _OVERLAY_MAX_CHARS)

    def pages_with_text(self) -> int:
        return sum(1 for page, lines in self.pages.items()
                   if any(l.char_count() for l in lines) and not self.overlay_only(page))

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
            layer.image_share[page_no] = _largest_image_share(page)
        except Exception as exc:  # noqa: BLE001 — одна битая страница не роняет документ
            logger.warning("Страница %d не прочиталась из текстового слоя: %s", page_no, exc)
            layer.pages[page_no] = []

    logger.info(
        "Текстовый слой: %d стр., %d стр. с текстом, %d символов",
        layer.page_count, layer.pages_with_text(), layer.char_count(),
    )
    return layer


def _largest_image_share(page: Any) -> float:
    """Доля страницы, которую закрывает самая большая растровая картинка."""
    area = abs(page.rect)
    if not area:
        return 0.0
    try:
        images = page.get_image_info()
    except Exception as exc:  # noqa: BLE001 — без картинок страница остаётся текстовой
        logger.debug("Картинки страницы не прочитаны: %s", exc)
        return 0.0
    shares = [abs(_pymupdf.Rect(item["bbox"]) & page.rect) / area for item in images if item.get("bbox")]
    return round(min(1.0, max(shares, default=0.0)), 3)


def _lines_from_page(page: Any, page_no: int, width: float, height: float) -> List[TextLine]:
    """Строки страницы с нормализованными координатами."""
    if width <= 0 or height <= 0:
        return []

    raw = page.get_text("dict")
    lines: List[TextLine] = []
    # У повёрнутой страницы (/Rotate, альбомный лист) слой отдаёт координаты
    # неповёрнутой, а `page.rect`, растр и рамки MinerU — повёрнутой. Без
    # пересчёта строки съезжали за край, прижимались к 1.0 и не попадали ни
    # в одну рамку: таблица оставалась с текстом модели, а слой уходил
    # плоским текстом мимо неё.
    turn = int(getattr(page, "rotation", 0) or 0) % 360
    matrix = page.rotation_matrix if turn else None

    for block in raw.get("blocks") or []:
        if block.get("type") != 0:      # 0 — текст, 1 — растровая картинка
            continue
        for line in block.get("lines") or []:
            spans = line.get("spans") or []
            text = "".join(span.get("text", "") for span in spans)
            if not text.strip():
                continue

            bbox = line.get("bbox") or [0, 0, 0, 0]
            if matrix is not None:
                turned = _pymupdf.Rect(bbox) * matrix
                bbox = [turned.x0, turned.y0, turned.x1, turned.y1]
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
                rotation=_turn_back(_direction_to_degrees(line.get("dir")), turn),
                color=statistics.mode(colors) if colors else 0,
                mono=any(marker in fonts for marker in _MONO_MARKERS),
            ))

    return lines


def _turn_back(angle: float, turn: int) -> float:
    """Угол письма на странице в том виде, в каком её видно (с учётом /Rotate)."""
    if not turn:
        return angle
    return round((angle - turn + 180.0) % 360.0 - 180.0, 1)


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

def group_lines(lines: Sequence[TextLine], turn: Optional[int] = None) -> List[List[TextLine]]:
    """
    Соседние строки -> абзацы. Разрыв — там, где вертикальный зазор заметно
    больше обычного межстрочного, меняется угол поворота или меняется
    моноширинность (граница «проза / код»).

    `turn` — поворот текста страницы (`page_turn`); по умолчанию считается
    по самим строкам, поэтому передавать сюда надо строки всей страницы.
    """
    if not lines:
        return []

    # На листе, повёрнутом боком, «сверху вниз» считается по прямо
    # стоящему листу, иначе вертикальные строки слипаются в один абзац.
    pairs = _reading_pairs(lines, page_turn(lines) if turn is None else turn)
    heights = [upright.height for upright, _ in pairs if upright.height > 0]
    typical = statistics.median(heights) if heights else 0.012
    gap_limit = max(typical * 0.85, 0.004)

    groups: List[List[Tuple[TextLine, TextLine]]] = [[pairs[0]]]
    for upright, line in pairs[1:]:
        previous = groups[-1][-1][0]
        gap = upright.bbox[1] - previous.bbox[3]
        same_style = (line.mono == previous.mono) and (abs(line.rotation - previous.rotation) < 5)
        if gap > gap_limit or not same_style:
            groups.append([(upright, line)])
        else:
            groups[-1].append((upright, line))
    return [[line for _, line in group] for group in groups]


# Колонка — это проза: столько строк и такая медианная длина строки.
_COLUMN_MIN_LINES = 4
_COLUMN_MIN_CHARS = 20
_COLUMN_SLACK = 0.005


def group_columns(lines: Sequence[TextLine], turn: Optional[int] = None) -> List[List[TextLine]]:
    """
    Как `group_lines`, но колонки прозы собираются в абзацы по отдельности,
    слева направо (`_prose_columns`). Нет колонок прозы — как `group_lines`.
    """
    if not lines:
        return []
    turn = page_turn(lines) if turn is None else turn
    columns = _prose_columns(lines, turn)
    if columns is None:
        return group_lines(lines, turn)
    groups: List[List[TextLine]] = []
    for column in columns:
        groups.extend(group_lines(column, turn))
    return groups


def column_lines(lines: Sequence[TextLine], turn: Optional[int] = None) -> List[TextLine]:
    """
    Строки страницы, где строка через разворот заменена своими кусками —
    но только если по кускам лист делится на колонки прозы. Иначе строки
    как есть: в таблице куски — это ячейки, и ряд должен остаться рядом.
    """
    if not any(line.parts for line in lines):
        return list(lines)
    flat = [part for line in lines for part in (line.parts or (line,))]
    turn = page_turn(flat) if turn is None else turn
    return flat if _prose_columns(flat, turn) is not None else list(lines)


def _prose_columns(
    lines: Sequence[TextLine], turn: int
) -> Optional[List[List[TextLine]]]:
    """
    Колонки — участки по горизонтали (на прямо стоящем листе), которые не
    пересекает ни одна строка. Делится только проза: в каждой колонке не
    меньше `_COLUMN_MIN_LINES` строк с медианной длиной от
    `_COLUMN_MIN_CHARS`. Столбцы таблицы (короткие ячейки) колонками не
    считаются, иначе ряд таблицы развалился бы на столбцы.
    """
    pairs = list(zip(_upright_lines(lines, turn), lines))
    pairs.sort(key=lambda pair: pair[0].bbox[0])
    columns: List[List[TextLine]] = []
    right = -1.0
    for upright, line in pairs:
        if columns and upright.bbox[0] <= right + _COLUMN_SLACK:
            columns[-1].append(line)
            right = max(right, upright.bbox[2])
        else:
            columns.append([line])
            right = upright.bbox[2]
    if len(columns) < 2:
        return None
    for column in columns:
        lengths = sorted(len(line.text.strip()) for line in column)
        if len(lengths) < _COLUMN_MIN_LINES or lengths[len(lengths) // 2] < _COLUMN_MIN_CHARS:
            return None
    return columns


def _reading_pairs(lines: Sequence[TextLine], turn: int) -> List[Tuple[TextLine, TextLine]]:
    """(строка с рамкой прямо стоящего листа, исходная строка) в порядке чтения."""
    pairs = list(zip(_upright_lines(lines, turn), lines))
    pairs.sort(key=lambda pair: _reading_key(pair[0]))
    return pairs


def lines_to_text(lines: Sequence[TextLine], turn: Optional[int] = None) -> str:
    """
    Текст абзаца. Моноширинный блок — это код или таблица, там перевод
    строки несёт смысл, и склеивать строки пробелом нельзя. `turn` — как у
    `group_lines`: поворот всей страницы, а не только этих строк.
    """
    ordered = [line for _, line in _reading_pairs(lines, page_turn(lines) if turn is None else turn)]
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
# Лист, повёрнутый боком
# ===========================================================================

def page_turn(lines: Sequence[TextLine]) -> int:
    """
    Поворот текста страницы кратно 90° против часовой (0, 90, 180, 270),
    если так повёрнута большая часть её текста; иначе 0.

    Скан альбомной страницы без /Rotate Tesseract читает после поворота по
    OSD, а рамки строк отдаёт в системе исходного листа — там строки
    вертикальные. Решает доля букв, а не всех символов: подписи размеров
    чертежа («100±0,2») повёрнуты поодиночке и почти без букв, и на
    разреженном листе их цифры перевешивали бы штамп и требования. Лист
    без букв считается по всем символам.
    """
    lettered = [line for line in lines if _letters(line.text) >= _TURN_MIN_LETTERS]
    weights: Dict[int, int] = {}
    total = 0
    for line in lettered or lines:
        chars = _letters(line.text) if lettered else line.char_count()
        total += chars
        turn = _quarter(line.rotation)
        if turn is not None:
            weights[turn] = weights.get(turn, 0) + chars
    if not total or not weights:
        return 0
    turn, chars = max(weights.items(), key=lambda item: item[1])
    return turn if turn and chars >= total * _PAGE_TURN_SHARE else 0


def _letters(text: str) -> int:
    return sum(1 for ch in text if ch.isalpha())


def _quarter(rotation: float) -> Optional[int]:
    """Угол письма -> 0/90/180/270, если он кратен 90° с допуском; иначе None."""
    quarter = round(rotation / 90.0)
    if abs(rotation - quarter * 90.0) > _TURN_TOLERANCE:
        return None
    return (quarter * 90) % 360


def upright_bbox(bbox: Sequence[float], turn: int) -> List[float]:
    """
    Рамка в долях исходного листа -> рамка на том же листе, повёрнутом так,
    чтобы текст стоял прямо. `turn` — поворот текста против часовой
    (`page_turn`): при 90° строки идут снизу вверх, и лист поворачивается
    на 90° по часовой.
    """
    x1, y1, x2, y2 = (float(v) for v in bbox)
    if turn == 90:
        return [1 - y2, x1, 1 - y1, x2]
    if turn == 180:
        return [1 - x2, 1 - y2, 1 - x1, 1 - y1]
    if turn == 270:
        return [y1, 1 - x2, y2, 1 - x1]
    return [x1, y1, x2, y2]


def _upright_lines(lines: Sequence[TextLine], turn: int) -> List[TextLine]:
    """
    Копии строк с рамками прямо стоящего листа — только для геометрии.
    Переводятся строки, написанные в направлении страницы; строки другого
    направления (горизонтальная подпись на листе, повёрнутом боком)
    остаются в системе страницы — как до поворота, иначе их «сверху вниз»
    становилось бы «слева направо» и далёкие строки слипались бы в абзац.
    """
    if not turn:
        return list(lines)
    return [
        dataclasses.replace(line, bbox=upright_bbox(line.bbox, turn))
        if _quarter(line.rotation) == turn else line
        for line in lines
    ]


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
    mangled_formulas: int = 0
    """Формульных блоков, в которых лежала проза под разметкой."""
    repaired_formulas: int = 0
    """Из них заменено текстом строк слоя."""
    decoded_formulas: int = 0
    """Из них заменить было нечем — снята разметка, блок на проверку."""
    duplicate_lines: int = 0
    """Строк, погашенных как повтор содержимого разобранной таблицы."""
    dropped_empty_tables: int = 0
    """Пустых рамок-продолжений многостраничной таблицы, убранных из выдачи."""
    mangled_tables: int = 0
    #: Из них починенные по строкам слоя.
    repaired_tables: int = 0
    #: Из них разобранные на текст и формулы, потому что починить не вышло.
    degraded_tables: int = 0
    #: Склеенных токенов в ячейках таблиц, разрезанных по словам слоя.
    split_tokens: int = 0
    #: Строк слоя из рамки таблицы, возвращённых текстом: их слов в ячейках нет.
    table_lines_kept: int = 0
    #: «Таблиц» во весь лист с прозой внутри — рамок ЕСКД, разобранных в текст.
    frame_tables: int = 0
    #: Строк слоя из рамки таблицы, отброшенных по уверенности.
    low_confidence_table_lines: int = 0
    #: Неразобранных таблиц, перечитанных моделью зрения (`table_reader`).
    vlm_tables: int = 0
    layer_chars_by_page: Dict[int, int] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "rebuilt_blocks": self.rebuilt_blocks,
            "recovered_blocks": self.recovered_blocks,
            "recovered_chars": self.recovered_chars,
            "repaired_cells": self.repaired_cells,
            "mangled_formulas": self.mangled_formulas,
            "repaired_formulas": self.repaired_formulas,
            "decoded_formulas": self.decoded_formulas,
            "duplicate_lines": self.duplicate_lines,
            "dropped_empty_tables": self.dropped_empty_tables,
            "mangled_tables": self.mangled_tables,
            "repaired_tables": self.repaired_tables,
            "degraded_tables": self.degraded_tables,
            "split_tokens": self.split_tokens,
            "table_lines_kept": self.table_lines_kept,
            "frame_tables": self.frame_tables,
            "low_confidence_table_lines": self.low_confidence_table_lines,
            "vlm_tables": self.vlm_tables,
        }


def reconcile_with_text_layer(
    blocks: Sequence[ParsedBlock],
    layer: TextLayer,
    source: LayerSource = VECTOR_LAYER,
    frame_table_guard: bool = False,
    keep_structured_tables: bool = False,
    table_line_min_confidence: float = 0.0,
    split_columns: bool = False,
    table_reader: Optional[Callable[[ParsedBlock], Optional[Dict[str, Any]]]] = None,
) -> Tuple[List[ParsedBlock], ReconcileStats]:
    """
    Текст блоков берётся из слоя, а не из OCR; всё, что не попало ни в одну
    рамку, возвращается отдельными блоками. Структура (таблицы, формулы,
    картинки) остаётся за парсером — слой про неё ничего не знает.

    `source` говорит, откуда слой: у векторного PDF это сам файл, у растра —
    построчное распознавание. Разбор от этого не меняется, меняются подпись
    метода и уверенность: выдавать распознанное за текст из файла нельзя.

    `keep_structured_tables` — рамка во весь лист с настоящими строками
    таблицы (спецификация ЕСКД) не снимается защитой рамок, даже если под
    таблицей проза примечаний. `table_line_min_confidence` — строки
    распознавания из рамки таблицы, которые иначе вернулись бы текстом,
    отбрасываются ниже этой уверенности: это повтор той же таблицы,
    прочитанный через линии сетки. Оба — только для слоя распознавания.
    `split_columns` — строки вне рамок собираются в абзацы по колонкам
    (`group_columns`).

    `table_reader` — последняя попытка для таблицы, которую не удалось ни
    выправить, ни собрать по строкам слоя: блок -> `table_data` или None.
    Без него (или при None) таблица разбирается на текст, как раньше.
    """
    stats = ReconcileStats(layer_chars_by_page=layer.char_counts_by_page())
    if frame_table_guard:
        blocks = _drop_frame_tables(
            blocks, layer, stats, keep_structured=keep_structured_tables and not source.exact
        )
    line_floor = table_line_min_confidence if not source.exact else 0.0
    by_page: Dict[int, List[ParsedBlock]] = {}
    for block in blocks:
        by_page.setdefault(block.page, []).append(block)

    pages = sorted(set(by_page) | set(p for p in layer.pages if layer.char_count(p)))
    result: List[ParsedBlock] = []
    # Содержимое всех разобранных таблиц документа одной строкой. Нужно на
    # многостраничной таблице: MinerU отдаёт её целиком в рамке первой
    # страницы, а на страницах-продолжениях — пустую рамку. Строки из такой
    # рамки иначе уезжают в индекс плоским текстом второй копией.
    parsed_tables = _table_index(blocks)

    for page in pages:
        page_blocks = by_page.get(page, [])
        page_lines = [l for l in layer.lines(page) if l.char_count()]
        if split_columns and not source.exact:
            # Разворот в две колонки: строка через обе колонки заменяется
            # кусками, и каждый уходит в свою рамку и свою колонку.
            page_lines = column_lines(page_lines)

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
        # Лист, повёрнутый боком: порядок блоков, место штампа и колонки
        # таблиц считаются по прямо стоящему листу, рамки остаются как есть.
        turn = page_turn(page_lines)
        # Словарь страницы: по нему режутся склейки в ячейках MinerU. Модель
        # таблиц склеивает строки ячейки и на векторном листе — слой файла
        # там точный, и словарь из него только надёжнее.
        vocab = layer_vocabulary(page_lines)

        leftovers = list(assignments.get(None) or [])

        extra_blocks: List[ParsedBlock] = []
        dropped: List[ParsedBlock] = []
        for block in page_blocks:
            matched = assignments.get(id(block)) or []
            if block.type == "text":
                if matched:
                    _rebuild_text_block(block, matched, stats, source, turn)
                else:
                    block.method = block.method or METHOD_PARSER_OCR
                if not source.exact and block.text:
                    # Распознанный текст вне таблиц страдает тем же: латиница
                    # в русских словах, склейки. Правила те же, без штампа.
                    block.text, count = split_glued_text(block.text, vocab)
                    stats.split_tokens += count
            elif block.type == "formula" and latex_text.is_mangled_text(block.text or ""):
                # Модель формул MinerU заворачивает в LaTeX любую строку,
                # которую сочла формулой, — на скане так уезжают заголовки
                # и подписи целиком. Строки слоя внутри такой рамки раньше
                # просто отбрасывались вместе с их текстом.
                _fix_formula(block, matched, stats, source, turn)
            elif block.type == "table" and matched and _carries_own_text(block):
                # Структуру таблицы строит MinerU — слой про неё ничего не
                # знает. Но текст ячеек всё равно распознан, поэтому его
                # значения выправляются по слою.
                replacements, kept = _fix_table(
                    block, matched, stats, vocab, exact=source.exact, turn=turn,
                    table_reader=table_reader,
                )
                extra_blocks.extend(replacements)
                leftovers.extend(_confident(kept, line_floor, stats))
            elif matched and not _carries_own_text(block):
                # Рамка распознана как таблица или картинка, но содержимого
                # в ней не оказалось (у MinerU это обычное дело на сложной
                # вёрстке). Текст из слоя внутри такой рамки иначе пропал бы
                # молча — возвращаем его отдельным блоком. Кроме случая,
                # когда это продолжение таблицы, уже разобранной целиком:
                # тогда возвращать нечего, а пустую рамку незачем держать.
                if block.type == "table":
                    fresh = _drop_duplicates(matched, parsed_tables, stats)
                    if parsed_tables:
                        # Пустая рамка — продолжение уже разобранной таблицы:
                        # неуверенные строки в ней — та же таблица через сетку.
                        # Без разобранных таблиц слой — единственный источник.
                        fresh = _confident(fresh, line_floor, stats)
                    if not fresh:
                        stats.dropped_empty_tables += 1
                        dropped.append(block)
                else:
                    # Гашение повторов — про таблицы. Подпись под картинкой
                    # может повторять строку таблицы на законных основаниях.
                    fresh = list(matched)
                leftovers.extend(fresh)

        if dropped:
            page_blocks = [b for b in page_blocks if not any(b is d for d in dropped)]
        page_blocks.extend(extra_blocks)

        grouping = group_columns if split_columns else group_lines
        for group in grouping(leftovers, turn):
            text = lines_to_text(group, turn)
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

        page_blocks.sort(key=lambda block: _block_reading_key(block, turn))
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


# «Таблица» MinerU, которая занимает столько листа, — рамка формата.
_FRAME_TABLE_AREA = 0.5
# Настоящая таблица: строк хотя бы с двумя заполненными ячейками не меньше
# стольких и не меньше такой доли строк, а ячейки короткие (не строки прозы).
_STRUCTURED_MIN_ROWS = 5
_STRUCTURED_MIN_SHARE = 0.4
_STRUCTURED_MAX_CELL = 40


def _confident(
    lines: Sequence[TextLine], floor: float, stats: "ReconcileStats"
) -> List[TextLine]:
    """Строки не ниже `floor` по уверенности; остальные считаются в статистике."""
    if floor <= 0:
        return list(lines)
    kept = [line for line in lines if line.confidence >= floor]
    stats.low_confidence_table_lines += len(lines) - len(kept)
    return kept


def _structured_table(block: ParsedBlock) -> bool:
    """
    В рамке настоящая таблица, а не текст листа: много строк с несколькими
    короткими заполненными ячейками (позиция, обозначение, наименование).
    Проза в рамке ЕСКД так не выглядит: у неё одна длинная ячейка на строку.
    """
    rows = (block.table_data or {}).get("rows") or []
    if not rows:
        return False
    filled_rows = 0
    lengths: List[int] = []
    for row in rows:
        filled = [str(c).strip() for c in row if str(c).strip()]
        lengths.extend(len(c) for c in filled)
        if len(filled) >= 2:
            filled_rows += 1
    if not lengths:
        return False
    return (
        filled_rows >= _STRUCTURED_MIN_ROWS
        and filled_rows >= len(rows) * _STRUCTURED_MIN_SHARE
        and statistics.median(lengths) <= _STRUCTURED_MAX_CELL
    )


def _drop_frame_tables(
    blocks: Sequence[ParsedBlock], layer: TextLayer, stats: ReconcileStats,
    keep_structured: bool = False,
) -> List[ParsedBlock]:
    """
    Текст в рамке ЕСКД MinerU принимает за одну таблицу во весь лист, а
    многостраничную — за одну таблицу на всю книгу (ПМ1: 3626 подчанков).
    Такая рамка убирается до сшивки: строки слоя внутри неё возвращаются
    обычным текстом, и в индекс таблицы этой книги она не попадает.
    Настоящая таблица во весь лист остаётся: в её ячейках не проза.
    """
    from core.providers import page_prose

    kept: List[ParsedBlock] = []
    dropped_lines: List[str] = []
    for block in blocks:
        if (block.type == "table" and block.area() >= _FRAME_TABLE_AREA
                and not (keep_structured and _structured_table(block))):
            x1, y1, x2, y2 = block.bbox
            inside = [
                line.text for line in layer.lines(block.page)
                if x1 <= line.center[0] <= x2 and y1 <= line.center[1] <= y2
            ]
            if inside and page_prose.assess(inside).is_prose:
                stats.frame_tables += 1
                dropped_lines.extend(inside)
                continue
        kept.append(block)

    # Многостраничную таблицу MinerU отдаёт целиком в рамке первого листа,
    # а продолжения — пустыми рамками. Если первый лист сам не проза
    # (титул, «УТВЕРЖДАЮ»), его «таблица» осталась бы и несла весь текст
    # книги, а строки продолжений вернулись бы текстом второй копией.
    # Поэтому снимается и та рамка во весь лист, в которой лежат строки
    # уже снятых рамок.
    probes = [_squash(t) for t in dropped_lines if len(_squash(t)) >= 20]
    if not probes:
        return kept
    result: List[ParsedBlock] = []
    for block in kept:
        if (block.type == "table" and block.area() >= _FRAME_TABLE_AREA
                and not (keep_structured and _structured_table(block))):
            own = _squash(_table_text(block))
            hits = sum(1 for p in probes if p in own)
            if own and hits >= min(3, len(probes)):
                stats.frame_tables += 1
                continue
        result.append(block)
    return result


def _squash(text: str) -> str:
    return "".join(fold_homoglyphs(text or "").split()).lower()


def _table_text(block: ParsedBlock) -> str:
    return f"{block.text or ''} {_table_index([block])}"


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
    turn: int = 0,
) -> None:
    """
    Текст блока — из слоя. Если из слоя вышло заметно меньше, чем распознал
    парсер, сшивка сработала неверно (например, рамка съехала) — тогда
    оставляем распознанное и честно помечаем это как OCR.
    """
    rebuilt = lines_to_text(matched, turn)
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


def _fix_formula(
    block: ParsedBlock,
    matched: Sequence[TextLine],
    stats: "ReconcileStats",
    source: LayerSource = VECTOR_LAYER,
    turn: int = 0,
) -> None:
    """
    Формульный блок, в котором лежит не формула, а проза под разметкой.

    Модель формул MinerU заворачивает в LaTeX всё, что сочла формулой, и на
    скане так уезжает обычный текст: `\\mathsf { A K T T E X H M 4 E C K O
    P O O C M O T P A }` — это «АКТ ТЕХНИЧЕСКОГО ОСМОТРА». Собрать строку
    обратно из разметки нельзя, буквы потеряны при распознавании. Зато тот
    же кусок листа прочитан построчно, и текст можно взять оттуда — ровно
    так же, как это делается для ячеек таблиц.

    Заменить нечем — снимаем разметку и помечаем блок фоллбэком: голый
    текст хотя бы читается и ищется, а `is_fallback` отправит фрагмент на
    проверку человеку. Отдавать побуквенную разметку в индекс хуже.
    """
    stats.mangled_formulas += 1
    rebuilt = lines_to_text(matched, turn)
    if rebuilt:
        block.type = "text"
        block.text = rebuilt
        block.bbox = merge_line_bboxes(matched)
        block.method = source.method
        block.confidence = source.confidence_of(matched)
        stats.repaired_formulas += 1
        return

    block.type = "text"
    block.text = latex_text.decode(block.text or "")
    block.method = METHOD_LATEX_DECODED
    block.is_fallback = True
    stats.decoded_formulas += 1


def _table_index(blocks: Sequence[ParsedBlock]) -> str:
    """Содержимое всех разобранных таблиц документа одной свёрнутой строкой."""
    parts: List[str] = []
    for block in blocks:
        table = block.table_data or {}
        parts.extend(str(h) for h in (table.get("headers") or []))
        for row in table.get("rows") or []:
            parts.extend(str(cell) for cell in row)
    return fold_homoglyphs(" ".join(parts)) if parts else ""


def _drop_duplicates(
    lines: Sequence[TextLine], parsed_tables: str, stats: "ReconcileStats"
) -> List[TextLine]:
    """
    Строки, которых ещё нет ни в одной разобранной таблице документа.

    Многостраничную таблицу MinerU отдаёт целиком в рамке первой страницы,
    а дальше присылает пустые рамки-продолжения. Строки слоя из такой рамки
    — это те же самые данные во второй раз, уже без колонок: в индексе от
    них один шум, а в поиске они перебивают разобранную таблицу.

    Сравнение по словам, а не по строке целиком: модель таблиц теряет и
    переставляет содержимое ячеек, так что дословно строка в разобранной
    таблице не находится почти никогда. Короткие слова не в счёт — из
    «шт» и «5» повтора не соберёшь.
    """
    if not parsed_tables:
        return list(lines)

    fresh: List[TextLine] = []
    for line in lines:
        words = [w for w in _TOKEN_RE.findall(line.text or "") if len(w) >= 3]
        folded = [fold_homoglyphs(w) for w in words]
        found = sum(1 for w in folded if w and w in parsed_tables)
        if len(folded) >= _DUPLICATE_MIN_WORDS and found >= len(folded) * _DUPLICATE_RATIO:
            stats.duplicate_lines += 1
            continue
        fresh.append(line)
    return fresh


def fold_homoglyphs(text: str) -> str:
    """
    Строка -> представление, в котором визуально одинаковые буквы разных
    алфавитов неразличимы. Нужно только для сравнения: наружу всегда уходит
    текст из файла, а не свёрнутый.
    """
    return "".join(_HOMOGLYPHS.get(ch, ch) for ch in text.lower() if not ch.isspace())


def _fix_table(
    block: ParsedBlock,
    matched: Sequence[TextLine],
    stats: ReconcileStats,
    vocab: Optional[Dict[str, str]] = None,
    exact: bool = False,
    turn: int = 0,
    table_reader: Optional[Callable[[ParsedBlock], Optional[Dict[str, Any]]]] = None,
) -> Tuple[List[ParsedBlock], List[TextLine]]:
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

    `vocab` — словарь слоя страницы. С ним склеенные строки ячеек режутся
    по словам, а буквы чужого алфавита внутри слова выправляются. Строки
    слоя, слов которых в таблице нет, возвращаются вторым значением —
    текстом; только у распознанного слоя (`exact=False`): у векторного это
    не измерялось.

    `turn` — поворот текста страницы (`page_turn`): колонки и строки
    таблицы ищутся по прямо стоящему листу.
    """
    if latex_text.mangled_cells(block.table_data):
        stats.mangled_tables += 1
        if not table_from_lines.repair_cells(block, _upright_lines(matched, turn)):
            # Строки слоя такой таблицы — обычно каша (альбомная таблица на
            # книжном листе читается поперёк), поэтому сначала модель
            # зрения: её таблица заменяет и разметку, и строки слоя.
            read = table_reader(block) if table_reader is not None else None
            if read:
                block.table_data = read
                # HTML от MinerU — та же разметка, что и в ячейках.
                block.table_html = None
                block.method = METHOD_VLM_TABLE
                stats.vlm_tables += 1
                # Модель могла пропустить строку: строки слоя с настоящими
                # словами, которых в её таблице нет, возвращаются текстом.
                kept = _lines_with_new_words(matched, block)
                stats.table_lines_kept += len(kept)
                return [], kept
            stats.degraded_tables += 1
            return table_from_lines.degrade(block, matched), []
        stats.repaired_tables += 1

    unused = _repair_table_cells(block, matched, stats)
    if vocab is None:
        return [], []
    _split_glued_cells(block, vocab, stats, exact, turn)
    if exact:
        return [], []
    kept = _lines_with_new_words(unused, block)
    stats.table_lines_kept += len(kept)
    return [], kept


def _repair_table_cells(
    block: ParsedBlock, matched: Sequence[TextLine], stats: ReconcileStats
) -> List[TextLine]:
    """
    Значения ячеек заменяются на текст из слоя, если это заведомо та же
    строка. Каждая строка слоя расходуется один раз, поэтому одинаковые
    ячейки не схлопываются в одну. Возвращает строки, не пошедшие в дело.
    """
    table = block.table_data or {}
    pool_lines = [line for line in matched if line.text.strip()]
    pool = [" ".join(line.text.split()) for line in pool_lines]
    if not pool:
        return []

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
    return [line for index, line in enumerate(pool_lines) if index not in used]


# ===========================================================================
# Склеенные строки ячеек
# ===========================================================================

# Буквы, которые распознавание подставляет в русское слово вместо
# кириллицы, по виду («Ha», «Нoмep», «Πo»): латиница и греческий.
_TO_CYRILLIC = dict(zip(
    "aceopxykmACEHKMOPTXYBΠποΟαΑεΕρΡτΤκΚχΧΒΗΜΥ",
    "асеорхукмАСЕНКМОРТХУВПпоОаАеЕрРтТкКхХВНМУ",
))
# Обратное — для латинского слова с кириллицей внутри («exрires»).
_TO_LATIN = {cyr: lat for lat, cyr in _TO_CYRILLIC.items() if "a" <= lat.lower() <= "z"}
_TO_LATIN.update(dict(zip("οΟαΑεΕρΡτΤκΚχΧΒΗΜ", "oOaAeEpPtTkKxXBHM")))
# Транслитерация по звуку: модель MinerU иногда пишет русское слово
# латиницей вперемешку («priёmki» — «приёмки»). Нужна только для сверки со
# словарём слоя, и только у слова, где смешаны алфавиты.
_TRANSLIT = dict(zip("abvgdezijklmnoprstufhcy", "абвгдезийклмнопрстуфхцы"))
_LETTER_RUN_RE = re.compile(r"[^\W\d_]+", re.UNICODE)


def _script(ch: str) -> str:
    if "\u0400" <= ch <= "\u04ff":
        return "cyr"
    if "\u0370" <= ch <= "\u03ff":
        return "grk"
    if ch.isascii() and ch.isalpha():
        return "lat"
    return "other" if ch.isalpha() else ""


def _scripts(text: str) -> set:
    return {_script(ch) for ch in text} - {""}


def _single_script(text: str) -> bool:
    return len(_scripts(text)) <= 1


def unify_script(word: str) -> str:
    """
    Слово из одного алфавита, если смесь в нём объяснима подменой похожих
    букв. Здесь — только кириллица с латиницей: слово с греческой буквой
    не трогается никогда, потому что «τср», «αт», «ρв» — законные
    обозначения ГОСТ, а «kΩ», «μm», «τmax» — единицы и индексы; греческая
    буква становится кириллицей только сверкой со словарём слоя. Есть в
    слове буква, которой нет в латинице («НачальникOTK», «HOCТЬ»), — похожие
    латинские буквы становятся кириллицей; есть буква, которой нет в
    кириллице («exрires»), — наоборот; есть и те и другие («priёmki») —
    слово не трогается: это не подмена, а каша. Латинская буква рядом с
    цифрой не меняется никогда: это код («ТипA1»), а не подмена.
    """
    def fix(match: "re.Match[str]") -> str:
        run = match.group(0)
        if _scripts(run) != {"cyr", "lat"}:
            return run
        cyr_only = any(_script(ch) == "cyr" and ch not in _TO_LATIN for ch in run)
        lat_only = any(_script(ch) == "lat" and ch not in _TO_CYRILLIC for ch in run)
        if cyr_only and lat_only:
            return run
        if not cyr_only and not lat_only:
            cyr = sum(_script(ch) == "cyr" for ch in run)
            to_cyrillic = cyr * 2 >= len(run)
        else:
            to_cyrillic = cyr_only
        table = _TO_CYRILLIC if to_cyrillic else _TO_LATIN
        source = "lat" if to_cyrillic else "cyr"
        return "".join(
            table.get(ch, ch)
            if _script(ch) == source and not _near_digit(word, match.start() + i) else ch
            for i, ch in enumerate(run)
        )

    return _LETTER_RUN_RE.sub(fix, word)


def _near_digit(text: str, index: int) -> bool:
    """Буква стоит вплотную к цифре — часть кода («A1», «M8»)."""
    return (index > 0 and text[index - 1].isdigit()) or (
        index + 1 < len(text) and text[index + 1].isdigit()
    )


def _fold_char(ch: str) -> str:
    """Посимвольная свёртка: длина строки сохраняется, позиции совпадают."""
    low = _TO_CYRILLIC.get(ch, ch) if _script(ch) == "grk" else ch
    low = low.lower()
    folded = _HOMOGLYPHS.get(low, low)
    return folded if len(folded) == 1 else ch


def _fold_word(word: str) -> str:
    return "".join(_fold_char(ch) for ch in word)


def _cursive_fold(word: str) -> str:
    folded = _fold_word(word)
    return "".join(_CURSIVE.get(ch, ch) for ch in folded)


def _unambiguous(words: Sequence[str]) -> Dict[str, str]:
    """Курсивная свёртка -> подпись; свёртки, общие для двух подписей, выкинуты."""
    seen: Dict[str, set] = {}
    for word in words:
        seen.setdefault(_cursive_fold(word), set()).add(word.lower())
    first = {}
    for word in words:
        first.setdefault(_cursive_fold(word), word)
    return {key: first[key] for key, labels in seen.items() if len(labels) == 1}


_STAMP_LEXICON = _unambiguous(_STAMP_LABELS)
_STAMP_EXACT = {
    _fold_word(word): (index, word)
    for index, group in enumerate(_STAMP_GROUPS) for word in group
}


def steel_grade(token: str) -> Optional[str]:
    """
    Марка стали по ГОСТ в кириллице, если токен — она, записанная латиницей.
    Нужны хром или никель (Х/Н): «10AT», «25MBA», «1K5M», «100KT» — коды, а
    не марки. Вызывается только внутри штампа чертежа.
    """
    if not _STEEL_GRADE_RE.match(token):
        return None
    latin = {ch for ch in token if _script(ch) == "lat"}
    if not latin or latin <= {"X"} or not ({"X", "H"} & set(token.upper())):
        return None
    return "".join(_TO_CYRILLIC.get(ch, ch) if _script(ch) == "lat" else ch for ch in token)


def _stamp_label(word: str, vocab: Dict[str, str]) -> Optional[str]:
    """Подпись штампа, которой является слово по курсивной свёртке, или None."""
    lexicon = getattr(vocab, "lexicon", None)
    if not lexicon or sum(ch.isalpha() for ch in word) < 2 or sum(ch.isalnum() for ch in word) < 3:
        return None
    label = lexicon.get(_cursive_fold(word))
    if label is None or _fold_word(word) == _fold_word(label):
        return label
    # Совпадение только по курсиву: слово, которое слой прочитал так же
    # буква в букву, — скорее настоящее слово, чем искажённая подпись.
    if word in getattr(vocab, "verbatim", ()):
        return None
    return label


def _match_case(spelling: str, original: str) -> str:
    """Написание словаря с регистром исходного слова (длины равны)."""
    if len(spelling) != len(original):
        return spelling
    return "".join(new.upper() if old.isupper() else new.lower()
                   for old, new in zip(original, spelling))


def _with_stamp_lexicon(vocab: Dict[str, str]) -> Dict[str, str]:
    """Словарь страницы плюс подписи штампа по курсивной свёртке."""
    merged = _Vocabulary()
    merged.update(vocab)
    merged.verbatim = set(getattr(vocab, "verbatim", ()))
    merged.lexicon = _STAMP_LEXICON
    return merged


def _looks_like_stamp(cells: Sequence[str], bbox: Sequence[float], turn: int = 0) -> bool:
    """
    Таблица — основная надпись чертежа (см. `_STAMP_MIN_LABELS`). «Внизу
    листа» — на прямо стоящем листе: у альбомного листа, отсканированного
    книжным, штамп лежит у края исходного листа сбоку или вверху.
    """
    if not bbox or upright_bbox(bbox, turn)[3] < _STAMP_MIN_BOTTOM:
        return False
    found = {}
    for cell in cells:
        for raw in re.split(r"[\s.]+", str(cell)):
            hit = _STAMP_EXACT.get(_fold_word(raw.strip(_WORD_EDGE)))
            if hit is not None:
                found[hit[1]] = hit[0]
    return (
        len(found) >= _STAMP_MIN_LABELS
        and len(set(found.values())) >= _STAMP_MIN_GROUPS
        and any(label in found for label in _STAMP_SIGNATURES)
    )


class _Vocabulary(dict):
    """Свёрнутое написание -> написание слоя; `verbatim` — слова как в слое."""

    def __init__(self) -> None:
        super().__init__()
        self.verbatim: set = set()
        #: Закрытый словарь по курсивной свёртке (подписи штампа) или None.
        self.lexicon: Optional[Dict[str, str]] = None


def layer_vocabulary(lines: Sequence[TextLine]) -> Dict[str, str]:
    """
    Слова строк слоя: свёрнутое написание -> написание из слоя (без знаков
    по краям). Из нескольких написаний берётся слово одного алфавита.
    """
    vocab = _Vocabulary()
    for line in lines:
        for raw in (line.text or "").split():
            word = raw.strip(_WORD_EDGE)
            if not word:
                continue
            vocab.verbatim.add(word)
            key = _fold_word(word)
            if key not in vocab or (not _single_script(vocab[key]) and _single_script(word)):
                vocab[key] = word
    return vocab


def _lookup(piece: str, vocab: Dict[str, str]) -> Optional[str]:
    """Написание куска по словарю слоя или None."""
    found = vocab.get(_fold_word(piece))
    if found is not None:
        return found
    scripts = _scripts(piece)
    if "cyr" in scripts and "lat" in scripts:
        spelled = "".join(_TRANSLIT.get(ch.lower(), ch) if _script(ch) == "lat" else ch
                          for ch in piece)
        found = vocab.get(_fold_word(spelled))
        if found is not None:
            return found
    lexicon = getattr(vocab, "lexicon", None)
    if lexicon and sum(ch.isalpha() for ch in piece) >= 2 and sum(ch.isalnum() for ch in piece) >= 3:
        return lexicon.get(_cursive_fold(piece))
    return None


def _respell(
    piece: str, spelling: str, trusted: bool, token: str = "", offset: int = 0
) -> str:
    """
    Кусок в написании слоя, с регистром куска. Только если слой надёжен
    (`trusted`: векторный слой или смешанный алфавит в самом токене) и
    написание слоя из одного алфавита.
    """
    if not trusted or spelling == piece or len(spelling) != len(piece):
        return piece
    if not _single_script(spelling):
        return piece
    # Меняются только буквы на буквы: цифра не становится буквой («3» и «з»
    # сворачиваются одинаково), а латинская буква у цифры — часть кода.
    context = token or piece
    out = []
    for i, (old, new) in enumerate(zip(piece, spelling)):
        keep = not (old.isalpha() and new.isalpha()) or (
            _script(old) == "lat" and _near_digit(context, offset + i)
        )
        out.append(old if keep else (new.upper() if old.isupper() else new.lower()))
    return "".join(out)


def _letter_run(text: str, from_end: bool) -> int:
    """Длина серии букв у края строки."""
    chars = reversed(text) if from_end else iter(text)
    run = 0
    for ch in chars:
        if not ch.isalpha():
            break
        run += 1
    return run


def _split_token(
    token: str, vocab: Dict[str, str]
) -> Optional[List[Tuple[int, int, str]]]:
    """
    Куски склеенного токена — (начало, конец, написание по слою) — или None.

    Токен должен целиком разложиться на слова словаря; из разложений берётся
    то, где кусков меньше. Кусок короче трёх букв допустим, только если это
    служебное слово, сокращение с точкой или число. Граница ставится:
    - в чистой склейке без знаков: между буквами, а между буквами и цифрами —
      только если с буквенной стороны слово не короче трёх букв («ГОСТ7798»,
      «Сталь45»; не «М12х50», «Ст3сп»);
    - после точки, запятой и подобных знаков — только перед словом из букв
      («См.ТАДУ»). Знак перед цифрами, «/» и дефис означают шифр или перенос:
      «ТАДУ.301611.002» и «мо-дуле» не режутся. Между двумя цифрами граница
      не ставится никогда.
    """
    if len(token) > _GLUED_MAX_LEN or sum(ch.isalnum() for ch in token) < _GLUED_MIN_ALNUM:
        return None
    folded = _fold_word(token)
    if folded.strip(_WORD_EDGE) in vocab:
        return None
    size = len(folded)
    spellings: Dict[Tuple[int, int], str] = {}
    short_words = {_fold_word(w) for w in _SHORT_WORDS}
    # best[stop] — (кусков, начало последнего куска, конец предыдущего или
    # None) для лучшего разбора folded[:stop].
    best: Dict[int, Tuple[int, int, Optional[int]]] = {}

    def piece_ok(start: int, stop: int) -> bool:
        piece = folded[start:stop]
        original = token[start:stop]
        # Число и марка стали — куски и без словаря: слово к ним приклеено
        # («Крышка2», «Лист12X18H10T»), а сами они в слое бывают искажены.
        stamp = bool(getattr(vocab, "lexicon", None))
        spelling = (_lookup(original, vocab) or (stamp and steel_grade(original))
                    or (original if _NUMBER_RE.match(original) else None))
        if spelling is None:
            return False
        spellings[(start, stop)] = spelling
        letters = sum(ch.isalpha() for ch in token[start:stop])
        # Сокращение с точкой («См.», «Рис.») — тоже слово.
        abbreviation = token[stop:stop + 1] == "."
        return letters == 0 or letters >= 3 or piece in short_words or abbreviation

    def cut_ok(prev_end: int, start: int, stop: int) -> bool:
        gap = token[prev_end:start]
        left, right = token[prev_end - 1], token[start]
        word = token[start:stop]
        if gap:
            return all(ch in _CUT_AFTER for ch in gap) and _letter_run(word, from_end=False) >= 3
        if left.isdigit() and right.isdigit():
            return False
        if left.isalpha() and right.isdigit():
            return _letter_run(token[:prev_end], from_end=True) >= 3
        if left.isdigit() and right.isalpha():
            return _letter_run(word, from_end=False) >= 3
        return True

    def extend(start: int, pieces: int, prev_end: Optional[int]) -> None:
        for stop in range(start + 1, size + 1):
            if not folded[stop - 1].isalnum() or not piece_ok(start, stop):
                continue
            if prev_end is not None and not cut_ok(prev_end, start, stop):
                continue
            if stop not in best or best[stop][0] > pieces + 1:
                best[stop] = (pieces + 1, start, prev_end)

    lead = 0
    while lead < size and not folded[lead].isalnum():
        lead += 1
    extend(lead, 0, None)
    # Кусок кончается правее, чем начинается, поэтому к моменту обхода
    # позиции `end` лучший разбор до неё уже окончателен.
    for end in range(1, size):
        if end not in best:
            continue
        gap = end
        while gap < size and not folded[gap].isalnum():
            gap += 1
        if gap < size:
            extend(gap, best[end][0], end)

    finishes = [end for end in best if not any(ch.isalnum() for ch in folded[end:])]
    if not finishes:
        return None
    end = min(finishes, key=lambda e: best[e][0])
    if best[end][0] < 2:
        return None
    pieces: List[Tuple[int, int, str]] = []
    while True:
        _, start, prev_end = best[end]
        pieces.append((start, end, spellings[(start, end)]))
        if prev_end is None:
            break
        end = prev_end
    return pieces[::-1]


def _fix_token(token: str, vocab: Dict[str, str], exact: bool) -> str:
    """
    Токен ячейки: алфавит выправлен, склейка разрезана по словам слоя, а
    куски и целые слова взяты в написании слоя, если слою можно верить.
    """
    core = token.strip(_WORD_EDGE)
    label = _stamp_label(core, vocab)
    if label is not None:
        # Подпись штампа — из закрытого словаря целиком: цифра в ней («И3м.»)
        # и латиница («Macca») — заведомо ошибка чтения, а не код.
        start = token.index(core)
        return token[:start] + _match_case(label, core) + token[start + len(core):]
    # Слово, которое в слое стоит буква в букву, — уже правильное; токен без
    # кириллицы с греческим («kΩ», «τmax») — законная запись, не подмена.
    if core in getattr(vocab, "verbatim", ()):
        return token
    if "grk" in _scripts(token) and "cyr" not in _scripts(token):
        return token
    token = unify_script(token)
    trusted = exact or len(_scripts(token)) > 1
    pieces = _split_token(token, vocab)
    if pieces:
        # Кусок из одних похожих латинских букв внутри русского токена
        # («Haстадии») — русское слово, даже если слой его так не написал.
        russian = "cyr" in _scripts(token)
        out = [token[:pieces[0][0]]]
        for index, (start, stop, spelling) in enumerate(pieces):
            if index:
                out.append(" ")
            original = token[start:stop]
            label = _stamp_label(original, vocab)
            stamp = bool(getattr(vocab, "lexicon", None))
            piece = ((stamp and steel_grade(original))
                     or (label and _match_case(label, original))
                     or _respell(original, spelling, trusted, token, start))
            if russian and all(_script(ch) == "lat" and ch in _TO_CYRILLIC
                               for ch in piece if ch.isalpha()):
                piece = "".join(
                    ch if _near_digit(token, start + i) else _TO_CYRILLIC.get(ch, ch)
                    for i, ch in enumerate(piece)
                )
            out.append(piece)
            following = pieces[index + 1][0] if index + 1 < len(pieces) else len(token)
            out.append(token[stop:following])
        return "".join(out)
    core = token.strip(_WORD_EDGE)
    if not core or not any(ch.isalpha() for ch in core):
        return token
    start = token.index(core)
    grade = steel_grade(core) if getattr(vocab, "lexicon", None) else None
    if grade is not None:
        return token[:start] + grade + token[start + len(core):]
    spelling = _lookup(core, vocab)
    if spelling is None:
        # Только в штампе: в открытом тексте частичная склейка не режется.
        cut = None
        if getattr(vocab, "lexicon", None):
            cut = _known_prefix_cut(core, vocab)
            glued = _WORD_NUMBER_RE.match(core)
            if cut is None and glued and _scripts(glued.group(1)) == {"cyr"}:
                cut = len(glued.group(1))
        if cut is None:
            return token
        return token[:start] + core[:cut] + " " + core[cut:] + token[start + len(core):]
    fixed = _respell(core, spelling, trusted, token, start)
    return token[:start] + fixed + token[start + len(core):]


def _known_prefix_cut(core: str, vocab: Dict[str, str]) -> Optional[int]:
    """
    Склейка двух слов в штампе чертежа, из которых в слое есть только одно
    («усилитель» + «короткий» — второе OCR прочитал кашей; наименование
    изделия в штампе — короткая фраза из отдельных слов). Режется, только если известное
    слово длинное (`_KNOWN_PART_MIN`), остаток — тоже длинное слово из одних
    букв того же алфавита, и весь токен — одно кириллическое слово: короткие
    основы («тепло», «электро») в составных словах так не отрезаются.
    """
    if not core.isalpha() or _scripts(core) != {"cyr"}:
        return None
    size = len(core)
    for cut in range(size - _UNKNOWN_PART_MIN, _KNOWN_PART_MIN - 1, -1):
        if _fold_word(core[:cut]) in vocab:
            return cut
    for cut in range(_UNKNOWN_PART_MIN, size - _KNOWN_PART_MIN + 1):
        if _fold_word(core[cut:]) in vocab:
            return cut
    return None


def split_glued_text(
    text: str, vocab: Dict[str, str], exact: bool = False
) -> Tuple[str, int]:
    """
    Текст (или HTML таблицы) с разрезанными склейками и выправленным
    алфавитом; второе значение — число изменённых токенов. Разметка не
    трогается: правится только текст между тегами. `exact` — словарь из
    векторного слоя: его написанию верим всегда, а написанию OCR — только
    в токене, где алфавиты смешаны.
    """
    if not text:
        return text, 0
    vocab = vocab or {}
    count = 0

    def fix(match: "re.Match[str]") -> str:
        nonlocal count
        token = match.group(0)
        fixed = _fix_token(token, vocab, exact)
        if fixed != token:
            count += 1
        return fixed

    parts = _MARKUP_RE.split(text)
    for index in range(0, len(parts), 2):
        parts[index] = re.sub(r"\S+", fix, parts[index])
    return "".join(parts), count


def _split_glued_cells(
    block: ParsedBlock, vocab: Dict[str, str], stats: ReconcileStats, exact: bool = False,
    turn: int = 0,
) -> None:
    """Склейки режутся во всех видах таблицы: ячейки, HTML и текст блока."""
    table = block.table_data or {}
    cells = list(table.get("headers") or []) + [c for row in table.get("rows") or [] for c in row]
    if not exact and _looks_like_stamp(cells, block.bbox, turn):
        vocab = _with_stamp_lexicon(vocab)

    def fix(value: Any) -> Any:
        if not isinstance(value, str):
            return value
        fixed, count = split_glued_text(value, vocab, exact)
        stats.split_tokens += count
        return fixed

    if table:
        table["headers"] = [fix(h) for h in (table.get("headers") or [])]
        table["rows"] = [[fix(c) for c in row] for row in (table.get("rows") or [])]
        block.table_data = table
    if block.table_html:
        block.table_html = split_glued_text(block.table_html, vocab, exact)[0]
    if block.text:
        block.text = split_glued_text(block.text, vocab, exact)[0]


def _looks_like_word(word: str) -> bool:
    """
    Слово, а не обрывок линий сетки: длиннее трёх знаков, почти одни буквы,
    есть гласная и не одна и та же буква подряд («ООО», «ВКЛМ», «||||»).
    """
    if len(word) < 4:
        return False
    letters = [ch for ch in word.lower() if ch.isalpha()]
    if len(letters) < len(word) * 0.8 or len(set(letters)) < 2:
        return False
    return any(ch in _VOWELS for ch in letters)


def _lines_with_new_words(lines: Sequence[TextLine], block: ParsedBlock) -> List[TextLine]:
    """
    Строки слоя из рамки таблицы, чьих слов в ячейках нет.

    Модель таблиц теряет подписи столбцов и куски ячеек целиком; слой их
    читает, но раньше такие строки молча выбрасывались вместе с рамкой.
    Повтором считается слово, которое уже есть в таблице, — строка из
    повторов не возвращается, чтобы содержимое не двоилось.
    """
    table = block.table_data or {}
    cells = [str(h) for h in (table.get("headers") or [])]
    for row in table.get("rows") or []:
        cells.extend(str(c) for c in row)
    known = {_fold_word(w) for w in _TOKEN_RE.findall(" ".join(cells))}
    kept: List[TextLine] = []
    for line in lines:
        words = [w for w in _TOKEN_RE.findall(line.text or "") if len(w) >= 3]
        # Новым считается только настоящее слово: обрывки линий сетки
        # Tesseract читает как «ООО», «ОИ», «ne».
        new = [_fold_word(w) for w in words
               if _fold_word(w) not in known and _looks_like_word(w)]
        if len(new) >= _TABLE_LINE_MIN_NEW_WORDS and len(new) >= len(words) * _TABLE_LINE_MIN_NEW_SHARE:
            kept.append(line)
            known.update(new)
    return kept


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


def _block_reading_key(block: ParsedBlock, turn: int = 0) -> Tuple[float, float]:
    x1, y1, _x2, _y2 = upright_bbox(block.bbox, turn)
    return (round(y1, 3), round(x1, 3))
