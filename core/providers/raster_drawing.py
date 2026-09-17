"""
Растровые чертежи: опознание листа по геометрии картинки и разбор полей по OCR.

Векторный лист разбирается по текстовому слою (`vector_drawing`) — там
распознавать нечего. Со сканом и фотографией так нельзя: текста в файле нет,
а решать, чертёж перед нами или страница текста, всё равно надо — до всякого
ML и до тяжёлого конвейера. Раньше этого решения не было вовсе: любая
картинка получала жанр `unknown`, ветка чертежей на неё не включалась, а на
приёме чертёж отличался от фотографии по «доле белого и плотности контуров»
на первых 64 КБ файла.

Признаки здесь геометрические, а не смысловые: рамка формата по краям листа,
длинные прямые линии (рамка, выносные, размерные, линии таблиц штампа),
редкая тёмная графика на белом фоне. У страницы текста длинных линий нет,
зато есть регулярные полосы строк; у фотографии — цвет и тёмный фон.

Разрешение — не придирка, а условие применимости: на картинке 80x50 не видно
ни рамки, ни строк, и любое суждение о её содержании будет выдумкой. Такой
файл честно помечается `unreadable` и уходит человеку, а не назначается
чертежом с уверенностью 0.8.

Разбор надписей отсюда убран вместе со сплошным распознаванием листа: поля
чертежа собирает `core.drawing_processor` из ответа внешней модели. Здесь
осталось опознание рода растра, которым пользуются приём и лестница.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field
from typing import Any, Dict

logger = logging.getLogger(__name__)

# Род растра.
KIND_DRAWING = "drawing"        # чертёж: рамка формата и длинные линии
KIND_SCHEME = "scheme"          # схема: графика есть, рамки формата нет
KIND_TEXT_SCAN = "text_scan"    # скан страницы текста
KIND_SCREENSHOT = "screenshot"  # снимок экрана: тёмный или ровный фон без графики
KIND_PHOTO = "photo"            # фотография: цвет
KIND_UNREADABLE = "unreadable"  # судить не о чем: мал размер, пусто, не открылось

# Ниже этого разрешения признаки листа неразличимы физически.
MIN_SIDE = 320
MIN_PIXELS = 150_000
# Рабочий размер анализа: больше не нужно, меньше — теряются тонкие линии.
WORK_SIDE = 1000

# Пороги признаков. Вынесены наверх, потому что их придётся подкручивать под
# конкретный корпус, и лучше делать это в одном месте.
_COLOUR_PHOTO = 0.18        # разброс каналов: выше — цветная съёмка
_COLOUR_GREY = 0.06         # ниже — практически монохром
_WHITE_DOCUMENT = 0.45      # доля светлых пикселей у документа
_INK_DRAWING = 0.25         # чертёж — редкая графика на белом
_INK_BLANK = 0.002          # ниже — лист пустой
_RULE_COVERAGE = 0.5        # линия считается длинной с этой доли стороны
_FRAME_COVERAGE = 0.7       # сторона рамки формата почти во всю сторону листа
_FRAME_MARGIN = 0.06        # рамка ищется в этой полосе у края
_TEXT_ROW_MIN = 0.005       # плотность строки текста: от
_TEXT_ROW_MAX = 0.60        #                          до
_TEXT_BANDS_PAGE = 6        # столько полос строк — это уже страница текста
_TEXT_PAGE_HARD = 18        # столько — страница текста даже в рамке


@dataclass
class RasterVerdict:
    """Что за лист перед нами и насколько это надёжно."""

    kind: str
    confidence: float
    reason: str = ""
    signals: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_drawing(self) -> bool:
        """Чертёж и схема одинаково требуют разбора графики, а не текста."""
        return self.kind in (KIND_DRAWING, KIND_SCHEME)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "confidence": round(self.confidence, 3),
            "reason": self.reason,
            **self.signals,
        }


# ===========================================================================
# Опознание
# ===========================================================================

def analyse(data: bytes) -> RasterVerdict:
    """Классифицирует растр по геометрии. Байты — весь файл, а не образец."""
    image = open_image(data)
    if image is None:
        return RasterVerdict(
            KIND_UNREADABLE, 0.0, "изображение не открылось", {}
        )
    return analyse_image(image)


def analyse_image(image) -> RasterVerdict:
    """То же, но по уже открытой картинке — чтобы не декодировать дважды."""
    try:
        import numpy
    except ImportError:  # pragma: no cover — numpy есть в рантайме
        return RasterVerdict(KIND_UNREADABLE, 0.0, "numpy недоступен", {})

    width, height = image.size
    signals: Dict[str, Any] = {
        "width": width, "height": height,
        "megapixels": round(width * height / 1_000_000, 3),
    }

    if min(width, height) < MIN_SIDE or width * height < MIN_PIXELS:
        return RasterVerdict(
            KIND_UNREADABLE, 0.3,
            f"разрешение {width}x{height} слишком мало: ни рамки, ни строк "
            f"на нём не различить, решение о содержании было бы выдумкой",
            signals,
        )

    work = _downscale(image, WORK_SIDE)
    rgb = numpy.asarray(work.convert("RGB"), dtype=numpy.float32)
    grey = rgb.mean(axis=2)

    colourfulness = float(rgb.std(axis=2).mean() / 128.0)
    whiteness = float((grey > 200).mean())
    ink_mask = grey < _threshold(grey, numpy)
    ink = float(ink_mask.mean())

    rows = ink_mask.sum(axis=1)
    columns = ink_mask.sum(axis=0)
    horizontal = _rule_bands(rows, ink_mask.shape[1])
    vertical = _rule_bands(columns, ink_mask.shape[0])
    rules = horizontal + vertical
    frame = _has_frame(ink_mask)

    # Полосы строк считаются без вертикальных линеек: боковая сторона рамки
    # добавляет тёмное в каждую строку, и без этой поправки лист текста в
    # рамке выглядит как одна сплошная полоса, то есть как чертёж.
    long_columns = columns >= _RULE_COVERAGE * ink_mask.shape[0]
    body = ink_mask[:, ~long_columns]
    text_bands = _text_bands(body.sum(axis=1), body.shape[1])

    signals.update({
        "colourfulness": round(colourfulness, 3),
        "whiteness": round(whiteness, 3),
        "ink": round(ink, 4),
        "long_rules": rules,
        "frame": frame,
        "text_bands": text_bands,
    })

    if colourfulness > _COLOUR_PHOTO:
        return RasterVerdict(
            KIND_PHOTO, 0.75, "цветовой разброс как у съёмки, а не у листа",
            signals,
        )

    if whiteness < _WHITE_DOCUMENT:
        kind = KIND_SCREENSHOT if colourfulness < _COLOUR_GREY else KIND_PHOTO
        return RasterVerdict(
            kind, 0.6, "фон тёмный: на лист документа не похоже", signals
        )

    if ink < _INK_BLANK:
        return RasterVerdict(
            KIND_UNREADABLE, 0.35, "лист практически пуст", signals
        )

    # Рамка формата — самый сильный признак листа. Но лист текста в рамке
    # остаётся листом текста, поэтому полос строк должно быть немного.
    if frame and rules >= 3 and ink <= _INK_DRAWING and text_bands <= _TEXT_PAGE_HARD:
        confidence = 0.85 if rules >= 5 else 0.75
        return RasterVerdict(
            KIND_DRAWING, confidence,
            f"рамка формата и {rules} длинных линий при плотности графики "
            f"{ink:.1%}",
            signals,
        )

    # Сетка таблицы тоже даёт длинные линии, и по одному их числу ведомость
    # неотличима от чертежа. Отличает соотношение: у ведомости строк текста
    # столько же, сколько линеек, у чертежа графика преобладает над текстом.
    if text_bands >= _TEXT_BANDS_PAGE and 2 * text_bands >= rules:
        return RasterVerdict(
            KIND_TEXT_SCAN, 0.75,
            f"{text_bands} полос строк при {rules} длинных линиях — страница "
            f"текста или ведомость, а не чертёж",
            signals,
        )

    if rules >= 4 and ink <= _INK_DRAWING:
        return RasterVerdict(
            KIND_DRAWING, 0.7,
            f"{rules} длинных линий без рамки формата — обрезанный лист или "
            f"фрагмент чертежа",
            signals,
        )

    if text_bands >= _TEXT_BANDS_PAGE:
        return RasterVerdict(
            KIND_TEXT_SCAN, 0.75,
            f"{text_bands} полос строк и {rules} длинных линий — страница "
            f"текста, а не чертёж",
            signals,
        )

    if rules >= 1:
        return RasterVerdict(
            KIND_SCHEME, 0.6,
            "графика есть, рамки формата и строк текста нет", signals
        )

    return RasterVerdict(
        KIND_UNREADABLE, 0.4,
        "ни рамки, ни длинных линий, ни строк — род листа не определён",
        signals,
    )


# ===========================================================================
# Помощники
# ===========================================================================

def open_image(data: bytes):
    """Целая картинка из байтов. `None` — не открылась (обрезок, не растр)."""
    try:
        from PIL import Image
    except ImportError:  # pragma: no cover
        return None
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
        return image
    except Exception as exc:  # noqa: BLE001 — битый или обрезанный файл
        logger.debug("Растр не открылся: %s", exc)
        return None


def _downscale(image, side: int):
    if max(image.size) <= side:
        return image
    copy = image.copy()
    copy.thumbnail((side, side))
    return copy


def _threshold(grey, numpy) -> float:
    """
    Порог Оцу: разделяет фон и графику там, где лист сер от съёмки. Ровно
    128 годится только для идеального скана.
    """
    histogram, edges = numpy.histogram(grey, bins=64, range=(0.0, 255.0))
    total = histogram.sum()
    if total == 0:
        return 128.0
    centres = (edges[:-1] + edges[1:]) / 2.0
    weight_background = numpy.cumsum(histogram)
    weight_foreground = total - weight_background
    valid = (weight_background > 0) & (weight_foreground > 0)
    if not valid.any():
        return 128.0
    sums = numpy.cumsum(histogram * centres)
    total_sum = sums[-1]
    mean_background = numpy.where(weight_background > 0, sums / numpy.maximum(weight_background, 1), 0)
    mean_foreground = numpy.where(
        weight_foreground > 0, (total_sum - sums) / numpy.maximum(weight_foreground, 1), 0
    )
    variance = weight_background * weight_foreground * (mean_background - mean_foreground) ** 2
    variance[~valid] = -1
    return float(centres[int(numpy.argmax(variance))])


def _bands(flags) -> int:
    """Число непрерывных участков в булевом профиле."""
    count, previous = 0, False
    for value in flags:
        if value and not previous:
            count += 1
        previous = bool(value)
    return count


def _rule_bands(profile, length: int) -> int:
    """Полосы, в которых линия тянется больше чем на половину стороны."""
    if length <= 0:
        return 0
    return _bands([value >= _RULE_COVERAGE * length for value in profile])


def _text_bands(rows, width: int) -> int:
    """Полосы строк: тёмного немного, но и не ноль, и они разделены пробелами."""
    if width <= 0:
        return 0
    return _bands([
        _TEXT_ROW_MIN * width <= value <= _TEXT_ROW_MAX * width for value in rows
    ])


def _has_frame(ink_mask) -> bool:
    """
    Рамка формата: тёмная линия почти во всю сторону в узкой полосе у края.
    Хватает трёх сторон из четырёх — сканы часто обрезаны с одного края.
    """
    height, width = ink_mask.shape
    margin_y = max(1, int(height * _FRAME_MARGIN))
    margin_x = max(1, int(width * _FRAME_MARGIN))

    rows = ink_mask.sum(axis=1)
    columns = ink_mask.sum(axis=0)
    needed_h = _FRAME_COVERAGE * width
    needed_v = _FRAME_COVERAGE * height

    sides = [
        bool((rows[:margin_y] >= needed_h).any()),
        bool((rows[height - margin_y:] >= needed_h).any()),
        bool((columns[:margin_x] >= needed_v).any()),
        bool((columns[width - margin_x:] >= needed_v).any()),
    ]
    return sum(sides) >= 3
