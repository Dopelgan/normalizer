"""
Подготовка растровой страницы к распознаванию.

Без неё Tesseract теряет на сканах больше всего, и потери не случайные:

* повёрнутый лист (альбомная страница, скан вверх ногами) не читается
  вовсе — ноль строк, и документ уходит в карантин «нечитаем»;
* таблица с сеткой пропадает целиком: линии ячеек сливаются с цифрами, и
  именно числа протоколов, спецификаций и актов в индекс не доезжают;
* зерно и пыль скана вдвое снижают долю прочитанных реквизитов.

На синтетическом протоколе испытаний (тест `test_ocr_prepare.py`) полнота
ключевых реквизитов поднимается с 0–67 % до 80–93 %.

Всё сделано на Pillow и numpy, без новых зависимостей образа. Геометрия
страницы меняется только поворотом, и он обратим: `PreparedPage.to_source`
переводит рамку строки обратно в доли исходного листа — по ним OCR-слой
сшивается с блоками MinerU.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageFilter, ImageOps

from core.config import settings

logger = logging.getLogger(__name__)

# Во сколько раз чётче должны стать строки, чтобы лист стоило поворачивать.
_SKEW_MIN_GAIN = 0.05


@dataclass
class PreparedPage:
    """Подготовленный лист и цепочка поворотов, которой он получен."""

    image: Image.Image
    source_size: Tuple[int, int]
    # Повороты в порядке применения: (угол против часовой, размер до, размер после).
    rotations: List[Tuple[float, Tuple[int, int], Tuple[int, int]]] = field(default_factory=list)
    orientation: int = 0
    skew: float = 0.0
    removed_line_pixels: int = 0
    speckle: float = 0.0

    def to_source(self, bbox: Sequence[float]) -> List[float]:
        """Рамка в долях подготовленного листа → доли исходного."""
        if not self.rotations:
            return [float(v) for v in bbox]
        w, h = self.image.size
        x1, y1, x2, y2 = bbox[0] * w, bbox[1] * h, bbox[2] * w, bbox[3] * h
        points = [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]
        for angle, before, after in reversed(self.rotations):
            points = [_unrotate(p, angle, before, after) for p in points]
        sw, sh = self.source_size
        xs = [min(max(p[0] / sw, 0.0), 1.0) for p in points]
        ys = [min(max(p[1] / sh, 0.0), 1.0) for p in points]
        return [min(xs), min(ys), max(xs), max(ys)]


def _unrotate(point, angle, before, after):
    """
    Точка листа после `Image.rotate(angle, expand=True)` → точка до поворота.
    Pillow вращает против часовой стрелки вокруг центра; ось y смотрит вниз.
    """
    theta = math.radians(angle)
    nx, ny = point[0] - after[0] / 2.0, point[1] - after[1] / 2.0
    ox = nx * math.cos(theta) - ny * math.sin(theta)
    oy = nx * math.sin(theta) + ny * math.cos(theta)
    return ox + before[0] / 2.0, oy + before[1] / 2.0


def prepare_page(image: Image.Image, upscaled: bool = False) -> PreparedPage:
    """
    Повернуть, выровнять, очистить лист и снять линии таблиц перед OCR.

    На выходе — оттенки серого, а не чёрно-белый лист: жёсткая бинаризация
    вдвое замедляла Tesseract на плотных страницах. Линии сетки
    закрашиваются белым по маске. `upscaled` — лист увеличен перед OCR
    (`_fit`): штрихи букв на нём толстые и длинные, линии не снимаются.
    """
    gray = ImageOps.grayscale(image)
    page = PreparedPage(image=gray, source_size=gray.size)

    orientation = _orientation(gray)
    if orientation:
        _rotate(page, -orientation)
        page.orientation = orientation

    skew = _skew_angle(page.image, settings.OCR_DESKEW_LIMIT)
    if abs(skew) >= 0.25:
        _rotate(page, skew)
        page.skew = skew

    cleaned = page.image
    # Медианный фильтр убирает зерно скана, но стирает и тонкие штрихи
    # мелкого шрифта: на чистой плотной странице он обнулял распознавание.
    # Поэтому он включается только там, где зерно действительно есть.
    page.speckle = _speckle_share(cleaned)
    if page.speckle >= settings.OCR_DESPECKLE_MIN_SHARE:
        cleaned = cleaned.filter(ImageFilter.MedianFilter(3))
    if settings.OCR_REMOVE_TABLE_LINES and not upscaled:
        pixels = np.asarray(cleaned)
        lines = _line_mask(pixels <= _otsu(pixels))
        page.removed_line_pixels = int(lines.sum())
        if page.removed_line_pixels:
            pixels = pixels.copy()
            pixels[lines] = 255
            cleaned = Image.fromarray(pixels)
    page.image = cleaned
    return page


def _rotate(page: PreparedPage, angle: float) -> None:
    before = page.image.size
    page.image = page.image.rotate(
        angle, expand=True, fillcolor=255, resample=Image.BICUBIC
    )
    page.rotations.append((angle, before, page.image.size))


def _orientation(gray: Image.Image) -> int:
    """
    Поворот листа кратно 90°, по которому Tesseract OSD узнаёт текст.

    На разреженных и зашумлённых листах OSD угадывает верно, но с низкой
    уверенностью (1–2,5 на протоколах испытаний), а на листах с малым
    количеством текста может ошибиться. Поэтому уверенный ответ принимается
    сразу, а неуверенный — только если повёрнутый лист читается заметно
    лучше исходного. Ошибка OSD — «не поворачивать».
    """
    found = osd_rotation(gray)
    if found is None:
        return 0
    angle, confidence = found
    if not angle:
        return 0
    if confidence >= settings.OCR_OSD_MIN_CONFIDENCE:
        return angle
    return angle if _reads_better(gray, angle) else 0


def osd_rotation(gray: Image.Image) -> Optional[Tuple[int, float]]:
    """
    Ответ Tesseract OSD: на сколько градусов по часовой повернуть лист, чтобы
    текст встал прямо, и уверенность в этом. `None` — OSD недоступен или
    ответа нет; это не то же самое, что «поворачивать не нужно».
    """
    try:
        import pytesseract

        osd = pytesseract.image_to_osd(gray, config="--psm 0 -c min_characters_to_try=5")
    except Exception as exc:  # noqa: BLE001 — OSD не обязателен
        logger.debug("Ориентация листа не определена: %s", exc)
        return None
    rotate = re.search(r"Rotate:\s*(\d+)", osd)
    confidence = re.search(r"Orientation confidence:\s*([\d.]+)", osd)
    if not rotate or not confidence:
        return None
    return int(rotate.group(1)) % 360, float(confidence.group(1))


def _reads_better(gray: Image.Image, angle: int) -> bool:
    """Повёрнутый на `angle` лист даёт заметно больше уверенно прочитанных слов."""
    small = gray.copy()
    small.thumbnail((1400, 1400))
    small = small.filter(ImageFilter.MedianFilter(3))
    before = _reading_score(small)
    after = _reading_score(small.rotate(-angle, expand=True, fillcolor=255))
    return after > max(before * 1.5, 20.0)


def _reading_score(image: Image.Image) -> float:
    """Сумма длин уверенно прочитанных слов — грубая мера читаемости листа."""
    try:
        import pytesseract

        data = pytesseract.image_to_data(
            image, lang="rus+eng", config="--psm 6", output_type=pytesseract.Output.DICT
        )
    except Exception:  # noqa: BLE001
        return 0.0
    score = 0.0
    for word, conf in zip(data.get("text") or [], data.get("conf") or []):
        try:
            conf = float(conf)
        except (TypeError, ValueError):
            continue
        word = (word or "").strip()
        if conf >= 60 and len(word) >= 3:
            score += len(word)
    return score


def _speckle_share(gray: Image.Image) -> float:
    """Доля одиночных чёрных пикселей (без чёрных соседей) среди всех чёрных."""
    pixels = np.asarray(gray)
    black = pixels <= _otsu(pixels)
    total = int(black.sum())
    if not total:
        return 0.0
    padded = np.pad(black, 1)
    neighbours = np.zeros(black.shape, dtype=np.uint8)
    rows, cols = black.shape
    for dy in (0, 1, 2):
        for dx in (0, 1, 2):
            if dy == 1 and dx == 1:
                continue
            neighbours += padded[dy:dy + rows, dx:dx + cols]
    return float((black & (neighbours == 0)).sum()) / total


def _otsu(pixels: np.ndarray) -> int:
    hist = np.bincount(pixels.ravel(), minlength=256).astype(float)
    total = hist.sum()
    if not total:
        return 128
    p = hist / total
    w = np.cumsum(p)
    m = np.cumsum(p * np.arange(256))
    with np.errstate(divide="ignore", invalid="ignore"):
        between = (m[-1] * w - m) ** 2 / (w * (1.0 - w))
    between = np.nan_to_num(between, nan=-1.0, posinf=-1.0)
    return int(np.argmax(between))


def _skew_angle(gray: Image.Image, limit: float) -> float:
    """
    Наклон строк методом проекций: при верном угле строки дают резкие пики
    в сумме по рядам. Сначала грубый шаг, потом уточнение около лучшего.
    """
    if limit <= 0:
        return 0.0
    small = gray.copy()
    small.thumbnail((1000, 1000))
    pixels = np.asarray(small)
    mask = Image.fromarray(((pixels <= _otsu(pixels)) * 255).astype(np.uint8))

    def score(angle: float) -> float:
        rotated = np.asarray(mask.rotate(angle, fillcolor=0)) > 0
        return float(np.var(rotated.sum(axis=1)))

    coarse = np.arange(-limit, limit + 1e-9, 1.0)
    best = max(coarse, key=score)
    fine = np.arange(max(-limit, best - 1.0), min(limit, best + 1.0) + 1e-9, 0.25)
    angle = float(max(fine, key=score))
    # Поворачивать только при заметном выигрыше: на пустом листе или листе
    # без строк все углы равны, и «лучший» угол — случайность.
    base = score(0.0)
    if base <= 0 or score(angle) <= base * (1.0 + _SKEW_MIN_GAIN):
        return 0.0
    return round(angle, 2)


def _long_runs(black: np.ndarray, min_len: int) -> np.ndarray:
    """Пиксели горизонтальных серий длиной не меньше `min_len`."""
    rows, _ = black.shape
    out = np.zeros_like(black, dtype=bool)
    padded = np.pad(black.astype(np.int8), ((0, 0), (1, 1)))
    edges = np.diff(padded, axis=1)
    for y in range(rows):
        starts = np.flatnonzero(edges[y] == 1)
        ends = np.flatnonzero(edges[y] == -1)
        for start, end in zip(starts, ends):
            if end - start >= min_len:
                out[y, start:end] = True
    return out


def _thin(mask: np.ndarray, max_thickness: int) -> np.ndarray:
    """Оставить из маски горизонтальных серий только тонкие по вертикали."""
    thick = _long_runs(mask.T, max_thickness + 1).T
    return mask & ~thick


def _line_mask(black: np.ndarray) -> np.ndarray:
    """
    Линии таблиц и рамок: длинные (больше доли листа) и тонкие серии.
    Толщина отсекает буквы: у крупного заголовка верхняя черта «Т» длинная,
    но толстая, а линия сетки на скане — несколько пикселей.
    """
    height, width = black.shape
    share = settings.OCR_LINE_MIN_SHARE
    max_thickness = max(3, round(min(height, width) * settings.OCR_LINE_MAX_THICKNESS))
    horizontal = _thin(_long_runs(black, max(20, int(width * share))), max_thickness)
    vertical = _thin(_long_runs(black.T, max(20, int(height * share))), max_thickness).T
    return horizontal | vertical
