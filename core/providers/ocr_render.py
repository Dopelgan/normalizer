"""
С каким разрешением растрировать страницу PDF под OCR.

`OCR_DPI` — единственное число на весь корпус, и оно решает другое, чем
`OCR_MAX_SIDE`: DPI задаёт, с каким разрешением poppler **рисует**
страницу, то есть сколько настоящих деталей встроенного в PDF растра
доедет до Tesseract; потолок только **уменьшает** уже нарисованное. Поэтому
одно вместо другого не работает ни в ту, ни в другую сторону.

Замеренное (14.11.2026, `ocr-dpi-eval-20261114`) и объясняющее этот модуль:

* В корпусе 605 сканированных страниц. Собственное разрешение встроенного
  растра: 300 ppi у 415, 150 ppi у 89, 200 ppi у 56, выше 400 ppi у 45.
  При `OCR_DPI=200` две трети страниц рисуются **ниже** собственного
  разрешения — детали, которые в файле есть, выбрасываются до распознавания.
  Выше собственного разрешения рисовать бессмысленно: poppler растянет тот
  же растр, и Tesseract получит интерполяцию.
* Выигрыш и проигрыш решает **не DPI сам по себе, а высота символа**.
  Tesseract приводит каждую строку к 36 px перед сетью. На плотной мелкой
  печати (спецификация, 601 ppi) при 200 dpi высота слова 8 px, читается
  2919 символов листа и CER блока 0.269; при 600 dpi — 16 px, 20721 символа
  и CER 0.023, причём **ни одна из 17 строк эталона не стала хуже**. На
  разрежённом чертёжном листе (A3, 300 ppi) при 200 dpi высота уже 23 px,
  и подъём DPI только вредит: CER блока A 0.214 → 0.313, 13 строк из 19
  стали хуже.
* **Лучшее разрешение — собственное разрешение растра, не больше.** На
  плотной странице со своим растром 300 ppi: 200 dpi — CER 0.204,
  300 dpi — 0.088 (8 строк из 11 исправлены, ни одной худшей),
  400 dpi — 0.128, 600 dpi — 0.133. Выше собственного разрешения идёт
  интерполяция, и она уже вредит.
* **Безусловный подъём не бесплатен.** На тех же чертёжных листах
  выбранные правилом 211 dpi (A3 при потолке 3500) против нынешних 200
  дали CER блока A 0.214 → 0.247 и 9 худших строк из 19, блока B
  0.260 → 0.266 (8 из 13), легенды 0.296 → 0.326 (8 из 32). Поэтому
  `render_dpi` сам по себе — не рекомендованное умолчание: разрешение
  стоит поднимать там, где строки вышли мелкими, то есть через
  `small_print`/`rescan` вторым проходом.
* **Потолок забирает выигрыш обратно.** A3 при потолке 3500 — это 212 dpi
  (3500 / 16.53 дюйма): 300, 400 и 600 dpi дают один и тот же рабочий лист
  3500x2475 и одну и ту же высоту 25 px, различаясь только ценой
  растрирования (1.2 → 2.1 с) и памятью (183 → 627 МБ). На той же
  спецификации при 600 dpi потолок срезал прочитанное с 20721 символа до
  4855. Поэтому рисовать выше того, что потолок сохранит, здесь запрещено
  прямо — это чистый расход.

Отсюда два правила, и оба нужны:

1. Не рисовать выше собственного разрешения растра страницы — выше него
   деталей нет.
2. Не рисовать выше того, что сохранит `OCR_MAX_SIDE`, — иначе
   нарисованное тут же уменьшается обратно.

Флагов два, и они независимы. `OCR_DPI_FROM_SOURCE` включает безусловный
выбор разрешения по странице (`render_dpi`); `OCR_SMALL_PRINT_RESCAN` —
второй проход только для листов с мелкими строками (`rescan`). Оба
выключены: `render_dpi` возвращает `settings.OCR_DPI`, `rescan` возвращает
None, то есть прежнее поведение в точности.
"""

import logging
import math
import subprocess
from functools import lru_cache
from typing import Optional, Tuple

from core.config import settings

logger = logging.getLogger(__name__)

# Сколько ждать `pdfinfo`/`pdfimages`. Это чтение заголовков, доли секунды;
# секунды здесь означают, что файл не читается, и тогда остаётся `OCR_DPI`.
PROBE_TIMEOUT = 20


def _probe(command) -> str:
    try:
        done = subprocess.run(
            command, capture_output=True, text=True,
            timeout=PROBE_TIMEOUT, errors="replace",
        )
    except Exception as exc:  # noqa: BLE001 — не прочиталось, решает `OCR_DPI`
        logger.debug("%s не отработал: %s", command[0], exc)
        return ""
    return done.stdout or ""


@lru_cache(maxsize=256)
def page_inches(path: str, page: int) -> Optional[float]:
    """
    Длинная сторона страницы в дюймах, или None.

    Нужна, чтобы перевести потолок длинной стороны (в пикселях) в потолок
    разрешения (в точках на дюйм): это одна и та же граница, выраженная
    по-разному, и без перевода их нельзя сравнить.
    """
    for line in _probe(["pdfinfo", "-f", str(page), "-l", str(page), path]).splitlines():
        if not line.startswith("Page") or "size:" not in line:
            continue
        parts = line.split(":", 1)[1].split()
        try:
            width, height = float(parts[0]), float(parts[2])
        except (IndexError, ValueError):
            continue
        if width <= 0 or height <= 0:
            continue
        return max(width, height) / 72.0
    return None


@lru_cache(maxsize=256)
def source_dpi(path: str, page: int) -> Optional[int]:
    """
    Собственное разрешение самого крупного растра страницы, точек на дюйм.

    None — разрешения нет: страница векторная, картинок на ней не нашлось
    или `pdfimages` не отработал. Тогда судить не по чему и решает `OCR_DPI`.

    Берётся самая крупная картинка страницы: у скана она одна и есть сама
    страница, а мелкие (печать, логотип, подпись) разрешение страницы не
    задают и сбили бы оценку в любую сторону.
    """
    best_pixels, best_ppi = 0, None
    output = _probe(["pdfimages", "-list", "-f", str(page), "-l", str(page), path])
    for line in output.splitlines():
        parts = line.split()
        # Строки шапки и разделителя: первой колонкой у них не номер страницы.
        if len(parts) < 14 or not parts[0].isdigit():
            continue
        try:
            pixels = int(parts[3]) * int(parts[4])
            ppi = max(float(parts[12]), float(parts[13]))
        except ValueError:
            continue
        if ppi <= 0 or pixels <= best_pixels:
            continue
        best_pixels, best_ppi = pixels, int(round(ppi))
    return best_ppi


def dpi_kept_by_cap(path: str, page: int) -> Optional[int]:
    """
    Разрешение, выше которого нарисованное срежет `_fit`, или None.

    `OCR_MAX_SIDE <= 0` — потолка нет, и срезать нечего.
    """
    max_side = settings.OCR_MAX_SIDE
    if max_side <= 0:
        return None
    inches = page_inches(path, page)
    if not inches:
        return None
    return max(1, int(math.floor(max(512, max_side) / inches)))


def render_dpi(path: str, page: int) -> int:
    """
    С каким разрешением рисовать эту страницу.

    Флаг выключен — прежнее поведение, `settings.OCR_DPI` для всех страниц.
    """
    configured = max(1, settings.OCR_DPI)
    if not settings.OCR_DPI_FROM_SOURCE:
        return configured

    native = source_dpi(path, page)
    if not native:
        return configured
    # Ниже настроенного не опускаемся: `OCR_DPI` остаётся нижней границей,
    # иначе страница со 150-ppi растром читалась бы хуже, чем сегодня.
    chosen = min(max(configured, native), max(configured, settings.OCR_DPI_CEILING))
    kept = dpi_kept_by_cap(path, page)
    if kept is not None:
        chosen = min(chosen, max(configured, kept))
    if chosen != configured:
        logger.info(
            "Страница %d файла %s: рисуем в %d dpi (свой растр %d ppi, "
            "потолок сохраняет %s dpi)",
            page, path, chosen, native, kept if kept is not None else "—",
        )
    return chosen


def batch_runs(path: str, first: int, last: int, batch: int):
    """
    Пачки рендера как (первая, последняя, разрешение).

    Рендер идёт пачками страниц, а разрешение выбирается на страницу, и
    одним вызовом `convert_from_path` две страницы с разным разрешением не
    нарисуешь. Поэтому пачка рвётся там, где разрешение меняется: иначе
    пришлось бы рисовать всю пачку по самой требовательной странице — то
    самое лишнее рисование, против которого весь модуль.

    Длина пачки сохраняется: разрыв только делит её, но не удлиняет.
    """
    batch = max(1, batch)
    start, current = first, None
    for page in range(first, last + 1):
        dpi = render_dpi(path, page)
        if current is None:
            start, current = page, dpi
            continue
        if dpi != current or page - start >= batch:
            yield start, page - 1, current
            start, current = page, dpi
    if current is not None:
        yield start, last, current


def small_print(median_word_height: int) -> bool:
    """
    Строки листа вышли мельче, чем Tesseract умеет читать.

    Сеть Tesseract приводит строку к 36 px. Измеренное: при высоте 8–16 px
    лист отдаёт считанные проценты своего текста, и подъём разрешения
    исправляет это без единой испорченной строки; при 23 px и выше подъём
    уже только вредит. Порог взят между этими двумя областями.
    """
    if median_word_height <= 0:
        return False
    return median_word_height < max(1, settings.OCR_SMALL_PRINT_HEIGHT)


def rescan(path: str, page: int, current_dpi: int, median_word_height: int
           ) -> Optional[Tuple[int, int]]:
    """
    Перечитать ли страницу крупнее, и с чем: (разрешение, потолок стороны).

    None — перечитывать незачем. Возвращается и потолок: поднять одно
    разрешение бесполезно, `_fit` уменьшит лист обратно, и вся цена уйдёт
    впустую (замерено: 20721 символ против 4855 на одной и той же
    странице в 600 dpi с потолком и без).

    Второй проход стоит отдельного растрирования и отдельного Tesseract,
    поэтому он не для всех: только для листа, который на первом проходе
    отдал неприлично мелкие строки, и только пока его собственный растр
    даёт что рисовать.
    """
    if not settings.OCR_SMALL_PRINT_RESCAN or not small_print(median_word_height):
        return None
    native = source_dpi(path, page)
    if not native or native <= current_dpi:
        return None
    target = min(native, max(current_dpi, settings.OCR_DPI_CEILING))
    if target <= current_dpi:
        return None
    inches = page_inches(path, page)
    if not inches:
        return None
    # Потолок на второй проход — ровно под выбранное разрешение, не больше:
    # лист и так крупный, а память растёт как квадрат стороны.
    return target, int(math.ceil(target * inches))
