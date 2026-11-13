"""
Выбор каталога моделей Tesseract для распознавания.

Одна строка конфигурации, `--tessdata-dir`, собирается здесь, а не в
месте вызова: вызовов Tesseract в конвейере несколько, и каталог у них
обязан быть один и тот же. Иначе страница, прочитанная одной моделью на
приёме, на разборе читается другой, и числа читаемости перестают
сравниваться между этапами.

Почему это вообще настройка. В образе стоят модели из пакетов Debian
(`tesseract-ocr-rus` 1:4.1.0-2, файлы 2019 года) — это `tessdata_fast`,
самый маленький и самый неточный из трёх опубликованных вариантов.
Подмена на `tessdata_best`, без какого-либо обучения, снижает ошибку на
символ на настоящих листах с 0,267 до 0,211 и с 0,262 до 0,229 (замеры —
`docs/ocr-models.md`). Платится временем: `best` читает лист примерно в
полтора раза дольше, поэтому по умолчанию каталог не подменяется.

Почему не `TESSDATA_PREFIX`, который Tesseract понимает сам и который
переключил бы модель вообще без кода. Он переключает её для всех вызовов
разом, а неверное значение в нём означает не возврат к системной модели,
а полный отказ Tesseract: каждый скан становится нечитаемым, и приём
отправляет документы в карантин. Здесь каталог проверяется до вызова, и
при любой беде с ним лист читается системной моделью, а в лог уходит
предупреждение. Для разового опыта `TESSDATA_PREFIX` по-прежнему годится
— он описан в `docs/ocr-models.md`.

`osd.traineddata` в каталоге не нужен: `--tessdata-dir` ставится только
вызовам распознавания, а они идут с языками `rus+eng`. Определение
поворота листа (`ocr_prepare`) эту настройку не получает и остаётся на
системной модели — `osd` во всех трёх вариантах один и тот же файл.
"""

from __future__ import annotations

import logging
import os
import shlex
from typing import Optional, Set

from core.config import settings

logger = logging.getLogger(__name__)

# Файлы, без которых каталог бесполезен: `--tessdata-dir` подменяет его
# целиком, и без любого из двух не соберётся связка `rus+eng`.
REQUIRED = ("rus.traineddata", "eng.traineddata")

_warned: Set[str] = set()


def tessdata_dir() -> Optional[str]:
    """Каталог моделей для распознавания или `None` — системный."""
    directory = (settings.OCR_TESSDATA_DIR or "").strip()
    if not directory:
        return None
    if not os.path.isdir(directory):
        _warn(directory, "каталог недоступен")
        return None
    missing = [name for name in REQUIRED if not os.path.isfile(os.path.join(directory, name))]
    if missing:
        _warn(directory, f"нет файлов {', '.join(missing)}")
        return None
    return directory


def tessdata_option() -> str:
    """
    Хвост строки конфигурации Tesseract: ` --tessdata-dir <путь>` или пусто.

    Путь экранируется: pytesseract разбирает строку конфигурации через
    `shlex.split`, и пробел в пути иначе разваливает вызов на два слова.
    """
    directory = tessdata_dir()
    return f" --tessdata-dir {shlex.quote(directory)}" if directory else ""


def _warn(directory: str, reason: str) -> None:
    """Предупредить один раз на каталог: иначе лог забивается постранично."""
    if directory in _warned:
        return
    _warned.add(directory)
    logger.warning(
        "OCR_TESSDATA_DIR=%r не используется (%s) — читаем системной моделью",
        directory, reason,
    )
