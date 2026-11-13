"""
Выбор каталога моделей Tesseract.

Одна строка конфигурации распознавания, `--tessdata-dir`, собирается здесь,
а не в месте вызова: вызовов Tesseract в конвейере несколько, и каталог у
них обязан быть один и тот же. Иначе страница, прочитанная одной моделью
на приёме, на разборе читается другой, и числа читаемости перестают
сравниваться между этапами.

Почему это вообще настройка. В образе стоят модели из пакетов Debian
(`tesseract-ocr-rus` 1:4.1.0-2, файлы 2019 года) — это `tessdata_fast`,
самый маленький и самый неточный из трёх опубликованных вариантов. Подмена
на `tessdata_best`, без какого-либо обучения, снижает ошибку на символ на
настоящих листах с 0,267 до 0,211 и с 0,262 до 0,229 (замеры —
`docs/ocr-models.md`). Платится временем: `best` читает лист примерно в
полтора раза дольше, поэтому по умолчанию каталог не подменяется.

Недоступный каталог не повод терять документ: распознавание идёт системной
моделью, а в лог уходит предупреждение. Молча подменять модель на другую
нельзя — иначе непонятно, чем прочитан лист.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from core.config import settings

logger = logging.getLogger(__name__)

# Файлы, без которых каталог бесполезен: `--tessdata-dir` подменяет его
# целиком. Без `eng` не собирается связка `rus+eng`, без `osd` отваливается
# определение поворота листа в `ocr_prepare`.
REQUIRED = ("rus.traineddata", "eng.traineddata", "osd.traineddata")

_warned: set = set()


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
    """Хвост строки конфигурации Tesseract: ` --tessdata-dir <путь>` или пусто."""
    directory = tessdata_dir()
    return f" --tessdata-dir {directory}" if directory else ""


def _warn(directory: str, reason: str) -> None:
    """Предупредить один раз на каталог: иначе лог забивается постранично."""
    if directory in _warned:
        return
    _warned.add(directory)
    logger.warning(
        "OCR_TESSDATA_DIR=%r не используется (%s) — читаем системной моделью",
        directory, reason,
    )
