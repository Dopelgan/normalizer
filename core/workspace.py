"""
Рабочий каталог процесса для временных файлов.

Всё временное — выкачанный из S3 исходник, промежуточные растры poppler,
файлы, которые tesseract пишет рядом с собой, — должно лежать в томе
сервиса, а не на слое контейнера: слой не резиновый, при падении воркера
мусор в нём не виден и не чистится, а на readonly-S3 деться этим файлам
больше некуда.

`configure_process_tempdir()` вызывается один раз на старте процесса и
выставляет TMPDIR/TEMP/TMP, поэтому каталог наследуют и сторонние
библиотеки, и запускаемые из них подпроцессы (pdftoppm, tesseract).
"""

import logging
import os
import tempfile
from pathlib import Path

from core.config import settings

logger = logging.getLogger(__name__)


def temp_dir() -> Path:
    """Каталог для временных файлов; создаётся при первом обращении."""
    path = Path(settings.TEMP_DIR)
    try:
        path.mkdir(parents=True, exist_ok=True)
        return path
    except OSError as exc:
        # Том не смонтирован или права не те: работать без временных файлов
        # сервис не может, но и падать на импорте не должен — уходим в
        # системный каталог и говорим об этом в лог.
        fallback = Path(tempfile.gettempdir())
        logger.warning(
            "Каталог временных файлов %s недоступен (%s), используется %s",
            path, exc, fallback,
        )
        return fallback


def configure_process_tempdir() -> None:
    """Направить временные файлы процесса и его подпроцессов в том сервиса."""
    path = temp_dir()
    tempfile.tempdir = str(path)
    for name in ("TMPDIR", "TEMP", "TMP"):
        os.environ[name] = str(path)
    logger.info("Временные файлы: %s", path)
