"""
Старые офисные форматы: конвертация в современные через LibreOffice.

DOC — двоичный формат Word 97-2003. Своей библиотеки под него нет: всё,
что умеет питон, — это вытащить из него плоский текст, потеряв таблицы, а
таблицы в наших документах и есть содержание. Поэтому файл один раз
конвертируется в DOCX, и дальше конвейер работает с ним как с любым DOCX —
и проба текста на приёме, и проверка открываемости, и уровень 2 лестницы.

Конвертер запускается подпроцессом и потому обставлен оговорками:

* LibreOffice может не стоять в образе — тогда формат отклоняется на
  приёме с внятной причиной, а не падает в середине разбора;
* конвертация одного файла идёт в своём каталоге профиля, иначе два
  одновременных запуска дерутся за общий профиль в домашнем каталоге и
  один из них молча возвращает пустой файл;
* результат кешируется по хешу содержимого: один и тот же файл за время
  обработки открывают трижды (проба текста, проверка открываемости,
  разбор), а конвертация стоит секунды.
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Optional, Tuple

from core import telemetry
from core.config import settings
from core.workspace import temp_dir

logger = logging.getLogger(__name__)

# Какой формат во что превращается. Расширение слева — то, что опознано по
# содержимому; справа — целевой формат и фильтр LibreOffice.
CONVERSIONS: Dict[str, Tuple[str, str]] = {
    "doc": ("docx", "docx:MS Word 2007 XML"),
}

# Кеш на один файл: больше не нужно, обработка идёт по одному документу.
_CACHE: Optional[Tuple[str, bytes]] = None


class ConversionUnavailable(RuntimeError):
    """Конвертер не установлен в этом образе."""


class ConversionFailed(RuntimeError):
    """Конвертер запустился, но результата не дал."""


def converts(file_type: str) -> bool:
    """Нужна ли этому типу конвертация перед разбором."""
    return (file_type or "").lower() in CONVERSIONS


def target_type(file_type: str) -> Optional[str]:
    entry = CONVERSIONS.get((file_type or "").lower())
    return entry[0] if entry else None


def available() -> bool:
    """Есть ли конвертер. Ответ нужен приёму, чтобы объяснить отказ."""
    if not settings.DOC_CONVERT_ENABLED:
        return False
    return shutil.which(settings.SOFFICE_BIN) is not None


def unavailable_reason(file_type: str) -> str:
    target = target_type(file_type) or "docx"
    if not settings.DOC_CONVERT_ENABLED:
        return (
            f"Конвертация .{file_type} выключена настройкой DOC_CONVERT_ENABLED. "
            f"Пересохраните файл как .{target}."
        )
    return (
        f".{file_type} — двоичный формат, для разбора он конвертируется в "
        f".{target}, но конвертер (LibreOffice) в образе не найден. "
        f"Пересохраните файл как .{target} или поставьте пакет "
        f"libreoffice-writer."
    )


def convert(data: bytes, file_type: str) -> bytes:
    """
    Содержимое файла в целевом формате. Для типа без конвертации возвращает
    байты как есть — вызывающему не нужно про это знать.
    """
    global _CACHE

    file_type = (file_type or "").lower()
    if not converts(file_type):
        return data
    if not data:
        raise ConversionFailed("нечего конвертировать: файл пуст")

    digest = hashlib.sha256(data).hexdigest()
    if _CACHE is not None and _CACHE[0] == digest:
        return _CACHE[1]

    if not available():
        raise ConversionUnavailable(unavailable_reason(file_type))

    target, filter_name = CONVERSIONS[file_type]
    with telemetry.measure(
        "convert.office", source=file_type, target=target, bytes=len(data)
    ) as span:
        converted = _run(data, file_type, target, filter_name)
        span["result_bytes"] = len(converted)

    _CACHE = (digest, converted)
    return converted


def reset_cache() -> None:
    """Сбросить кеш конвертации — нужен тестам."""
    global _CACHE
    _CACHE = None


# ===========================================================================
# Запуск
# ===========================================================================

def _run(data: bytes, source: str, target: str, filter_name: str) -> bytes:
    with tempfile.TemporaryDirectory(dir=str(temp_dir()), prefix="soffice-") as work:
        root = Path(work)
        source_path = root / f"document.{source}"
        source_path.write_bytes(data)
        out_dir = root / "out"
        out_dir.mkdir()
        profile = root / "profile"

        command = [
            settings.SOFFICE_BIN,
            f"-env:UserInstallation=file://{profile}",
            "--headless",
            "--norestore",
            "--nolockcheck",
            "--nodefault",
            "--nofirststartwizard",
            "--convert-to", filter_name,
            "--outdir", str(out_dir),
            str(source_path),
        ]
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                timeout=settings.DOC_CONVERT_TIMEOUT,
                env={**os.environ, "HOME": str(root)},
                check=False,
            )
        except FileNotFoundError as exc:
            raise ConversionUnavailable(unavailable_reason(source)) from exc
        except subprocess.TimeoutExpired as exc:
            raise ConversionFailed(
                f"конвертация .{source} не уложилась в "
                f"{settings.DOC_CONVERT_TIMEOUT} с"
            ) from exc

        produced = out_dir / f"document.{target}"
        if not produced.exists():
            # LibreOffice возвращает код 0 и на неудаче тоже — судить
            # приходится по тому, появился файл или нет.
            detail = (completed.stderr or completed.stdout or b"").decode(
                "utf-8", errors="replace"
            ).strip()[:300]
            raise ConversionFailed(
                f"конвертер не создал .{target}"
                + (f": {detail}" if detail else "")
            )
        result = produced.read_bytes()

    if not result:
        raise ConversionFailed(f"конвертер вернул пустой .{target}")
    logger.info("Файл .%s сконвертирован в .%s (%d Б)", source, target, len(result))
    return result
