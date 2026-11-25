"""
Отпечаток сборки разбора (N-12, NORMALIZER_FINGERPRINT_REPARSE).

Уже проиндексированный файл отдаётся из БД без нового разбора, если совпала
версия профиля. Версия профиля — константа, и после выпуска с другим кодом
или другими флагами `create` того же файла отдавал прежний результат: на
вопрос «какие документы разобраны до флага X» отвечала только дата.

Отпечаток — две части:
- `release_sha` — SHA выпуска: переменная окружения RELEASE_SHA, иначе
  первая строка файла RELEASE_SHA_FILE, иначе «unknown»;
- `settings_sha256` — хеш настроек, от которых зависит разбор: тот же
  снимок, что входит в ключ кэша лестницы (`result_cache.settings_snapshot`),
  поэтому флаг с выключенным значением (WHEN_ON) хеша не меняет.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

from core.config import settings

UNKNOWN = "unknown"


def release_sha() -> str:
    """SHA выпуска из окружения или файла; без них — «unknown»."""
    value = (os.environ.get("RELEASE_SHA") or "").strip()
    if value:
        return value
    path = getattr(settings, "RELEASE_SHA_FILE", "") or ""
    if path:
        try:
            first = Path(path).read_text(encoding="utf-8").strip().splitlines()
        except OSError:
            first = []
        if first and first[0].strip():
            return first[0].strip()
    return UNKNOWN


def settings_sha256() -> str:
    """Хеш настроек разбора — снимка ключа кэша лестницы."""
    from core.ladder import result_cache

    snapshot = result_cache.settings_snapshot()
    blob = json.dumps(snapshot, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def current() -> Dict[str, str]:
    """Отпечаток этой сборки: {"release_sha", "settings_sha256"}."""
    return {"release_sha": release_sha(), "settings_sha256": settings_sha256()}


def matches(stored: Optional[Any], fingerprint: Optional[Dict[str, str]] = None) -> bool:
    """Сохранённый отпечаток тот же, что у этой сборки. Нет отпечатка — не тот."""
    if not isinstance(stored, dict):
        return False
    fingerprint = fingerprint or current()
    return all(stored.get(key) == value for key, value in fingerprint.items())
