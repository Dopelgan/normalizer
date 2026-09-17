"""
Поиск исходного файла по `s3_fileid`.

По контракту RAG передаёт только идентификатор файла в общем S3, а Parser
сам находит объект. Здесь идентификатор превращается в URI хранилища и тип
файла. Если идентификатор пришёл без расширения — перебираются известные.
"""

import logging
import os
from dataclasses import dataclass
from typing import Optional

from core import filetypes
from core.config import settings
from core.providers.storage import StorageProvider, StorageProviderFactory

logger = logging.getLogger(__name__)


class SourceFileNotFound(FileNotFoundError):
    """Файл по s3_fileid не найден ни с одним из проверенных расширений."""


@dataclass(frozen=True)
class LocatedFile:
    s3_fileid: str
    uri: str
    file_type: str      # pdf, png, docx, ...

    @property
    def is_image(self) -> bool:
        return filetypes.is_image(self.file_type)

    @property
    def kind(self) -> Optional[str]:
        """Род содержимого: по нему лестница стратегий отбирает применимые."""
        return filetypes.kind_of(self.file_type)


class FileLocator:
    """Превращает s3_fileid в URI хранилища."""

    def __init__(self, storage: Optional[StorageProvider] = None):
        self.storage = storage or StorageProviderFactory.default()
        self.is_s3 = settings.STORAGE_TYPE.lower() == "s3"

    def locate(self, s3_fileid: str) -> LocatedFile:
        fileid = s3_fileid.strip().lstrip("/")
        if not fileid:
            raise SourceFileNotFound("Пустой s3_fileid")

        candidates = [fileid]
        if not os.path.splitext(fileid)[1]:
            candidates += [fileid + ext for ext in settings.SOURCE_PROBE_EXTENSIONS]

        for candidate in candidates:
            uri = self._build_uri(candidate)
            try:
                if self.storage.exists(uri):
                    return LocatedFile(
                        s3_fileid=s3_fileid,
                        uri=uri,
                        file_type=self._file_type(candidate),
                    )
            except Exception as exc:  # недоступное хранилище — не молчим
                logger.warning("Проверка %s не удалась: %s", uri, exc)

        raise SourceFileNotFound(
            f"Файл для s3_fileid={s3_fileid!r} не найден "
            f"(префикс {settings.SOURCE_PREFIX!r}, проверено {len(candidates)} вариантов)"
        )

    # ------------------------------------------------------------ внутреннее
    def _build_uri(self, key: str) -> str:
        prefix = settings.SOURCE_PREFIX or ""
        if prefix and not prefix.endswith("/"):
            prefix += "/"
        path = f"{prefix}{key}"
        if self.is_s3:
            return f"s3://{settings.S3_BUCKET}/{path}"
        return path

    @staticmethod
    def _file_type(key: str) -> str:
        return filetypes.extension_of(key) or "pdf"
