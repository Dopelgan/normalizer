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

# Сигнатуре формата хватает первых байтов: качать весь объект ради типа
# незачем, тем более что хранилище может быть удалённым.
_HEAD_BYTES = 8192


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
                        file_type=self._file_type(candidate, uri),
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

    def _file_type(self, key: str, uri: str) -> str:
        """
        Тип найденного файла. Расширения в ключе может не быть вовсе —
        тогда спрашиваем не догадку, а сам файл: прежний код в этом случае
        возвращал "pdf", и DOCX под безымянным ключом уезжал на разбор PDF.
        """
        extension = filetypes.extension_of(key)
        if extension:
            return extension
        detected = filetypes.sniff(self._head(uri))
        if detected:
            logger.info("Тип %s определён по сигнатуре: .%s", uri, detected)
            return detected
        # Формат не опознан по голове файла (так бывает у docx и xlsx —
        # это zip, и его состав виден только целиком). Пустой тип честнее
        # выдуманного: он уточнится, когда файл будет прочитан.
        return ""

    def _head(self, uri: str) -> bytes:
        """Первые байты файла: дешевле полного чтения и хватает сигнатуре."""
        try:
            stream = self.storage.get_stream(uri)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Голова файла %s не прочитана: %s", uri, exc)
            return b""
        try:
            return stream.read(_HEAD_BYTES)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Голова файла %s не прочитана: %s", uri, exc)
            return b""
        finally:
            close = getattr(stream, "close", None)
            if callable(close):
                close()
