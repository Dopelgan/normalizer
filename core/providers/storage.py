"""
Абстракция хранилища для работы с файлами из локальной ФС и S3.
Поддерживает стриминг, проверку существования, подписанные URL и запись.
"""

import logging
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import BinaryIO, Dict, Optional, Tuple
from urllib.parse import urlparse

import boto3
from botocore.client import Config
from botocore.exceptions import ClientError

from core.config import settings

logger = logging.getLogger(__name__)


class ReadOnlyStorageError(RuntimeError):
    """Попытка записать в хранилище, выданное только на чтение."""


class StorageProvider(ABC):
    """Базовый интерфейс для всех провайдеров хранилища."""

    @abstractmethod
    def get_stream(self, uri: str) -> BinaryIO:
        """Возвращает файлоподобный объект для чтения (поток)."""

    @abstractmethod
    def exists(self, uri: str) -> bool:
        """Проверяет существование файла по URI."""

    @abstractmethod
    def get_presigned_url(self, uri: str, expires_in: int = 3600) -> str:
        """Подписанный URL (S3) или абсолютный путь (local)."""

    @abstractmethod
    def write_file(self, uri: str, content: bytes) -> None:
        """Записывает байтовое содержимое по указанному URI."""

    @abstractmethod
    def delete_file(self, uri: str) -> None:
        """Удаляет файл по URI (если существует)."""

    @abstractmethod
    def get_uri_type(self, uri: str) -> str:
        """Возвращает тип URI: 's3' или 'local'."""

    @abstractmethod
    def get_accessible_uri(self, uri: str) -> str:
        """URI, пригодный для чтения из текущего контекста."""

    def size(self, uri: str) -> Optional[int]:
        """
        Размер объекта в байтах, не выкачивая его. `None` — узнать не вышло.
        Нужен слою G-1: отсев по размеру должен быть дешевле чтения файла.
        """
        raise NotImplementedError

    def read_bytes(self, uri: str) -> bytes:
        stream = self.get_stream(uri)
        try:
            return stream.read()
        finally:
            close = getattr(stream, "close", None)
            if callable(close):
                close()


class LocalStorageProvider(StorageProvider):
    """Локальная ФС с защитой от Path Traversal."""

    def __init__(self, base_path: Optional[str] = None):
        self.base_path = Path(base_path or settings.STORAGE_LOCAL_MOUNT).resolve()

    # ----------------------------------------------------------- интерфейс
    def get_stream(self, uri: str) -> BinaryIO:
        return open(self._resolve_path(uri), "rb")

    def exists(self, uri: str) -> bool:
        try:
            return self._resolve_path(uri).exists()
        except ValueError:
            return False

    def get_presigned_url(self, uri: str, expires_in: int = 3600) -> str:
        return str(self._resolve_path(uri))

    def write_file(self, uri: str, content: bytes) -> None:
        path = self._resolve_path(uri)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            f.write(content)

    def delete_file(self, uri: str) -> None:
        path = self._resolve_path(uri)
        if path.exists():
            path.unlink()

    def size(self, uri: str) -> Optional[int]:
        try:
            return self._resolve_path(uri).stat().st_size
        except OSError:
            return None

    def get_uri_type(self, uri: str) -> str:
        return "local"

    def get_accessible_uri(self, uri: str) -> str:
        """Для локального хранилища это абсолютный путь внутри base_path."""
        return str(self._resolve_path(uri))

    # ------------------------------------------------------------ внутреннее
    def _resolve_path(self, uri: str) -> Path:
        """
        Извлекает путь из URI и проверяет, что он внутри base_path.
        Защита от Path Traversal через Path.is_relative_to().
        """
        raw_path = uri[7:] if uri.startswith("file://") else uri
        path = Path(raw_path)
        if not path.is_absolute():
            path = self.base_path / path
        resolved = path.resolve()
        if not resolved.is_relative_to(self.base_path):
            raise ValueError(f"Path traversal detected: {resolved} is outside {self.base_path}")
        return resolved


class S3StorageProvider(StorageProvider):
    """S3-совместимое хранилище."""

    def __init__(self, bucket: Optional[str] = None):
        self.endpoint = settings.S3_ENDPOINT
        self.access_key = settings.S3_ACCESS_KEY
        self.secret_key = settings.S3_SECRET_KEY
        self.bucket = bucket or settings.S3_BUCKET
        if not all([self.endpoint, self.access_key, self.secret_key, self.bucket]):
            raise ValueError(
                "S3 configuration incomplete. "
                "Check S3_ENDPOINT, S3_ACCESS_KEY, S3_SECRET_KEY, S3_BUCKET."
            )
        self.s3_client = boto3.client(
            "s3",
            endpoint_url=self.endpoint,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
            config=Config(signature_version="s3v4"),
        )

    # ----------------------------------------------------------- интерфейс
    def get_stream(self, uri: str) -> BinaryIO:
        bucket, key = self._parse_s3_uri(uri)
        return self.s3_client.get_object(Bucket=bucket, Key=key)["Body"]

    def exists(self, uri: str) -> bool:
        bucket, key = self._parse_s3_uri(uri)
        try:
            self.s3_client.head_object(Bucket=bucket, Key=key)
            return True
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code")
            if code in ("404", "NoSuchKey", "NotFound"):
                return False
            raise

    def get_presigned_url(self, uri: str, expires_in: int = 3600) -> str:
        bucket, key = self._parse_s3_uri(uri)
        return self.s3_client.generate_presigned_url(
            "get_object", Params={"Bucket": bucket, "Key": key}, ExpiresIn=expires_in
        )

    def write_file(self, uri: str, content: bytes) -> None:
        # Бакет выдан на чтение. Молча уронить запись в AccessDenied из
        # глубины boto3 хуже, чем сказать прямо: производное пишется в
        # ARTIFACTS_MOUNT, а сюда — только если S3_READONLY снят осознанно.
        self._ensure_writable(uri)
        bucket, key = self._parse_s3_uri(uri)
        self.s3_client.put_object(Bucket=bucket, Key=key, Body=content)

    def delete_file(self, uri: str) -> None:
        self._ensure_writable(uri)
        bucket, key = self._parse_s3_uri(uri)
        self.s3_client.delete_object(Bucket=bucket, Key=key)

    @staticmethod
    def _ensure_writable(uri: str) -> None:
        if settings.S3_READONLY:
            raise ReadOnlyStorageError(
                f"S3 доступен только на чтение, запись в {uri} запрещена. "
                "Производные файлы пишутся в ARTIFACTS_MOUNT; снимите "
                "S3_READONLY, только если бакет действительно writable."
            )

    def size(self, uri: str) -> Optional[int]:
        bucket, key = self._parse_s3_uri(uri)
        try:
            return int(self.s3_client.head_object(Bucket=bucket, Key=key)["ContentLength"])
        except ClientError as exc:
            # Раньше здесь стоял `self.client` (такого атрибута нет) под голым
            # `except Exception`: AttributeError проглатывался, размер всегда
            # получался None, и отсев по размеру на S3 молча не работал.
            logger.warning("Размер объекта %s не получен: %s", uri, exc)
            return None

    def get_uri_type(self, uri: str) -> str:
        return "s3"

    def get_accessible_uri(self, uri: str) -> str:
        return self.get_presigned_url(uri)

    # ------------------------------------------------------------ внутреннее
    def _parse_s3_uri(self, uri: str) -> Tuple[str, str]:
        """
        s3://bucket/path/to/file -> (bucket, key).
        Ключ без схемы трактуется как путь внутри бакета по умолчанию.
        """
        if not uri.startswith("s3://"):
            return self.bucket, uri.lstrip("/")
        parsed = urlparse(uri)
        bucket = parsed.netloc or self.bucket
        key = parsed.path.lstrip("/")
        if not key:
            raise ValueError(f"В S3 URI отсутствует ключ объекта: {uri}")
        return bucket, key


class ReadThroughS3Provider(StorageProvider):
    """
    Чтение из S3, запись — в собственный том.

    Рабочая конфигурация сервиса: документы лежат в общем S3, доступном
    только на чтение, а всё, что мы производим сами, остаётся у нас. Чтение
    маршрутизируется по схеме URI (`s3://` — в S3, остальное — локально),
    поэтому вызывающему коду не нужно знать, где именно лежит файл: он
    получает URI от `write_file` и потом тем же провайдером его читает.
    """

    def __init__(
        self,
        remote: Optional[StorageProvider] = None,
        local: Optional[StorageProvider] = None,
    ):
        self.remote = remote or S3StorageProvider()
        self.local = local or LocalStorageProvider(settings.ARTIFACTS_MOUNT)

    # ----------------------------------------------------------- интерфейс
    def _for(self, uri: str) -> StorageProvider:
        return self.remote if uri.startswith("s3://") else self.local

    def get_stream(self, uri: str) -> BinaryIO:
        return self._for(uri).get_stream(uri)

    def exists(self, uri: str) -> bool:
        return self._for(uri).exists(uri)

    def get_presigned_url(self, uri: str, expires_in: int = 3600) -> str:
        return self._for(uri).get_presigned_url(uri, expires_in)

    def write_file(self, uri: str, content: bytes) -> None:
        if uri.startswith("s3://"):
            raise ReadOnlyStorageError(
                f"S3 доступен только на чтение, запись в {uri} запрещена."
            )
        self.local.write_file(uri, content)

    def delete_file(self, uri: str) -> None:
        if uri.startswith("s3://"):
            raise ReadOnlyStorageError(
                f"S3 доступен только на чтение, удаление {uri} запрещено."
            )
        self.local.delete_file(uri)

    def size(self, uri: str) -> Optional[int]:
        return self._for(uri).size(uri)

    def get_uri_type(self, uri: str) -> str:
        return self._for(uri).get_uri_type(uri)

    def get_accessible_uri(self, uri: str) -> str:
        return self._for(uri).get_accessible_uri(uri)

    def read_bytes(self, uri: str) -> bytes:
        return self._for(uri).read_bytes(uri)


class StorageProviderFactory:
    """Фабрика провайдера по префиксу URI."""

    # Провайдер по умолчанию — один на процесс. За ним стоит клиент boto3 с
    # собственным пулом соединений, а спрашивают его почти все: контекст
    # документа, парсер, распознавание, разбор чертежей, поиск файла. На
    # документ выходило шесть-семь клиентов, каждый со своим пулом и своим
    # рукопожатием к S3.
    _default: Dict[str, StorageProvider] = {}
    _default_lock = threading.Lock()

    @staticmethod
    def get(uri: str) -> StorageProvider:
        if uri.startswith("s3://"):
            return S3StorageProvider()
        return LocalStorageProvider()

    @classmethod
    def default(cls) -> StorageProvider:
        """Провайдер, соответствующий STORAGE_TYPE из конфигурации."""
        kind = settings.STORAGE_TYPE.lower()
        # Ключ кэша учитывает режим доступа: смена S3_READONLY меняет
        # провайдера, а не только его поведение.
        key = f"{kind}:ro" if kind == "s3" and settings.S3_READONLY else kind
        with cls._default_lock:
            provider = cls._default.get(key)
            if provider is None:
                if kind == "s3":
                    provider = (
                        ReadThroughS3Provider()
                        if settings.S3_READONLY
                        else S3StorageProvider()
                    )
                else:
                    provider = LocalStorageProvider()
                cls._default[key] = provider
            return provider

    @classmethod
    def reset_default(cls) -> None:
        """Забыть кэш. Нужна тестам и смене настроек на лету."""
        with cls._default_lock:
            cls._default.clear()
