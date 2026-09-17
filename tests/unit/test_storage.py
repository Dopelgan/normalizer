"""Провайдеры хранилища: локальная ФС, защита от path traversal, фабрика."""

from pathlib import Path

import pytest

from core.config import settings
from core.providers.storage import (
    LocalStorageProvider,
    S3StorageProvider,
    StorageProviderFactory,
)


class TestLocalStorageProvider:
    def test_write_read_delete(self, tmp_path):
        provider = LocalStorageProvider(base_path=str(tmp_path))
        provider.write_file("test.txt", b"Hello")
        assert provider.exists("test.txt")
        with provider.get_stream("test.txt") as stream:
            assert stream.read() == b"Hello"
        assert provider.read_bytes("test.txt") == b"Hello"
        provider.delete_file("test.txt")
        assert provider.exists("test.txt") is False

    def test_creates_nested_directories(self, tmp_path):
        provider = LocalStorageProvider(base_path=str(tmp_path))
        provider.write_file("a/b/c/file.json", b"{}")
        assert (tmp_path / "a" / "b" / "c" / "file.json").exists()

    @pytest.mark.parametrize("uri", ["/etc/passwd", "../../etc/passwd", "file:///etc/passwd"])
    def test_path_traversal_blocked(self, tmp_path, uri):
        provider = LocalStorageProvider(base_path=str(tmp_path))
        with pytest.raises(ValueError, match="Path traversal detected"):
            provider.get_stream(uri)

    def test_exists_is_false_for_traversal(self, tmp_path):
        provider = LocalStorageProvider(base_path=str(tmp_path))
        assert provider.exists("../../etc/passwd") is False

    def test_file_uri_prefix_stripped(self, tmp_path):
        provider = LocalStorageProvider(base_path=str(tmp_path))
        provider.write_file("doc.txt", b"x")
        assert provider.exists(f"file://{tmp_path}/doc.txt")

    def test_accessible_uri_is_resolved_absolute_path(self, tmp_path):
        """Раньше здесь был мёртвый hasattr('_extract_path') и URI возвращался как есть."""
        provider = LocalStorageProvider(base_path=str(tmp_path))
        assert provider.get_accessible_uri("sub/doc.txt") == str(tmp_path / "sub" / "doc.txt")

    def test_accessible_uri_also_blocks_traversal(self, tmp_path):
        provider = LocalStorageProvider(base_path=str(tmp_path))
        with pytest.raises(ValueError):
            provider.get_accessible_uri("../../etc/passwd")

    def test_presigned_url_is_path(self, tmp_path):
        provider = LocalStorageProvider(base_path=str(tmp_path))
        assert provider.get_presigned_url("test.txt") == str(Path(tmp_path) / "test.txt")

    def test_uri_type(self, tmp_path):
        assert LocalStorageProvider(base_path=str(tmp_path)).get_uri_type("x") == "local"


class TestS3Config:
    def test_incomplete_configuration_raises(self, monkeypatch):
        monkeypatch.setattr(settings, "S3_ENDPOINT", None)
        with pytest.raises(ValueError, match="S3 configuration incomplete"):
            S3StorageProvider()

    def test_parse_uri(self, monkeypatch):
        for key, value in [
            ("S3_ENDPOINT", "http://minio:9000"), ("S3_ACCESS_KEY", "k"),
            ("S3_SECRET_KEY", "s"), ("S3_BUCKET", "bucket"),
        ]:
            monkeypatch.setattr(settings, key, value)
        provider = S3StorageProvider()
        assert provider._parse_s3_uri("s3://my-bucket/folder/file.pdf") == ("my-bucket", "folder/file.pdf")
        # Ключ без схемы попадает в бакет по умолчанию.
        assert provider._parse_s3_uri("folder/file.pdf") == ("bucket", "folder/file.pdf")

    def test_uri_without_key_rejected(self, monkeypatch):
        for key, value in [
            ("S3_ENDPOINT", "http://minio:9000"), ("S3_ACCESS_KEY", "k"),
            ("S3_SECRET_KEY", "s"), ("S3_BUCKET", "bucket"),
        ]:
            monkeypatch.setattr(settings, key, value)
        with pytest.raises(ValueError, match="отсутствует ключ"):
            S3StorageProvider()._parse_s3_uri("s3://bucket")


    def test_size_uses_s3_client(self, monkeypatch):
        """
        Регрессия: здесь стоял несуществующий `self.client` под голым
        `except Exception`, размер всегда получался None, и отсев по
        размеру на S3 молча не работал.
        """
        for key, value in [
            ("S3_ENDPOINT", "http://minio:9000"), ("S3_ACCESS_KEY", "k"),
            ("S3_SECRET_KEY", "s"), ("S3_BUCKET", "bucket"),
        ]:
            monkeypatch.setattr(settings, key, value)
        provider = S3StorageProvider()

        class _Stub:
            @staticmethod
            def head_object(Bucket, Key):
                assert (Bucket, Key) == ("bucket", "a.pdf")
                return {"ContentLength": 4096}

        provider.s3_client = _Stub()
        assert provider.size("a.pdf") == 4096


class TestFactory:
    def test_local_for_plain_and_file_uri(self):
        assert isinstance(StorageProviderFactory.get("test.txt"), LocalStorageProvider)
        assert isinstance(StorageProviderFactory.get("file:///test.txt"), LocalStorageProvider)

    def test_default_follows_storage_type(self, monkeypatch):
        monkeypatch.setattr(settings, "STORAGE_TYPE", "local")
        assert isinstance(StorageProviderFactory.default(), LocalStorageProvider)


@pytest.mark.integration
class TestS3Integration:
    """Требуют поднятого MinIO: docker compose up -d minio minio-init."""

    @pytest.fixture
    def s3_provider(self):
        try:
            provider = S3StorageProvider()
            provider.s3_client.head_bucket(Bucket=provider.bucket)
        except Exception as exc:  # noqa: BLE001
            pytest.skip(f"S3 недоступен: {exc}")
        return provider

    def test_roundtrip(self, s3_provider):
        uri = f"s3://{s3_provider.bucket}/tests/roundtrip.txt"
        s3_provider.write_file(uri, b"data")
        try:
            assert s3_provider.exists(uri)
            assert s3_provider.read_bytes(uri) == b"data"
        finally:
            s3_provider.delete_file(uri)
        assert s3_provider.exists(uri) is False

    def test_presigned_url_downloads(self, s3_provider):
        import requests

        uri = f"s3://{s3_provider.bucket}/tests/presigned.txt"
        s3_provider.write_file(uri, b"presigned")
        try:
            url = s3_provider.get_presigned_url(uri, expires_in=60)
            response = requests.get(url, timeout=10)
            assert response.status_code == 200
            assert response.content == b"presigned"
        finally:
            s3_provider.delete_file(uri)
