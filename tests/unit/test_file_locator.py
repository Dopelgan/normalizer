"""Поиск исходного файла по s3_fileid."""

import pytest

from core.config import settings
from core.providers.file_locator import FileLocator, SourceFileNotFound


@pytest.fixture
def locator(temp_storage, monkeypatch):
    monkeypatch.setattr(settings, "STORAGE_TYPE", "local")
    monkeypatch.setattr(settings, "SOURCE_PREFIX", "documents/")
    return FileLocator(temp_storage)


class TestLocate:
    def test_exact_name(self, locator, temp_storage):
        temp_storage.write_file("documents/file-001.pdf", b"%PDF")
        located = locator.locate("file-001.pdf")
        assert located.uri == "documents/file-001.pdf"
        assert located.file_type == "pdf"
        assert located.is_image is False

    def test_extension_probed_when_absent(self, locator, temp_storage):
        temp_storage.write_file("documents/file-002.png", b"\x89PNG")
        located = locator.locate("file-002")
        assert located.uri == "documents/file-002.png"
        assert located.file_type == "png"
        assert located.is_image is True

    def test_pdf_wins_over_later_extensions(self, locator, temp_storage):
        temp_storage.write_file("documents/f.pdf", b"%PDF")
        temp_storage.write_file("documents/f.png", b"\x89PNG")
        assert locator.locate("f").file_type == "pdf"

    def test_missing_file(self, locator):
        with pytest.raises(SourceFileNotFound, match="не найден"):
            locator.locate("нет-такого-файла")

    def test_empty_id(self, locator):
        with pytest.raises(SourceFileNotFound):
            locator.locate("   ")

    def test_nested_key(self, locator, temp_storage):
        temp_storage.write_file("documents/2026/09/file.pdf", b"%PDF")
        assert locator.locate("2026/09/file.pdf").uri == "documents/2026/09/file.pdf"

    def test_s3_uri_built(self, temp_storage, monkeypatch):
        monkeypatch.setattr(settings, "STORAGE_TYPE", "s3")
        monkeypatch.setattr(settings, "S3_BUCKET", "docs")
        monkeypatch.setattr(settings, "SOURCE_PREFIX", "incoming/")

        class AlwaysExists:
            def exists(self, uri):
                return uri == "s3://docs/incoming/file-1.pdf"

        located = FileLocator(AlwaysExists()).locate("file-1.pdf")
        assert located.uri == "s3://docs/incoming/file-1.pdf"

    def test_traversal_attempt_is_not_found(self, locator):
        """Путь наружу из хранилища не должен ни находиться, ни падать."""
        with pytest.raises(SourceFileNotFound):
            locator.locate("../../etc/passwd")
