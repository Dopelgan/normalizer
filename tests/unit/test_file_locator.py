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

    def test_type_of_extensionless_key_comes_from_signature(self, locator, temp_storage):
        """
        Ключ без расширения — обычное дело в контракте RAG. Раньше такому
        файлу молча приписывался тип pdf, и DOCX уезжал на разбор PDF.
        """
        temp_storage.write_file("documents/8e00bee8", b"%PDF-1.5\n1 0 obj")
        assert locator.locate("8e00bee8").file_type == "pdf"

    def test_png_without_extension(self, locator, temp_storage):
        temp_storage.write_file("documents/img-1", b"\x89PNG\r\n\x1a\nIHDR")
        located = locator.locate("img-1")
        assert located.file_type == "png"
        assert located.is_image is True

    def test_unknown_signature_gives_empty_type(self, locator, temp_storage):
        """Пустой тип честнее выдуманного: он уточнится при чтении файла."""
        temp_storage.write_file("documents/unknown-1", "просто текст".encode())
        assert locator.locate("unknown-1").file_type == ""

    def test_traversal_attempt_is_not_found(self, locator):
        """Путь наружу из хранилища не должен ни находиться, ни падать."""
        with pytest.raises(SourceFileNotFound):
            locator.locate("../../etc/passwd")


# ---------------------------------------------------------------- вложения чата
_UUID = "0b9f3c2e-5d1a-4c6b-9e8f-1a2b3c4d5e6f"
_ZIP_HEAD = b"PK\x03\x04" + b"\x00" * 60


class _FakeS3:
    """Хранилище из двух бакетов: основной (docs) и вложений чата."""

    def __init__(self, objects, meta=None):
        self.objects = objects          # uri -> bytes
        self.meta = meta or {}          # uri -> dict
        self.checked = []

    def exists(self, uri):
        self.checked.append(uri)
        return uri in self.objects

    def get_stream(self, uri):
        import io
        return io.BytesIO(self.objects[uri])

    def metadata(self, uri):
        return self.meta.get(uri, {})


@pytest.fixture
def s3_settings(monkeypatch):
    monkeypatch.setattr(settings, "STORAGE_TYPE", "s3")
    monkeypatch.setattr(settings, "S3_BUCKET", "knowledge-base")
    monkeypatch.setattr(settings, "SOURCE_PREFIX", "")
    monkeypatch.setattr(settings, "S3_ATTACHMENTS_BUCKET", "chat-attachments")


class TestChatAttachments:
    """
    Uuid вложения чата ищется и в бакете вложений: в основном бакете
    (uuid + 17 расширений) его нет, и поиск дал бы SourceFileNotFound.
    """

    def test_docx_attachment_found_with_type_from_metadata(self, s3_settings):
        uri = f"s3://chat-attachments/{_UUID}"
        storage = _FakeS3(
            {uri: _ZIP_HEAD},
            {uri: {"filename": "%D0%BF%D1%80%D0%B8%D0%BA%D0%B0%D0%B7.docx"}},
        )
        located = FileLocator(storage).locate(_UUID)
        assert located.uri == uri
        assert located.file_type == "docx"
        assert located.s3_fileid == _UUID

    def test_xlsx_attachment_type_from_metadata(self, s3_settings):
        uri = f"s3://chat-attachments/{_UUID}"
        storage = _FakeS3({uri: _ZIP_HEAD}, {uri: {"filename": "report%202026.XLSX"}})
        assert FileLocator(storage).locate(_UUID).file_type == "xlsx"

    def test_without_metadata_type_is_sniffed(self, s3_settings):
        uri = f"s3://chat-attachments/{_UUID}"
        storage = _FakeS3({uri: b"%PDF-1.7\n"})
        assert FileLocator(storage).locate(_UUID).file_type == "pdf"

    def test_knowledge_base_checked_first(self, s3_settings):
        kb = f"s3://knowledge-base/{_UUID}.pdf"
        att = f"s3://chat-attachments/{_UUID}"
        storage = _FakeS3({kb: b"%PDF-1.7", att: _ZIP_HEAD})
        assert FileLocator(storage).locate(_UUID).uri == kb
        assert att not in storage.checked

    def test_attachment_probed_exactly_once_without_extensions(self, s3_settings):
        storage = _FakeS3({})
        with pytest.raises(SourceFileNotFound):
            FileLocator(storage).locate(_UUID)
        attachment_checks = [u for u in storage.checked if "chat-attachments" in u]
        assert attachment_checks == [f"s3://chat-attachments/{_UUID}"]

    def test_disabled_when_setting_empty(self, s3_settings, monkeypatch):
        monkeypatch.setattr(settings, "S3_ATTACHMENTS_BUCKET", "")
        att = f"s3://chat-attachments/{_UUID}"
        storage = _FakeS3({att: _ZIP_HEAD})
        with pytest.raises(SourceFileNotFound):
            FileLocator(storage).locate(_UUID)
        assert att not in storage.checked

    @pytest.mark.parametrize("fileid", [
        _UUID.upper(),
        _UUID + ".docx",
        "x/" + _UUID,
        "../" + _UUID,
        _UUID + "/..",
        "cluster-a/document",
        "not-a-uuid",
    ])
    def test_only_bare_lowercase_uuid_goes_to_attachments(self, s3_settings, fileid):
        storage = _FakeS3({})
        with pytest.raises(SourceFileNotFound):
            FileLocator(storage).locate(fileid)
        assert not any("chat-attachments" in u for u in storage.checked)
