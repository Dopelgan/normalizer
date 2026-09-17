"""Реестр типов файлов: опознание, MIME, род содержимого, отказы."""

import pytest

from core import filetypes


class TestRecognition:
    @pytest.mark.parametrize("value", ["pdf", "PDF", ".pdf", " .PDF "])
    def test_normalization(self, value):
        assert filetypes.normalize(value) == "pdf"

    @pytest.mark.parametrize("path,expected", [
        ("documents/чертёж.DXF", "dxf"),
        ("a/b/файл.tar.gz", "gz"),
        ("без_расширения", ""),
    ])
    def test_extension_of(self, path, expected):
        assert filetypes.extension_of(path) == expected

    @pytest.mark.parametrize("ext", [
        "pdf", "jpg", "jpeg", "png", "tiff", "tif", "bmp",
        "docx", "txt", "csv", "md", "xlsx", "dxf",
    ])
    def test_every_requested_format_is_supported(self, ext):
        """Полный список типов из задания принимается конвейером."""
        assert filetypes.is_supported(ext)
        assert filetypes.rejection_reason(ext) is None


class TestKinds:
    @pytest.mark.parametrize("ext,kind", [
        ("pdf", filetypes.KIND_PDF),
        ("png", filetypes.KIND_IMAGE),
        ("tif", filetypes.KIND_IMAGE),
        ("docx", filetypes.KIND_OFFICE_TEXT),
        ("xlsx", filetypes.KIND_SPREADSHEET),
        ("csv", filetypes.KIND_SPREADSHEET),
        ("md", filetypes.KIND_PLAIN_TEXT),
        ("dxf", filetypes.KIND_CAD),
    ])
    def test_kind(self, ext, kind):
        assert filetypes.kind_of(ext) == kind

    def test_images_are_listed_once(self):
        assert filetypes.IMAGE_EXTENSIONS == {"jpg", "jpeg", "png", "tiff", "tif", "bmp"}


class TestMime:
    @pytest.mark.parametrize("ext,mime", [
        ("pdf", "application/pdf"),
        ("jpeg", "image/jpeg"),
        ("csv", "text/csv"),
        ("dxf", "image/vnd.dxf"),
    ])
    def test_known(self, ext, mime):
        assert filetypes.mime_for(ext) == mime

    def test_unknown_falls_back(self):
        assert filetypes.mime_for("xyz") == "application/octet-stream"


class TestRejection:
    def test_dwg_is_known_but_not_supported(self):
        """DWG опознаётся — отказ объясняет, что делать, а не просто ругается."""
        assert filetypes.is_known("dwg")
        assert not filetypes.is_supported("dwg")
        reason = filetypes.rejection_reason("dwg")
        assert "DXF" in reason and "конвертер" in reason

    def test_unknown_format_lists_supported(self):
        reason = filetypes.rejection_reason("exe")
        assert ".pdf" in reason and ".xlsx" in reason

    def test_missing_extension(self):
        assert "расширения" in filetypes.rejection_reason("")
