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


class TestSniff:
    """Формат по первым байтам: расширение может врать или отсутствовать."""

    def test_pdf(self):
        assert filetypes.sniff(b"%PDF-1.7\n1 0 obj") == "pdf"

    def test_png(self):
        assert filetypes.sniff(b"\x89PNG\r\n\x1a\nIHDR") == "png"

    def test_jpeg(self):
        assert filetypes.sniff(b"\xff\xd8\xff\xe0JFIF") == "jpg"

    def test_tiff_both_byte_orders(self):
        assert filetypes.sniff(b"II*\x00") == "tiff"
        assert filetypes.sniff(b"MM\x00*") == "tiff"

    def test_dxf(self):
        assert filetypes.sniff(b"0\nSECTION\n2\nENTITIES\n") == "dxf"

    def test_dwg(self):
        assert filetypes.sniff(b"AC1027\x00\x00") == "dwg"

    def test_docx_and_xlsx_are_told_apart(self):
        import io
        import zipfile

        def archive(part: str) -> bytes:
            buffer = io.BytesIO()
            with zipfile.ZipFile(buffer, "w") as zf:
                zf.writestr("[Content_Types].xml", "<Types/>")
                zf.writestr(part, "<xml/>")
            return buffer.getvalue()

        assert filetypes.sniff(archive("word/document.xml")) == "docx"
        assert filetypes.sniff(archive("xl/workbook.xml")) == "xlsx"

    def test_plain_text_has_no_signature(self):
        assert filetypes.sniff("просто текст".encode()) is None

    def test_empty(self):
        assert filetypes.sniff(b"") is None


class TestResolve:
    def test_signature_wins_over_extension(self):
        verdict = filetypes.resolve("documents/договор.pdf", b"%PDF-1.4")
        assert verdict.file_type == "pdf"
        assert verdict.mismatch is False

    def test_mismatch_is_reported(self):
        verdict = filetypes.resolve("documents/скан.pdf", b"\x89PNG\r\n\x1a\n")
        assert verdict.file_type == "png"
        assert verdict.declared == "pdf"
        assert verdict.mismatch is True
        assert "png" in verdict.explanation

    def test_aliases_are_not_a_mismatch(self):
        verdict = filetypes.resolve("фото.jpeg", b"\xff\xd8\xff\xe0")
        assert verdict.mismatch is False
        assert verdict.file_type == "jpg"

    def test_no_signature_keeps_extension(self):
        verdict = filetypes.resolve("таблица.csv", "a;b;c".encode())
        assert verdict.file_type == "csv"
        assert verdict.detected is None

    def test_missing_extension_resolved_by_content(self):
        verdict = filetypes.resolve("documents/8e00bee8", b"%PDF-1.5")
        assert verdict.file_type == "pdf"
        assert verdict.declared == ""

    def test_declared_separately(self):
        verdict = filetypes.resolve_declared("pdf", b"\x89PNG\r\n\x1a\n")
        assert verdict.file_type == "png"
        assert verdict.mismatch is True


class TestOfficeLegacyFormats:
    """Форматы, которые реально приходят из бухгалтерии и кадров."""

    def test_xls_by_ole_signature(self):
        head = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"\x00" * 64
        head += "Workbook".encode("utf-16-le")
        assert filetypes.sniff(head) == "xls"

    def test_doc_by_ole_signature(self):
        head = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"\x00" * 64
        head += "WordDocument".encode("utf-16-le")
        assert filetypes.sniff(head) == "doc"

    def test_ods_by_mimetype_entry(self):
        import io
        import zipfile

        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("mimetype", "application/vnd.oasis.opendocument.spreadsheet")
            archive.writestr("content.xml", "<x/>")
        assert filetypes.sniff(buffer.getvalue()) == "ods"

    def test_xlsm_is_told_from_xlsx(self):
        import io
        import zipfile

        def book(with_macros: bool) -> bytes:
            buffer = io.BytesIO()
            with zipfile.ZipFile(buffer, "w") as archive:
                archive.writestr("[Content_Types].xml", "<Types/>")
                archive.writestr("xl/workbook.xml", "<workbook/>")
                if with_macros:
                    archive.writestr("xl/vbaProject.bin", "x")
            return buffer.getvalue()

        assert filetypes.sniff(book(True)) == "xlsm"
        assert filetypes.sniff(book(False)) == "xlsx"

    def test_legacy_spreadsheets_are_supported(self):
        for extension in ("xls", "xlsm", "ods"):
            assert filetypes.is_supported(extension), extension
            assert filetypes.kind_of(extension) == filetypes.KIND_SPREADSHEET

    def test_doc_is_known_but_not_supported(self):
        assert filetypes.is_known("doc")
        assert not filetypes.is_supported("doc")
        assert "docx" in filetypes.rejection_reason("doc")
