"""
Формат .doc: разбор через конвертацию в .docx.

DOC — двоичный Word 97-2003. Всё, что умеет питон без посторонней помощи,
— вытащить из него плоский текст, потеряв таблицы, а таблицы в этих
документах и есть содержание. Поэтому файл конвертируется в DOCX и дальше
идёт штатным путём: проба текста на приёме, проверка открываемости, второй
уровень лестницы.

Часть проверок требует установленного LibreOffice и пропускается там, где
его нет. То, что от него не зависит — реестр типов, причина отказа,
опознание по сигнатуре, — проверяется всегда.
"""

import io
import subprocess

import pytest

from core import filetypes
from core.gateway import text_probe
from core.providers import office_convert


@pytest.fixture(autouse=True)
def clean_cache():
    office_convert.reset_cache()
    yield
    office_convert.reset_cache()


def docx_bytes(*lines: str, table=None) -> bytes:
    docx = pytest.importorskip("docx")
    document = docx.Document()
    for line in lines:
        document.add_paragraph(line)
    if table:
        grid = document.add_table(rows=len(table), cols=len(table[0]))
        for row_no, row in enumerate(table):
            for col_no, value in enumerate(row):
                grid.cell(row_no, col_no).text = str(value)
    buffer = io.BytesIO()
    document.save(buffer)
    return buffer.getvalue()


@pytest.fixture
def doc_bytes(tmp_path):
    """Настоящий .doc: собирается тем же LibreOffice из .docx."""
    if not office_convert.available():
        pytest.skip("LibreOffice не установлен")

    source = tmp_path / "src.docx"
    source.write_bytes(docx_bytes(
        "Приказ № 17 от 12.03.2026",
        "Об утверждении регламента приёмки",
        table=[("Позиция", "Срок"), ("Согласование", "10 дней")],
    ))
    subprocess.run(
        ["soffice", f"-env:UserInstallation=file://{tmp_path}/profile",
         "--headless", "--convert-to", "doc:MS Word 97",
         "--outdir", str(tmp_path), str(source)],
        capture_output=True, timeout=180, check=False,
    )
    produced = tmp_path / "src.doc"
    if not produced.exists():
        pytest.skip("LibreOffice не собрал .doc")
    return produced.read_bytes()


# ===========================================================================
# Реестр типов
# ===========================================================================

class TestRegistry:
    def test_doc_is_supported(self):
        assert filetypes.is_supported("doc")
        assert filetypes.kind_of("doc") == filetypes.KIND_OFFICE_TEXT

    def test_doc_is_in_probe_order(self):
        """Файл без расширения должен находиться и в этом формате тоже."""
        for extension in ("doc", "xlsm", "xls", "ods"):
            assert extension in filetypes.PROBE_ORDER, extension

    def test_missing_converter_is_explained_on_intake(self, monkeypatch):
        monkeypatch.setattr(office_convert, "available", lambda: False)
        reason = filetypes.rejection_reason("doc")
        assert reason and "LibreOffice" in reason

    def test_conversion_is_declared_only_for_legacy_formats(self):
        assert office_convert.converts("doc")
        assert office_convert.target_type("doc") == "docx"
        assert not office_convert.converts("docx")

    def test_convert_passes_modern_format_through(self):
        data = docx_bytes("Договор поставки")
        assert office_convert.convert(data, "docx") is data


# ===========================================================================
# Опознание по содержимому
# ===========================================================================

class TestSignature:
    def test_word_document_is_not_a_workbook(self, doc_bytes):
        """
        Регресс. Имена потоков внутри OLE2 искались в порядке «Workbook,
        Book, WordDocument», а короткое «Book» встречается в теле .doc
        случайно: документ Word объявлялся книгой Excel, уезжал в табличную
        ветку и там не открывался.
        """
        assert filetypes.sniff(doc_bytes) == "doc"

    def test_doc_named_docx_is_recognized(self, doc_bytes):
        resolved = filetypes.resolve("documents/приказ.docx", doc_bytes)
        assert resolved.file_type == "doc"
        assert resolved.mismatch


# ===========================================================================
# Разбор
# ===========================================================================

class TestParsing:
    def test_text_probe_reads_doc(self, doc_bytes):
        probe = text_probe.extract("doc", doc_bytes)
        assert probe.source == text_probe.SOURCE_DOCX
        assert "Приказ" in probe.text

    def test_table_content_survives_conversion(self, doc_bytes):
        """
        Ради содержимого таблиц конвертация и делается: antiword и прочие
        читалки плоского текста теряют ячейки целиком.

        Сохранится ли таблица именно таблицей, зависит от самого файла:
        двойной проход docx -> doc -> docx через LibreOffice иногда
        раскладывает простую сетку в абзацы. Значения ячеек при этом на
        месте, и проверяем мы их.
        """
        probe = text_probe.extract("doc", doc_bytes)
        assert "Согласование" in probe.text and "10 дней" in probe.text

    def test_second_conversion_is_taken_from_cache(self, doc_bytes, monkeypatch):
        """Один файл за обработку открывают трижды; конвертировать хватит раз."""
        first = office_convert.convert(doc_bytes, "doc")

        def fail(*_args, **_kwargs):
            raise AssertionError("конвертер вызван повторно")

        monkeypatch.setattr(office_convert, "_run", fail)
        assert office_convert.convert(doc_bytes, "doc") == first

    def test_quality_gate_opens_doc(self, doc_bytes):
        from core.quality.service import _can_open

        opened, detail = _can_open(doc_bytes, "doc")
        assert opened, detail

    def test_ladder_level_two_takes_doc(self, doc_bytes):
        from core.ladder.context import DocumentContext
        from core.ladder.strategies.native_text import DocxStrategy

        context = DocumentContext(uri="документ.doc", file_type="doc", data=doc_bytes)
        strategy = DocxStrategy()
        assert strategy.applicable(context)
        result = strategy.run(context)
        assert any("Приказ" in (b.text or "") for b in result.blocks)
        assert any("регламента" in (b.text or "") for b in result.blocks)


class TestWithoutConverter:
    def test_probe_says_why_it_could_not_read(self, monkeypatch):
        monkeypatch.setattr(office_convert, "available", lambda: False)
        probe = text_probe.extract("doc", b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"\x00" * 64)
        assert probe.source == text_probe.SOURCE_NONE
        assert "LibreOffice" in (probe.reason or "")

    def test_ladder_skips_the_level(self, monkeypatch):
        from core.ladder.context import DocumentContext
        from core.ladder.strategies.native_text import DocxStrategy

        monkeypatch.setattr(office_convert, "available", lambda: False)
        context = DocumentContext(
            uri="документ.doc", file_type="doc",
            data=b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"\x00" * 64,
        )
        # Уровень неприменим — лестница идёт выше, а не падает здесь.
        assert not DocxStrategy().applicable(context)

    def test_empty_file_is_not_sent_to_the_converter(self):
        with pytest.raises(office_convert.ConversionFailed):
            office_convert.convert(b"", "doc")
