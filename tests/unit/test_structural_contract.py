"""Согласованный Parser → RAG: структурные данные, координаты, качество."""

from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from core.contract_finalize import finalize_fragments, stable_fragment_id
from core.drawing_quality import apply_drawing_quality
from core.mappers import fragment_to_row, row_to_fragment
from core.models.contract import Content, DrawingField, Fragment, Position, Provenance, SourceSpan
from core.models.parse_result import ParsedBlock, ParseResult
from core.normalizer import TextNormalizer
from core.processing_contract import PROCESSING_PROFILE_VERSION


META = {"doc_id": "doc-contract", "s3_fileid": "original.pdf"}


def normalize(*blocks):
    return TextNormalizer().normalize(ParseResult(blocks=list(blocks), exhaustive=True), META)


def drawing(confidence=0.95, completeness=None, mechanisms=None):
    return Fragment(
        fragment_id="drawing-1", type="drawing",
        content=Content(structured_drawing_fields=[DrawingField(
            category="size", value="20", confidence=confidence,
        )]),
        position=Position(page=1), confidence=confidence, completeness=completeness,
        completeness_mechanisms=mechanisms or {},
        provenance=Provenance(method="vector_text_layer", strategy_level=2, source="drawing"),
    )


def test_complete_unicode_block_is_preserved_without_tokenization():
    raw = "  Начало 😀\n" + "длинный текст " * 1500 + "\nКонец  "
    fragment = normalize(ParsedBlock(type="text", text=raw, page=2, sheet_name="А"))[0]
    assert fragment.content.text == raw
    span = fragment.position.source_spans[0]
    assert raw[span.char_start:span.char_end] == raw
    assert span.char_end == len(raw)
    assert fragment.position.text_offsets_exact is True
    assert fragment.position.bbox_exact is False
    assert fragment.position.page_start == fragment.position.page_end == 2
    assert span.sheet == "А"
    assert fragment.provenance.processing_profile_version == PROCESSING_PROFILE_VERSION
    assert span.source_unit_id == fragment.provenance.source_unit_id


def test_structural_units_never_merge_even_on_the_same_page():
    blocks = [ParsedBlock(type="text", text="А", page=1, sheet_name="A"),
              ParsedBlock(type="text", text="Б", page=1, sheet_name="B")]
    fragments = normalize(*blocks)
    assert [f.content.text for f in fragments] == ["А", "Б"]
    assert [f.position.sheet for f in fragments] == ["A", "B"]


def test_full_table_including_oversized_row_and_extra_metadata_is_preserved():
    rows = [[str(i), "x" * 10000 if i == 500 else "Значение"] for i in range(1001)]
    table = {"headers": ["№", "Значение"], "rows": rows,
             "merged_cells": [{"start": [0, 0], "end": [0, 1]}]}
    fragment = normalize(ParsedBlock(type="table", page=3, table_data=table))[0]
    result = fragment.content.table_data.model_dump()
    assert result["rows"] == rows
    assert result["merged_cells"] == table["merged_cells"]
    assert result["row_start"] == 0 and result["row_end"] == 1001
    assert result["table_id"] == fragment.provenance.source_unit_id


def test_source_unit_ids_and_fragment_ids_survive_unrelated_insertions():
    tail = ParsedBlock(type="text", text="Хвост", page=2, order=0)
    original = normalize(tail)[0]
    again = normalize(ParsedBlock(type="text", text="Вставка", page=1), tail)[1]
    assert original.fragment_id == again.fragment_id
    assert original.provenance.source_unit_id == again.provenance.source_unit_id


def test_duplicate_source_units_stay_distinct():
    fragments = normalize(*[ParsedBlock(type="text", text="Повтор", page=1) for _ in range(2)])
    assert len({f.fragment_id for f in fragments}) == 2
    assert len({f.provenance.source_unit_id for f in fragments}) == 2


def test_finalization_is_idempotent_for_generated_drawing():
    fragment = drawing()
    first, _ = finalize_fragments([fragment], "doc-1", "file-1")
    identity = first[0].fragment_id
    second, _ = finalize_fragments(first, "doc-1", "file-1")
    assert second[0].fragment_id == identity
    assert stable_fragment_id("doc-1", second[0]) == identity


def test_changed_table_only_drawing_changes_fragment_id():
    one = drawing()
    two = drawing()
    one.content.structured_drawing_fields = two.content.structured_drawing_fields = []
    one.content.structured_payload = {"tables": [{"headers": ["A"], "rows": [["1"]]}]}
    two.content.structured_payload = {"tables": [{"headers": ["A"], "rows": [["2"]]}]}
    assert stable_fragment_id("doc-1", one) != stable_fragment_id("doc-1", two)


@pytest.mark.parametrize("confidence,expected", [(0.9499, False), (0.95, True), (1.0, True)])
def test_individual_drawing_facts_have_explicit_threshold(confidence, expected):
    fragment = drawing(confidence=confidence)
    fragments, _ = finalize_fragments([fragment], "doc-1", "file-1")
    assert fragment.content.structured_drawing_fields[0].confirmed is expected
    assert fragments[0].extracted_facts[0].confirmed is expected
    assert fragments[0].extracted_facts[0].confidence == confidence
    assert fragment.completeness is None
    assert fragment.review_required is True


@pytest.mark.parametrize("completeness,review", [(0.8999, True), (0.9, False)])
def test_independent_completeness_threshold(completeness, review):
    fragment = drawing(completeness=completeness, mechanisms={"independent_reference": completeness})
    apply_drawing_quality(fragment)
    assert fragment.review_required is review


def test_legacy_structure_score_is_not_accepted_as_completeness():
    fragment = drawing(completeness=1.0)
    apply_drawing_quality(fragment)
    assert fragment.completeness is None
    assert fragment.review_required


def test_dev_value_cannot_be_confirmed_even_at_high_confidence():
    fragment = drawing(confidence=1.0, completeness=1.0, mechanisms={"reference": 1})
    fragment.content.structured_drawing_fields[0].value = "20 (dev)"
    apply_drawing_quality(fragment)
    assert not fragment.content.structured_drawing_fields[0].confirmed
    assert "SYNTHETIC_DRAWING_VALUE" in fragment.warnings


def test_unknown_completeness_and_quality_survive_database_roundtrip(db_session):
    from core.db.models import ChunkDB, DocumentDB
    db_session.add(DocumentDB(id="doc-1", source_path="source.pdf"))
    fragment = drawing()
    fragment.missing_regions = [{"page": 1, "bbox": [0, 0, 0.5, 0.5], "reason": "unread"}]
    finalize_fragments([fragment], "doc-1", "file-1")
    db_session.add(ChunkDB(**fragment_to_row(fragment, "doc-1")))
    db_session.commit()
    restored = row_to_fragment(db_session.get(ChunkDB, fragment.fragment_id))
    assert restored.model_dump() == fragment.model_dump()


def test_source_coordinates_survive_database_roundtrip(db_session):
    from core.db.models import ChunkDB, DocumentDB
    db_session.add(DocumentDB(id="doc-1", source_path="source.pdf"))
    fragment = normalize(ParsedBlock(type="text", text=" 😀\n А ", page=1,
                                    section_path=["Глава", "Пункт"]))[0]
    db_session.add(ChunkDB(**fragment_to_row(fragment, "doc-1")))
    db_session.commit()
    assert row_to_fragment(db_session.get(ChunkDB, fragment.fragment_id)).model_dump() == fragment.model_dump()


def test_old_drawing_rows_are_explicitly_unknown():
    row = SimpleNamespace(**fragment_to_row(drawing(completeness=1.0), "doc-1"))
    row.extra_data = {}
    row.id = "old-drawing"
    restored = row_to_fragment(row)
    assert restored.completeness is None
    assert restored.review_required


def test_invalid_source_ranges_are_rejected():
    with pytest.raises(ValidationError):
        SourceSpan(source_unit_id="a", page=1, char_start=5, char_end=2)
    with pytest.raises(ValidationError):
        Position(page=1, page_start=3, page_end=1)


def test_csv_over_5000_rows_is_not_truncated():
    from core.ladder.strategies.tabular import SpreadsheetStrategy
    data = "id;value\n" + "".join(f"{i};value-{i}\n" for i in range(5100))
    result = SpreadsheetStrategy()._from_csv(data.encode())[0]
    assert len(result.table_data["rows"]) == 5100
    assert result.table_data["rows"][-1] == ["5099", "value-5099"]


def test_xlsx_over_5000_rows_is_not_truncated():
    import io
    import openpyxl
    from core.ladder.strategies.tabular import SpreadsheetStrategy
    book = openpyxl.Workbook()
    book.active.append(["id", "value"])
    for i in range(5100):
        book.active.append([i, f"value-{i}"])
    data = io.BytesIO()
    book.save(data)
    book.close()
    result = SpreadsheetStrategy()._from_xlsx(data.getvalue())[0]
    assert len(result.table_data["rows"]) == 5100
    assert result.table_data["rows"][-1] == ["5099", "value-5099"]


def test_table_resource_limit_is_an_error_not_partial_success(monkeypatch):
    from core.config import settings
    from core.ladder.strategies.tabular import SpreadsheetStrategy
    monkeypatch.setattr(settings, "SPREADSHEET_MAX_CELLS", 3)
    with pytest.raises(ValueError, match="SPREADSHEET_MAX_CELLS"):
        SpreadsheetStrategy()._from_csv(b"a;b\n1;2\n")


def test_markdown_extra_cell_is_preserved():
    from core.ladder.strategies.native_text import PlainTextStrategy
    table, _ = PlainTextStrategy._read_table(["|A|B|", "|---|---|", "|1|2|3|"], 0)
    assert table["rows"] == [["1", "2", "3"]]


def test_table_read_from_drawing_has_no_assumed_confidence():
    fragment = drawing(completeness=1.0, mechanisms={"reference": 1})
    fragment.content.structured_payload = {"tables": [{"rows": [["part", "2"]]}]}
    apply_drawing_quality(fragment)
    assert fragment.content.structured_payload["tables"][0]["confirmed"] is False
    assert fragment.review_required


def test_docx_tables_keep_document_order_and_their_own_section():
    import io
    from docx import Document
    from core.ladder.context import DocumentContext
    from core.ladder.strategies.native_text import DocxStrategy
    book = Document()
    book.add_heading("Глава", level=1)
    book.add_heading("Пункт А", level=2)
    table = book.add_table(rows=2, cols=1)
    table.cell(0, 0).text = "Имя"
    table.cell(1, 0).text = "Данные"
    book.add_heading("Пункт Б", level=2)
    book.add_paragraph("После таблицы")
    data = io.BytesIO()
    book.save(data)
    result = DocxStrategy().run(DocumentContext(uri="a.docx", file_type="docx", data=data.getvalue()))
    assert [b.type for b in result.blocks] == ["text", "text", "table", "text", "text"]
    assert result.blocks[2].section_path == ["Глава", "Пункт А"]
    assert result.blocks[-1].section_path == ["Глава", "Пункт Б"]


def test_markdown_headers_are_boundaries_without_blank_lines():
    from core.ladder.context import DocumentContext
    from core.ladder.strategies.native_text import PlainTextStrategy
    data = "# Глава\n## Пункт\nСодержимое\n# Другая\nКонец".encode()
    result = PlainTextStrategy().run(DocumentContext(uri="a.md", file_type="md", data=data))
    assert [b.text for b in result.blocks] == ["# Глава", "## Пункт", "Содержимое", "# Другая", "Конец"]
    assert result.blocks[2].section_path == ["Глава", "Пункт"]
    assert result.blocks[-1].section_path == ["Другая"]


def test_ods_repeat_counts_are_not_silently_truncated():
    import io
    import zipfile
    from core.ladder.strategies.tabular import SpreadsheetStrategy
    xml = '''<office:document-content xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0"
     xmlns:table="urn:oasis:names:tc:opendocument:xmlns:table:1.0"
     xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0">
     <table:table table:name="Лист"><table:table-row><table:table-cell><text:p>A</text:p></table:table-cell></table:table-row>
     <table:table-row table:number-rows-repeated="5100"><table:table-cell table:number-columns-repeated="70"><text:p>X</text:p></table:table-cell></table:table-row>
     </table:table></office:document-content>'''
    data = io.BytesIO()
    with zipfile.ZipFile(data, "w") as archive:
        archive.writestr("content.xml", xml)
    result = SpreadsheetStrategy()._from_ods(data.getvalue())[0]
    assert len(result.table_data["rows"]) == 5100
    assert len(result.table_data["rows"][-1]) == 70


def test_rag_profile_matches_agreed_parameters():
    from pathlib import Path
    import yaml
    from core.processing_contract import RAG_EMBEDDING_PROFILE
    profile = yaml.safe_load((Path(__file__).resolve().parents[2] / "config/rag-consumer-profile.yaml").read_text(encoding="utf-8"))
    assert profile["embedding"]["revision"] == RAG_EMBEDDING_PROFILE["revision"]
    assert profile["chunking"]["target_tokens"] == 512
    assert profile["chunking"]["max_tokens"] == 768
    assert profile["chunking"]["overlap_ratio"] == 0.15


def test_geometry_and_confidence_do_not_change_source_identity():
    original = ParsedBlock(type="drawing", page=1, drawing_fields=[{
        "category": "size", "value": "20", "confidence": 0.9,
        "bbox": [0.1, 0.1, 0.2, 0.2],
    }])
    changed = original.model_copy(deep=True)
    changed.bbox = [0.2, 0.2, 0.3, 0.3]
    changed.drawing_fields[0]["confidence"] = 0.95
    changed.drawing_fields[0]["bbox"] = [0.2, 0.2, 0.3, 0.3]
    one, two = normalize(original)[0], normalize(changed)[0]
    assert one.provenance.source_unit_id == two.provenance.source_unit_id
    assert one.fragment_id == two.fragment_id
