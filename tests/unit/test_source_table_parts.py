"""Physical source gold must survive transport parts and archive replay."""

import io
import json
import zipfile

import pytest

from core.ladder.strategies.tabular import SpreadsheetStrategy, _coordinate
from core.models.contract import Fragment, TableData
from core.models.parse_result import ParseResult
from core.normalizer.normalizer import TextNormalizer
from core.normalizer.table_parts import TABLE_MAX_ITEMS, _digest, _items, table_parts


@pytest.fixture(autouse=True)
def structural_v2(monkeypatch):
    """Эти проверки — про профиль structural-rag-v2 (флаг STRUCTURAL_RAG_V2)."""
    from core.config import settings
    monkeypatch.setattr(settings, "STRUCTURAL_RAG_V2", True)

DATA_ROWS = {"ods": 300, "xlsx": 500}


def source_fixture(kind):
    if kind == "xlsx":
        import openpyxl
        book = openpyxl.Workbook()
        sheet = book.active
        sheet.title = "Source"
        sheet.append(["Title"])
        sheet.merge_cells(start_row=1, start_column=1, end_row=1, end_column=20)
        sheet.append([f"Column {i}" for i in range(20)])
        for row in range(DATA_ROWS[kind]):
            for column in range(1, 21):
                sheet.cell(row=row + 4, column=column,
                           value="Group" if column == 1 else f"Value {row}:{column}")
        sheet.merge_cells(start_row=4, start_column=1, end_row=DATA_ROWS[kind] + 3, end_column=1)
        stream = io.BytesIO()
        book.save(stream)
        book.close()
        return SpreadsheetStrategy()._from_xlsx(stream.getvalue())
    body = ['''<table:table-row><table:table-cell table:number-columns-spanned="20"><text:p>Title</text:p></table:table-cell>
      <table:covered-table-cell table:number-columns-repeated="19"/></table:table-row>
      <table:table-header-rows>''']
    for level in range(2):
        body.append('<table:table-row>' + ''.join(
            f'<table:table-cell><text:p>Header {level}:{i}</text:p></table:table-cell>'
            for i in range(20)) + '</table:table-row>')
    body.append('</table:table-header-rows><table:table-row><table:table-cell/></table:table-row>')
    for row in range(300):
        anchor = ('<table:table-cell table:number-rows-spanned="300"><text:p>Group</text:p></table:table-cell>'
                  if row == 0 else '<table:covered-table-cell/>')
        body.append('<table:table-row>' + anchor + ''.join(
            f'<table:table-cell><text:p>Value {row}:{i}</text:p></table:table-cell>'
            for i in range(1, 20)) + '</table:table-row>')
    xml = '''<office:document-content xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0"
      xmlns:table="urn:oasis:names:tc:opendocument:xmlns:table:1.0"
      xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0"><table:table table:name="Source">'''
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr("content.xml", xml + ''.join(body) + '</table:table></office:document-content>')
    return SpreadsheetStrategy()._from_ods(stream.getvalue())


@pytest.mark.parametrize("kind", ["ods", "xlsx"])
def test_large_source_roundtrips_parts_with_cross_boundary_merge(kind):
    blocks = source_fixture(kind)
    result = ParseResult(blocks=blocks, parser_name="spreadsheet", source_kind="spreadsheet", page_count=1)
    normalizer = TextNormalizer()
    fragments = normalizer.normalize(result, {"doc_id": "doc", "s3_fileid": "source"})
    assert len(fragments) > 1
    tables = [fragment.content.table_data.model_dump() for fragment in fragments]
    sources = [table["source_structure"] for table in tables]
    original = TableData(**blocks[0].table_data).model_dump(exclude={"table_id", "row_start", "row_end"})
    assert len(original["rows"]) == DATA_ROWS[kind]
    assert _items(original) > TABLE_MAX_ITEMS
    assert {source["whole_table_digest"] for source in sources} == {_digest(original)}
    assert [source["part_index"] for source in sources] == list(range(len(tables)))
    assert all(source["part_count"] == len(tables) for source in sources)
    assert all(_items(table) <= TABLE_MAX_ITEMS for table in tables)
    assert [row for source in sources for row in source["rows"]] == original["source_structure"]["rows"]
    assert [row for table in tables for row in table["rows"]] == original["rows"]
    assert [row for source in sources for row in source["data_rows"]] == original["source_structure"]["data_rows"]
    assert len({fragment.provenance.source_unit_id for fragment in fragments}) == 1
    # Каждая часть — свой table_id: первая с id целой таблицы, далее :part-N.
    unit = fragments[0].provenance.source_unit_id
    assert [table["table_id"] for table in tables] == [unit] + [
        f"{unit}:part-{index + 1}" for index in range(1, len(tables))]
    assert any(source["merge_anchors"] for source in sources[1:])
    assert all(source["merges"] == original["source_structure"]["merges"] for source in sources)
    assert all(left["row_end"] == right["row_start"] for left, right in zip(tables, tables[1:]))
    replay = [Fragment.model_validate_json(fragment.model_dump_json()) for fragment in fragments]
    assert [item.model_dump() for item in replay] == [item.model_dump() for item in fragments]
    again = normalizer.normalize(result, {"doc_id": "doc", "s3_fileid": "source"})
    assert [item.fragment_id for item in again] == [item.fragment_id for item in fragments]


def test_oversized_single_source_row_is_explicit_error():
    cells = [{"coordinate": _coordinate(1, i), "row": 1, "column": i, "display": "x", "raw_value": "x"}
             for i in range(1, 20001)]
    table = TableData(table_id="t", headers=["Header"], rows=[["x"]], source_structure={
        "schema": "source-table-v1", "rows": [{"row": 1, "cells": cells}], "data_rows": [1], "merges": []})
    with pytest.raises(ValueError, match="TABLE_RESOURCE_LIMIT"):
        table_parts(table)


def test_table_part_count_has_explicit_resource_ceiling(monkeypatch):
    import core.normalizer.table_parts as parts_module
    monkeypatch.setattr(parts_module, "TABLE_MAX_PARTS", 1)
    block = source_fixture("ods")[0]
    with pytest.raises(ValueError, match="fragment ceiling"):
        parts_module.table_parts(TableData(**{**block.table_data, "table_id": "t"}))


def test_full_http_envelope_is_measured_before_return(monkeypatch):
    from pydantic import BaseModel
    from ingest import response_budget
    class Envelope(BaseModel):
        documents: list[str]
        status: str = "completed"
    model = Envelope(documents=["я" * 60, "я" * 60])
    full = response_budget.bounded_response(model)
    monkeypatch.setattr(response_budget, "MAX_RESPONSE_BYTES", len(full.body) - 1)
    blocked = response_budget.bounded_response(model)
    assert blocked.status_code == 413
    body = json.loads(blocked.body)
    assert body["status"] == "error" and body["documents"] == []
    assert "PARSER_RESPONSE_RESOURCE_LIMIT" in body["error"]["message"]
    monkeypatch.setattr(response_budget, "MAX_RESPONSE_BYTES", len(full.body))
    assert response_budget.bounded_response(model).body == full.body


def test_callback_admission_measures_actual_ascii_wire_before_provider(monkeypatch):
    from core import result_publisher
    sent = []
    class Callback:
        def get_name(self):
            return "callback"
        def send(self, payload):
            sent.append(payload)
            return True
    monkeypatch.setattr(result_publisher, "MAX_RESPONSE_BYTES", 2000)
    monkeypatch.setattr(result_publisher, "get_delivery_provider", lambda: Callback())
    assert result_publisher.publish_result({"request_id": "r", "event_id": "e", "operation": "create",
                                            "status": "completed", "documents": ["я" * 1000]})
    assert len(sent) == 1
    assert sent[0]["request_id"] == "r" and sent[0]["event_id"] == "e"
    assert sent[0]["status"] == "error" and sent[0]["documents"] == []
    assert len(json.dumps(sent[0]).encode("utf-8")) <= 2000


def test_table_parts_require_table_id():
    block = source_fixture("xlsx")[0]
    table = TableData(**{**block.table_data, "table_id": None})
    with pytest.raises(ValueError, match="table parts require a table_id"):
        table_parts(table)


def test_payload_tables_follow_part_table_id(monkeypatch):
    """Вложенные таблицы части получают тот же table_id, что и сама часть."""
    import core.normalizer.normalizer as normalizer_module
    original = normalizer_module.attach_source

    def attach(fragment, block, unit_id, **kwargs):
        fragment.content.structured_payload = {"tables": [{"rows": []}]}
        original(fragment, block, unit_id, **kwargs)

    monkeypatch.setattr(normalizer_module, "attach_source", attach)
    blocks = source_fixture("xlsx")
    result = ParseResult(blocks=blocks, parser_name="spreadsheet", source_kind="spreadsheet", page_count=1)
    fragments = TextNormalizer().normalize(result, {"doc_id": "doc", "s3_fileid": "source"})
    unit = fragments[0].provenance.source_unit_id
    assert len(fragments) > 1
    for number, fragment in enumerate(fragments[1:], 2):
        assert fragment.content.table_data.table_id == f"{unit}:part-{number}"
        assert [t["table_id"] for t in fragment.content.structured_payload["tables"]] == [
            f"{unit}:part-{number}"]


def test_xlsx_merges_are_clamped_to_data_grid():
    import openpyxl
    book = openpyxl.Workbook()
    sheet = book.active
    sheet.append(["a", "b"])
    sheet.append(["c", "d"])
    sheet.merge_cells(start_row=2, start_column=2, end_row=50, end_column=30)
    stream = io.BytesIO()
    book.save(stream)
    book.close()
    block = SpreadsheetStrategy()._from_xlsx(stream.getvalue())[0]
    merges = block.table_data["source_structure"]["merges"]
    assert merges and all(m["row_end"] <= 2 and m["column_end"] <= 2 for m in merges)
