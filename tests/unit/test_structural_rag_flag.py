"""Флаг STRUCTURAL_RAG_V2: выключен — выдача как у structural-rag-v1, включён — v2."""

import io
import json
import zipfile
from types import SimpleNamespace

import pytest

from core.config import settings
from core.ladder.strategies.tabular import SpreadsheetStrategy
from core.models.contract import Content, Fragment, Position, Provenance
from core.models.parse_result import ParsedBlock, ParseResult
from core.normalizer import TextNormalizer
from core.processing_contract import processing_profile_version
from core.providers import mixed_script, text_layer
from core.source_contract import source_unit_ids


@pytest.fixture(params=[False, True], ids=["v1", "v2"])
def v2(request, monkeypatch):
    monkeypatch.setattr(settings, "STRUCTURAL_RAG_V2", request.param)
    return request.param


def test_default_is_v1():
    assert settings.STRUCTURAL_RAG_V2 is False
    assert processing_profile_version() == "structural-rag-v1"


def test_profile_version_follows_flag(v2):
    assert processing_profile_version() == ("structural-rag-v2" if v2 else "structural-rag-v1")


def _line(*spans):
    page = SimpleNamespace(rotation=0, get_text=lambda kind: {"blocks": [{
        "type": 0, "lines": [{"spans": list(spans), "bbox": [10, 10, 180, 20], "dir": [1, 0]}]}]})
    return text_layer._lines_from_page(page, 1, 200, 100)[0]


def _span(text, left, right):
    return {"text": text, "bbox": [left, 10, right, 20], "size": 10, "font": "Example", "color": 0}


def test_span_spacing_only_under_flag(v2):
    line = _line(_span("Поставка", 10, 50), _span("разрешена", 54, 110))
    if v2:
        assert line.text == "Поставка разрешена" and line.text_edits
    else:
        # v1: куски склеиваются как есть.
        assert line.text == "Поставкаразрешена" and line.text_edits == []


def _ods(body):
    xml = f'''<office:document-content
      xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0"
      xmlns:table="urn:oasis:names:tc:opendocument:xmlns:table:1.0"
      xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0">
      <table:table table:name="Example">{body}</table:table></office:document-content>'''
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr("content.xml", xml)
    return stream.getvalue()


ODS_BODY = '''
  <table:table-row><table:table-cell><text:p>Header</text:p></table:table-cell>
    <table:table-cell><text:p>Value</text:p></table:table-cell></table:table-row>
  <table:table-row><table:table-cell><text:p>a</text:p></table:table-cell>
    <table:table-cell office:value-type="float" office:value="1"><text:p>1</text:p></table:table-cell></table:table-row>
  <table:table-row table:number-rows-repeated="1048570"><table:table-cell table:number-columns-repeated="1024"/></table:table-row>'''


def _xlsx():
    import openpyxl
    book = openpyxl.Workbook()
    book.active.title = "Empty"
    sheet = book.create_sheet("Data")
    sheet.append(["Header", "Value"])
    sheet.append(["a", 1])
    stream = io.BytesIO()
    book.save(stream)
    book.close()
    return stream.getvalue()


@pytest.mark.parametrize("kind", ["ods", "xlsx", "csv"])
def test_tabular_source_structure_only_under_flag(v2, kind):
    strategy = SpreadsheetStrategy()
    if kind == "ods":
        blocks = strategy._from_ods(_ods(ODS_BODY))
    elif kind == "xlsx":
        blocks = strategy._from_xlsx(_xlsx())
    else:
        blocks = strategy._from_csv("Header;Value\na;1\n".encode("utf-8"))
    assert len(blocks) == 1
    table = blocks[0].table_data
    assert table["headers"] == ["Header", "Value"] and table["rows"] == [["a", "1"]]
    if kind == "xlsx":
        # Пустой первый лист пропущен, номер страницы — порядковый рабочего листа.
        assert blocks[0].page == 2
    if v2:
        assert table["source_structure"]["schema"] == "source-table-v1"
    else:
        assert table == {"headers": ["Header", "Value"], "rows": [["a", "1"]], "total_row": None}


def _v1_unit_id(block, doc_id):
    import hashlib
    table = {key: block.table_data.get(key) for key in ("headers", "rows", "total_row")}
    payload = [doc_id, block.type, block.page, block.sheet_name, block.text, table, None, block.image_ref]
    digest = hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True,
                                       default=str).encode("utf-8")).hexdigest()[:24]
    return f"unit-{digest}"


def test_v1_table_fragment_has_v1_identity_and_shape(monkeypatch):
    monkeypatch.setattr(settings, "STRUCTURAL_RAG_V2", False)
    block = ParsedBlock(type="table", page=1, sheet_name="Data", confidence=1,
                        table_data={"headers": ["A"], "rows": [["1"]], "total_row": None})
    assert source_unit_ids([block], "doc")[id(block)] == _v1_unit_id(block, "doc")
    fragment = TextNormalizer().normalize(ParseResult(blocks=[block], exhaustive=True),
                                          {"doc_id": "doc", "s3_fileid": "source.xlsx"})[0]
    dumped = fragment.model_dump(mode="json")
    assert "source_structure" not in dumped["content"]["table_data"]
    assert dumped["provenance"]["processing_profile_version"] == "structural-rag-v1"
    assert "text_observations" not in block.model_dump(mode="json")


def test_large_table_split_only_under_flag(v2):
    rows = [[f"r{index}", "x", "y"] for index in range(40_000)]
    block = ParsedBlock(type="table", page=1, sheet_name="Data", confidence=1,
                        table_data={"headers": ["A", "B", "C"], "rows": rows, "total_row": None})
    result = ParseResult(blocks=[block], exhaustive=True)
    if v2:
        # v2 делит только таблицу с исходной сеткой; без неё — явная ошибка.
        with pytest.raises(ValueError, match="TABLE_RESOURCE_LIMIT"):
            TextNormalizer().normalize(result, {"doc_id": "doc", "s3_fileid": "source.xlsx"})
    else:
        fragments = TextNormalizer().normalize(result, {"doc_id": "doc", "s3_fileid": "source.xlsx"})
        assert len(fragments) == 1 and len(fragments[0].content.table_data.rows) == 40_000


def test_mixed_script_evidence_only_under_flag(v2):
    fragment = Fragment(fragment_id="source", type="text", content=Content(text="Код ККS"),
                        position=Position(page=1), confidence=1, completeness=1,
                        provenance=Provenance(method="text_layer", strategy_level=2, source="vector_pdf"))
    mixed_script.repair_fragment(fragment)
    assert fragment.content.text == "Код KKS"
    payload = fragment.content.structured_payload
    assert payload[mixed_script.PAYLOAD_KEY] == [{"before": "ККS", "after": "KKS"}]
    assert ("text_correction_evidence" in payload) is v2
