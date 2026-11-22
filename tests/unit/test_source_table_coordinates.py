"""Gold is the physical XML grid, before any display/header inference."""

import io
import zipfile

import pytest

from core.ladder.strategies.tabular import SpreadsheetStrategy
from core.models.parse_result import ParsedBlock
from core.source_contract import source_unit_ids


@pytest.fixture(autouse=True)
def structural_v2(monkeypatch):
    """Эти проверки — про профиль structural-rag-v2 (флаг STRUCTURAL_RAG_V2)."""
    from core.config import settings
    monkeypatch.setattr(settings, "STRUCTURAL_RAG_V2", True)


def ods(body):
    xml = f'''<office:document-content
      xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0"
      xmlns:table="urn:oasis:names:tc:opendocument:xmlns:table:1.0"
      xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0">
      <table:table table:name="Example">{body}</table:table>
      </office:document-content>'''
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr("content.xml", xml)
    return stream.getvalue()


def test_ods_merge_nested_text_and_blank_are_source_bound():
    data = ods('''
      <table:table-row><table:table-cell table:number-columns-spanned="4">
        <text:p>Report</text:p></table:table-cell>
        <table:covered-table-cell table:number-columns-repeated="3"/></table:table-row>
      <table:table-row><table:table-cell><text:p>Group</text:p></table:table-cell>
        <table:table-cell><text:p>Part</text:p></table:table-cell>
        <table:table-cell><text:p>Serial A</text:p></table:table-cell>
        <table:table-cell><text:p>Serial B</text:p></table:table-cell></table:table-row>
      <table:table-row><table:table-cell table:number-rows-spanned="2"><text:p>G</text:p></table:table-cell>
        <table:table-cell><text:p>Nested <text:span>part</text:span> tail</text:p></table:table-cell>
        <table:table-cell office:value-type="float" office:value="123"><text:p>123</text:p></table:table-cell>
        <table:table-cell><text:p>456</text:p></table:table-cell></table:table-row>
      <table:table-row><table:covered-table-cell/><table:table-cell/>
        <table:table-cell><text:p>124</text:p></table:table-cell>
        <table:table-cell><text:p>457</text:p></table:table-cell></table:table-row>
      <table:table-row><table:table-cell/><table:table-cell/>
        <table:table-cell><text:p>125</text:p></table:table-cell></table:table-row>''')
    table = SpreadsheetStrategy()._from_ods(data)[0].table_data
    assert table["headers"] == ["Group", "Part", "Serial A", "Serial B"]
    assert table["rows"][0] == ["G", "Nested part tail", "123", "456"]
    assert table["rows"][1] == ["", "", "124", "457"]
    source = table["source_structure"]
    assert source["title_rows"] == [1]
    assert source["header_rows"] == [2]
    assert source["data_rows"] == [3, 4, 5]
    covered = source["rows"][3]["cells"][0]
    assert covered["coordinate"] == "A4"
    assert covered["merge_anchor"] == "A3"
    assert covered["effective_display"] == "G"
    assert source["rows"][4]["cells"][0]["effective_display"] == ""
    assert source["rows"][2]["cells"][2]["raw_value"] == "123"


def test_source_identity_includes_raw_table_metadata():
    one = ParsedBlock(type="table", page=1, table_data={
        "headers": ["A"], "rows": [["same"]],
        "source_structure": {"raw_value": "1"},
    })
    two = one.model_copy(deep=True)
    two.table_data["source_structure"]["raw_value"] = "2"
    assert source_unit_ids([one], "doc")[id(one)] != source_unit_ids([two], "doc")[id(two)]


def test_ods_sixty_pairs_stay_in_columns_e_f_across_group_boundary():
    # Gold is generated from source positions, with two independent merged groups.
    body = ['''<table:table-header-rows><table:table-row>
      <table:table-cell table:number-columns-repeated="6"><text:p>Header</text:p></table:table-cell>
      </table:table-row></table:table-header-rows>''']
    for index in range(60):
        group_start = index in (0, 40)
        span = 40 if index == 0 else 20
        group = "First" if index < 40 else "Second"
        middle = (f'''<table:table-cell table:number-rows-spanned="{span}"><text:p>{group}</text:p></table:table-cell>
                      <table:table-cell table:number-rows-spanned="{span}"><text:p>{group} item</text:p></table:table-cell>'''
                  if group_start else '<table:covered-table-cell table:number-columns-repeated="2"/>')
        body.append(f'''<table:table-row>
          <table:table-cell table:number-columns-repeated="2"><text:p>{index + 1}</text:p></table:table-cell>
          {middle}<table:table-cell><text:p>A{index}</text:p></table:table-cell>
          <table:table-cell><text:p>B{index}</text:p></table:table-cell></table:table-row>''')
    table = SpreadsheetStrategy()._from_ods(ods("".join(body)))[0].table_data
    assert len(table["rows"]) == 60
    assert [(row[4], row[5]) for row in table["rows"]] == [
        (f"A{i}", f"B{i}") for i in range(60)
    ]
    source = table["source_structure"]
    assert source["header_detection"] == "explicit"
    for index, row in enumerate(source["rows"][1:]):
        anchor_row = 2 if index < 40 else 42
        assert row["cells"][2]["merge_anchor"] == f"C{anchor_row}"
        assert row["cells"][3]["merge_anchor"] == f"D{anchor_row}"
        assert row["cells"][4]["coordinate"] == f"E{index + 2}"
        assert row["cells"][5]["coordinate"] == f"F{index + 2}"


def test_ods_blank_repeated_rows_keep_physical_coordinates():
    table = SpreadsheetStrategy()._from_ods(ods('''
      <table:table-row><table:table-cell><text:p>Header</text:p></table:table-cell></table:table-row>
      <table:table-row table:number-rows-repeated="2"><table:table-cell/></table:table-row>
      <table:table-row><table:table-cell office:value-type="float" office:value="9"
         table:formula="of:=3*3"><text:p>9</text:p></table:table-cell></table:table-row>
    '''))[0].table_data
    assert table["rows"] == [["9"]]
    source = table["source_structure"]
    assert source["data_rows"] == [4]
    assert source["rows"][1]["cells"][0]["row_repeat"] == 2
    assert source["rows"][3]["cells"][0]["coordinate"] == "A4"
    assert source["rows"][3]["cells"][0]["cached_value"] == "9"


def test_xlsx_raw_serial_survives_date_style_and_formula_cache():
    import openpyxl
    book = openpyxl.Workbook()
    book.active.append(["Certificate", "Calculated"])
    book.active.append([256852, "=1+1"])
    book.active["A2"].number_format = "dd.mm.yyyy"
    hidden = book.create_sheet("Hidden")
    hidden.sheet_state = "hidden"
    stream = io.BytesIO()
    book.save(stream)
    book.close()
    # A computed cache is a source fact, independent of openpyxl coercion.
    patched = io.BytesIO()
    with zipfile.ZipFile(stream) as original, zipfile.ZipFile(patched, "w") as output:
        for name in original.namelist():
            content = original.read(name)
            if name == "xl/worksheets/sheet1.xml":
                content = content.replace(b"<f>1+1</f><v></v>", b"<f>1+1</f><v>2</v>")
            output.writestr(name, content)
    tables = SpreadsheetStrategy()._from_xlsx(patched.getvalue())
    source = tables[0].table_data["source_structure"]
    cell = source["rows"][1]["cells"][0]
    assert cell["coordinate"] == "A2"
    assert cell["raw_value"] == "256852"
    assert cell["number_format"] == "dd.mm.yyyy"
    assert cell["style_id"] != "0"
    formula = source["rows"][1]["cells"][1]
    assert formula["formula"] == "1+1"
    assert formula["cached_value"] == "2"
    # Пустой скрытый лист фрагмента не даёт, как в v1.
    assert [table.sheet_name for table in tables] == ["Sheet"]


def xlsx_fixture(*, chartsheet=False):
    import openpyxl
    from openpyxl.chart import BarChart, Reference
    book = openpyxl.Workbook()
    book.active.title = "Data"
    book.active.append(["Serial"])
    book.active.append([123])
    if chartsheet:
        chart = BarChart()
        chart.add_data(Reference(book.active, min_col=1, min_row=2, max_row=2))
        sheet = book.create_chartsheet("Chart", index=0)
        sheet.add_chart(chart)
    stream = io.BytesIO()
    book.save(stream)
    book.close()
    return stream.getvalue()


def test_chartsheet_before_worksheet_does_not_rebind_source_xml():
    # Номер страницы — порядковый номер рабочего листа, как в v1: лист
    # диаграммы его не сдвигает, а XML ячеек остаётся своего листа.
    from core.ladder.context import DocumentContext
    from core.models.parse_result import ParseResult
    from ingest.tasks import page_report

    data = xlsx_fixture(chartsheet=True)
    table = SpreadsheetStrategy()._from_xlsx(data)[0]
    assert table.sheet_name == "Data"
    assert table.page == 1
    assert table.table_data["rows"] == [["123"]]
    cell = table.table_data["source_structure"]["rows"][1]["cells"][0]
    assert cell["coordinate"] == "A2"
    assert cell["raw_value"] == "123"
    context = DocumentContext("example.xlsx", "xlsx", data=data, storage=object())
    assert context.source_page_count == 2
    report = page_report(ParseResult(blocks=[table], exhaustive=True, page_count=1,
                                    source_page_count=context.source_page_count))
    assert report["processed_pages"] == [1]
    assert report["unknown_pages"] == [2]
    assert report["coverage_unknown"] is True


def test_xlsx_understated_dimension_does_not_hide_xml_cells():
    output = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(xlsx_fixture())) as source, zipfile.ZipFile(output, "w") as dest:
        for name in source.namelist():
            content = source.read(name)
            if name == "xl/worksheets/sheet1.xml":
                content = content.replace(b'<dimension ref="A1:A2"/>', b'<dimension ref="A1"/>')
            dest.writestr(name, content)
    table = SpreadsheetStrategy()._from_xlsx(output.getvalue())[0].table_data
    assert table["rows"] == [["123"]]
    assert table["source_structure"]["rows"][1]["cells"][0]["raw_value"] == "123"
    assert table["source_structure"]["coverage"] == "complete"


def test_xlsx_expansion_cap_is_checked_before_any_member_is_opened(monkeypatch):
    from core.config import settings
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("xl/workbook.xml", " " * 20000)
    monkeypatch.setattr(settings, "QG_MAX_SIZE_BYTES", 10000)
    def forbidden_open(*args, **kwargs):
        raise AssertionError("entry opened before expansion preflight")
    monkeypatch.setattr(zipfile.ZipFile, "open", forbidden_open)
    with pytest.raises(ValueError, match="ZIP.*budget"):
        SpreadsheetStrategy()._from_xlsx(output.getvalue())


def test_xlsx_xml_cell_budget_stops_before_openpyxl(monkeypatch):
    from core.config import settings
    from core.ladder.strategies import tabular
    data = xlsx_fixture()
    monkeypatch.setattr(settings, "SPREADSHEET_MAX_CELLS", 1)
    def forbidden_load(*args, **kwargs):
        raise AssertionError("openpyxl called before source cell preflight")
    monkeypatch.setattr(tabular.openpyxl, "load_workbook", forbidden_load)
    with pytest.raises(ValueError, match="SPREADSHEET_MAX_CELLS"):
        SpreadsheetStrategy()._from_xlsx(data)


def test_ods_expansion_cap_is_checked_before_any_member_is_opened(monkeypatch):
    from core.config import settings

    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("content.xml", " " * 20000)
    monkeypatch.setattr(settings, "QG_MAX_SIZE_BYTES", 10000)

    def forbidden_open(*args, **kwargs):
        raise AssertionError("ODS member opened before expansion preflight")

    monkeypatch.setattr(zipfile.ZipFile, "open", forbidden_open)
    with pytest.raises(ValueError, match="ODS ZIP.*budget"):
        SpreadsheetStrategy()._from_ods(output.getvalue())


def test_duplicate_ods_content_member_is_rejected_before_reading(monkeypatch):
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr("content.xml", "first")
        with pytest.warns(UserWarning, match="Duplicate name"):
            archive.writestr("content.xml", "second")

    def forbidden_open(*args, **kwargs):
        raise AssertionError("ambiguous ODS archive member was read")

    monkeypatch.setattr(zipfile.ZipFile, "open", forbidden_open)
    with pytest.raises(ValueError, match="Duplicate ODS ZIP member"):
        SpreadsheetStrategy()._from_ods(output.getvalue())


def test_ods_trailing_million_empty_rows_do_not_count_toward_budget(monkeypatch):
    from core.config import settings
    monkeypatch.setattr(settings, "SPREADSHEET_MAX_CELLS", 1000)
    table = SpreadsheetStrategy()._from_ods(ods('''
      <table:table-row><table:table-cell><text:p>Header</text:p></table:table-cell>
        <table:table-cell table:number-columns-repeated="1024"/></table:table-row>
      <table:table-row><table:table-cell><text:p>1</text:p></table:table-cell>
        <table:table-cell table:number-columns-repeated="1024"/></table:table-row>
      <table:table-row table:number-rows-repeated="1048570">
        <table:table-cell table:number-columns-repeated="1025"/></table:table-row>
    '''))[0].table_data
    assert table["headers"] == ["Header"] and table["rows"] == [["1"]]
    source = table["source_structure"]
    assert [row["row"] for row in source["rows"]] == [1, 2]
    assert all(len(row["cells"]) == 1 for row in source["rows"])


def test_ods_empty_sheet_gives_no_fragment():
    assert SpreadsheetStrategy()._from_ods(ods('''
      <table:table-row table:number-rows-repeated="1048576">
        <table:table-cell table:number-columns-repeated="1024"/></table:table-row>''')) == []


def _patched_sheet(data, replace):
    output = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(data)) as source, zipfile.ZipFile(output, "w") as dest:
        for name in source.namelist():
            content = source.read(name)
            if name == "xl/worksheets/sheet1.xml":
                content = replace(content)
            dest.writestr(name, content)
    return output.getvalue()


def test_xlsx_full_sheet_dimension_and_far_merge_do_not_count_toward_budget(monkeypatch):
    from core.config import settings
    monkeypatch.setattr(settings, "SPREADSHEET_MAX_CELLS", 1000)

    def huge(content):
        content = content.replace(b'<dimension ref="A1:A2"/>', b'<dimension ref="A1:XFD1048576"/>')
        # Объединение далеко за данными и объединение от данных до конца листа.
        return content.replace(b"</sheetData>", b'</sheetData><mergeCells count="2">'
                               b'<mergeCell ref="Z500000:XFD1048576"/><mergeCell ref="A2:XFD1048576"/>'
                               b'</mergeCells>')

    data = _patched_sheet(xlsx_fixture(), huge)
    assert b'<dimension ref="A1:XFD1048576"/>' in data
    table = SpreadsheetStrategy()._from_xlsx(data)[0].table_data
    assert table["headers"] == ["Serial"] and table["rows"] == [["123"]]
    source = table["source_structure"]
    assert [len(row["cells"]) for row in source["rows"]] == [1, 1]
    assert [merge["anchor"] for merge in source["merges"]] == ["A2"]


def test_xlsx_styled_empty_cells_do_not_extend_the_grid(monkeypatch):
    from core.config import settings
    monkeypatch.setattr(settings, "SPREADSHEET_MAX_CELLS", 1000)
    styled = b"".join(b'<row r="%d"><c r="A%d" s="1"/><c r="XFD%d" s="1"/></row>' % (n, n, n)
                      for n in range(3, 2003))
    data = _patched_sheet(xlsx_fixture(), lambda c: c.replace(b"</sheetData>", styled + b"</sheetData>"))
    table = SpreadsheetStrategy()._from_xlsx(data)[0].table_data
    assert table["rows"] == [["123"]]
    assert len(table["source_structure"]["rows"]) == 2


def test_xlsx_empty_worksheet_gives_no_fragment():
    import openpyxl
    book = openpyxl.Workbook()
    book.active.title = "Empty"
    data = book.create_sheet("Data")
    data.append(["Serial"])
    data.append([7])
    stream = io.BytesIO()
    book.save(stream)
    book.close()
    tables = SpreadsheetStrategy()._from_xlsx(stream.getvalue())
    assert [(table.sheet_name, table.page) for table in tables] == [("Data", 2)]


@pytest.mark.parametrize("text", ["", "   \n\n   \n\t\n"])
def test_whitespace_only_csv_gives_no_fragment(text):
    assert SpreadsheetStrategy()._from_csv(text.encode("utf-8")) == []
