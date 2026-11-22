"""
Уровень 3 — табличный разбор жёсткого формата.

XLSX и CSV не нужно ни распознавать, ни размечать: структура задана самим
форматом. Лист книги — самостоятельная единица, его имя уезжает в
`position.sheet`, как и требует контракт.

Формулы XLSX читаются по вычисленному значению: в индекс должно попадать
то, что видит человек, а не текст формулы. Когда книга сохранена без
кэша значений, вместо числа придёт формула — это отмечается в логе.
"""

from __future__ import annotations

import csv
import io
import logging
import posixpath
import zipfile
from typing import List

from defusedxml import ElementTree

from core import filetypes
from core.config import settings
from core.ladder.base import Strategy
from core.ladder.context import DocumentContext
from core.models.parse_result import ParsedBlock, ParseResult
from core.processing_contract import structural_v2

logger = logging.getLogger(__name__)

try:
    import openpyxl
    OPENPYXL_AVAILABLE = True
except ImportError:  # pragma: no cover
    openpyxl = None
    OPENPYXL_AVAILABLE = False

try:
    # Excel 97-2003 (BIFF) openpyxl не открывает вовсе — для него отдельная
    # библиотека. Счета, УПД и акты в бухгалтерских выгрузках до сих пор
    # приходят именно в этом формате.
    import xlrd
    XLRD_AVAILABLE = True
except ImportError:  # pragma: no cover
    xlrd = None
    XLRD_AVAILABLE = False

def _check_budget(cells: int) -> None:
    if cells > settings.SPREADSHEET_MAX_CELLS:
        raise ValueError(
            "Таблица превышает SPREADSHEET_MAX_CELLS; результат не усечён, "
            "нужен больший ресурсный бюджет"
        )


def _check_zip_budget(data: bytes, archive: zipfile.ZipFile, format_name: str) -> int:
    """Reject oversized or ambiguous containers before opening a source member."""
    limit = settings.QG_MAX_SIZE_BYTES
    entries = archive.infolist()
    if (len(data) > limit or sum(info.file_size for info in entries) > limit or
            any(info.file_size > limit or info.compress_size > limit for info in entries)):
        raise ValueError(f"{format_name} ZIP expansion exceeds byte budget")
    if len({info.filename for info in entries}) != len(entries):
        raise ValueError(f"Duplicate {format_name} ZIP member")
    return limit


class SpreadsheetStrategy(Strategy):
    """XLSX и CSV -> таблицы без единой догадки о структуре."""

    level = 3
    name = "spreadsheet"
    method = "tabular_extraction"
    exhaustive = True

    def applicable(self, context: DocumentContext) -> bool:
        if context.kind != filetypes.KIND_SPREADSHEET:
            return False
        if context.file_type == "csv":
            return True
        if context.file_type == "xls":
            return XLRD_AVAILABLE
        if context.file_type == "ods":
            return True
        return OPENPYXL_AVAILABLE

    def run(self, context: DocumentContext) -> ParseResult:
        if context.file_type == "csv":
            blocks = self._from_csv(context.data)
            source_kind = "csv"
        elif context.file_type == "xls":
            blocks = self._from_xls(context.data)
            source_kind = "spreadsheet"
        elif context.file_type == "ods":
            blocks = self._from_ods(context.data)
            source_kind = "spreadsheet"
        else:
            blocks = self._from_xlsx(context.data)
            source_kind = "spreadsheet"

        if not blocks:
            raise ValueError("В таблице нет данных")

        return self.build_result(blocks, source_kind=source_kind, page_count=len(blocks))

    # ------------------------------------------------------------------ CSV
    def _from_csv(self, data: bytes) -> List[ParsedBlock]:
        if not structural_v2():
            return self._from_csv_v1(data)
        text = self._decode(data)
        dialect = self._dialect(text)
        physical = []
        cells = 0
        for row_no, row in enumerate(csv.reader(io.StringIO(text), dialect), 1):
            cells += max(1, len(row))
            _check_budget(cells)
            physical.append({"row": row_no, "cells": [
                {"coordinate": _coordinate(row_no, col_no), "row": row_no,
                 "column": col_no, "raw_value": cell, "value_type": "string",
                 "display": cell.strip(), "effective_display": cell.strip()}
                for col_no, cell in enumerate(row, 1)
            ]})
        table_data = _source_table(physical, [], format_name="csv", visibility="not_applicable")
        # Файл из одних пробелов и пустых строк — не таблица, фрагмента нет.
        if _blank(table_data):
            return []
        return [ParsedBlock(
            type="table", page=1, confidence=1.0, method=self.method,
            table_data=table_data,
        )]

    def _from_csv_v1(self, data: bytes) -> List[ParsedBlock]:
        """Профиль v1: только отображаемые строки, без исходной сетки."""
        text = self._decode(data)
        dialect = self._dialect(text)
        rows = []
        cells = 0
        for row in csv.reader(io.StringIO(text), dialect):
            cells += max(1, len(row))
            _check_budget(cells)
            values = [str(cell).strip() for cell in row]
            if any(values):
                rows.append(values)
        if not rows:
            return []
        return [ParsedBlock(
            type="table", page=1, confidence=1.0, method=self.method,
            table_data={"headers": rows[0], "rows": rows[1:], "total_row": None},
        )]

    @staticmethod
    def _decode(data: bytes) -> str:
        for encoding in ("utf-8-sig", "utf-8", "cp1251"):
            try:
                return data.decode(encoding)
            except UnicodeDecodeError:
                continue
        return data.decode("utf-8", errors="replace")

    @staticmethod
    def _dialect(text: str):
        """Разделитель определяется по образцу: в русских выгрузках это `;`."""
        sample = "\n".join(text.splitlines()[:20])
        try:
            return csv.Sniffer().sniff(sample, delimiters=",;\t|")
        except csv.Error:
            logger.debug("Разделитель CSV не определён, берём запятую")
            return csv.get_dialect("excel")

    # ----------------------------------------------------------------- XLSX
    def _from_xlsx(self, data: bytes) -> List[ParsedBlock]:
        if not structural_v2():
            return self._from_xlsx_v1(data)
        # Read XML before openpyxl turns date-styled numeric identifiers into dates.
        source_sheets = _xlsx_sources(data)
        workbook = openpyxl.load_workbook(
            io.BytesIO(data), read_only=True, data_only=True
        )
        # Вторая копия — с формулами. Значение формулы хранится в файле только
        # если его сохранила программа с пересчётом; у выгрузок и файлов,
        # собранных скриптами, кэша нет, и `data_only` отдаёт пустую ячейку.
        # Тогда лучше текст формулы, чем молча пропавшее значение.
        formulas = openpyxl.load_workbook(io.BytesIO(data), read_only=True)
        blocks: List[ParsedBlock] = []
        try:
            # Номер страницы — порядковый номер рабочего листа, как в v1:
            # листы диаграмм в `worksheets` не входят и номер не сдвигают.
            for page_no, (sheet, formula_sheet) in enumerate(
                zip(workbook.worksheets, formulas.worksheets), start=1
            ):
                source = source_sheets[sheet.title]
                _check_budget(source["max_row"] * max(1, source["max_column"]))
                physical = []
                cells = 0
                bounds = {"min_row": 1, "min_col": 1, "max_row": source["max_row"],
                          "max_col": source["max_column"]}
                cached_rows = sheet.iter_rows(**bounds) if source["max_row"] else iter(())
                formula_rows = formula_sheet.iter_rows(**bounds) if source["max_row"] else iter(())
                for row_no, (cached_row, formula_row) in enumerate(zip(
                    cached_rows, formula_rows,
                ), start=1):
                    cells += max(1, len(cached_row))
                    _check_budget(cells)
                    records = []
                    for col_no, (cached, raw) in enumerate(zip(cached_row, formula_row), 1):
                        coordinate = _coordinate(row_no, col_no)
                        record = dict(source["cells"].get(coordinate, {}))
                        display = _cell_text(cached.value, raw.value)
                        record.update(coordinate=coordinate, row=row_no, column=col_no,
                                      display=display, effective_display=display,
                                      number_format=getattr(raw, "number_format", None))
                        records.append(record)
                    physical.append({"row": row_no, "cells": records})
                _apply_merges(physical, source["merges"])
                table_data = _source_table(physical, source["merges"],
                                           visibility=sheet.sheet_state, format_name="xlsx")
                # Пустой лист фрагмента не даёт, как в v1.
                if _blank(table_data):
                    continue
                blocks.append(ParsedBlock(
                    type="table", page=page_no, sheet_name=sheet.title,
                    confidence=1.0, method=self.method,
                    section_title=sheet.title,
                    table_data=table_data,
                ))
        finally:
            workbook.close()
            formulas.close()
        return blocks

    def _from_xlsx_v1(self, data: bytes) -> List[ParsedBlock]:
        """Профиль v1: значения листа через openpyxl, без исходной сетки."""
        workbook = openpyxl.load_workbook(
            io.BytesIO(data), read_only=True, data_only=True
        )
        # Вторая копия — с формулами: без кэша значений лучше текст формулы,
        # чем молча пропавшее значение.
        formulas = openpyxl.load_workbook(io.BytesIO(data), read_only=True)
        blocks: List[ParsedBlock] = []
        try:
            for page_no, (sheet, formula_sheet) in enumerate(
                zip(workbook.worksheets, formulas.worksheets), start=1
            ):
                rows = self._sheet_rows(sheet, formula_sheet)
                if not rows:
                    continue
                blocks.append(ParsedBlock(
                    type="table", page=page_no, sheet_name=sheet.title,
                    confidence=1.0, method=self.method,
                    section_title=sheet.title,
                    table_data={
                        "headers": rows[0],
                        "rows": rows[1:],
                        "total_row": None,
                    },
                ))
        finally:
            workbook.close()
            formulas.close()
        return blocks

    # ------------------------------------------------------------------ XLS
    def _from_xls(self, data: bytes) -> List[ParsedBlock]:
        """Excel 97-2003. Даты приводим к виду, который видит человек."""
        book = xlrd.open_workbook(file_contents=data)
        blocks: List[ParsedBlock] = []
        for page_no, sheet in enumerate(book.sheets(), start=1):
            rows: List[List[str]] = []
            _check_budget(sheet.nrows * max(1, sheet.ncols))
            for index in range(sheet.nrows):
                values = [
                    self._xls_cell(cell, book.datemode) for cell in sheet.row(index)
                ]
                if any(values):
                    rows.append(values)
            if not rows:
                continue
            blocks.append(ParsedBlock(
                type="table", page=page_no, sheet_name=sheet.name,
                confidence=1.0, method=self.method, section_title=sheet.name,
                table_data={"headers": rows[0], "rows": rows[1:], "total_row": None},
            ))
        return blocks

    @staticmethod
    def _xls_cell(cell, datemode) -> str:
        if cell.ctype == xlrd.XL_CELL_DATE:
            try:
                parts = xlrd.xldate_as_tuple(cell.value, datemode)
                if parts[:3] == (0, 0, 0):
                    return "{:02d}:{:02d}:{:02d}".format(*parts[3:])
                return "{2:02d}.{1:02d}.{0:04d}".format(*parts[:3])
            except Exception:  # noqa: BLE001 — битая дата не повод терять строку
                return str(cell.value)
        if cell.ctype == xlrd.XL_CELL_NUMBER:
            # Целое в xls хранится как float: «12345.0» в счёте выглядит дико.
            number = cell.value
            return str(int(number)) if float(number).is_integer() else str(number)
        if cell.ctype == xlrd.XL_CELL_BOOLEAN:
            return "да" if cell.value else "нет"
        if cell.ctype in (xlrd.XL_CELL_EMPTY, xlrd.XL_CELL_BLANK):
            return ""
        return str(cell.value).strip()

    # ------------------------------------------------------------------ ODS
    def _from_ods(self, data: bytes) -> List[ParsedBlock]:
        """
        OpenDocument — zip с content.xml. Разбираем штатным XML-парсером:
        отдельная библиотека ради одной таблицы не нужна.
        """
        if not structural_v2():
            return self._from_ods_v1(data)
        table_ns = "{urn:oasis:names:tc:opendocument:xmlns:table:1.0}"
        text_ns = "{urn:oasis:names:tc:opendocument:xmlns:text:1.0}"
        office_ns = "{urn:oasis:names:tc:opendocument:xmlns:office:1.0}"

        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            _check_zip_budget(data, archive, "ODS")
            content = archive.read("content.xml")
        root = ElementTree.fromstring(content)

        blocks: List[ParsedBlock] = []
        for page_no, table in enumerate(root.iter(f"{table_ns}table"), start=1):
            name = table.get(f"{table_ns}name") or f"Лист {page_no}"
            # Первый проход: строки как серии ячеек, повторы не разворачиваются.
            # Так хвост из пустых строк (number-rows-repeated на миллион) и
            # пустые колонки за данными не попадают в бюджет ячеек.
            source_rows = []
            for row, is_header in _ods_rows(table, table_ns):
                row_repeat = _positive(row.get(f"{table_ns}number-rows-repeated"))
                runs = []
                width = used = 0
                for cell in row:
                    if cell.tag not in (f"{table_ns}table-cell", f"{table_ns}covered-table-cell"):
                        continue
                    repeat = _positive(cell.get(f"{table_ns}number-columns-repeated"))
                    width += repeat
                    display = "\n".join(_ods_text(p, text_ns) for p in cell.iter(f"{text_ns}p"))
                    record = {
                        "display": display, "effective_display": display,
                        "covered": cell.tag == f"{table_ns}covered-table-cell",
                        "raw_value": next((cell.get(f"{office_ns}{key}") for key in
                                           ("value", "date-value", "time-value", "boolean-value", "string-value")
                                           if cell.get(f"{office_ns}{key}") is not None), None),
                        "value_type": cell.get(f"{office_ns}value-type"),
                        "formula": cell.get(f"{table_ns}formula"),
                        "style_id": cell.get(f"{table_ns}style-name"),
                        "column_repeat": repeat, "row_repeat": row_repeat,
                        "row_span": _positive(cell.get(f"{table_ns}number-rows-spanned")),
                        "column_span": _positive(cell.get(f"{table_ns}number-columns-spanned")),
                    }
                    record["cached_value"] = record["raw_value"] if record["formula"] else None
                    if _ods_content(record):
                        used = width
                    runs.append((record, repeat))
                source_rows.append((row_repeat, runs, used, is_header))

            # Протяжённость данных: до последней строки и колонки с содержимым.
            last_row = max_col = row_no = 0
            for row_repeat, _runs, used, _header in source_rows:
                row_no += row_repeat
                if used:
                    last_row = row_no
                    max_col = max(max_col, used)

            physical = []
            merges = []
            explicit_headers = []
            cells = 0
            for row_repeat, runs, _used, is_header in source_rows:
                if len(physical) >= last_row:
                    break
                row_repeat = min(row_repeat, last_row - len(physical))
                # Бюджет — до разворота повторов, по ширине внутри данных.
                cells += row_repeat * max(1, min(max_col, sum(repeat for _record, repeat in runs)))
                _check_budget(cells)
                templates = []
                for record, repeat in runs:
                    take = min(repeat, max_col - len(templates))
                    if take <= 0:
                        break
                    templates.extend(dict(record) for _ in range(take))
                for _ in range(row_repeat):
                    row_no = len(physical) + 1
                    records = []
                    for col_no, template in enumerate(templates, 1):
                        record = dict(template, row=row_no, column=col_no,
                                      coordinate=_coordinate(row_no, col_no))
                        if record["row_span"] > 1 or record["column_span"] > 1:
                            merges.append({"anchor": record["coordinate"], "row_start": row_no,
                                           "column_start": col_no,
                                           "row_end": row_no + record["row_span"] - 1,
                                           "column_end": col_no + record["column_span"] - 1})
                        records.append(record)
                    physical.append({"row": row_no, "cells": records,
                                     "explicit_header": is_header})
                    if is_header:
                        explicit_headers.append(row_no)
            _apply_merges(physical, merges)
            table_data = _source_table(physical, merges, format_name="ods",
                                       header_rows=explicit_headers)
            # Лист без текста фрагмента не даёт, как в v1.
            if _blank(table_data):
                continue
            blocks.append(ParsedBlock(
                type="table", page=page_no, sheet_name=name,
                confidence=1.0, method=self.method, section_title=name,
                table_data=table_data,
            ))
        return blocks

    def _from_ods_v1(self, data: bytes) -> List[ParsedBlock]:
        """Профиль v1: отображаемые строки листа, без исходной сетки."""
        table_ns = "{urn:oasis:names:tc:opendocument:xmlns:table:1.0}"
        text_ns = "{urn:oasis:names:tc:opendocument:xmlns:text:1.0}"

        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            content = archive.read("content.xml")
        root = ElementTree.fromstring(content)

        blocks: List[ParsedBlock] = []
        for page_no, table in enumerate(root.iter(f"{table_ns}table"), start=1):
            name = table.get(f"{table_ns}name") or f"Лист {page_no}"
            rows: List[List[str]] = []
            cells = 0
            for row in table.iter(f"{table_ns}table-row"):
                values: List[str] = []
                for cell in row.iter(f"{table_ns}table-cell"):
                    text = "".join(
                        node.text or "" for node in cell.iter(f"{text_ns}p")
                    ).strip()
                    repeat = int(cell.get(f"{table_ns}number-columns-repeated") or 1)
                    if repeat < 1:
                        raise ValueError("Некорректный повтор колонок ODS")
                    _check_budget(cells + len(values) + repeat)
                    values.extend([text] * repeat)
                while values and not values[-1]:
                    values.pop()
                if values:
                    repeat = int(row.get(f"{table_ns}number-rows-repeated") or 1)
                    if repeat < 1:
                        raise ValueError("Некорректный повтор строк ODS")
                    cells += repeat * len(values)
                    _check_budget(cells)
                    rows.extend([list(values) for _ in range(repeat)])
            if not rows:
                continue
            blocks.append(ParsedBlock(
                type="table", page=page_no, sheet_name=name,
                confidence=1.0, method=self.method, section_title=name,
                table_data={"headers": rows[0], "rows": rows[1:], "total_row": None},
            ))
        return blocks

    @staticmethod
    def _sheet_rows(sheet, formula_sheet=None) -> List[List[str]]:
        rows: List[List[str]] = []
        cells = 0
        formula_rows = (
            formula_sheet.iter_rows(values_only=True) if formula_sheet is not None else None
        )
        for raw in sheet.iter_rows(values_only=True):
            cells += max(1, len(raw))
            _check_budget(cells)
            source = next(formula_rows, ()) if formula_rows is not None else ()
            values = [
                _cell_text(cell, source[index] if index < len(source) else None)
                for index, cell in enumerate(raw)
            ]
            if any(values):
                rows.append(values)
        return rows


def _cell_text(cached, raw) -> str:
    """Значение ячейки; для формулы без сохранённого значения — её текст."""
    if cached is not None:
        return str(cached).strip()
    if isinstance(raw, str) and raw.startswith("="):
        return raw.strip()
    return ""


def _positive(value):
    number = int(value or 1)
    if number < 1:
        raise ValueError("Некорректный повтор или размер объединения ODS")
    return number


def _coordinate(row, column):
    letters = ""
    while column:
        column, remainder = divmod(column - 1, 26)
        letters = chr(65 + remainder) + letters
    return f"{letters}{row}"


def _ods_text(node, namespace):
    if node.tag == f"{namespace}s":
        count = _positive(node.get(f"{namespace}c"))
        _check_budget(count)
        return " " * count
    if node.tag == f"{namespace}tab":
        return "\t"
    if node.tag == f"{namespace}line-break":
        return "\n"
    return (node.text or "") + "".join(
        _ods_text(child, namespace) + (child.tail or "") for child in node
    )


def _ods_rows(parent, namespace, header=False):
    for child in parent:
        if child.tag == f"{namespace}table-row":
            yield child, header
        elif child.tag in {f"{namespace}{name}" for name in (
            "table-header-rows", "table-row-group", "table-rows",
        )}:
            yield from _ods_rows(child, namespace, header or child.tag == f"{namespace}table-header-rows")


def _blank(table_data) -> bool:
    """Лист без единой непустой ячейки: фрагмента не даёт."""
    return table_data["source_structure"]["coverage"] == "blank"


def _ods_content(record) -> bool:
    """Ячейка ODS, которая что-то несёт: текст, значение, формулу или объединение."""
    return bool(record["display"] or record["raw_value"] is not None or record["formula"]
                or record["covered"] or record["row_span"] > 1 or record["column_span"] > 1)


def _apply_merges(rows, merges):
    # The reference is separate from the original blank display. No fill-down.
    # В бюджет идёт только часть объединения внутри прочитанной сетки:
    # диапазон до XFD1048576 над тремя ячейками данных — не миллиард ячеек.
    width = max((len(row["cells"]) for row in rows), default=0)
    cost = sum(max(0, min(m["row_end"], len(rows)) - m["row_start"] + 1) *
               max(0, min(m["column_end"], width) - m["column_start"] + 1) for m in merges)
    _check_budget(cost)
    for merge in merges:
        start_row, start_col = merge["row_start"], merge["column_start"]
        if start_row > len(rows) or start_col > len(rows[start_row - 1]["cells"]):
            continue
        anchor = rows[start_row - 1]["cells"][start_col - 1]
        for row in rows[start_row - 1:merge["row_end"]]:
            for cell in row["cells"][start_col - 1:merge["column_end"]]:
                cell["merge_anchor"] = merge["anchor"]
                if cell is not anchor and not cell["display"]:
                    cell["effective_display"] = anchor["display"]


def _source_table(rows, merges, *, format_name, visibility="unknown", header_rows=None):
    nonempty = [row for row in rows if any(cell["display"] for cell in row["cells"])]
    header_rows = list(header_rows or [])
    titles = []
    if header_rows:
        titles = [row["row"] for row in nonempty if row["row"] < min(header_rows)]
    # A leading full-width merge is a title, not the column vocabulary.
    if not header_rows and nonempty:
        candidate = nonempty[0]
        while len(nonempty) > 1 and sum(bool(c["display"]) for c in candidate["cells"]) == 1:
            populated = next(c for c in candidate["cells"] if c["display"])
            if not any(m["anchor"] == populated["coordinate"] and
                       m["column_end"] > m["column_start"] for m in merges):
                break
            titles.append(candidate["row"])
            nonempty = nonempty[1:]
            candidate = nonempty[0]
        header_rows = [nonempty[0]["row"]]
    header_set = set(header_rows)
    data = [row for row in nonempty if row["row"] not in header_set and row["row"] not in titles]

    def values(row):
        result = [cell["display"] for cell in row["cells"]]
        while result and not result[-1]:
            result.pop()
        return result

    header = next((row for row in rows if row["row"] in header_set), None)
    return {
        "headers": values(header) if header else [],
        "rows": [values(row) for row in data], "total_row": None,
        "source_structure": {
            "schema": "source-table-v1", "format": format_name,
            "coordinate_system": "one_based_physical_sheet", "sheet_visibility": visibility,
            "coverage": "complete" if nonempty else "blank",
            "header_detection": "explicit" if header_rows and format_name == "ods" and
                any(row.get("explicit_header") for row in rows) else "layout_candidate",
            "header_rows": header_rows, "title_rows": titles,
            "data_rows": [row["row"] for row in data],
            "rows": rows, "merges": merges,
            "display_origin": "openpyxl_value" if format_name == "xlsx" else "source_text",
        },
    }


def _xlsx_sources(data):
    ns = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    rel_ns = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
    package_ns = "{http://schemas.openxmlformats.org/package/2006/relationships}"
    from openpyxl.utils.cell import coordinate_to_tuple, range_boundaries

    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        # The existing file-size ceiling also bounds expanded ZIP input. Check all
        # entries before openpyxl may read styles/shared strings or any XML member.
        limit = _check_zip_budget(data, archive, "XLSX")
        targets = {r.get("Id"): dict(r.attrib) for r in _xml_records(
            archive, "xl/_rels/workbook.xml.rels", {f"{package_ns}Relationship"}, limit)}
        sheet_records = [dict(node.attrib) for node in _xml_records(
            archive, "xl/workbook.xml", {f"{ns}sheet"}, limit)]
        sheets = {}
        total_cells = 0
        for sheet in sheet_records:
            relationship = targets[sheet[f"{rel_ns}id"]]
            if relationship.get("Type") != f"{rel_ns[1:-1]}/worksheet":
                continue
            if relationship.get("TargetMode") == "External":
                raise ValueError("External XLSX worksheet relationship")
            target = relationship["Target"]
            path = target.lstrip("/") if target.startswith("/") else posixpath.normpath(posixpath.join("xl", target))
            cells = {}
            merges = []
            max_row = max_column = 0
            for node in _xml_records(archive, path, {f"{ns}c", f"{ns}mergeCell", f"{ns}dimension"}, limit):
                if node.tag != f"{ns}c":
                    left, top, right, bottom = range_boundaries(node.get("ref"))
                    if any(type(value) is not int or value < 1 for value in (left, top, right, bottom)):
                        raise ValueError("Invalid XLSX source range")
                    # Объявленный dimension и диапазоны объединений границу сетки
                    # не задают: граница — непустые ячейки (ниже).
                    if node.tag == f"{ns}mergeCell":
                        merges.append({"anchor": _coordinate(top, left), "row_start": top,
                                       "column_start": left, "row_end": bottom, "column_end": right})
                    continue
                coordinate = node.get("r")
                if not coordinate or coordinate in cells:
                    raise ValueError("Missing or duplicate XLSX source cell coordinate")
                row, column = coordinate_to_tuple(coordinate)
                if row < 1 or column < 1:
                    raise ValueError("Invalid XLSX source cell coordinate")
                value, formula = node.find(f"{ns}v"), node.find(f"{ns}f")
                inline = "".join(n.text or "" for n in node.iter(f"{ns}t"))
                # Ячейка только со стилем (форматирование пустых строк до конца
                # листа) ни границу, ни бюджет не двигает и записи не получает.
                if formula is None and not inline and (value is None or not value.text):
                    continue
                total_cells += 1
                _check_budget(total_cells)
                max_row, max_column = max(max_row, row), max(max_column, column)
                _check_budget(max_row * max_column)
                cells[coordinate] = {
                    "raw_value": value.text if value is not None else None,
                    "value_type": node.get("t", "n"), "style_id": node.get("s", "0"),
                    "formula": formula.text if formula is not None else None,
                    "formula_attributes": dict(formula.attrib) if formula is not None else {},
                    "cached_value": value.text if formula is not None and value is not None else None,
                    "formula_cache_status": ("available" if value is not None and value.text is not None
                                             else "missing") if formula is not None else "not_applicable",
                    "inline_text": inline,
                }
            name = sheet["name"]
            if name in sheets:
                raise ValueError("Duplicate XLSX worksheet name")
            # Объединение, начатое за границей данных, покрывает только пустые
            # ячейки: в сетку оно не входит.
            # Конец оставленного объединения обрезается по сетке данных.
            merges = [{**merge, "row_end": min(merge["row_end"], max_row),
                       "column_end": min(merge["column_end"], max_column)}
                      for merge in merges
                      if merge["row_start"] <= max_row and merge["column_start"] <= max_column]
            sheets[name] = {"cells": cells, "merges": merges,
                             "max_row": max_row, "max_column": max_column}
    return sheets


def _xml_records(archive, path, tags, limit):
    """Stream bounded source records; release cells/rows instead of retaining a DOM."""
    class BoundedReader:
        def __init__(self, source):
            self.source = source
            self.consumed = 0

        def read(self, size=-1):
            remaining = limit - self.consumed
            chunk = self.source.read(min(size, remaining + 1) if size >= 0 else remaining + 1)
            self.consumed += len(chunk)
            if self.consumed > limit:
                raise ValueError("XLSX ZIP XML exceeds byte budget")
            return chunk

    stack = []
    with archive.open(path) as source:
        for event, node in ElementTree.iterparse(BoundedReader(source), events=("start", "end")):
            if event == "start":
                stack.append(node)
                continue
            if node.tag in tags:
                yield node
            if node.tag in tags or node.tag.endswith("}row"):
                node.clear()
                if len(stack) > 1:
                    stack[-2].remove(node)
            stack.pop()
