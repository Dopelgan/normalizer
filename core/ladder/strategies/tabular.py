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
from typing import List

from core import filetypes
from core.ladder.base import Strategy
from core.ladder.context import DocumentContext
from core.models.parse_result import ParsedBlock, ParseResult

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

# Больше этого числа строк на лист не забираем: остальное всё равно не
# помещается в осмысленный фрагмент, а память съедает.
_MAX_ROWS = 5000


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
        text = self._decode(data)
        dialect = self._dialect(text)
        rows = [
            [str(cell).strip() for cell in row]
            for row in csv.reader(io.StringIO(text), dialect)
        ]
        rows = [row for row in rows if any(row)][:_MAX_ROWS]
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
        workbook = openpyxl.load_workbook(
            io.BytesIO(data), read_only=True, data_only=True
        )
        blocks: List[ParsedBlock] = []
        try:
            for page_no, sheet in enumerate(workbook.worksheets, start=1):
                rows = self._sheet_rows(sheet)
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
        return blocks

    # ------------------------------------------------------------------ XLS
    def _from_xls(self, data: bytes) -> List[ParsedBlock]:
        """Excel 97-2003. Даты приводим к виду, который видит человек."""
        book = xlrd.open_workbook(file_contents=data)
        blocks: List[ParsedBlock] = []
        for page_no, sheet in enumerate(book.sheets(), start=1):
            rows: List[List[str]] = []
            for index in range(min(sheet.nrows, _MAX_ROWS)):
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
        import zipfile

        from defusedxml import ElementTree

        table_ns = "{urn:oasis:names:tc:opendocument:xmlns:table:1.0}"
        text_ns = "{urn:oasis:names:tc:opendocument:xmlns:text:1.0}"

        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            content = archive.read("content.xml")
        root = ElementTree.fromstring(content)

        blocks: List[ParsedBlock] = []
        for page_no, table in enumerate(root.iter(f"{table_ns}table"), start=1):
            name = table.get(f"{table_ns}name") or f"Лист {page_no}"
            rows: List[List[str]] = []
            for row in table.iter(f"{table_ns}table-row"):
                values: List[str] = []
                for cell in row.iter(f"{table_ns}table-cell"):
                    text = "".join(
                        node.text or "" for node in cell.iter(f"{text_ns}p")
                    ).strip()
                    # Повторы пустых ячеек в ODS кодируются числом, а не
                    # копиями: разворачиваем, но в разумных пределах.
                    repeat = int(cell.get(f"{table_ns}number-columns-repeated") or 1)
                    values.extend([text] * min(repeat, 64))
                while values and not values[-1]:
                    values.pop()
                if values:
                    rows.append(values)
                if len(rows) >= _MAX_ROWS:
                    logger.warning("Лист %s обрезан на %d строках", name, _MAX_ROWS)
                    break
            if not rows:
                continue
            blocks.append(ParsedBlock(
                type="table", page=page_no, sheet_name=name,
                confidence=1.0, method=self.method, section_title=name,
                table_data={"headers": rows[0], "rows": rows[1:], "total_row": None},
            ))
        return blocks

    @staticmethod
    def _sheet_rows(sheet) -> List[List[str]]:
        rows: List[List[str]] = []
        for raw in sheet.iter_rows(values_only=True):
            values = ["" if cell is None else str(cell).strip() for cell in raw]
            if any(values):
                rows.append(values)
            if len(rows) >= _MAX_ROWS:
                logger.warning("Лист %s обрезан на %d строках", sheet.title, _MAX_ROWS)
                break
        return rows
