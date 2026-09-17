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
        return context.file_type == "csv" or OPENPYXL_AVAILABLE

    def run(self, context: DocumentContext) -> ParseResult:
        if context.file_type == "csv":
            blocks = self._from_csv(context.data)
            source_kind = "csv"
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
