"""
Текст документа для классификации приёма.

Слой G-2 раньше искал маркеры в первых 64 КБ файла как есть. Для PDF, DOCX
и XLSX это сжатый двоичный поток: декодированный в cp1251, он даёт случайную
кириллицу и случайные же совпадения. Реальный случай: приложение к договору
было признано перепиской, потому что внутри сжатого потока нашлась
последовательность `re:`.

Здесь из файла достаётся именно текст — тем инструментом, который понимает
формат. У PDF без текстового слоя (скан) текст берётся быстрым OCR первых
страниц (`GATEWAY_OCR_PROBE_PAGES`, `GATEWAY_OCR_PROBE_SECONDS`) — тем же
`TesseractFallbackProvider.parse_pages`, что у проверки читаемости Quality
Gate. Если текста нет и так (картинка, чертёж, OCR недоступен), это честно
сообщается: классифицировать такой файл по содержимому нельзя, решают путь
и имя.
"""

from __future__ import annotations

import io
import logging
import re
import time
from dataclasses import dataclass
from typing import Optional

from celery.exceptions import SoftTimeLimitExceeded

from core import filetypes
from core.config import settings

logger = logging.getLogger(__name__)

# Сколько текста достаточно, чтобы понять род документа. Первые страницы
# несут титул, шапку и предмет — дальше начинается тело, которое род уже
# не меняет.
TEXT_LIMIT = 20_000
PDF_PAGES = 5
XLSX_ROWS = 60

# Источник текста — попадает в сигналы приёма.
SOURCE_PDF = "pdf"
SOURCE_DOCX = "docx"
SOURCE_XLSX = "xlsx"
SOURCE_XLS = "xls"
SOURCE_ODS = "ods"
SOURCE_PLAIN = "plain"
SOURCE_OCR = "ocr"
SOURCE_NONE = "none"

_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


@dataclass(frozen=True)
class TextProbe:
    """Что удалось прочитать из файла и можно ли этому верить."""

    text: str
    source: str
    reason: Optional[str] = None
    # Страниц в PDF без текстового слоя — столько можно отдать OCR.
    pages: int = 0

    @property
    def reliable(self) -> bool:
        """Текст извлечён инструментом формата и не пуст."""
        return self.source != SOURCE_NONE and bool(self.text.strip())

    def as_dict(self) -> dict:
        return {
            "text_source": self.source,
            "text_chars": len(self.text),
            **({"text_reason": self.reason} if self.reason else {}),
        }


def extract(
    file_type: str, data: bytes, uri: Optional[str] = None, storage=None,
) -> TextProbe:
    """
    Текст документа по его типу. Байты — весь файл, а не образец.

    `uri` и `storage` нужны быстрому OCR скана: без них PDF без текстового
    слоя остаётся без текста, как раньше.
    """
    kind = filetypes.kind_of(file_type)

    if kind == filetypes.KIND_PDF:
        probe = _from_pdf(data)
        if probe.source == SOURCE_NONE and probe.pages and uri:
            return _from_ocr(uri, data, storage, probe)
        return probe
    if kind == filetypes.KIND_OFFICE_TEXT:
        return _from_office_text(file_type, data)
    if kind == filetypes.KIND_SPREADSHEET:
        # Раньше здесь по типу отбирался только `xlsx`, а всё остальное
        # семейство таблиц — xlsm, xls, ods — уезжало в `_from_plain` и
        # декодировалось как текст. Для zip и BIFF это случайная кириллица
        # из двоичного потока, то есть случайные же совпадения маркеров:
        # книга с макросами классифицировалась по мусору.
        return _from_spreadsheet(file_type, data)
    if kind == filetypes.KIND_PLAIN_TEXT:
        return _from_plain(data)
    if kind == filetypes.KIND_CAD:
        # DXF — текст, но это коды и координаты, а не проза: маркеры по
        # нему искать бессмысленно, род и так известен из формата.
        return TextProbe("", SOURCE_NONE, "формат CAD: текста для разбора нет")
    if kind == filetypes.KIND_IMAGE:
        return TextProbe("", SOURCE_NONE, "растр: текста без распознавания нет")
    return TextProbe("", SOURCE_NONE, f"тип {file_type!r} без извлечения текста")


# ---------------------------------------------------------------- форматы

def _from_pdf(data: bytes) -> TextProbe:
    from core.providers.text_layer import open_pdf

    document = open_pdf(data)
    if document is None:
        return TextProbe("", SOURCE_NONE, "PDF не открылся")
    page_count = document.page_count
    try:
        chunks = []
        for index in range(min(document.page_count, PDF_PAGES)):
            try:
                chunks.append(document[index].get_text() or "")
            except Exception as exc:  # noqa: BLE001 — битая страница не роняет разбор
                logger.debug("Страница %d не прочиталась: %s", index + 1, exc)
            if sum(len(c) for c in chunks) >= TEXT_LIMIT:
                break
        text = _clean("\n".join(chunks))
    finally:
        document.close()

    if not text.strip():
        # Скан: текстового слоя нет. Это не ошибка, но и судить по
        # содержимому слоя не о чем — текст может дать только OCR.
        return TextProbe(
            "", SOURCE_NONE, "PDF без текстового слоя (скан)", pages=page_count,
        )
    return TextProbe(text, SOURCE_PDF)


def _from_ocr(uri: str, data: bytes, storage, scan: TextProbe) -> TextProbe:
    """
    Текст скана быстрым OCR первых страниц.

    Прод, 13.11: пятнадцать сканов без текстового слоя и без маркера в имени
    («ЦО-11-101-13225 (конструирование оборудования) 2028.pdf», «Лист
    ознакомления.pdf») получали запасной «деловой, 0.50» и уходили в
    карантин: судить было не по чему, кроме имени. Текст у них есть — на
    картинке.

    Распознавание то же, что у быстрого OCR Quality Gate, страницы и
    разрешение те же: в задаче приёма (`ocr_memo`) второй этап их не
    распознаёт заново. Страницы читаются по одной, и после бюджета времени
    следующая не начинается. Сбой OCR — не ошибка приёма: остаётся прежнее
    «текста нет».
    """
    limit = min(scan.pages, max(0, settings.GATEWAY_OCR_PROBE_PAGES))
    if limit <= 0:
        return scan

    from core.providers.tesseract_fallback import TesseractFallbackProvider

    provider = TesseractFallbackProvider(storage=storage)
    deadline = time.monotonic() + max(0.0, settings.GATEWAY_OCR_PROBE_SECONDS)
    chunks = []
    read = 0
    for page in range(1, limit + 1):
        if page > 1 and time.monotonic() >= deadline:
            logger.info("OCR приёма %s остановлен по бюджету на странице %d", uri, page)
            break
        try:
            blocks = provider.parse_pages(uri, [page], file_type="pdf", data=data)
        except SoftTimeLimitExceeded:
            raise
        except Exception as exc:  # noqa: BLE001 — OCR не обязан быть поднят
            logger.info("OCR приёма недоступен для %s: %s", uri, exc)
            break
        read += 1
        chunks.extend(block.text for block in blocks if (block.text or "").strip())

    text = _clean("\n".join(chunks))
    if not text.strip():
        reason = scan.reason
        if read:
            reason = f"{reason}; OCR {read} стр. текста не дал"
        return TextProbe("", SOURCE_NONE, reason, pages=scan.pages)
    return TextProbe(
        text, SOURCE_OCR, f"PDF без текстового слоя, OCR {read} стр.",
        pages=scan.pages,
    )


def _from_office_text(file_type: str, data: bytes) -> TextProbe:
    """DOCX как есть; DOC — через конвертацию в DOCX."""
    from core.providers import office_convert

    if office_convert.converts(file_type):
        try:
            data = office_convert.convert(data, file_type)
        except office_convert.ConversionUnavailable as exc:
            return TextProbe("", SOURCE_NONE, str(exc))
        except Exception as exc:  # noqa: BLE001 — приём не роняем
            return TextProbe("", SOURCE_NONE, f"конвертация не удалась: {exc}")
    return _from_docx(data)


def _from_spreadsheet(file_type: str, data: bytes) -> TextProbe:
    """Таблица своим инструментом: csv — текстом, xlsx/xlsm, xls, ods."""
    if file_type in ("xlsx", "xlsm"):
        return _from_xlsx(data)
    if file_type == "xls":
        return _from_xls(data)
    if file_type == "ods":
        return _from_ods(data)
    # CSV и всё, что осталось, — обычный текст.
    return _from_plain(data)


def _from_xls(data: bytes) -> TextProbe:
    """Excel 97-2003: двоичный BIFF, открывается xlrd."""
    try:
        import xlrd  # type: ignore
    except ImportError:
        return TextProbe("", SOURCE_NONE, "xlrd недоступен")
    try:
        book = xlrd.open_workbook(file_contents=data)
    except Exception as exc:  # noqa: BLE001
        return TextProbe("", SOURCE_NONE, f"XLS не открылся: {exc}")

    parts = []
    for sheet in book.sheets():
        parts.append(str(sheet.name))
        for row_no in range(min(sheet.nrows, XLSX_ROWS)):
            parts.append(" ".join(
                str(cell.value) for cell in sheet.row(row_no) if cell.value not in (None, "")
            ))
        if sum(len(p) for p in parts) >= TEXT_LIMIT:
            break

    text = _clean("\n".join(parts))
    if not text.strip():
        return TextProbe("", SOURCE_NONE, "XLS без содержимого")
    return TextProbe(text, SOURCE_XLS)


def _from_ods(data: bytes) -> TextProbe:
    """OpenDocument: zip, внутри которого content.xml — обычный XML."""
    import io as _io
    import zipfile

    try:
        with zipfile.ZipFile(_io.BytesIO(data)) as archive:
            raw = archive.read("content.xml")
    except Exception as exc:  # noqa: BLE001
        return TextProbe("", SOURCE_NONE, f"ODS не открылся: {exc}")

    try:
        from defusedxml import ElementTree  # type: ignore
    except ImportError:  # pragma: no cover — defusedxml в зависимостях есть
        from xml.etree import ElementTree  # type: ignore

    try:
        root = ElementTree.fromstring(raw)
    except Exception as exc:  # noqa: BLE001
        return TextProbe("", SOURCE_NONE, f"ODS разобран не полностью: {exc}")

    parts = []
    for node in root.iter():
        if node.text and node.text.strip():
            parts.append(node.text.strip())
        if sum(len(p) for p in parts) >= TEXT_LIMIT:
            break

    text = _clean(" ".join(parts))
    if not text.strip():
        return TextProbe("", SOURCE_NONE, "ODS без содержимого")
    return TextProbe(text, SOURCE_ODS)


def _from_docx(data: bytes) -> TextProbe:
    try:
        import docx  # type: ignore
    except ImportError:
        return TextProbe("", SOURCE_NONE, "python-docx недоступен")
    try:
        document = docx.Document(io.BytesIO(data))
    except Exception as exc:  # noqa: BLE001
        return TextProbe("", SOURCE_NONE, f"DOCX не открылся: {exc}")

    parts = [p.text for p in document.paragraphs if p.text]
    for table in document.tables:
        for row in table.rows:
            parts.append(" ".join(cell.text for cell in row.cells if cell.text))
            if sum(len(p) for p in parts) >= TEXT_LIMIT:
                break
    text = _clean("\n".join(parts))
    if not text.strip():
        return TextProbe("", SOURCE_NONE, "DOCX без текста")
    return TextProbe(text, SOURCE_DOCX)


def _from_xlsx(data: bytes) -> TextProbe:
    try:
        from openpyxl import load_workbook  # type: ignore
    except ImportError:
        return TextProbe("", SOURCE_NONE, "openpyxl недоступен")
    try:
        book = load_workbook(io.BytesIO(data), read_only=True, data_only=True)
    except Exception as exc:  # noqa: BLE001
        return TextProbe("", SOURCE_NONE, f"XLSX не открылся: {exc}")

    parts = []
    try:
        for sheet in book.worksheets:
            parts.append(str(sheet.title))
            for row_no, row in enumerate(sheet.iter_rows(values_only=True)):
                parts.append(" ".join(str(v) for v in row if v is not None))
                if row_no + 1 >= XLSX_ROWS:
                    break
            if sum(len(p) for p in parts) >= TEXT_LIMIT:
                break
    finally:
        book.close()

    text = _clean("\n".join(parts))
    if not text.strip():
        return TextProbe("", SOURCE_NONE, "XLSX без содержимого")
    return TextProbe(text, SOURCE_XLSX)


def _from_plain(data: bytes) -> TextProbe:
    for encoding in ("utf-8", "cp1251"):
        try:
            text = data.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:  # pragma: no cover — cp1251 декодирует почти всё
        text = data.decode("utf-8", errors="ignore")

    text = _clean(text)
    if not text.strip():
        return TextProbe("", SOURCE_NONE, "файл пуст")
    return TextProbe(text, SOURCE_PLAIN)


# ------------------------------------------------------------- внутреннее

def _clean(text: str) -> str:
    """Убрать управляющие символы и подрезать до предела."""
    return _CONTROL.sub(" ", text)[:TEXT_LIMIT]
