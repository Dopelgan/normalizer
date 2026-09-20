"""
Единый реестр поддерживаемых типов файлов.

Раньше знание о типах было размазано: список расширений для поиска файла
лежал в настройках, таблица MIME — внутри парсера MinerU, признак «это
картинка» — ещё в двух местах своим набором. Стоило добавить формат, и его
приходилось дописывать в каждое из них по отдельности.

Здесь тип объявляется один раз: расширение, MIME и род содержимого. Род
нужен лестнице стратегий — он отвечает на вопрос «чем это в принципе можно
разобрать», ещё до того, как файл открыт.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Set

# Род содержимого. Определяет, какие уровни лестницы вообще применимы.
KIND_CAD = "cad"                  # исходник системы проектирования
KIND_PDF = "pdf"                  # может быть и векторным, и сканом
KIND_IMAGE = "image"              # только растр
KIND_OFFICE_TEXT = "office_text"  # документ с извлекаемым текстом
KIND_SPREADSHEET = "spreadsheet"  # таблица жёсткого формата
KIND_PLAIN_TEXT = "plain_text"    # текст без разметки положения


@dataclass(frozen=True)
class FileType:
    extension: str
    mime: str
    kind: str
    # Формат опознаётся, но разобрать его нечем. Такой файл отклоняется на
    # Quality Gate с внятной причиной, а не падает в середине конвейера.
    supported: bool = True
    note: str = ""


_TYPES: Dict[str, FileType] = {
    t.extension: t for t in (
        FileType("pdf",  "application/pdf", KIND_PDF),

        FileType("jpg",  "image/jpeg", KIND_IMAGE),
        FileType("jpeg", "image/jpeg", KIND_IMAGE),
        FileType("png",  "image/png",  KIND_IMAGE),
        FileType("tiff", "image/tiff", KIND_IMAGE),
        FileType("tif",  "image/tiff", KIND_IMAGE),
        FileType("bmp",  "image/bmp",  KIND_IMAGE),

        FileType("docx", "application/vnd.openxmlformats-officedocument."
                         "wordprocessingml.document", KIND_OFFICE_TEXT),

        FileType("xlsx", "application/vnd.openxmlformats-officedocument."
                         "spreadsheetml.sheet", KIND_SPREADSHEET),
        # Книга с макросами — тот же формат, что и xlsx: openpyxl читает её
        # без оговорок, макросы разбору не мешают.
        FileType("xlsm", "application/vnd.ms-excel.sheet.macroEnabled.12",
                 KIND_SPREADSHEET),
        # Excel 97-2003: двоичный BIFF, читается отдельной библиотекой.
        FileType("xls",  "application/vnd.ms-excel", KIND_SPREADSHEET),
        # OpenDocument: zip с content.xml, таблица разбирается штатным XML.
        FileType("ods",  "application/vnd.oasis.opendocument.spreadsheet",
                 KIND_SPREADSHEET),
        FileType("csv",  "text/csv", KIND_SPREADSHEET),

        FileType("txt",  "text/plain",    KIND_PLAIN_TEXT),
        FileType("md",   "text/markdown", KIND_PLAIN_TEXT),

        # Word 97-2003: двоичный формат, своей библиотеки под него нет.
        # Разбирается через конвертацию в DOCX (core.providers.office_convert),
        # поэтому поддержан, но зависит от конвертера в образе: если его нет,
        # файл отклоняется на приёме с внятной причиной, а не падает посреди
        # разбора.
        FileType("doc", "application/msword", KIND_OFFICE_TEXT),

        FileType("dxf",  "image/vnd.dxf", KIND_CAD),
        FileType(
            "dwg", "image/vnd.dwg", KIND_CAD, supported=False,
            note="DWG — закрытый двоичный формат. Нужен внешний конвертер "
                 "в DXF (ODA File Converter); пришлите DXF или включите "
                 "конвертер в образ.",
        ),
    )
}

# Порядок перебора, когда s3_fileid пришёл без расширения. Сначала то, что
# встречается чаще, — лишних обращений к хранилищу так меньше.
PROBE_ORDER: List[str] = [
    "pdf", "png", "jpg", "jpeg", "tiff", "tif", "bmp",
    # Семейство офисных форматов целиком: раньше в переборе не было ни
    # xlsm, ни xls, ни ods, ни doc, и файл без расширения в этих форматах
    # не находился в хранилище вовсе.
    "docx", "doc", "xlsx", "xlsm", "xls", "ods", "csv", "txt", "md", "dxf",
]

IMAGE_EXTENSIONS: Set[str] = {e for e, t in _TYPES.items() if t.kind == KIND_IMAGE}
SUPPORTED_EXTENSIONS: Set[str] = {e for e, t in _TYPES.items() if t.supported}

_FALLBACK_MIME = "application/octet-stream"


# ===========================================================================
# Доступ
# ===========================================================================

def normalize(file_type: str) -> str:
    """`.PDF`, `PDF`, `pdf` -> `pdf`."""
    return (file_type or "").strip().lower().lstrip(".")


def extension_of(path: str) -> str:
    """Расширение файла или пути, без точки и в нижнем регистре."""
    return normalize(os.path.splitext(path or "")[1])


def get(file_type: str) -> Optional[FileType]:
    return _TYPES.get(normalize(file_type))


def is_known(file_type: str) -> bool:
    """Формат опознан — но, возможно, не поддержан (см. `is_supported`)."""
    return normalize(file_type) in _TYPES


def is_supported(file_type: str) -> bool:
    return normalize(file_type) in SUPPORTED_EXTENSIONS


def kind_of(file_type: str) -> Optional[str]:
    entry = get(file_type)
    return entry.kind if entry else None


def mime_for(file_type: str) -> str:
    entry = get(file_type)
    return entry.mime if entry else _FALLBACK_MIME


def is_image(file_type: str) -> bool:
    return normalize(file_type) in IMAGE_EXTENSIONS


# ===========================================================================
# Опознание по содержимому
# ===========================================================================

# Расширение — это утверждение отправителя, а не факт. Оно бывает пустым
# (`s3_fileid` без расширения), бывает чужим (скан переименовали в .pdf,
# выгрузка отдала .xlsx с CSV внутри). От типа зависит, какой уровень
# лестницы вообще возьмётся за файл, поэтому там, где байты уже в руках,
# формат определяется по ним.

_ZIP_OFFICE_PARTS = (
    ("word/document.xml", "docx"),
    ("xl/workbook.xml", "xlsx"),
)

_ODF_MIMETYPES = (
    (b"mimetypeapplication/vnd.oasis.opendocument.spreadsheet", "ods"),
)

# Имена потоков внутри OLE2 записаны в UTF-16.
_OLE_WORKBOOK = "Workbook".encode("utf-16-le")
_OLE_BOOK = "Book".encode("utf-16-le")
_OLE_WORD = "WordDocument".encode("utf-16-le")


def sniff(data: bytes) -> Optional[str]:
    """
    Формат по сигнатуре первых байтов. `None` — сигнатуры нет.

    Текстовые форматы (txt, md, csv) сигнатуры не имеют и здесь не
    угадываются: для них расширение — единственный доступный признак, а
    ошибиться между ними дешевле, чем принять таблицу за документ.
    """
    if not data:
        return None

    if data[:5] == b"%PDF-":
        return "pdf"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if data[:3] == b"\xff\xd8\xff":
        return "jpg"
    if data[:4] in (b"II*\x00", b"MM\x00*"):
        return "tiff"
    if data[:2] == b"BM":
        return "bmp"
    # DWG пишет версию в первых шести байтах: AC1015, AC1027, AC1032, ...
    if data[:2] == b"AC" and data[2:6].isdigit():
        return "dwg"
    if data[:8] == b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1":
        return _sniff_ole(data)
    if data[:4] == b"PK\x03\x04":
        return _sniff_zip(data)

    head = data[:4096].lstrip()
    if head.startswith((b"SECTION", b"0\r\n", b"0\n")):
        # DXF — текстовый формат: пары «код / значение», первая секция
        # начинается с кода 0 и слова SECTION.
        window = data[:4096]
        if b"SECTION" in window and (b"HEADER" in window or b"ENTITIES" in window):
            return "dxf"
    return None


def _sniff_zip(data: bytes) -> Optional[str]:
    """
    Офисные форматы поверх zip различаются составом частей.

    OpenDocument кладёт `mimetype` первым файлом без сжатия — его видно уже
    в первых байтах, даже если архив обрезан.
    """
    head = data[:256]
    for marker, extension in _ODF_MIMETYPES:
        if marker in head:
            return extension

    import io
    import zipfile

    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            names = set(archive.namelist())
    except Exception:  # noqa: BLE001 — обрезанный или битый архив
        return None

    if "xl/workbook.xml" in names:
        # Книга с макросами — тот же формат, отличается наличием проекта VBA.
        return "xlsm" if any(n.endswith("vbaProject.bin") for n in names) else "xlsx"
    for part, extension in _ZIP_OFFICE_PARTS:
        if part in names:
            return extension
    return None


def _sniff_ole(data: bytes) -> Optional[str]:
    """
    Формат внутри контейнера OLE2 (Excel 97-2003, Word 97-2003).

    Имена потоков лежат в каталоге контейнера в UTF-16, поэтому ищем их в
    таком виде: «Workbook» — книга Excel, «WordDocument» — документ Word.

    Порядок проверки не произвольный. Короткое имя «Book» (так называется
    поток книги Excel 5.0) — четыре буквы, и в теле документа Word оно
    находится случайно: в реальном .doc из LibreOffice `Book` в UTF-16
    встречается за тринадцать килобайт до настоящего каталога. Документ
    Word объявлялся книгой Excel, попадал в табличную ветку, там не
    открывался — и классификация уезжала на имя файла. Поэтому сперва
    ищутся длинные однозначные имена, и только потом короткое.
    """
    window = data[:65536]
    if _OLE_WORD in window:
        return "doc"
    if _OLE_WORKBOOK in window:
        return "xls"
    if _OLE_BOOK in window:
        return "xls"
    return None


def resolve(path: str, data: Optional[bytes]) -> "TypeVerdict":
    """
    Итоговый тип файла по пути и содержимому: сигнатура важнее расширения.

    Возвращается и то, и другое: расхождение — не ошибка сама по себе, но
    его видно в сигналах приёма и в логе разбора, а не только по странному
    поведению дальше по конвейеру.
    """
    return resolve_declared(extension_of(path), data)


def resolve_declared(declared: str, data: Optional[bytes]) -> "TypeVerdict":
    """То же, но когда тип объявлен отдельно от пути (контракт RAG)."""
    declared = normalize(declared)
    detected = sniff(data or b"")

    if detected and detected != declared:
        # Сигнатура есть и спорит с расширением — верим байтам. Частный
        # случай: jpg/jpeg и tif/tiff — один формат под двумя именами.
        if not _same_format(declared, detected):
            return TypeVerdict(detected, declared, detected, mismatch=True)
    return TypeVerdict(detected or declared, declared, detected, mismatch=False)


def _same_format(left: str, right: str) -> bool:
    aliases = {"jpeg": "jpg", "tif": "tiff"}
    return aliases.get(left, left) == aliases.get(right, right)


@dataclass(frozen=True)
class TypeVerdict:
    """Чем файл оказался, чем назвался и спорят ли эти двое."""

    file_type: str
    declared: str
    detected: Optional[str]
    mismatch: bool

    @property
    def explanation(self) -> Optional[str]:
        if not self.mismatch:
            return None
        declared = f".{self.declared}" if self.declared else "без расширения"
        return (
            f"Содержимое опознано как .{self.detected}, "
            f"файл назван {declared} — тип взят по содержимому."
        )


def rejection_reason(file_type: str) -> Optional[str]:
    """
    Почему файл нельзя принять. `None` — можно. Текст рассчитан на то, что
    его увидит человек в отчёте приёма, поэтому он объясняет, а не ругается.
    """
    normalized = normalize(file_type)
    if not normalized:
        return "У файла нет расширения, определить формат невозможно."
    entry = _TYPES.get(normalized)
    if entry is None:
        return (
            f"Формат .{normalized} не поддерживается. Поддерживаются: "
            + ", ".join("." + e for e in sorted(SUPPORTED_EXTENSIONS)) + "."
        )
    if not entry.supported:
        return entry.note
    return _conversion_reason(normalized)


def _conversion_reason(file_type: str) -> Optional[str]:
    """
    Формат поддержан, но разбирается только через конвертер. Нет конвертера
    — нет и разбора, и сказать об этом нужно на приёме, а не в середине
    конвейера.
    """
    from core.providers import office_convert

    if not office_convert.converts(file_type) or office_convert.available():
        return None
    return office_convert.unavailable_reason(file_type)
