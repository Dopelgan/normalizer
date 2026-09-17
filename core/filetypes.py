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
        FileType("csv",  "text/csv", KIND_SPREADSHEET),

        FileType("txt",  "text/plain",    KIND_PLAIN_TEXT),
        FileType("md",   "text/markdown", KIND_PLAIN_TEXT),

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
    "docx", "xlsx", "csv", "txt", "md", "dxf",
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
    return None
