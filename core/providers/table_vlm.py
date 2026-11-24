"""
Перечитывание неразобранной таблицы скана моделью зрения (Qwen-VL).

Таблица уходит сюда, когда модель таблиц MinerU отдала ячейки разметкой, а
собрать их заново по строкам OCR не вышло. Типичный случай — альбомная
таблица на книжном листе паспорта: MinerU читает её поперёк, Tesseract
тоже, и строки слоя — каша из обрывков. Разобранная на эти обрывки
(`table_from_lines.degrade`) таблица теряет значения ячеек.

Здесь страница рендерится, таблица вырезается по рамке MinerU, ставится
прямо по OSD Tesseract и читается моделью. Не вышло на любом шаге — None,
и вызывающий разбирает таблицу на текст: эта ветка только добавляет
шанс, но не отнимает обычный исход.
"""

from __future__ import annotations

import io
import logging
from typing import Any, Callable, Dict, List, Optional

from celery.exceptions import SoftTimeLimitExceeded

from core import deadline, filetypes
import re

from core.config import settings
from core.models.parse_result import ParsedBlock

logger = logging.getLogger(__name__)

TableReader = Callable[[ParsedBlock], Optional[Dict[str, Any]]]


def table_reader(file_bytes: Optional[bytes], file_type: str, client=None) -> Optional[TableReader]:
    """
    Функция «блок таблицы -> table_data или None» для `reconcile_with_text_layer`.
    None — ветка выключена, файла нет или модели нет.
    """
    if not settings.TABLE_VLM_RETRY_ENABLED or not file_bytes:
        return None
    if client is None:
        from core.providers.qwen_client import QwenVisionClient

        client = QwenVisionClient()
    if not client.available:
        return None

    pages: Dict[int, Any] = {}
    budget = {"left": max(0, int(settings.TABLE_VLM_MAX_TABLES))}

    def read(block: ParsedBlock) -> Optional[Dict[str, Any]]:
        if budget["left"] <= 0:
            logger.warning("Таблица на стр. %s не перечитана: исчерпан предел %d на документ",
                           block.page, settings.TABLE_VLM_MAX_TABLES)
            return None
        left = deadline.remaining()
        if left is not None and left < settings.VLM_TIMEOUT:
            # До срока задачи не уложится вызов модели: таблица разбирается
            # на текст, а документ не теряется по сроку.
            logger.warning("Таблица на стр. %s не перечитана: до срока задачи %.0f с",
                           block.page, left)
            return None
        budget["left"] -= 1
        try:
            spec = settings.TABLE_VLM_ESKD_SPEC and eskd_spec(block.table_data)
            key = (block.page, spec)
            if key not in pages:
                pages[key] = _render_page(file_bytes, file_type, block.page,
                                          dpi=settings.TABLE_VLM_SPEC_DPI if spec else None)
            page = pages[key]
            if page is None:
                return None
            piece = _upright(_crop(page, block.bbox))
            if spec:
                table = client.read_table(piece, max_side=settings.TABLE_VLM_SPEC_MAX_SIDE, spec=True)
            else:
                table = client.read_table(piece)
        except SoftTimeLimitExceeded:
            raise
        except Exception as exc:  # noqa: BLE001 — сбой перечитывания не роняет разбор
            logger.warning("Таблица на стр. %s не перечитана моделью: %s", block.page, exc)
            return None
        data = _table_data(table)
        if data is None:
            logger.info("Модель не прочитала таблицу на стр. %s", block.page)
        else:
            logger.info("Таблица на стр. %s перечитана моделью: строк %d",
                        block.page, len(data["rows"]))
        return data

    return read


# Графы спецификации ЕСКД (ГОСТ 2.106, форма 1) в шапке таблицы MinerU.
_SPEC_GRAPHS = ("обозначение", "наименование")
_SPEC_ANY = ("поз", "кол", "примеч", "приме-", "формат", "зона")


def eskd_spec(table_data: Optional[Dict[str, Any]]) -> bool:
    """
    Таблица — спецификация ЕСКД: в шапке или первых строках есть графы
    «Обозначение» и «Наименование» и хотя бы одна из «Поз.», «Кол.»,
    «Примечание», «Формат», «Зона». Шапка скана приходит с мусором
    («шowdoф» вместо «Формат»), поэтому достаточно двух слов и одного.
    """
    if not table_data:
        return False
    rows = [table_data.get("headers") or []] + list(table_data.get("rows") or [])[:2]
    text = " ".join(str(cell) for row in rows for cell in row).casefold()
    if all(word in text for word in _SPEC_GRAPHS) and any(word in text for word in _SPEC_ANY):
        return True
    # Шапка первого листа спецификации на скане приходит неузнаваемой
    # (ТАДУ 687000.001, изм. 2): зато в строках — обозначения документов
    # и изделий «ТАДУ 687000.001 СБ». Три таких ячейки — спецификация.
    cells = [str(cell) for row in table_data.get("rows") or [] for cell in row]
    return sum(1 for cell in cells if _SPEC_DESIGNATION.search(cell)) >= 3


# Обозначение по ГОСТ 2.201: код организации и «xxxxxx.xxx» (точка может
# потеряться на скане: «ТАДУ 687000001001»).
_SPEC_DESIGNATION = re.compile(r"[А-ЯЁA-Z]{2,4}\s?\d{6}\.?\d{3}")

# Коды документов ЕСКД (ГОСТ 2.102), которые в спецификации стоят после
# обозначения: скан отдаёт их латиницей и цифрами («P3» вместо «РЭ»).
DOC_CODES = frozenset({
    "СБ", "ТУ", "РЭ", "ЭТ", "ПС", "ФО", "ВО", "ГЧ", "МЧ", "ТЧ", "УЧ", "ВС", "ВП", "ВД", "ПТ", "ЭП", "ТП",
    "ИМ", "МЭ", "ТБ", "РР", "ПЭ3", "Э3", "Э4", "Э5", "Э6", "Э7", "Д1", "Д2", "Д3", "ПЗ", "КД", "ЗИ", "ЭД",
})
_CODE_LOOKALIKE = str.maketrans({"P": "Р", "C": "С", "B": "В", "T": "Т", "E": "Е", "O": "О", "H": "Н",
                                 "K": "К", "M": "М", "X": "Х", "A": "А", "3": "Э", "6": "Б"})
_CODE_AFTER = re.compile(r"(\d{6}\.?\d{3}(?:\.\d{2,3})*(?:-\d{2,4})?)\s+([A-ZА-ЯЁ0-9]{2,3})\b")
# Латинская буква, неотличимая от кириллической, внутри русского слова.
_CYR_LOOKALIKE = str.maketrans({"a": "а", "e": "е", "o": "о", "p": "р", "c": "с", "x": "х", "y": "у",
                                "A": "А", "B": "В", "E": "Е", "K": "К", "M": "М", "H": "Н", "O": "О",
                                "P": "Р", "C": "С", "T": "Т", "X": "Х"})
_WORD = re.compile(r"[A-Za-zА-Яа-яЁё]+")
_CYRILLIC = re.compile(r"[А-Яа-яЁё]")
_LATIN = re.compile(r"[A-Za-z]")


def _doc_code(match: "re.Match") -> str:
    code = match.group(2)
    if code in DOC_CODES:
        return match.group(0)
    for candidate in (code.translate(_CODE_LOOKALIKE),):
        # «3T» -> «ЭТ», «P3» -> «РЭ»: цифра 3 — это «Э» только в коде документа.
        if candidate in DOC_CODES:
            return f"{match.group(1)} {candidate}"
    return match.group(0)


def clean_spec_cell(text: Any) -> str:
    """
    Ячейка спецификации: латинские буквы-двойники внутри русского слова —
    кириллицей («Шaйба» -> «Шайба»), код документа после обозначения — по
    ГОСТ 2.102 («ТАДУ 687000.001 P3» -> «… РЭ»). Слово целиком латиницей
    («DIN», тип ввода «k») не трогается.
    """
    value = str(text or "")

    def word(match: "re.Match") -> str:
        token = match.group(0)
        if _CYRILLIC.search(token) and _LATIN.search(token):
            fixed = token.translate(_CYR_LOOKALIKE)
            if not _LATIN.search(fixed):
                return fixed
        return token

    value = _WORD.sub(word, value)
    return _CODE_AFTER.sub(_doc_code, value)


def clean_spec(table_data: Dict[str, Any]) -> Dict[str, Any]:
    rows = [[clean_spec_cell(cell) for cell in row] for row in table_data.get("rows") or []]
    return dict(table_data, rows=rows)


def _table_data(table: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Ответ модели -> `table_data`; таблица без единой непустой ячейки — None."""
    if not table:
        return None
    rows: List[List[str]] = [row for row in (table.get("rows") or []) if any(c.strip() for c in row)]
    if not rows:
        return None
    data: Dict[str, Any] = {
        "headers": list(table.get("headers") or []),
        "rows": rows,
        "total_row": None,
    }
    if table.get("title"):
        data["title"] = table["title"]
    return data


def _render_page(file_bytes: bytes, file_type: str, page: int, dpi: Optional[int] = None):
    """
    Страница `page` (с 1) -> PIL RGB. None — не открылась.

    `dpi` — разрешение рендера; по умолчанию `TABLE_VLM_DPI`. Своё
    разрешение нужно дочитыванию пустых рамок: у него свой потолок цены.
    """
    from PIL import Image

    if filetypes.is_image(file_type):
        if page != 1:
            return None
        return Image.open(io.BytesIO(file_bytes)).convert("RGB")
    import fitz

    document = fitz.open(stream=file_bytes, filetype="pdf")
    try:
        if not 1 <= page <= document.page_count:
            return None
        pixmap = document[page - 1].get_pixmap(dpi=int(dpi or settings.TABLE_VLM_DPI))
        return Image.open(io.BytesIO(pixmap.tobytes("png"))).convert("RGB")
    finally:
        document.close()


def _crop(image, bbox):
    from core.providers.drawing_vision import crop

    return crop(image, bbox, upscale=1.0)


def _upright(image):
    """
    Вырезка, повёрнутая так, чтобы текст стоял прямо. Решает OSD Tesseract
    и только при уверенном ответе: ложный поворот хуже никакого.
    """
    from core.providers.ocr_prepare import osd_rotation

    gray = image.convert("L")
    gray.thumbnail((2400, 2400))
    answer = osd_rotation(gray)
    if answer is None or answer[1] < settings.OCR_OSD_MIN_CONFIDENCE or not answer[0]:
        return image
    # OSD говорит, на сколько повернуть по часовой; PIL крутит против.
    return image.rotate(-answer[0], expand=True, fillcolor="white")
