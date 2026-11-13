"""
Перечитывание неразобранной таблицы скана моделью зрения (Qwen-VL).

Таблица уходит сюда, когда модель таблиц MinerU отдала ячейки разметкой, а
собрать их заново по строкам OCR не вышло. Типичный случай — альбомная
таблица на книжном листе паспорта: MinerU читает её поперёк, Tesseract
тоже, и строки слоя — каша из обрывков. Раньше такая таблица разбиралась
на эти обрывки (`table_from_lines.degrade`): значения ячеек терялись.

Здесь страница рендерится, таблица вырезается по рамке MinerU, ставится
прямо по OSD Tesseract и читается моделью. Не вышло на любом шаге — None,
и вызывающий разбирает таблицу на текст, как раньше: эта ветка только
добавляет шанс, но не отнимает прежний исход.
"""

from __future__ import annotations

import io
import logging
from typing import Any, Callable, Dict, List, Optional

from celery.exceptions import SoftTimeLimitExceeded

from core import filetypes
from core.config import settings
from core.models.parse_result import ParsedBlock

logger = logging.getLogger(__name__)

TableReader = Callable[[ParsedBlock], Optional[Dict[str, Any]]]


def table_reader(file_bytes: Optional[bytes], file_type: str, client=None) -> Optional[TableReader]:
    """
    Функция «блок таблицы -> table_data или None» для `reconcile_with_text_layer`.
    None — ветка выключена, файла нет или модели нет: тогда всё как раньше.
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
        budget["left"] -= 1
        try:
            if block.page not in pages:
                pages[block.page] = _render_page(file_bytes, file_type, block.page)
            page = pages[block.page]
            if page is None:
                return None
            piece = _upright(_crop(page, block.bbox))
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


def _render_page(file_bytes: bytes, file_type: str, page: int):
    """Страница `page` (с 1) -> PIL RGB. None — не открылась."""
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
        pixmap = document[page - 1].get_pixmap(dpi=settings.TABLE_VLM_DPI)
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
