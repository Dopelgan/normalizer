"""
Дочитывание пустой рамки таблицы по рендеру страницы (Tesseract).

Рамка попадает сюда, когда у неё нет ни собственного содержимого
(`table_data`), ни строк текстового слоя внутри, ни вырезки. До этой ветки
такая рамка уезжала в индекс фрагментом без текста или снималась уборкой —
и в обоих случаях её содержимое пропадало молча. На проде таких рамок 1180
в 376 документах; на замере 24 настоящих рамок 18 читаются, и в них лежат
штампы («СТО 04.18 03 Лист 56 из 56»), шапки таблиц («№ Наименование Дата
Уведомление План качества Договор») и надписи организации. Остальные 6 —
4 действительно пустые (Tesseract не отдаёт ни символа), 1 повёрнутый
мусор и 1 полушум.

Полушумная рамка — не наш случай: её содержимое лежит в текстовом слое
страницы (73 символа внутри рамки), и то, что сшивка его не отдала, —
отдельный дефект сопоставления. Затирать его дочитыванием нельзя, поэтому
затвор такую рамку отклоняет, а страница уходит непрочитанной.

Поворот рамку с мусором не спасает: на замере уверенность под всеми
четырьмя углами 62.6 / 61.0 / 56.1 / 57.9, а OSD отвечает 180° с
уверенностью 0.29 против `OCR_OSD_MIN_CONFIDENCE = 2.0`, то есть
`table_vlm._upright` на ней и так ничего не делает. Отдельной обработки
поворота здесь нет намеренно.

Единый слот `table_reader` сшивки обслуживает оба состояния блока: мятую
таблицу (ячейки разметкой) отдаём модели зрения, пустую рамку читаем
здесь. Два состояния исключают друг друга по `_carries_own_text`, поэтому
второй параметр сшивке не нужен.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

from celery.exceptions import SoftTimeLimitExceeded

from core import deadline
from core.config import settings
from core.models.parse_result import ParsedBlock
from core.providers import table_vlm
from core.providers.table_vlm import TableReader

logger = logging.getLogger(__name__)

#: Язык распознавания — тот же, что у фоллбэка и слоя распознавания.
_LANG = "rus+eng"
#: Вырезка рамки — однородный блок текста.
_PSM = 6
# Запас до срока задачи. На замере полоса читается 0.1-0.4 с, целый лист —
# до 4.9 с на `image_to_string`, и `image_to_data` стоит примерно столько
# же снова. 15 с покрывают худшую рамку с обоими проходами; меньше запаса —
# рамка не читается, и страница уходит непрочитанной, а не теряется по сроку.
_MIN_SECONDS_LEFT = 15.0


def table_reader(
    file_bytes: Optional[bytes], file_type: str, client=None
) -> Optional[TableReader]:
    """
    Функция «блок таблицы -> что удалось прочитать или None» для
    `reconcile_with_text_layer`, на оба состояния блока:

    * мятая таблица (`table_data` с разметкой) -> `table_data` модели зрения;
    * пустая рамка (`table_data` пуст) -> `{"text": ..., "confidence": ...}`.

    None — обе ветки выключены или файла нет.
    """
    vlm = table_vlm.table_reader(file_bytes, file_type, client=client)
    blank = _blank_reader(file_bytes, file_type)
    if vlm is None and blank is None:
        return None

    def read(block: ParsedBlock) -> Optional[Dict[str, Any]]:
        if _carries_own_text(block):
            return vlm(block) if vlm is not None else None
        return blank(block) if blank is not None else None

    return read


def _blank_reader(file_bytes: Optional[bytes], file_type: str) -> Optional[TableReader]:
    """Чтение пустых рамок документа с общим на документ пределом рамок."""
    if not settings.BLANK_TABLE_OCR_ENABLED or not file_bytes:
        return None

    pages: Dict[int, Any] = {}
    budget = {"left": max(0, int(settings.BLANK_TABLE_OCR_MAX_REGIONS))}

    def read(block: ParsedBlock) -> Optional[Dict[str, Any]]:
        if budget["left"] <= 0:
            logger.warning(
                "Пустая рамка на стр. %s не дочитана: исчерпан предел %d на документ",
                block.page, settings.BLANK_TABLE_OCR_MAX_REGIONS,
            )
            return None
        left = deadline.remaining()
        if left is not None and left < _MIN_SECONDS_LEFT:
            logger.warning("Пустая рамка на стр. %s не дочитана: до срока задачи %.0f с",
                           block.page, left)
            return None
        try:
            if block.page not in pages:
                pages[block.page] = table_vlm._render_page(
                    file_bytes, file_type, block.page, dpi=settings.BLANK_TABLE_OCR_DPI
                )
            page = pages[block.page]
            if page is None:
                return None
            piece = table_vlm._upright(table_vlm._crop(page, block.bbox))
        except SoftTimeLimitExceeded:
            raise
        except Exception as exc:  # noqa: BLE001 — сбой дочитывания не роняет разбор
            logger.warning("Пустая рамка на стр. %s не дочитана: %s", block.page, exc)
            return None

        pixels = int(piece.width) * int(piece.height)
        if pixels > max(0, int(settings.BLANK_TABLE_OCR_MAX_PIXELS)):
            logger.warning(
                "Пустая рамка на стр. %s не дочитана: вырезка %dx%d больше предела %d пкс",
                block.page, piece.width, piece.height, settings.BLANK_TABLE_OCR_MAX_PIXELS,
            )
            return None

        budget["left"] -= 1
        try:
            text, confidence = read_region(piece)
        except SoftTimeLimitExceeded:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.warning("Пустая рамка на стр. %s не прочитана: %s", block.page, exc)
            return None

        reason = reject_reason(text, confidence)
        if reason is not None:
            logger.info("Пустая рамка на стр. %s отклонена: %s (уверенность %.1f)",
                        block.page, reason, confidence)
            return None
        logger.info("Пустая рамка на стр. %s дочитана: символов %d, уверенность %.1f",
                    block.page, len(text), confidence)
        return {"text": text, "confidence": round(confidence / 100.0, 4)}

    return read


def read_region(image) -> Tuple[str, float]:
    """
    Текст вырезки и средняя уверенность слов (0-100).

    Уверенность — единственный признак, который на замере отличает
    читаемую рамку от мусора, поэтому берётся `image_to_data`, а не
    `image_to_string`: он стоит примерно столько же снова, и это
    сознательная плата.
    """
    import pytesseract

    from core.providers import ocr_model

    data = pytesseract.image_to_data(
        image, lang=_LANG,
        config=f"--psm {_PSM}{ocr_model.tessdata_option()}",
        output_type=pytesseract.Output.DICT,
    )
    words = []
    confidences = []
    for index, raw in enumerate(data.get("text") or []):
        word = (raw or "").strip()
        if not word:
            continue
        try:
            confidence = float(data["conf"][index])
        except (KeyError, IndexError, TypeError, ValueError):
            continue
        if confidence < 0:
            continue
        words.append(word)
        confidences.append(confidence)
    if not words:
        return "", 0.0
    return " ".join(words), sum(confidences) / len(confidences)


def reject_reason(text: str, confidence: float) -> Optional[str]:
    """
    Затвор прочитанной рамки: None — текст принят, строка — причина отказа.

    Четыре условия, и только одно из них разделяет данные замера:

    * `confidence` — средняя уверенность слов. Шум 62.6, полушум 66.1, все
      18 читаемых рамок 75.9-95.1: порог 70 стоит в настоящем разрыве.
    * длина, доля букв и односкриптовый токен — нижние полы, не признаки.
      На замере повёрнутый мусор держал долю односкриптовых 0.73 и долю
      букв 0.66, то есть лучше вполне читаемой рамки с 0.478 и 0.449, а
      два настоящих штампа несли по одному односкриптовому токену. Любое
      правило вида «не меньше трёх токенов» выбросило бы настоящий текст,
      поэтому требование токена действует только на длинном тексте, где
      одно слово уже не объяснение.
    """
    min_chars = max(1, int(settings.BLANK_TABLE_OCR_MIN_CHARS))
    min_confidence = float(settings.BLANK_TABLE_OCR_MIN_CONFIDENCE)
    min_alpha_share = float(settings.BLANK_TABLE_OCR_MIN_ALPHA_SHARE)

    dense = "".join(text.split())
    if len(dense) < min_chars:
        return f"символов {len(dense)} из {min_chars}"
    if confidence < min_confidence:
        return f"уверенность {confidence:.1f} из {min_confidence:.0f}"
    alpha = sum(1 for ch in dense if ch.isalpha())
    share = alpha / len(dense)
    if share < min_alpha_share:
        return f"доля букв {share:.2f} из {min_alpha_share:.2f}"
    # Длинный текст без ни одного целого слова одного алфавита — каша из
    # обрывков. На коротком штампе такого требования нет: «Лист 56 из 56».
    if len(dense) >= _TOKEN_CHARS and not _has_single_script_token(text):
        return "ни одного односкриптового слова"
    return None


#: Длина, с которой требуется хотя бы одно целое слово одного алфавита.
_TOKEN_CHARS = 12
#: Букв в таком слове.
_TOKEN_LETTERS = 4


def _has_single_script_token(text: str) -> bool:
    """Есть ли слово из >= 4 букв одного алфавита (кириллица или латиница)."""
    for token in text.split():
        letters = [ch for ch in token if ch.isalpha()]
        if len(letters) < _TOKEN_LETTERS:
            continue
        scripts = {_script(ch) for ch in letters}
        if len(scripts) == 1 and scripts != {""}:
            return True
    return False


def _script(ch: str) -> str:
    """Алфавит буквы: cyr, lat или пусто для всего остального."""
    code = ord(ch.lower())
    if 0x0410 <= code <= 0x044F or code in (0x0401, 0x0451):
        return "cyr"
    if 0x0061 <= code <= 0x007A:
        return "lat"
    return ""


def _carries_own_text(block: ParsedBlock) -> bool:
    """То же, что у сшивки: есть ли у таблицы собственное содержимое."""
    from core.providers.text_layer import _carries_own_text as carries

    return carries(block)
