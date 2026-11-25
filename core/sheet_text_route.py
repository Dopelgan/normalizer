"""
Текст листа-чертежа читает модель, а не OCR (C8, DRAWING_ROUTE_SHEET_TEXT_VLM).

Зачем. На скане чертежа MinerU делит страницу на рисунок и текстовые блоки.
Рисунок уходит в ветку чертежей, а штамп, технические требования (ТТ) и
технические характеристики остаются текстом распознавания. Лист 1 сборочного
чертежа так получил 0 граф штампа и мусор OCR в ТТ: «схеме сварки» вместо
«сборки», «128» вместо 138, «20+10» вместо 20±10. Модель эти же строки
читает верно, но её к ним не подпускали: решение «это чертёж» принималось
для области рисунка, а не для страницы.

Что делается. Страница признаётся чертежом, если на ней:
- рисунок не меньше DRAWING_ROUTE_SHEET_TEXT_MIN_SHARE площади листа или
  разбор листа целиком;
- или рисунок любого размера и рамка формата (ГОСТ 2.301) на самой странице;
- или весь документ — растровый чертёж по классификации лестницы.

На такой странице:
1. если полей основной надписи нет — штамп читается проходом штампа
   (`DrawingProcessor._stamp_page`), отдельным чертёжным фрагментом;
2. текстовые блоки вне рисунка перечитываются моделью по вырезке в
   разрешении DRAWING_ROUTE_SHEET_TEXT_DPI (`QwenVisionClient.transcribe`).
   Текст OCR остаётся, только если модель не ответила, ответила пусто или
   заметно короче (меньше `_KEEP_SHARE` знаков OCR) — это отказ, а не чтение.

Таблицы MinerU не трогаются: строки модели в ячейки не раскладываются, а
сломанная таблица хуже неточной.
"""

from __future__ import annotations

import contextvars
import logging
import math
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Sequence

from celery.exceptions import SoftTimeLimitExceeded

from core import deadline
from core.config import settings
from core.models.contract import Fragment

logger = logging.getLogger(__name__)

# Рисунок от такой доли листа — разбор листа целиком, а не вид: блоки под
# ним не исключаются, иначе страница не перечитывалась бы вовсе.
_WHOLE_SHEET = 0.8
# Блок, который на столько своей площади лежит в рисунке, ветка чертежей
# уже прочитала вместе с рисунком.
_INSIDE_PICTURE = 0.5
# Блок шире этой доли листа — рамка неизвестна (по умолчанию [0,0,1,1]):
# вырезать нечего.
_MAX_BLOCK_AREA = 0.6
# Ответ модели короче такой доли знаков OCR — отказ или обрыв, а не чтение.
_KEEP_SHARE = 0.3

SOURCE_VLM = "vlm_crop"
SOURCE_OCR = "ocr_fallback"
METHOD_VLM = "vlm_sheet_text"


def apply(processor, fragments: List[Fragment], metadata: Dict[str, Any]) -> List[Fragment]:
    """Штамп и текстовые блоки страниц-чертежей — модели. Без флага — те же фрагменты."""
    if not settings.DRAWING_ROUTE_SHEET_TEXT_VLM:
        return fragments
    source_uri = str(metadata.get("source_path") or "")
    if not source_uri or not processor.qwen.available:
        return fragments
    whole_document = processor._source_is_raster_drawing(metadata)
    pages = _candidate_pages(fragments, whole_document)
    budget = [max(0, int(settings.DRAWING_ROUTE_SHEET_TEXT_MAX_BLOCKS))]
    stamps: List[Fragment] = []
    with _budget():
        for page in pages:
            if not _time_left():
                logger.warning("Текст страниц-чертежей не перечитан: исчерпан срок задачи или бюджет чертежей")
                break
            try:
                stamp = _route_page(processor, fragments, metadata, source_uri, page, whole_document, budget)
            except (SoftTimeLimitExceeded, deadline.Cancelled):
                raise
            except Exception as exc:  # noqa: BLE001 — страница не роняет документ
                logger.error("Текст страницы-чертежа %d не перечитан: %s", page, exc)
                continue
            if stamp is not None:
                stamps.append(stamp)
    for stamp in stamps:
        # Штамп встаёт перед фрагментами своей страницы: порядок страниц сохраняется.
        index = next((i for i, f in enumerate(fragments)
                      if f.position is not None and f.position.page >= stamp.position.page), len(fragments))
        fragments.insert(index, stamp)
    return fragments


def attach_rejected(qwen, fragments: List[Fragment]) -> None:
    """
    Отказы модели за документ — в payload первого чертёжного фрагмента
    (DRAWING_VLM_REJECT_DETAILS): страницы у отказа нет, а без записи в
    результате разброс прогонов виден только в логе контейнера.
    """
    if not settings.DRAWING_VLM_REJECT_DETAILS:
        return
    rejected = list(getattr(qwen, "rejected_answers", None) or [])
    if not rejected:
        return
    for fragment in fragments:
        if fragment.type == "drawing":
            payload = fragment.content.structured_payload or {}
            payload["vlm_rejected"] = rejected
            fragment.content.structured_payload = payload
            return


# ------------------------------------------------------------------ страницы
def _candidate_pages(fragments: Sequence[Fragment], whole_document: bool) -> List[int]:
    """Страницы с текстовыми блоками, где есть рисунок (или весь документ — чертёж)."""
    pictured = {int(f.position.page) for f in fragments
                if f.position is not None and f.type in ("drawing", "image")}
    texted = {int(f.position.page) for f in fragments
              if f.position is not None and f.type == "text" and (f.content.text or "").strip()}
    stampless = {page for page in pictured if not _has_stamp(fragments, page)}
    pages = (texted | stampless) & pictured
    if whole_document:
        pages |= texted
    return sorted(pages)


def _route_page(processor, fragments, metadata, source_uri: str, page: int,
                whole_document: bool, budget: List[int]) -> Optional[Fragment]:
    on_page = [f for f in fragments if f.position is not None and int(f.position.page) == page]
    pictures = [f for f in on_page if f.type in ("drawing", "image")]
    image = _page_image(processor, source_uri, page)
    if image is None:
        return None
    if not (whole_document or _is_drawing_page(pictures, image, processor)):
        return None
    logger.info("Страница %d — чертёж: штамп и текст листа читает модель", page)
    stamp = None
    if not _has_stamp(fragments, page):
        stamp = processor._stamp_page(source_uri, metadata, page)
    crops = [_bbox(f) for f in pictures]
    crops = [box for box in crops if box is not None and _area(box) < _WHOLE_SHEET]
    for fragment in on_page:
        if fragment.type != "text" or not (fragment.content.text or "").strip():
            continue
        box = _bbox(fragment)
        if box is None or _area(box) > _MAX_BLOCK_AREA or _area(box) <= 0:
            continue
        if any(_inside(box, picture) >= _INSIDE_PICTURE for picture in crops):
            continue
        if budget[0] <= 0 or not _time_left():
            logger.warning("Текстовые блоки страницы %d не перечитаны: исчерпан предел вырезок или срок", page)
            break
        budget[0] -= 1
        try:
            _reread_block(processor, fragment, image, box)
        except (SoftTimeLimitExceeded, deadline.Cancelled):
            raise
        except Exception as exc:  # noqa: BLE001 — сбой блока не теряет штамп страницы
            # Штамп этой страницы уже прочитан: он возвращается, а блок
            # остаётся с текстом OCR и причиной в учёте.
            logger.error("Блок %s не перечитан: %s", fragment.fragment_id, exc)
            _note(fragment, {"source": SOURCE_OCR, "reason": "vlm_error"})
    return stamp


def _is_drawing_page(pictures: Sequence[Fragment], image, processor) -> bool:
    """Рисунок крупный, лист прочитан целиком или на странице рамка формата."""
    for picture in pictures:
        box = _bbox(picture)
        if box is not None and _area(box) >= settings.DRAWING_ROUTE_SHEET_TEXT_MIN_SHARE:
            return True
    if not pictures:
        return False
    from core.providers import drawing_vision

    if not processor.vision_enabled:
        return False
    return bool(drawing_vision.analyse(_analysis_image(image)).frame)


def _analysis_image(image):
    """
    Рамку формата ищем на растре в рабочем разрешении чертежей
    (DRAWING_RENDER_DPI), а не на рендере для вырезок: рамка видна и там, а
    разбор листа в 300 dpi в разы дольше и тяжелее по памяти.
    """
    dpi = float((getattr(image, "info", None) or {}).get("render_dpi") or 0)
    work = float(settings.DRAWING_RENDER_DPI or 0)
    if dpi <= 0 or work <= 0 or work >= dpi:
        return image
    scale = work / dpi
    return image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))))


def _has_stamp(fragments: Sequence[Fragment], page: int) -> bool:
    return any(
        f.type == "drawing" and f.position is not None and int(f.position.page) == page
        and any(field.category == "title_block" for field in f.content.structured_drawing_fields or [])
        for f in fragments
    )


# ------------------------------------------------------------------ вырезки
def _reread_block(processor, fragment: Fragment, image, box: List[float]) -> None:
    """Блок по вырезке моделью; текст OCR остаётся только при отказе модели."""
    from core.providers import drawing_vision

    ocr_text = fragment.content.text or ""
    # Поле вокруг рамки (padding вырезки) нужно: рамка MinerU прижата к
    # буквам, и край строки иначе срезается. Потолок увеличения — тот же,
    # с которым вырезка уйдёт модели, иначе её ужмут обратно.
    piece = drawing_vision.crop(image, box, max_side=settings.DRAWING_TRANSCRIBE_MAX_SIDE)
    lines = processor._fidelity(processor.qwen.transcribe(piece))
    vlm_text = "\n".join(line for line in lines or [] if line.strip())
    ocr_chars, vlm_chars = _visible(ocr_text), _visible(vlm_text)
    note: Dict[str, Any] = {"ocr_chars": ocr_chars, "vlm_chars": vlm_chars}
    if not vlm_text:
        note.update(source=SOURCE_OCR, reason="vlm_no_answer" if lines is None else "vlm_empty")
    elif vlm_chars < _KEEP_SHARE * ocr_chars:
        note.update(source=SOURCE_OCR, reason="vlm_shorter")
    else:
        note.update(source=SOURCE_VLM)
        fragment.content.text = vlm_text
        if fragment.provenance is not None:
            fragment.provenance.method = METHOD_VLM
            fragment.provenance.strategy_level = 7
        position = fragment.position
        # Смещения и строки с рамками считались по тексту OCR: к тексту модели
        # они не относятся, и ссылаться на них — ссылаться на чужой текст.
        position.source_spans = []
        position.text_offsets_exact = False
        position.lines = None
    _note(fragment, note)
    logger.info("Блок %s: %s (OCR %d знаков, модель %d)", fragment.fragment_id,
                note["source"], ocr_chars, vlm_chars)


def _note(fragment: Fragment, note: Dict[str, Any]) -> None:
    payload = fragment.content.structured_payload or {}
    payload["sheet_text_route"] = note
    fragment.content.structured_payload = payload


def render_dpi(width_pt: float, height_pt: float) -> float:
    """
    dpi рендера страницы для вырезок: DRAWING_ROUTE_SHEET_TEXT_DPI, но не
    больше DRAWING_ROUTE_SHEET_TEXT_RENDER_MAX_MP мегапикселей на страницу
    (A0 в 300 dpi — 140 Мп, 420 МБ в памяти) — как у плиток в родном
    разрешении (`native_tiles.render_dpi`).
    """
    target = float(settings.DRAWING_ROUTE_SHEET_TEXT_DPI)
    limit = float(settings.DRAWING_ROUTE_SHEET_TEXT_RENDER_MAX_MP) * 1e6
    width_in, height_in = width_pt / 72.0, height_pt / 72.0
    if width_in > 0 and height_in > 0 and limit > 0:
        target = min(target, math.sqrt(limit / (width_in * height_in)))
    return target


def _page_image(processor, source_uri: str, page: int):
    """
    Страница в рабочем разрешении вырезок текста. PDF рендерится заново в
    DRAWING_ROUTE_SHEET_TEXT_DPI: мелкий шрифт ТТ при 200 dpi модель путает
    («128» и «138»), растр берётся как есть — это и есть его разрешение.
    """
    data = processor._read(source_uri)
    if not data:
        return None
    if data[:5] != b"%PDF-":
        return processor._open(source_uri, page)
    from core import drawing_processor

    try:
        import fitz
        from PIL import Image
    except ImportError:  # pragma: no cover
        return None
    with drawing_processor._PDF_RENDER_LOCK:
        try:
            document = fitz.open(stream=data, filetype="pdf")
        except Exception as exc:  # noqa: BLE001
            logger.error("PDF не открылся для вырезок текста: %s", exc)
            return None
        try:
            if not 1 <= page <= document.page_count:
                return None
            rect = document[page - 1].rect
            dpi = max(1, int(render_dpi(rect.width, rect.height)))
            pixmap = document[page - 1].get_pixmap(dpi=dpi)
            image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
            image.info["render_dpi"] = dpi
            return image
        finally:
            document.close()


# Начало бюджета прохода на часах `gpu_gate.budget_now` (без ожидания
# места на видеокарте); None — вне `_budget`.
_started: contextvars.ContextVar[Optional[float]] = contextvars.ContextVar(
    "sheet_text_route_started", default=None)


@contextmanager
def _budget():
    """
    Проход по документу укладывается в DRAWING_TIME_BUDGET — тот же учёт,
    что у листа в ветке чертежей: свои часы без ожидания видеокарты.
    """
    from core.providers import gpu_gate

    with gpu_gate.budget_clock():
        token = _started.set(gpu_gate.budget_now())
        try:
            yield
        finally:
            _started.reset(token)


def _time_left() -> bool:
    """Хватит ли на вызов модели и срока задачи, и бюджета чертежей."""
    left = deadline.remaining()
    if left is not None and left < settings.VLM_TIMEOUT:
        return False
    budget = settings.DRAWING_TIME_BUDGET
    started = _started.get()
    if budget and started is not None:
        from core.providers import gpu_gate

        if gpu_gate.budget_now() + settings.VLM_TIMEOUT > started + budget:
            return False
    return True


def _bbox(fragment: Fragment) -> Optional[List[float]]:
    """Рамка в долях листа или None (рамки нет или она не в долях)."""
    raw = fragment.position.bbox if fragment.position is not None else None
    if not raw or len(raw) != 4:
        return None
    box = [float(v) for v in raw]
    if min(box) < 0.0 or max(box) > 1.0 or box[2] <= box[0] or box[3] <= box[1]:
        return None
    return box


def _area(box: Sequence[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _inside(box: Sequence[float], other: Sequence[float]) -> float:
    """Какая доля `box` лежит в `other`."""
    width = min(box[2], other[2]) - max(box[0], other[0])
    height = min(box[3], other[3]) - max(box[1], other[1])
    area = _area(box)
    if width <= 0 or height <= 0 or area <= 0:
        return 0.0
    return width * height / area


def _visible(text: str) -> int:
    return sum(1 for ch in text if not ch.isspace())
