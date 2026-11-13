"""
Клиент MinerU (контейнер mineru-api, `MINERU_ENDPOINT`) с фоллбэком на Tesseract.

Вызывается уровнем 5 лестницы (`core.ladder.strategies.recognition`, метод
`mineru_ocr`). На вход — PDF или картинка из хранилища, на выход —
`ParseResult` с блоками в долях страницы [0, 1].

Как устроено:

* запрос — `POST /file_parse` с multipart-загрузкой; таймаут —
  `MINERU_TIMEOUT`, на длинном документе больше: `MINERU_TIMEOUT_PER_PAGE`
  на страницу, но в пределах срока задачи (`_timeout_for`);
* приоритет отдаётся `middle_json` — только он содержит координаты блоков
  и `page_size`, без которых bbox нельзя нормализовать в [0, 1];
* изображения запрашиваются у MinerU (`return_images`) и выкладываются в
  хранилище, иначе этап обработки чертежей остаётся без входных данных;
* таблицы разбираются из `table_body` (HTML), как их реально отдаёт MinerU;
* сетевые сбои (фоллбэк уместен) отделены от ошибок разбора (это дефект);
  недоступность запоминается на процесс (`UNAVAILABLE_TTL`).

Текст и структура сводятся в `parse`. У векторного PDF текст берётся из
слоя файла (`core.providers.text_layer`), у растра — из слоя построчного
распознавания (`TesseractFallbackProvider.text_layer`, `OCR_LAYER_ENABLED`);
структура (таблицы, формулы, картинки) остаётся за MinerU. Флаги сшивки:
`OCR_LAYER_DROP_LOW_CONFIDENCE` с порогом `OCR_LAYER_MIN_LINE_CONFIDENCE`
(только растр) и `FRAME_TABLE_GUARD_ENABLED` (оба слоя).
"""

import base64
import json
import logging
import os
import re
import time
from dataclasses import replace
from difflib import SequenceMatcher
from html.parser import HTMLParser
from typing import Any, Dict, List, Optional, Tuple

import requests

from core import filetypes
from celery.exceptions import SoftTimeLimitExceeded

from core import deadline
from core.config import settings
from core.models.parse_result import ParsedBlock, ParseResult
from core.providers.document_parser import (
    DocumentParserProvider,
    ParserFailed,
    ParserUnavailable,
)
from core.providers import dimension_fidelity, ocr_cleanup, table_vlm
from core.providers.storage import StorageProvider, StorageProviderFactory
from core.providers.tesseract_fallback import TesseractFallbackProvider
from core.providers.vector_drawing import (
    DrawingPage,
    drawing_pages_from_document,
    extract_drawing_fields,
)
from core.providers.text_layer import (
    METHOD_OCR_LAYER_RECOVERED,
    METHOD_PARSER_OCR,
    OCR_LAYER,
    TextLayer,
    group_lines,
    layer_from_document,
    lines_to_text,
    merge_line_bboxes,
    open_pdf,
    page_turn,
    reconcile_with_text_layer,
)

logger = logging.getLogger(__name__)


# Столько символов текста MinerU на странице, чтобы считать его основным.
_MINERU_PAGE_TEXT_CHARS = 200


def drop_low_confidence_lines(
    layer: TextLayer, blocks: List[ParsedBlock], threshold: float
) -> TextLayer:
    """
    Слой распознавания без строк ниже `threshold` — но только на страницах,
    где у MinerU есть свой текст. Там мусор Tesseract («едприяувие = Fe SES»)
    дублирует MinerU и уходит в индекс отдельными блоками. На странице, где
    MinerU текста не дал, слой — единственный источник, и он не трогается.
    """
    # Колонтитул «Лист 3» — ещё не текст страницы: тело могло уехать в
    # формулу или таблицу, и тогда слой — единственный его источник.
    mineru_chars: Dict[int, int] = {}
    for b in blocks:
        if b.type in _TEXTISH and b.text:
            mineru_chars[b.page] = mineru_chars.get(b.page, 0) + len("".join(b.text.split()))
    mineru_pages = {p for p, n in mineru_chars.items() if n >= _MINERU_PAGE_TEXT_CHARS}
    pages: Dict[int, List[Any]] = {}
    dropped = 0
    for page, lines in layer.pages.items():
        if page in mineru_pages:
            kept = [l for l in lines if l.confidence >= threshold]
            dropped += len(lines) - len(kept)
            pages[page] = kept
        else:
            pages[page] = list(lines)
    if dropped:
        logger.info("Слой распознавания: отброшено строк с уверенностью < %.2f: %d",
                    threshold, dropped)
    return replace(layer, pages=pages)

_RUS_WORD_RE = re.compile(r"[А-Яа-яЁё]{4,}")
_LAT_WORD_RE = re.compile(r"[A-Za-z]{4,}")
# Основа слова для сравнения копий: окончания у Tesseract и MinerU расходятся
# («определятся» / «определяются»), начало слова — реже.
_STEM = 5


def _stems(text: str) -> List[str]:
    return [w.lower().replace("ё", "е")[:_STEM] for w in _RUS_WORD_RE.findall(text or "")]


# Обозначения, которые нельзя терять вместе с «графикой»: код KKS
# («10JEB10CT001»), диаметр («⌀14»), радиус или резьба с дробной частью
# («R1,5», «M2,5»). Без русских слов они похожи на мусор чертежа, но это
# данные, и если у MinerU/модели на странице их нет — блок остаётся.
# «R263» и «125» из графики под правило не попадают: нет дроби и букв кода.
_DESIGNATION_RE = re.compile(
    r"(?<![A-Z0-9])\d{0,3}[A-Z]{2,}\d{1,}[A-Z0-9]*\d(?![A-Z0-9])"
    r"|[⌀Ø]\d+(?:[.,]\d+)?"
    r"|(?<![A-Z0-9])[RM]\d+[.,]\d+"
)
# Кириллица, похожая на латиницу: Tesseract и MinerU путают «СТ» и «CT».
_LOOKALIKE = str.maketrans("АВЕКМНОРСТХУ", "ABEKMHOPCTXY")


def _norm_codes(text: str) -> str:
    text = (text or "").upper().translate(_LOOKALIKE).replace("Ø", "⌀")
    return re.sub(r"⌀\s+", "⌀", text)


def _designations(text: str) -> List[str]:
    return [m.group(0) for m in _DESIGNATION_RE.finditer(_norm_codes(text))]


def page_ranges(pages: List[int]) -> str:
    """«51-158» вместо списка из ста восьми чисел — для лога и пометки."""
    ranges: List[str] = []
    for page in sorted(set(pages)):
        if ranges and page == _range_end(ranges[-1]) + 1:
            ranges[-1] = f"{ranges[-1].split('-')[0]}-{page}"
        else:
            ranges.append(str(page))
    return ", ".join(ranges)


def _range_end(chunk: str) -> int:
    return int(chunk.split("-")[-1])


def _contained_share(part: List[str], whole: List[str]) -> float:
    """
    Какая доля основ `part` нашлась в `whole` одной последовательностью.

    Сравнение идёт по порядку слов: совпадающие отрезки
    `SequenceMatcher` монотонны в обеих последовательностях, поэтому
    случайные пересечения словаря в счёт не идут — только тот же текст,
    прочитанный с пропусками и другими окончаниями.
    """
    if not part:
        return 0.0
    matcher = SequenceMatcher(a=part, b=whole, autojunk=False)
    matched = sum(block.size for block in matcher.get_matching_blocks())
    return matched / len(part)


def drop_repeated_recovered_blocks(
    blocks: List[ParsedBlock], min_contained: float
) -> Tuple[List[ParsedBlock], Dict[str, int]]:
    """
    Блоки, восстановленные из слоя распознавания, без повторов текста MinerU.

    Только на страницах, где у MinerU свой текст (как у
    `drop_low_confidence_lines`): там восстановленный блок либо повторяет
    этот текст хуже, либо собран из графики («a С» 125) \/ R263»).

    Повтором считается блок, который целиком содержится в одном другом
    блоке страницы: не меньше `min_contained` его основ нашлись в основах
    того блока одной последовательностью, по порядку слов. Прежнее правило
    сравнивало блок с объединением всех слов страницы и без порядка, и
    этого хватало, чтобы потерять настоящий текст: предложение «проводится
    расследование. Покупатель письменно извещает Поставщика о необходимости
    прибытия для расследования выявленных несоответствий.» делит с
    соседним фрагментом только хвост «о необходимости прибытия…», а
    остальные его слова нашлись по странице вразнобой — доля вышла выше
    порога, и пункт договора остался без сказуемого. По новому правилу у
    него совпадает 7 основ из 11 с соседом (0.64) и меньше с остальными: ни
    сравнение порога не достигает, и блок остаётся.

    Второе правило — графика: блок без русских слов на русской странице
    (русских слов у MinerU не меньше латинских). Отбрасывается только
    заведомый мусор обводки — ни русского, ни латинского слова от четырёх
    букв. Блок, который на текст всё-таки похож, остаётся и считается в
    `kept_uncertain`: сомнительное удаление заменено сомнительным
    сохранением, потому что лишняя строка в индексе дешевле потерянной.

    На странице без текста MinerU слой — единственный источник и не
    трогается. Блок с обозначением (KKS, ⌀, R1,5), которого нет в тексте
    страницы у MinerU/модели, не отбрасывается ни по одному из правил.

    Возвращает оставшиеся блоки и счётчики: `repeat` — доказанных повторов,
    `graphics` — мусора обводки, `kept_uncertain` — сохранённых спорных,
    `total` — всего удалено.
    """
    page_chars: Dict[int, int] = {}
    # Основы каждого блока страницы по отдельности: повтор доказывается
    # сравнением с одним блоком, а не с объединением слов страницы.
    page_block_stems: Dict[int, List[List[str]]] = {}
    # Русских и латинских слов в тексте MinerU страницы: правило «нет
    # русского слова — графика» верно только для русской страницы. На
    # английском скане восстановленный блок — текст, которого у MinerU нет.
    page_rus: Dict[int, int] = {}
    page_lat: Dict[int, int] = {}
    # Весь текст страницы помимо слоя (любые типы блоков, включая ответы
    # модели по чертежу) — для проверки обозначений.
    page_codes: Dict[int, List[str]] = {}
    for b in blocks:
        if b.method == METHOD_OCR_LAYER_RECOVERED:
            continue
        text = b.text or ""
        page_codes.setdefault(b.page, []).append(_norm_codes(text))
        if b.type == "table" and b.table_data:
            cells = [str(h) for h in (b.table_data.get("headers") or [])]
            for row in b.table_data.get("rows") or []:
                cells.extend(str(c) for c in row)
            text = " ".join(cells)
            page_codes[b.page].append(_norm_codes(text))
        if b.type in _TEXTISH or b.type == "table":
            stems = _stems(text)
            if stems:
                page_block_stems.setdefault(b.page, []).append(stems)
            page_rus[b.page] = page_rus.get(b.page, 0) + len(stems)
            page_lat[b.page] = page_lat.get(b.page, 0) + len(_LAT_WORD_RE.findall(text))
        if b.type in _TEXTISH and b.text:
            page_chars[b.page] = page_chars.get(b.page, 0) + len("".join(b.text.split()))
    kept: List[ParsedBlock] = []
    counts = {"repeat": 0, "graphics": 0, "kept_uncertain": 0, "total": 0}
    for b in blocks:
        if (b.method == METHOD_OCR_LAYER_RECOVERED
                and page_chars.get(b.page, 0) >= _MINERU_PAGE_TEXT_CHARS):
            text = b.text or ""
            stems = _stems(text)
            russian_page = page_rus.get(b.page, 0) > 0 and (
                page_rus.get(b.page, 0) >= page_lat.get(b.page, 0))
            repeat = any(
                _contained_share(stems, other) >= min_contained
                for other in page_block_stems.get(b.page, ())
            ) if stems else False
            # Мусор обводки: ни русского, ни латинского слова — читать в нём
            # нечего. Латиница от четырёх букв означает, что блок всё-таки
            # похож на текст, пусть и прочитанный неверно.
            graphics = (not stems and russian_page
                        and not _LAT_WORD_RE.search(text))
            uncertain = not stems and russian_page and not graphics
            if repeat or graphics:
                # Пробелы внутри обозначения у MinerU («10JEB10 CT001») не в счёт.
                page_text = "".join("|".join(page_codes.get(b.page, [])).split())
                if any(code not in page_text for code in _designations(text)):
                    kept.append(b)
                    continue
                counts["repeat" if repeat else "graphics"] += 1
                counts["total"] += 1
                continue
            if uncertain:
                counts["kept_uncertain"] += 1
        kept.append(b)
    if counts["total"] or counts["kept_uncertain"]:
        logger.info(
            "Слой распознавания: доказанных повторов %d, мусора обводки %d, "
            "спорных блоков сохранено %d",
            counts["repeat"], counts["graphics"], counts["kept_uncertain"],
        )
    return kept, counts


# Типы блоков MinerU, которые сводятся к текстовому.
_TEXTISH = {"text", "title", "paragraph", "list", "header", "footer", "index", "text_level"}
_FORMULA = {"formula", "equation", "interline_equation", "inline_equation", "isolate_formula"}


# ===========================================================================
# Разбор HTML-таблиц MinerU
# ===========================================================================

class _TableHTMLParser(HTMLParser):
    """Минимальный разбор `<table>` из ответа MinerU в headers/rows."""

    def __init__(self) -> None:
        super().__init__()
        self.rows: List[List[str]] = []
        self.header_flags: List[bool] = []
        self._row: Optional[List[str]] = None
        self._cell: Optional[List[str]] = None
        self._row_is_header = False

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self._row = []
            self._row_is_header = False
        elif tag in ("td", "th"):
            self._cell = []
            if tag == "th":
                self._row_is_header = True

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self._cell is not None and self._row is not None:
            self._row.append(" ".join("".join(self._cell).split()))
            self._cell = None
        elif tag == "tr" and self._row is not None:
            if any(c for c in self._row):
                self.rows.append(self._row)
                self.header_flags.append(self._row_is_header)
            self._row = None

    def handle_data(self, data):
        if self._cell is not None:
            self._cell.append(data)


def parse_table_html(html: str) -> Optional[Dict[str, Any]]:
    """HTML таблицы -> {'headers': [...], 'rows': [[...]]}. None, если пусто."""
    if not html or "<t" not in html.lower():
        return None
    parser = _TableHTMLParser()
    try:
        parser.feed(html)
        parser.close()
    except Exception as exc:  # noqa: BLE001 — битый HTML не должен ронять парсинг
        logger.warning("Не удалось разобрать HTML таблицы: %s", exc)
        return None

    if not parser.rows:
        return None

    headers: List[str] = []
    rows = parser.rows
    if parser.header_flags and parser.header_flags[0]:
        headers = rows[0]
        rows = rows[1:]
    elif len(rows) > 1:
        # Заголовков <th> нет — считаем заголовком первую строку.
        headers = rows[0]
        rows = rows[1:]

    width = max([len(headers)] + [len(r) for r in rows]) if (headers or rows) else 0
    headers = headers + [""] * (width - len(headers))
    rows = [r + [""] * (width - len(r)) for r in rows]
    return {"headers": headers, "rows": rows, "total_row": None}


# ===========================================================================
# Парсер
# ===========================================================================

class ParserTimedOut(ParserUnavailable):
    """MinerU доступен, но не уложился в таймаут: деградация документа, не сервиса."""


class MinerUParserProvider(DocumentParserProvider):
    # Когда сервис не отвечает, об этом помнит весь процесс: иначе каждый
    # уровень лестницы заново ждёт таймаут соединения и заново прогоняет
    # полный OCR — на сканах это удваивает время разбора и ничего не
    # добавляет к результату.
    _unavailable_until: float = 0.0
    UNAVAILABLE_TTL = 60.0

    @classmethod
    def mark_unavailable(cls) -> None:
        cls._unavailable_until = time.time() + cls.UNAVAILABLE_TTL

    @classmethod
    def is_available(cls) -> bool:
        return time.time() >= cls._unavailable_until

    @classmethod
    def reset_availability(cls) -> None:
        """Нужна тестам и ручной проверке после подъёма сервиса."""
        cls._unavailable_until = 0.0

    def __init__(self, storage: Optional[StorageProvider] = None):
        self.storage = storage or StorageProviderFactory.default()
        self.tesseract = TesseractFallbackProvider(storage=self.storage)
        self.mineru_endpoint = settings.MINERU_ENDPOINT.rstrip("/")
        self.timeout = settings.MINERU_TIMEOUT

    # ------------------------------------------------------------- публичное
    def parse(
        self,
        uri: str,
        file_type: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ParseResult:
        metadata = metadata or {}

        # Текстовый слой читается до похода в MinerU: он нужен и как источник
        # истины по тексту, и как страховка, если MinerU недоступен. Вместе с
        # ним с того же открытого документа снимается вердикт по чертёжным
        # листам — файл за проход открывается один раз.
        file_bytes = self._read_source(uri)
        layer, drawing_verdicts, pages = self._read_pdf(file_bytes, file_type)

        try:
            result = self._parse_with_mineru(uri, file_type, metadata, file_bytes, pages)
        except ParserUnavailable as exc:
            # Недоступность сервиса — это деградация системы, а не свойство
            # документа, и путать их нельзя: лежащий mineru-api иначе
            # неотличим от плохого скана, приём «работает», а качество падает
            # молча и на всём потоке сразу.
            note = (f"mineru_timeout: {exc}" if isinstance(exc, ParserTimedOut)
                    else f"mineru_unavailable: {exc}")
            if layer is not None:
                logger.error(
                    "MinerU недоступен (%s), но у %s есть текстовый слой — читаем его", exc, uri
                )
                degraded = self._parse_from_text_layer(layer)
            else:
                logger.error("MinerU недоступен (%s), полный фоллбэк на Tesseract: %s", exc, uri)
                degraded = self._full_tesseract_fallback(uri, file_type)
            degraded.degraded.append(note)
            return degraded

        if layer is not None:
            # Векторный PDF: текст берём из слоя, структуру оставляем за MinerU.
            blocks, stats = reconcile_with_text_layer(
                result.blocks, layer,
                frame_table_guard=settings.FRAME_TABLE_GUARD_ENABLED,
            )
            blocks = self._apply_vector_drawings(blocks, layer, drawing_verdicts)
            dimension_fixes = self._fix_dimensions(blocks, drawing_verdicts)
            result.blocks = blocks
            result.text_layer_chars = stats.layer_chars_by_page
            result.text_layer_stats = stats.as_dict()
            if dimension_fixes:
                result.text_layer_stats["dimension_fixes"] = dimension_fixes
            return result

        # Растр: слоя в файле нет, но его можно собрать распознаванием. Схема
        # та же, что у вектора, — структура от MinerU, текст со стороны:
        # модель таблиц MinerU отдаёт русскую прозу побуквенной LaTeX-
        # разметкой, а Tesseract с `rus` читает её как текст.
        if self._has_content(result):
            ocr_layer, unread_pages = self._ocr_layer(uri, file_type, total_pages=pages)
            if ocr_layer is not None and settings.OCR_LAYER_DROP_LOW_CONFIDENCE:
                ocr_layer = drop_low_confidence_lines(
                    ocr_layer, result.blocks, settings.OCR_LAYER_MIN_LINE_CONFIDENCE
                )
            vocab = ocr_cleanup.load_vocabulary() if settings.OCR_WORD_FIX_ENABLED else None
            word_fix = ocr_cleanup.CleanupStats()
            if ocr_layer is not None and vocab is not None:
                ocr_layer, word_fix = ocr_cleanup.clean_layer(ocr_layer, vocab)
            if ocr_layer is not None:
                blocks, stats = reconcile_with_text_layer(
                    result.blocks, ocr_layer, source=OCR_LAYER,
                    frame_table_guard=settings.FRAME_TABLE_GUARD_ENABLED,
                    keep_structured_tables=settings.OCR_KEEP_STRUCTURED_TABLES,
                    table_line_min_confidence=settings.OCR_TABLE_LINE_MIN_CONFIDENCE,
                    split_columns=settings.OCR_SPLIT_WIDE_GAPS,
                    table_reader=table_vlm.table_reader(file_bytes, file_type),
                )
                if vocab is not None:
                    # Ячейки таблиц и текст MinerU, не перекрытые слоем.
                    word_fix.add(ocr_cleanup.clean_blocks(blocks, vocab))
                dimension_fixes = self._fix_dimensions(blocks, drawing_verdicts)
                repeated = None
                if settings.OCR_LAYER_DROP_REPEATED:
                    blocks, repeated = drop_repeated_recovered_blocks(
                        blocks, settings.OCR_LAYER_REPEAT_CONTAIN
                    )
                result.blocks = blocks
                result.text_layer_chars = stats.layer_chars_by_page
                result.text_layer_stats = {**stats.as_dict(), "layer": "ocr"}
                if dimension_fixes:
                    result.text_layer_stats["dimension_fixes"] = dimension_fixes
                if vocab is not None:
                    result.text_layer_stats["word_fix"] = word_fix.as_dict()
                if repeated is not None:
                    # Доказанных повторов — отдельно от мусора обводки и от
                    # спорных блоков, которые оставлены в тексте. Одно число
                    # «отброшено N» не позволяло отличить верное удаление от
                    # потери, и спросить было не у чего.
                    result.text_layer_stats["dropped_repeated"] = repeated["total"]
                    result.text_layer_stats["dropped_detail"] = repeated
                    if repeated["kept_uncertain"]:
                        result.degraded.append(
                            f"ocr_layer_uncertain_kept: спорных блоков слоя "
                            f"оставлено в тексте {repeated['kept_uncertain']}"
                        )
                # Происхождение известно только здесь: слоя в файле не
                # оказалось и текст собран распознаванием. До этой ветки
                # `source_kind` ставился по расширению, и скан уезжал в
                # провенанс векторным PDF.
                result.source_kind = (
                    "image" if filetypes.is_image(file_type) else "scanned_pdf"
                )
                if stats.degraded_tables:
                    result.degraded.append(
                        f"table_not_parsed: таблиц разобрано на текст и формулы "
                        f"{stats.degraded_tables}"
                    )
                if stats.decoded_formulas:
                    # Проза под разметкой, которой нечем заменить: текст
                    # остался нечитаемым, и молчать об этом нельзя.
                    result.degraded.append(
                        f"formula_not_text: блоков с разметкой вместо текста "
                        f"{stats.decoded_formulas}"
                    )
                if unread_pages:
                    # Блоки MinerU на этих страницах есть, покрытие страниц
                    # полное, но текста на них не прочитано: без этой записи
                    # две трети документа пропадали молча.
                    result.unread_text_pages = unread_pages
                    result.degraded.append(
                        f"text_not_read: распознавание не дошло до страниц "
                        f"{page_ranges(unread_pages)} "
                        f"(предел OCR_FULL_MAX_PAGES или бюджет времени)"
                    )
                return result

        # MinerU ответил, но текста не дал — типично для скана без OCR-слоя.
        if not self._has_text(result):
            logger.warning("MinerU не вернул текста для %s, пробуем Tesseract", uri)
            fallback = self._full_tesseract_fallback(uri, file_type)
            if fallback.blocks:
                # Сохраняем найденные MinerU изображения и таблицы.
                non_text = [b for b in result.blocks if b.type in ("image", "table")]
                fallback.blocks = fallback.blocks + non_text
                fallback.raw_output = result.raw_output
                fallback.degraded.append(
                    "mineru_no_text: MinerU ответил без текста, страница прочитана OCR"
                )
                return fallback
        return result

    # ------------------------------------------------------- текстовый слой
    def _read_source(self, uri: str) -> Optional[bytes]:
        try:
            return self.storage.read_bytes(uri)
        except Exception as exc:  # noqa: BLE001 — ошибку поднимет _parse_with_mineru
            logger.warning("Не удалось прочитать файл из хранилища: %s: %s", uri, exc)
            return None

    @staticmethod
    def _read_pdf(
        file_bytes: Optional[bytes], file_type: str
    ) -> Tuple[Optional[TextLayer], Dict[int, DrawingPage], int]:
        """
        Всё, что читается из самого файла, за одно открытие: текстовый слой
        векторного PDF и вердикт по чертёжным листам. Раньше файл открывался
        на каждый вопрос к нему отдельно — на альбоме чертежей это давало
        несколько полных проходов по страницам за один разбор.

        Вердикт по чертежам снимается только там, где слой пригоден: для
        скана он всё равно не нужен, а `get_drawings()` на каждой странице
        стоит дорого.
        """
        if not file_bytes or file_type.lower() != "pdf":
            return None, {}, 1

        document = open_pdf(file_bytes)
        if document is None:
            return None, {}, 1
        try:
            pages = max(1, document.page_count)
            layer = layer_from_document(document)
            if not layer.is_usable():
                logger.info("Текстовый слой пуст или разрежен — работаем по распознаванию")
                return None, {}, pages
            return layer, drawing_pages_from_document(document), pages
        finally:
            document.close()

    def _ocr_layer(
        self, uri: str, file_type: Optional[str] = None, total_pages: int = 0,
    ) -> Tuple[Optional[TextLayer], List[int]]:
        """
        Построчное распознавание растра как слой и страницы, текст которых
        прочитать не удалось. Недоступность OCR здесь не фатальна: без слоя
        разбор останется как был, поэтому ошибка уходит в лог, а не наружу.

        Проход по страницам ограничен `OCR_FULL_MAX_PAGES` и бюджетом
        времени. Прежде усечение прохода здесь просто терялось: слой
        обрывался на полусотне страниц, MinerU при этом отдавал блоки по
        всему документу, покрытие страниц выходило полным — и документ со
        ста непрочитанными страницами выдавался как разобранный целиком.
        Теперь проход продолжается с места обрыва (`OCR_LAYER_CONTINUATION_
        PASSES` дополнительных проходов), а то, что не прочитано и после
        них, возвращается списком страниц и доезжает до выдачи.
        """
        if not settings.OCR_LAYER_ENABLED:
            return None, []
        pages: Dict[int, Any] = {}
        first = 1
        truncated: Optional[tuple] = None
        for _ in range(max(0, settings.OCR_LAYER_CONTINUATION_PASSES) + 1):
            try:
                part = self.tesseract.text_layer(uri, file_type, first_page=first)
            except SoftTimeLimitExceeded:
                raise
            except Exception as exc:  # noqa: BLE001 — слой не обязателен
                logger.warning("Слой распознавания для %s не собрался: %s", uri, exc)
                break
            truncated = getattr(self.tesseract, "truncated_at", None)
            if part is not None:
                pages.update(part.pages)
            if not truncated:
                break
            done, total = truncated
            if total and done >= total:
                truncated = None
                break
            if done < first:
                # Проход не продвинулся ни на страницу (кончился бюджет
                # времени задачи) — следующий не продвинется тем более.
                break
            first = done + 1
        if not pages:
            logger.info("Распознавание не дало строк для %s — сшивать нечего", uri)
            return None, self._unread_pages(pages, total_pages, truncated)
        layer = TextLayer(pages=pages, page_count=max(pages))
        unread = self._unread_pages(pages, total_pages, truncated)
        logger.info(
            "Слой распознавания для %s: страниц %d, символов %d%s",
            uri, layer.page_count, layer.char_count(),
            f", не прочитано страниц {len(unread)}: {page_ranges(unread)}" if unread else "",
        )
        return layer, unread

    @staticmethod
    def _unread_pages(
        pages: Dict[int, Any], total_pages: int, truncated: Optional[tuple],
    ) -> List[int]:
        """
        Страницы, до которых распознавание не дошло.

        Пустая страница скана и страница, которую не успели прочитать, —
        разные вещи, и путать их нельзя. Непрочитанными считаются только
        страницы за местом обрыва прохода: страница внутри прочитанного
        участка без строк — это пустая страница, а не потеря.
        """
        if not truncated:
            return []
        done, total = truncated
        last = max(1, int(done))
        total = max(int(total or 0), int(total_pages or 0))
        if total <= last:
            return []
        return [page for page in range(last + 1, total + 1) if page not in pages]

    @staticmethod
    def _has_content(result: ParseResult) -> bool:
        """В результате есть хоть что-то: текст, таблица, формула, картинка."""
        return bool(result.blocks)

    def _parse_from_text_layer(self, layer: TextLayer) -> ParseResult:
        """
        Разбор одним текстовым слоем, без MinerU. Таблиц и формул тут не
        будет — зато текст точный и полный, что лучше молчаливой потери.
        """
        blocks: List[ParsedBlock] = []
        for page in sorted(layer.pages):
            if layer.overlay_only(page):
                continue
            lines = [l for l in layer.lines(page) if l.char_count()]
            turn = page_turn(lines)
            for group in group_lines(lines, turn):
                text = lines_to_text(group, turn)
                if not text:
                    continue
                blocks.append(ParsedBlock(
                    type="text", text=text, page=page,
                    bbox=merge_line_bboxes(group), order=len(blocks),
                    confidence=1.0, method="text_layer",
                ))
        return ParseResult(
            blocks=blocks,
            is_fallback=True,
            parser_name="text_layer",
            source_kind="vector_pdf",
            raw_output={},
            failed_pages=[],
            page_count=layer.page_count,
            text_layer_chars=layer.char_counts_by_page(),
            text_layer_stats={"mode": "text_layer_only"},
        )

    # -------------------------------------------------------------- MinerU
    def _parse_with_mineru(
        self,
        uri: str,
        file_type: str,
        metadata: Dict[str, Any],
        file_bytes: Optional[bytes] = None,
        pages: int = 1,
    ) -> ParseResult:
        if file_bytes is None:
            try:
                file_bytes = self.storage.read_bytes(uri)
            except Exception as exc:
                raise ParserFailed(
                    f"Не удалось прочитать файл из хранилища: {uri}: {exc}"
                ) from exc

        filename = self._filename(uri, file_type)
        url = f"{self.mineru_endpoint}/file_parse"
        data = {
            "return_md": "true",
            "return_content_list": "true",
            "return_middle_json": "true",
            "return_images": "true",
            "lang_list": metadata.get("lang") or settings.MINERU_DEFAULT_LANG,
            "backend": settings.MINERU_BACKEND,
        }
        files = [("files", (filename, file_bytes, filetypes.mime_for(file_type)))]

        timeout = self._timeout_for(pages)
        logger.info(
            "MinerU: %s file=%s size=%d B timeout=%ds", url, filename, len(file_bytes), timeout
        )
        try:
            response = requests.post(url, files=files, data=data, timeout=timeout)
        except requests.ReadTimeout as exc:
            # Долгий разбор большого скана — свойство документа, а не отказ
            # сервиса. Если по таймауту помечать MinerU недоступным на минуту, все
            # соседние документы уйдут в плоский OCR без таблиц.
            raise ParserTimedOut(f"MinerU не уложился в {timeout} с: {exc}") from exc
        except requests.RequestException as exc:
            self.mark_unavailable()
            raise ParserUnavailable(f"нет связи с MinerU: {exc}") from exc

        if response.status_code >= 500:
            raise ParserUnavailable(f"MinerU вернул {response.status_code}: {response.text[:300]}")
        if response.status_code != 200:
            raise ParserFailed(
                f"MinerU отклонил запрос {response.status_code}: {response.text[:500]}"
            )

        try:
            payload = response.json()
        except ValueError as exc:
            raise ParserFailed(f"MinerU вернул не-JSON: {response.text[:300]}") from exc

        asset_prefix = metadata.get("asset_prefix") or settings.ASSETS_PREFIX
        blocks, page_count = self._extract(payload, asset_prefix)

        is_image = filetypes.is_image(file_type)
        return ParseResult(
            blocks=blocks,
            is_fallback=False,
            parser_name="mineru",
            source_kind="image" if is_image else "vector_pdf",
            raw_output=payload,
            failed_pages=[],
            page_count=page_count or (1 if is_image else len({b.page for b in blocks})),
        )

    def _timeout_for(self, pages: int) -> int:
        """
        Таймаут разбора: не меньше общего и бюджета на страницу, но не больше
        40 % бюджета задачи — иначе задачу оборвёт Celery и документ
        потеряется целиком вместо деградации до OCR.
        """
        wanted = max(self.timeout, max(1, pages) * settings.MINERU_TIMEOUT_PER_PAGE)
        wanted = min(wanted, settings.PARSE_TASK_SOFT_TIME_LIMIT * 0.4)
        # Остаток срока задачи: половина — MinerU, половина — OCR-слою/фоллбэку.
        left = deadline.remaining()
        if left is not None:
            wanted = min(wanted, max(1.0, left / 2))
        return int(max(1, wanted))

    # ------------------------------------------------------ разбор ответа
    def _extract(self, payload: Dict[str, Any], asset_prefix: str) -> Tuple[List[ParsedBlock], int]:
        results = payload.get("results") or {}
        if not results:
            logger.warning("В ответе MinerU нет ключа 'results'")
            return [], 0

        blocks: List[ParsedBlock] = []
        page_count = 0

        for filename, result in results.items():
            images = self._store_images(result.get("images") or {}, asset_prefix, filename)

            middle = self._loads(result.get("middle_json"))
            if middle:
                got, pages = self._blocks_from_middle_json(middle, images)
                if got:
                    blocks.extend(got)
                    page_count = max(page_count, pages)
                    continue

            content_list = self._loads(result.get("content_list"))
            if content_list:
                got = self._blocks_from_content_list(content_list, images)
                if got:
                    blocks.extend(got)
                    page_count = max(page_count, max((b.page for b in got), default=0))
                    continue

            md_content = result.get("md_content")
            if md_content:
                blocks.append(
                    ParsedBlock(type="text", text=md_content, page=1, bbox=[0, 0, 1, 1])
                )
                page_count = max(page_count, 1)

        for order, block in enumerate(blocks):
            if block.order is None:
                block.order = order
        return blocks, page_count

    @staticmethod
    def _loads(value: Any) -> Any:
        if not value:
            return None
        if isinstance(value, (list, dict)):
            return value
        try:
            return json.loads(value)
        except (TypeError, ValueError):
            logger.error("Не удалось разобрать JSON из ответа MinerU")
            return None

    def _store_images(
        self, images: Dict[str, Any], asset_prefix: str, source_name: str
    ) -> Dict[str, str]:
        """Base64-картинки из ответа MinerU -> URI в хранилище."""
        stored: Dict[str, str] = {}
        if not images:
            return stored
        prefix = asset_prefix if asset_prefix.endswith("/") else asset_prefix + "/"
        for name, payload in images.items():
            try:
                raw = payload
                if isinstance(raw, dict):
                    raw = raw.get("data") or raw.get("base64") or ""
                if not isinstance(raw, str) or not raw:
                    continue
                if raw.startswith("data:"):
                    raw = raw.split(",", 1)[1]
                content = base64.b64decode(raw)
                safe = re.sub(r"[^A-Za-z0-9._-]", "_", os.path.basename(name)) or "image.jpg"
                target = f"{prefix}{safe}"
                self.storage.write_file(target, content)
                stored[name] = target
                stored[os.path.basename(name)] = target
            except Exception as exc:  # noqa: BLE001
                logger.warning("Не удалось сохранить изображение %s из %s: %s", name, source_name, exc)
        if stored:
            logger.info("Сохранено изображений из MinerU: %d", len({v for v in stored.values()}))
        return stored

    # ------------------------------------------------------- middle_json
    def _blocks_from_middle_json(
        self, middle_json: Dict[str, Any], images: Dict[str, str]
    ) -> Tuple[List[ParsedBlock], int]:
        pdf_info = middle_json.get("pdf_info") or []
        blocks: List[ParsedBlock] = []
        order = 0

        for page in pdf_info:
            page_no = int(page.get("page_idx", 0)) + 1
            page_size = page.get("page_size") or []
            width = float(page_size[0]) if len(page_size) >= 2 and page_size[0] else 0.0
            height = float(page_size[1]) if len(page_size) >= 2 and page_size[1] else 0.0

            raw_blocks = list(
                page.get("para_blocks")
                or page.get("preproc_blocks")
                or page.get("layout_dets")
                or []
            )
            # `discarded_blocks` — то, что MinerU решил не показывать
            # (колонтитулы, боковые подписи, куски вне основной колонки).
            # Для полноты документа это всё равно содержимое, и терять его
            # молча нельзя.
            raw_blocks.extend(page.get("discarded_blocks") or [])

            for raw in raw_blocks:
                block = self._block_from_raw(raw, page_no, width, height, images, order)
                if block is not None:
                    blocks.append(block)
                    order += 1

        return blocks, len(pdf_info)

    def _block_from_raw(
        self,
        raw: Dict[str, Any],
        page_no: int,
        width: float,
        height: float,
        images: Dict[str, str],
        order: int,
    ) -> Optional[ParsedBlock]:
        raw_type = str(raw.get("type", "text")).lower()
        bbox = self._norm_bbox(raw.get("bbox") or raw.get("poly"), width, height)
        confidence = float(raw.get("score", 1.0) or 1.0)

        if raw_type in _TEXTISH:
            text = self._text_from_block(raw)
            if not text:
                return None
            return ParsedBlock(
                type="text", text=text, page=page_no, bbox=bbox, order=order,
                confidence=confidence, page_size=[width, height] if width else None,
                section_title=text[:200] if raw.get("type") == "title" else None,
                method=METHOD_PARSER_OCR,
            )

        if raw_type in _FORMULA:
            text = self._text_from_block(raw) or raw.get("latex") or raw.get("html")
            if not text:
                return None
            return ParsedBlock(
                type="formula", text=str(text), page=page_no, bbox=bbox, order=order,
                confidence=confidence, page_size=[width, height] if width else None,
                method="mineru_formula",
            )

        if raw_type == "table":
            html = self._nested_value(raw, ("html", "table_body", "latex"))
            table_data = parse_table_html(html) if html else None
            return ParsedBlock(
                type="table", table_data=table_data, table_html=html,
                text=self._text_from_block(raw) or None,
                image_ref=self._image_uri(raw, images), page=page_no, bbox=bbox,
                order=order, confidence=confidence,
                page_size=[width, height] if width else None,
                method="mineru_table",
            )

        if raw_type in ("image", "figure", "image_body"):
            image_ref = self._image_uri(raw, images)
            if not image_ref:
                return None
            return ParsedBlock(
                type="image", image_ref=image_ref, page=page_no, bbox=bbox, order=order,
                confidence=confidence, page_size=[width, height] if width else None,
                method="mineru_image",
            )

        return None

    @staticmethod
    def _text_from_block(raw: Dict[str, Any]) -> str:
        """Собирает текст из lines[].spans[].content или прямого поля text."""
        direct = raw.get("text")
        if isinstance(direct, str) and direct.strip():
            return direct.strip()

        parts: List[str] = []
        for container in ("lines", "blocks"):
            for line in raw.get(container) or []:
                if isinstance(line, dict):
                    for span in line.get("spans") or []:
                        content = span.get("content") or span.get("text") or span.get("html")
                        if content:
                            parts.append(str(content))
                    nested = MinerUParserProvider._text_from_block(line)
                    if nested and not (line.get("spans")):
                        parts.append(nested)
        return " ".join(" ".join(parts).split())

    @staticmethod
    def _nested_value(raw: Dict[str, Any], keys: Tuple[str, ...]) -> Optional[str]:
        for key in keys:
            value = raw.get(key)
            if isinstance(value, str) and value.strip():
                return value
        for line in raw.get("blocks") or raw.get("lines") or []:
            if isinstance(line, dict):
                found = MinerUParserProvider._nested_value(line, keys)
                if found:
                    return found
                for span in line.get("spans") or []:
                    for key in keys:
                        value = span.get(key)
                        if isinstance(value, str) and value.strip():
                            return value
        return None

    @staticmethod
    def _image_uri(raw: Dict[str, Any], images: Dict[str, str]) -> Optional[str]:
        path = MinerUParserProvider._nested_value(raw, ("img_path", "image_path", "image_ref"))
        if not path:
            return None
        return images.get(path) or images.get(os.path.basename(path)) or path

    @staticmethod
    def _norm_bbox(bbox: Any, width: float, height: float) -> List[float]:
        """Координаты MinerU (пиксели страницы) -> доли [0, 1]."""
        if not bbox:
            return [0.0, 0.0, 1.0, 1.0]
        coords = [float(c) for c in bbox if isinstance(c, (int, float))]
        if len(coords) == 8:  # полигон -> объемлющий прямоугольник
            xs, ys = coords[0::2], coords[1::2]
            coords = [min(xs), min(ys), max(xs), max(ys)]
        if len(coords) != 4:
            return [0.0, 0.0, 1.0, 1.0]
        if width > 0 and height > 0:
            coords = [coords[0] / width, coords[1] / height, coords[2] / width, coords[3] / height]
        elif max(coords) > 1.0:
            # Размер страницы неизвестен, а координаты явно не нормализованы.
            return [0.0, 0.0, 1.0, 1.0]
        return coords

    # ----------------------------------------------------- content_list
    def _blocks_from_content_list(
        self, content_list: List[Dict[str, Any]], images: Dict[str, str]
    ) -> List[ParsedBlock]:
        """
        Резервный путь: координат в content_list нет, поэтому bbox остаётся
        на всю страницу, а полнота по площади в нормализаторе не считается.
        """
        blocks: List[ParsedBlock] = []
        for order, item in enumerate(content_list):
            raw_type = str(item.get("type", "text")).lower()
            page_no = int(item.get("page_idx", 0)) + 1

            if raw_type in _TEXTISH:
                text = item.get("text") or item.get("content")
                if not text or not str(text).strip():
                    continue
                blocks.append(ParsedBlock(
                    type="text", text=str(text).strip(), page=page_no, order=order,
                    bbox=[0, 0, 1, 1],
                    section_title=str(text)[:200] if item.get("text_level") == 1 else None,
                    method=METHOD_PARSER_OCR,
                ))
            elif raw_type in _FORMULA:
                text = item.get("text") or item.get("latex")
                if not text:
                    continue
                blocks.append(ParsedBlock(
                    type="formula", text=str(text), page=page_no, order=order, bbox=[0, 0, 1, 1]
                ))
            elif raw_type == "table":
                html = item.get("table_body") or item.get("html")
                blocks.append(ParsedBlock(
                    type="table", table_data=parse_table_html(html) if html else None,
                    table_html=html, image_ref=self._image_uri(item, images),
                    page=page_no, order=order, bbox=[0, 0, 1, 1],
                ))
            elif raw_type in ("image", "figure"):
                image_ref = self._image_uri(item, images)
                if image_ref:
                    blocks.append(ParsedBlock(
                        type="image", image_ref=image_ref, page=page_no, order=order,
                        bbox=[0, 0, 1, 1],
                    ))
        return blocks

    # ------------------------------------------------------------ фоллбэк
    def _full_tesseract_fallback(self, uri: str, file_type: str) -> ParseResult:
        blocks = self.tesseract.parse_all_pages(uri, file_type)
        for block in blocks:
            block.method = block.method or "tesseract"
        degraded: List[str] = []
        truncated = getattr(self.tesseract, "truncated_at", None)
        if truncated:
            done, total = truncated
            degraded.append(
                f"OCR выполнен для первых {done} страниц из {total or 'неизвестного числа'}: "
                f"предел OCR_FULL_MAX_PAGES или OCR_TIME_BUDGET."
            )
        return ParseResult(
            blocks=blocks,
            is_fallback=True,
            degraded=degraded,
            parser_name="tesseract_full",
            source_kind="image" if filetypes.is_image(file_type) else "scanned_pdf",
            raw_output={},
            # failed_pages — страницы, которые распознать НЕ удалось. Раньше
            # сюда клались как раз успешные, и сырой результат врал наоборот.
            failed_pages=list(getattr(self.tesseract, "failed_pages", []) or []),
            page_count=len({b.page for b in blocks}),
        )

    # ------------------------------------------------------------- чертежи
    @staticmethod
    def _fix_dimensions(
        blocks: List[ParsedBlock], verdicts: Optional[Dict[int, DrawingPage]] = None,
    ) -> int:
        """
        Числа и знаки размеров в тексте и ячейках MinerU
        (`DRAWING_DIMENSION_FIDELITY`); сколько записей поправлено. Своя
        таблица MinerU лежит в выдаче рядом с чтением модели: пока в ячейке
        оставался «M24×15», верная запись «M24x1,5» из чтения модели
        находила лишь одну из двух.

        Правятся только страницы-чертежи, если вердикт по страницам есть:
        в прозе договора или ТЗ размеров по ГОСТ нет, и трогать её незачем.
        Вердикта нет только у входа-картинки — там лист и есть весь файл.
        """
        if not settings.DRAWING_DIMENSION_FIDELITY:
            return 0
        chosen = blocks
        if verdicts:
            drawings = {page for page, verdict in verdicts.items() if verdict.is_drawing}
            if not drawings:
                return 0
            chosen = [block for block in blocks if block.page in drawings]
        return dimension_fidelity.repair_blocks(chosen)

    def _apply_vector_drawings(
        self,
        blocks: List[ParsedBlock],
        layer: TextLayer,
        verdicts: Dict[int, DrawingPage],
    ) -> List[ParsedBlock]:
        """
        Страницы, которые по собственной геометрии являются чертежами,
        заменяются одним блоком `drawing` с разобранными полями. Текстовые
        блоки такой страницы убираются: их содержимое уезжает в поля, и
        дублировать его отдельными фрагментами незачем.
        """
        drawing_pages = {p for p, v in verdicts.items() if v.is_drawing}
        if not drawing_pages:
            return blocks

        result: List[ParsedBlock] = [
            b for b in blocks if not (b.page in drawing_pages and b.type == "text")
        ]

        for page in sorted(drawing_pages):
            fields = extract_drawing_fields(layer, page)
            if not fields:
                # Скан-чертёж без текстового слоя: оставляем страницу как была,
                # ею занимается детектор с распознаванием.
                continue
            result.append(ParsedBlock(
                type="drawing",
                text=" ".join(f["value"] for f in fields),
                page=page,
                bbox=[0.0, 0.0, 1.0, 1.0],
                confidence=verdicts[page].score,
                method="vector_text_layer",
                drawing_fields=fields,
            ))

        result.sort(key=lambda b: (b.page, b.bbox[1], b.bbox[0]))
        for order, block in enumerate(result):
            block.order = order
        return result

    @staticmethod
    def _has_text(result: ParseResult) -> bool:
        return any(b.type == "text" and b.text and b.text.strip() for b in result.blocks)

    @staticmethod
    def _filename(uri: str, file_type: str) -> str:
        name = uri.rstrip("/").split("/")[-1] or "document"
        if "." not in name:
            name = f"{name}.{file_type.lower()}"
        stem, ext = name.rsplit(".", 1)
        return f"{stem}.{ext.lower()}"
