"""
Парсер документов поверх MinerU HTTP API с фоллбэком на Tesseract.

Отличия от черновика плана 45 (там предполагалась ручка `/parse`, которой у
MinerU нет):

* используется настоящая ручка `POST /file_parse` с multipart-загрузкой;
* приоритет отдаётся `middle_json` — только он содержит координаты блоков
  и `page_size`, без которых bbox нельзя нормализовать в [0, 1];
* изображения запрашиваются у MinerU (`return_images`) и выкладываются в
  хранилище, иначе этап обработки чертежей остаётся без входных данных;
* таблицы разбираются из `table_body` (HTML), как их реально отдаёт MinerU;
* сетевые сбои (фоллбэк уместен) отделены от ошибок разбора (это дефект).
"""

import base64
import json
import logging
import os
import re
from html.parser import HTMLParser
from typing import Any, Dict, List, Optional, Tuple

import requests

from core import filetypes
from core.config import settings
from core.models.parse_result import ParsedBlock, ParseResult
from core.providers.document_parser import (
    DocumentParserProvider,
    ParserFailed,
    ParserUnavailable,
)
from core.providers.storage import StorageProvider, StorageProviderFactory
from core.providers.tesseract_fallback import TesseractFallbackProvider
from core.providers.vector_drawing import (
    DrawingPage,
    drawing_pages_from_document,
    extract_drawing_fields,
)
from core.providers.text_layer import (
    METHOD_PARSER_OCR,
    OCR_LAYER,
    TextLayer,
    group_lines,
    layer_from_document,
    lines_to_text,
    merge_line_bboxes,
    open_pdf,
    reconcile_with_text_layer,
)

logger = logging.getLogger(__name__)

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

class MinerUParserProvider(DocumentParserProvider):
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
        layer, drawing_verdicts = self._read_pdf(file_bytes, file_type)

        try:
            result = self._parse_with_mineru(uri, file_type, metadata, file_bytes)
        except ParserUnavailable as exc:
            # Недоступность сервиса — это деградация системы, а не свойство
            # документа, и путать их нельзя: лежащий mineru-api иначе
            # неотличим от плохого скана, приём «работает», а качество падает
            # молча и на всём потоке сразу.
            note = f"mineru_unavailable: {exc}"
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
            blocks, stats = reconcile_with_text_layer(result.blocks, layer)
            blocks = self._apply_vector_drawings(blocks, layer, drawing_verdicts)
            result.blocks = blocks
            result.text_layer_chars = stats.layer_chars_by_page
            result.text_layer_stats = stats.as_dict()
            return result

        # Растр: слоя в файле нет, но его можно собрать распознаванием. Схема
        # та же, что у вектора, — структура от MinerU, текст со стороны:
        # модель таблиц MinerU отдаёт русскую прозу побуквенной LaTeX-
        # разметкой, а Tesseract с `rus` читает её как текст.
        if self._has_content(result):
            ocr_layer = self._ocr_layer(uri)
            if ocr_layer is not None:
                blocks, stats = reconcile_with_text_layer(
                    result.blocks, ocr_layer, source=OCR_LAYER
                )
                result.blocks = blocks
                result.text_layer_chars = stats.layer_chars_by_page
                result.text_layer_stats = {**stats.as_dict(), "layer": "ocr"}
                if stats.degraded_tables:
                    result.degraded.append(
                        f"table_not_parsed: таблиц разобрано на текст и формулы "
                        f"{stats.degraded_tables}"
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
    ) -> Tuple[Optional[TextLayer], Dict[int, DrawingPage]]:
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
            return None, {}

        document = open_pdf(file_bytes)
        if document is None:
            return None, {}
        try:
            layer = layer_from_document(document)
            if not layer.is_usable():
                logger.info("Текстовый слой пуст или разрежен — работаем по распознаванию")
                return None, {}
            return layer, drawing_pages_from_document(document)
        finally:
            document.close()

    def _ocr_layer(self, uri: str) -> Optional[TextLayer]:
        """
        Построчное распознавание растра как слой. Недоступность OCR здесь не
        фатальна: без слоя разбор останется как был, поэтому ошибка уходит в
        лог, а не наружу.
        """
        if not settings.OCR_LAYER_ENABLED:
            return None
        try:
            layer = self.tesseract.text_layer(uri)
        except Exception as exc:  # noqa: BLE001 — слой не обязателен
            logger.warning("Слой распознавания для %s не собрался: %s", uri, exc)
            return None
        if layer is None or not layer.char_count():
            logger.info("Распознавание не дало строк для %s — сшивать нечего", uri)
            return None
        logger.info(
            "Слой распознавания для %s: страниц %d, символов %d",
            uri, layer.page_count, layer.char_count(),
        )
        return layer

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
            for group in group_lines([l for l in layer.lines(page) if l.char_count()]):
                text = lines_to_text(group)
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

        logger.info("MinerU: %s file=%s size=%d B", url, filename, len(file_bytes))
        try:
            response = requests.post(url, files=files, data=data, timeout=self.timeout)
        except requests.RequestException as exc:
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
        blocks = self.tesseract.parse_all_pages(uri)
        for block in blocks:
            block.method = block.method or "tesseract"
        return ParseResult(
            blocks=blocks,
            is_fallback=True,
            parser_name="tesseract_full",
            source_kind="image" if filetypes.is_image(file_type) else "scanned_pdf",
            raw_output={},
            # failed_pages — страницы, которые распознать НЕ удалось. Раньше
            # сюда клались как раз успешные, и сырой результат врал наоборот.
            failed_pages=[],
            page_count=len({b.page for b in blocks}),
        )

    # ------------------------------------------------------------- чертежи
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
