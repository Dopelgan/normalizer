"""
Уровни 4-7 — всё, что требует распознавания.

4. Распознавание текста без структуры. Дёшево, годится для чистого растра
   с простой вёрсткой.
5. Детекция областей и структурированный разбор. Сюда же сведён основной
   конвейер MinerU вместе со сшивкой текстового слоя и разбором чертежей.
6. Восстановление изображения плюс уровень 5. Отдельный этап: от десятой
   до трети корпуса предприятия с историей — фотографии и копии низкого
   качества, и без выправления уровень 5 на них даёт мало.
7. Эскалация к основной мультимодальной модели. Применяется, когда после
   уровня 5 или 6 остаётся неоднозначность, и только если модель настроена.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any, Dict, Optional

import requests

from core import filetypes
from core.config import settings
from core.ladder.base import Strategy
from core.ladder.context import SOURCE_RASTER, DocumentContext
from core.models.parse_result import ParsedBlock, ParseResult
from core.providers.mineru_parser import MinerUParserProvider
from core.providers.storage import StorageProvider
from core.providers.tesseract_fallback import TesseractFallbackProvider

logger = logging.getLogger(__name__)


class PlainOcrStrategy(Strategy):
    """Уровень 4: распознавание без разметки областей."""

    level = 4
    name = "plain_ocr"
    method = "tesseract"

    def __init__(
        self,
        provider: Optional[TesseractFallbackProvider] = None,
        storage: Optional[StorageProvider] = None,
    ):
        self.provider = provider or TesseractFallbackProvider(storage=storage)

    def applicable(self, context: DocumentContext) -> bool:
        return context.kind in (filetypes.KIND_IMAGE, filetypes.KIND_PDF)

    def run(self, context: DocumentContext) -> ParseResult:
        blocks = self.provider.parse_all_pages(context.uri)
        if not blocks:
            raise ValueError("Распознавание не дало текста")
        return self.build_result(
            blocks,
            source_kind="image" if context.kind == filetypes.KIND_IMAGE else "scanned_pdf",
            is_fallback=True,
        )


class LayoutStrategy(Strategy):
    """
    Уровень 5: детекция областей и структурированный разбор.

    Здесь же работает сшивка с текстовым слоем и разбор векторных чертежей —
    всё это части одного вызова парсера и разделять их нет смысла.
    """

    level = 5
    name = "layout"
    method = "mineru_ocr"

    def __init__(
        self,
        provider: Optional[MinerUParserProvider] = None,
        storage: Optional[StorageProvider] = None,
    ):
        self._provider = provider
        self._storage = storage

    @property
    def provider(self) -> MinerUParserProvider:
        if self._provider is None:
            self._provider = MinerUParserProvider(storage=self._storage)
        return self._provider

    def applicable(self, context: DocumentContext) -> bool:
        if context.kind not in (filetypes.KIND_PDF, filetypes.KIND_IMAGE):
            return False
        # Сервис разбора лежит — этот уровень выродится в тот же полный OCR,
        # который уже сделал уровень 4, только потратив на него вдвое больше
        # времени. Лучше честно пропустить уровень: деградация уже отмечена.
        if not MinerUParserProvider.is_available():
            logger.warning(
                "Уровень %d пропущен: MinerU недоступен, разбор остаётся на OCR",
                self.level,
            )
            return False
        return True

    def run(self, context: DocumentContext) -> ParseResult:
        result = self.provider.parse(context.uri, context.file_type, context.metadata)
        if not result.blocks:
            raise ValueError("Разбор с детекцией областей не дал блоков")
        result.parser_name = result.parser_name or self.name
        return result


class RestoreThenLayoutStrategy(Strategy):
    """
    Уровень 6: геометрическое и фотометрическое выправление, затем уровень 5.

    Порядок этапов из архитектуры: оценка, выправление геометрии,
    выправление освещения и контраста, избирательное повышение разрешения,
    повторная оценка. Если после выправления лучше не стало, документ
    честно помечается нечитаемым в оригинале — это не то же самое, что
    «не распознано системой».
    """

    level = 6
    name = "restore_then_layout"
    method = "restored_layout"

    def __init__(
        self,
        layout: Optional[LayoutStrategy] = None,
        storage: Optional[StorageProvider] = None,
    ):
        self.layout = layout or LayoutStrategy(storage=storage)

    def applicable(self, context: DocumentContext) -> bool:
        classification = context.classification
        return (
            classification.source == SOURCE_RASTER
            and classification.needs_restoration
            and context.kind in (filetypes.KIND_IMAGE, filetypes.KIND_PDF)
            # Уровень заканчивается вызовом уровня 5: без сервиса он
            # бессмыслен ровно по той же причине.
            and MinerUParserProvider.is_available()
        )

    def run(self, context: DocumentContext) -> ParseResult:
        from core.ladder.restoration import restore

        restored, applied = restore(context.data, pdf=context.kind == filetypes.KIND_PDF)
        if restored is None:
            raise ValueError("Восстановление изображения недоступно")

        quality = self._quality_after(restored, context.classification.signals)
        before = context.classification.raster_quality or 0.0
        logger.info(
            "Восстановление %s: качество %.2f -> %.2f, применено: %s",
            context.uri, before, quality or 0.0, ", ".join(applied) or "ничего",
        )
        if quality is not None and quality <= before + 0.02:
            raise ValueError(
                "После выправления изображение не стало пригоднее — "
                "документ нечитаем в оригинале"
            )

        uri = self._store(context, restored)
        restored_context = DocumentContext(
            uri=uri, file_type="png", metadata=context.metadata,
            storage=context.storage, data=restored,
        )
        result = self.layout.run(restored_context)
        for block in result.blocks:
            block.method = self.method
        result.parser_name = self.name
        return result

    @staticmethod
    def _quality_after(restored: bytes, signals: Dict[str, Any]) -> Optional[float]:
        """
        Оценка выправленного растра в той же шкале, что и оценка исходного.

        В оценку входит разрешение, а этап В-4 увеличивает мелкий лист вдвое:
        вчетверо больше пикселей — и «стало лучше» получалось у любого
        увеличения, даже когда читаемость не менялась ни на сколько. Тем же
        путём сравнивались рендер PDF в 150 dpi и увеличенный вдвое PNG.
        Поэтому перед сравнением выправленный растр приводится к числу
        пикселей исходного — с сохранением пропорций, не растяжением.
        """
        from core.ladder.context import _load_image, assess_raster

        width, height = signals.get("width"), signals.get("height")
        image = _load_image(restored)
        if image is None or not width or not height:
            quality, _ = assess_raster(restored, image=image)
            return quality

        before_pixels = int(width) * int(height)
        after_pixels = image.width * image.height
        if after_pixels > before_pixels * 1.02:
            factor = (before_pixels / after_pixels) ** 0.5
            try:
                from PIL import Image

                image = image.resize(
                    (max(1, int(image.width * factor)), max(1, int(image.height * factor))),
                    Image.LANCZOS,
                )
            except ImportError:  # pragma: no cover — без Pillow сюда не дойдут
                pass

        quality, _ = assess_raster(restored, image=image)
        return quality

    @staticmethod
    def _store(context: DocumentContext, data: bytes) -> str:
        """
        Выправленный растр кладётся рядом с ассетами: парсер читает файл из
        хранилища по URI, а не принимает байты.
        """
        digest = hashlib.sha256(data).hexdigest()[:16]
        prefix = settings.ASSETS_PREFIX.rstrip("/")
        uri = f"{prefix}/restored/{digest}.png"
        context.storage.write_file(uri, data)
        return uri


class VlmEscalationStrategy(Strategy):
    """
    Уровень 7: эскалация к основной мультимодальной модели.

    Самый дорогой уровень, поэтому применяется только при настроенном
    эндпоинте и только к документам, где дешёвые уровни оставили
    неоднозначность. Модель не настроена — уровень просто неприменим, и
    лестница молча заканчивается на предыдущем.
    """

    level = 7
    name = "vlm_escalation"
    method = "vlm_escalation"

    def applicable(self, context: DocumentContext) -> bool:
        return bool(settings.VLM_ENDPOINT) and context.kind in (
            filetypes.KIND_PDF, filetypes.KIND_IMAGE
        )

    def run(self, context: DocumentContext) -> ParseResult:
        import base64

        payload = {
            "image_b64": base64.b64encode(context.data).decode("ascii"),
            "file_type": context.file_type,
            "prompt": settings.VLM_PROMPT,
        }
        try:
            response = requests.post(
                f"{settings.VLM_ENDPOINT.rstrip('/')}/parse",
                json=payload, timeout=settings.VLM_TIMEOUT,
            )
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as exc:
            raise ValueError(f"Мультимодальная модель недоступна: {exc}") from exc
        except ValueError as exc:
            raise ValueError(f"Мультимодальная модель вернула не-JSON: {exc}") from exc

        text = (data.get("text") or "").strip()
        if not text:
            raise ValueError("Мультимодальная модель не вернула текста")

        confidence = float(data.get("confidence", 0.7) or 0.7)
        blocks = [ParsedBlock(
            type="text", text=text, page=int(data.get("page", 1) or 1),
            confidence=max(0.0, min(1.0, confidence)), method=self.method,
        )]
        return self.build_result(blocks, source_kind="vlm", page_count=1)
