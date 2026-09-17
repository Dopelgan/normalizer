"""
DOCUMENT QUALITY GATE — проверка технической пригодности.

Отвечает на вопрос «пригоден ли документ технически», и только на него.
Вопрос «относится ли это к корпоративным знаниям» решён раньше, на Data
Gateway: сюда файл приходит уже признанным знанием.

Быстрый OCR здесь нужен лишь для предварительной оценки читаемости —
полный разбор выполняется позже и только после прохождения проверки. Если
у документа есть текстовый слой, распознавать вообще нечего: читаемость
доказана самим файлом.

Пять этапов: техническая валидация, дубликаты, дата и актуальность,
качество распознавания, классификация и маршрутизация. Предложение
размещения формирует система — утверждает человек.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from core import filetypes
from core.config import settings
from core.gateway.profile import GatewayProfile, load_profile
from core.gateway.service import ACCEPT, QUARANTINE, REJECT
from core.ladder.context import (
    GENRE_DRAWING,
    GENRE_MIXED,
    GENRE_SPREADSHEET,
    GENRE_TEXT,
    SOURCE_RASTER,
    DocumentContext,
)
from core.providers.storage import StorageProvider, StorageProviderFactory

logger = logging.getLogger(__name__)

# Типы маршрутизации из QG-5.
ROUTE_TEXT = "текстовый"
ROUTE_TABULAR = "табличный"
ROUTE_MIXED = "смешанный"
ROUTE_MULTIMODAL = "мультимодальный"
ROUTE_DRAWING = "чертёж"

# Расстояние Хэмминга между отпечатками, ниже которого документы считаются
# почти одинаковыми. 64-битный отпечаток, 6 бит — это около 90% сходства.
_SIMHASH_BITS = 64
_NEAR_DUPLICATE_DISTANCE = 6

_DATE_IN_NAME = re.compile(r"(19|20)\d{2}[-_.]?(0[1-9]|1[0-2])?")


@dataclass
class QualityVerdict:
    """Решение Quality Gate вместе с предупреждениями и предложением места."""

    outcome: str
    reason: str
    stage: str
    warnings: List[str] = field(default_factory=list)
    confidence: float = 1.0
    file_hash: Optional[str] = None
    routing: Dict[str, Any] = field(default_factory=dict)
    signals: Dict[str, Any] = field(default_factory=dict)

    @property
    def accepted(self) -> bool:
        return self.outcome == ACCEPT

    def as_dict(self) -> Dict[str, Any]:
        return {
            "outcome": self.outcome,
            "reason": self.reason,
            "stage": self.stage,
            "warnings": self.warnings,
            "confidence": round(self.confidence, 3),
            "file_hash": self.file_hash,
            "routing": self.routing,
            "signals": self.signals,
        }


class QualityGate:
    """Пять этапов проверки перед тем, как пускать файл в нормализацию."""

    def __init__(
        self,
        profile: Optional[GatewayProfile] = None,
        storage: Optional[StorageProvider] = None,
        known_hashes: Optional[Dict[str, str]] = None,
        known_fingerprints: Optional[List[Tuple[str, int]]] = None,
    ):
        self.profile = profile or load_profile()
        self.storage = storage or StorageProviderFactory.default()
        # Отпечатки и хеши уже принятого приходят снаружи: ходить в БД из
        # проверки не нужно, а тестировать так заметно проще.
        self.known_hashes = dict(known_hashes or {})
        self.known_fingerprints = list(known_fingerprints or [])

    # ------------------------------------------------------------- публичное
    def evaluate(self, s3_fileid: str, uri: Optional[str] = None) -> QualityVerdict:
        path = uri or s3_fileid
        warnings: List[str] = []

        # Размер проверяется до чтения: файл на гигабайты нельзя сперва
        # затащить целиком в память, а потом решить, что он слишком большой.
        declared = self._declared_size(path)
        if declared is not None and declared > settings.QG_MAX_SIZE_BYTES:
            return QualityVerdict(
                QUARANTINE,
                f"Файл больше предела ({declared} Б против "
                f"{settings.QG_MAX_SIZE_BYTES} Б) — нужно решение человека.",
                "QG-1",
            )

        try:
            data = self.storage.read_bytes(path)
        except Exception as exc:  # noqa: BLE001
            return QualityVerdict(
                REJECT, f"Файл не читается из хранилища: {exc}", "QG-1"
            )

        verdict = self.technical_validation(path, data)
        if verdict is not None:
            return verdict

        file_hash = hashlib.sha256(data).hexdigest()

        context = DocumentContext(
            uri=path, file_type=filetypes.extension_of(path),
            storage=self.storage, data=data,
        )

        duplicate = self.duplicate_check(s3_fileid, file_hash, context, warnings)
        if duplicate is not None:
            return duplicate

        self.freshness_check(path, data, warnings)

        readability = self.ocr_check(context, warnings)
        if readability.outcome != ACCEPT:
            readability.file_hash = file_hash
            readability.warnings = warnings + readability.warnings
            return readability

        routing = self.routing(context)
        return QualityVerdict(
            outcome=ACCEPT,
            reason="Документ технически пригоден к нормализации.",
            stage="QG-5",
            warnings=warnings,
            confidence=readability.confidence,
            file_hash=file_hash,
            routing=routing,
            signals={"fingerprint": _simhash(_text_sample(context))},
        )

    def _declared_size(self, path: str) -> Optional[int]:
        """Размер по метаданным хранилища. None — хранилище его не отдало."""
        getter = getattr(self.storage, "size", None)
        if getter is None:
            return None
        try:
            return getter(path)
        except Exception as exc:  # noqa: BLE001 — неизвестный размер не отказ
            logger.warning("Размер %s не определён: %s", path, exc)
            return None

    # -------------------------------------------- QG-1 техническая валидация
    def technical_validation(self, path: str, data: bytes) -> Optional[QualityVerdict]:
        extension = filetypes.extension_of(path)

        reason = filetypes.rejection_reason(extension)
        if reason:
            return QualityVerdict(REJECT, reason, "QG-1")

        if len(data) > settings.QG_MAX_SIZE_BYTES:
            return QualityVerdict(
                QUARANTINE,
                f"Файл больше предела ({len(data)} Б против "
                f"{settings.QG_MAX_SIZE_BYTES} Б) — нужно решение человека.",
                "QG-1",
            )

        if len(data) < self.profile.min_size_bytes:
            return QualityVerdict(
                REJECT,
                f"Файл слишком мал ({len(data)} Б) — содержимого в нём нет.",
                "QG-1",
            )

        opened, detail = _can_open(data, extension)
        if not opened:
            return QualityVerdict(
                REJECT, f"Файл повреждён и не открывается: {detail}", "QG-1"
            )
        return None

    # -------------------------------------------- QG-2 проверка на дубликаты
    def duplicate_check(
        self,
        s3_fileid: str,
        file_hash: str,
        context: DocumentContext,
        warnings: List[str],
    ) -> Optional[QualityVerdict]:
        """
        Точный дубль отклоняется. Семантическое сходство выше порога — это
        не отказ, а флаг администратору: «обновление или дубль».
        """
        twin = self.known_hashes.get(file_hash)
        if twin and twin != s3_fileid:
            return QualityVerdict(
                REJECT,
                f"Точный дубликат уже принятого файла {twin!r} "
                f"(совпадает контрольная сумма).",
                "QG-2", file_hash=file_hash,
            )

        fingerprint = _simhash(_text_sample(context))
        for other_id, other_print in self.known_fingerprints:
            if other_id == s3_fileid:
                continue
            distance = _hamming(fingerprint, other_print)
            if distance <= _NEAR_DUPLICATE_DISTANCE:
                similarity = 1 - distance / _SIMHASH_BITS
                if similarity >= settings.QG_SIMILARITY_THRESHOLD:
                    warnings.append(
                        f"Похож на {other_id!r} (сходство {similarity:.0%}) — "
                        f"обновление или дубль, нужно решение администратора."
                    )
                    break
        return None

    # ------------------------------------------- QG-3 дата и актуальность
    def freshness_check(self, path: str, data: bytes, warnings: List[str]) -> None:
        created = _document_date(path, data)
        if created is None:
            warnings.append(
                "У документа не удалось определить дату — актуальность "
                "не подтверждена, уверенность снижена."
            )
            return

        age_days = (datetime.now(timezone.utc) - created).days
        if age_days > settings.QG_MAX_AGE_DAYS:
            warnings.append(
                f"Документу {age_days // 365} лет — он может быть неактуален."
            )

    # ------------------------------- QG-4 проверка качества распознавания
    def ocr_check(self, context: DocumentContext, warnings: List[str]) -> QualityVerdict:
        """
        Быстрый OCR на первых страницах. У векторного документа читаемость
        доказана текстовым слоем — распознавать нечего.
        """
        if context.kind in (
            filetypes.KIND_PLAIN_TEXT, filetypes.KIND_SPREADSHEET,
            filetypes.KIND_OFFICE_TEXT, filetypes.KIND_CAD,
        ):
            return QualityVerdict(ACCEPT, "Содержимое читается напрямую.", "QG-4")

        if context.text_layer is not None:
            return QualityVerdict(
                ACCEPT, "У документа есть текстовый слой — читаемость доказана.",
                "QG-4", confidence=1.0,
            )

        confidence = self._quick_ocr_confidence(context)
        if confidence is None:
            warnings.append(
                "Быстрый OCR недоступен — читаемость не проверена, "
                "оценка отложена до полного разбора."
            )
            return QualityVerdict(ACCEPT, "Читаемость не проверена.", "QG-4",
                                  confidence=0.5)

        if confidence < settings.QG_OCR_BLOCK_CONFIDENCE:
            return QualityVerdict(
                QUARANTINE,
                f"Распознавание почти ничего не даёт (уверенность "
                f"{confidence:.2f}). Документ нечитаем в оригинале — "
                f"нужно решение человека, а не повторная попытка.",
                "QG-4", confidence=confidence,
            )
        if confidence < settings.QG_OCR_WARN_CONFIDENCE:
            warnings.append(
                f"Низкое качество распознавания (уверенность {confidence:.2f}): "
                f"часть содержимого может быть потеряна."
            )
        return QualityVerdict(ACCEPT, "Читаемость подтверждена.", "QG-4",
                              confidence=confidence)

    def _quick_ocr_confidence(self, context: DocumentContext) -> Optional[float]:
        from core.providers.tesseract_fallback import (
            OcrUnavailable,
            TesseractFallbackProvider,
        )

        try:
            pages = list(range(1, max(1, settings.QG_OCR_PAGES) + 1))
            blocks = TesseractFallbackProvider(storage=self.storage).parse_pages(
                context.uri, pages
            )
        except OcrUnavailable as exc:
            logger.info("Быстрый OCR недоступен для %s: %s", context.uri, exc)
            return None
        except Exception as exc:  # noqa: BLE001 — OCR не обязан быть поднят
            logger.info("Быстрый OCR не отработал на %s: %s", context.uri, exc)
            return None

        if not blocks:
            return 0.0
        weights = [max(1, len(b.text or "")) for b in blocks]
        return sum(b.confidence * w for b, w in zip(blocks, weights)) / sum(weights)

    # ------------------------------- QG-5 классификация и маршрутизация
    def routing(self, context: DocumentContext) -> Dict[str, Any]:
        """
        Тип документа и предложение размещения. Метки доступа проставляются
        здесь, на приёме, но утверждает их человек — предложение уходит в
        очередь модерации, а не применяется само.
        """
        classification = context.classification
        document_type = self._document_type(context, classification)

        sensitivity = "internal"
        if document_type == ROUTE_DRAWING:
            sensitivity = "confidential"

        return {
            "document_type": document_type,
            "genre": classification.genre,
            "source": classification.source,
            "suggested_container": self._container(document_type),
            "suggested_sensitivity": sensitivity,
            "suggested_access_labels": self._labels(document_type),
            "needs_moderation": True,
            "note": "Предложение сформировано системой, утверждает человек.",
        }

    @staticmethod
    def _document_type(context: DocumentContext, classification) -> str:
        if context.kind == filetypes.KIND_CAD or classification.genre == GENRE_DRAWING:
            return ROUTE_DRAWING
        if classification.genre == GENRE_SPREADSHEET:
            return ROUTE_TABULAR
        if classification.source == SOURCE_RASTER:
            return ROUTE_MULTIMODAL
        if classification.genre == GENRE_MIXED:
            return ROUTE_MIXED
        if classification.genre == GENRE_TEXT:
            return ROUTE_TEXT
        return ROUTE_MIXED

    @staticmethod
    def _container(document_type: str) -> str:
        return {
            ROUTE_DRAWING: "Конструкторская документация",
            ROUTE_TABULAR: "Реестры и ведомости",
            ROUTE_TEXT: "Организационные документы",
            ROUTE_MIXED: "Общая документация",
            ROUTE_MULTIMODAL: "Сканы и фотокопии",
        }.get(document_type, "Общая документация")

    @staticmethod
    def _labels(document_type: str) -> List[str]:
        if document_type == ROUTE_DRAWING:
            return ["конструкторский отдел"]
        if document_type == ROUTE_TABULAR:
            return ["планово-экономический отдел"]
        return ["все сотрудники"]


# ===========================================================================
# Помощники
# ===========================================================================

def _can_open(data: bytes, extension: str) -> Tuple[bool, str]:
    """Открывается ли файл своим форматом. Повреждённый отсекается сразу."""
    kind = filetypes.kind_of(extension)
    try:
        if kind == filetypes.KIND_PDF:
            import pymupdf
            document = pymupdf.open(stream=data, filetype="pdf")
            pages = document.page_count
            document.close()
            return (pages > 0, "в PDF нет страниц")
        if kind == filetypes.KIND_IMAGE:
            import io

            from PIL import Image
            image = Image.open(io.BytesIO(data))
            image.verify()
            return True, ""
        if extension == "xlsx":
            import io

            import openpyxl
            workbook = openpyxl.load_workbook(io.BytesIO(data), read_only=True)
            workbook.close()
            return True, ""
        if extension == "docx":
            import io

            import docx
            docx.Document(io.BytesIO(data))
            return True, ""
    except ImportError:  # pragma: no cover — библиотеки может не быть
        return True, ""
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)[:200]
    return True, ""


def _text_sample(context: DocumentContext, limit: int = 4000) -> str:
    """Небольшой кусок текста документа для отпечатка сходства."""
    layer = None
    try:
        layer = context.text_layer
    except Exception:  # noqa: BLE001
        layer = None

    if layer is not None:
        parts: List[str] = []
        for page in sorted(layer.pages)[:3]:
            parts.extend(line.text for line in layer.lines(page))
            if sum(len(p) for p in parts) > limit:
                break
        return " ".join(parts)[:limit]

    if context.kind in (filetypes.KIND_PLAIN_TEXT, filetypes.KIND_SPREADSHEET):
        for encoding in ("utf-8", "cp1251"):
            try:
                return context.data[:limit * 2].decode(encoding)[:limit]
            except UnicodeDecodeError:
                continue
    return ""


def _simhash(text: str) -> int:
    """
    64-битный отпечаток текста. Почти одинаковые документы дают отпечатки,
    различающиеся единицами бит, и это ловит переименованные копии и
    обновлённые редакции, которых точная контрольная сумма не видит.
    """
    tokens = re.findall(r"\w{3,}", (text or "").lower())
    if not tokens:
        return 0

    vector = [0] * _SIMHASH_BITS
    for token in tokens:
        digest = int(hashlib.md5(token.encode("utf-8")).hexdigest()[:16], 16)
        for bit in range(_SIMHASH_BITS):
            vector[bit] += 1 if (digest >> bit) & 1 else -1

    fingerprint = 0
    for bit, value in enumerate(vector):
        if value > 0:
            fingerprint |= 1 << bit
    return fingerprint


def _hamming(left: int, right: int) -> int:
    if not left or not right:
        return _SIMHASH_BITS
    return bin(left ^ right).count("1")


def _document_date(path: str, data: bytes) -> Optional[datetime]:
    """Дата документа: из метаданных формата, иначе из имени файла."""
    extension = filetypes.extension_of(path)

    if filetypes.kind_of(extension) == filetypes.KIND_PDF:
        try:
            import pymupdf
            document = pymupdf.open(stream=data, filetype="pdf")
            raw = (document.metadata or {}).get("creationDate") or ""
            document.close()
            parsed = _parse_pdf_date(raw)
            if parsed:
                return parsed
        except Exception:  # noqa: BLE001
            pass

    if extension == "docx":
        try:
            import io

            import docx
            created = docx.Document(io.BytesIO(data)).core_properties.created
            if created:
                return created.replace(tzinfo=created.tzinfo or timezone.utc)
        except Exception:  # noqa: BLE001
            pass

    match = _DATE_IN_NAME.search(os.path.basename(path))
    if match:
        try:
            year = int(match.group(0)[:4])
            if 1950 <= year <= datetime.now(timezone.utc).year:
                return datetime(year, 1, 1, tzinfo=timezone.utc)
        except ValueError:
            pass
    return None


def _parse_pdf_date(raw: str) -> Optional[datetime]:
    """PDF хранит дату как `D:20240115120000+03'00'`."""
    match = re.match(r"D:(\d{4})(\d{2})(\d{2})", raw or "")
    if not match:
        return None
    try:
        return datetime(
            int(match.group(1)), int(match.group(2)), int(match.group(3)),
            tzinfo=timezone.utc,
        )
    except ValueError:
        return None
