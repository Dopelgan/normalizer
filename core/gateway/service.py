"""
DATA GATEWAY — отсев непригодных данных.

Отвечает ровно на один вопрос: относится ли файл к корпоративным знаниям.
Вопрос «пригоден ли документ технически» — это Quality Gate, и смешивать их
нельзя: личная фотография технически безупречна и любую проверку качества
пройдёт. Поэтому Data Gateway строго предшествует Quality Gate.

Четыре слоя, от дешёвого к дорогому: формальные признаки, быстрый
классификатор, классификация изображений, разбор спорных случаев. Как
только слой вынес уверенное решение, дальше не идём — в этом и смысл
порядка.

Исходов три, а не два. Карантин — не роскошь: без него система либо
пропускает мусор, либо молча теряет нужные документы.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from core import filetypes
from core.gateway.profile import (
    CATEGORY_BUSINESS,
    CATEGORY_CORRESPONDENCE,
    CATEGORY_PERSONAL,
    CATEGORY_SYSTEM,
    CATEGORY_TECHNICAL,
    CATEGORY_UNRECOGNIZABLE,
    IMAGE_DOCUMENT_SCAN,
    IMAGE_DRAWING,
    IMAGE_OBJECT_PHOTO,
    IMAGE_PERSONAL_PHOTO,
    IMAGE_SCHEME,
    IMAGE_SCREENSHOT,
    GatewayProfile,
    load_profile,
)
from core.providers import raster_drawing
from core.providers.classifier_client import ClassifierClient
from core.providers.storage import StorageProvider, StorageProviderFactory

logger = logging.getLogger(__name__)

ACCEPT = "accept"
QUARANTINE = "quarantine"
REJECT = "reject"

# Сколько байт содержимого осматривает быстрый классификатор.
_SAMPLE_BYTES = 64 * 1024

_SYSTEM_NAMES = (
    "thumbs.db", "desktop.ini", ".ds_store", "~$", "index.dat", "ntuser.dat",
)
_TECHNICAL_MARKERS = (
    "гост", "ост ", "ту ", "чертёж", "чертеж", "спецификац", "техническ",
    "допуск", "шероховат", "сборочн", "деталь", "узел", "схема",
)
_BUSINESS_MARKERS = (
    "договор", "приказ", "регламент", "инструкц", "положение", "акт ",
    "накладн", "счёт", "счет-фактур", "протокол", "устав", "отчёт", "отчет",
)
_CORRESPONDENCE_MARKERS = (
    "re:", "fwd:", "кому:", "от кого:", "с уважением", "добрый день",
    "здравствуйте", "переписка",
)
_PERSONAL_MARKERS = (
    "отпуск", "свадьб", "день рождения", "личное", "семья", "фото с",
)

# Род растра (core.providers.raster_drawing) -> категория приёма.
_IMAGE_CATEGORY_BY_KIND = {
    raster_drawing.KIND_DRAWING: IMAGE_DRAWING,
    raster_drawing.KIND_SCHEME: IMAGE_SCHEME,
    raster_drawing.KIND_TEXT_SCAN: IMAGE_DOCUMENT_SCAN,
    raster_drawing.KIND_SCREENSHOT: IMAGE_SCREENSHOT,
    raster_drawing.KIND_PHOTO: IMAGE_OBJECT_PHOTO,
    raster_drawing.KIND_UNREADABLE: CATEGORY_UNRECOGNIZABLE,
}


@dataclass
class GatewayVerdict:
    """Решение по файлу и почему оно принято."""

    outcome: str
    reason: str
    layer: str
    category: Optional[str] = None
    confidence: float = 1.0
    signals: Dict[str, Any] = field(default_factory=dict)

    @property
    def accepted(self) -> bool:
        return self.outcome == ACCEPT

    def as_dict(self) -> Dict[str, Any]:
        return {
            "outcome": self.outcome,
            "reason": self.reason,
            "layer": self.layer,
            "category": self.category,
            "confidence": round(self.confidence, 3),
            "signals": self.signals,
        }


class DataGateway:
    """Четыре слоя отсева, от дешёвого к дорогому."""

    def __init__(
        self,
        profile: Optional[GatewayProfile] = None,
        storage: Optional[StorageProvider] = None,
        classifier: Optional[ClassifierClient] = None,
    ):
        self.profile = profile or load_profile()
        self.storage = storage or StorageProviderFactory.default()
        self.classifier = classifier or ClassifierClient()

    # ------------------------------------------------------------- публичное
    def evaluate(
        self, s3_fileid: str, uri: Optional[str] = None, size: Optional[int] = None
    ) -> GatewayVerdict:
        """Пропустить файл дальше, отправить в карантин или отклонить."""
        path = uri or s3_fileid

        verdict = self.formal_layer(path, size)
        if verdict is not None:
            return verdict

        if filetypes.is_image(filetypes.extension_of(path)):
            # Картинку нельзя судить по первым 64 КБ. У сжатых форматов в них
            # умещается весь файл, у BMP и TIFF — кусок заголовка и несколько
            # первых строк развёртки: декодер такой обрезок не открывает, и
            # безобидный чертёж получал «нераспознаваемое» с уверенностью
            # 0.06 только из-за формата хранения. Картинка читается целиком.
            image_bytes = self._read_all(path)
            sample = image_bytes[:_SAMPLE_BYTES]
            verdict = self.image_layer(path, image_bytes, 1.0 if image_bytes else 0.0)
        else:
            sample, inspected = self._sample(path, size)
            verdict = self.content_layer(path, sample, inspected)

        if verdict.confidence >= self.profile.min_confidence:
            return verdict
        return self.escalation_layer(path, sample, verdict)

    # -------------------------------------------------- G-1 формальные признаки
    def formal_layer(self, path: str, size: Optional[int]) -> Optional[GatewayVerdict]:
        """
        Тип, размер, путь, имя. Самый дешёвый слой отсекает большую часть
        очевидного мусора, не открывая файл вовсе.
        """
        name = os.path.basename(path).lower()
        extension = filetypes.extension_of(path)

        if any(marker in name for marker in _SYSTEM_NAMES):
            return GatewayVerdict(
                REJECT, f"Системный файл {name!r} не является знанием.", "G-1",
                category=CATEGORY_SYSTEM,
            )

        if self.profile.is_forbidden_path(path):
            return GatewayVerdict(
                REJECT, "Файл лежит в каталоге, исключённом профилем приёма.",
                "G-1", category=CATEGORY_SYSTEM,
            )

        if self.profile.is_personal_path(path) and not self.profile.accept_personal_areas:
            # Личная область — риск и мусор, но там же встречаются
            # единственные экземпляры документов. Поэтому карантин, а не отказ.
            return GatewayVerdict(
                QUARANTINE,
                "Файл в личной области хранилища: по умолчанию такие в базу "
                "знаний не принимаются, решение за администратором.",
                "G-1", category=CATEGORY_PERSONAL, confidence=1.0,
            )

        if extension not in {e.lower() for e in self.profile.allowed_extensions}:
            reason = filetypes.rejection_reason(extension) or (
                f"Тип .{extension} не разрешён профилем приёма."
            )
            return GatewayVerdict(REJECT, reason, "G-1")

        if size is not None:
            if size < self.profile.min_size_bytes:
                return GatewayVerdict(
                    REJECT,
                    f"Файл слишком мал ({size} Б): содержимого в нём нет.",
                    "G-1",
                )
            if size > self.profile.max_size_bytes:
                return GatewayVerdict(
                    QUARANTINE,
                    f"Файл больше предела профиля "
                    f"({size} Б против {self.profile.max_size_bytes} Б).",
                    "G-1",
                )

        return None

    # ------------------------------------------- G-2 быстрый классификатор
    def content_layer(
        self, path: str, sample: bytes, inspected: float
    ) -> GatewayVerdict:
        """
        Категория документа. Сначала спрашиваем модель, если она настроена;
        иначе решают правила по имени и содержимому.

        Доля осмотренного важна не меньше самой категории: классификатор,
        увидевший пять процентов документа, не может быть в нём уверен,
        сколько бы он ни заявлял.
        """
        remote = self.classifier.classify_document(path, sample)
        if remote is not None:
            category, confidence = remote["category"], float(remote["confidence"])
            source = "модель"
        else:
            category, confidence = self._categorize(path, sample)
            source = "правила"

        confidence = self._discount(confidence, inspected)
        outcome = self.profile.outcome_for_category(category)
        return GatewayVerdict(
            outcome=outcome,
            reason=self._explain(category, outcome, source),
            layer="G-2",
            category=category,
            confidence=confidence,
            signals={"inspected_fraction": round(inspected, 3), "decided_by": source},
        )

    # ---------------------------------------- G-3 классификация изображений
    def image_layer(self, path: str, data: bytes, inspected: float) -> GatewayVerdict:
        """
        Изображения разбираются отдельно: чертёж и селфи — разные вещи.
        `data` — весь файл, а не образец: решение принимается по геометрии
        картинки, и обрезок для этого не годится.
        """
        remote = self.classifier.classify_image(path, data[:_SAMPLE_BYTES])
        raster: Dict[str, Any] = {}
        if remote is not None:
            category, confidence = remote["category"], float(remote["confidence"])
            source = "модель"
        else:
            category, confidence, raster = self._categorize_image(path, data)
            source = "правила"

        outcome = self.profile.outcome_for_image(category)
        return GatewayVerdict(
            outcome=outcome,
            reason=self._explain(category, outcome, source),
            layer="G-3",
            category=category,
            confidence=self._discount(confidence, inspected),
            signals={
                "inspected_fraction": round(inspected, 3),
                "decided_by": source,
                **({"raster": raster} if raster else {}),
            },
        )

    # ------------------------------------------ G-4 разбор спорных случаев
    def escalation_layer(
        self, path: str, sample: bytes, previous: GatewayVerdict
    ) -> GatewayVerdict:
        """
        Дорого и на малой доле файлов. Спорный случай уходит основной модели
        вместе с описанием деятельности организации из профиля. Модель не
        настроена — документ отправляется в карантин, а не выбрасывается.
        """
        resolved = self.classifier.resolve_ambiguous(
            path, sample, self.profile.company_profile
        )
        if resolved is None:
            return GatewayVerdict(
                QUARANTINE,
                f"Классификатор не уверен (уверенность {previous.confidence:.2f}), "
                "а разбор спорных случаев не настроен — решение за администратором.",
                "G-4", category=previous.category,
                confidence=previous.confidence, signals=previous.signals,
            )

        category = resolved["category"]
        outcome = self.profile.outcome_for_category(category)
        return GatewayVerdict(
            outcome=outcome,
            reason=self._explain(category, outcome, "основная модель"),
            layer="G-4",
            category=category,
            confidence=float(resolved.get("confidence", 0.8)),
            signals={**previous.signals, "escalated_from": previous.layer},
        )

    # ------------------------------------------------------------ внутреннее
    def _read_all(self, path: str) -> bytes:
        """Весь файл: нужен там, где решение принимается по содержимому целиком."""
        try:
            return self.storage.read_bytes(path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Не удалось прочитать %s для классификации: %s", path, exc)
            return b""

    def _sample(self, path: str, size: Optional[int]) -> tuple:
        """Кусок содержимого и доля файла, которую он составляет."""
        try:
            data = self.storage.read_bytes(path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Не удалось прочитать %s для классификации: %s", path, exc)
            return b"", 0.0

        total = size or len(data) or 1
        sample = data[:_SAMPLE_BYTES]
        return sample, min(1.0, len(sample) / total)

    def _discount(self, confidence: float, inspected: float) -> float:
        """Уверенность падает пропорционально тому, как мало было осмотрено."""
        if inspected >= self.profile.min_inspected_fraction:
            return max(0.0, min(1.0, confidence))
        factor = max(0.1, inspected / max(self.profile.min_inspected_fraction, 1e-6))
        return max(0.0, min(1.0, confidence * factor))

    @staticmethod
    def _explain(category: str, outcome: str, source: str) -> str:
        titles = {
            CATEGORY_BUSINESS: "деловой документ",
            CATEGORY_TECHNICAL: "техническая документация",
            CATEGORY_CORRESPONDENCE: "переписка",
            CATEGORY_PERSONAL: "личное",
            CATEGORY_SYSTEM: "системный файл",
            CATEGORY_UNRECOGNIZABLE: "нераспознаваемое",
            IMAGE_DRAWING: "чертёж",
            IMAGE_SCHEME: "схема",
            IMAGE_DOCUMENT_SCAN: "скан документа",
            IMAGE_SCREENSHOT: "снимок экрана",
            IMAGE_OBJECT_PHOTO: "фотография объекта",
            IMAGE_PERSONAL_PHOTO: "личная фотография",
        }
        verdicts = {
            ACCEPT: "принимается",
            QUARANTINE: "отправляется в карантин",
            REJECT: "отклоняется",
        }
        return (
            f"Определено как {titles.get(category, category)} ({source}); "
            f"по профилю приёма {verdicts[outcome]}."
        )

    def _categorize(self, path: str, sample: bytes) -> tuple:
        """Правила по имени файла и началу содержимого."""
        haystack = (os.path.basename(path) + " " + _as_text(sample)).lower()

        if _matches(haystack, _TECHNICAL_MARKERS):
            return CATEGORY_TECHNICAL, 0.8
        if _matches(haystack, _BUSINESS_MARKERS):
            return CATEGORY_BUSINESS, 0.78
        if _matches(haystack, _CORRESPONDENCE_MARKERS):
            return CATEGORY_CORRESPONDENCE, 0.72
        if _matches(haystack, _PERSONAL_MARKERS):
            return CATEGORY_PERSONAL, 0.7

        extension = filetypes.extension_of(path)
        if extension in ("dxf",):
            return CATEGORY_TECHNICAL, 0.9
        if extension in ("xlsx", "csv"):
            return CATEGORY_BUSINESS, 0.65

        if not _as_text(sample).strip():
            return CATEGORY_UNRECOGNIZABLE, 0.55
        # Ничего характерного не нашлось: это не повод выбрасывать документ,
        # но и уверенно принимать его не за что — решит слой G-4.
        return CATEGORY_BUSINESS, 0.5

    @staticmethod
    def _categorize_image(path: str, data: bytes) -> tuple:
        """
        Род изображения по геометрии листа, а не по «доле белого».

        Раньше здесь было два числа: если картинка светлая и неяркая, она
        объявлялась чертежом с уверенностью 0.8. Под это правило подходил
        любой блёклый скан и даже картинка 80x50, на которой ничего не
        разглядеть. Теперь решает `raster_drawing`: рамка формата, длинные
        линии, полосы строк — и отдельный ответ «разрешение не позволяет
        судить» вместо выдуманного чертежа.

        Имя файла остаётся подсказкой, но подсказкой второго порядка: оно
        не отменяет содержимое, а расходится с ним — и тогда решение
        принимает человек.
        """
        name = os.path.basename(path).lower()
        verdict = raster_drawing.analyse(data)
        signals = verdict.as_dict()

        category = _IMAGE_CATEGORY_BY_KIND.get(verdict.kind, CATEGORY_UNRECOGNIZABLE)
        confidence = verdict.confidence

        # Имя говорит «личное», а на листе чертёж — это противоречие, а не
        # повод выбросить документ: уверенность падает, и случай уходит на
        # разбор спорных (G-4), то есть человеку.
        if _matches(name, _PERSONAL_MARKERS):
            if category in (IMAGE_OBJECT_PHOTO, IMAGE_PERSONAL_PHOTO, CATEGORY_UNRECOGNIZABLE):
                return IMAGE_PERSONAL_PHOTO, max(confidence, 0.7), signals
            signals["name_conflict"] = "имя файла помечено как личное"
            return category, min(confidence, 0.5), signals

        if re.search(r"(img_|dsc|photo|снимок|selfie)", name) and category in (
            IMAGE_OBJECT_PHOTO, CATEGORY_UNRECOGNIZABLE
        ):
            return IMAGE_PERSONAL_PHOTO, max(confidence, 0.7), signals
        if re.search(r"(screen|скрин)", name) and category in (
            IMAGE_OBJECT_PHOTO, CATEGORY_UNRECOGNIZABLE
        ):
            return IMAGE_SCREENSHOT, max(confidence, 0.7), signals

        return category, confidence, signals


# ===========================================================================
# Помощники
# ===========================================================================

def _matches(haystack: str, markers) -> bool:
    return any(marker in haystack for marker in markers)


def _as_text(sample: bytes) -> str:
    for encoding in ("utf-8", "cp1251"):
        try:
            return sample.decode(encoding)
        except UnicodeDecodeError:
            continue
    return sample.decode("utf-8", errors="ignore")
