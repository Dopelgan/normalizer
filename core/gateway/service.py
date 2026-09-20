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
from core.gateway import text_probe
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
# Маркеры ищутся по границам слов, а не подстрокой. Подстрочный поиск по
# двухбуквенным маркерам («ту », «ост », «re:») срабатывал на случайных
# сочетаниях — в том числе внутри сжатого потока PDF, где «re:» встречается
# просто как последовательность байтов.
def _markers(*patterns: str) -> tuple:
    return tuple(re.compile(pattern, re.IGNORECASE) for pattern in patterns)


_TECHNICAL_MARKERS = _markers(
    r"\bгост\b", r"\bост\s*\d", r"\bту\s*\d", r"\bчерт[её]ж\w*",
    r"\bспецификац\w+", r"\bтехническ\w+", r"\bдопуск\w*",
    r"\bшероховат\w+", r"\bсборочн\w+", r"\bдеталь\w*", r"\bузел\b",
    r"\bсхема\w*", r"\bчертеж\w*",
)
_BUSINESS_MARKERS = _markers(
    r"\bдоговор\w*", r"\bприказ\w*", r"\bрегламент\w*", r"\bинструкц\w+",
    r"\bположение\b", r"\bакт\b", r"\bнакладн\w+", r"\bсч[ёе]т\b",
    r"\bсчет-фактур\w*", r"\bпротокол\w*", r"\bустав\b", r"\bотч[ёе]т\w*",
    r"\bприложение\s*№?\s*\d", r"\bграфик\b", r"\bсоглашени\w+",
    r"\bспецификация\s+к\s+договору",
    # Кадровый документооборот — это деловые документы организации, а не
    # «личное»: заявление на отпуск пишется по форме и хранится в кадрах.
    # Без этих маркеров слово «отпуск» уводило такие файлы в «личное».
    r"\bзаявлени\w+", r"\bдолжностн\w+\s+инструкц\w+", r"\bтрудов\w+\s+договор",
    r"\bсчет[- ]фактур\w*", r"\bнакладная\b", r"\bупд\b", r"\bтз\b",
)
# Переписка опознаётся по обороту письма, а не по одному слову: «кому:» в
# бланке и «с уважением» в подписи — это переписка, а слово «письмо» в
# названии приложения к договору — ещё нет.
_CORRESPONDENCE_MARKERS = _markers(
    r"(?:^|\n)\s*(?:re|fwd|fw)\s*:", r"\bкому\s*:", r"\bот\s+кого\s*:",
    r"\bс\s+уважением\b", r"\bдобрый\s+(?:день|вечер)\b",
    r"\bздравствуйте\b", r"\bпереписка\w*", r"\bисх\.\s*№", r"\bвх\.\s*№",
)
_PERSONAL_MARKERS = _markers(
    r"\bотпуск\w*", r"\bсвадьб\w+", r"\bдень\s+рождения\b", r"\bличное\b",
    r"\bсемья\b", r"\bфото\s+с\b",
)

# Категории в порядке разбора и уверенность, с которой правило их называет.
_RULE_ORDER = (
    (CATEGORY_TECHNICAL, _TECHNICAL_MARKERS, 0.8),
    (CATEGORY_BUSINESS, _BUSINESS_MARKERS, 0.78),
    (CATEGORY_CORRESPONDENCE, _CORRESPONDENCE_MARKERS, 0.72),
    (CATEGORY_PERSONAL, _PERSONAL_MARKERS, 0.7),
)

# Категории, которые нельзя объявить по одному лишь пути. Путь к файлу —
# слабый признак: «1-2. Договора и ДС/…/Приложение 2. График.pdf» говорит о
# деловом документе, но ничего не говорит о переписке.
_TEXT_ONLY_CATEGORIES = (CATEGORY_CORRESPONDENCE,)

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

        # Имя, каталог и размер известны без файла — эти проверки первыми.
        verdict = self.path_layer(path) or self.size_layer(size)
        if verdict is not None:
            return verdict

        # Дальше решение принимается по содержимому, поэтому файл читается
        # один раз: и для опознания формата, и для классификации. Картинку
        # судить по первым 64 КБ нельзя (у BMP и TIFF в них только заголовок
        # и пара строк развёртки), а формат по обрезку zip не определяется.
        data = self._read_all(path)
        resolved = filetypes.resolve(path, data)

        verdict = self.type_layer(path, resolved)
        if verdict is not None:
            return verdict

        if filetypes.is_image(resolved.file_type):
            verdict = self.image_layer(path, data, 1.0 if data else 0.0)
        else:
            inspected = 1.0 if data else 0.0
            verdict = self.content_layer(path, data, inspected, resolved)

        if resolved.mismatch:
            verdict.signals.setdefault("type_mismatch", resolved.explanation)

        if verdict.confidence >= self.profile.min_confidence:
            return verdict
        return self.escalation_layer(path, data[:_SAMPLE_BYTES], verdict)

    # -------------------------------------------------- G-1 формальные признаки
    def formal_layer(
        self,
        path: str,
        size: Optional[int],
        resolved: Optional[filetypes.TypeVerdict] = None,
    ) -> Optional[GatewayVerdict]:
        """
        Тип, размер, путь, имя. Самый дешёвый слой отсекает большую часть
        очевидного мусора, не открывая файл вовсе.

        `resolved` — тип, уже определённый по содержимому. Без него формат
        берётся из расширения: так слой работает и там, где файла в руках
        ещё нет.
        """
        return (
            self.path_layer(path)
            or self.size_layer(size)
            or self.type_layer(path, resolved)
        )

    def path_layer(self, path: str) -> Optional[GatewayVerdict]:
        """Имя и каталог: системный файл, запрещённая или личная область."""
        name = os.path.basename(path).lower()

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

        return None

    def type_layer(
        self, path: str, resolved: Optional[filetypes.TypeVerdict] = None
    ) -> Optional[GatewayVerdict]:
        """
        Формат файла. Расширение — утверждение отправителя, поэтому когда
        байты уже прочитаны, тип берётся по сигнатуре: от него зависит, какой
        уровень разбора возьмётся за документ дальше.
        """
        if resolved is None:
            resolved = filetypes.TypeVerdict(
                filetypes.extension_of(path), filetypes.extension_of(path), None, False
            )
        file_type = resolved.file_type

        if file_type not in {e.lower() for e in self.profile.allowed_extensions}:
            reason = filetypes.rejection_reason(file_type) or (
                f"Тип .{file_type} не разрешён профилем приёма."
            )
            if resolved.mismatch:
                reason = f"{reason} {resolved.explanation}"
            return GatewayVerdict(
                REJECT, reason, "G-1",
                signals={"file_type": file_type, "declared_type": resolved.declared},
            )
        return None

    def size_layer(self, size: Optional[int]) -> Optional[GatewayVerdict]:
        """Размер, известный из метаданных хранилища."""
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
        self,
        path: str,
        data: bytes,
        inspected: float,
        resolved: Optional[filetypes.TypeVerdict] = None,
    ) -> GatewayVerdict:
        """
        Категория документа. Сначала спрашиваем модель, если она настроена;
        иначе решают правила по тексту документа и пути к нему.

        `data` — файл целиком: текст достаётся инструментом формата, а не
        поиском подстрок в байтах. Раньше здесь искали маркеры прямо в первых
        64 КБ, и сжатый поток PDF давал случайные совпадения — приложение к
        договору становилось перепиской из-за байтов «re:» внутри потока.

        Доля осмотренного важна не меньше самой категории: классификатор,
        увидевший пять процентов документа, не может быть в нём уверен,
        сколько бы он ни заявлял.
        """
        file_type = (resolved.file_type if resolved else filetypes.extension_of(path))
        sample = data[:_SAMPLE_BYTES]

        remote = self.classifier.classify_document(path, sample)
        if remote is not None:
            category, confidence = remote["category"], float(remote["confidence"])
            source = "модель"
            signals: Dict[str, Any] = {}
        else:
            category, confidence, signals = self._categorize(path, data, file_type)
            source = "правила"

        confidence = self._discount(confidence, inspected)
        outcome = self.profile.outcome_for_category(category)
        return GatewayVerdict(
            outcome=outcome,
            reason=self._explain(category, outcome, source),
            layer="G-2",
            category=category,
            confidence=confidence,
            signals={
                "inspected_fraction": round(inspected, 3),
                "decided_by": source,
                **signals,
            },
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

    def _categorize(self, path: str, data: bytes, file_type: str) -> tuple:
        """
        Правила по тексту документа и пути к нему.

        Текст и путь — признаки разного веса. Текст извлечён из файла и
        говорит о содержании; путь говорит лишь о том, куда документ
        положили, поэтому категория по одному пути объявляется с меньшей
        уверенностью, а переписку по нему не объявляют вовсе.
        """
        probe = text_probe.extract(file_type, data)
        signals: Dict[str, Any] = probe.as_dict()

        if probe.reliable:
            category, confidence, hit = _first_match(probe.text)
            if category is not None:
                signals["matched_in"] = "текст"
                signals["matched_marker"] = hit
                return category, confidence, signals

        # Путь целиком, а не только имя файла: каталог «Договоры и ДС» —
        # такой же признак, как слово в названии, и раньше он отбрасывался.
        category, confidence, hit = _first_match(path.replace("/", " "))
        if category is not None and category not in _TEXT_ONLY_CATEGORIES:
            signals["matched_in"] = "путь"
            signals["matched_marker"] = hit
            # Путь — признак второго порядка: уверенность ниже, и спорный
            # случай уходит на разбор (G-4), а не решается молча.
            return category, round(confidence * 0.85, 3), signals

        if file_type == "dxf":
            return CATEGORY_TECHNICAL, 0.9, signals
        if file_type in ("xlsx", "csv"):
            return CATEGORY_BUSINESS, 0.65, signals

        if not probe.reliable and not data:
            return CATEGORY_UNRECOGNIZABLE, 0.55, signals
        # Ничего характерного не нашлось: это не повод выбрасывать документ,
        # но и уверенно принимать его не за что — решит слой G-4.
        return CATEGORY_BUSINESS, 0.5, signals

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

def _haystack(value: str) -> str:
    """
    Строка, пригодная для поиска по границам слов.

    В именах файлов слова разделяют не пробелы, а `_`, `-` и точки, причём
    `_` для регулярного выражения — такой же символ слова, как буква:
    в «личное_отпуск.jpg» маркер `\bличное\b` без этой нормализации не
    находится.
    """
    return re.sub(r"[_\-./\\]+", " ", value or "")


def _matches(haystack: str, markers) -> bool:
    """Есть ли в тексте хоть один маркер набора."""
    prepared = _haystack(haystack)
    return any(marker.search(prepared) for marker in markers)


def _first_match(haystack: str) -> tuple:
    """
    Первая подходящая категория по порядку разбора: техническое, деловое,
    переписка, личное. Возвращает и сам сработавший маркер — он попадает в
    сигналы приёма, чтобы решение можно было проверить, а не принять на веру.
    """
    prepared = _haystack(haystack)
    for category, markers, confidence in _RULE_ORDER:
        for marker in markers:
            found = marker.search(prepared)
            if found:
                return category, confidence, found.group(0).strip()
    return None, 0.0, None


def _as_text(sample: bytes) -> str:
    for encoding in ("utf-8", "cp1251"):
        try:
            return sample.decode(encoding)
        except UnicodeDecodeError:
            continue
    return sample.decode("utf-8", errors="ignore")
