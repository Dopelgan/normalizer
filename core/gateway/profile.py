"""
Профиль приёма: какие данные эта установка считает корпоративным знанием.

Правила отсева описывают конкретную организацию, а не конвейер, поэтому
они вынесены из кода в профиль. Профиль читается из YAML; если файла нет,
берутся умолчания — конвейер обязан работать и без настройки.

Категории и допустимость задаются здесь же: одна установка принимает
переписку, другая считает её мусором, и спорить об этом в коде бессмысленно.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from core import filetypes

logger = logging.getLogger(__name__)

# Категории слоя G-2.
CATEGORY_BUSINESS = "business_document"
CATEGORY_TECHNICAL = "technical_documentation"
CATEGORY_CORRESPONDENCE = "correspondence"
CATEGORY_PERSONAL = "personal"
CATEGORY_SYSTEM = "system_file"
CATEGORY_UNRECOGNIZABLE = "unrecognizable"

# Категории слоя G-3 (только изображения).
IMAGE_DRAWING = "drawing"
IMAGE_SCHEME = "scheme"
# Скан страницы текста: это документ, а не картинка объекта, и принимать его
# нужно наравне с чертежом. Отдельная категория нужна, чтобы блёклый скан
# перестал записываться в чертежи только за то, что он светлый.
IMAGE_DOCUMENT_SCAN = "document_scan"
IMAGE_SCREENSHOT = "screenshot"
IMAGE_OBJECT_PHOTO = "object_photo"
IMAGE_PERSONAL_PHOTO = "personal_photo"


@dataclass
class GatewayProfile:
    """Правила отсева одной установки."""

    allowed_extensions: List[str] = field(
        default_factory=lambda: sorted(filetypes.SUPPORTED_EXTENSIONS)
    )
    # Каталоги, содержимое которых в базу знаний не принимается.
    forbidden_paths: List[str] = field(
        default_factory=lambda: ["/temp/", "/tmp/", "/cache/", "/.trash/", "/recycle"]
    )
    # Личные области: по умолчанию не принимаются, но могут содержать
    # единственные экземпляры документов — поэтому запрет настраиваемый.
    personal_paths: List[str] = field(default_factory=lambda: ["/personal/", "/личное/"])
    personal_exceptions: List[str] = field(default_factory=list)
    accept_personal_areas: bool = False

    min_size_bytes: int = 512
    max_size_bytes: int = 200 * 1024 * 1024

    accepted_categories: List[str] = field(
        default_factory=lambda: [CATEGORY_BUSINESS, CATEGORY_TECHNICAL]
    )
    quarantined_categories: List[str] = field(
        default_factory=lambda: [CATEGORY_CORRESPONDENCE, CATEGORY_UNRECOGNIZABLE]
    )
    rejected_categories: List[str] = field(
        default_factory=lambda: [CATEGORY_PERSONAL, CATEGORY_SYSTEM]
    )

    accepted_image_categories: List[str] = field(
        default_factory=lambda: [IMAGE_DRAWING, IMAGE_SCHEME, IMAGE_DOCUMENT_SCAN]
    )
    quarantined_image_categories: List[str] = field(
        default_factory=lambda: [IMAGE_OBJECT_PHOTO, IMAGE_SCREENSHOT]
    )
    rejected_image_categories: List[str] = field(
        default_factory=lambda: [IMAGE_PERSONAL_PHOTO]
    )

    # Ниже этой уверенности решение слоя G-2 или G-3 не считается
    # достаточным, и документ уходит на разбор спорных случаев (G-4).
    min_confidence: float = 0.6
    # Если классификатор осмотрел меньше этой доли документа, уверенность
    # понижается независимо от того, насколько он в себе уверен.
    min_inspected_fraction: float = 0.2

    # Краткое описание деятельности организации — передаётся основной
    # модели на слое G-4, чтобы спорный случай разбирался в контексте.
    company_profile: str = ""

    # ---------------------------------------------------------------- методы
    def outcome_for_category(self, category: str) -> str:
        if category in self.accepted_categories:
            return "accept"
        if category in self.rejected_categories:
            return "reject"
        return "quarantine"

    def outcome_for_image(self, category: str) -> str:
        if category in self.accepted_image_categories:
            return "accept"
        if category in self.rejected_image_categories:
            return "reject"
        return "quarantine"

    def is_personal_path(self, path: str) -> bool:
        lowered = "/" + path.replace("\\", "/").lower().lstrip("/")
        if any(exception.lower() in lowered for exception in self.personal_exceptions):
            return False
        return any(marker.lower() in lowered for marker in self.personal_paths)

    def is_forbidden_path(self, path: str) -> bool:
        lowered = "/" + path.replace("\\", "/").lower().lstrip("/")
        return any(marker.lower() in lowered for marker in self.forbidden_paths)


# ===========================================================================
# Загрузка
# ===========================================================================

_CACHE: Dict[str, GatewayProfile] = {}


def load_profile(path: Optional[str] = None) -> GatewayProfile:
    """
    Профиль из YAML. Файла нет или он битый — работаем на умолчаниях.

    Если по указанному пути файла нет, а рядом лежит образец
    (`profile.example.yaml` при `profile.yaml`), читается он. В штатной
    поставке `config/profile.yaml` не существует, и механизм профиля был
    выключен целиком: правила правили в образце, а работали умолчания из
    кода, и разницы никто не видел. Образец — это поставляемый профиль, а
    файл рядом с ним — его переопределение на месте.
    """
    from core.config import settings

    path = path or settings.GATEWAY_PROFILE_PATH
    if not path:
        return GatewayProfile()
    if path in _CACHE:
        return _CACHE[path]

    profile = GatewayProfile()
    source = _source_file(path)
    if source is None:
        logger.info(
            "Профиль приёма не найден (%s) — работают умолчания из кода", path
        )
    else:
        logger.info("Профиль приёма: %s", source)
        raw = _read_yaml(source)
        if raw:
            profile = _from_mapping(raw.get("data_gateway") or {})
    _CACHE[path] = profile
    return profile


def _source_file(path: str) -> Optional[str]:
    """Сам файл, иначе поставляемый образец рядом с ним, иначе ничего."""
    if os.path.exists(path):
        return path
    base, ext = os.path.splitext(path)
    example = f"{base}.example{ext or '.yaml'}"
    return example if os.path.exists(example) else None


def reset_cache() -> None:
    """Сбрасывает кеш профиля — нужен тестам и горячей перезагрузке правил."""
    _CACHE.clear()


def _read_yaml(path: str) -> Optional[Dict[str, Any]]:
    try:
        import yaml
    except ImportError:  # pragma: no cover
        logger.warning("PyYAML недоступен — профиль приёма не прочитан")
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}
    except Exception as exc:  # noqa: BLE001 — битый профиль не роняет сервис
        logger.error("Профиль приёма %s не разобран: %s", path, exc)
        return None


def _from_mapping(data: Dict[str, Any]) -> GatewayProfile:
    """
    Значения профиля поверх умолчаний, неизвестные ключи игнорируются.

    Значение приводится к типу умолчания: `min_size_bytes: "512"` в YAML
    иначе доезжает строкой до сравнения с размером файла и роняет приём
    пятисотой на каждом документе.
    """
    profile = GatewayProfile()
    for key, value in (data or {}).items():
        if not hasattr(profile, key):
            logger.warning("Неизвестный ключ профиля приёма: %s", key)
            continue
        if value is None:
            continue
        default = getattr(profile, key)
        try:
            setattr(profile, key, _coerce(value, default))
        except (TypeError, ValueError) as exc:
            logger.error(
                "Ключ профиля приёма %s=%r не подходит по типу (%s) — берём умолчание",
                key, value, exc,
            )
    return profile


def _coerce(value: Any, default: Any) -> Any:
    """Приводит значение из YAML к типу умолчания."""
    if isinstance(default, bool):
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "да")
        return bool(value)
    if isinstance(default, int):
        return int(value)
    if isinstance(default, float):
        return float(value)
    if isinstance(default, list):
        if not isinstance(value, (list, tuple)):
            raise TypeError("ожидался список")
        return [str(item) for item in value]
    if isinstance(default, str):
        return str(value)
    return value
