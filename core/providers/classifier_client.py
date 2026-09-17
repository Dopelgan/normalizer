"""
Клиент классификатора приёма (слои G-2, G-3, G-4).

Сервис необязателен. Когда эндпоинт не настроен или недоступен, клиент
возвращает `None`, и Data Gateway решает правилами: конвейер обязан
работать без ML-сервиса, просто с меньшей точностью отсева.

Устроен так же, как клиенты детектора и Florence: адрес из настроек,
таймаут, отказ не роняет обработку.
"""

from __future__ import annotations

import base64
import logging
from typing import Any, Dict, Optional

import requests

from core.config import settings

logger = logging.getLogger(__name__)


class ClassifierClient:
    def __init__(self, endpoint: Optional[str] = None, timeout: Optional[int] = None):
        raw = endpoint if endpoint is not None else settings.CLASSIFIER_ENDPOINT
        self.endpoint = raw.rstrip("/") if raw else None
        self.timeout = timeout or settings.CLASSIFIER_TIMEOUT

    @property
    def available(self) -> bool:
        return bool(self.endpoint)

    # --------------------------------------------------------------- G-2
    def classify_document(self, path: str, sample: bytes) -> Optional[Dict[str, Any]]:
        return self._call("classify/document", {
            "path": path,
            "sample_b64": base64.b64encode(sample).decode("ascii"),
        })

    # --------------------------------------------------------------- G-3
    def classify_image(self, path: str, sample: bytes) -> Optional[Dict[str, Any]]:
        return self._call("classify/image", {
            "path": path,
            "image_b64": base64.b64encode(sample).decode("ascii"),
        })

    # --------------------------------------------------------------- G-4
    def resolve_ambiguous(
        self, path: str, sample: bytes, company_profile: str
    ) -> Optional[Dict[str, Any]]:
        """Спорный случай — основной модели, вместе с профилем организации."""
        return self._call("resolve", {
            "path": path,
            "sample_b64": base64.b64encode(sample).decode("ascii"),
            "company_profile": company_profile,
        })

    # -------------------------------------------------------------- общее
    def _call(self, route: str, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not self.endpoint:
            return None
        try:
            response = requests.post(
                f"{self.endpoint}/{route}", json=payload, timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as exc:
            logger.warning("Классификатор недоступен (%s): %s", route, exc)
            return None
        except ValueError as exc:
            logger.warning("Классификатор вернул не-JSON (%s): %s", route, exc)
            return None

        category = data.get("category")
        if not category:
            logger.warning("Классификатор не вернул категорию (%s)", route)
            return None
        return {"category": str(category), "confidence": float(data.get("confidence", 0.5))}
