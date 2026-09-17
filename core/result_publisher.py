"""Публикация готового результата запроса через настроенный канал."""

import logging
from typing import Union

from core.models.contract import ParseResponse, ResultsResponse
from core.result_delivery import ResultDeliveryError, get_delivery_provider

logger = logging.getLogger(__name__)


def publish_result(response: Union[ParseResponse, ResultsResponse, dict]) -> bool:
    """
    Отправляет результат push-каналом, если он включён.
    Ошибка доставки не считается ошибкой обработки: результат уже лежит в
    Redis и доступен через GET /internal/v1/parse/results/{request_id}.
    """
    payload = response if isinstance(response, dict) else response.model_dump(mode="json")
    request_id = payload.get("request_id", "?")
    try:
        provider = get_delivery_provider()
        provider.send(payload)
        if provider.get_name() != "none":
            logger.info("Результат %s отправлен через %s", request_id, provider.get_name())
        return True
    except ResultDeliveryError as exc:
        logger.error("Не удалось доставить результат %s: %s", request_id, exc)
        return False
    except Exception as exc:  # noqa: BLE001
        logger.exception("Неожиданная ошибка доставки результата %s: %s", request_id, exc)
        return False
