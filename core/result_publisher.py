"""Публикация готового результата запроса через настроенный канал."""

import logging
from typing import Union

from core.models.contract import ParseResponse, ResultsResponse, SyncPendingResponse
from core.result_delivery import ResultDeliveryError, get_delivery_provider
from core.transport_budget import MAX_RESPONSE_BYTES, encode_payload, resource_error

logger = logging.getLogger(__name__)


def publish_result(
    response: Union[ParseResponse, ResultsResponse, SyncPendingResponse, dict]
) -> bool:
    """
    Отправляет результат push-каналом, если он включён.
    Ошибка доставки не считается ошибкой обработки: результат уже лежит в
    Redis и доступен через GET /internal/v1/parse/results/{request_id}.
    """
    payload = response if isinstance(response, dict) else response.model_dump(mode="json")
    request_id = payload.get("request_id", "?")
    try:
        provider = get_delivery_provider()
        # Match the actual provider representation: requests' json= uses ASCII
        # escapes/default separators, Redis/NATS use UTF-8/default separators.
        ascii_wire = provider.get_name() == "callback"
        if encode_payload(payload, MAX_RESPONSE_BYTES, ensure_ascii=ascii_wire, compact=False) is None:
            payload = {key: payload[key] for key in ("request_id", "event_id", "operation", "dialog_id")
                       if key in payload}
            payload.update(status="error", documents=[], error=resource_error())
            if encode_payload(payload, MAX_RESPONSE_BYTES, ensure_ascii=ascii_wire, compact=False) is None:
                logger.error("Результат %s превышает бюджет даже в error envelope", request_id)
                return False
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
