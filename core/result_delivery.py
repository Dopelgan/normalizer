"""
Опциональный push-канал доставки результата.

По контракту RAG забирает результат сам через
`GET /internal/v1/parse/results/{request_id}`, поэтому доставка по умолчанию
выключена (`RESULT_DELIVERY_TYPE=none`). Каналы ниже — для интеграций, где
push всё-таки нужен: Redis Streams, HTTP callback, NATS JetStream.
"""

import json
import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Dict

import redis
import requests

from core.config import settings

logger = logging.getLogger(__name__)


class ResultDeliveryError(Exception):
    """Не удалось доставить результат ни одной попыткой."""


class ResultDeliveryProvider(ABC):
    @abstractmethod
    def send(self, response: Dict[str, Any]) -> bool:
        """Отправляет результат. True при успехе."""

    @abstractmethod
    def get_name(self) -> str:
        """Имя провайдера для логов."""


class NullDelivery(ResultDeliveryProvider):
    """Заглушка: результат забирается поллингом, push не нужен."""

    def send(self, response: Dict[str, Any]) -> bool:
        logger.debug("Push-доставка отключена, результат ждёт поллинга")
        return True

    def get_name(self) -> str:
        return "none"


class RedisStreamsDelivery(ResultDeliveryProvider):
    def __init__(self, client=None):
        self.redis_client = client or redis.Redis.from_url(settings.EVENT_BROKER_URL)
        self.stream_name = settings.RESULT_STREAM_NAME
        self.max_retries = 3
        self.retry_delay = 1

    def send(self, response: Dict[str, Any]) -> bool:
        data = json.dumps(response, default=str, ensure_ascii=False)
        last_error = None
        for attempt in range(self.max_retries):
            try:
                self.redis_client.xadd(self.stream_name, {"data": data}, maxlen=10000)
                logger.info("Результат отправлен в Redis Stream %s", self.stream_name)
                return True
            except redis.exceptions.RedisError as exc:
                last_error = exc
                logger.warning("Redis Stream, попытка %d: %s", attempt + 1, exc)
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
        raise ResultDeliveryError(f"Redis Streams: {last_error}")

    def get_name(self) -> str:
        return "redis_streams"


class CallbackDelivery(ResultDeliveryProvider):
    def __init__(self):
        self.callback_url = settings.RESULT_CALLBACK_URL
        if not self.callback_url:
            raise ValueError("RESULT_CALLBACK_URL не задан")
        self.timeout = 30
        self.max_retries = 3
        self.retry_delay = 2

    def send(self, response: Dict[str, Any]) -> bool:
        last_error = None
        for attempt in range(self.max_retries):
            try:
                resp = requests.post(
                    self.callback_url, json=response, timeout=self.timeout,
                    headers={"Content-Type": "application/json"},
                )
                if resp.status_code in (200, 201, 202, 204):
                    logger.info("Callback %s: %d", self.callback_url, resp.status_code)
                    return True
                last_error = f"HTTP {resp.status_code}"
                logger.warning("Callback вернул %d, попытка %d", resp.status_code, attempt + 1)
            except requests.RequestException as exc:
                last_error = exc
                logger.warning("Callback, попытка %d: %s", attempt + 1, exc)
            if attempt < self.max_retries - 1:
                time.sleep(self.retry_delay * (attempt + 1))
        raise ResultDeliveryError(f"Callback: {last_error}")

    def get_name(self) -> str:
        return "callback"


class NatsDelivery(ResultDeliveryProvider):
    def __init__(self):
        self.servers = [
            s.strip() for s in (settings.NATS_SERVERS or "nats://nats:4222").split(",") if s.strip()
        ]
        self.subject = settings.NATS_SUBJECT

    def send(self, response: Dict[str, Any]) -> bool:
        try:
            import asyncio

            import nats
        except ImportError as exc:  # pragma: no cover
            raise ResultDeliveryError(
                "Для RESULT_DELIVERY_TYPE=nats нужен пакет nats-py"
            ) from exc

        payload = json.dumps(response, default=str, ensure_ascii=False).encode("utf-8")

        async def _publish() -> None:
            connection = await nats.connect(servers=self.servers)
            try:
                js = connection.jetstream()
                ack = await js.publish(self.subject, payload)
                logger.info("Результат отправлен в NATS %s, seq=%s", self.subject, ack.seq)
            finally:
                await connection.close()

        try:
            asyncio.run(_publish())
            return True
        except Exception as exc:  # noqa: BLE001
            raise ResultDeliveryError(f"NATS: {exc}") from exc

    def get_name(self) -> str:
        return "nats"


def get_delivery_provider() -> ResultDeliveryProvider:
    delivery_type = (settings.RESULT_DELIVERY_TYPE or "none").lower()
    if delivery_type in ("none", "", "polling"):
        return NullDelivery()
    if delivery_type == "callback":
        return CallbackDelivery()
    if delivery_type == "nats":
        return NatsDelivery()
    if delivery_type == "redis_streams":
        return RedisStreamsDelivery()
    logger.warning("Неизвестный RESULT_DELIVERY_TYPE=%r, push отключён", delivery_type)
    return NullDelivery()
