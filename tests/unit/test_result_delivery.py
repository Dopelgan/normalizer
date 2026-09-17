"""Каналы push-доставки результата."""

import pytest
import redis
import requests_mock

from core.config import settings
from core.models.contract import ParseResponse
from core.result_delivery import (
    CallbackDelivery,
    NullDelivery,
    RedisStreamsDelivery,
    ResultDeliveryError,
    get_delivery_provider,
)
from core.result_publisher import publish_result


class TestFactory:
    def test_default_is_polling(self, monkeypatch):
        monkeypatch.setattr(settings, "RESULT_DELIVERY_TYPE", "none")
        assert isinstance(get_delivery_provider(), NullDelivery)

    def test_callback(self, monkeypatch):
        monkeypatch.setattr(settings, "RESULT_DELIVERY_TYPE", "callback")
        monkeypatch.setattr(settings, "RESULT_CALLBACK_URL", "http://rag/result")
        assert isinstance(get_delivery_provider(), CallbackDelivery)

    def test_unknown_type_degrades_to_null(self, monkeypatch):
        monkeypatch.setattr(settings, "RESULT_DELIVERY_TYPE", "carrier-pigeon")
        assert isinstance(get_delivery_provider(), NullDelivery)

    def test_callback_without_url_raises(self, monkeypatch):
        monkeypatch.setattr(settings, "RESULT_DELIVERY_TYPE", "callback")
        monkeypatch.setattr(settings, "RESULT_CALLBACK_URL", None)
        with pytest.raises(ValueError):
            get_delivery_provider()


class TestRedisStreams:
    def test_send(self, fake_redis):
        provider = RedisStreamsDelivery(client=fake_redis)
        assert provider.send({"request_id": "req-1"}) is True
        assert fake_redis.xlen(provider.stream_name) == 1

    def test_retries_then_raises(self, mocker):
        client = mocker.MagicMock()
        client.xadd.side_effect = redis.exceptions.ConnectionError("нет связи")
        provider = RedisStreamsDelivery(client=client)
        provider.retry_delay = 0
        with pytest.raises(ResultDeliveryError):
            provider.send({"request_id": "req-1"})
        assert client.xadd.call_count == provider.max_retries


class TestCallback:
    @pytest.fixture
    def provider(self, monkeypatch):
        monkeypatch.setattr(settings, "RESULT_CALLBACK_URL", "http://rag/result")
        instance = CallbackDelivery()
        instance.retry_delay = 0
        return instance

    def test_success(self, provider):
        with requests_mock.Mocker() as m:
            m.post("http://rag/result", status_code=202)
            assert provider.send({"request_id": "req-1"}) is True

    def test_retries_on_5xx(self, provider):
        with requests_mock.Mocker() as m:
            m.post("http://rag/result", status_code=500)
            with pytest.raises(ResultDeliveryError):
                provider.send({"request_id": "req-1"})
            assert m.call_count == provider.max_retries


class TestPublisher:
    def test_failure_does_not_raise(self, monkeypatch):
        monkeypatch.setattr(settings, "RESULT_DELIVERY_TYPE", "callback")
        monkeypatch.setattr(settings, "RESULT_CALLBACK_URL", "http://rag/result")
        with requests_mock.Mocker() as m:
            m.post("http://rag/result", exc=OSError("сеть отвалилась"))
            response = ParseResponse(request_id="req-1", dialog_id="dlg-1", documents=[])
            assert publish_result(response) is False

    def test_null_delivery_reports_success(self, monkeypatch):
        monkeypatch.setattr(settings, "RESULT_DELIVERY_TYPE", "none")
        response = ParseResponse(request_id="req-1", dialog_id="dlg-1", documents=[])
        assert publish_result(response) is True
