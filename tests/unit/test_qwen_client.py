"""Клиент внешней модели Qwen3-VL: запрос, разбор ответа, повторы, отказы."""

import json

import pytest
import requests

from core.providers import qwen_client
from core.providers.qwen_client import QwenVisionClient, extract_json

pytest.importorskip("PIL")
from PIL import Image  # noqa: E402


def picture(size=(400, 300)):
    return Image.new("RGB", size, "white")


def answer(payload) -> dict:
    body = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
    return {"choices": [{"message": {"content": body}}]}


class FakeResponse:
    def __init__(self, payload=None, status=200, text=""):
        self._payload = payload
        self.status_code = status
        self.text = text

    def json(self):
        if self._payload is None:
            raise ValueError("не JSON")
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code), response=self)


class FakeSession:
    """Отдаёт заготовленные ответы по очереди и запоминает запросы."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests = []

    def post(self, url, json=None, headers=None, timeout=None):
        self.requests.append({"url": url, "json": json, "headers": headers or {}})
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    def get(self, url, timeout=None):
        self.requests.append({"url": url})
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


@pytest.fixture(autouse=True)
def no_sleep(mocker):
    mocker.patch.object(qwen_client.time, "sleep")


def client(session, **kwargs):
    kwargs.setdefault("endpoint", "http://gpu-host:8000")
    return QwenVisionClient(session=session, **kwargs)


class TestAvailability:
    def test_empty_endpoint_means_unavailable(self):
        assert not QwenVisionClient(endpoint="").available

    def test_unavailable_client_does_not_call_anything(self):
        session = FakeSession()
        assert QwenVisionClient(endpoint="", session=session).read_sheet(picture()) is None
        assert session.requests == []

    def test_v1_is_added_once(self):
        assert client(FakeSession())._base() == "http://gpu-host:8000/v1"
        assert client(FakeSession(), endpoint="http://gpu:8000/v1")._base() == "http://gpu:8000/v1"


class TestRequest:
    def test_image_goes_as_data_url_with_the_prompt(self):
        session = FakeSession(FakeResponse(answer({"annotations": []})))
        client(session).read_sheet(picture())

        sent = session.requests[0]
        assert sent["url"] == "http://gpu-host:8000/v1/chat/completions"
        content = sent["json"]["messages"][1]["content"]
        assert content[0]["image_url"]["url"].startswith("data:image/png;base64,")
        assert "ГОСТ" in sent["json"]["messages"][0]["content"]
        assert sent["json"]["temperature"] == 0.0
        assert sent["json"]["response_format"] == {"type": "json_object"}

    def test_api_key_goes_into_the_header(self):
        session = FakeSession(FakeResponse(answer({"annotations": []})))
        client(session, api_key="secret").read_sheet(picture())
        assert session.requests[0]["headers"]["Authorization"] == "Bearer secret"

    def test_no_key_means_no_header(self, mocker):
        # Ключ по умолчанию задан (vLLM в контуре поднят с --api-key local),
        # поэтому «ключа нет» здесь приходится изображать явно.
        mocker.patch.object(qwen_client.settings, "QWEN_API_KEY", None)
        session = FakeSession(FakeResponse(answer({"annotations": []})))
        client(session, api_key=None).read_sheet(picture())
        assert "Authorization" not in session.requests[0]["headers"]

    def test_large_sheet_is_downscaled(self, mocker):
        mocker.patch.object(qwen_client.settings, "QWEN_SHEET_MAX_SIDE", 512)
        session = FakeSession(FakeResponse(answer({"annotations": []})))
        client(session).read_sheet(picture((4000, 3000)))

        url = session.requests[0]["json"]["messages"][1]["content"][0]["image_url"]["url"]
        import base64
        import io

        decoded = Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1])))
        assert max(decoded.size) <= 512


class TestSheetReading:
    def test_annotations_are_returned(self):
        payload = {
            "sheet_type": "detail",
            "annotations": [
                {"category": "size", "value": "⌀44H7", "bbox": [0.1, 0.2, 0.2, 0.3]},
                "мусор, который не словарь",
            ],
        }
        reading = client(FakeSession(FakeResponse(answer(payload)))).read_sheet(picture())

        assert reading.sheet_type == "detail"
        assert [a["value"] for a in reading.annotations] == ["⌀44H7"]

    def test_json_in_markdown_fence_is_understood(self):
        fenced = "```json\n{\"annotations\": [{\"value\": \"20\"}]}\n```"
        reading = client(FakeSession(FakeResponse(answer(fenced)))).read_sheet(picture())
        assert reading.annotations[0]["value"] == "20"

    def test_answer_without_annotations_is_empty_not_broken(self):
        reading = client(FakeSession(FakeResponse(answer({"sheet_type": "scheme"})))).read_sheet(picture())
        assert reading.sheet_type == "scheme"
        assert reading.annotations == []

    def test_not_json_at_all_is_a_refusal(self):
        session = FakeSession(FakeResponse(answer("Извините, я не могу прочитать чертёж")))
        assert client(session).read_sheet(picture()) is None

    def test_hint_from_vision_reaches_the_prompt(self):
        session = FakeSession(FakeResponse(answer({"annotations": []})))
        client(session).read_sheet(picture(), hint="штамп в правом нижнем углу")
        prompt = session.requests[0]["json"]["messages"][1]["content"][1]["text"]
        assert "штамп в правом нижнем углу" in prompt


class TestRegions:
    def test_title_block_fields_and_rows(self):
        payload = {
            "fields": {"Наименование": "Вилка", "Материал": "Сталь 45", "Масштаб": ""},
            "rows": [["Разраб.", "Иванов"], ["Пров.", "Петров"]],
        }
        block = client(FakeSession(FakeResponse(answer(payload)))).read_title_block(picture())

        assert block["fields"] == {"Наименование": "Вилка", "Материал": "Сталь 45"}
        assert block["rows"][1] == ["Пров.", "Петров"]

    def test_table_rows_are_strings(self):
        payload = {"headers": ["Поз.", "Кол."], "rows": [[1, 2], {"a": 3, "b": None}]}
        table = client(FakeSession(FakeResponse(answer(payload)))).read_table(picture())

        assert table["headers"] == ["Поз.", "Кол."]
        assert table["rows"] == [["1", "2"], ["3", ""]]

    def test_broken_answer_gives_none(self):
        session = FakeSession(FakeResponse(answer("не таблица")))
        assert client(session).read_table(picture()) is None


class TestFailures:
    def test_timeout_is_retried_then_given_up(self):
        session = FakeSession(
            requests.Timeout("вышло время"),
            requests.Timeout("вышло время"),
            requests.Timeout("вышло время"),
        )
        assert client(session).read_sheet(picture()) is None
        assert len(session.requests) == 3           # первая попытка плюс два повтора

    def test_server_error_is_retried_and_then_succeeds(self):
        session = FakeSession(
            FakeResponse(status=503, text="model is loading"),
            FakeResponse(answer({"annotations": [{"value": "20"}]})),
        )
        reading = client(session).read_sheet(picture())
        assert reading.annotations[0]["value"] == "20"
        assert len(session.requests) == 2

    def test_client_error_is_not_retried(self):
        session = FakeSession(FakeResponse(status=400, text="bad request"))
        assert client(session).read_sheet(picture()) is None
        assert len(session.requests) == 1

    def test_answer_without_choices(self):
        assert client(FakeSession(FakeResponse({"choices": []}))).read_sheet(picture()) is None

    def test_health_says_no_when_host_is_silent(self):
        session = FakeSession(requests.ConnectionError("нет маршрута"))
        assert client(session).health() is False

    def test_health_says_yes(self):
        assert client(FakeSession(FakeResponse({"data": []}))).health() is True


class TestExtractJson:
    @pytest.mark.parametrize("text,expected", [
        ('{"a": 1}', {"a": 1}),
        ('```json\n{"a": 1}\n```', {"a": 1}),
        ('Вот результат: {"a": {"b": 2}} — всё.', {"a": {"b": 2}}),
        ('{"value": "фигурная } внутри строки"}', {"value": "фигурная } внутри строки"}),
        ("[1, 2, 3]", None),
        ("", None),
        ("совсем не json", None),
    ])
    def test_cases(self, text, expected):
        assert extract_json(text) == expected
