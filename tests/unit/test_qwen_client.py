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
        """
        Строка в списке надписей — не мусор, а прочтение без места: на
        плотном листе модель бросает обёртку объекта и перечисляет значения
        подряд. Раньше такие строки выбрасывались здесь (`isinstance(a,
        dict)`), и на настоящем листе-паспорте так пропали 83 прочтения из
        108. Читается ли строка как надпись чертежа, решают правила разбора
        дальше (`drawing_text.looks_like_text`), а не вид элемента JSON.
        """
        payload = {
            "sheet_type": "detail",
            "annotations": [
                {"category": "size", "value": "⌀44H7", "bbox": [0.1, 0.2, 0.2, 0.3]},
                "2.13 Степень защиты по ГОСТ 14254-2015",
            ],
        }
        reading = client(FakeSession(FakeResponse(answer(payload)))).read_sheet(picture())

        assert reading.sheet_type == "detail"
        assert [a["value"] for a in reading.annotations] == [
            "⌀44H7", "2.13 Степень защиты по ГОСТ 14254-2015",
        ]
        assert reading.lost == 0

    def test_an_element_that_is_no_reading_is_counted_not_dropped(self):
        """Доля из bbox отдельным элементом в надпись не превращается."""
        payload = {"annotations": [
            {"category": "size", "value": "⌀44H7", "bbox": [0.1, 0.2, 0.2, 0.3]},
            0.52, None,
        ]}
        reading = client(FakeSession(FakeResponse(answer(payload)))).read_sheet(picture())

        assert [a["value"] for a in reading.annotations] == ["⌀44H7"]
        assert reading.lost == 2 and reading.complete is False

    def test_json_in_markdown_fence_is_understood(self):
        fenced = "```json\n{\"annotations\": [{\"value\": \"20\"}]}\n```"
        reading = client(FakeSession(FakeResponse(answer(fenced)))).read_sheet(picture())
        assert reading.annotations[0]["value"] == "20"

    def test_bbox_in_thousandths_becomes_fractions(self):
        """
        Запрос просит доли, а модель отвечает тысячными: на ГЧ 302182.001 все
        57 надписей пришли так, и поля легли рамкой всего листа.
        """
        payload = {"annotations": [
            {"category": "title_block", "value": "ТАДУ 302182.001 ГЧ", "bbox": [539, 798, 853, 828]},
            {"category": "size", "value": "⌀44H7", "bbox": [0.1, 0.2, 0.2, 0.3]},
            {"category": "size", "value": "20", "bbox": [1200, 300, 1300, 340]},
        ]}
        reading = client(FakeSession(FakeResponse(answer(payload)))).read_sheet(picture())

        assert [a["bbox"] for a in reading.annotations] == [
            [0.539, 0.798, 0.853, 0.828],
            [0.1, 0.2, 0.2, 0.3],        # доли — как были
            [1200, 300, 1300, 340],      # больше 1000 — пиксели, не переводим
        ]

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


class TestResponseCache:
    """Кэш ответов модели (VLM_RESPONSE_CACHE): что попадает в ключ."""

    @pytest.fixture
    def cache(self, mocker, tmp_path):
        mocker.patch.object(qwen_client.settings, "VLM_RESPONSE_CACHE", True)
        mocker.patch.object(
            qwen_client.settings, "VLM_RESPONSE_CACHE_DIR", str(tmp_path)
        )
        return tmp_path

    def test_second_same_sheet_does_not_reach_the_model(self, cache):
        session = FakeSession(FakeResponse(answer({"annotations": [{"value": "20"}]})))
        sheet = picture()
        assert client(session).read_sheet(sheet).annotations[0]["value"] == "20"
        # Второй клиент и пустая сессия: ответ может прийти только из кэша,
        # а лишний запрос уронил бы FakeSession на пустом списке.
        again = client(FakeSession()).read_sheet(sheet)
        assert again.annotations[0]["value"] == "20"
        assert len(session.requests) == 1

    def test_cache_is_off_by_default(self, mocker, tmp_path):
        mocker.patch.object(
            qwen_client.settings, "VLM_RESPONSE_CACHE_DIR", str(tmp_path)
        )
        session = FakeSession(
            FakeResponse(answer({"annotations": []})),
            FakeResponse(answer({"annotations": []})),
        )
        sheet = picture()
        client(session).read_sheet(sheet)
        client(session).read_sheet(sheet)
        assert len(session.requests) == 2
        assert list(tmp_path.iterdir()) == []

    def test_other_sheet_is_another_key(self, cache):
        session = FakeSession(
            FakeResponse(answer({"annotations": [{"value": "20"}]})),
            FakeResponse(answer({"annotations": [{"value": "30"}]})),
        )
        own = client(session)
        assert own.read_sheet(picture((400, 300))).annotations[0]["value"] == "20"
        assert own.read_sheet(picture((401, 300))).annotations[0]["value"] == "30"
        assert len(session.requests) == 2

    def test_smaller_image_is_another_key(self, mocker, cache):
        session = FakeSession(
            FakeResponse(answer({"annotations": []})),
            FakeResponse(answer({"annotations": []})),
        )
        sheet = picture((4000, 3000))
        client(session).read_sheet(sheet)
        mocker.patch.object(qwen_client.settings, "QWEN_SHEET_MAX_SIDE", 512)
        client(session).read_sheet(sheet)
        assert len(session.requests) == 2

    def test_other_prompt_is_another_key(self, mocker, cache):
        """Флаг, который меняет текст запроса, обязан промахнуться мимо кэша."""
        session = FakeSession(
            FakeResponse(answer({"annotations": []})),
            FakeResponse(answer({"annotations": []})),
        )
        sheet = picture()
        client(session).read_sheet(sheet)
        mocker.patch.object(
            qwen_client.settings, "DRAWING_DIMENSION_FIDELITY", True
        )
        client(session).read_sheet(sheet)
        assert len(session.requests) == 2
        first, second = (r["json"]["messages"][1]["content"][1]["text"]
                         for r in session.requests)
        assert first != second

    def test_other_model_is_another_key(self, cache):
        session = FakeSession(
            FakeResponse(answer({"annotations": []})),
            FakeResponse(answer({"annotations": []})),
        )
        sheet = picture()
        client(session, model="Qwen/A").read_sheet(sheet)
        client(session, model="Qwen/B").read_sheet(sheet)
        assert len(session.requests) == 2

    def test_other_token_ceiling_is_another_key(self, mocker, cache):
        session = FakeSession(
            FakeResponse(answer({"annotations": []})),
            FakeResponse(answer({"annotations": []})),
        )
        mocker.patch.object(qwen_client.settings, "DRAWING_SHEET_HIGH_TOKENS", True)
        mocker.patch.object(qwen_client.settings, "DRAWING_SHEET_MAX_TOKENS", 8192)
        sheet = picture()
        client(session).read_sheet(sheet)
        mocker.patch.object(qwen_client.settings, "DRAWING_SHEET_MAX_TOKENS", 12288)
        client(session).read_sheet(sheet)
        assert len(session.requests) == 2

    def test_failed_call_is_not_cached(self, cache):
        """Кэш не продлевает жизнь отказу: модель спросят заново."""
        session = FakeSession(
            FakeResponse(status=400, text="bad request"),
            FakeResponse(answer({"annotations": [{"value": "20"}]})),
        )
        sheet = picture()
        assert client(session).read_sheet(sheet) is None
        assert client(session).read_sheet(sheet).annotations[0]["value"] == "20"
        assert len(session.requests) == 2

    def test_truncated_answer_stays_truncated_on_replay(self, mocker, cache):
        """`finish_reason` хранится вместе с ответом: дочитывание частями
        должно вести себя на кэше ровно так же, как на свежем ответе."""
        mocker.patch.object(qwen_client.settings, "DRAWING_SHEET_TILES", False)
        cut = {"choices": [{"message": {"content": '{"annotations": [{"value": "20"}]'},
                            "finish_reason": "length"}]}
        session = FakeSession(FakeResponse(cut))
        sheet = picture()
        assert client(session).read_sheet(sheet).cut is True
        assert client(FakeSession()).read_sheet(sheet).cut is True
        assert len(session.requests) == 1

    def test_other_endpoint_is_another_key(self, cache):
        """Одно имя модели на другом сервере — другие веса: ответ одного
        сервера не выдаётся за ответ другого."""
        session = FakeSession(
            FakeResponse(answer({"annotations": [{"value": "20"}]})),
            FakeResponse(answer({"annotations": [{"value": "30"}]})),
        )
        sheet = picture()
        own = client(session, endpoint="http://gpu-a:8000", model="qwen")
        other = client(session, endpoint="http://gpu-b:8000", model="qwen")
        assert own.read_sheet(sheet).annotations[0]["value"] == "20"
        assert other.read_sheet(sheet).annotations[0]["value"] == "30"
        assert len(session.requests) == 2

    def test_overlapping_writers_of_one_key_do_not_lose_the_write(self, mocker, cache):
        """Два писателя одного ключа в одном процессе (плитки идут потоками):
        у каждого свой временный файл, ни одна запись не теряется и мусора
        не остаётся."""
        key = "ab" + "0" * 62
        real_replace = qwen_client.os.replace
        warnings = mocker.spy(qwen_client.logger, "warning")
        nested = []

        def overlapping_replace(src, dst):
            # Второй писатель целиком успевает между записью и переименованием первого.
            if not nested:
                nested.append(True)
                qwen_client.cache_put(key, ("второй", None))
            return real_replace(src, dst)

        mocker.patch.object(qwen_client.os, "replace", side_effect=overlapping_replace)
        qwen_client.cache_put(key, ("первый", "stop"))

        assert warnings.call_count == 0
        assert qwen_client.cache_get(key) == ("первый", "stop")
        assert [p.name for p in cache.rglob("*") if p.is_file()] == [f"{key}.json"]

    def test_identical_body_gives_identical_key(self):
        """Одно и то же тело запроса — один ключ, независимо от порядка полей
        и от того, что это другой объект."""
        def body(order):
            fields = {
                "model": "qwen", "temperature": 0.1, "max_tokens": 512,
                "response_format": {"type": "json_object"},
                "messages": [{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": ""}},
                    {"type": "text", "text": "Перепиши лист"},
                ]}],
            }
            return {name: fields[name] for name in order}

        names = ["model", "temperature", "max_tokens", "response_format", "messages"]
        first = qwen_client.cache_key("http://gpu:8000/v1", body(names), "data:x")
        second = qwen_client.cache_key("http://gpu:8000/v1", body(names[::-1]), "data:x")
        assert first == second
        changed = body(names)
        changed["max_tokens"] = 513
        assert qwen_client.cache_key("http://gpu:8000/v1", changed, "data:x") != first

    def test_v1_key_is_computed_as_the_old_code_did(self, mocker):
        """Ключ прежней версии сверен с выводом старого cache_key на том же запросе."""
        mocker.patch.object(qwen_client.settings, "QWEN_TEMPERATURE", 0.1)
        key = qwen_client.cache_key_v1(
            "Qwen/Qwen3.8-27B-FP8", 8192, "Перепиши весь текст листа.",
            "data:image/png;base64,iVBORw0KGgo=",
        )
        assert key == "1a87ef451fb7f384cf7927394a32f052de55322320f00325350d6888313ea24c"

    def test_v1_hit_is_migrated_once_then_served_from_v2(self, mocker, cache):
        mocker.patch.object(qwen_client.settings, "VLM_CACHE_V1_FALLBACK", True)
        v1_keys = mocker.spy(qwen_client, "cache_key_v1")
        sheet = picture()
        # Первый вызов: ключ v1 этого запроса, модель спрошена, файлы убраны.
        client(FakeSession(FakeResponse(answer({"annotations": []})))).read_sheet(sheet)
        old_key = v1_keys.spy_return
        for path in cache.rglob("*.json"):
            path.unlink()
        # Запись старого кода лежит под ключом v1.
        old_path = cache / old_key[:2] / f"{old_key}.json"
        old_path.parent.mkdir(parents=True, exist_ok=True)
        old_path.write_text(
            '{"content": "{\\"annotations\\": [{\\"value\\": \\"20\\"}]}", "finish": "stop"}',
            encoding="utf-8",
        )
        migrated = client(FakeSession()).read_sheet(sheet)
        assert migrated.annotations[0]["value"] == "20"
        assert old_path.exists()
        # Без старого файла ответ берётся из v2, модель не спрашивается.
        old_path.unlink()
        calls = v1_keys.call_count
        again = client(FakeSession()).read_sheet(sheet)
        assert again.annotations[0]["value"] == "20"
        assert v1_keys.call_count == calls

    def test_without_v1_file_the_model_is_asked(self, mocker, cache):
        mocker.patch.object(qwen_client.settings, "VLM_CACHE_V1_FALLBACK", True)
        session = FakeSession(FakeResponse(answer({"annotations": [{"value": "30"}]})))
        assert client(session).read_sheet(picture()).annotations[0]["value"] == "30"
        assert len(session.requests) == 1

    def test_broken_cache_file_is_a_miss(self, cache):
        session = FakeSession(
            FakeResponse(answer({"annotations": []})),
            FakeResponse(answer({"annotations": []})),
        )
        sheet = picture()
        client(session).read_sheet(sheet)
        for path in cache.rglob("*.json"):
            path.write_text("не json", encoding="utf-8")
        client(session).read_sheet(sheet)
        assert len(session.requests) == 2


class TestDimensionLinksAnswer:
    """Ответ без списка `links` не замерзает в кэше ответов модели."""

    @pytest.mark.parametrize("content,ok", [
        ("{}", False), ('{"links": "нет"}', False), ("не JSON", False),
        ('{"links": []}', True), ('{"links": [{"id": 1}]}', True),
    ])
    def test_only_answers_with_links_are_cached(self, content, ok):
        assert qwen_client.links_answer_accepted(content) is ok

    def test_empty_object_is_not_cached_and_is_asked_again(self, mocker, tmp_path):
        mocker.patch.object(qwen_client.settings, "VLM_RESPONSE_CACHE", True)
        mocker.patch.object(qwen_client.settings, "VLM_RESPONSE_CACHE_DIR", str(tmp_path))
        session = FakeSession(FakeResponse(answer({})), FakeResponse(answer({"links": []})),
                              FakeResponse(answer({"links": [{"id": 9}]})))
        model = client(session)

        assert model.read_dimension_links(picture(), "связи", 2) == {}
        assert model.read_dimension_links(picture(), "связи", 2) == {"links": []}
        # Принятый ответ взят из кэша: третьего вызова нет.
        assert model.read_dimension_links(picture(), "связи", 2) == {"links": []}
        assert len(session.requests) == 2


class TestStableCalls:
    """Карточка E (DRAWING_VLM_STABLE): рассуждение выключено, «{}» переспрашивается."""

    def test_thinking_off_only_with_flag(self, mocker):
        session = FakeSession(FakeResponse(answer({"fields": {"Масса": "1"}})),
                              FakeResponse(answer({"fields": {"Масса": "1"}})))
        mocker.patch.object(qwen_client.settings, "DRAWING_VLM_STABLE", False)
        client(session).read_title_block_fields(picture())
        assert "chat_template_kwargs" not in session.requests[0]["json"]
        mocker.patch.object(qwen_client.settings, "DRAWING_VLM_STABLE", True)
        client(session).read_title_block_fields(picture((310, 200)))
        assert session.requests[1]["json"]["chat_template_kwargs"] == {"enable_thinking": False}

    def test_empty_object_is_asked_once_more_and_logged(self, mocker, tmp_path):
        log = tmp_path / "rejected.jsonl"
        mocker.patch.object(qwen_client.settings, "DRAWING_VLM_STABLE", True)
        mocker.patch.object(qwen_client.settings, "DRAWING_VLM_REJECTED_LOG", str(log))
        session = FakeSession(FakeResponse(answer("{}")), FakeResponse(answer({"fields": {"Масса": "3,74"}})))
        parsed = client(session).read_title_block_fields(picture())
        assert len(session.requests) == 2
        assert parsed["fields"]["mass"] == "3,74"
        rows = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
        assert rows and rows[0]["head"] == "{}"

    def test_empty_object_without_flag_is_taken_as_is(self, mocker):
        mocker.patch.object(qwen_client.settings, "DRAWING_VLM_STABLE", False)
        session = FakeSession(FakeResponse(answer("{}")))
        client(session).read_title_block_fields(picture())
        assert len(session.requests) == 1


class TestContentPass:
    """Карточка B (DRAWING_SHEET_CONTENT_PASS): содержание без рамок, место — от прохода с рамками."""

    def test_items_and_codes(self):
        items = qwen_client.content_items([["s", "15±0,1"], ["n", "1. * Размеры для справок."], ["x", "п.4"],
                                           ["tb", ""], "мусор", ["s"]])
        assert items == [{"category": "size", "value": "15±0,1"},
                         {"category": "note", "value": "1. * Размеры для справок."},
                         {"category": "note", "value": "п.4"}]

    def test_repeats_take_boxes_in_order_and_located_rest_is_kept(self):
        content = qwen_client.SheetReading(sheet_type="detail", annotations=[
            {"category": "size", "value": "15±0,1"}, {"category": "size", "value": "15±0,1"},
            {"category": "size", "value": "15 ± 0,1"}, {"category": "note", "value": "Острые кромки притупить"}])
        located = qwen_client.SheetReading(sheet_type="unknown", annotations=[
            {"category": "size", "value": "15±0,1", "bbox": [0.1, 0.1, 0.2, 0.2]},
            {"category": "size", "value": "15±0,1", "bbox": [0.5, 0.5, 0.6, 0.6]},
            {"category": "radius", "value": "R2", "bbox": [0.7, 0.7, 0.72, 0.72]}])
        merged = qwen_client.merge_content(content, located)
        assert merged.sheet_type == "detail"
        boxes = [a.get("bbox") for a in merged.annotations]
        assert boxes[:4] == [[0.1, 0.1, 0.2, 0.2], [0.5, 0.5, 0.6, 0.6], None, None]
        assert merged.annotations[-1]["value"] == "R2"

    def test_schema_requires_both_keys(self):
        schema = qwen_client.content_schema()["json_schema"]["schema"]
        assert schema["required"] == ["sheet_type", "annotations"]

    def test_flag_on_makes_content_call_at_its_own_side(self, mocker):
        mocker.patch.object(qwen_client.settings, "DRAWING_SHEET_CONTENT_PASS", True)
        mocker.patch.object(qwen_client.settings, "DRAWING_SHEET_CONTENT_MAX_SIDE", 700)
        located = answer({"sheet_type": "detail", "annotations": [
            {"category": "size", "value": "⌀315", "bbox": [0.4, 0.3, 0.45, 0.4]}]})
        content = answer({"sheet_type": "detail", "annotations": [["s", "⌀315"], ["s", "18*"]]})
        session = FakeSession(FakeResponse(located), FakeResponse(content))
        reading = client(session).read_sheet(picture((1400, 1000)))
        assert len(session.requests) == 2
        assert session.requests[1]["json"]["response_format"]["type"] == "json_schema"
        values = [(a["value"], a.get("bbox")) for a in reading.annotations]
        assert values == [("⌀315", [0.4, 0.3, 0.45, 0.4]), ("18*", None)]

    def test_flag_off_is_one_call(self, mocker):
        mocker.patch.object(qwen_client.settings, "DRAWING_SHEET_CONTENT_PASS", False)
        session = FakeSession(FakeResponse(answer({"sheet_type": "detail", "annotations": []})))
        client(session).read_sheet(picture())
        assert len(session.requests) == 1


@pytest.mark.parametrize("flag", [True, False])
@pytest.mark.parametrize("compact", [True, False])
def test_keep_repeats_rule_in_sheet_prompt(mocker, flag, compact):
    """Карточка D: правило «не повторяй» заменяется правилом «столько раз, сколько стоит»."""
    mocker.patch.object(qwen_client.settings, "DRAWING_KEEP_REPEATS", flag)
    mocker.patch.object(qwen_client.settings, "DRAWING_SHEET_COMPACT_JSON", compact)
    prompt = client(FakeSession())._sheet_prompt("")
    assert ("столько раз, сколько она стоит" in prompt) is flag
    assert ("Не повторяй одну надпись дважды" in prompt) is (not flag)
