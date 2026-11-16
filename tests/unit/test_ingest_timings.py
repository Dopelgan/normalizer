"""
Тайминги разбора по документу (`extra_data.timings_ms`): уровни 4 и 5,
запрос к MinerU отдельно от работы после него, вызовы модели листов.

Повод: скорость конвейера мерилась разбором логов воркера, и сказать, какую
долю `parse.document` съедает Tesseract, а какую — ожидание MinerU, было
нельзя. Каждый новый ключ проверяется отдельно.
"""

import threading
import time

import pytest
import requests
import requests_mock

from core import telemetry
from core.config import settings
from core.ladder.base import Strategy
from core.ladder.context import DocumentContext
from core.ladder.router import LadderRouter
from core.models.parse_result import ParsedBlock
from core.providers.mineru_parser import MINERU_HTTP_MS, MinerUParserProvider


def context(file_type: str = "png") -> DocumentContext:
    return DocumentContext(uri="documents/x", file_type=file_type, metadata={}, data=b"x")


class TimedStrategy(Strategy):
    """Уровень с заданной длительностью и, для уровня 5, долей «запроса»."""

    def __init__(self, level, seconds=0.0, http_ms=0.0, fail=False, fallback=False):
        self.level = level
        self.name = f"timed-{level}"
        self.method = f"timed_{level}"
        self.exhaustive = True
        self._seconds = seconds
        self._http_ms = http_ms
        self._fail = fail
        self._fallback = fallback

    def applicable(self, context):
        return True

    def run(self, context):
        time.sleep(self._seconds)
        if self._http_ms:
            telemetry.add(MINERU_HTTP_MS, self._http_ms)
        if self._fail:
            raise ValueError("уровень не справился")
        block = ParsedBlock(type="text", text="Достаточно длинный текст фрагмента документа.",
                            page=1, confidence=1.0, method=self.method)
        return self.build_result([block], source_kind="test", page_count=1,
                                 is_fallback=self._fallback)


class TestLadderLevelTimings:
    def test_level4_time_is_recorded(self):
        with telemetry.collect() as timings:
            LadderRouter([TimedStrategy(4, 0.05, fallback=True)], threshold=0.75).parse(context())
        assert timings["parse.level4_ms"] >= 45

    def test_level5_is_split_into_request_and_the_rest(self):
        with telemetry.collect() as timings:
            LadderRouter([TimedStrategy(5, 0.08, http_ms=30.0)], threshold=0.75).parse(context())
        assert timings["parse.level5_http_ms"] == pytest.approx(30.0)
        assert timings["parse.level5_post_ms"] >= 45
        assert timings["parse.level5_post_ms"] < 80 + 50

    def test_level5_request_of_level6_is_not_counted_as_level5(self):
        """Уровень 6 тоже ходит в MinerU — его запрос в level5_http_ms не попадает."""
        with telemetry.collect() as timings:
            LadderRouter(
                [TimedStrategy(5, http_ms=10.0, fail=True), TimedStrategy(6, http_ms=500.0)],
                threshold=0.75,
            ).parse(context())
        assert timings["parse.level5_http_ms"] == pytest.approx(10.0)
        assert timings[MINERU_HTTP_MS] == pytest.approx(510.0)

    def test_failed_level_is_timed_too(self):
        with telemetry.collect() as timings:
            LadderRouter(
                [TimedStrategy(4, 0.03, fail=True), TimedStrategy(5)], threshold=0.75
            ).parse(context())
        assert timings["parse.level4_ms"] >= 25

    def test_other_levels_add_no_keys(self):
        with telemetry.collect() as timings:
            LadderRouter([TimedStrategy(2)], threshold=0.75).parse(context("txt"))
        assert not any(key.startswith("parse.level") for key in timings)


class TestMineruRequestTime:
    def setup_method(self):
        MinerUParserProvider.reset_availability()

    teardown_method = setup_method

    def test_request_time_is_recorded_on_success_and_on_failure(self, temp_storage):
        temp_storage.write_file("documents/test.pdf", b"%PDF-1.4 fake")
        provider = MinerUParserProvider(temp_storage)
        url = f"{settings.MINERU_ENDPOINT.rstrip('/')}/file_parse"
        with telemetry.collect() as timings, requests_mock.Mocker() as m:
            m.post(url, exc=requests.exceptions.ConnectTimeout("нет связи"))
            with pytest.raises(Exception):
                provider._parse_with_mineru("documents/test.pdf", "pdf", {}, b"%PDF-1.4", 1)
        assert MINERU_HTTP_MS in timings


class FakeResponse:
    def __init__(self, payload=None, status=200):
        self._payload = payload
        self.status_code = status
        self.text = ""

    def json(self):
        return self._payload


class FakeSession:
    def __init__(self, *responses):
        self.responses = list(responses)

    def post(self, url, json=None, headers=None, timeout=None):
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class TestVlmCallCounts:
    @pytest.fixture(autouse=True)
    def no_sleep(self, mocker):
        from core.providers import qwen_client

        mocker.patch.object(qwen_client.time, "sleep")

    def client(self, session):
        from core.providers.qwen_client import QwenVisionClient

        client = QwenVisionClient(endpoint="http://gpu-host:8000", session=session)
        client.retries = 1
        return client

    def test_successful_call_is_counted_with_its_time(self):
        ok = FakeResponse({"choices": [{"message": {"content": "{}"}}]})
        with telemetry.collect() as timings:
            assert self.client(FakeSession(ok))._post({}) is not None
        assert timings["parse.vlm_calls"] == 1
        assert "parse.vlm_call_ms_sum" in timings
        assert "parse.vlm_failed_calls" not in timings

    def test_retries_are_one_call_and_a_final_failure_is_counted(self):
        broken = requests.ConnectionError("обрыв")
        with telemetry.collect() as timings:
            assert self.client(FakeSession(broken, broken))._post({}) is None
        assert timings["parse.vlm_calls"] == 1
        assert timings["parse.vlm_failed_calls"] == 1


class TestCollectorIsThreadSafe:
    def test_concurrent_additions_are_not_lost(self):
        """Потоки листов и плиток пишут в один словарь (копия контекста)."""
        import contextvars

        with telemetry.collect() as timings:
            def worker():
                for _ in range(2000):
                    telemetry.add("parse.vlm_calls", 1)

            threads = [threading.Thread(target=contextvars.copy_context().run, args=(worker,))
                       for _ in range(8)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
        assert timings["parse.vlm_calls"] == 16000
