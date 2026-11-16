"""
LADDER_PARALLEL_LAYOUT: на скане запрос к MinerU идёт одновременно с
уровнем 4 (Tesseract), а не после него. Результат тот же, время — максимум
из двух, ошибка запроса — ошибка уровня 5, брошенный запрос не держит задачу
и не объявляет MinerU недоступным.
"""

import threading
import time

import pytest
import requests
import requests_mock
from celery.exceptions import SoftTimeLimitExceeded

from core import telemetry
from core.config import settings
from core.ladder.base import Strategy
from core.ladder.context import SOURCE_RASTER, SOURCE_VECTOR_TEXT, Classification, DocumentContext
from core.ladder.router import LadderRouter
from core.ladder.strategies.recognition import LayoutStrategy
from core.models.parse_result import ParsedBlock
from core.providers.mineru_parser import MinerUParserProvider
from tests.unit.test_mineru_parser import PARSE_URL, middle_json_response

DELAY = 0.4


class SlowOcr(Strategy):
    """Уровень 4: читает «страницы» `DELAY` секунд."""

    level = 4
    name = "slow_ocr"
    method = "tesseract"

    def __init__(self, raises=None):
        self.raises = raises

    def applicable(self, context):
        return True

    def run(self, context):
        time.sleep(DELAY)
        if self.raises is not None:
            raise self.raises
        block = ParsedBlock(type="text", text="Первая строка абзаца. Вторая строка абзаца.",
                            page=1, confidence=0.8, method=self.method)
        return self.build_result([block], source_kind="scanned_pdf", page_count=1, is_fallback=True)


@pytest.fixture(autouse=True)
def available():
    MinerUParserProvider.reset_availability()
    yield
    MinerUParserProvider.reset_availability()


@pytest.fixture
def layout(temp_storage, mocker):
    temp_storage.write_file("documents/test.pdf", b"%PDF-1.4 fake")
    provider = MinerUParserProvider(temp_storage)
    # OCR-слой скана — то же, что делает уровень 4; здесь он не нужен.
    mocker.patch.object(provider, "_ocr_layer", return_value=(None, None))
    return LayoutStrategy(provider=provider)


def scan(source=SOURCE_RASTER):
    ctx = DocumentContext(uri="documents/test.pdf", file_type="pdf", metadata={}, data=b"%PDF-1.4 fake")
    ctx._classification = Classification(source=source, genre="unknown")
    ctx._source_count_loaded, ctx._source_page_count = True, 1
    return ctx


def slow_mineru(request, context):
    time.sleep(DELAY)
    return middle_json_response()


def run(monkeypatch, flag, layout, ocr=None, source=SOURCE_RASTER):
    monkeypatch.setattr(settings, "LADDER_PARALLEL_LAYOUT", flag)
    started = time.perf_counter()
    with telemetry.collect() as timings:
        result = LadderRouter([ocr or SlowOcr(), layout], threshold=0.75).parse(scan(source))
    return result, time.perf_counter() - started, timings


def blocks(result):
    return [(b.type, b.text, b.page, b.bbox, b.table_data) for b in result.blocks]


class TestParallelLayout:
    def test_wall_time_is_max_not_sum_and_result_is_the_same(self, monkeypatch, layout):
        with requests_mock.Mocker() as m:
            m.post(PARSE_URL, json=slow_mineru)
            off, off_s, _ = run(monkeypatch, False, layout)
            on, on_s, timings = run(monkeypatch, True, layout)
        assert off_s >= 2 * DELAY
        assert on_s < 2 * DELAY - 0.1
        assert blocks(on) == blocks(off)
        assert on.ladder["chosen_level"] == off.ladder["chosen_level"] == 5
        assert [a["level"] for a in on.ladder["attempts"]] == [4, 5]
        assert timings["parse.level5_http_ms"] >= DELAY * 1000 * 0.9
        assert timings["parse.level5_post_ms"] < DELAY * 1000

    def test_request_error_is_the_level5_error(self, monkeypatch, layout):
        with requests_mock.Mocker() as m:
            m.post(PARSE_URL, status_code=400, text="bad request")
            off, _, _ = run(monkeypatch, False, layout)
            on, _, _ = run(monkeypatch, True, layout)
        errors = [[a.get("error") for a in r.ladder["attempts"] if a["level"] == 5] for r in (off, on)]
        assert errors[0] == errors[1] and "отклонил" in errors[1][0]
        assert on.ladder["chosen_level"] == 4

    def test_level4_failure_still_consumes_the_request(self, monkeypatch, layout):
        with requests_mock.Mocker() as m:
            m.post(PARSE_URL, json=slow_mineru)
            result, _, _ = run(monkeypatch, True, layout, ocr=SlowOcr(raises=ValueError("нет текста")))
        assert result.ladder["chosen_level"] == 5

    def test_cancelled_task_abandons_the_request(self, monkeypatch, layout):
        handles = []
        original = LayoutStrategy.prefetch

        def recording(self, context):
            handle = original(self, context)
            handles.append(handle)
            return handle

        monkeypatch.setattr(LayoutStrategy, "prefetch", recording)
        release = threading.Event()

        def hanging(request, context):
            release.wait(5)
            raise requests.ConnectionError("сессия закрыта")

        with requests_mock.Mocker() as m:
            m.post(PARSE_URL, json=hanging)
            with pytest.raises(SoftTimeLimitExceeded):
                run(monkeypatch, True, layout, ocr=SlowOcr(raises=SoftTimeLimitExceeded()))
            [handle] = handles
            assert handle.abandoned and not handle.consumed
            release.set()
            with pytest.raises(Exception):
                handle._future.result(timeout=5)
        # Обрыв брошенного запроса — не отказ сервиса.
        assert MinerUParserProvider.is_available()
        assert not [t for t in threading.enumerate() if t.name.startswith("mineru-prefetch") and t.is_alive()]

    def test_flag_off_starts_no_thread(self, monkeypatch, layout):
        monkeypatch.setattr(LayoutStrategy, "prefetch",
                            lambda self, ctx: pytest.fail("запрос заранее без флага"))
        with requests_mock.Mocker() as m:
            m.post(PARSE_URL, json=middle_json_response())
            result, _, _ = run(monkeypatch, False, layout)
        assert result.ladder["chosen_level"] == 5

    def test_not_a_scan_is_not_prefetched(self, monkeypatch, layout):
        monkeypatch.setattr(LayoutStrategy, "prefetch",
                            lambda self, ctx: pytest.fail("запрос заранее не на скане"))
        with requests_mock.Mocker() as m:
            m.post(PARSE_URL, json=middle_json_response())
            run(monkeypatch, True, layout, source=SOURCE_VECTOR_TEXT)

    def test_unavailable_mineru_is_not_prefetched(self, monkeypatch, layout):
        MinerUParserProvider.mark_unavailable()
        monkeypatch.setattr(LayoutStrategy, "prefetch",
                            lambda self, ctx: pytest.fail("запрос заранее к лежащему MinerU"))
        result, _, _ = run(monkeypatch, True, layout)
        assert result.ladder["chosen_level"] == 4
