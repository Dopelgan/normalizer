"""
Отказ сервиса разбора ставит документ в очередь, а не деградирует его
(PARSE_REQUEUE_ON_OUTAGE).

16.11 MinerU убили по памяти, и 21 скан ушёл в индекс плоским OCR без таблиц;
vLLM отказал 38 вызовам. С флагом растровый документ при лежащем MinerU и
документ, которому модель не ответила ни разу, разбираются позже. Таймаут
чтения — свойство документа и остаётся сегодняшней деградацией. Срок
ожидания вышел — разбор как сегодня. Пока документ ждёт, RAG видит его «в
обработке».
"""

import io
import time

import pytest
import requests

from core import outage
from core.config import settings
from core.ladder.context import DocumentContext
from core.ladder.router import LadderRouter
from core.ladder.strategies.recognition import LayoutStrategy
from core.models.parse_result import ParsedBlock, ParseResult
from core.providers.document_parser import ParserUnavailable
from core.providers.mineru_parser import ParserTimedOut
from core.providers.mineru_parser import MinerUParserProvider
from core.providers.qwen_client import QwenVisionClient
from core.repositories import DocumentRepository
from core.result_aggregator import get_request_state, init_request_state
from ingest import tasks
from ingest.tasks import generate_doc_id, process_document
from tests.unit.test_tasks import pipeline, task_db  # noqa: F401 — фикстуры

Image = pytest.importorskip("PIL.Image")


@pytest.fixture
def requeue_on(monkeypatch):
    monkeypatch.setattr(settings, "PARSE_REQUEUE_ON_OUTAGE", True)
    MinerUParserProvider.reset_availability()
    yield
    MinerUParserProvider.reset_availability()


def _png() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (40, 40), "white").save(buffer, format="PNG")
    return buffer.getvalue()


def _image_context():
    return DocumentContext(uri="documents/scan.png", file_type="png", data=_png())


# ------------------------------------------------------------- область
def test_no_requeue_outside_task_or_without_flag(monkeypatch):
    monkeypatch.setattr(settings, "PARSE_REQUEUE_ON_OUTAGE", True)
    assert not outage.requeue_allowed()          # вне задачи
    monkeypatch.setattr(settings, "PARSE_REQUEUE_ON_OUTAGE", False)
    with outage.scope():
        assert not outage.requeue_allowed()


def test_wait_is_bounded(requeue_on, monkeypatch):
    monkeypatch.setattr(settings, "PARSE_OUTAGE_MAX_WAIT_SECONDS", 3600)
    with outage.scope(time.time() - 60):
        assert outage.requeue_allowed()
    with outage.scope(time.time() - 3601):
        assert not outage.requeue_allowed()


# --------------------------------------------------------------- MinerU
def test_layout_level_raises_for_raster_when_mineru_is_down(requeue_on):
    MinerUParserProvider.mark_unavailable()
    with outage.scope(), pytest.raises(outage.ServiceOutage):
        # Через лестницу: общий перехват неудачи уровня его не глотает.
        LadderRouter(strategies=[LayoutStrategy()]).parse(_image_context())


def test_document_finished_below_level5_does_not_wait(requeue_on):
    # Уровню 4 хватило оценки — до уровня 5 лестница не доходит, и лежащий
    # MinerU документ не задерживает.
    from tests.unit.test_ladder import FakeStrategy

    MinerUParserProvider.mark_unavailable()
    with outage.scope():
        result = LadderRouter(strategies=[FakeStrategy(4, confidence=1.0), LayoutStrategy()],
                              threshold=0.75).parse(_image_context())
    assert result.ladder["chosen_level"] == 4


def test_layout_level_skipped_as_before_without_flag(monkeypatch):
    monkeypatch.setattr(settings, "PARSE_REQUEUE_ON_OUTAGE", False)
    MinerUParserProvider.mark_unavailable()
    try:
        with outage.scope():
            assert LayoutStrategy().applicable(_image_context()) is False
    finally:
        MinerUParserProvider.reset_availability()


def test_vector_pdf_keeps_text_layer_route(requeue_on):
    # Документ с текстовым слоем при лежащем MinerU читается по слою, как сегодня.
    context = DocumentContext(uri="documents/x.pdf", file_type="pdf", data=b"%PDF-1.4")
    context._classification = type("C", (), {"source": "vector_text"})()
    MinerUParserProvider.mark_unavailable()
    with outage.scope():
        assert LayoutStrategy().applicable(context) is False


def _mineru(monkeypatch, error):
    provider = MinerUParserProvider.__new__(MinerUParserProvider)
    monkeypatch.setattr(provider, "_read_source", lambda uri: b"png", raising=False)
    monkeypatch.setattr(provider, "_read_pdf", lambda data, file_type: (None, {}, 1), raising=False)

    def down(*args, **kwargs):
        raise error

    monkeypatch.setattr(provider, "_parse_with_mineru", down, raising=False)
    monkeypatch.setattr(
        provider, "_full_tesseract_fallback",
        lambda uri, file_type: ParseResult(
            blocks=[ParsedBlock(type="text", text="плоский OCR", page=1)], is_fallback=True,
        ),
        raising=False,
    )
    return provider


def test_mineru_refused_on_scan_requeues(requeue_on, monkeypatch):
    provider = _mineru(monkeypatch, ParserUnavailable("нет связи с MinerU"))
    with outage.scope(), pytest.raises(outage.ServiceOutage):
        provider.parse("documents/scan.png", "png", {})


def test_mineru_timeout_stays_degradation(requeue_on, monkeypatch):
    provider = _mineru(monkeypatch, ParserTimedOut("MinerU не уложился"))
    with outage.scope():
        result = provider.parse("documents/scan.png", "png", {})
    assert any(note.startswith("mineru_timeout") for note in result.degraded)


def test_max_wait_exceeded_gives_todays_result(requeue_on, monkeypatch):
    provider = _mineru(monkeypatch, ParserUnavailable("нет связи с MinerU"))
    with outage.scope(time.time() - settings.PARSE_OUTAGE_MAX_WAIT_SECONDS - 1):
        result = provider.parse("documents/scan.png", "png", {})
    assert [block.text for block in result.blocks] == ["плоский OCR"]
    assert any(note.startswith("mineru_unavailable") for note in result.degraded)


# ---------------------------------------------------------------- Qwen-VL
class _Session:
    def __init__(self, error=None, status=200):
        self.error, self.status = error, status

    def post(self, url, json=None, headers=None, timeout=None):
        if self.error is not None:
            raise self.error
        status = self.status

        class Response:
            status_code = status
            text = "upstream down"

            @staticmethod
            def json():
                return {"choices": [{"message": {"content": '{"lines": ["А"]}'},
                                     "finish_reason": "stop"}]}

        return Response()


def _ask(session):
    client = QwenVisionClient(endpoint="http://vlm.test/v1", model="qwen", session=session)
    client.retries = 0
    return client.transcribe(Image.new("RGB", (32, 32), "white"))


@pytest.mark.parametrize("session", [
    _Session(error=requests.ConnectionError("Connection refused")),
    _Session(status=503),
])
def test_vlm_refused_for_whole_document_requeues(requeue_on, session):
    with outage.scope():
        assert _ask(session) is None
        with pytest.raises(outage.ServiceOutage):
            outage.check_vlm()


def test_vlm_read_timeout_is_per_sheet(requeue_on):
    with outage.scope():
        assert _ask(_Session(error=requests.ReadTimeout("read timed out"))) is None
        outage.check_vlm()          # не отказ сервиса


def test_one_answer_means_service_is_up(requeue_on):
    with outage.scope():
        _ask(_Session(error=requests.ConnectionError("Connection refused")))
        assert _ask(_Session()) == ["А"]
        outage.check_vlm()


# ------------------------------------------------------------------ задача
def test_task_requeues_and_stays_processing(requeue_on, pipeline, task_db, fake_redis, mocker):  # noqa: F811
    requeue = mocker.patch.object(tasks.process_document, "apply_async")
    real_parse = pipeline.parse
    pipeline.parse = mocker.Mock(side_effect=outage.ServiceOutage("MinerU недоступен"))
    init_request_state("req-out", ["file-out"])

    before = time.time()
    result = process_document("req-out", "file-out")

    assert result["status"] == "requeued_on_outage"
    requeue.assert_called_once()
    kwargs = requeue.call_args.kwargs["kwargs"]
    assert kwargs["request_id"] == "req-out" and kwargs["s3_fileid"] == "file-out"
    assert kwargs["outage_since"] >= before
    assert requeue.call_args.kwargs["countdown"] == settings.PARSE_OUTAGE_RETRY_SECONDS
    # RAG видит файл «в обработке», а не ошибку; файл не считается пропавшим.
    assert get_request_state("req-out")["status"] == "processing"
    assert not fake_redis.hexists("req:req-out:since", "file-out")
    session = task_db()
    assert DocumentRepository(session).get(generate_doc_id("file-out")).status == "pending"
    session.close()

    # Сервис поднялся: повтор находит свой pending-документ и разбирает его.
    pipeline.parse = real_parse
    result = process_document("req-out", "file-out", outage_since=kwargs["outage_since"])
    assert result["status"] == "success"
    state = get_request_state("req-out")
    assert state["status"] == "completed"
    assert len(state["documents"]) == 1


def test_task_without_flag_is_unchanged(pipeline, task_db, fake_redis, mocker, monkeypatch):  # noqa: F811
    monkeypatch.setattr(settings, "PARSE_REQUEUE_ON_OUTAGE", False)
    requeue = mocker.patch.object(tasks.process_document, "apply_async")
    init_request_state("req-plain", ["file-plain"])
    assert process_document("req-plain", "file-plain")["status"] == "success"
    requeue.assert_not_called()


# ------------------------------------------- пометка LAYOUT_UNAVAILABLE
def _ladder_with_ocr():
    from tests.unit.test_ladder import FakeStrategy

    return LadderRouter(strategies=[FakeStrategy(4, confidence=1.0, fallback=True),
                                    LayoutStrategy()], threshold=0.75)


@pytest.fixture
def note_on(monkeypatch):
    monkeypatch.setattr(settings, "PARSE_LAYOUT_UNAVAILABLE_NOTE", True)
    MinerUParserProvider.reset_availability()
    yield
    MinerUParserProvider.reset_availability()


def test_scan_left_on_ocr_is_marked(note_on, monkeypatch):
    from core.ladder.router import LAYOUT_UNAVAILABLE

    monkeypatch.setattr(settings, "PARSE_REQUEUE_ON_OUTAGE", False)
    MinerUParserProvider.mark_unavailable()
    with outage.scope():
        result = _ladder_with_ocr().parse(_image_context())
    assert result.ladder["chosen_level"] == 4
    assert LAYOUT_UNAVAILABLE in result.degraded
    assert LAYOUT_UNAVAILABLE in result.ladder["degraded"]


def test_scan_marked_after_outage_wait_ran_out(note_on, monkeypatch):
    from core.ladder.router import LAYOUT_UNAVAILABLE

    monkeypatch.setattr(settings, "PARSE_REQUEUE_ON_OUTAGE", True)
    monkeypatch.setattr(settings, "PARSE_OUTAGE_MAX_WAIT_SECONDS", 3600)
    MinerUParserProvider.mark_unavailable()
    with outage.scope(time.time() - 3601):
        result = _ladder_with_ocr().parse(_image_context())
    assert LAYOUT_UNAVAILABLE in result.degraded


def test_no_mark_without_flag(monkeypatch):
    monkeypatch.setattr(settings, "PARSE_REQUEUE_ON_OUTAGE", False)
    MinerUParserProvider.mark_unavailable()
    try:
        with outage.scope():
            result = _ladder_with_ocr().parse(_image_context())
    finally:
        MinerUParserProvider.reset_availability()
    assert result.degraded == []


def test_scan_marked_when_layout_level_failed(note_on, monkeypatch):
    # MinerU отклонил задачу (409 с нехваткой памяти) или не дал блоков:
    # скан остался плоским OCR — тоже с пометкой.
    def run(self, context, prefetched=None):
        raise ValueError("нет блоков")

    monkeypatch.setattr(LayoutStrategy, "run", run)
    with outage.scope():
        result = _ladder_with_ocr().parse(_image_context())
    assert result.ladder["chosen_level"] == 4
    assert "LAYOUT_UNAVAILABLE" in result.degraded


def test_no_mark_when_layout_level_delivered(note_on, monkeypatch):
    from tests.unit.test_ladder import FakeStrategy

    with outage.scope():
        result = LadderRouter(strategies=[FakeStrategy(4, confidence=1.0, fallback=True),
                                          FakeStrategy(5, confidence=1.0, structured=True)],
                              threshold=0.75).parse(_image_context())
    assert result.ladder["chosen_level"] == 5
    assert "LAYOUT_UNAVAILABLE" not in result.degraded


def test_vector_pdf_is_not_marked(note_on):
    context = DocumentContext(uri="documents/x.pdf", file_type="pdf", data=b"%PDF-1.4")
    context._classification = type("C", (), {"source": "vector_text"})()
    MinerUParserProvider.mark_unavailable()
    assert LayoutStrategy().skipped_as_unavailable(context) is False
