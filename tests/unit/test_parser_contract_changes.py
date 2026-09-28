"""
Изменения контракта Parser (docs/PARSER-CONTRACT-CHANGES.md).

Проверяется ровно то, что перечислено в разделе «Что нужно изменить»:
идентификаторы, факты, стабильный `fragment_id`, provenance, частичный
результат, объект ошибки и независимость результатов по `event_id`.
"""

import pytest
from fastapi.testclient import TestClient

from core.contract_finalize import (
    finalize_fragments,
    normalized_content_hash,
    stable_fragment_id,
)
from core.models.api import BackgroundParseRequest, ParseRequest
from core.models.contract import (
    BackgroundAcceptedResponse,
    Content,
    DocumentMetadata,
    DocumentResult,
    Fragment,
    ParserError,
    Position,
    Provenance,
    ResultsResponse,
)
from core.models.parse_result import ParsedBlock, ParseResult
from core.normalizer import TextNormalizer
from core.providers.document_parser import ParserFailed, ParserUnavailable
from core.providers.file_locator import SourceFileNotFound
from core.result_aggregator import get_request_state, init_request_state
from ingest import tasks
from ingest.api import app
from ingest.tasks import build_error_document, classify_error, page_report

META = {"doc_id": "doc-001", "s3_fileid": "cluster-a/document.pdf"}


def text_block(text, page=1, order=0, **kwargs):
    return ParsedBlock(type="text", text=text, page=page,
                       bbox=[0.0, 0.0, 1.0, 1.0], order=order, **kwargs)


def fragment(text="Текст фрагмента достаточной длины.", page=1, **kwargs):
    defaults = dict(
        fragment_id="",
        type="text",
        content=Content(text=text),
        position=Position(page=page, bbox=[0.0, 0.0, 1.0, 1.0], order=1),
        confidence=0.9,
        completeness=1.0,
        provenance=Provenance(method="text_layer_extraction", strategy_level=2,
                              source="vector_pdf"),
    )
    defaults.update(kwargs)
    return Fragment(**defaults)


def document(fileid, **kwargs):
    payload = dict(
        s3_fileid=fileid,
        status="completed",
        document_metadata=DocumentMetadata(doc_id=f"doc-{fileid}", status="indexed"),
        fragments=[],
    )
    payload.update(kwargs)
    return DocumentResult(**payload).model_dump(mode="json")


# ===========================================================================
# 1-2. Идентификаторы: event_id в фоне, dialog_id в синхронном разборе
# ===========================================================================

class TestIdentifiers:
    def test_background_request_has_no_dialog_id_field(self):
        request = BackgroundParseRequest(
            request_id="req-1", event_id="event-123", s3_fileid=["f1"]
        )
        assert "dialog_id" not in request.model_dump()
        assert request.event_id == "event-123"

    def test_background_request_ignores_incoming_dialog_id(self):
        request = BackgroundParseRequest.model_validate({
            "request_id": "req-1", "event_id": "event-123",
            "dialog_id": "event-123", "s3_fileid": ["f1"],
        })
        assert request.event_id == "event-123"
        assert "dialog_id" not in request.model_dump()

    def test_sync_request_still_requires_dialog_id(self):
        with pytest.raises(Exception):
            ParseRequest(request_id="req-1", s3_fileid=["f1"])
        assert ParseRequest(
            request_id="req-1", dialog_id="dlg", s3_fileid=["f1"]
        ).dialog_id == "dlg"

    def test_accepted_response_shape(self):
        payload = BackgroundAcceptedResponse(
            request_id="parser-request-123", event_id="event-123", operation="create"
        ).model_dump()
        assert set(payload) == {
            "request_id", "event_id", "operation", "accepted", "status",
        }

    def test_results_envelope_shape_matches_contract(self):
        payload = ResultsResponse(
            request_id="parser-request-123", event_id="event-123",
            operation="create", status="completed",
        ).model_dump()
        assert set(payload) == {
            "request_id", "event_id", "operation", "status", "documents", "error",
        }


# ===========================================================================
# 3. normalized_content_hash
# ===========================================================================

class TestNormalizedHash:
    def test_same_text_same_hash(self):
        first = normalized_content_hash([fragment()])
        second = normalized_content_hash([fragment()])
        assert first == second and first.startswith("sha256:")

    def test_other_text_other_hash(self):
        assert normalized_content_hash([fragment()]) != normalized_content_hash(
            [fragment(text="Совершенно другой текст документа.")]
        )

    def test_whitespace_does_not_change_meaning(self):
        assert normalized_content_hash([fragment(text="А  Б\nВ")]) == \
               normalized_content_hash([fragment(text="А Б В")])

    def test_empty_document(self):
        assert normalized_content_hash([]) is None


# ===========================================================================
# 4. extracted_facts
# ===========================================================================

class TestExtractedFacts:
    def test_facts_attached_to_fragment(self):
        fragments, _ = finalize_fragments(
            [fragment(text="Максимальная температура — 80 °C.")], "doc-001", "f1"
        )
        facts = fragments[0].extracted_facts
        assert facts and facts[0].value.number == 80
        assert facts[0].fragment_id == fragments[0].fragment_id
        assert facts[0].fact_id.startswith(fragments[0].fragment_id)

    def test_facts_can_be_skipped(self):
        fragments, _ = finalize_fragments(
            [fragment(text="Температура 80 °C.")], "doc-001", "f1", extract=False
        )
        assert fragments[0].extracted_facts == []

    def test_fragment_without_values_has_no_facts(self):
        fragments, _ = finalize_fragments(
            [fragment(text="Настоящий документ описывает порядок работ.")],
            "doc-001", "f1",
        )
        assert fragments[0].extracted_facts == []


# ===========================================================================
# 5. Стабильный fragment_id
# ===========================================================================

class TestStableFragmentId:
    def test_same_file_gives_same_ids(self):
        blocks = [text_block(f"Абзац {i} достаточной длины для фрагмента. " * 2, order=i)
                  for i in range(4)]
        first = TextNormalizer().normalize(ParseResult(blocks=blocks), META)
        second = TextNormalizer().normalize(ParseResult(blocks=blocks), META)
        assert [f.fragment_id for f in first] == [f.fragment_id for f in second]

    def test_new_fragment_does_not_shift_others(self):
        """
        Обработчик чертежей вставляет разбор листа в начало. Раньше это
        сдвигало нумерацию и RAG получал дубли вместо обновления.
        """
        tail = [fragment(text="Первый абзац."), fragment(text="Второй абзац.")]
        before, _ = finalize_fragments([f.model_copy(deep=True) for f in tail], "doc-001", "f1")
        with_sheet, _ = finalize_fragments(
            [fragment(text="Лист чертежа.")] + [f.model_copy(deep=True) for f in tail],
            "doc-001", "f1",
        )
        assert [f.fragment_id for f in before] == [f.fragment_id for f in with_sheet[1:]]

    def test_id_survives_reordering(self):
        one, two = fragment(text="Первый абзац."), fragment(text="Второй абзац.")
        direct, _ = finalize_fragments([one.model_copy(deep=True), two.model_copy(deep=True)],
                                       "doc-001", "f1")
        reversed_, _ = finalize_fragments([two.model_copy(deep=True), one.model_copy(deep=True)],
                                          "doc-001", "f1")
        assert {f.fragment_id for f in direct} == {f.fragment_id for f in reversed_}

    def test_duplicates_stay_unique(self):
        fragments, _ = finalize_fragments(
            [fragment(text="Повтор шапки."), fragment(text="Повтор шапки.")],
            "doc-001", "f1",
        )
        assert len({f.fragment_id for f in fragments}) == 2

    def test_id_fits_database_column(self):
        long_id = stable_fragment_id("doc-" + "0" * 32, fragment(text="Т" * 10_000))
        assert len(long_id) <= 80

    def test_id_map_reports_renames(self):
        source = fragment(text="Текст.", fragment_id="doc-001-frag-001")
        fragments, id_map = finalize_fragments([source], "doc-001", "f1")
        assert id_map["doc-001-frag-001"] == fragments[0].fragment_id


# ===========================================================================
# 6. Provenance
# ===========================================================================

class TestProvenance:
    def test_source_file_and_page_filled(self):
        fragments, _ = finalize_fragments(
            [fragment(page=4)], "doc-001", "cluster-a/document.pdf"
        )
        assert fragments[0].provenance.source_file_id == "cluster-a/document.pdf"
        assert fragments[0].provenance.source_page == 4

    def test_normalizer_fills_source_file_id(self):
        fragments = TextNormalizer().normalize(
            ParseResult(blocks=[text_block("Абзац достаточной длины для фрагмента. " * 2)]),
            META,
        )
        assert fragments[0].provenance.source_file_id == "cluster-a/document.pdf"
        assert fragments[0].provenance.source_page == 1


# ===========================================================================
# 7. Частичный результат
# ===========================================================================

class TestPartialResult:
    def test_pages_split_into_processed_and_failed(self):
        parse_result = ParseResult(
            blocks=[text_block("текст", page=page) for page in (1, 2, 3)],
            page_count=4, failed_pages=[3],
        )
        report = page_report(parse_result)
        assert report["pages"] == 4
        assert report["failed_pages"] == [3]
        assert report["processed_pages"] == [1, 2, 4]

    def test_warnings_mention_failed_page(self):
        parse_result = ParseResult(blocks=[], page_count=2, failed_pages=[2],
                                   degraded=["OCR_LOW_CONFIDENCE: page 2"])
        warnings = tasks.build_warnings(parse_result, [2])
        assert "OCR_LOW_CONFIDENCE: page 2" in warnings
        assert "PAGE_NOT_PARSED: page 2" in warnings

    def test_partial_document_makes_request_partial(self, fake_redis):
        init_request_state("req-partial", ["f1"], event_id="event-1")
        tasks.put_document_result("req-partial", "f1", document(
            "f1", status="partial", pages=18, processed_pages=[1, 2], failed_pages=[7],
            warnings=["OCR_LOW_CONFIDENCE: page 7"],
        ))
        assert get_request_state("req-partial")["status"] == "partial"


# ===========================================================================
# 8. Структурированная ошибка
# ===========================================================================

class TestStructuredError:
    @pytest.mark.parametrize("exc,code,retryable", [
        (SourceFileNotFound("нет файла"), "SOURCE_NOT_FOUND", False),
        (OSError("хранилище недоступно"), "SOURCE_READ_ERROR", True),
        (ParserFailed("формат не поддерживается"), "UNSUPPORTED_FORMAT", False),
        (ParserUnavailable("mineru лёг"), "EXTRACTION_FAILED", True),
        (ParserFailed("разбор не удался"), "EXTRACTION_FAILED", False),
        (RuntimeError("что-то пошло не так"), "INTERNAL_ERROR", True),
    ])
    def test_reason_codes(self, exc, code, retryable):
        error = classify_error(exc, attempt=2)
        assert error.reason_code == code
        assert error.retryable is retryable
        assert error.attempt == 2

    def test_error_is_object_in_document(self):
        payload = build_error_document("f1", "doc-1", "", ParserError(
            reason_code="UNSUPPORTED_FORMAT",
            message="Parser does not support this MIME type",
        ))
        assert payload["error"] == {
            "reason_code": "UNSUPPORTED_FORMAT",
            "message": "Parser does not support this MIME type",
            "retryable": False,
            "attempt": 1,
        }

    def test_legacy_string_error_is_coerced(self):
        """В Redis мог остаться документ со строковой ошибкой — не теряем его."""
        result = DocumentResult(
            s3_fileid="f1",
            document_metadata=DocumentMetadata(doc_id="d1"),
            error="Parser failed",
        )
        assert result.error.reason_code == "INTERNAL_ERROR"
        assert result.error.message == "Parser failed"

    def test_unknown_reason_code_rejected(self):
        with pytest.raises(Exception):
            ParserError(reason_code="СВОЙ_КОД", message="")


# ===========================================================================
# 9-10. Пачка: независимые результаты по event_id
# ===========================================================================

@pytest.fixture
def client(fake_redis, mocker):
    mocker.patch("ingest.api._dispatch")
    with TestClient(app) as test_client:
        yield test_client


class TestBatchIndependence:
    def test_each_event_gets_its_own_request(self, client):
        """
        RAG разворачивает пачку lifecycle-событий в отдельные запросы
        Parser — по одному файлу и одному `event_id` на запрос.
        """
        for number, (request_id, event_id, fileid) in enumerate([
            ("parser-request-1", "event-1", "cluster-a/a.pdf"),
            ("parser-request-2", "event-2", "cluster-a/b.pdf"),
        ]):
            response = client.post("/internal/v1/parse/background", json={
                "request_id": request_id, "event_id": event_id,
                "operation": "create" if number == 0 else "update",
                "s3_fileid": [fileid],
            })
            assert response.status_code == 202
            assert response.json()["event_id"] == event_id

    def test_failure_of_one_event_does_not_touch_another(self, client):
        init_request_state("parser-request-1", ["f1"], event_id="event-1")
        init_request_state("parser-request-2", ["f2"], event_id="event-2")
        tasks.put_document_result("parser-request-1", "f1", build_error_document(
            "f1", "d1", "", SourceFileNotFound("нет файла")
        ))
        tasks.put_document_result("parser-request-2", "f2", document("f2"))

        first = client.get("/internal/v1/parse/results/parser-request-1").json()
        second = client.get("/internal/v1/parse/results/parser-request-2").json()

        assert first["status"] == "error" and first["event_id"] == "event-1"
        assert first["error"]["reason_code"] == "SOURCE_NOT_FOUND"
        assert second["status"] == "completed" and second["event_id"] == "event-2"
        assert second["error"] is None

    def test_result_carries_exactly_one_document_per_event(self, client):
        init_request_state("parser-request-3", ["cluster-a/document.pdf"], event_id="event-3")
        tasks.put_document_result(
            "parser-request-3", "cluster-a/document.pdf", document("cluster-a/document.pdf")
        )
        body = client.get("/internal/v1/parse/results/parser-request-3").json()
        assert len(body["documents"]) == 1
        assert body["documents"][0]["s3_fileid"] == "cluster-a/document.pdf"

    def test_repeat_delivery_does_not_duplicate(self, fake_redis, mocker):
        published = mocker.patch("ingest.tasks.publish_result")
        init_request_state("parser-request-4", ["f1"], event_id="event-4")
        for _ in range(3):
            tasks.finalize("parser-request-4", "f1", document("f1"), event_id="event-4")
        published.assert_called_once()

    def test_background_publishes_envelope_with_event_id(self, fake_redis, mocker):
        published = mocker.patch("ingest.tasks.publish_result")
        init_request_state("parser-request-5", ["f1"], event_id="event-5")
        tasks.finalize("parser-request-5", "f1", document("f1"), event_id="event-5")

        envelope = published.call_args[0][0]
        assert isinstance(envelope, ResultsResponse)
        assert envelope.event_id == "event-5"
        assert "dialog_id" not in envelope.model_dump()

    def test_sync_publishes_envelope_with_dialog_id(self, fake_redis, mocker):
        published = mocker.patch("ingest.tasks.publish_result")
        init_request_state("req-sync", ["f1"], dialog_id="dialog-456")
        tasks.finalize("req-sync", "f1", document("f1"), dialog_id="dialog-456")

        envelope = published.call_args[0][0]
        assert envelope.dialog_id == "dialog-456"
        assert "event_id" not in envelope.model_dump()

    def test_polling_and_push_share_the_envelope(self, client, fake_redis, mocker):
        published = mocker.patch("ingest.tasks.publish_result")
        init_request_state("parser-request-6", ["f1"], event_id="event-6", operation="create")
        tasks.finalize("parser-request-6", "f1", document("f1"), event_id="event-6")

        pushed = published.call_args[0][0].model_dump(mode="json")
        polled = client.get("/internal/v1/parse/results/parser-request-6").json()
        assert set(pushed) == set(polled)
        assert pushed["event_id"] == polled["event_id"] == "event-6"
        assert pushed["status"] == polled["status"] == "completed"
