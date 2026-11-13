"""
Второе чтение надписи не тащит служебную приписку в текст
(DRAWING_TRANSCRIPT_VARIANT_CLEAN).

Когда строки модели очень похожи на абзац распознавания, но расходятся
цифрой («Св-04х19н1им3» против «Св-04Х19Н11М3»), кто прав — неизвестно, и
оба чтения сохраняются. Приписка «(вариант чтения модели, требует сверки)»
при этом уходила прямо в `content.text` — то есть в выдачу поиска у
заказчика — и обещала проверку, которой не будет: качество чтения листа
человеку не адресовано. С флагом в тексте остаются оба чтения и ничего
больше, а чем они друг другу приходятся, сказано в payload.
"""

import pytest

from core.config import settings
from core.drawing_transcript import (
    VARIANT_MARK, VARIANT_WARNING, WARNING, apply_transcripts,
)
from core.models.contract import Content, Fragment, Position, Provenance

# Та же надпись, прочитанная двумя способами: расхождение в одной цифре.
OCR_TEXT = "Св-04х19н1им3"
MODEL_TEXT = "Св-04Х19Н11М3"


def ocr_paragraph(text=OCR_TEXT, page=1):
    return Fragment(
        fragment_id="t1", type="text",
        content=Content(text=text),
        position=Position(page=page, bbox=[0.1, 0.1, 0.9, 0.2]),
        confidence=0.5, completeness=None,
        provenance=Provenance(method="mineru_ocr", strategy_level=4,
                              source="ocr", source_unit_id="u1"),
    )


def sheet(lines, page=1):
    return Fragment(
        fragment_id="d1", type="drawing",
        content=Content(text="\n".join(lines),
                        structured_payload={"transcript_line_count": len(lines)}),
        position=Position(page=page, bbox=[0, 0, 1, 1]),
        confidence=0.8, completeness=None,
        provenance=Provenance(method="vlm_full_page", strategy_level=7,
                              source="drawing", source_unit_id="u2"),
    )


@pytest.fixture(autouse=True)
def fixes_on(monkeypatch):
    monkeypatch.setattr(settings, "DRAWING_TRANSCRIPT_FIXES_OCR", True)


@pytest.fixture
def clean(monkeypatch):
    monkeypatch.setattr(settings, "DRAWING_TRANSCRIPT_VARIANT_CLEAN", True)


def run(paragraph_text=OCR_TEXT, model_lines=(MODEL_TEXT,)):
    fragments = [ocr_paragraph(paragraph_text), sheet(list(model_lines))]
    return apply_transcripts(fragments)[0]


class TestTheMarkerLeavesTheIndexedText:
    def test_flag_off_keeps_todays_text_exactly(self):
        """Выключенный флаг ничего не меняет — ни строки, ни пометки."""
        fragment = run()
        assert fragment.content.text == f"{OCR_TEXT}\n{MODEL_TEXT}{VARIANT_MARK}"
        assert VARIANT_WARNING in fragment.warnings

    def test_flag_on_both_readings_stay_and_the_phrase_goes(self, clean):
        fragment = run()
        assert fragment.content.text == f"{OCR_TEXT}\n{MODEL_TEXT}"
        assert VARIANT_MARK not in (fragment.content.text or "")

    def test_the_promise_of_a_check_is_not_shown_to_anyone(self, clean):
        """«Требует сверки» обещало проверку, которой не будет."""
        assert "требует сверки" not in (run().content.text or "")

    def test_both_readings_are_still_findable(self, clean):
        """Оба чтения ищутся: ради этого второе и сохраняется."""
        text = run().content.text or ""
        assert OCR_TEXT in text and MODEL_TEXT in text


class TestThePayloadCarriesTheFact:
    def test_the_disagreement_is_written_down(self, clean):
        payload = run().content.structured_payload or {}
        assert payload["transcript_variants"] == [{"ocr": OCR_TEXT, "model": MODEL_TEXT}]
        assert payload["transcript_variant_count"] == 1

    def test_the_sheet_asks_to_be_verified_by_machine(self, clean):
        """Правильное действие при расхождении — сверить, а не спросить."""
        assert (run().content.structured_payload or {})["transcript_verify_candidate"] is True

    def test_flag_off_writes_no_payload(self):
        assert not (run().content.structured_payload or {})

    def test_nobody_is_summoned_by_a_variant(self, clean):
        """Качество чтения листа человеку не адресовано."""
        assert run().review_required is False

    def test_the_existing_warning_is_kept(self, clean):
        """На пометке стоит сегодняшний учёт: снимать её — отдельная перемена."""
        assert VARIANT_WARNING in run().warnings


class TestNothingElseMoves:
    def test_a_matching_reading_still_replaces_the_paragraph(self, clean):
        """Цифры совпали — это замена, а не вариант: путь не затронут."""
        fragment = run("Шероховатость Ра 16", ("Шероховатость Ra 16",))
        assert fragment.content.text == "Шероховатость Ra 16"
        assert WARNING in fragment.warnings
        assert not (fragment.content.structured_payload or {})

    def test_an_unrelated_paragraph_is_left_alone(self, clean):
        fragment = run("Совершенно другой абзац про поставку", (MODEL_TEXT,))
        assert fragment.content.text == "Совершенно другой абзац про поставку"
        assert not fragment.warnings

    def test_an_existing_payload_is_not_thrown_away(self, clean):
        fragments = [ocr_paragraph(), sheet([MODEL_TEXT])]
        fragments[0].content.structured_payload = {"своё": 1}
        payload = apply_transcripts(fragments)[0].content.structured_payload
        assert payload["своё"] == 1 and payload["transcript_variant_count"] == 1
