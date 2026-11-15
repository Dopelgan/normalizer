"""
Перечитывание плохого скана (DRAWING_RESCAN_POOR_OCR_PAGES) по договору
`process`: строки текста лежат в `payload.transcript_lines`, а
`structured_payload.transcript_line_count` появляется только после
`build_fragment`. Проверка ответа `process` по полю фрагмента выбрасывала
каждое удачное прочтение.

Модель поддельная; проверяется, дошло ли прочтение до фрагмента и
сохранились ли непрочитанные части листа.
"""

import pytest

from core import drawing_processor as dp
from core.drawing_processor import DrawingProcessor
from core.drawing_quality import apply_drawing_quality, document_quality
from core.models.contract import Content, Fragment, Position, Provenance
from core.scan_text_rescan import Verdict

pytest.importorskip("numpy")
pytest.importorskip("PIL")

from tests.fixtures import synthetic_sheet as sheet  # noqa: E402

SCAN_URI = "assets/doc-1/scan.png"


class TranscribingQwen:
    """Модель, которая умеет только «весь текст листа»."""

    def __init__(self, lines):
        self.available = True
        self.model = "fake-vl"
        self.lines = lines
        self.calls = 0

    def transcribe(self, image):
        self.calls += 1
        return self.lines


def _ocr_fragment(page):
    return Fragment(
        fragment_id=f"ocr-{page}", type="text",
        content=Content(text="испорченный текст"),
        position=Position(page=page, bbox=[0.0, 0.0, 1.0, 1.0], order=1),
        confidence=0.3, completeness=1.0,
        provenance=Provenance(method="ocr_layer", strategy_level=5, source="scanned_pdf"),
    )


@pytest.fixture
def rescan_on(monkeypatch, temp_storage):
    temp_storage.write_file(SCAN_URI, sheet.png_bytes())
    monkeypatch.setattr(dp.settings, "DRAWING_RESCAN_POOR_OCR_PAGES", True)
    monkeypatch.setattr(dp.settings, "DRAWING_TRANSCRIBE_TILES", False)
    monkeypatch.setattr(dp.settings, "DRAWING_DIMENSION_FIDELITY", False)
    monkeypatch.setattr(dp, "poor_pages", lambda fragments, *_: {
        1: Verdict(rescan=True, score=0.2, reasons=["low_score"]),
    })
    monkeypatch.setattr(dp.deadline, "remaining", lambda: None)
    return temp_storage


def test_successful_rescan_becomes_a_fragment(rescan_on):
    qwen = TranscribingQwen(["Технические требования", "1. Размеры для справок"])
    processor = DrawingProcessor(rescan_on, qwen=qwen)

    sheets = processor._rescan_poor_pages(
        [_ocr_fragment(1)], {"source_path": SCAN_URI, "doc_id": "doc-1"},
    )

    assert qwen.calls == 1
    assert len(sheets) == 1
    sheet_fragment = sheets[0]
    assert sheet_fragment.content.text == "Технические требования\n1. Размеры для справок"
    payload = sheet_fragment.content.structured_payload
    assert payload["transcript_line_count"] == 2
    assert payload["rescan_reason"]["rescan"] is True


def test_empty_rescan_adds_nothing(rescan_on):
    qwen = TranscribingQwen([])
    processor = DrawingProcessor(rescan_on, qwen=qwen)

    sheets = processor._rescan_poor_pages(
        [_ocr_fragment(1)], {"source_path": SCAN_URI, "doc_id": "doc-1"},
    )

    assert qwen.calls == 1
    assert sheets == []


def test_partial_rescan_keeps_missing_tiles(rescan_on, monkeypatch):
    """Часть листа не прочитана: строки идут во фрагмент, пропуск — в missing_regions."""
    processor = DrawingProcessor(rescan_on, qwen=TranscribingQwen([]))

    def partial(image, payload, missing_regions, started):
        payload["transcript_complete"] = False
        missing_regions.append({"kind": "transcript_tile", "bbox": [0.5, 0.0, 1.0, 1.0],
                                "reason": "timeout"})
        return ["Технические требования"]

    monkeypatch.setattr(processor, "_transcribe", partial)
    sheets = processor._rescan_poor_pages(
        [_ocr_fragment(1)], {"source_path": SCAN_URI, "doc_id": "doc-1"},
    )

    assert len(sheets) == 1
    assert sheets[0].content.structured_payload["transcript_complete"] is False
    assert [r["kind"] for r in sheets[0].missing_regions] == ["transcript_tile"]


def test_already_transcribed_page_is_not_read_again(rescan_on):
    qwen = TranscribingQwen(["строка"])
    processor = DrawingProcessor(rescan_on, qwen=qwen)
    done = Fragment(
        fragment_id="sheet-1", type="drawing",
        content=Content(text="строка", structured_payload={"transcript_line_count": 1}),
        position=Position(page=1, bbox=[0.0, 0.0, 1.0, 1.0], order=0),
        confidence=0.5, completeness=None,
        provenance=Provenance(method="vlm_full_page", strategy_level=7, source="drawing"),
    )

    sheets = processor._rescan_poor_pages(
        [_ocr_fragment(1), done], {"source_path": SCAN_URI, "doc_id": "doc-1"},
    )

    assert sheets == []
    assert qwen.calls == 0


@pytest.mark.parametrize("unmeasured_not_partial", [True, False])
def test_rescan_sheet_does_not_make_document_partial(rescan_on, monkeypatch,
                                                     unmeasured_not_partial):
    """Перечитанная страница уже покрыта распознаванием: недочитанная часть
    или неизмеренная полнота листа-перечитывания ставит на проверку сам
    лист, но не делает документ частичным."""
    monkeypatch.setattr(dp.settings, "DRAWING_UNMEASURED_NOT_PARTIAL", unmeasured_not_partial)
    processor = DrawingProcessor(rescan_on, qwen=TranscribingQwen([]))

    def partial(image, payload, missing_regions, started):
        payload["transcript_complete"] = False
        missing_regions.append({"kind": "transcript_tile", "bbox": [0.5, 0.0, 1.0, 1.0],
                                "reason": "timeout"})
        return ["Технические требования"]

    monkeypatch.setattr(processor, "_transcribe", partial)
    ocr = _ocr_fragment(1)
    sheets = processor._rescan_poor_pages([ocr], {"source_path": SCAN_URI, "doc_id": "doc-1"})
    assert len(sheets) == 1
    apply_drawing_quality(sheets[0])
    assert sheets[0].review_required

    partial_doc, _ = document_quality([ocr, *sheets])
    assert not partial_doc

    # Тот же лист не из перечитывания документ частичным делает.
    del sheets[0].content.structured_payload["rescan_reason"]
    assert document_quality([ocr, *sheets])[0]
