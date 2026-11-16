"""
Лист, который ветка чертежей начала и потеряла (таймаут чтения модели,
исключение листа, пустой ответ), попадает в `DRAWING_SHEETS_SKIPPED:`
(DRAWING_LOST_SHEETS_SKIPPED).
"""

import pytest

from core.config import settings
from core.drawing_processor import DrawingProcessor
from core.models.contract import Content, Fragment, Position, Provenance
from core.providers.qwen_client import QwenVisionClient

META = {
    "source_path": "src/scan.pdf",
    "classification": {"signals": {"raster_drawing_pages": [1, 2, 3]}},
}


def _fragment(page, kind="drawing"):
    return Fragment(
        fragment_id=f"f{page}", type=kind, content=Content(text=f"лист {page}"),
        position=Position(page=page, bbox=[0, 0, 1, 1]), confidence=0.5, completeness=None,
        provenance=Provenance(method="vlm_full_page", strategy_level=7, source="drawing"),
    )


@pytest.fixture
def processor(temp_storage, monkeypatch):
    processor = DrawingProcessor(temp_storage, qwen=QwenVisionClient(endpoint="http://vlm.test/v1"))
    monkeypatch.setattr(processor, "_source_is_raster_drawing", lambda metadata: True)
    monkeypatch.setattr(processor, "_full_read_image", lambda metadata: False)
    monkeypatch.setattr(processor, "_full_read_pdf", lambda metadata: False)

    def readings(answers):
        monkeypatch.setattr(processor, "_read_sheets",
                            lambda uri, metadata, pages, full, fields: list(answers))
        monkeypatch.setattr(
            processor, "build_fragment",
            lambda result, anchor, metadata: None if result == "empty"
            else _fragment(anchor.position.page),
        )
        return processor

    return readings


def _warnings(fragments):
    return [w for f in fragments for w in f.warnings]


def test_lost_sheet_is_silent_without_flag(processor):
    sheets = processor(["r1", None, "empty"])._sheet_drawings([_fragment(9, "text")], META)
    assert [s.position.page for s in sheets] == [1]
    assert _warnings(sheets) == []


def test_lost_sheets_are_listed_with_flag(processor, monkeypatch):
    monkeypatch.setattr(settings, "DRAWING_LOST_SHEETS_SKIPPED", True)
    sheets = processor(["r1", None, "empty"])._sheet_drawings([_fragment(9, "text")], META)
    assert [s.position.page for s in sheets] == [1]
    assert _warnings(sheets) == ["DRAWING_SHEETS_SKIPPED:2,3"]
    assert sheets[0].review_required


def test_no_sheet_read_marks_other_fragment(processor, monkeypatch):
    monkeypatch.setattr(settings, "DRAWING_LOST_SHEETS_SKIPPED", True)
    others = [_fragment(1, "text"), _fragment(2, "text")]
    sheets = processor([None, None, None])._sheet_drawings(others, META)
    assert sheets == []
    assert others[0].warnings == ["DRAWING_SHEETS_SKIPPED:1,2,3"]


def test_all_sheets_read_gives_no_mark(processor, monkeypatch):
    monkeypatch.setattr(settings, "DRAWING_LOST_SHEETS_SKIPPED", True)
    sheets = processor(["r1", "r2", "r3"])._sheet_drawings([_fragment(9, "text")], META)
    assert len(sheets) == 3 and _warnings(sheets) == []
