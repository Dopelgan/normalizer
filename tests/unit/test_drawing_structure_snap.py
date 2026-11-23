"""Структура листа, S3: рамка элемента по строке текстового слоя или распознавания."""

from core.drawing_structure import candidates, merge, snap
from core.models.contract import Content, DrawingField, Fragment, Position, Provenance


def _element(value, bbox, category="size"):
    pool = candidates.collect([Fragment(
        fragment_id="w", type="drawing",
        content=Content(structured_drawing_fields=[DrawingField(category=category, value=value, bbox=bbox)],
                        structured_payload={}),
        position=Position(page=1), confidence=0.8, completeness=1.0,
        provenance=Provenance(method="vlm_full_page", strategy_level=7, source="drawing"),
    )])
    return merge.merge(pool.candidates)[0]


def test_vlm_element_takes_line_bbox_when_inscription_matches():
    element = _element("⌀8,5", [0.30, 0.30, 0.36, 0.34])
    lines = [{"text": "⌀8,5", "bbox": [0.312, 0.318, 0.335, 0.329], "source": "ocr"}]
    assert snap.snap([element], lines) == 1
    assert element.bbox == [0.312, 0.318, 0.335, 0.329] and element.exact
    assert element.value == "⌀8,5"


def test_garbage_ocr_never_changes_value_and_gives_no_geometry():
    element = _element("⌀8,5", [0.30, 0.30, 0.36, 0.34])
    lines = [{"text": "Каарёинаты отверстий 88.5", "bbox": [0.30, 0.31, 0.50, 0.33], "source": "ocr"}]
    assert snap.snap([element], lines) == 0
    assert element.value == "⌀8,5" and element.bbox == [0.30, 0.30, 0.36, 0.34]
    assert not element.exact


def test_far_line_with_same_text_is_ignored():
    element = _element("R1", [0.20, 0.20, 0.22, 0.22])
    lines = [{"text": "R1", "bbox": [0.70, 0.70, 0.72, 0.72], "source": "text_layer"}]
    assert snap.snap([element], lines) == 0


def test_long_line_gives_span_of_the_inscription():
    element = _element("15±0,1", [0.40, 0.40, 0.50, 0.44])
    lines = [{"text": "15±0,1 п4", "bbox": [0.41, 0.41, 0.50, 0.43], "source": "ocr"}]
    assert snap.snap([element], lines) == 1
    assert element.bbox[0] == 0.41 and element.bbox[2] < 0.50


def test_unplaced_note_snaps_only_to_a_unique_line():
    element = _element("Острые кромки притупить", [0.0, 0.0, 1.0, 1.0], category="note")
    unique = [{"text": "3. Острые кромки притупить.", "bbox": [0.1, 0.8, 0.4, 0.82], "source": "ocr"}]
    assert snap.snap([element], unique) == 1 and element.exact
    twice = _element("Острые кромки притупить", [0.0, 0.0, 1.0, 1.0], category="note")
    assert snap.snap([twice], unique + [{"text": "Острые кромки притупить", "bbox": [0.5, 0.5, 0.8, 0.52],
                                         "source": "ocr"}]) == 0
