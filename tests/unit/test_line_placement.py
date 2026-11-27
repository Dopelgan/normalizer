"""Место надписи без места по строкам распознавания (DRAWING_PLACE_FROM_LINES)."""

from core.config import settings
from core.models.contract import (
    Content, DrawingField, Fragment, Position, Provenance, RecognitionLine,
)
from core.providers import field_placement, line_placement


def line(text, bbox):
    return RecognitionLine(text=text, bbox=bbox, source="ocr_layer")


def drawing(fields, lines):
    return Fragment(
        fragment_id="sheet", type="drawing",
        content=Content(structured_drawing_fields=fields,
                        structured_payload={"coordinate_reference": "page"}),
        position=Position(page=1, bbox=[0.0, 0.0, 1.0, 1.0], lines=lines),
        confidence=0.0, completeness=0.0,
        provenance=Provenance(method="vision_plus_vlm", strategy_level=7, source="drawing"),
    )


def field(value, bbox, exact=None, category="size"):
    return DrawingField(category=category, value=value, bbox=bbox, confidence=0.8, bbox_exact=exact)


LINES = [line("88", [0.59, 0.55, 0.62, 0.57]), line("162", [0.69, 0.55, 0.72, 0.57]),
         line("Масса 1,26", [0.80, 0.90, 0.90, 0.92]), line("3", [0.1, 0.1, 0.11, 0.11]),
         line("3", [0.3, 0.3, 0.31, 0.31]),
         line("Сварка аргонодуговая по ГОСТ 14771", [0.10, 0.70, 0.50, 0.72])]


def test_unplaced_fields_take_unique_line_boxes():
    sheet = drawing([
        field("88", [0.0, 0.0, 1.0, 1.0], exact=False),
        field("162", [0.0, 0.0, 1.0, 1.0], exact=False),
        field("Масса: 1,26", [0.0, 0.0, 1.0, 1.0], exact=False, category="title_block"),
        field("3", [0.0, 0.0, 1.0, 1.0], exact=False),             # две строки «3» — место неизвестно
        field("ГОСТ 14771", [0.0, 0.0, 1.0, 1.0], exact=False, category="note"),
        field("88", [0.2, 0.2, 0.22, 0.21], exact=True),            # с местом — не трогается
    ], LINES)
    stats = line_placement.apply([sheet])
    got = [(f.value, f.bbox, f.bbox_exact) for f in sheet.content.structured_drawing_fields]
    assert got[0] == ("88", [0.59, 0.55, 0.62, 0.57], True)
    assert got[1] == ("162", [0.69, 0.55, 0.72, 0.57], True)
    # «Масса: 1,26» ищется без подписи: «1.26» — конец строки «Масса 1,26».
    assert got[2][2] is True and got[2][1][2] == 0.9 and got[2][1][0] > 0.85
    assert got[3] == ("3", [0.0, 0.0, 1.0, 1.0], False)
    x1, y1, x2, y2 = got[4][1]
    assert got[4][2] is True and 0.10 < x1 < x2 <= 0.50 and (y1, y2) == (0.70, 0.72)
    assert got[5] == ("88", [0.2, 0.2, 0.22, 0.21], True)
    assert stats["placed"] == 4


def test_label_is_stripped_and_inside_match_needs_length():
    assert line_placement.find_box("Масса: 1,26", [("масса1.26", [0.8, 0.9, 0.9, 0.92])]) is not None
    assert line_placement.find_box("Масса: 1,26", [("1.26", [0.8, 0.9, 0.9, 0.92])]) == [0.8, 0.9, 0.9, 0.92]
    assert line_placement.find_box("26", [("масса1.26", [0.8, 0.9, 0.9, 0.92])]) is None


def test_flag_off_leaves_fields(monkeypatch):
    monkeypatch.setattr(settings, "DRAWING_PLACE_FROM_LINES", False)
    sheet = drawing([field("88", [0.0, 0.0, 1.0, 1.0], exact=False)], LINES)
    sheet.content.structured_payload = {"coordinate_reference": "page"}
    field_placement.apply([sheet])
    assert sheet.content.structured_drawing_fields[0].bbox == [0.0, 0.0, 1.0, 1.0]


WHOLE = ("2. Сварка аргонодуговая ГОСТ 14771-76. Присадочная проволока Св-04Х19Н11М3 ГОСТ 2246-70. "
         "Масса наплавленного металла 20 г.")


def test_whole_note_takes_its_pieces_box_and_pieces_go():
    sheet = drawing([
        field("1. Размеры для справок.", [0.09, 0.65, 0.3, 0.67], exact=True, category="note"),
        field(WHOLE, [0.0, 0.0, 1.0, 1.0], exact=False, category="note"),
        field("2. Сварка аргонодуговая ГОСТ 14771-76.", [0.09, 0.67, 0.45, 0.69], exact=True, category="note"),
        field("Присадочная проволока Св-04Х19Н11М3 ГОСТ 2246-70. Масса наплавленного металла 20 г.",
              [0.09, 0.69, 0.9, 0.71], exact=True, category="note"),
    ], [])
    stats = line_placement.merge_note_pieces([sheet])
    got = [(f.value, f.bbox, f.bbox_exact) for f in sheet.content.structured_drawing_fields]
    assert got == [("1. Размеры для справок.", [0.09, 0.65, 0.3, 0.67], True),
                   (WHOLE, [0.09, 0.67, 0.9, 0.71], True)]
    assert stats == {"merged": 1, "pieces": 2}


def test_pieces_that_do_not_cover_the_whole_note_stay():
    sheet = drawing([
        field("2. Сварка аргонодуговая ГОСТ 14771-76. Присадочная проволока Св-04.", [0.0, 0.0, 1.0, 1.0],
              exact=False, category="note"),
        field("2. Сварка аргонодуговая ГОСТ 14771-76.", [0.09, 0.67, 0.45, 0.69], exact=True, category="note"),
    ], [])
    assert line_placement.merge_note_pieces([sheet]) == {"merged": 0, "pieces": 0}
    assert len(sheet.content.structured_drawing_fields) == 2
