"""
Сведение прочтений полей чертежа (DRAWING_READINGS_RECONCILED, карточка D3).

Одна надпись листа, прочитанная проходом «лист целиком» и вырезкой вида,
прежде давала два поля в двух фрагментах страницы — «⌀280±0,3» и
«⌀280+0,3» — без пометки, какое из них спорное. Теперь — одно поле с
прочтениями, согласием и уверенностью по согласию, а запись листа называет
расхождение одной строкой.
"""

import pytest

from core import drawing_record
from core.config import settings
from core.models.contract import Content, DrawingField, Fragment, Position, Provenance
from core.providers import reading_reconcile
from core.providers.reading_reconcile import AGREED, CONFLICT, SINGLE, VECTOR


def field(value, bbox, category="size", confidence=0.9):
    return DrawingField(category=category, value=value, bbox=bbox, confidence=confidence)


def drawing(fields, bbox=(0.0, 0.0, 1.0, 1.0), text=None, method="vision_plus_vlm", name="sheet"):
    payload = {"transcript_line_count": len(text.split("\n"))} if text else {}
    return Fragment(
        fragment_id=name, type="drawing",
        content=Content(text=text, structured_drawing_fields=fields, structured_payload=payload),
        position=Position(page=1, bbox=list(bbox)), confidence=0.0, completeness=0.0,
        provenance=Provenance(method=method, strategy_level=7, source="drawing"),
    )


def ocr(text):
    return Fragment(fragment_id="ocr", type="text", content=Content(text=text),
                    position=Position(page=1, bbox=[0.0, 0.0, 1.0, 1.0]), confidence=0.0, completeness=0.0,
                    provenance=Provenance(method="ocr", strategy_level=5, source="ocr"))


# Вырезка вида — левая половина листа: рамка поля в ней в долях вырезки.
CROP = (0.0, 0.0, 0.5, 1.0)


def flange():
    """Фланец: ⌀280 прочитан листом с «±», вырезкой — с «+»."""
    sheet = drawing([field("⌀280±0,3", [0.30, 0.34, 0.32, 0.40]), field("⌀315", [0.42, 0.35, 0.45, 0.41])],
                    text="ТАДУ 302389.001.01.02.001\nМасса 3,74")
    crop = drawing([field("⌀280+0,3", [0.60, 0.34, 0.64, 0.40]), field("⌀18", [0.1, 0.8, 0.14, 0.84])],
                   bbox=CROP, name="crop")
    return sheet, crop


class TestFlange:
    def test_one_field_with_conflict_mark(self):
        sheet, crop = flange()
        reading_reconcile.apply([sheet, crop])
        f280 = sheet.content.structured_drawing_fields[0]
        assert f280.value == "⌀280±0,3" and f280.agreement == CONFLICT
        assert f280.readings == [{"source": "sheet", "value": "⌀280±0,3"},
                                 {"source": "crop", "value": "⌀280+0,3"}]
        assert f280.confidence == 0.4
        # Прочтение вырезки ушло в поле листа; своё поле вырезки осталось.
        assert [f.value for f in crop.content.structured_drawing_fields] == ["⌀18"]
        assert crop.content.structured_drawing_fields[0].agreement == SINGLE

    def test_record_prints_one_line_with_both_readings(self, monkeypatch):
        monkeypatch.setattr(settings, "DRAWING_READINGS_RECONCILED", True)
        sheet, crop = flange()
        reading_reconcile.apply([sheet, crop])
        text = drawing_record.compose_fragment(sheet)
        assert "⌀280±0,3 (прочтения расходятся: ⌀280±0,3 | ⌀280+0,3)" in text
        assert text.count("⌀280+0,3") == 1

    def test_far_twin_is_another_inscription(self):
        sheet, crop = flange()
        crop.content.structured_drawing_fields[0].bbox = [0.6, 0.9, 0.64, 0.95]
        reading_reconcile.apply([sheet, crop])
        assert sheet.content.structured_drawing_fields[0].agreement == SINGLE
        assert len(crop.content.structured_drawing_fields) == 2


class TestTip:
    """Наконечник: угол 10° прочитан тремя проходами по-разному."""

    def test_three_readings_without_majority_are_a_conflict(self):
        sheet = drawing([field("10°±2°", [0.46, 0.25, 0.48, 0.30], category="tolerance")],
                        text="Угол 10°±3°")
        crop = drawing([field("10°±1°", [0.92, 0.25, 0.96, 0.30]), field("⌀18", [0.1, 0.8, 0.14, 0.84])],
                       bbox=CROP, name="crop")
        reading_reconcile.apply([sheet, crop])
        angle = sheet.content.structured_drawing_fields[0]
        assert angle.agreement == CONFLICT and angle.value == "10°±2°"
        assert [r["value"] for r in angle.readings] == ["10°±2°", "10°±1°", "10°±3°"]
        assert [f.value for f in crop.content.structured_drawing_fields] == ["⌀18"]

    def test_two_sources_outvote_the_sheet(self):
        sheet = drawing([field("70*2", [0.32, 0.12, 0.35, 0.14])], text="70+2")
        crop = drawing([field("70+2", [0.62, 0.11, 0.68, 0.13])], bbox=CROP, name="crop")
        reading_reconcile.apply([sheet, crop])
        size = sheet.content.structured_drawing_fields[0]
        assert size.value == "70+2" and size.agreement == AGREED and size.confidence == 0.9

    def test_lost_decimal_comma_restored_by_majority(self):
        sheet = drawing([field("134,9", [0.5, 0.5, 0.52, 0.52])], text="1349")
        crop = drawing([field("1349", [0.0, 0.5, 0.04, 0.52])], bbox=(0.5, 0.0, 1.0, 1.0), name="crop")
        reading_reconcile.apply([sheet, crop])
        assert sheet.content.structured_drawing_fields[0].value == "1349"


class TestSources:
    def test_vector_layer_wins(self):
        sheet = drawing([field("⌀4,5-0,2", [0.46, 0.39, 0.48, 0.46])], text="⌀4,5-0,2")
        layer = drawing([field("⌀4,5+0,2", [0.46, 0.39, 0.48, 0.46])], method="vector_text_layer",
                        name="vector")
        reading_reconcile.apply([sheet, layer])
        size = sheet.content.structured_drawing_fields[0]
        assert size.value == "⌀4,5+0,2" and size.agreement == VECTOR
        assert size.confidence < 0.95       # подтверждает только человек

    def test_far_vector_text_is_another_inscription(self):
        sheet = drawing([field("M6x0,75", [0.1, 0.1, 0.14, 0.12], category="thread")])
        layer = drawing([field("M6x1", [0.8, 0.8, 0.84, 0.82], category="thread")],
                        method="vector_text_layer", name="vector")
        reading_reconcile.apply([sheet, layer])
        assert sheet.content.structured_drawing_fields[0].value == "M6x0,75"

    def test_vector_digits_without_decimal_form_are_another_inscription(self):
        sheet = drawing([field("1,5", [0.30, 0.30, 0.33, 0.32])])
        layer = drawing([field("15", [0.31, 0.30, 0.33, 0.32]), field("1,5", [0.60, 0.60, 0.63, 0.62])],
                        method="vector_text_layer", name="vector")
        reading_reconcile.apply([sheet, layer])
        size = sheet.content.structured_drawing_fields[0]
        assert size.value == "1,5" and size.agreement == SINGLE

    def test_nearest_vector_field_is_taken(self):
        sheet = drawing([field("⌀4,5-0,2", [0.46, 0.39, 0.48, 0.46])])
        layer = drawing([field("⌀4,5+0,2", [0.49, 0.40, 0.51, 0.47]), field("⌀4,5-0,2", [0.46, 0.39, 0.48, 0.46])],
                        method="vector_text_layer", name="vector")
        reading_reconcile.apply([sheet, layer])
        assert sheet.content.structured_drawing_fields[0].value == "⌀4,5-0,2"

    def test_text_alone_does_not_change_the_value(self):
        sheet = drawing([field("1,5*", [0.9, 0.73, 0.92, 0.75])], text="15*")
        fragments = [sheet, ocr("15*")]
        reading_reconcile.apply(fragments)
        size = sheet.content.structured_drawing_fields[0]
        assert size.value == "1,5*" and size.agreement == CONFLICT

    def test_page_without_whole_sheet_is_left_as_is(self):
        one = drawing([field("⌀18", [0.1, 0.1, 0.2, 0.2])], bbox=(0.0, 0.0, 0.5, 0.5), name="a")
        two = drawing([field("⌀18", [0.2, 0.2, 0.4, 0.4])], bbox=(0.0, 0.0, 0.5, 0.5), name="b")
        reading_reconcile.apply([one, two])
        assert len(two.content.structured_drawing_fields) == 1
        assert one.content.structured_drawing_fields[0].agreement is None

    def test_notation_difference_is_agreement(self):
        sheet = drawing([field("⌀6,6−0,05", [0.16, 0.25, 0.18, 0.33])], text="⌀6,6-0.05")
        reading_reconcile.apply([sheet])
        size = sheet.content.structured_drawing_fields[0]
        assert size.agreement == AGREED and size.value == "⌀6,6−0,05"

    def test_ocr_layer_is_a_reading(self):
        sheet = drawing([field("⌀16,3*", [0.56, 0.25, 0.58, 0.32])])
        fragments = [sheet, ocr("Фаска\n⌀16,3*")]
        reading_reconcile.apply(fragments)
        assert sheet.content.structured_drawing_fields[0].readings[-1] == {"source": "ocr", "value": "⌀16,3*"}

    def test_two_different_text_variants_say_nothing(self):
        sheet = drawing([field("15*", [0.9, 0.73, 0.92, 0.75])], text="1,5*\n15*\n1.5*\n0,15*")
        reading_reconcile.apply([sheet])
        assert sheet.content.structured_drawing_fields[0].agreement == AGREED

    def test_list_of_values_is_not_one_inscription(self):
        sheet = drawing([field("M1, M2", [0.1, 0.1, 0.2, 0.2], category="thread")], text="M1.2")
        reading_reconcile.apply([sheet])
        thread = sheet.content.structured_drawing_fields[0]
        assert thread.value == "M1, M2" and thread.agreement == SINGLE

    def test_suffix_letters_are_not_a_disagreement(self):
        sheet = drawing([field("M18x1,5-8g", [0.62, 0.23, 0.64, 0.32], category="thread")],
                        text="Резьба M18x1,5-8g")
        reading_reconcile.apply([sheet])
        assert sheet.content.structured_drawing_fields[0].agreement == AGREED

    def test_text_duplicates_across_fragments_merge(self):
        sheet = drawing([field("Поз. 3", [0.30, 0.30, 0.34, 0.32], category="position")])
        crop = drawing([field("Поз. 3", [0.60, 0.30, 0.68, 0.32], category="position"),
                        field("Поз. 7", [0.10, 0.80, 0.18, 0.82], category="position")],
                       bbox=CROP, name="crop")
        reading_reconcile.apply([sheet, crop])
        assert [f.value for f in crop.content.structured_drawing_fields] == ["Поз. 7"]
        assert sheet.content.structured_drawing_fields[0].agreement == AGREED

    def test_crop_is_not_left_empty(self):
        sheet = drawing([field("70*2", [0.32, 0.12, 0.35, 0.14])], text="70+2")
        crop = drawing([field("70+2", [0.62, 0.11, 0.68, 0.13])], bbox=CROP, name="crop")
        reading_reconcile.apply([sheet, crop])
        kept = crop.content.structured_drawing_fields
        assert [f.value for f in kept] == ["70+2"] and kept[0].agreement == AGREED


class TestFlagOff:
    def test_contract_without_new_keys(self):
        dumped = field("⌀315", [0.1, 0.1, 0.2, 0.2]).model_dump(mode="json")
        assert "readings" not in dumped and "agreement" not in dumped

    def test_record_without_mark(self, monkeypatch):
        monkeypatch.setattr(settings, "DRAWING_READINGS_RECONCILED", False)
        sheet, crop = flange()
        reading_reconcile.apply([sheet, crop])
        assert "расходятся" not in drawing_record.compose_fragment(sheet)

    def test_enrich_does_not_reconcile_without_flag(self, monkeypatch):
        from core.drawing_processor import DrawingProcessor

        monkeypatch.setattr(settings, "DRAWING_READINGS_RECONCILED", False)
        sheet, crop = flange()
        processor = DrawingProcessor(storage=object(), qwen=object())
        processor._pages_read_whole = lambda metadata: frozenset()
        processor._sheet_drawings = lambda fragments, metadata, page_fields=None: []
        out = processor.enrich_fragments([sheet, crop], {})
        assert [f.value for f in out[1].content.structured_drawing_fields] == ["⌀280+0,3", "⌀18"]

    def test_enrich_reconciles_with_flag(self, monkeypatch):
        from core.drawing_processor import DrawingProcessor

        monkeypatch.setattr(settings, "DRAWING_READINGS_RECONCILED", True)
        sheet, crop = flange()
        processor = DrawingProcessor(storage=object(), qwen=object())
        processor._pages_read_whole = lambda metadata: frozenset()
        processor._sheet_drawings = lambda fragments, metadata, page_fields=None: []
        out = processor.enrich_fragments([sheet, crop], {})
        assert [f.value for f in out[1].content.structured_drawing_fields] == ["⌀18"]


@pytest.mark.parametrize("value, plausible", [
    ("70*2", False), ("2Nº1", False), ("⌀18+0,25 0,10", False),
    ("⌀18+0,25+0,10", True), ("⌀280±0,3", True), ("8 отв.", True),
])
def test_plausible_notation(value, plausible):
    assert reading_reconcile.plausible(value) is plausible


class TestItemReferences:
    """Пилот 17.11: «п.10» в тексте и полях листа прочитано латинской «n»."""

    def test_latin_n_item_reference_becomes_cyrillic(self):
        sheet = drawing([field("n.6", [0.1, 0.1, 0.12, 0.12], category="position")],
                        text="Установить по n.10, n. 11\nRun.5 и n.a")
        reading_reconcile.apply([sheet])
        assert sheet.content.text == "Установить по п.10, п. 11\nRun.5 и n.a"
        assert sheet.content.structured_drawing_fields[0].value == "п.6"

    def test_other_fragments_untouched(self):
        text = ocr("см. n.3")
        reading_reconcile.apply([text])
        assert text.content.text == "см. n.3"
