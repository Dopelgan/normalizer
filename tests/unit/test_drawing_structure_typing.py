"""Структура листа, S4: тип надписи и разбор величины (синтетические записи)."""

import pytest

from core.drawing_structure import typing as typing_


@pytest.mark.parametrize("value", ["п4", "п.5", "п. 8", "n.10", "пп.3, 4"])
def test_item_reference_is_note_ref_not_size(value):
    typed = typing_.classify(value, "size")
    assert typed.kind == typing_.NOTE_REF
    assert typed.note_refs


def test_double_reference_numbers():
    assert typing_.note_refs("пп.3, 4") == [3, 4]


@pytest.mark.parametrize("value, count", [("4 отв.", "4"), ("N отв.", "N"), ("4N01", "4"),
                                          ("4, Nот", "4"), ("4 места", "4")])
def test_hole_counts(value, count):
    typed = typing_.classify(value, "callout")
    assert typed.kind == typing_.COUNT and typed.count == count


def test_diameter_with_two_sided_tolerance():
    dim = typing_.parse_dimension("⌀245-0,20+0,05")
    assert dim["kind"] == "diameter" and dim["nominal"] == 245
    assert dim["tolerance"] == {"upper": 0.05, "lower": -0.2, "source": "explicit"}


def test_plus_minus_and_reference_marker():
    dim = typing_.parse_dimension("18*")
    assert dim["nominal"] == 18 and dim["reference"] is True
    dim = typing_.parse_dimension("15±0,1")
    assert dim["tolerance"]["upper"] == 0.1 and dim["tolerance"]["lower"] == -0.1


def test_fit_notation():
    dim = typing_.parse_dimension("⌀30H7")
    assert dim["tolerance"]["fit"] == "H7" and dim["tolerance"]["source"] == "fit_notation"


def test_parametric_size_keeps_formula():
    dim = typing_.parse_dimension("(X+38)±0,2")
    assert dim["formula"] == {"expr": "X+38", "variables": ["X"]}
    assert dim["nominal"] is None
    assert dim["tolerance"]["upper"] == 0.2


def test_thread_with_pitch_and_field():
    dim = typing_.parse_dimension("М20×1,5-6g", "thread")
    assert dim["kind"] == "thread" and dim["nominal"] == 20 and dim["pitch"] == 1.5
    assert dim["tolerance"]["fit"] == "6g"


def test_diameter_with_hole_count_in_one_inscription():
    typed = typing_.classify("⌀8,5 4 отв.", "size")
    assert typed.kind == typing_.DIMENSION
    assert typed.dimension["nominal"] == 8.5 and typed.count == "4"


def test_chamfer_radius_angle():
    assert typing_.parse_dimension("1×45°")["kind"] == "chamfer"
    assert typing_.parse_dimension("R1")["kind"] == "radius"
    assert typing_.parse_dimension("30°")["kind"] == "angle"


def test_size_with_item_reference_keeps_both():
    typed = typing_.classify("15 п4", "size")
    assert typed.kind == typing_.DIMENSION and typed.note_refs == [4]
    assert typed.dimension["nominal"] == 15


def test_balloon_on_callout_is_position():
    assert typing_.classify("3", "callout").kind == typing_.POSITION
    assert typing_.classify("12", "position").number == "12"


def test_view_title_and_surface_callout():
    typed = typing_.classify("Б (4:1)", "callout")
    assert typed.kind == typing_.VIEW_TITLE and typed.view == ("Б", "detail", "4:1")
    surface = typing_.classify("Лицевая сторона п.8", "callout")
    assert surface.kind == typing_.SURFACE and surface.note_refs == [8]
    assert surface.label == "Лицевая сторона"


def test_numbered_note_and_marker():
    typed = typing_.classify("1. * Размеры для справок.", "note")
    assert typed.kind == typing_.NOTE and typed.note_number == 1 and typed.marker == "*"


def test_roughness_symbol():
    typed = typing_.classify("Ra 1,6", "roughness")
    assert typed.kind == typing_.SYMBOL and typed.symbol["value"] == "Ra 1,6"


def test_transcript_notes_join_continuation():
    lines = ["Технические требования", "1. * Размеры для справок.",
             "2. Неуказанные предельные отклонения размеров:", "валов h14, отверстий H14.", "Крышка"]
    notes = typing_.transcript_notes(lines)
    assert [n for n, _ in notes] == [1, 2]
    assert "валов h14" in notes[1][1]
