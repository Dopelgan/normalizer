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
    assert typing_.classify("3", "callout", "assembly").kind == typing_.POSITION
    # На детали число на выноске — размер.
    assert typing_.classify("3", "callout", "detail").kind == typing_.DIMENSION


def test_scheme_designators_are_positions_not_sizes():
    for value in ("K20, K21", "HL1, HL2", "M1, M2", "QF3..QF11"):
        typed = typing_.classify(value, "position", "scheme")
        assert typed.kind == typing_.POSITION and typed.number == value


def test_words_in_brackets_are_not_a_formula():
    assert typing_.classify("Датчик ДП-003Т (с кабелем)", "size").kind == typing_.TEXT
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


def test_variable_sizes_of_a_parametric_sheet():
    for value, expr in (("X", "X"), ("L1", "L1"), ("Z+10", "Z+10"), ("L*", "L")):
        typed = typing_.classify(value, "size")
        assert typed.kind == typing_.DIMENSION and typed.dimension["formula"]["expr"] == expr
    assert typing_.parse_dimension("L±10")["tolerance"]["upper"] == 10
    assert typing_.parse_dimension("R1")["kind"] == "radius"
    assert typing_.parse_dimension("M8")["kind"] == "thread"


@pytest.mark.parametrize("value", ["A", "Б", "B", "Д"])
def test_view_letter_is_never_a_dimension(value):
    assert typing_.classify(value, "size").kind == typing_.TEXT


@pytest.mark.parametrize("value", ["380 В", "4,№91", "10ABC70XY001 (ЗН)"])
def test_callout_number_with_word_is_text(value):
    assert typing_.classify(value, "callout").kind == typing_.TEXT


@pytest.mark.parametrize("value", ["15±0,1", "⌀8,5", "45°", "20*"])
def test_callout_with_dimension_notation_stays_dimension(value):
    assert typing_.classify(value, "callout", "detail").kind == typing_.DIMENSION


def test_scheme_numbers_and_designators_are_positions():
    assert typing_.classify("14", "callout", "scheme").kind == typing_.POSITION
    typed = typing_.classify("M1, M2", "thread", "scheme")
    assert typed.kind == typing_.POSITION and typed.number == "M1, M2"
    assert typing_.classify("M20", "thread", "detail").kind == typing_.DIMENSION


@pytest.mark.parametrize("value", ["14. 10.16 (рукописно)", "02.2022", "19.10.26; 29.09.25 (рукописно)"])
def test_signing_date_is_title_not_note_or_size(value):
    assert typing_.is_date(value)
    assert typing_.classify(value, "note").kind == typing_.TITLE
    assert typing_.classify(value, "size").kind == typing_.TITLE


@pytest.mark.parametrize("value", ["1.5", "12.50", "15-0,1"])
def test_decimals_are_not_dates(value):
    assert not typing_.is_date(value)


def test_number_inside_title_block_is_not_position_or_size():
    assert typing_.classify("1", "position", "assembly", in_title=True).kind == typing_.TITLE
    assert typing_.classify("1", "callout", "detail", in_title=True).kind == typing_.TITLE
    assert typing_.classify("1", "position", "assembly").kind == typing_.POSITION


def test_footnote_is_separate_note_kind():
    typed = typing_.classify("1) - нормативный документ по договору поставки.", "note")
    assert typed.kind == typing_.NOTE and typed.footnote and typed.note_number == 1
    assert typed.note_text.startswith("нормативный")
    assert not typing_.classify("1. Размеры для справок.", "note").footnote
    assert typing_.footnote_marks("контроль по ГОСТ 1-2018 1) - 100%.") == [1]
    assert typing_.footnote_marks("1) - документ") == []


@pytest.mark.parametrize("value, expected", [
    ("- визуальный контроль 100%.", True),
    ("валов h14, отверстий H14.", True),
    ("Нормы оценки по ГОСТ 1-2009; уровень качества - D.", True),
    ("4 Для фиксации труб использовать скобы.", False),
    ("Технические требования", False),
    ("СБ210/СП-02-100П-В-4 IP67 ГОСТ 1-2009", False),
    ("МЕТРОЛОГИЧЕСКАЯ ЭКСПЕРТИЗА ПРОВЕДЕНА Петров И. И.", False),
])
def test_note_continuation_candidates(value, expected):
    assert typing_.continues_note(value) is expected


def test_transcript_notes_skip_date_and_footnote_and_table_rows():
    lines = ["Технические требования", "6. Маркировать на бирке обозначение чертежа.",
             "10.02.2022 (рукописно)", "см.таблицу\t1:1", "-01\t247\t321",
             "7. Покрытие по ГОСТ 9.032.", "- грунт в один слой;", "1) - документ по договору."]
    notes = typing_.transcript_notes(lines)
    assert [n for n, _ in notes] == [6, 7]
    assert notes[0][1] == "Маркировать на бирке обозначение чертежа."
    assert notes[1][1].endswith("- грунт в один слой;")
