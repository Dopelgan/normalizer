"""
Правила полей чертежа под своими флагами: классы (C11), позиции по
спецификации (C12), ряд шероховатости ГОСТ 2789 (S1), общий знак
шероховатости в углу листа (C14). Значения синтетические.
"""

import copy

import pytest

from core.config import settings
from core.models.contract import (
    Content, DimensionLink, DrawingField, Fragment, Position, Provenance, TableData,
)
from core.providers import sense_rules

FLAGS = ("DRAWING_SENSE_CLASS_RULES", "DRAWING_SENSE_SPEC_POSITIONS",
         "DRAWING_SENSE_RA_SERIES", "DRAWING_CORNER_ROUGHNESS_V2")


def field(value, bbox, category="size", confidence=0.9, **extra):
    return DrawingField(category=category, value=value, bbox=bbox, confidence=confidence, **extra)


def link(label="элемент"):
    return DimensionLink(target_label=label, target_bbox=[0.5, 0.5, 0.6, 0.6], method="vlm",
                         confidence=0.6)


def drawing(fields, bbox=(0.0, 0.0, 1.0, 1.0), name="sheet", text=None, page=1, **payload):
    return Fragment(
        fragment_id=name, type="drawing",
        content=Content(text=text, structured_drawing_fields=fields, structured_payload=dict(payload)),
        position=Position(page=page, bbox=list(bbox)), confidence=0.0, completeness=0.0,
        provenance=Provenance(method="vision_plus_vlm", strategy_level=7, source="drawing"),
    )


def table(headers, rows, page=2):
    return Fragment(
        fragment_id="spec", type="table",
        content=Content(table_data=TableData(headers=headers, rows=rows)),
        position=Position(page=page, bbox=[0, 0, 1, 1]), confidence=0.9, completeness=1.0,
        provenance=Provenance(method="vision_plus_vlm", strategy_level=7, source="table"),
    )


def categories(fragment):
    return [(f.value, f.category) for f in fragment.content.structured_drawing_fields]


@pytest.fixture
def flags_off(monkeypatch):
    for name in FLAGS:
        monkeypatch.setattr(settings, name, False)


def flag_on(monkeypatch, name):
    for other in FLAGS:
        monkeypatch.setattr(settings, other, other == name)


# --- C11 --------------------------------------------------------------------

def class_sheet():
    return drawing([
        field("2 места", [0.10, 0.10, 0.14, 0.12], category="callout"),
        field("4 отв.", [0.20, 0.10, 0.24, 0.12], category="position"),
        field("4 отв. ⌀6", [0.30, 0.10, 0.36, 0.12], category="size"),
        field("30°±2°", [0.40, 0.40, 0.45, 0.42], category="tolerance"),
        field("±0,1", [0.50, 0.40, 0.53, 0.42], category="tolerance"),
        field("№2", [0.60, 0.40, 0.62, 0.42], category="position"),
        field("5", [0.70, 0.40, 0.71, 0.42], category="position"),
    ], sheet_type="detail")


def test_class_rules(monkeypatch):
    flag_on(monkeypatch, "DRAWING_SENSE_CLASS_RULES")
    sheet = class_sheet()
    stats = sense_rules.apply([sheet])
    assert categories(sheet) == [
        ("2 места", "note"), ("4 отв.", "note"), ("4 отв. ⌀6", "size"),
        ("30°±2°", "size"), ("±0,1", "tolerance"), ("№2", "callout"), ("5", "position"),
    ]
    assert stats["class_count_note"] == 2 and stats["class_pm_size"] == 1 and stats["class_weld_mark"] == 1


@pytest.mark.parametrize("value, category", [
    ("3 Допускается применять любой другой металлопрокат из стали 12Х18Н10Т по ГОСТ 5632-2014.", "note"),
    ("4. Материал заменять только по согласованию с разработчиком", "note"),
    ("3 ГОСТ 19904-90 12Х18Н10Т ГОСТ 5582-75", "material"),
    ("2 Лист Б-ПН-2 ГОСТ 19903-74 12Х18Н10Т ГОСТ 5582-75", "material"),
    ("Круг 12Х18Н10Т ГОСТ 5949-75", "material"),
])
def test_numbered_requirement_in_material_is_note(monkeypatch, value, category):
    flag_on(monkeypatch, "DRAWING_SENSE_CLASS_RULES")
    sheet = drawing([field(value, [0.5, 0.7, 0.9, 0.72], category="material")], sheet_type="detail")
    sense_rules.apply([sheet])
    assert categories(sheet) == [(value, category)]


def test_pm_twin_keeps_one_size(monkeypatch):
    flag_on(monkeypatch, "DRAWING_SENSE_CLASS_RULES")
    sheet = drawing([
        field("30°±2°", [0.40, 0.40, 0.45, 0.42], category="size"),
        field("30°±2°", [0.40, 0.40, 0.45, 0.42], category="tolerance", confidence=0.5),
        field("R5", [0.1, 0.1, 0.12, 0.12], category="radius"),
    ])
    stats = sense_rules.apply([sheet])
    assert categories(sheet) == [("30°±2°", "size"), ("R5", "radius")]
    assert stats["class_pm_merged"] == 1


def test_class_rules_off_changes_nothing(flags_off):
    sheet = class_sheet()
    before = copy.deepcopy(sheet)
    stats = sense_rules.apply([sheet])
    assert sheet == before
    assert not [k for k in stats if k.startswith("class_")]


# --- C12 --------------------------------------------------------------------

SPEC = (["Поз.", "Обозначение", "Наименование", "Кол."],
        [["1", "АБВ.001", "Корпус", "1"], ["2", "АБВ.002", "Крышка", "1"],
         ["3", "АБВ.003", "Втулка", "2"], ["5", "", "Болт М6", "4"],
         ["", "", "Стандартные изделия", ""]])


def assembly(fields, **payload):
    return drawing(fields, sheet_type="assembly", **payload)


def test_spec_positions_from_table_fragment_and_sheet_table():
    assert sense_rules.spec_positions([table(*SPEC)]) == {1, 2, 3, 5}
    sheet = drawing([], tables=[{"kind": "table", "headers": [], "rows": [list(SPEC[0])] + SPEC[1]}])
    assert sense_rules.spec_positions([sheet]) == {1, 2, 3, 5}
    # Одна строка — не спецификация; таблица без «Поз.» — тоже.
    assert sense_rules.spec_positions([table(SPEC[0], SPEC[1][:1])]) is None
    assert sense_rules.spec_positions([table(["Обозначение", "Кол."], [["А", "1"], ["Б", "2"]])]) is None


def spec_sheet():
    return assembly([
        field("2", [0.30, 0.20, 0.31, 0.22], category="position"),      # в спецификации
        field("7", [0.40, 0.20, 0.41, 0.22], category="position"),      # нет -> выноска
        field("9", [0.45, 0.20, 0.46, 0.22], category="position", link=link()),  # нет, связь -> размер
        field("5", [0.50, 0.20, 0.51, 0.22], category="callout"),       # в спецификации, на выноске
        field("1", [0.55, 0.20, 0.56, 0.22], category="size"),          # без выноски — не трогается
        field("8", [0.60, 0.20, 0.61, 0.22], category="callout"),       # нет в спецификации
        field("№1", [0.70, 0.20, 0.72, 0.22], category="callout"),      # метка шва
    ])


def test_spec_positions_rule(monkeypatch):
    flag_on(monkeypatch, "DRAWING_SENSE_SPEC_POSITIONS")
    sheet = spec_sheet()
    stats = sense_rules.apply([sheet, table(*SPEC)])
    assert categories(sheet) == [
        ("2", "position"), ("7", "callout"), ("9", "size"), ("5", "position"),
        ("1", "size"), ("8", "callout"), ("№1", "callout"),
    ]
    fields = sheet.content.structured_drawing_fields
    assert fields[1].needs_review and fields[2].needs_review
    assert stats["spec_position_dropped"] == 2 and stats["spec_position_added"] == 1


def test_spec_positions_without_spec_do_nothing(monkeypatch):
    flag_on(monkeypatch, "DRAWING_SENSE_SPEC_POSITIONS")
    sheet = spec_sheet()
    before = copy.deepcopy(sheet)
    sense_rules.apply([sheet])
    assert sheet == before


def test_spec_positions_incomplete_spec_does_nothing(monkeypatch):
    # Из номеров 1…5 прочитаны три (60 % < 80 %): строки спецификации
    # потеряны, «нет в спецификации» ничего не значит.
    flag_on(monkeypatch, "DRAWING_SENSE_SPEC_POSITIONS")
    sheet = spec_sheet()
    before = copy.deepcopy(sheet)
    rows = [row for row in SPEC[1] if row[0] != "3"]
    stats = sense_rules.apply([sheet, table(SPEC[0], rows)])
    assert sheet == before
    assert not stats.get("spec_position_dropped") and not stats.get("spec_position_added")


def test_spec_positions_only_on_assembly(monkeypatch):
    flag_on(monkeypatch, "DRAWING_SENSE_SPEC_POSITIONS")
    sheet = drawing([field("7", [0.4, 0.2, 0.41, 0.22], category="position")], sheet_type="detail")
    sense_rules.apply([sheet, table(*SPEC)])
    assert categories(sheet) == [("7", "position")]


def test_spec_positions_off(flags_off):
    sheet = spec_sheet()
    before = copy.deepcopy(sheet)
    sense_rules.apply([sheet, table(*SPEC)])
    assert sheet == before


# --- S1 ---------------------------------------------------------------------

@pytest.mark.parametrize("raw, fixed", [
    # Не из полного ряда — правится всегда.
    ("Ra 08", "Ra 0,8"), ("Ra 04", "Ra 0,4"), ("Rz 08", "Rz 0,8"),
])
def test_ra_series_fix(raw, fixed):
    assert sense_rules.ra_series_fix(raw) == fixed
    assert sense_rules.ra_series_fix(raw, fine_sheet=True) == fixed


@pytest.mark.parametrize("raw, fixed", [
    ("Ra 16", "Ra 1,6"), ("Ra 32", "Ra 3,2"), ("Ra 63", "Ra 6,3"), ("Ra 125", "Ra 12,5"),
    ("√Ra 63 (√)", "√Ra 6,3 (√)"),
])
def test_ra_full_series_fixed_only_on_fine_sheet(raw, fixed):
    # Значения полного ряда Ra верны сами по себе: правка — только при
    # другой шероховатости листа не грубее 12,5.
    assert sense_rules.ra_series_fix(raw) is None
    assert sense_rules.ra_series_fix(raw, fine_sheet=True) == fixed


@pytest.mark.parametrize("value", [
    "Ra 25", "Ra 50", "Ra 100", "Ra 1,6", "Ra 12",  # верные или неоднозначные
    "Ra 160", "Ra 200", "Ra 320", "Ra 400",          # «1,60» и прочее — никогда
    "Rz 16", "Rz 20", "Rz 40", "Rz 320", "Rz 125",   # полный ряд Rz
    "R 16", "⌀16",
])
def test_ra_series_keeps_valid(value):
    assert sense_rules.ra_series_fix(value) is None
    assert sense_rules.ra_series_fix(value, fine_sheet=True) is None


def ra_sheet(other="Ra 3,2"):
    return drawing([
        field("Ra 16", [0.2, 0.6, 0.24, 0.62], category="roughness", link=link()),
        field("Ra 25", [0.3, 0.6, 0.34, 0.62], category="roughness"),
        field(other, [0.5, 0.6, 0.54, 0.62], category="roughness"),
        field("16", [0.4, 0.6, 0.42, 0.62], category="size"),
    ])


def test_ra_series_rule(monkeypatch):
    flag_on(monkeypatch, "DRAWING_SENSE_RA_SERIES")
    sheet = ra_sheet()
    stats = sense_rules.apply([sheet])
    fixed, valid, _, size = sheet.content.structured_drawing_fields
    assert fixed.value == "Ra 1,6" and fixed.needs_review and fixed.link is not None
    assert [r["value"] for r in fixed.readings] == ["Ra 16", "Ra 1,6"]
    assert fixed.readings[1]["source"] == sense_rules.SENSE
    assert valid.value == "Ra 25" and valid.readings is None
    assert size.value == "16"
    assert stats["ra_series"] == 1


def test_ra_series_rule_keeps_full_series_on_coarse_sheet(monkeypatch):
    # Тонкой шероховатости на листе нет: «Ra 16» — значение ряда, остаётся.
    flag_on(monkeypatch, "DRAWING_SENSE_RA_SERIES")
    sheet = ra_sheet(other="Ra 50")
    stats = sense_rules.apply([sheet])
    assert sheet.content.structured_drawing_fields[0].value == "Ra 16"
    assert not stats.get("ra_series")


def test_ra_series_off(flags_off):
    sheet = ra_sheet()
    sense_rules.apply([sheet])
    assert sheet.content.structured_drawing_fields[0].value == "Ra 16"


# --- C14 --------------------------------------------------------------------

CORNER_FIELD = [0.90, 0.02, 0.97, 0.06]      # правый верхний угол листа
VIEW_IN_CORNER = (0.80, 0.0, 1.0, 0.3)       # вырезка вида, накрывающая угол


@pytest.mark.parametrize("raw, value", [
    ("Ra 6,3(V)", "Ra 6,3 (√)"), ("√ (√)", "√ (√)"), ("Ra 6,3 (√)", "Ra 6,3 (√)"),
])
def test_general_roughness_record(monkeypatch, raw, value):
    flag_on(monkeypatch, "DRAWING_CORNER_ROUGHNESS_V2")
    sheet = drawing([field(raw, [0.5, 0.5, 0.55, 0.52], category="note")], sheet_type="detail")
    sense_rules.apply([sheet])
    (got,) = sheet.content.structured_drawing_fields
    assert (got.value, got.category) == (value, "roughness")


def test_corner_roughness_gets_check_mark(monkeypatch):
    flag_on(monkeypatch, "DRAWING_CORNER_ROUGHNESS_V2")
    sheet = drawing([
        field("Ra 6,3", CORNER_FIELD, category="roughness"),
        field("Ra 3,2", [0.3, 0.3, 0.34, 0.32], category="roughness"),
    ], sheet_type="detail", layout={"frame": [0.0, 0.0, 1.0, 1.0]})
    stats = sense_rules.apply([sheet])
    corner, inner = sheet.content.structured_drawing_fields
    assert corner.value == "Ra 6,3 (√)" and not corner.needs_review
    assert inner.value == "Ra 3,2"
    assert stats["corner_marked"] == 1


def test_unplaced_general_roughness_goes_to_corner(monkeypatch):
    flag_on(monkeypatch, "DRAWING_CORNER_ROUGHNESS_V2")
    frame = [0.02, 0.02, 0.98, 0.98]
    sheet = drawing([
        field("Ra 6,3 (√)", [0.0, 0.0, 1.0, 1.0], category="roughness"),
        field("Ra 1,6 (√)", [0.3, 0.3, 0.34, 0.32], category="roughness"),
    ], sheet_type="detail", layout={"frame": frame})
    stats = sense_rules.apply([sheet])
    unplaced, placed = sheet.content.structured_drawing_fields
    assert unplaced.bbox == sense_rules.corner_sign_box(sense_rules.corner_box(frame))
    assert placed.bbox == [0.3, 0.3, 0.34, 0.32]
    assert stats["corner_placed"] == 1


def test_unplaced_general_roughness_stays_with_flag_off(flags_off):
    sheet = drawing([field("Ra 6,3 (√)", [0.0, 0.0, 1.0, 1.0], category="roughness")], sheet_type="detail")
    sense_rules.apply([sheet])
    assert sheet.content.structured_drawing_fields[0].bbox == [0.0, 0.0, 1.0, 1.0]


def test_roughness_inside_view_is_not_general(monkeypatch):
    flag_on(monkeypatch, "DRAWING_CORNER_ROUGHNESS_V2")
    sheet = drawing([field("Ra 6,3", CORNER_FIELD, category="roughness")], sheet_type="detail")
    view = drawing([field("Ra 6,3", [0.5, 0.1, 0.6, 0.2], category="roughness")],
                   bbox=VIEW_IN_CORNER, name="view")
    sense_rules.apply([sheet, view])
    assert sheet.content.structured_drawing_fields[0].value == "Ra 6,3"
    assert view.content.structured_drawing_fields[0].value == "Ra 6,3"


def test_corner_not_promoted_on_assembly_or_outside_corner(monkeypatch):
    flag_on(monkeypatch, "DRAWING_CORNER_ROUGHNESS_V2")
    sheet = drawing([field("Ra 6,3", CORNER_FIELD, category="roughness")], sheet_type="assembly")
    middle = drawing([field("Ra 6,3", [0.5, 0.5, 0.55, 0.52], category="roughness")], sheet_type="detail")
    sense_rules.apply([sheet])
    sense_rules.apply([middle])
    assert sheet.content.structured_drawing_fields[0].value == "Ra 6,3"
    assert middle.content.structured_drawing_fields[0].value == "Ra 6,3"


def test_two_corner_values_are_left_alone(monkeypatch):
    flag_on(monkeypatch, "DRAWING_CORNER_ROUGHNESS_V2")
    sheet = drawing([
        field("Ra 6,3", CORNER_FIELD, category="roughness"),
        field("Ra 3,2", [0.80, 0.02, 0.85, 0.06], category="roughness"),
    ], sheet_type="detail")
    sense_rules.apply([sheet])
    assert [f.value for f in sheet.content.structured_drawing_fields] == ["Ra 6,3", "Ra 3,2"]


def test_corner_from_sheet_text(monkeypatch):
    flag_on(monkeypatch, "DRAWING_CORNER_ROUGHNESS_V2")
    sheet = drawing([field("⌀20", [0.3, 0.3, 0.33, 0.32])], sheet_type="detail",
                    text="Ra 6,3 (√)\n1. Острые кромки притупить")
    stats = sense_rules.apply([sheet])
    added = sheet.content.structured_drawing_fields[-1]
    assert (added.value, added.category) == ("Ra 6,3 (√)", "roughness")
    assert added.bbox[0] >= 0.7 and added.bbox[3] <= 0.12
    assert stats["corner_from_text"] == 1


def test_corner_off(flags_off):
    sheet = drawing([field("Ra 6,3", CORNER_FIELD, category="roughness")], sheet_type="detail",
                    text="Ra 6,3 (√)")
    before = copy.deepcopy(sheet)
    stats = sense_rules.apply([sheet])
    assert sheet == before
    assert not [k for k in stats if k.startswith(("corner_", "ra_", "spec_"))]
