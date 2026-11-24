"""
Правила здравого смысла для полей чертежа (DRAWING_SENSE_RULES).

Обведённая толщина листа «S0,5*» читается моделью тремя способами: «50,5*»
(уверенно и со связью «радиус скругления»), «S0,5*» (неуверенно) и на
вырезке вида — шероховатостью «Ra 0,5*». Сведение оставляло самое
уверенное, «50,5*». Штамп говорит «Лист 0,5 ГОСТ …»: значит, это «S0,5*».
Значения здесь синтетические.
"""

import pytest

from core.config import settings
from core.models.contract import Content, DimensionLink, DrawingField, Fragment, Position, Provenance
from core.providers import reading_reconcile, sense_rules


def field(value, bbox, category="size", confidence=0.9, **extra):
    return DrawingField(category=category, value=value, bbox=bbox, confidence=confidence, **extra)


def link(label, confidence=0.6):
    return DimensionLink(target_label=label, target_bbox=[0.5, 0.5, 0.6, 0.6], method="vlm",
                         confidence=confidence)


def drawing(fields, bbox=(0.0, 0.0, 1.0, 1.0), name="sheet", text=None):
    return Fragment(
        fragment_id=name, type="drawing",
        content=Content(text=text, structured_drawing_fields=fields, structured_payload={}),
        position=Position(page=1, bbox=list(bbox)), confidence=0.0, completeness=0.0,
        provenance=Provenance(method="vision_plus_vlm", strategy_level=7, source="drawing"),
    )


# Вырезка вида — правая верхняя четверть листа.
VIEW = (0.5, 0.0, 1.0, 0.5)
MARK = [0.80, 0.08, 0.84, 0.11]          # место «S0,5*» на листе
MARK_IN_VIEW = [0.60, 0.16, 0.68, 0.22]  # то же место в долях вырезки


def three_readings():
    sheet = drawing([
        field("Лист 0,5 ГОСТ 19904-90", [0.70, 0.90, 0.95, 0.93], category="material"),
        field("50,5*", MARK, link=link("радиус скругления")),
        field("S0,5*", [0.80, 0.08, 0.845, 0.11], confidence=0.4, agreement="conflict",
              readings=[{"source": "sheet", "value": "S0,5*"}]),
        field("R1", [0.30, 0.30, 0.32, 0.32], category="radius", link=link("радиус скругления")),
        field("⌀18", [0.40, 0.40, 0.43, 0.42], link=link("отверстие")),
        field("Ra 6,3", [0.20, 0.60, 0.24, 0.62], category="roughness", confidence=0.7),
    ], text="Лист 0,5 ГОСТ 19904-90")
    view = drawing([
        field("Ra 0,5*", MARK_IN_VIEW, category="roughness", confidence=0.4, agreement="conflict",
              readings=[{"source": "crop", "value": "Ra 0,5*"}, {"source": "crop_reread", "value": "S0,5*"}]),
        field("⌀6", [0.10, 0.80, 0.14, 0.84]),
    ], bbox=VIEW, name="view")
    return sheet, view


def all_fields(*fragments):
    return [f for fragment in fragments for f in fragment.content.structured_drawing_fields]


@pytest.mark.parametrize("reconciled", [False, True])
def test_three_readings_end_as_one_thickness(reconciled):
    sheet, view = three_readings()
    if reconciled:
        reading_reconcile.apply([sheet, view])
    stats = sense_rules.apply([sheet, view])
    fields = all_fields(sheet, view)
    marks = [f for f in fields if f.value == "S0,5*"]
    assert len(marks) == 1
    (mark,) = marks
    # Рамка — прочтения листа, связь «радиус» снята, прежнее прочтение видно.
    assert mark.bbox == MARK and mark.link is None
    assert "50,5*" in [r["value"] for r in mark.readings]
    assert mark.needs_review is True and mark.agreement == "conflict"
    assert mark.confidence >= 0.6     # не прячется в просмотре
    assert not [f for f in fields if "50,5" in f.value or "Ra 0,5" in f.value]
    assert stats["thickness"] >= 1 and stats["roughness_dropped"] == 1
    # Запись листа называет расхождение.
    assert "50,5*" in reading_reconcile.conflict_note(mark.model_dump())


def test_true_links_and_true_roughness_survive():
    sheet, view = three_readings()
    sense_rules.apply([sheet, view])
    by_value = {f.value: f for f in all_fields(sheet, view)}
    assert by_value["R1"].link is not None
    assert by_value["⌀18"].link is not None
    rough = by_value["Ra 6,3"]
    assert rough.category == "roughness" and rough.readings is None and rough.confidence == 0.7


def test_radius_link_on_diameter_is_dropped_value_kept():
    fields = [field("⌀18", [0.4, 0.4, 0.43, 0.42], link=link("радиус скругления")),
              field("12", [0.5, 0.5, 0.52, 0.52], link=link("диаметр вала")),
              field("M8", [0.6, 0.6, 0.62, 0.62], link=link("резьбовое отверстие")),
              field("4×R2", [0.7, 0.7, 0.72, 0.72], link=link("радиус"))]
    assert sense_rules.check_links(fields) == 2
    assert [f.link is not None for f in fields] == [False, False, True, True]
    assert [f.value for f in fields] == ["⌀18", "12", "M8", "4×R2"]


def test_check_links_on_sheet_pass_dicts():
    fields = [{"value": "50,5*", "link": {"target_label": "радиус скругления"}},
              {"value": "R1", "link": {"target_label": "радиус скругления"}}]
    assert sense_rules.check_links(fields) == 1
    assert "link" not in fields[0] and "link" in fields[1]


@pytest.mark.parametrize("texts, expected", [
    (["Лист 0,5 ГОСТ 19904-90"], "0,5"),
    (["Лист Б-ПН-1,5×1000×2000 ГОСТ 19903-74"], "1,5"),
    (["Лента 0,8-Н ГОСТ 503-81"], "0,8"),
    (["S2 ГОСТ 19903"], "2"),
    (["Лист 2 Листов 5"], None),          # номер листа штампа, не толщина
    (["Сталь 45 ГОСТ 1050-88"], None),
    (["Лист 0,5 ГОСТ 19904-90", "Лист 1 ГОСТ 19904-90"], None),   # разные — не гадать
])
def test_thickness_from_material(texts, expected):
    assert sense_rules.thickness(texts) == expected


def test_misread_thickness_rules():
    assert sense_rules.misread_thickness("50,5*", "0,5") == "S0,5*"
    # Голый «50,5» без звёздочки — правдоподобный размер (правка ревью 1).
    assert sense_rules.misread_thickness("50.5", "0,5") is None
    assert sense_rules.misread_thickness("55", "5") is None
    assert sense_rules.misread_thickness("55*", "5") == "S5*"
    assert sense_rules.misread_thickness("52", "2") is None      # размер 52 правдоподобен
    assert sense_rules.misread_thickness("52*", "2") == "S2*"
    assert sense_rules.misread_thickness("150,5*", "0,5") is None


def test_roughness_with_star_away_from_sizes_becomes_size():
    sheet = drawing([field("Лист 0,5 ГОСТ 19904-90", [0.7, 0.9, 0.95, 0.93], category="material"),
                     field("Ra 0,5*", [0.2, 0.2, 0.24, 0.22], category="roughness")])
    sense_rules.apply([sheet])
    (rough,) = [f for f in sheet.content.structured_drawing_fields if f.category != "material"]
    assert rough.category == "size" and rough.value == "S0,5*"
    assert [r["value"] for r in rough.readings] == ["Ra 0,5*", "S0,5*"]


def test_without_thickness_size_is_untouched():
    sheet = drawing([field("Сталь 45 ГОСТ 1050-88", [0.7, 0.9, 0.95, 0.93], category="material"),
                     field("50,5*", MARK, link=link("радиус скругления"))])
    sense_rules.apply([sheet])
    size = sheet.content.structured_drawing_fields[1]
    assert size.value == "50,5*" and size.readings is None
    assert size.link is None      # вид связи проверяется и без толщины


def test_flag_off_does_not_call_rules(temp_storage, monkeypatch):
    """С выключенным флагом обработчик не зовёт правила вовсе: вывод прежний."""
    from core.drawing_processor import DrawingProcessor

    assert settings.DRAWING_SENSE_RULES is False

    def boom(*_a, **_k):
        raise AssertionError("правила вызваны при выключенном флаге")

    monkeypatch.setattr(sense_rules, "apply", boom)
    monkeypatch.setattr(sense_rules, "check_links", boom)
    monkeypatch.setattr(settings, "DRAWING_READINGS_RECONCILED", True)
    sheet, view = three_readings()
    processor = DrawingProcessor(temp_storage, qwen=type("Q", (), {"available": False, "model": "x"})())
    out = processor.enrich_fragments([sheet, view], {})
    reference_sheet, reference_view = three_readings()
    reading_reconcile.apply([reference_sheet, reference_view])
    assert [f.model_dump() for f in out] == [f.model_dump() for f in (reference_sheet, reference_view)]


# ---------------------------------------------------------------- правки ревью

MATERIAL = ("Лист 0,5 ГОСТ 19904-90", [0.7, 0.9, 0.95, 0.93])


def sheet_with(*fields):
    return drawing([field(MATERIAL[0], MATERIAL[1], category="material"), *fields])


def test_bare_size_on_thin_sheet_stays():
    sheet = sheet_with(field("50,5", MARK))
    sense_rules.apply([sheet])
    size = sheet.content.structured_drawing_fields[1]
    assert size.value == "50,5" and size.readings is None


def test_bare_size_with_s_reading_becomes_thickness():
    sheet = sheet_with(field("50,5", MARK, readings=[{"source": "sheet", "value": "50,5"},
                                                     {"source": "crop", "value": "S0,5"}]))
    sense_rules.apply([sheet])
    assert sheet.content.structured_drawing_fields[1].value == "S0,5"


def test_unrelated_size_with_stray_s_reading_stays():
    sheet = sheet_with(field("120", MARK, readings=[{"source": "sheet", "value": "120"},
                                                    {"source": "crop", "value": "S0,5"}]))
    sense_rules.apply([sheet])
    assert sheet.content.structured_drawing_fields[1].value == "120"


def test_starred_roughness_without_s_or_thickness_stays_for_review():
    sheet = drawing([field("Сталь 45 ГОСТ 1050-88", MATERIAL[1], category="material"),
                     field("Ra 3,2*", [0.2, 0.2, 0.24, 0.22], category="roughness")])
    sense_rules.apply([sheet])
    rough = sheet.content.structured_drawing_fields[1]
    assert rough.category == "roughness" and rough.value == "Ra 3,2*" and rough.needs_review is True


def test_roughness_not_removable_is_not_converted_into_duplicate_size():
    sheet = sheet_with(field("50,5*", MARK))
    view = drawing([field("Ra 0,5*", MARK_IN_VIEW, category="roughness")], bbox=VIEW, name="view")
    stats = sense_rules.apply([sheet, view])
    (rough,) = view.content.structured_drawing_fields
    assert rough.category == "roughness" and rough.needs_review is True
    assert stats["roughness_to_size"] == 0 and stats["roughness_dropped"] == 0


def test_dropped_roughness_readings_go_to_survivor():
    sheet = sheet_with(field("50,5*", MARK), field("Ra 0,5*", [0.805, 0.085, 0.835, 0.105],
                                                   category="roughness"))
    sense_rules.apply([sheet])
    values = [f.value for f in sheet.content.structured_drawing_fields]
    assert "Ra 0,5*" not in values
    mark = next(f for f in sheet.content.structured_drawing_fields if f.value == "S0,5*")
    assert "Ra 0,5*" in [r["value"] for r in mark.readings]


def test_large_box_centre_does_not_swallow_small_roughness():
    # Центр крупной рамки попал в мелкую надпись, а центр надписи — вне
    # крупной: по строгому правилу это не одно место.
    big = [0.4, 0.3, 0.6, 0.7]
    small = [0.45, 0.48, 0.95, 0.52]
    assert sense_rules.overlaps(small, big) is True
    assert sense_rules.overlaps(small, big, strict=True) is False
    assert sense_rules.overlaps(big, small, strict=True) is False
    # Мелкая надпись внутри крупной — одно место и по строгому правилу.
    assert sense_rules.overlaps([0.45, 0.4, 0.5, 0.45], big, strict=True) is True


@pytest.mark.parametrize("value, label, fits", [
    ("R5", "радиус", True), ("SR10", "радиус сферы", True), ("(R5)", "радиус", True),
    ("4×R1", "радиус", True), ("Ra 6,3", "радиус", False), ("Rz 20", "радиус", False),
    ("⌀6", "диаметр", True), ("Ø6", "диаметр", True), ("Ф6", "диаметр", True), ("ф6", "диаметр", True),
    ("Φ6", "диаметр", True), ("S⌀20", "диаметр сферы", True), ("(⌀6)", "диаметр", True), ("12", "диаметр вала", False),
    ("M8", "резьбовое отверстие", True), ("G1/2", "отверстие", True), ("Rc1/4", "отверстие", True),
    ("Tr20x4", "отверстие", True), ("К1/8", "отверстие", True),
    ("4 отверстия ⌀6", "отверстие", True), ("2 места ⌀8", "отверстие", True),
    ("2 отв. ⌀8", "отверстие", True), ("R5", "отверстие", False),
])
def test_link_fits_forms(value, label, fits):
    assert sense_rules.link_fits(value, label) is fits


def test_rewrite_recomputes_derived_metadata():
    from core.providers import vector_drawing

    sheet = drawing([field(MATERIAL[0], MATERIAL[1], category="material"),
                     field("Ra 0,5*", [0.2, 0.2, 0.24, 0.22], category="roughness",
                           unit="мкм", tolerance="±0,1")])
    sense_rules.apply([sheet])
    size = sheet.content.structured_drawing_fields[1]
    assert size.value == "S0,5*" and size.category == "size"
    assert size.tolerance == vector_drawing.extract_tolerance("S0,5*")
    assert size.unit == vector_drawing.extract_unit("S0,5*", "size")


@pytest.mark.parametrize("texts, expected", [
    (["Лист 0,5"], "0,5"),                               # конец записи — только в графе материала
    ([("Лист 0,5", False)], None),
    ([("S0,5 ГОСТ 19903", False)], None),                # «S» — только из графы материала
    (["С 20 ГОСТ 1050-88"], None),                       # кириллическая «С» — не толщина
    (["Лист 1/2"], None), (["Лист 1 Листов 2"], None),   # счётчик листов
    ([("Лист 1,5 ГОСТ 19903-74", False)], "1,5"),        # с ГОСТ — и в прочей графе
    (["Лист 5 ГОСТ 19903-74"], "5"),
    (["Лист 0,5 ГОСТ 19904-90", "S0,8"], None),          # противоречие — не гадать
])
def test_thickness_sources(texts, expected):
    assert sense_rules.thickness(texts) == expected


def test_whole_thickness_bare_55_stays_starred_55_rewritten():
    sheet = drawing([field("Лист 5 ГОСТ 19903-74", MATERIAL[1], category="material"),
                     field("55", [0.3, 0.3, 0.33, 0.32]), field("55*", MARK)])
    sense_rules.apply([sheet])
    values = [f.value for f in sheet.content.structured_drawing_fields]
    assert values[1:] == ["55", "S5*"]


# ---------------------------------------------------------------------------
# Резьба, которой на листе нет (метка шва, обозначение элемента схемы)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("value", ["2M3", "2M", "2N*1", "4№1"])
def test_weld_mark_read_as_thread_becomes_callout(value):
    """Пробка ЭК150 СБ: метка шва «2№3» приходила резьбой «2M3»."""
    sheet = drawing([field(value, [0.7, 0.2, 0.75, 0.23], category="thread")])
    stats = sense_rules.apply([sheet])
    (mark,) = sheet.content.structured_drawing_fields
    assert mark.category == "callout" and mark.value == value and mark.needs_review
    assert stats["thread_weld"] == 1


@pytest.mark.parametrize("value", ["M1", "M11", "М2", "M1(2)", "M111", "M12-15"])
def test_scheme_designator_is_not_a_thread(value):
    """Схема Э4: «M1» — двигатель по ГОСТ 2.710, а не резьба М1."""
    sheet = drawing([field(value, [0.3, 0.3, 0.32, 0.32], category="thread")])
    sheet.content.structured_payload = {"sheet_type": "scheme"}
    sense_rules.apply([sheet])
    assert sheet.content.structured_drawing_fields[0].category == "position"


@pytest.mark.parametrize("value, sheet_type", [
    ("M1", "detail"), ("M6", "assembly"), ("M6-6H", "scheme"), ("M12-6g", "scheme"), ("M20x1,5", "scheme"),
    ("2 отв. M3", "detail"), ("2 M3", "detail"), ("G1/2", "detail"),
])
def test_real_threads_stay(value, sheet_type):
    sheet = drawing([field(value, [0.3, 0.3, 0.32, 0.32], category="thread")])
    sheet.content.structured_payload = {"sheet_type": sheet_type}
    sense_rules.apply([sheet])
    assert sheet.content.structured_drawing_fields[0].category == "thread"


# ---------------------------------------------------------------------------
# N-S1: толщина из текста страницы, справочная толщина, двойники одного места
# ---------------------------------------------------------------------------

def test_thickness_from_transcript_only():
    """«Лист 0,5 ГОСТ 19904-90» только в тексте листа: «50,5*» -> «S0,5*»."""
    sheet = drawing([field("50,5*", MARK)])
    sheet.content.structured_payload = {"transcript_lines": ["Лист 0,5 ГОСТ 19904-90", "Масса 0,02"]}
    sense_rules.apply([sheet])
    assert sheet.content.structured_drawing_fields[0].value == "S0,5*"


def test_thickness_from_page_ocr_text():
    sheet = drawing([field("50,5*", MARK)])
    text = Fragment(fragment_id="t", type="text", content=Content(text="Бирка. Лист 0,5 ГОСТ 19904-90"),
                    position=Position(page=1, bbox=[0, 0, 1, 1]), confidence=0.9, completeness=1.0,
                    provenance=Provenance(method="ocr_layer", strategy_level=5, source="scanned_pdf"))
    sense_rules.apply([sheet, text])
    assert sheet.content.structured_drawing_fields[0].value == "S0,5*"


def test_sheet_number_in_transcript_is_not_thickness():
    """«Лист 1» вне графы материала — номер листа: толщины нет, размер не трогается."""
    sheet = drawing([field("51*", MARK)])
    sheet.content.structured_payload = {"transcript_lines": ["Лист 1", "Листов 1", "Лист 1 Листов 2"]}
    sense_rules.apply([sheet])
    assert sheet.content.structured_drawing_fields[0].value == "51*"


def test_no_thickness_anywhere_leaves_size():
    sheet = drawing([field("50,5*", MARK)])
    sense_rules.apply([sheet])
    assert sheet.content.structured_drawing_fields[0].value == "50,5*"


def test_reference_star_and_s_reading_make_thickness_without_material():
    sheet = drawing([
        field("50,5*", MARK, readings=[{"source": "sheet", "value": "50,5*"}, {"source": "crop", "value": "S0,5*"}]),
        field("* Размеры для справок.", [0.1, 0.8, 0.4, 0.82], category="note"),
    ])
    sense_rules.apply([sheet])
    assert sheet.content.structured_drawing_fields[0].value == "S0,5*"


def test_reference_rule_needs_the_note():
    sheet = drawing([field("50,5*", MARK, readings=[{"source": "sheet", "value": "50,5*"},
                                                    {"source": "crop", "value": "S0,5*"}])])
    sense_rules.apply([sheet])
    assert sheet.content.structured_drawing_fields[0].value == "50,5*"


def test_roughness_twin_over_thickness_collapses():
    """«Ra 0,5*» на месте «S0,5*» — то же прочтение: остаётся толщина."""
    sheet = sheet_with(field("S0,5*", MARK), field("Ra 0,5", MARK, category="roughness"))
    stats = sense_rules.apply([sheet])
    values = [f.value for f in sheet.content.structured_drawing_fields if f.category != "material"]
    assert values == ["S0,5*"] and stats["twins_merged"] == 1


def test_roughness_elsewhere_and_two_sizes_stay():
    sheet = drawing([
        field("3,2", MARK), field("Ra 3,2", [0.1, 0.1, 0.13, 0.12], category="roughness"),
        field("32", [0.5, 0.5, 0.53, 0.52]), field("52", [0.5, 0.5, 0.53, 0.52]),
    ])
    sense_rules.apply([sheet])
    assert len(sheet.content.structured_drawing_fields) == 4
