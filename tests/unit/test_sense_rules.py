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
    assert sense_rules.misread_thickness("50.5", "0,5") == "S0,5"
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
