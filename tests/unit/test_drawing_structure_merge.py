"""
Структура листа, S0–S2: пул прочтений и сведение повторов.

Лист синтетический: фрагмент «лист целиком», вырезка вида с полями в долях
вырезки, сводная запись штампа и клетки штампа. Данных заказчика нет.
"""

from core.drawing_structure import candidates, merge
from core.drawing_structure.assemble import build_sheet
from core.models.contract import Content, DrawingField, Fragment, Position, Provenance
from core.providers import title_block


def _field(category, value, bbox, confidence=0.8, **extra):
    return DrawingField(category=category, value=value, bbox=bbox, confidence=confidence, **extra)


def _drawing(fields, bbox=(0.0, 0.0, 1.0, 1.0), payload=None, text=None, method="vlm_full_page", fid="f"):
    return Fragment(
        fragment_id=fid, type="drawing",
        content=Content(text=text, structured_drawing_fields=fields, structured_payload=payload or {}),
        position=Position(page=1, bbox=list(bbox)), confidence=0.8, completeness=1.0,
        provenance=Provenance(method=method, strategy_level=7, source="drawing"),
    )


STAMP = [0.55, 0.75, 0.98, 0.97]


def _sheet():
    whole = _drawing([
        _field("size", "(X+38)±0,2", [0.53, 0.25, 0.60, 0.27]),
        _field("size", "R1", [0.20, 0.30, 0.22, 0.32]),
        _field("size", "R1", [0.40, 0.30, 0.42, 0.32]),
        _field("callout", "Б (4:1)", [0.70, 0.10, 0.76, 0.12]),
        _field("title_block", "Крышка", [0.60, 0.85, 0.75, 0.88]),
        _field("title_block", "Масса", [0.80, 0.80, 0.85, 0.82]),
        _field("title_block", "0,15", [0.80, 0.82, 0.85, 0.84]),
        _field("title_block", "Утв. Глинка 10.02.2022", [0.56, 0.94, 0.70, 0.96]),
        _field("title_block", "Обозначение: ТАДУ 687000.001.001", STAMP),
        _field("title_block", "Наименование: Крышка", STAMP),
        _field("title_block", "Масса: 0,15", STAMP),
        _field("title_block", "Утв.: Глинко", STAMP),
        _field("material", "Материал: 12Х18Н10Т ГОСТ 5632-2014", STAMP),
    ], payload={"title_block_bbox": STAMP, "layout": {"angle": 0.0, "turn": 0}, "sheet_type": "detail",
                "title_block": {"designation": "ТАДУ 687000.001.001"}}, fid="whole")
    # Вырезка вида: рамка на странице [0.5, 0.2, 0.7, 0.4]; поле — в долях вырезки.
    crop = _drawing([
        _field("size", "(X+38)±0,2", [0.15, 0.25, 0.50, 0.35]),
        _field("size", "(Х+38)±0,2", [0.15, 0.25, 0.50, 0.35]),
    ], bbox=(0.5, 0.2, 0.7, 0.4), fid="crop")
    return whole, crop


def test_crop_fields_are_lifted_to_page_coordinates():
    whole, crop = _sheet()
    pool = candidates.collect([whole, crop])
    lifted = [c for c in pool.candidates if c.pass_ == candidates.PASS_VIEW_CROP]
    assert lifted and all(0.5 <= c.bbox[0] <= 0.7 and 0.2 <= c.bbox[1] <= 0.4 for c in lifted)
    assert lifted[0].field_ref == (1, 0)
    assert pool.crop_frames == [[0.5, 0.2, 0.7, 0.4]]


def test_sheet_and_crop_readings_collapse_into_one_dimension():
    sheet, warnings = build_sheet(list(_sheet()))
    assert sheet is not None
    formulas = [d for d in sheet.dimensions if d.formula is not None]
    assert len(formulas) == 1, [d.value_raw for d in sheet.dimensions]
    dim = formulas[0]
    assert {"sheet", "view_crop"} <= set(dim.provenance.passes)
    assert {(r.fragment, r.field) for r in dim.field_refs} >= {(0, 0), (1, 0)}
    assert sheet.stats.merged_duplicates >= 2


def test_two_equal_values_at_different_places_stay_two_elements():
    sheet, _ = build_sheet(list(_sheet()))
    radii = [d for d in sheet.dimensions if d.kind == "radius"]
    assert len(radii) == 2


def test_variant_reading_goes_to_alternatives_and_needs_check():
    whole = _drawing([
        _field("size", "⌀6,6+0,05", [0.30, 0.30, 0.34, 0.32]),
        _field("size", "⌀6,6-0,05", [0.305, 0.301, 0.345, 0.321]),
    ], fid="w")
    elements = merge.merge(candidates.collect([whole]).candidates)
    assert len(elements) == 1
    assert [a["value"] for a in elements[0].alternatives] == ["⌀6,6-0,05"]
    assert elements[0].needs_check()


def test_vector_reading_wins_and_is_exact():
    whole = _drawing([_field("size", "⌀8,8", [0.30, 0.30, 0.34, 0.32])], fid="w")
    vector = _drawing([_field("size", "⌀8,5", [0.31, 0.30, 0.34, 0.315], provenance="vector_text_layer")],
                      method="vector_text_layer", fid="v")
    elements = merge.merge(candidates.collect([whole, vector]).candidates)
    assert len(elements) == 1
    assert elements[0].value == "⌀8,5" and elements[0].exact
    assert not elements[0].needs_check()


def test_title_block_one_field_per_key_with_cell_bbox():
    sheet, _ = build_sheet(list(_sheet()))
    fields = sheet.title_block.fields
    assert fields["mass"].value == "0,15"
    assert fields["mass"].bbox == [0.8, 0.82, 0.85, 0.84]
    assert fields["name"].bbox == [0.6, 0.85, 0.75, 0.88]
    assert fields["designation"].value == "ТАДУ 687000.001.001"
    assert "material" in fields
    # Графа штампа ровно одна на ключ: дублей «Масса» нет ни в полях, ни в надписях.
    assert not any(t.text in ("0,15", "Масса", "Крышка") for t in sheet.texts)


def test_title_roles_are_personal_and_keep_variants():
    sheet, _ = build_sheet(list(_sheet()))
    roles = {r.role: r for r in sheet.title_block.roles}
    assert roles["approved"].personal is True
    assert roles["approved"].name in ("Глинка", "Глинко")
    assert roles["approved"].date == "10.02.2022"


def test_nothing_dropped_every_field_is_referenced():
    whole, crop = _sheet()
    sheet, _ = build_sheet([whole, crop])
    data = sheet.model_dump(mode="json")
    refs = set()

    def walk(node):
        if isinstance(node, dict):
            for ref in node.get("field_refs") or []:
                refs.add((ref["fragment"], ref["field"]))
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)
    walk(data)
    # Строки подписантов уходят в `roles` (графа, фамилия, дата): у роли в
    # схеме v1 нет `field_refs`, поэтому они сверяются по графе.
    roles = {r["role"] for r in data["title_block"]["roles"]}
    for index, item in enumerate(whole.content.structured_drawing_fields):
        if title_block._signer(item.value.split()[0]) in roles:
            refs.add((0, index))
    expected = {(0, i) for i in range(len(whole.content.structured_drawing_fields))}
    expected |= {(1, i) for i in range(len(crop.content.structured_drawing_fields))}
    assert expected <= refs, sorted(expected - refs)


def test_one_pass_never_merges_two_numbers_in_one_table_row():
    whole = _drawing([
        _field("size", "-01", [0.58, 0.57, 0.61, 0.59]),
        _field("size", "247", [0.59, 0.57, 0.62, 0.59]),
    ], fid="w")
    assert len(merge.merge(candidates.collect([whole]).candidates)) == 2


def test_fragment_already_placed_on_page_is_taken_as_is():
    # Размещение полей (DRAWING_FIELD_PLACEMENT_V2) уже перевело рамки вырезки в доли страницы.
    crop = _drawing([_field("size", "R2", [0.55, 0.25, 0.57, 0.27], bbox_exact=True,
                            alternatives=["R3"], needs_review=True)],
                    bbox=(0.5, 0.2, 0.7, 0.4), payload={"coordinate_reference": "page"}, fid="crop")
    whole = _drawing([_field("size", "⌀5", [0.1, 0.1, 0.12, 0.12])], fid="whole")
    pool = candidates.collect([whole, crop])
    placed = next(c for c in pool.candidates if c.value == "R2")
    assert placed.bbox == [0.55, 0.25, 0.57, 0.27] and placed.exact
    element = next(e for e in merge.merge(pool.candidates) if e.value == "R2")
    assert [a["value"] for a in element.alternatives] == ["R3"] and element.needs_check()


def test_crop_mapping_uses_field_placement_turn():
    # Вырезка, повёрнутая на 90°: точка (0, 0) выправленной картинки — правый верхний угол вырезки.
    crop = _drawing([_field("size", "R2", [0.0, 0.0, 0.1, 0.1])], bbox=(0.5, 0.2, 0.7, 0.4),
                    payload={"layout": {"angle": 0.0, "turn": 90}}, fid="crop")
    whole = _drawing([_field("size", "⌀5", [0.1, 0.1, 0.12, 0.12])], fid="whole")
    placed = next(c for c in candidates.collect([whole, crop]).candidates if c.value == "R2")
    assert placed.bbox[2] == 0.7 and placed.bbox[1] == 0.2
