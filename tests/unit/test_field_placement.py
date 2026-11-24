"""Размещение полей чертежа на странице (DRAWING_FIELD_PLACEMENT_V2)."""

import pytest

from core.models.contract import Content, DimensionLink, DrawingField, Fragment, Position, Provenance
from core.providers import field_placement as fp


def _field(category, value, bbox, confidence=0.9, **extra):
    return DrawingField(category=category, value=value, bbox=bbox, confidence=confidence, **extra)


def _drawing(fields, frame, page=1, angle=0.0, turn=0, size=None, method="vlm_full_page"):
    layout = {"angle": angle, "turn": turn}
    if size is not None:
        layout["source_size"] = list(size)
    return Fragment(
        fragment_id=f"f-{frame}", type="drawing",
        content=Content(structured_drawing_fields=fields,
                        structured_payload={"layout": layout, "coordinate_reference": "deskewed_page"}),
        position=Position(page=page, bbox=frame),
        provenance=Provenance(method=method, strategy_level=7, source="drawing"),
        confidence=0.8, completeness=1.0,
    )


def _close(a, b, tol=1e-3):
    return all(abs(x - y) <= tol for x, y in zip(a, b))


def test_crop_field_lands_on_page_by_crop_origin_and_scale():
    # Вырезка занимает правую верхнюю четверть страницы: поле в её левом верхнем углу.
    crop = _drawing([_field("callout", "Б", [0.1, 0.2, 0.3, 0.4])], [0.5, 0.0, 1.0, 0.5])
    fp.apply([crop])
    field = crop.content.structured_drawing_fields[0]
    assert _close(field.bbox, [0.55, 0.1, 0.65, 0.2])
    assert field.bbox_exact is True
    assert crop.content.structured_payload["coordinate_reference"] == "page"


@pytest.mark.parametrize("turn, turned, expected", [
    # Картинка 200×100, надпись в пикселях [20..60]×[10..30] до поворота.
    (90, [10 / 100, 140 / 200, 30 / 100, 180 / 200], [0.1, 0.1, 0.3, 0.3]),
    (180, [140 / 200, 70 / 100, 180 / 200, 90 / 100], [0.1, 0.1, 0.3, 0.3]),
    (270, [70 / 100, 20 / 200, 90 / 100, 60 / 200], [0.1, 0.1, 0.3, 0.3]),
])
def test_turned_crop_is_turned_back(turn, turned, expected):
    crop = _drawing([_field("size", "20", turned)], [0.0, 0.0, 1.0, 1.0], turn=turn, size=(200, 100))
    fp.apply([crop])
    assert _close(crop.content.structured_drawing_fields[0].bbox, expected)


def test_turn_inverse_matches_pil_rotation():
    pil = pytest.importorskip("PIL.Image")
    image = pil.new("L", (200, 100), 255)
    image.putpixel((40, 20), 0)
    for turn in (90, 180, 270):
        turned = image.rotate(turn, expand=True)
        x, y = next((x, y) for x in range(turned.size[0]) for y in range(turned.size[1])
                    if turned.getpixel((x, y)) < 128)
        u, v = fp.point_to_page((x + 0.5) / turned.size[0], (y + 0.5) / turned.size[1],
                                [0, 0, 1, 1], 0.0, turn, (200, 100))
        assert abs(u * 200 - 40.5) <= 1.0 and abs(v * 100 - 20.5) <= 1.0


def test_skew_inverse_matches_pil_rotation():
    pil = pytest.importorskip("PIL.Image")
    image = pil.new("L", (400, 300), 255)
    for dx in range(-1, 2):
        for dy in range(-1, 2):
            image.putpixel((300 + dx, 60 + dy), 0)
    skewed = image.rotate(3.0, expand=False, fillcolor=255)
    dark = [(x, y) for x in range(400) for y in range(300) if skewed.getpixel((x, y)) < 200]
    x = sum(p[0] for p in dark) / len(dark) + 0.5
    y = sum(p[1] for p in dark) / len(dark) + 0.5
    u, v = fp.point_to_page(x / 400, y / 300, [0, 0, 1, 1], 3.0, 0, (400, 300))
    assert abs(u * 400 - 300.5) <= 1.5 and abs(v * 300 - 60.5) <= 1.5


def test_link_coordinates_follow_the_field():
    link = DimensionLink(target_label="отверстие", target_bbox=[0.5, 0.5, 0.6, 0.6],
                         leader=[[0.5, 0.5], [1.0, 1.0]], method="vlm", confidence=0.8)
    crop = _drawing([_field("size", "⌀8,5", [0.1, 0.1, 0.2, 0.2], link=link)], [0.2, 0.2, 0.6, 0.6])
    fp.apply([crop])
    moved = crop.content.structured_drawing_fields[0].link
    assert _close(moved.target_bbox, [0.4, 0.4, 0.44, 0.44])
    assert _close(moved.leader[1], [0.6, 0.6])


def test_default_bbox_is_only_the_crop_region():
    crop = _drawing([_field("note", "Линия сгиба", [0, 0, 1, 1])], [0.5, 0.1, 0.9, 0.5])
    fp.apply([crop])
    field = crop.content.structured_drawing_fields[0]
    assert field.bbox == [0.5, 0.1, 0.9, 0.5]
    assert field.bbox_exact is False


def test_vector_drawing_is_not_moved():
    vector = _drawing([_field("size", "20", [0.1, 0.1, 0.2, 0.2])], [0.5, 0.5, 1.0, 1.0],
                      method="vector_text_layer")
    fp.apply([vector])
    assert vector.content.structured_drawing_fields[0].bbox == [0.1, 0.1, 0.2, 0.2]


def test_item_reference_read_as_size_becomes_callout():
    sheet = _drawing([_field("size", "п4", [0.5, 0.5, 0.52, 0.52], confidence=0.7)], [0, 0, 1, 1])
    fp.apply([sheet])
    field = sheet.content.structured_drawing_fields[0]
    assert (field.category, field.value) == ("callout", "п.4")


def test_item_reference_loses_dimension_link():
    link = DimensionLink(target_label="габаритная ширина", target_bbox=[0.3, 0.3, 0.4, 0.4],
                         leader=[], method="vlm", confidence=0.5)
    sheet = _drawing([_field("size", "п4", [0.5, 0.5, 0.52, 0.52], link=link)], [0, 0, 1, 1])
    fp.apply([sheet])
    field = sheet.content.structured_drawing_fields[0]
    assert field.category == "callout" and field.link is None


def test_sheet_and_crop_reading_of_one_inscription_merge_into_the_precise_one():
    sheet = _drawing([_field("size", "п4", [0.55, 0.26, 0.57, 0.30], confidence=0.7)], [0, 0, 1, 1])
    crop = _drawing([_field("callout", "п.4", [0.04, 0.40, 0.06, 0.45], confidence=0.75)],
                    [0.537, 0.089, 0.967, 0.522])
    fp.apply([sheet, crop])
    fields = sheet.content.structured_drawing_fields + crop.content.structured_drawing_fields
    assert len(fields) == 1
    assert fields[0].value == "п.4" and fields[0].confidence == 0.75


def test_same_value_in_two_places_is_kept_twice():
    sheet = _drawing([_field("size", "15±0,1", [0.1, 0.1, 0.15, 0.12]),
                      _field("size", "15±0,1", [0.6, 0.6, 0.65, 0.62])], [0, 0, 1, 1])
    fp.apply([sheet])
    assert len(sheet.content.structured_drawing_fields) == 2


def test_title_block_cell_and_aggregated_record_merge_keeping_label_and_variants():
    stamp = [0.55, 0.75, 0.98, 0.97]
    sheet = _drawing([
        _field("title_block", "Постанов", [0.65, 0.88, 0.72, 0.90], confidence=0.75),
        _field("title_block", "Глинка", [0.65, 0.95, 0.72, 0.97], confidence=0.75),
        _field("title_block", "ТАДУ 687000.001.001", [0.75, 0.79, 0.97, 0.82], confidence=0.75),
        _field("title_block", "2 ГОСТ 19904-90", [0.75, 0.92, 0.85, 0.94], confidence=0.75),
        _field("title_block", "12X18X10T ГОСТ 5582-75", [0.71, 0.94, 0.85, 0.96], confidence=0.75),
        _field("title_block", "1", [0.97, 0.89, 0.98, 0.91], confidence=0.6),
        _field("title_block", "Пров.: Постнов", stamp, confidence=0.85),
        _field("title_block", "Утв.: Глинко", stamp, confidence=0.85),
        _field("title_block", "Обозначение: ТАДУ 687000.001.001", stamp, confidence=0.85),
        _field("material", "Материал: 2 ГОСТ 19904-90 12Х18Н10Т ГОСТ 5582-75", stamp, confidence=0.85),
        _field("title_block", "Лист: 1", stamp, confidence=0.85),
        _field("title_block", "Листов: 1", stamp, confidence=0.85),
    ], [0, 0, 1, 1])
    fp.apply([sheet])
    by_value = {f.value: f for f in sheet.content.structured_drawing_fields}
    checked = by_value["Пров.: Постнов"]
    assert checked.bbox == [0.65, 0.88, 0.72, 0.9] and checked.bbox_exact
    assert checked.alternatives == ["Постанов"] and checked.needs_review
    assert by_value["Утв.: Глинко"].alternatives == ["Глинка"]
    assert by_value["Обозначение: ТАДУ 687000.001.001"].bbox == [0.75, 0.79, 0.97, 0.82]
    material = by_value["Материал: 2 ГОСТ 19904-90 12Х18Н10Т ГОСТ 5582-75"]
    assert material.bbox == [0.71, 0.92, 0.85, 0.96]
    assert material.alternatives == ["12X18X10T ГОСТ 5582-75"] and material.needs_review
    # «1» — и «Лист», и «Листов»: не угадываем, все три остаются.
    assert {"1", "Лист: 1", "Листов: 1"} <= set(by_value)
    assert by_value["Лист: 1"].bbox_exact is False
    assert len(sheet.content.structured_drawing_fields) == 7


@pytest.mark.parametrize("flag, kept", [(True, 3), (False, 2)])
def test_margin_graph_is_not_a_part_of_the_stamp_row_next_to_it(monkeypatch, flag, kept):
    """
    Лист фланца ТАДУ 302389.001.01.02.001: клетка «Подп. и дата» поля
    подшивки стоит вплотную к строке штампа «Разраб. Бабанин 10.02.2022».
    С DRAWING_MARGIN_COLUMNS графа не сводится в строку как её часть, а та
    же запись графы без подписи по-прежнему сводится.
    """
    from core.config import settings

    monkeypatch.setattr(settings, "DRAWING_MARGIN_COLUMNS", flag)
    sheet = _drawing([
        _field("title_block", "Разраб. Бабанин 10.02.2022", [0.118, 0.859, 0.408, 0.88]),
        _field("title_block", "Подп. и дата: 10.02.2022 (рукописно)", [0.06, 0.77, 0.1146, 0.883]),
        _field("title_block", "Инв. № подл.: 1012-05 (рукописно)", [0.06, 0.883, 0.1146, 0.964]),
        _field("title_block", "1012-05", [0.07, 0.89, 0.11, 0.95], confidence=0.6),
    ], [0, 0, 1, 1])
    fp.apply([sheet])
    values = [f.value for f in sheet.content.structured_drawing_fields]
    assert len(values) == kept
    assert "1012-05" not in values
    assert ("Подп. и дата: 10.02.2022 (рукописно)" in values) is flag


def test_garbled_readings_are_marked_low_quality():
    sheet = _drawing([
        _field("callout", "4, Nот", [0.1, 0.1, 0.12, 0.12], confidence=0.75),
        _field("callout", "4N01", [0.2, 0.2, 0.22, 0.22], confidence=0.6),
        _field("note", "11. Цвета побежалости", [0.3, 0.3, 0.5, 0.32], confidence=0.5),
        _field("callout", "N отв.", [0.6, 0.6, 0.62, 0.62], confidence=0.75),
    ], [0, 0, 1, 1])
    fp.apply([sheet])
    low = {f.value: bool(f.low_quality) for f in sheet.content.structured_drawing_fields}
    assert low == {"4, Nот": True, "4N01": True, "11. Цвета побежалости": True, "N отв.": False}


def test_new_keys_absent_without_placement():
    data = _field("size", "20", [0.1, 0.1, 0.2, 0.2]).model_dump()
    assert not {"bbox_exact", "alternatives", "needs_review", "low_quality"} & set(data)
