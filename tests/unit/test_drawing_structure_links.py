"""
Структура листа, S5–S7: виды, связи, проверки и предел размера.

Синтетический лист: размеры с отверстиями, выноски «п.8», звёздочка,
поверхность, шероховатость, позиции и спецификация. Данных заказчика нет.
"""

from core.drawing_structure import validate
from core.drawing_structure.assemble import build_sheet
from core.models.contract import Content, DimensionLink, DrawingField, Fragment, Position, Provenance
from core.models.drawing_sheet import SIZE_CAP_BYTES, sheet_size_bytes


def _field(category, value, bbox, confidence=0.8, **extra):
    return DrawingField(category=category, value=value, bbox=bbox, confidence=confidence, **extra)


def _drawing(fields, bbox=(0.0, 0.0, 1.0, 1.0), payload=None, text=None, fid="f"):
    return Fragment(
        fragment_id=fid, type="drawing",
        content=Content(text=text, structured_drawing_fields=fields, structured_payload=payload or {}),
        position=Position(page=2, bbox=list(bbox)), confidence=0.8, completeness=1.0,
        provenance=Provenance(method="vlm_full_page", strategy_level=7, source="drawing"),
    )


def _detail():
    link = DimensionLink(target_label="отверстие", target_bbox=[0.30, 0.40, 0.33, 0.43],
                         leader=[[0.31, 0.35], [0.31, 0.40]], method="vlm", confidence=0.6)
    fields = [
        _field("size", "⌀8,5", [0.30, 0.33, 0.34, 0.35], link=link),
        _field("callout", "N отв.", [0.30, 0.355, 0.34, 0.37]),
        _field("callout", "п.5", [0.345, 0.33, 0.37, 0.35]),
        _field("size", "18*", [0.50, 0.50, 0.53, 0.52]),
        _field("thread", "М20×1,5-6g", [0.20, 0.60, 0.27, 0.62]),
        _field("callout", "Лицевая сторона п.8", [0.60, 0.40, 0.72, 0.42]),
        _field("roughness", "Ra 1,6", [0.62, 0.43, 0.66, 0.45]),
        _field("roughness", "Ra 6,3 (√)", [0.85, 0.03, 0.95, 0.06]),
        _field("callout", "А (2:1)", [0.10, 0.08, 0.16, 0.10]),
        _field("note", "1. * Размеры для справок.", [0.10, 0.70, 0.40, 0.72]),
        _field("note", "5. Отверстия по схеме п.6.", [0.10, 0.72, 0.40, 0.74]),
        _field("note", "8. Лицевую сторону полировать.", [0.10, 0.74, 0.40, 0.76]),
        _field("title_block", "Масштаб: 1:1", [0.55, 0.75, 0.98, 0.97]),
        _field("title_block", "Лист: 1", [0.55, 0.75, 0.98, 0.97]),
        _field("title_block", "Листов: 2", [0.55, 0.75, 0.98, 0.97]),
    ]
    payload = {"title_block_bbox": [0.55, 0.75, 0.98, 0.97], "sheet_type": "detail",
               "layout": {"angle": 0.0, "turn": 0, "frame": [0.02, 0.02, 0.98, 0.98]}}
    whole = _drawing(fields, payload=payload, fid="whole")
    crop = _drawing([], bbox=(0.08, 0.11, 0.45, 0.65), fid="crop")
    return whole, crop


def test_hole_count_and_note_reference_attach_to_diameter():
    sheet, _ = build_sheet(list(_detail()))
    dim = next(d for d in sheet.dimensions if d.kind == "diameter")
    assert dim.value_raw == "⌀8,5" and dim.count == "N"
    note5 = next(n for n in sheet.notes if n.number == 5)
    assert note5.id in dim.note_refs
    assert dim.id in note5.refs.dimensions
    assert dim.link.method == "vlm" and dim.leader


def test_derived_hole_feature_from_link_target():
    sheet, _ = build_sheet(list(_detail()))
    dim = next(d for d in sheet.dimensions if d.kind == "diameter")
    hole = next(f for f in sheet.features if f.id == dim.feature_id)
    assert hole.kind == "hole" and hole.bbox == [0.3, 0.4, 0.33, 0.43]
    assert hole.derived_from == [dim.id]
    thread = next(d for d in sheet.dimensions if d.kind == "thread")
    assert next(f for f in sheet.features if f.id == thread.feature_id).kind == "thread"


def test_marker_links_reference_size_to_defining_note():
    sheet, warnings = build_sheet(list(_detail()))
    ref = next(d for d in sheet.dimensions if d.nominal == 18)
    note1 = next(n for n in sheet.notes if n.number == 1)
    assert ref.reference and note1.defines_marker == "*"
    assert note1.id in ref.note_refs
    assert validate.W_MARKER not in warnings


def test_surface_callout_links_note_and_roughness():
    sheet, _ = build_sheet(list(_detail()))
    surface = next(f for f in sheet.features if f.kind == "surface")
    assert surface.label == "Лицевая сторона"
    note8 = next(n for n in sheet.notes if n.number == 8)
    assert surface.id in note8.refs.features
    rough = {s.value: s for s in sheet.symbols}
    assert rough["Ra 1,6"].scope == "surface" and rough["Ra 1,6"].feature_id == surface.id
    assert rough["Ra 6,3 (√)"].scope == "general"


def test_views_from_title_and_crop_frame():
    sheet, _ = build_sheet(list(_detail()))
    view = next(v for v in sheet.views if v.label == "А")
    assert view.kind == "detail" and view.scale == "2:1"
    assert view.bbox == [0.08, 0.11, 0.45, 0.65]
    dim = next(d for d in sheet.dimensions if d.kind == "diameter")
    assert dim.view_id == view.id


def test_validation_warnings_note_gap_and_sheet_count():
    sheet, warnings = build_sheet(list(_detail()))
    assert validate.W_NOTE_GAP in warnings      # пункты 1, 5, 8
    assert validate.W_SHEETS not in warnings
    assert sheet.title_block.fields["scale"].value == "1:1"
    assert sheet.stats.elements > 10


def test_sheet_greater_than_sheets_is_flagged():
    whole = _drawing([
        _field("title_block", "Лист: 3", [0.55, 0.75, 0.98, 0.97]),
        _field("title_block", "Листов: 2", [0.55, 0.75, 0.98, 0.97]),
    ], payload={"title_block_bbox": [0.55, 0.75, 0.98, 0.97]})
    sheet, warnings = build_sheet([whole])
    assert validate.W_SHEETS in warnings
    assert sheet.title_block.fields["sheet"].review.state == "needs_check"


def test_balloon_links_to_spec_row():
    fields = [
        _field("position", "1", [0.30, 0.30, 0.32, 0.32]),
        _field("position", "2", [0.40, 0.30, 0.42, 0.32]),
        _field("position", "7", [0.50, 0.30, 0.52, 0.32]),
    ]
    payload = {"sheet_type": "assembly", "tables": [{
        "kind": "table", "bbox": [0.55, 0.40, 0.98, 0.70], "title": "",
        "headers": ["Поз.", "Обозначение", "Наименование", "Кол."],
        "rows": [["1", "А.001", "Корпус", "1"], ["2", "А.002", "Крышка", "1"]],
    }]}
    sheet, warnings = build_sheet([_drawing(fields, payload=payload)])
    rows = {p.number: p.table_row for p in sheet.positions}
    assert rows["1"].row == 0 and rows["2"].row == 1 and rows["7"] is None
    assert sheet.tables[0].kind == "spec"
    assert validate.W_POSITION in warnings


def test_size_cap_drops_texts_and_marks_truncated():
    fields = [_field("callout", f"Надпись номер {i} " + "ж" * 300, [0.01 * (i % 90), 0.5, 0.01 * (i % 90) + 0.01, 0.51])
              for i in range(200)]
    sheet, warnings = build_sheet([_drawing(fields)])
    assert sheet is not None and sheet.stats.truncated
    assert sheet_size_bytes(sheet) <= SIZE_CAP_BYTES
    assert validate.W_TRUNCATED in warnings


def _notes_sheet(extra=(), payload_extra=None, text=None):
    fields = [
        _field("note", "9. Сварные швы по ГОСТ 1-76.", [0.10, 0.70, 0.45, 0.72]),
        _field("note", "10. Оценка качества сварных соединений:", [0.10, 0.72, 0.30, 0.74]),
        _field("note", "- визуальный контроль по РБ-1-14 1) - 100%.", [0.14, 0.74, 0.45, 0.76]),
        _field("note", "Нормы оценки по ГОСТ 2-2009 1); уровень качества - D.", [0.10, 0.76, 0.45, 0.78]),
        _field("note", "11. Цвета побежалости не допускаются.", [0.10, 0.78, 0.30, 0.80]),
        _field("note", "1) - нормативный документ по договору поставки.", [0.14, 0.82, 0.42, 0.84]),
        _field("note", "МЕТРОЛОГИЧЕСКАЯ ЭКСПЕРТИЗА ПРОВЕДЕНА Петров", [0.05, 0.80, 0.20, 0.82]),
        *extra,
    ]
    payload = {"sheet_type": "detail", "title_block_bbox": [0.55, 0.85, 0.98, 0.98],
               **(payload_extra or {})}
    return build_sheet([_drawing(fields, payload=payload, text=text)])


def test_note_continuation_lines_join_previous_note():
    sheet, _ = _notes_sheet()
    note10 = next(n for n in sheet.notes if n.number == 10 and n.kind == "item")
    assert "визуальный контроль" in note10.text and "Нормы оценки" in note10.text
    assert note10.bbox[3] == 0.78 and len(note10.field_refs) == 3
    texts = [t.text for t in sheet.texts]
    assert not any("визуальный" in t or "Нормы оценки" in t for t in texts)
    # Штамп прописными левее пункта продолжением не становится.
    assert any(t.startswith("МЕТРОЛОГИЧЕСКАЯ") for t in texts)
    assert not any("МЕТРОЛОГ" in n.text for n in sheet.notes)


def test_footnote_is_separate_and_linked_from_marker():
    sheet, warnings = _notes_sheet()
    footnote = next(n for n in sheet.notes if n.kind == "footnote")
    assert footnote.number == 1 and footnote.text.startswith("нормативный")
    note10 = next(n for n in sheet.notes if n.number == 10)
    assert footnote.id in note10.refs.notes
    # Сноска 1 не закрывает пропуск пунктов 1–8 и не подменяет пункт 1.
    assert validate.W_NOTE_GAP in warnings


def test_numbered_line_between_breaks_continuation():
    fields = [
        _field("note", "8. Остальные характеристики по ТЗ.", [0.57, 0.48, 0.81, 0.49]),
        _field("note", "4 Для блока использовать модули.", [0.57, 0.49, 0.81, 0.50]),
        _field("note", "- трубы крепить скобами.", [0.57, 0.50, 0.81, 0.51]),
    ]
    sheet, _ = build_sheet([_drawing(fields, payload={"sheet_type": "detail"})])
    note8 = next(n for n in sheet.notes if n.number == 8)
    assert note8.text == "Остальные характеристики по ТЗ."
    assert {t.text for t in sheet.texts} == {"4 Для блока использовать модули.", "- трубы крепить скобами."}


def test_date_reading_goes_to_title_not_notes():
    text = "Технические требования\n1. Размеры для справок.\n14. 10.16 (рукописно)"
    extra = [_field("callout", "14.10.16", [0.60, 0.90, 0.66, 0.92])]
    sheet, _ = _notes_sheet(extra=extra, text=text)
    assert not any(n.number == 14 for n in sheet.notes)
    assert sheet.title_block.fields["signed_date"].value == "14.10.16"
    assert not any(d.value_raw == "14.10.16" for d in sheet.dimensions)


def test_role_name_and_date_split_from_aggregated_cell():
    payload = {"title_block": {"developed": "Иванов 01.02.20 (рукописно)", "checked": "Петров"}}
    sheet, _ = _notes_sheet(payload_extra=payload)
    roles = {r.role: r for r in sheet.title_block.roles}
    assert roles["developed"].name == "Иванов" and roles["developed"].date == "01.02.20"
    assert roles["developed"].handwritten and roles["developed"].personal
    assert roles["checked"].name == "Петров"


def test_number_in_title_block_is_not_a_position():
    fields = [
        _field("position", "1", [0.30, 0.30, 0.32, 0.32]),
        _field("position", "1", [0.96, 0.90, 0.97, 0.92]),
        _field("title_block", "Масштаб: 1:1", [0.55, 0.85, 0.98, 0.98]),
    ]
    payload = {"sheet_type": "assembly", "title_block_bbox": [0.55, 0.85, 0.98, 0.98]}
    sheet, _ = build_sheet([_drawing(fields, payload=payload)])
    assert [p.bbox for p in sheet.positions] == [[0.3, 0.3, 0.32, 0.32]]
    assert any(t.text == "1" for t in sheet.texts)
