"""
Графы поля подшивки и графа 26 листа ЕСКД (DRAWING_MARGIN_COLUMNS).

Повёрнутые надписи слева от рамки проход листа не читает: с листа детали
пришло «Инв. № подл.: 05» с рамкой на пол-листа вместо «1012-05». Полоса и
графа 26 вырезаются по геометрии ГОСТ, читаются одним вызовом, поле графы
получает рамку своей клетки. Модель поддельная, значения синтетические.
"""

from io import BytesIO

import pytest

from core.config import settings
from core.providers import drawing_vision, margin_columns
from core.providers.qwen_client import SheetReading

pytest.importorskip("PIL")
from PIL import Image  # noqa: E402

# Лист A4 книжный, 4 пикселя на мм: внутренняя рамка 185×287 мм с полем
# подшивки 20 мм слева и 5 мм с остальных сторон.
SIZE = (840, 1188)
FRAME = [20 / 210, 5 / 297, 205 / 210, 292 / 297]


class MarginQwen:
    available = True
    model = "fake-vl"

    def __init__(self, answer, sheet_fields=(), sheet_type="detail"):
        self.answer = answer
        self.sheet_fields = list(sheet_fields)
        self.sheet_type = sheet_type
        self.margin_calls = []

    def read_sheet(self, image, hint="", **_):
        return SheetReading(sheet_type=self.sheet_type, annotations=list(self.sheet_fields))

    def read_title_block(self, image):
        return None

    def read_table(self, image):
        return None

    def read_margin_columns(self, image):
        self.margin_calls.append(image.size)
        return self.answer


def layout_with_frame(image):
    return drawing_vision.SheetLayout(image=image, signals={"inner_frame": list(FRAME)})


def run(temp_storage, monkeypatch, answer, sheet_fields=(), flag=True, budget_left=True, frame=True,
        sheet_type="detail"):
    from core.drawing_processor import DrawingProcessor
    from core.models.contract import Content, Fragment, Position, Provenance

    monkeypatch.setattr(settings, "DRAWING_MARGIN_COLUMNS", flag)
    monkeypatch.setattr(settings, "DRAWING_VISION_ENABLED", False)
    if frame:
        monkeypatch.setattr(drawing_vision, "plain", layout_with_frame)
    if not budget_left:
        monkeypatch.setattr(settings, "DRAWING_TIME_BUDGET", 10)
        monkeypatch.setattr(settings, "VLM_TIMEOUT", 30)
    data = BytesIO()
    Image.new("RGB", SIZE, "white").save(data, format="PNG")
    temp_storage.write_file("assets/doc/sheet.png", data.getvalue())
    fragment = Fragment(
        fragment_id="doc-frag-1", type="image", content=Content(image_ref="assets/doc/sheet.png"),
        position=Position(page=1, bbox=[0.0, 0.0, 1.0, 1.0], order=1), confidence=0.9, completeness=1.0,
        provenance=Provenance(method="mineru_image", strategy_level=2, source="vector_pdf"),
    )
    qwen = MarginQwen(answer, sheet_fields, sheet_type)
    out = DrawingProcessor(temp_storage, qwen=qwen).enrich_fragments([fragment], {})
    sheet = next(f for f in out if f.content.image_ref == "assets/doc/sheet.png")
    return sheet, qwen


DIM = {"category": "size", "value": "⌀18", "bbox": [0.4, 0.4, 0.43, 0.42]}
# Прочтение листа: хвост номера и рамка области штампа на пол-листа.
HALF_PAGE = {"category": "title_block", "value": "Инв. № подл.: 05 (рукописно)",
             "bbox": [0.057, 0.496, 0.96, 0.966]}
DESIGNATION = {"category": "title_block", "value": "Обозначение: АБВГ.123456.001",
               "bbox": [0.6, 0.9, 0.9, 0.95]}


def answer(**columns):
    return {"columns": {key: {"value": value, "handwritten": True} for key, value in columns.items()},
            "top_designation": ""}


def labelled(sheet, label):
    return [f for f in sheet.content.structured_drawing_fields or [] if f.value.startswith(label + ":")]


def inside(box, outer):
    return (box[0] >= outer[0] - 1e-4 and box[1] >= outer[1] - 1e-4
            and box[2] <= outer[2] + 1e-4 and box[3] <= outer[3] + 1e-4)


def test_margin_read_extends_half_page_tail(temp_storage, monkeypatch):
    sheet, qwen = run(temp_storage, monkeypatch, answer(inventory="1012-05"), [DIM, HALF_PAGE])
    assert len(qwen.margin_calls) == 1
    (field,) = labelled(sheet, "Инв. № подл.")
    assert field.value == "Инв. № подл.: 1012-05 (рукописно)"
    strip = [0.0, FRAME[1], FRAME[0], FRAME[3]]
    assert inside(field.bbox, strip)
    x1, y1, x2, y2 = field.bbox
    assert (x2 - x1) * (y2 - y1) < 0.01            # клетка графы, а не пол-листа
    assert y2 == pytest.approx(FRAME[3], abs=1e-3)  # нижняя графа колонки
    # «05» — хвост «1012-05»: остаётся длинное, прежнее — прочтением.
    assert [r["source"] for r in field.readings] == ["sheet", "margin_column"]
    assert field.readings[0]["value"] == HALF_PAGE["value"]
    stats = sheet.content.structured_payload["margin_columns"]
    assert stats["extended"] == 1 and stats["status"] == "read"


def test_conflicting_margin_read_wins_and_goes_to_review(temp_storage, monkeypatch):
    # Значение листа с подписью графы подшивки прочитано не с её клетки
    # (штамп отдал даты своих граф «Дата»): остаётся чтение клетки.
    other = dict(HALF_PAGE, value="Инв. № подл.: 7777 (рукописно)")
    sheet, _ = run(temp_storage, monkeypatch, answer(inventory="1012-05"), [DIM, other])
    (field,) = labelled(sheet, "Инв. № подл.")
    assert field.value == "Инв. № подл.: 1012-05 (рукописно)" and field.needs_review is True
    assert other["value"] in [r["value"] for r in field.readings]
    assert field.bbox[2] <= FRAME[0] + 1e-4          # рамка — клетка графы
    assert sheet.content.structured_payload["margin_columns"]["conflict"] == 1


def test_longer_sheet_value_kept_over_margin_tail():
    fields = [{"category": "title_block", "value": "Инв. № подл.: 1012-05", "bbox": [0.1, 0.5, 0.9, 0.9]}]
    read = {"category": "title_block", "value": "Инв. № подл.: 05", "bbox": [0.05, 0.9, 0.09, 0.98]}
    stats = margin_columns.merge(fields, [read])
    assert fields[0]["value"] == "Инв. № подл.: 1012-05" and stats["extended"] == 1
    assert fields[0]["bbox"] == read["bbox"] and not fields[0].get("needs_review")


def test_scheme_sheet_makes_no_call(temp_storage, monkeypatch):
    _, qwen = run(temp_storage, monkeypatch, answer(inventory="1012-05"), [DIM], sheet_type="scheme")
    assert qwen.margin_calls == []


def test_same_value_keeps_field_with_cell_box(temp_storage, monkeypatch):
    same = dict(HALF_PAGE, value="Инв. № подл.: 1012-05 (рукописно)")
    sheet, _ = run(temp_storage, monkeypatch, answer(inventory="1012-05"), [DIM, same])
    (field,) = labelled(sheet, "Инв. № подл.")
    assert field.bbox[2] <= FRAME[0] + 1e-4 and not field.readings


def test_empty_columns_add_nothing(temp_storage, monkeypatch):
    empty = answer(inventory="", signed_date="", first_use="Перв. примен.", reference="—")
    sheet, qwen = run(temp_storage, monkeypatch, empty, [DIM])
    assert len(qwen.margin_calls) == 1
    values = [f.value for f in sheet.content.structured_drawing_fields]
    assert values == ["⌀18"]
    assert sheet.content.structured_payload["margin_columns"]["added"] == 0


def test_filled_columns_get_their_own_cells(temp_storage, monkeypatch):
    sheet, _ = run(temp_storage, monkeypatch,
                   answer(inventory="1012-05", signed_date="10.02.2022", first_use="АБВГ.123456.001"),
                   [DIM])
    inv = labelled(sheet, "Инв. № подл.")[0].bbox
    date = labelled(sheet, "Подп. и дата")[0].bbox
    first = labelled(sheet, "Перв. примен.")[0]
    assert date[3] == pytest.approx(inv[1], abs=1e-3)   # «Подп. и дата» над «Инв. № подл.»
    assert first.bbox[1] == pytest.approx(FRAME[1], abs=1e-3)  # верхняя колонка
    assert first.value == "Перв. примен.: АБВГ.123456.001"     # не рукописная графа


def test_budget_exhausted_makes_no_call(temp_storage, monkeypatch):
    sheet, qwen = run(temp_storage, monkeypatch, answer(inventory="1012-05"), [DIM, HALF_PAGE],
                      budget_left=False)
    assert qwen.margin_calls == []
    assert sheet.content.structured_payload["margin_columns"] == {"status": "no_time"}
    assert labelled(sheet, "Инв. № подл.")[0].value == HALF_PAGE["value"]


def test_no_inner_frame_makes_no_call(temp_storage, monkeypatch):
    sheet, qwen = run(temp_storage, monkeypatch, answer(inventory="1012-05"), [DIM], frame=False)
    assert qwen.margin_calls == []
    assert sheet.content.structured_payload["margin_columns"] == {"status": "no_frame"}


def test_flag_off_makes_no_call(temp_storage, monkeypatch):
    assert settings.DRAWING_MARGIN_COLUMNS is False
    sheet, qwen = run(temp_storage, monkeypatch, answer(inventory="1012-05"), [DIM, HALF_PAGE], flag=False)
    assert qwen.margin_calls == []
    assert "margin_columns" not in sheet.content.structured_payload
    assert labelled(sheet, "Инв. № подл.")[0].value == HALF_PAGE["value"]


def test_top_box_confirms_designation_without_second_field(temp_storage, monkeypatch):
    data = answer(inventory="1012-05")
    data["top_designation"] = "АБВГ.123456.001"
    sheet, _ = run(temp_storage, monkeypatch, data, [DIM, DESIGNATION])
    assert len(labelled(sheet, "Обозначение")) == 1
    assert sheet.content.structured_payload["margin_columns"]["designation"] == "confirmed"


def test_top_box_never_changes_designation(temp_storage, monkeypatch):
    broken = dict(DESIGNATION, value="Обозначение: АБВГ 123456001")
    data = answer()
    data["top_designation"] = "АБВГ.123456.002"
    sheet, _ = run(temp_storage, monkeypatch, data, [DIM, broken])
    (field,) = labelled(sheet, "Обозначение")
    assert field.value == "Обозначение: АБВГ 123456001" and field.needs_review is True
    assert [r["value"] for r in field.readings] == ["Обозначение: АБВГ 123456001",
                                                    "Обозначение: АБВГ.123456.002"]
    assert sheet.content.structured_payload["margin_columns"]["designation"] == "conflict"


def test_top_box_without_main_designation_adds_nothing():
    fields = [dict(DIM)]
    assert margin_columns.check_designation(fields, "АБВГ.123456.001") == "no_main"
    assert fields == [DIM]


def test_geometry_cells_stay_inside_strip():
    layout = layout_with_frame(Image.new("RGB", SIZE, "white"))
    geo = margin_columns.geometry(layout)
    strip = geo["strip"]
    assert strip[2] == pytest.approx(FRAME[0], abs=1e-4)
    for key, box in geo["cells"].items():
        assert inside(box, strip), key
    # Графа 26 — 70×14 мм в левом верхнем углу рамки.
    x1, y1, x2, y2 = geo["top_box"]
    assert (x2 - x1) * 210 == pytest.approx(70, abs=0.5) and (y2 - y1) * 297 == pytest.approx(14, abs=0.5)


def test_frame_at_sheet_edge_has_no_margin():
    image = Image.new("RGB", SIZE, "white")
    layout = drawing_vision.SheetLayout(image=image, frame=[0.0, 0.0, 1.0, 1.0])
    assert margin_columns.geometry(layout) is None


def test_compose_rotates_strip_and_box():
    strip = Image.new("RGB", (40, 400), "white")
    box = Image.new("RGB", (280, 56), "white")
    image = margin_columns.compose(strip, box)
    assert image.width == 400 and image.height >= 40 + 56


@pytest.mark.parametrize("raw, expected", [
    ({"columns": {"inventory": "1012-05"}}, {"inventory": ("1012-05", False)}),
    ({"columns": {"inventory": {"value": "1012-05", "handwritten": True}}},
     {"inventory": ("1012-05", True)}),
    ({"columns": {"first_use": {"value": "АБВГ.123456.001", "handwritten": True}}},
     {"first_use": ("АБВГ.123456.001", False)}),
    ({"columns": {"inventory": "Инв. № подл.", "reference": "нет", "foo": "1"}}, {}),
    (None, {}),
])
def test_parse(raw, expected):
    assert margin_columns.parse(raw)["columns"] == expected


def test_stamp_date_under_margin_label_is_dropped():
    """
    ТАДУ 305179.001.01.01.003: «Подп. и дата: 20.06.2025» с рамкой штампа —
    даты граф «Дата» основной надписи; графы подшивки прочитаны с клетки.
    """
    strip = [0.0, 0.03, 0.057, 0.97]
    fields = [
        {"category": "title_block", "value": "Подп. и дата: 20.06.2025 (рукописно)", "bbox": [0.54, 0.76, 0.93, 0.96]},
        {"category": "title_block", "value": "Разраб.: Иванов", "bbox": [0.55, 0.86, 0.6, 0.88]},
    ]
    reads = [{"category": "title_block", "value": "Инв. № подл.: 2068-06", "bbox": [0.03, 0.89, 0.057, 0.97]}]
    stats = margin_columns.merge(fields, reads, strip)
    assert [f["value"] for f in fields] == ["Разраб.: Иванов", "Инв. № подл.: 2068-06"]
    assert stats["dropped"] == 1
    # Без полосы — прежнее поведение: ничего не снимается.
    again = [dict(fields[0]), {"category": "title_block", "value": "Подп. и дата: 1.1.24", "bbox": [0.6, 0.8, 0.9, 0.9]}]
    assert margin_columns.merge(again, [])["dropped"] == 0 and len(again) == 2


def test_margin_read_month_year_fixed(monkeypatch):
    monkeypatch.setattr(settings, "DRAWING_MARGIN_DATE_FIX", True)
    parsed = margin_columns.parse({"columns": {"signed_date": {"value": "18 22", "handwritten": True}}})
    assert parsed["columns"]["signed_date"][0] == "08.22"
