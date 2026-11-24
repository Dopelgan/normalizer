"""
Строки текста листа и таблиц — в поля (DRAWING_FIELDS_FROM_TEXT) и повторы
одной надписи в разных местах листа (DRAWING_KEEP_REPEATS).

Проход «весь текст листа» читает перечень элементов схемы и примечания, а
поля строились только из прохода по категориям: надписи терялись на стыке.
`_deduplicate` сводил одинаковые значения без учёта места: размер на двух
видах оставался одним полем. Модель поддельная.
"""

from io import BytesIO

import pytest

from core.config import settings
from core.providers.qwen_client import SheetReading

pytest.importorskip("PIL")
from PIL import Image  # noqa: E402


class TextQwen:
    available = True
    model = "fake-vl"

    def __init__(self, annotations, lines):
        self.annotations = list(annotations)
        self.lines = list(lines)

    def read_sheet(self, image, hint="", **_):
        return SheetReading(sheet_type="scheme", annotations=list(self.annotations))

    def transcribe(self, image):
        return list(self.lines)

    def read_title_block(self, image):
        return None

    def read_table(self, image, **_):
        return None


def _sheet_fields(temp_storage, monkeypatch, annotations, lines, from_text=True, repeats=False):
    from core.drawing_processor import DrawingProcessor

    monkeypatch.setattr(settings, "DRAWING_FIELDS_FROM_TEXT", from_text)
    monkeypatch.setattr(settings, "DRAWING_KEEP_REPEATS", repeats)
    monkeypatch.setattr(settings, "DRAWING_VISION_ENABLED", False)
    monkeypatch.setattr(settings, "DRAWING_TRANSCRIBE_TILES", False)
    monkeypatch.setattr(settings, "DRAWING_SHEET_GUARDED", False)
    data = BytesIO()
    Image.new("RGB", (1654, 1169), "white").save(data, format="PNG")
    temp_storage.write_file("assets/doc/sheet.png", data.getvalue())
    processor = DrawingProcessor(temp_storage, qwen=TextQwen(annotations, lines))
    result = processor.process("assets/doc/sheet.png", {}, 1, transcribe=True)
    return result["parsed_fields"], result["payload"]


STAMP = {"category": "title_block", "value": "Разраб.: Бабанин", "bbox": [0.7, 0.9, 0.8, 0.92]}
ROWS = [
    "Поз. обозначение | Наименование | Кол.",
    "HL1, HL2 | Лампа СКЛ 14 А-Ж-2-220 желтая | 2",
    "QF1, QF2 | Выключатель автоматический трехполюсный 10А | 2",
    "Бабанин",
    "3 Провода свить в витые пары.",
]


def test_scheme_table_rows_give_position_and_name_fields(temp_storage, monkeypatch):
    fields, payload = _sheet_fields(temp_storage, monkeypatch, [STAMP], ROWS)
    by_value = {f["value"]: f for f in fields}
    assert by_value["HL1, HL2"]["category"] == "position"
    assert by_value["QF1, QF2"]["category"] == "position"
    name = by_value["Лампа СКЛ 14 А-Ж-2-220 желтая"]
    # Место неизвестно, значение не подтверждено: запись скажет «сверить».
    assert name["bbox"] == [0.0, 0.0, 1.0, 1.0]
    assert name["confidence"] < 0.95 and name["provenance"] == "full_page_ocr"
    assert "3 Провода свить в витые пары." in by_value
    assert payload["fields_from_text"] == len(fields) - 1


def test_line_already_in_a_field_is_not_repeated(temp_storage, monkeypatch):
    fields, _ = _sheet_fields(temp_storage, monkeypatch, [STAMP], ROWS)
    values = [f["value"] for f in fields]
    # «Бабанин» уже есть в графе «Разраб.»; «2» — одно и то же количество.
    assert "Бабанин" not in values
    assert values.count("Разраб.: Бабанин") == 1
    assert len(values) == len(set(values))


def test_flag_off_keeps_fields_as_before(temp_storage, monkeypatch):
    fields, payload = _sheet_fields(temp_storage, monkeypatch, [STAMP], ROWS, from_text=False)
    assert [f["value"] for f in fields] == ["Разраб.: Бабанин"]
    assert "fields_from_text" not in payload


def test_table_cells_skip_headers_and_known_values():
    from core.drawing_processor import _fields_from_text
    from core.providers import drawing_vision

    table = {"kind": drawing_vision.KIND_TABLE, "headers": ["Поз.", "Наименование"],
             "rows": [["R1", "Резистор С2-33"], ["Поз.", "Наименование"], ["2", "Кнопка IP67"]]}
    known = [{"category": "note", "value": "Кнопка IP67", "bbox": [0.1, 0.1, 0.2, 0.12]}]
    added = _fields_from_text(known, [], [table])
    assert [(f["category"], f["value"]) for f in added] == [
        ("position", "R1"), ("note", "Резистор С2-33"), ("position", "2"),
    ]


SIZE_A = {"category": "size", "value": "15±0,1", "bbox": [0.10, 0.20, 0.14, 0.22]}
SIZE_B = {"category": "size", "value": "15±0,1", "bbox": [0.60, 0.70, 0.64, 0.72]}


def test_same_size_on_two_views_stays_twice(temp_storage, monkeypatch):
    fields, _ = _sheet_fields(temp_storage, monkeypatch, [SIZE_A, SIZE_B], [],
                              from_text=False, repeats=True)
    assert [f["bbox"] for f in fields if f["value"] == "15±0,1"] == [SIZE_A["bbox"], SIZE_B["bbox"]]


def test_repeats_flag_off_merges_as_before(temp_storage, monkeypatch):
    fields, _ = _sheet_fields(temp_storage, monkeypatch, [SIZE_A, SIZE_B], [],
                              from_text=False, repeats=False)
    assert len([f for f in fields if f["value"] == "15±0,1"]) == 1


def test_one_place_still_merges(monkeypatch):
    from core.drawing_processor import _deduplicate

    monkeypatch.setattr(settings, "DRAWING_KEEP_REPEATS", True)
    near = {**SIZE_A, "bbox": [0.11, 0.21, 0.15, 0.23], "confidence": 0.9}
    region = {"category": "title_block", "value": "Масштаб: 1:1", "bbox": [0.5, 0.8, 1.0, 1.0],
              "confidence": 0.9}
    sheet = {"category": "title_block", "value": "Масштаб: 1:1", "bbox": [0.9, 0.95, 0.93, 0.97],
             "confidence": 0.75}
    whole = {**SIZE_A, "bbox": [0.0, 0.0, 1.0, 1.0], "confidence": 0.5}
    out = _deduplicate([{**SIZE_A, "confidence": 0.75}, near, region, sheet, whole])
    # Тот же размер рядом, без места, и графа штампа внутри области — одно поле.
    assert [(f["value"], f["confidence"]) for f in out] == [("15±0,1", 0.9), ("Масштаб: 1:1", 0.9)]
