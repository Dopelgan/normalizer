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
NOTE = {"category": "note", "value": "1 Электромонтаж вести проводом ПуГВ-1,0.", "bbox": [0.1, 0.1, 0.5, 0.12]}
ROWS = [
    "Поз. обозначение | Наименование | Кол.",
    "HL1, HL2 | Лампа СКЛ 14 А-Ж-2-220 желтая | 2",
    "QF1, QF2 | Выключатель автоматический трехполюсный 10А | 2",
    "Разраб. | Бабанин",
    "Масса | 1,2",
    "3 Провода свить в витые пары.",
    "1. Электромонтаж вести проводом ПуГВ-1,0",
    "⌀57x3",
    "10",
    "M3",
    "M20x1,5",
    "Вход охлаждающей воды",
]


def test_only_strict_patterns_become_fields(temp_storage, monkeypatch):
    fields, payload = _sheet_fields(temp_storage, monkeypatch, [STAMP, NOTE], ROWS)
    added = [(f["category"], f["value"]) for f in fields if f["provenance"] == "full_page_ocr"]
    # Обозначения элементов («HL1, HL2») и резьба без шага («M3») — не поля.
    assert added == [
        ("title_block", "Масса: 1,2"), ("note", "3 Провода свить в витые пары."),
        ("size", "⌀57x3"), ("thread", "M20x1,5"),
    ]
    # Место неизвестно, значение не подтверждено: запись скажет «сверить».
    assert all(f["bbox"] == [0.0, 0.0, 1.0, 1.0] and f["confidence"] < 0.95
               for f in fields if f["provenance"] == "full_page_ocr")
    assert payload["fields_from_text"] == len(added)


def test_line_already_in_a_field_is_not_repeated(temp_storage, monkeypatch):
    fields, _ = _sheet_fields(temp_storage, monkeypatch, [STAMP, NOTE], ROWS)
    values = [f["value"] for f in fields]
    # «Разраб.: Бабанин» уже прочитан штампом; пункт 1 — тот же, что примечание.
    assert values.count("Разраб.: Бабанин") == 1
    assert not any(v.startswith("1. Электромонтаж") for v in values)
    assert len(values) == len(set(values))


def test_flag_off_keeps_fields_as_before(temp_storage, monkeypatch):
    fields, payload = _sheet_fields(temp_storage, monkeypatch, [STAMP, NOTE], ROWS, from_text=False)
    assert [f["value"] for f in fields] == [STAMP["value"], NOTE["value"]]
    assert "fields_from_text" not in payload


def test_table_gives_position_numbers_only_with_a_name():
    from core.drawing_processor import _fields_from_text
    from core.providers import drawing_vision

    table = {"kind": drawing_vision.KIND_TABLE, "headers": ["Поз.", "Наименование"],
             "rows": [["R1", "Резистор С2-33"], ["2", "Кнопка IP67"], ["3", ""], ["4", "5"]]}
    added = _fields_from_text([], [], [table])
    assert [(f["category"], f["value"]) for f in added] == [("position", "2")]
    # Без графы «Поз.» голое число позицией не считается.
    table["headers"] = ["Обозначение", "Наименование"]
    assert _fields_from_text([], [], [table]) == []


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
