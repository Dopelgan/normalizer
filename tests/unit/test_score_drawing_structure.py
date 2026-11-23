"""
Оценщик структуры чертёжного листа (`tools/score_drawing_structure.py`).

Лист выдуман. Проверяется: эталон структуры выводится из надписей листа
(штамп по ключам с рамкой ячейки, виды, примечания, ссылки «п.N», маркеры),
плоский выход с копией из вырезки вида и двойным штампом даёт ошибки места и
повторы, а структура со связями получает точность связей.
"""

import importlib.util
import json
from pathlib import Path

_PATH = Path(__file__).resolve().parents[2] / "tools" / "score_drawing_structure.py"
_spec = importlib.util.spec_from_file_location("score_drawing_structure", _PATH)
tool = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tool)

STAMP = [0.55, 0.75, 0.98, 0.97]


def _sheet():
    items = [
        ("title_block", "Обозначение: АБВГ 100.001", STAMP),
        ("title_block", "АБВГ 100.001", [0.75, 0.79, 0.97, 0.82]),
        ("title_block", "Наименование: Планка", STAMP),
        ("title_block", "Планка", [0.73, 0.85, 0.82, 0.89]),
        ("title_block", "Масштаб: 1:1", STAMP),
        ("title_block", "1:1", [0.94, 0.86, 0.97, 0.88]),
        ("title_block", "Листов: 1", STAMP),
        ("title_block", "Разраб.: Иванов", STAMP),
        ("material", "Материал: Сталь 20", STAMP),
        ("title_block", "Сталь 20", [0.75, 0.92, 0.85, 0.94]),
        ("note", "1. * Размеры для справок.", [0.58, 0.52, 0.7, 0.54]),
        ("note", "2. Неуказанные предельные отклонения размеров: H14.", [0.58, 0.54, 0.8, 0.56]),
        ("note", "3. Покрытие по ОСТ 1.", [0.58, 0.56, 0.7, 0.58]),
        ("callout", "Б (4:1) 2 места", [0.9, 0.33, 0.94, 0.38]),
        ("callout", "Б (4:1)", [0.9, 0.34, 0.95, 0.37]),
        ("callout", "Б", [0.56, 0.09, 0.58, 0.12]),
        ("callout", "А (1:2)", [0.25, 0.30, 0.30, 0.32]),
        ("callout", "A (1:2) Отверстия условно", [0.25, 0.30, 0.45, 0.35]),
        ("callout", "Торец п.3", [0.16, 0.11, 0.24, 0.15]),
        ("size", "⌀6,5", [0.40, 0.14, 0.42, 0.16]),
        ("callout", "4 отв.", [0.40, 0.17, 0.42, 0.19]),
        ("size", "п.2", [0.44, 0.14, 0.46, 0.16]),
        ("size", "20±0,1", [0.50, 0.30, 0.54, 0.32]),
        ("size", "12*", [0.30, 0.40, 0.32, 0.44]),
        ("radius", "R1", [0.88, 0.40, 0.90, 0.42]),
    ]
    return {"sheet_id": "X1", "kind": "detail", "page": 1, "format": "А4",
            "items": [{"category": c, "value": v, "bbox": b} for c, v, b in items]}


def test_parse_dimension_kinds_and_tolerances():
    assert tool.parse_dimension("⌀20H7")["kind"] == "diameter"
    assert tool.parse_dimension("⌀20H7")["tolerance"] == {"fit": "H7"}
    assert tool.parse_dimension("15±0,1")["tolerance"] == {"upper": 0.1, "lower": -0.1}
    thread = tool.parse_dimension("М20x1,5")
    assert thread["kind"] == "thread" and thread["tolerance"] is None
    formula = tool.parse_dimension("(X+38)±0,2")
    assert formula["formula"] == "X+38" and formula["nominal"] is None
    assert tool.parse_dimension("R2*2")["marker"] == "*2"
    assert tool.parse_dimension("12*")["marker"] == "*"


def test_build_reference_structure():
    ref = tool.build_sheet(_sheet())
    fields = ref["title_block"]["fields"]
    # Значение из сводного поля, рамка — из ячейки штампа.
    assert fields["designation"]["value"] == "АБВГ 100.001"
    assert fields["designation"]["bbox"] == [0.75, 0.79, 0.97, 0.82]
    assert fields["material"]["value"] == "Сталь 20"
    assert fields["sheets"]["value"] == "1" and fields["format"]["value"] == "А4"
    assert fields["mass"]["needs_human"] and "title_block.mass" in ref["human_labelling"]
    assert ref["title_block"]["roles"][0]["personal"] is True
    # Вид «Б» прочитан дважды — один вид-выносной элемент на 2 места; «Б» у стрелки — указатель.
    detail = [v for v in ref["views"] if v["label"] == "Б"]
    assert len(detail) == 1 and detail[0]["kind"] == "detail" and detail[0]["count"] == 2
    assert [v["kind"] for v in ref["views"]] == ["main", "detail", "view"]  # «А» и латинская «A»
    assert ref["view_pointers"] == [{"label": "Б", "bbox": [0.56, 0.09, 0.58, 0.12]}]
    assert [n["number"] for n in ref["notes"]] == [1, 2, 3]
    assert ref["markers"] == {"*": 1}
    assert ref["not_sizes"] == ["п.2"]
    values = [d["value_raw"] for d in ref["dimensions"]]
    assert "п.2" not in values and "⌀6,5" in values
    hole = next(d for d in ref["dimensions"] if d["value_raw"] == "⌀6,5")
    assert hole["feature"] == "hole" and hole["count"] == 4 and hole["note_refs"] == [2]
    kinds = {(link["type"], link["to"], link["status"]) for link in ref["links"]}
    assert ("note_ref", 3, "derived") in kinds  # «Торец п.3»
    assert ("marker", 1, "derived") in kinds  # «12*» → примечание 1
    assert ("feature", "fillet", "derived") in kinds  # R1


def _score(sheet, rows):
    reference = {sheet["sheet_id"]: sheet}
    structure = {sheet["sheet_id"]: tool.build_sheet(sheet)}
    return tool.score(reference, structure, rows)


def test_flat_output_crop_copy_and_double_stamp():
    sheet = _sheet()
    items = [dict(i) for i in sheet["items"] if i["value"] != "п.2"]
    # Копия из вырезки вида: то же значение в координатах вырезки.
    items.append({"category": "size", "value": "20±0,1", "bbox": [0.10, 0.80, 0.14, 0.82]})
    # Ссылка на пункт выдана размером, вариант прочтения фамилии на том же месте.
    items.append({"category": "size", "value": "п.2", "bbox": [0.44, 0.14, 0.46, 0.16]})
    items.append({"category": "title_block", "value": "Иванав", "bbox": [0.65, 0.86, 0.72, 0.88]})
    items.append({"category": "title_block", "value": "Иванов", "bbox": [0.65, 0.86, 0.72, 0.88]})
    result = _score(sheet, [{"sheet_id": "X1", "arm": "flat", "run": 1, "items": items}])
    row = result["flat"]["*"]
    assert row == result["flat"]["detail"]
    assert row["recall"] == 1.0 and row["precision"] < 1.0
    assert row["placement"] == 1.0  # лишняя копия не сопоставлена, сопоставленные на месте
    assert row["dup_excess"] > 0 and row["title_dup"] >= 4
    assert row["variants"] == 1 and row["noteref_as_size"] == 1
    assert row["links_emitted"] == 0 and row["link_precision"] is None
    assert row["link_coverage"] == 0.0


def test_misplaced_crop_field_fails_placement():
    sheet = _sheet()
    items = [dict(i) for i in sheet["items"]]
    for item in items:
        if item["value"] == "R1":
            item["bbox"] = [0.10, 0.10, 0.12, 0.12]  # рамка вырезки, не листа
    row = _score(sheet, [{"sheet_id": "X1", "arm": "flat", "items": items}])["flat"]["*"]
    assert row["placement"] < 1.0


def test_structure_reading_links_and_title():
    sheet = _sheet()
    structure = {
        "schema": "drawing-sheet-v1",
        "title_block": {"fields": {
            "designation": {"value": "АБВГ 100.001", "bbox": [0.75, 0.79, 0.97, 0.82]},
            "name": {"value": "Планка", "bbox": [0.73, 0.85, 0.82, 0.89]},
            "material": {"value": "Сталь 20", "bbox": [0.75, 0.92, 0.85, 0.94]},
            "scale": {"value": "1:1", "bbox": [0.94, 0.86, 0.97, 0.88]},
            "sheets": {"value": "1"}, "format": {"value": "А4"}}},
        "views": [{"id": "view-1", "label": "Б", "title_raw": "Б (4:1) 2 места",
                   "title_bbox": [0.9, 0.33, 0.94, 0.38]}],
        "features": [{"id": "feature-1", "kind": "fillet"}],
        "dimensions": [
            {"value_raw": "R1", "kind": "radius", "bbox": [0.88, 0.40, 0.90, 0.42],
             "feature_id": "feature-1", "view_id": "view-1"},
            {"value_raw": "12*", "kind": "linear", "bbox": [0.30, 0.40, 0.32, 0.44]},
        ],
        "notes": [{"number": 1, "text": "* Размеры для справок.", "defines_marker": "*",
                   "bbox": [0.58, 0.52, 0.7, 0.54]}],
        # Неверная связь: «Торец п.3» отнесён к пункту 2.
        "texts": [{"value": "Торец п.3", "bbox": [0.16, 0.11, 0.24, 0.15], "note_refs": [2]}],
    }
    rows = [{"sheet_id": "X1", "arm": "structure", "sheet": structure}]
    row = _score(sheet, rows)["structure"]["*"]
    assert row["title_typed"] == 1.0 and row["title_dup"] == 0
    # Верно: R1 → скругление, «*» → п.1; неверно: «Торец п.3» → 2. Вид без эталона не судится.
    assert row["link_precision"] == round(2 / 3, 3)
    assert row["links_emitted"] == 3


def test_cli_build_and_score(tmp_path, capsys):
    ref = tmp_path / "ref.jsonl"
    ref.write_text(json.dumps(_sheet(), ensure_ascii=False) + "\n", encoding="utf-8")
    out = tmp_path / "struct.jsonl"
    tool.main(["build", "--reference", str(ref), "--out", str(out)])
    reading = tmp_path / "read.jsonl"
    reading.write_text(json.dumps({"sheet_id": "X1", "arm": "a", "items": _sheet()["items"]},
                                  ensure_ascii=False) + "\n", encoding="utf-8")
    summary = tmp_path / "s.json"
    tool.main(["score", "--reference", str(ref), "--structure", str(out),
               "--readings", str(reading), "--json", str(summary)])
    assert "== a" in capsys.readouterr().out
    assert json.loads(summary.read_text(encoding="utf-8"))["a"]["*"]["recall"] == 1.0
