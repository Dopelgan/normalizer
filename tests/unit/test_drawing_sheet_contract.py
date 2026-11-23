"""
Схема структуры листа `drawing-sheet-v1` и ключ `Content.drawing_sheet`.

Листы синтетические: данных заказчика в публичном репозитории нет.
"""

import json

import pytest
from pydantic import ValidationError

from core.contract_finalize import stable_fragment_id
from core.models.contract import Content, Fragment, Provenance
from core.models.drawing_sheet import (
    SCHEMA_VERSION,
    DrawingSheet,
    TitleRole,
    cap_sheet_size,
    element_anchor,
    sheet_size_bytes,
)

# Фрагмент чертежа в том виде, как его отдавали до структуры листа.
OLD_DRAWING_CONTENT = {
    "text": "Крышка. Отверстие ⌀8,5",
    "table_data": None,
    "image_ref": "s3://bucket/sheet-1.png",
    "formula_mathml": None,
    "parsed_expression": None,
    "structured_drawing_fields": [
        {
            "category": "size",
            "value": "⌀8,5",
            "tolerance": None,
            "nature": "executive",
            "unit": "мм",
            "confidence": 0.7,
            "confirmed": False,
            "bbox": [0.1, 0.2, 0.15, 0.22],
            "source_of_tolerance": "unknown",
            "provenance": "detected_from_annotation",
        }
    ],
    "structured_payload": {"layout": {"views": []}},
}


def _base(prefix: str, n: int, raw: str, bbox, /, **extra):
    return {
        "id": f"{prefix}-{n}",
        "anchor": element_anchor(prefix, raw, bbox),
        "bbox": bbox,
        "provenance": {"source": "vlm", "passes": ["sheet"]},
        "confidence": 0.7,
        **extra,
    }


def _sheet(**overrides) -> DrawingSheet:
    data = {
        "page": 0,
        "sheet_type": "detail",
        "frame": {"angle": 0.4, "turn": 90, "format": "A3", "format_confidence": 0.8,
                  "inner_frame": [0.02, 0.02, 0.98, 0.98], "page_px": [4961, 3508]},
        "title_block": {
            "bbox": [0.6, 0.8, 0.98, 0.98],
            "fields": {"designation": _base("title", 1, "АБВГ.000001.001", [0.7, 0.85, 0.9, 0.88],
                                            value="АБВГ.000001.001")},
            "roles": [{"role": "Разраб.", "name": "Иванов", "date": "01.02.26"}],
            "changes": [{"rev": "1", "doc": "АБВГ.12-26"}],
        },
        "views": [_base("view", 1, "А-А", [0.1, 0.1, 0.5, 0.6], label="А-А", kind="section")],
        "features": [_base("feature", 1, "hole", [0.2, 0.2, 0.25, 0.25], kind="hole", count=4,
                           derived_from=["dimension-1"])],
        "dimensions": [_base(
            "dimension", 1, "⌀8,5", [0.18, 0.15, 0.24, 0.17],
            value_raw="⌀8,5", kind="diameter", nominal=8.5, count=4,
            tolerance={"upper": 0.1, "lower": -0.1, "source": "explicit"},
            formula={"expr": "(X+38)", "variables": ["X"]},
            view_id="view-1", feature_id="feature-1",
            leader=[[0.2, 0.16], [0.22, 0.22]], link={"method": "rule", "confidence": 0.6},
            note_refs=["note-1"],
            alternatives=[{"value": "⌀8.5", "source": "ocr", "pass": "transcribe", "confidence": 0.4}],
            field_refs=[{"fragment": 0, "field": 0}],
        )],
        "positions": [_base("position", 1, "3", [0.3, 0.3, 0.32, 0.32], number="3",
                            table_row={"table_id": "table-1", "row": 2})],
        "notes": [_base("note", 1, "Размеры для справок", [0.6, 0.5, 0.9, 0.52], number=1,
                        text="* Размеры для справок", defines_marker="*",
                        refs={"dimensions": ["dimension-1"]})],
        "tables": [_base("table", 1, "spec", [0.6, 0.1, 0.98, 0.4], kind="spec", headers=["Поз."],
                         rows=[{"cells": [{"text": "3", "bbox": [0.6, 0.1, 0.62, 0.12]}]}])],
        "symbols": [_base("symbol", 1, "Ra 6,3", [0.9, 0.02, 0.97, 0.06], kind="roughness",
                          value="Ra 6,3", scope="general")],
        "texts": [_base("text", 1, "Лицевая сторона", [0.4, 0.4, 0.5, 0.42], text="Лицевая сторона")],
        "stats": {"elements": 9, "alternatives": 1},
    }
    data.update(overrides)
    return DrawingSheet.model_validate(data)


# ---------------------------------------------------------------------------
# Ключ в контракте
# ---------------------------------------------------------------------------

def test_key_absent_when_unset():
    content = Content(text="x")
    assert "drawing_sheet" not in content.model_dump()
    assert "drawing_sheet" not in content.model_dump(mode="json")
    assert "drawing_sheet" not in json.loads(content.model_dump_json())


def test_old_payload_validates_and_serializes_unchanged():
    content = Content(**OLD_DRAWING_CONTENT)
    assert content.drawing_sheet is None
    assert content.model_dump(mode="json") == OLD_DRAWING_CONTENT


def test_fragment_id_does_not_depend_on_structure():
    def fragment(sheet):
        return Fragment(
            fragment_id="tmp", type="drawing", content=Content(**OLD_DRAWING_CONTENT, drawing_sheet=sheet),
            confidence=0.8, completeness=1.0,
            provenance=Provenance(method="vlm", strategy_level=7, source="drawing"),
        )

    assert stable_fragment_id("doc", fragment(None)) == stable_fragment_id("doc", fragment(_sheet()))


def test_content_round_trip_with_sheet():
    sheet = _sheet()
    content = Content(**OLD_DRAWING_CONTENT, drawing_sheet=sheet.model_dump(mode="json"))
    dumped = content.model_dump(mode="json")
    assert dumped["drawing_sheet"] == sheet.model_dump(mode="json")
    # Хранилище Normalizer поднимает фрагмент из словаря — тот же результат.
    assert Content(**dumped).model_dump(mode="json") == dumped


# ---------------------------------------------------------------------------
# Форма структуры
# ---------------------------------------------------------------------------

def test_reserved_keys_named_as_in_schema():
    for by_alias in (False, True):
        data = _sheet().model_dump(mode="json", by_alias=by_alias)
        assert data["schema"] == SCHEMA_VERSION
        assert "schema_" not in data
        alternative = data["dimensions"][0]["alternatives"][0]
        assert alternative["pass"] == "transcribe"
        assert "pass_" not in alternative


def test_none_values_are_omitted():
    data = _sheet(title_block=None).model_dump(mode="json")
    assert "title_block" not in data
    position = data["positions"][0]
    assert "target_bbox" not in position and "view_id" not in position
    assert data["frame"]["coordinate_reference"] == "deskewed_page"


def test_unknown_keys_are_ignored_for_additive_changes():
    data = _sheet().model_dump(mode="json")
    data["future_key"] = {"x": 1}
    data["dimensions"][0]["future_key"] = 1
    sheet = DrawingSheet.model_validate(data)
    assert "future_key" not in sheet.model_dump(mode="json")


def test_unknown_sheet_type_becomes_unknown():
    assert _sheet(sheet_type="схема-деталь").sheet_type == "unknown"
    assert _sheet(sheet_type="assembly").sheet_type == "assembly"


def test_bbox_ordered_and_clamped():
    sheet = _sheet(views=[_base("view", 1, "А", [0.5, 0.6, -0.1, 1.2], kind="main")])
    assert sheet.views[0].bbox == [0.0, 0.6, 0.5, 1.0]


@pytest.mark.parametrize("patch", [
    {"bbox": [0.1, 0.2, 0.3]},
    {"id": "Dimension 1"},
    {"anchor": "abc"},
    {"provenance": {"source": "guess"}},
    {"confidence": 1.5},
    {"leader": [[0.1, 0.2, 0.3]]},
    {"kind": "length"},
])
def test_invalid_element_rejected(patch):
    dimension = _sheet().model_dump(mode="json")["dimensions"][0]
    dimension.update(patch)
    with pytest.raises(ValidationError):
        _sheet(dimensions=[dimension])


def test_wrong_schema_version_rejected():
    with pytest.raises(ValidationError):
        DrawingSheet.model_validate({"schema": "drawing-sheet-v2", "page": 0})


def test_roles_are_always_personal():
    assert TitleRole(role="Пров.", name="Петров").model_dump()["personal"] is True
    with pytest.raises(ValidationError):
        TitleRole(role="Пров.", name="Петров", personal=False)


# ---------------------------------------------------------------------------
# Отпечаток элемента
# ---------------------------------------------------------------------------

def test_anchor_stable_across_rereads():
    bbox = [0.10, 0.10, 0.12, 0.12]
    anchor = element_anchor("dimension", "⌀8,5", bbox)
    # Запятая/точка, пробелы и сдвиг рамки внутри клетки 1/64 — та же надпись.
    assert element_anchor("dimension", " ⌀ 8.5 ", [0.101, 0.1, 0.121, 0.121]) == anchor
    assert element_anchor("dimension", "⌀8,5", [0.12, 0.12, 0.10, 0.10]) == anchor
    assert len(anchor) == 40


def test_anchor_differs_by_kind_value_and_place():
    bbox = [0.10, 0.10, 0.12, 0.12]
    anchor = element_anchor("dimension", "R1", bbox)
    assert element_anchor("text", "R1", bbox) != anchor
    assert element_anchor("dimension", "R2", bbox) != anchor
    # Два одинаковых «R1» в разных местах вида — разные элементы.
    assert element_anchor("dimension", "R1", [0.5, 0.5, 0.52, 0.52]) != anchor


# ---------------------------------------------------------------------------
# Предел размера
# ---------------------------------------------------------------------------

def _texts(count: int, length: int = 200):
    return [
        _base("text", i + 1, f"t{i}", [0.01 * (i % 90), 0.5, 0.01 * (i % 90) + 0.01, 0.51],
              text="Ж" * length)
        for i in range(count)
    ]


def test_cap_keeps_small_sheet_untouched():
    sheet = _sheet()
    capped, fits = cap_sheet_size(sheet)
    assert fits and capped is sheet
    assert capped.stats.truncated is False


def test_cap_drops_leaders_first():
    sheet = _sheet()
    with_leaders = sheet_size_bytes(sheet)
    capped, fits = cap_sheet_size(sheet, limit=with_leaders - 1)
    assert fits
    assert capped.stats.truncated is True
    assert capped.dimensions[0].leader == []
    assert len(capped.texts) == len(sheet.texts)
    # Исходная структура не меняется.
    assert sheet.dimensions[0].leader and sheet.stats.truncated is False


def test_cap_drops_texts_from_the_end():
    sheet = _sheet(texts=_texts(400))
    assert sheet_size_bytes(sheet) > 48 * 1024
    capped, fits = cap_sheet_size(sheet)
    assert fits
    assert sheet_size_bytes(capped) <= 48 * 1024
    assert capped.stats.truncated is True
    assert 0 < len(capped.texts) < 400
    # Снимаются последние надписи, первые остаются по порядку.
    assert [t.id for t in capped.texts] == [t.id for t in sheet.texts[: len(capped.texts)]]
    # Остальное не тронуто.
    assert len(capped.dimensions) == len(sheet.dimensions) == 1
    assert capped.notes == sheet.notes


def test_cap_drops_only_as_many_texts_as_needed():
    sheet = _sheet(texts=_texts(60))
    limit = sheet_size_bytes(sheet) - 1
    capped, fits = cap_sheet_size(sheet, limit=limit)
    assert fits
    # Снятых линий хватает — надписи остаются все.
    assert len(capped.texts) == 60
    no_leaders = sheet_size_bytes(capped)
    capped, fits = cap_sheet_size(sheet, limit=no_leaders - 1)
    assert fits and len(capped.texts) == 59


def test_cap_reports_sheet_that_cannot_fit():
    notes = [_base("note", i + 1, f"n{i}", [0.1, 0.1, 0.2, 0.2], number=i + 1, text="Ж" * 400)
             for i in range(100)]
    capped, fits = cap_sheet_size(_sheet(notes=notes, texts=_texts(5)))
    assert not fits
    assert capped.texts == []
    assert capped.stats.truncated is True
