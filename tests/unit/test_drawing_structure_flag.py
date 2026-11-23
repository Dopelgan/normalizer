"""
Структура листа в выдаче (флаг DRAWING_STRUCTURE, карточка N6).

Синтетический документ: лист целиком и вырезка вида на странице 2, абзац
распознавания той же страницы со строками, лист страницы 3. Данных
заказчика нет. Вызовов модели нет: фрагменты уже чертёжные.
"""

import copy
import json
import re

import pytest

from core import drawing_structure
from core.config import settings
from core.drawing_processor import DrawingProcessor
from core.drawing_structure import assemble
from core.ladder import result_cache
from core.models.contract import (
    Content,
    DimensionLink,
    DrawingField,
    Fragment,
    Position,
    Provenance,
    RecognitionLine,
)
from core.models.drawing_sheet import Dimension, DrawingSheet, Feature, TitleRole, View

STRUCTURE_FLAGS = ("DRAWING_STRUCTURE", "DRAWING_DIMENSION_LINKS", "DRAWING_FIELD_PLACEMENT_V2")


class _NoModel:
    """Модель не должна вызываться: любой вызов — падение теста."""

    available = False

    def __getattr__(self, name):
        raise AssertionError(f"вызов модели: {name}")


def _field(category, value, bbox, **extra):
    return DrawingField(category=category, value=value, bbox=bbox, confidence=0.8, **extra)


def _drawing(fid, page, fields, bbox=(0.0, 0.0, 1.0, 1.0), payload=None):
    return Fragment(
        fragment_id=fid, type="drawing",
        content=Content(text="Крышка", structured_drawing_fields=fields,
                        structured_payload=payload if payload is not None else {}),
        position=Position(page=page, bbox=list(bbox)), confidence=0.8, completeness=1.0,
        provenance=Provenance(method="vlm_full_page", strategy_level=7, source="drawing"),
    )


def _text(fid, page, lines):
    return Fragment(
        fragment_id=fid, type="text", content=Content(text=" ".join(line.text for line in lines)),
        position=Position(page=page, bbox=[0.0, 0.0, 1.0, 1.0], lines=lines),
        confidence=0.6, completeness=1.0,
        provenance=Provenance(method="tesseract", strategy_level=5, source="scan"),
    )


def _document(turn=0):
    link = DimensionLink(target_label="отверстие", target_bbox=[0.30, 0.40, 0.33, 0.43],
                         leader=[[0.31, 0.35], [0.31, 0.40]], method="vlm", confidence=0.6)
    whole = _drawing("whole", 2, [
        _field("size", "⌀8,5", [0.30, 0.33, 0.34, 0.35], link=link),
        _field("callout", "N отв.", [0.30, 0.355, 0.34, 0.37]),
        _field("size", "15±0,1", [0.50, 0.50, 0.53, 0.52]),
        _field("note", "1. Неуказанные предельные отклонения H14, h14.", [0.10, 0.70, 0.40, 0.72]),
        _field("title_block", "Масштаб: 1:1", [0.55, 0.75, 0.98, 0.97]),
    ], payload={"title_block_bbox": [0.55, 0.75, 0.98, 0.97], "sheet_type": "detail",
                "layout": {"angle": 0.0, "turn": turn, "frame": [0.02, 0.02, 0.98, 0.98]}})
    crop = _drawing("crop", 2, [_field("size", "15±0,1", [0.40, 0.50, 0.50, 0.55])],
                    bbox=(0.08, 0.11, 0.45, 0.65))
    lines = [
        RecognitionLine(text="⌀8,5", bbox=[0.305, 0.334, 0.331, 0.348], source="tesseract", confidence=0.9),
        # Строка соседней страницы соединённого абзаца — к листу не относится.
        RecognitionLine(text="15±0,1", bbox=[0.70, 0.10, 0.74, 0.12], source="tesseract", page=3),
    ]
    paragraph = _text("ocr", 2, lines)
    other = _drawing("p3", 3, [_field("size", "R5", [0.2, 0.2, 0.23, 0.22])])
    return [whole, crop, paragraph, other]


def _enrich(monkeypatch, fragments, flag):
    monkeypatch.setattr(settings, "DRAWING_STRUCTURE", flag)
    monkeypatch.setattr(settings, "DRAWING_READINGS_RECONCILED", False)
    monkeypatch.setattr(settings, "DRAWING_FIELD_PLACEMENT_V2", False)
    return DrawingProcessor(storage=object(), qwen=_NoModel()).enrich_fragments(fragments, {})


def _dump(fragments):
    return [json.loads(f.model_dump_json()) for f in fragments]


def test_flag_off_output_unchanged(monkeypatch):
    before = _dump(_document())
    out = _enrich(monkeypatch, _document(), False)
    assert _dump(out) == before
    assert all("drawing_sheet" not in d["content"] for d in _dump(out))


def test_flag_on_one_sheet_per_page_and_flat_output_unchanged(monkeypatch):
    off = _dump(_enrich(monkeypatch, _document(), False))
    out = _enrich(monkeypatch, _document(), True)
    on = _dump(out)
    carriers = [(f.position.page, f.fragment_id) for f in out if f.content.drawing_sheet is not None]
    assert carriers == [(2, "whole"), (3, "p3")]
    for before, after in zip(off, on):
        sheet = after["content"].pop("drawing_sheet", None)
        after.get("completeness_mechanisms", {}).get("drawing_diagnostics", {}).pop("structure", None)
        if not (after.get("completeness_mechanisms") or {}).get("drawing_diagnostics"):
            (after.get("completeness_mechanisms") or {}).pop("drawing_diagnostics", None)
        assert after == before
        if sheet is not None:
            # Структура сериализуется и поднимается обратно той же схемой.
            assert DrawingSheet.model_validate(sheet).page == after["position"]["page"]
    note = out[0].completeness_mechanisms["drawing_diagnostics"]["structure"]
    assert note["built"] is True and note["stats"]["elements"] > 0


def test_snap_takes_recognition_line_of_the_same_page(monkeypatch):
    out = _enrich(monkeypatch, _document(), True)
    sheet = out[0].content.drawing_sheet
    hole = next(d for d in sheet.dimensions if d.kind == "diameter")
    assert hole.bbox_exact and hole.bbox == [0.305, 0.334, 0.331, 0.348]
    size = next(d for d in sheet.dimensions if d.nominal == 15)
    # Строка «15±0,1» лежит на странице 3 — рамку размеру страницы 2 она не даёт.
    assert size.bbox[0] < 0.6


def test_turned_unplaced_sheet_ignores_recognition_lines(monkeypatch):
    out = _enrich(monkeypatch, _document(turn=90), True)
    sheet = out[0].content.drawing_sheet
    hole = next(d for d in sheet.dimensions if d.kind == "diameter")
    assert not hole.bbox_exact


def test_letter_count_kept_as_written(monkeypatch):
    out = _enrich(monkeypatch, _document(), True)
    sheet = out[0].content.drawing_sheet
    hole = next(d for d in sheet.dimensions if d.kind == "diameter")
    assert hole.count == "N" and hole.value_raw == "⌀8,5"
    feature = next(f for f in sheet.features if f.id == hole.feature_id)
    assert feature.kind == "hole" and feature.count == "N"


def test_assembler_failure_keeps_document(monkeypatch):
    def broken(*_args, **_kwargs):
        raise RuntimeError("сбой")

    monkeypatch.setattr(assemble, "build_sheet", broken)
    out = _enrich(monkeypatch, _document(), True)
    assert all(f.content.drawing_sheet is None for f in out)
    assert out[0].completeness_mechanisms["drawing_diagnostics"]["structure"] == {
        "built": False, "error": "RuntimeError"}


def test_nul_does_not_reach_the_structure(monkeypatch):
    fragments = _document()
    fragments[0].content.structured_drawing_fields.append(
        _field("callout", "Лицевая\x00 сторона", [0.6, 0.4, 0.7, 0.42]))
    out = drawing_structure.apply(copy.deepcopy(fragments))
    text = out[0].content.drawing_sheet.model_dump_json()
    assert "\\u0000" not in text and "Лицевая сторона" in text


def test_structure_flags_do_not_reach_ladder_cache():
    """
    Ветка чертежей идёт после попадания в кэш лестницы (`ingest/tasks.py`:
    `result_cache.parse` -> нормализация -> `enrich_fragments`), поэтому её
    флаги в ключ не входят: иначе каждый флаг сбрасывал бы MinerU и
    Tesseract всего корпуса. Падает, если флаг начнут читать в коде лестницы.
    """
    sources = result_cache.ladder_sources()
    names = {p.as_posix() for p in sources}
    assert not any(n.endswith("core/drawing_processor.py") or "/drawing_structure/" in n for n in names)
    for path in sources:
        text = path.read_text(encoding="utf-8")
        for flag in STRUCTURE_FLAGS:
            # Чтение — `settings.X` или `getattr(settings, "X")`; упоминание в тексте — нет.
            read = re.compile(rf"settings\.{flag}\b|getattr\(\s*settings\s*,\s*[\"']{flag}[\"']")
            assert not read.search(text), f"{flag} читается в {path.name}: нужен ключ кэша"
    assert not set(STRUCTURE_FLAGS) & set(result_cache.KEY_SETTINGS)


@pytest.mark.parametrize("value, expected", [(4, 4), ("4", 4), ("N", "N"), (" N ", "N"), (None, None)])
def test_count_accepts_letter(value, expected):
    base = {"id": "dimension-1", "anchor": "0" * 40, "bbox": [0.1, 0.1, 0.2, 0.2],
            "provenance": {"source": "vlm"}}
    assert Dimension(**base, value_raw="⌀8", kind="diameter", count=value).count == expected
    assert Feature(**{**base, "id": "feature-1"}, kind="hole", count=value).count == expected


@pytest.mark.parametrize("value", [0, -1, "", True, "x" * 17])
def test_count_rejects_garbage(value):
    with pytest.raises(ValueError):
        Dimension(id="dimension-1", anchor="0" * 40, bbox=[0.1, 0.1, 0.2, 0.2],
                  provenance={"source": "vlm"}, value_raw="⌀8", kind="diameter", count=value)


def test_view_title_bbox_and_role_readings_are_kept():
    view = View(id="view-1", anchor="0" * 40, bbox=[0.1, 0.1, 0.5, 0.5], provenance={"source": "vlm"},
                title_bbox=[0.2, 0.05, 0.3, 0.08])
    assert view.model_dump()["title_bbox"] == [0.2, 0.05, 0.3, 0.08]
    role = TitleRole(role="developed", name="Постанов", alternatives=["Постнов"],
                     bbox=[0.6, 0.8, 0.7, 0.82], field_refs=[{"fragment": 0, "field": 3}])
    dumped = role.model_dump()
    assert dumped["alternatives"] == ["Постнов"] and dumped["field_refs"] == [{"fragment": 0, "field": 3}]
    assert "bbox" not in TitleRole(role="checked", name="Глинка").model_dump()
