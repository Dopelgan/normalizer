"""
Чистота записи чертежа: учёт разбора не уходит в индексируемый текст
(`DRAWING_DIAGNOSTICS_NOT_INDEXED`), а фраза не получает размерной
категории по знаку внутри себя (`DRAWING_CATEGORY_PROSE_GUARD`).

Значения в тестах настоящие: надписи и ключи взяты из переобработанного
14.11 на `fd0c0e5` документа ТАДУ 687000.001.001 «Крышка (изм. 1)» и из
замера по 327 документам (42 чанка из 19 699 — это рамки надписей и
причины непривязки в искомом тексте; полей-примечаний с размерной
категорией — 10).

Флаги ставятся через `monkeypatch.setattr` на объявленное поле `Settings`:
прямое присваивание оставило бы флаг включённым до конца прогона.
"""

import pytest

from core.config import settings
from core.contract_finalize import DIAGNOSTICS_KEY, finalize_fragments
from core.models.contract import Content, DrawingField, Fragment, Position, Provenance
from core.providers.vector_drawing import classify, decisive_category

# Блок привязки величин к видам, как его отдаёт `drawing_views.attach`:
# именно эти ключи и становились искомым текстом.
VALUE_LINKS = {
    "coordinate_reference": "deskewed_page",
    "views": [{"view_id": "v1", "name": "Q (1:2,5)", "cells": [[0, 0]]}],
    "fields": [
        {"field_index": 1, "value": "Линия сгиба", "category": "note",
         "bbox": [0.49, 0.03, 0.61, 0.07], "view": None,
         "basis": "unattributed", "why": "outside_views"},
    ],
    "counts": {"linked": 0, "unlinked": 12, "ambiguous": 0, "no_coordinates": 0},
    "signals": {"titles": 3},
}


def drawing_fragment() -> Fragment:
    """Фрагмент чертежа с учётом разбора и с тем, что читает потребитель."""
    return Fragment(
        fragment_id="",
        type="drawing",
        content=Content(
            text="⌀8,5\n15±0,1\nЛиния сгиба",
            structured_drawing_fields=[
                DrawingField(category="size", value="⌀8,5", confidence=0.95, confirmed=True),
            ],
            structured_payload={
                # Читает потребитель.
                "title_block": {"designation": "ТАДУ 687000.001.001", "mass": "26,02"},
                "tables": [{"headers": ["Поз."], "rows": [["1"]]}],
                "sheet_reparse_candidate": True,
                "sheet_reparse_reason": "sheet_answer_cut",
                "transcript_line_count": 3,
                "recognition_mode": "sheet_only",
                # Наш учёт разбора.
                "value_links": VALUE_LINKS,
                "sheet_answer_cut": True,
                "sheet_annotations_lost": 4,
                "regions_read": 2,
                "model": "qwen2.5-vl",
                "side_column": {"bbox": [0, 0, 0.1, 1], "read": True},
            },
        ),
        position=Position(page=1, bbox=[0, 0, 1, 1]),
        confidence=0.8,
        completeness=None,
        provenance=Provenance(method="vlm_full_page", strategy_level=7, source="drawing"),
    )


@pytest.fixture
def aside(monkeypatch):
    monkeypatch.setattr(settings, "DRAWING_DIAGNOSTICS_NOT_INDEXED", True)


@pytest.fixture
def aside_off(monkeypatch):
    monkeypatch.setattr(settings, "DRAWING_DIAGNOSTICS_NOT_INDEXED", False)


class TestDiagnosticsAside:
    """Учёт разбора — в `completeness_mechanisms`, а не в payload."""

    def test_flag_off_keeps_payload(self, aside_off):
        fragments, _ = finalize_fragments([drawing_fragment()], doc_id="d1")
        payload = fragments[0].content.structured_payload
        assert "value_links" in payload
        assert DIAGNOSTICS_KEY not in (fragments[0].completeness_mechanisms or {})

    def test_links_leave_payload(self, aside):
        fragments, _ = finalize_fragments([drawing_fragment()], doc_id="d1")
        payload = fragments[0].content.structured_payload
        for key in ("value_links", "sheet_answer_cut", "sheet_annotations_lost",
                    "regions_read", "model", "side_column"):
            assert key not in payload, key

    def test_consumer_keys_stay(self, aside):
        fragments, _ = finalize_fragments([drawing_fragment()], doc_id="d1")
        payload = fragments[0].content.structured_payload
        # Чанкер, очередь перечитывания, качество и запись читают эти ключи.
        for key in ("title_block", "tables", "sheet_reparse_candidate",
                    "sheet_reparse_reason", "transcript_line_count", "recognition_mode"):
            assert key in payload, key

    def test_diagnostics_kept_aside(self, aside):
        fragments, _ = finalize_fragments([drawing_fragment()], doc_id="d1")
        kept = (fragments[0].completeness_mechanisms or {}).get(DIAGNOSTICS_KEY) or {}
        # Данные не выброшены: по ним видно, почему величина без вида.
        assert kept["value_links"]["counts"]["unlinked"] == 12
        assert kept["value_links"]["fields"][0]["why"] == "outside_views"
        assert kept["sheet_annotations_lost"] == 4

    def test_text_without_diagnostics(self, aside):
        """Ни одного следа учёта в том, что уходит в индекс."""
        fragments, _ = finalize_fragments([drawing_fragment()], doc_id="d1")
        indexed = (fragments[0].content.text or "") + repr(
            fragments[0].content.structured_payload)
        for trace in ("bbox", "outside_views", "unattributed", "field_index",
                      "unlinked", "no_coordinates"):
            assert trace not in indexed, trace

    def test_idempotent(self, aside):
        fragments, _ = finalize_fragments([drawing_fragment()], doc_id="d1")
        payload = dict(fragments[0].content.structured_payload or {})
        kept = dict((fragments[0].completeness_mechanisms or {})[DIAGNOSTICS_KEY])
        again, _ = finalize_fragments(fragments, doc_id="d1")
        assert dict(again[0].content.structured_payload or {}) == payload
        assert dict((again[0].completeness_mechanisms or {})[DIAGNOSTICS_KEY]) == kept

    def test_text_fragment_untouched(self, aside):
        text = Fragment(
            fragment_id="", type="text",
            content=Content(text="Технические требования", structured_payload={"any": 1}),
            position=Position(page=1, bbox=[0, 0, 1, 1]),
            confidence=0.9, completeness=None,
            provenance=Provenance(method="text_layer", strategy_level=1, source="pdf"),
        )
        fragments, _ = finalize_fragments([text], doc_id="d1")
        assert fragments[0].content.structured_payload == {"any": 1}


@pytest.fixture
def guard(monkeypatch):
    monkeypatch.setattr(settings, "DRAWING_CATEGORY_PROSE_GUARD", True)


@pytest.fixture
def guard_off(monkeypatch):
    monkeypatch.setattr(settings, "DRAWING_CATEGORY_PROSE_GUARD", False)


# Настоящие пункты требований с листа ТАДУ 687000.001.001 и короткие
# надписи того же листа.
NOTES = [
    "5. Координаты отверстий ⌀8,5 и их количество N определяется схемой "
    "размеров ТАДУ 687000.001.001-XXXX Д2",
    "8. Лицевую сторону шлифовать, шерох. не хуже чем Ra 1,6",
]
VALUES = [("⌀8,5", "size"), ("Ra 1,6", "roughness"), ("⌀8,5 N отв.", "size"),
          ("R2", "radius"), ("15±0,1", "tolerance"), ("M10x1,25", "thread")]


class TestProseGuard:
    """Примечание — не размер и не шероховатость."""

    @pytest.mark.parametrize("text,category", [(NOTES[0], "size"), (NOTES[1], "roughness")])
    def test_flag_off_mislabels(self, guard_off, text, category):
        """Без флага разбор прежний — тот, что дал прод."""
        assert classify(text)[0] == category

    @pytest.mark.parametrize("text", NOTES)
    def test_note_stays_note(self, guard, text):
        assert classify(text)[0] == "note"
        assert decisive_category(text) is None

    @pytest.mark.parametrize("text,category", VALUES)
    def test_values_keep_category(self, guard, text, category):
        """Короткая надпись категорию не теряет — чинится только фраза."""
        assert classify(text)[0] == category

    @pytest.mark.parametrize("text", [
        "2. Материал — сталь 12Х18Н10Т ГОСТ 5949-75, круг 100 ГОСТ 7417-75",
    ])
    def test_material_prose_stays_material(self, guard, text):
        """Марка названа словом: её фраза называет верно и до флага."""
        assert classify(text)[0] == "material"

    def test_it_grade_prose_stays_tolerance(self, guard):
        text = "Неуказанные предельные отклонения размеров H14; h14; ±IT14/2"
        assert classify(text)[0] == "tolerance"

    @pytest.mark.parametrize("strict", [False, True])
    def test_both_modes(self, guard, monkeypatch, strict):
        """Строгий разбор знака фразу тоже не метит."""
        monkeypatch.setattr(settings, "DRAWING_FIELD_CATEGORY_STRICT", strict)
        for text in NOTES:
            assert classify(text)[0] == "note"
