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


class TestRecordWithoutSheetText:
    """
    Лист без текста записывает Normalizer, а не адаптер RAG
    (`DRAWING_RECORD_WITHOUT_SHEET_TEXT`).

    Надписи здесь настоящие — с листа ТАДУ 305179.001.03 СБ (пробка
    ЭК150), `tests/fixtures/real_sheet_views.json`.
    """

    @staticmethod
    def textless() -> Fragment:
        from core.providers.vector_drawing import classify

        marks = ["⌀200±2", "250±2", "А(2:1)", "11", "8", "п.6", "В–В",
                 "Лицевую сторону шлифовать, шерох. не хуже чем Ra 1,6"]
        fields = []
        for value in marks:
            category, confidence = classify(value)
            fields.append(DrawingField(category=category, value=value,
                                       confidence=confidence, confirmed=False))
        return Fragment(
            fragment_id="", type="drawing",
            content=Content(text=None, structured_drawing_fields=fields,
                            structured_payload={"recognition_mode": "sheet_only"}),
            position=Position(page=1, bbox=[0, 0, 1, 1]),
            confidence=0.6, completeness=None,
            provenance=Provenance(method="vlm_full_page", strategy_level=7,
                                  source="drawing"),
        )

    @pytest.fixture
    def composed(self, monkeypatch):
        monkeypatch.setattr(settings, "DRAWING_RECORD_COMPOSED", True)

    @pytest.fixture
    def without_text(self, monkeypatch, composed):
        monkeypatch.setattr(settings, "DRAWING_RECORD_WITHOUT_SHEET_TEXT", True)

    def test_flag_off_leaves_text_empty(self, composed, monkeypatch):
        monkeypatch.setattr(settings, "DRAWING_RECORD_WITHOUT_SHEET_TEXT", False)
        fragments, _ = finalize_fragments([self.textless()], doc_id="d1")
        assert not fragments[0].content.text

    def test_record_composed(self, without_text):
        fragments, _ = finalize_fragments([self.textless()], doc_id="d1")
        text = fragments[0].content.text or ""
        assert "Размеры и обозначения величин на листе" in text

    def test_nothing_lost(self, without_text):
        """Ни одна надпись не исчезает — в том числе голое число."""
        source = self.textless()
        values = [field.value for field in source.content.structured_drawing_fields]
        fragments, _ = finalize_fragments([source], doc_id="d1")
        text = fragments[0].content.text or ""
        for value in values:
            assert value in text, value

    def test_bare_numbers_in_other_marks(self, without_text):
        """Голое число рода величины не получает, но из записи не уходит."""
        fragments, _ = finalize_fragments([self.textless()], doc_id="d1")
        text = fragments[0].content.text or ""
        head = "Остальные надписи листа"
        assert head in text
        tail = text[text.index(head):]
        assert "11" in tail and "8" in tail

    def test_caution_once_and_no_tails(self, without_text):
        """Осторожность — одной фразой на запись, а не припиской при значении."""
        fragments, _ = finalize_fragments([self.textless()], doc_id="d1")
        text = fragments[0].content.text or ""
        assert text.count("нужно сверить с самим чертежом") == 1
        assert "(требует сверки)" not in text

    def test_confidence_stays_per_value(self, without_text):
        """Уверенность по каждому значению остаётся в полях фрагмента."""
        fragments, _ = finalize_fragments([self.textless()], doc_id="d1")
        fields = fragments[0].content.structured_drawing_fields
        # Фраза на запись — одна, а уверенность осталась у каждого значения
        # своя: её читает всякий, кому она нужна.
        assert fields and all(field.confidence is not None for field in fields)
        assert len({field.confidence for field in fields}) > 1

    def test_idempotent(self, without_text):
        fragments, _ = finalize_fragments([self.textless()], doc_id="d1")
        once = fragments[0].content.text
        again, _ = finalize_fragments(fragments, doc_id="d1")
        assert again[0].content.text == once

    def test_empty_fragment_untouched(self, without_text):
        """Собирать нечего — фрагмент остаётся как был."""
        bare = Fragment(
            fragment_id="", type="drawing",
            content=Content(text=None, structured_drawing_fields=[], structured_payload={}),
            position=Position(page=1, bbox=[0, 0, 1, 1]),
            confidence=0.1, completeness=None,
            provenance=Provenance(method="vlm_full_page", strategy_level=7, source="drawing"),
        )
        fragments, _ = finalize_fragments([bare], doc_id="d1")
        assert not fragments[0].content.text


class TestRecordUsesValueLinks:
    """Запись называет вид при размере, когда привязка есть в данных."""

    @staticmethod
    def linked() -> Fragment:
        fields = [
            DrawingField(category="size", value="⌀200", tolerance="±2",
                         confidence=0.95, confirmed=True),
            DrawingField(category="size", value="⌀300", tolerance="±2",
                         confidence=0.95, confirmed=True),
        ]
        payload = {
            "recognition_mode": "sheet_only",
            "value_links": {
                "fields": [
                    {"field_index": 0, "value": "⌀200", "view": "v1",
                     "view_name": "А (2:1)", "basis": "inside"},
                    {"field_index": 1, "value": "⌀300", "view": None,
                     "basis": "unattributed", "why": "outside_views"},
                ],
                "counts": {"linked": 1, "unlinked": 1, "ambiguous": 0,
                           "no_coordinates": 0},
            },
        }
        return Fragment(
            fragment_id="", type="drawing",
            content=Content(text="⌀200±2\n⌀300±2", structured_drawing_fields=fields,
                            structured_payload=payload),
            position=Position(page=1, bbox=[0, 0, 1, 1]),
            confidence=0.95, completeness=None,
            provenance=Provenance(method="vlm_full_page", strategy_level=7,
                                  source="drawing"),
        )

    @pytest.fixture
    def uses_links(self, monkeypatch):
        monkeypatch.setattr(settings, "DRAWING_RECORD_COMPOSED", True)
        monkeypatch.setattr(settings, "DRAWING_RECORD_USES_VALUE_LINKS", True)

    @pytest.fixture
    def links_off(self, monkeypatch):
        monkeypatch.setattr(settings, "DRAWING_RECORD_COMPOSED", True)
        monkeypatch.setattr(settings, "DRAWING_RECORD_USES_VALUE_LINKS", False)

    def test_flag_off_says_nothing_known(self, links_off):
        fragments, _ = finalize_fragments([self.linked()], doc_id="d1")
        text = fragments[0].content.text or ""
        assert "вид А (2:1)" not in text
        assert "запись чертежа не указывает" in text

    def test_view_named_at_value(self, uses_links):
        fragments, _ = finalize_fragments([self.linked()], doc_id="d1")
        text = fragments[0].content.text or ""
        assert "⌀200 (допуск ±2) (вид А (2:1))" in text

    def test_hedge_speaks_of_unlinked_only(self, uses_links):
        fragments, _ = finalize_fragments([self.linked()], doc_id="d1")
        text = fragments[0].content.text or ""
        assert "Там, где вид не назван при размере" in text
        assert "относится каждый размер, запись чертежа не указывает" not in text

    def test_mismatched_index_ignored(self, uses_links):
        """Номер поля сдвинулся — привязка не используется: чужой вид хуже пустоты."""
        fragment = self.linked()
        payload = dict(fragment.content.structured_payload)
        payload["value_links"] = dict(payload["value_links"])
        payload["value_links"]["fields"] = [
            {"field_index": 0, "value": "⌀999", "view": "v1", "view_name": "А (2:1)"},
        ]
        fragment.content.structured_payload = payload
        fragments, _ = finalize_fragments([fragment], doc_id="d1")
        assert "(вид" not in (fragments[0].content.text or "")


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


class TestAllFieldsSaid:
    """
    Запись произносит каждое значение, поэтому адаптеру RAG нечего
    дописывать (`DRAWING_RECORD_ALL_FIELDS_SAID`,
    `DRAWING_RECORD_STAMP_FROM_FIELDS`).

    Значения — с настоящего листа ТАДУ 421451.003.07.003 «Табличка»,
    переобработанного 14.11 с семью флагами: именно на нём хвостов стало
    больше (13 → 29), потому что сборка собрала лист в один фрагмент, а
    значения, которых запись не назвала, адаптер дописал строками
    «Основная надпись: Фарфурат (требует сверки)».
    """

    @staticmethod
    def sheet() -> Fragment:
        fields = [
            # Графа штампа с именем: запись назовёт её графой.
            DrawingField(category="title_block", value="Разраб.: Бабанин",
                         confidence=0.85, confirmed=False),
            # Графа штампа без имени: имени по догадке не получает.
            DrawingField(category="title_block", value="Фарфурат",
                         confidence=0.85, confirmed=False),
            # Голое число: род величины ему не назвать (`_kind_of`).
            DrawingField(category="size", value="5", confidence=0.8, confirmed=False),
            DrawingField(category="size", value="5±1", confidence=0.8, confirmed=False),
            DrawingField(category="position", value="п.4", confidence=0.8,
                         confirmed=False),
            DrawingField(category="callout", value="4 места", confidence=0.8,
                         confirmed=False),
        ]
        payload = {
            "recognition_mode": "sheet_only",
            "title_block": {"designation": "ТАДУ 421451.003.07.003"},
            "tables": [{"kind": "other", "headers": ["Изм.", "Лист"],
                        "rows": [["1", "301"], ["2", "302"]]}],
        }
        return Fragment(
            fragment_id="", type="drawing",
            content=Content(text="ТАДУ 421451.003.07.003\n5±1\n4 места",
                            structured_drawing_fields=fields,
                            structured_payload=payload),
            position=Position(page=1, bbox=[0, 0, 1, 1]),
            confidence=0.8, completeness=None,
            provenance=Provenance(method="vlm_full_page", strategy_level=7,
                                  source="drawing"),
        )

    @pytest.fixture
    def said(self, monkeypatch):
        monkeypatch.setattr(settings, "DRAWING_RECORD_COMPOSED", True)
        monkeypatch.setattr(settings, "DRAWING_RECORD_ALL_FIELDS_SAID", True)
        monkeypatch.setattr(settings, "DRAWING_RECORD_STAMP_FROM_FIELDS", True)

    @pytest.fixture
    def said_off(self, monkeypatch):
        monkeypatch.setattr(settings, "DRAWING_RECORD_COMPOSED", True)
        monkeypatch.setattr(settings, "DRAWING_RECORD_ALL_FIELDS_SAID", False)
        monkeypatch.setattr(settings, "DRAWING_RECORD_STAMP_FROM_FIELDS", False)

    @staticmethod
    def values_of(fragment: Fragment):
        return [field.value for field in fragment.content.structured_drawing_fields]

    def test_flag_off_leaves_values_unsaid(self, said_off):
        """Без флага «5» и «Фарфурат» в тексте не называются — их и дописывал адаптер."""
        source = self.sheet()
        fragments, _ = finalize_fragments([source], doc_id="d1")
        from core.drawing_record import _whole_value_said

        text = fragments[0].content.text or ""
        assert not _whole_value_said("5", text)
        assert "Фарфурат" not in text

    def test_every_value_said_whole(self, said):
        """Каждое значение стоит в тексте целым значением — правилом адаптера."""
        from core.drawing_record import _whole_value_said

        source = self.sheet()
        fields = list(source.content.structured_drawing_fields)
        fragments, _ = finalize_fragments([source], doc_id="d1")
        text = fragments[0].content.text or ""
        for field in fields:
            value = field.value
            if field.category == "title_block" and ": " in value:
                # Адаптер RAG разбирает графу на подпись и значение
                # (`_DRAWING_LABELLED_VALUE`) и ищет в тексте значение,
                # а не строку «Разраб.: Бабанин» целиком.
                value = value.split(":", 1)[1].strip()
            assert _whole_value_said(value, text), value

    def test_table_rows_said(self, said):
        from core.drawing_record import _whole_value_said

        fragments, _ = finalize_fragments([self.sheet()], doc_id="d1")
        text = fragments[0].content.text or ""
        assert _whole_value_said("1\t301", text)
        assert _whole_value_said("2\t302", text)

    def test_named_graph_keeps_its_name(self, said):
        """Графа, которую поле назвало, стоит в записи графой, а не в остатке."""
        fragments, _ = finalize_fragments([self.sheet()], doc_id="d1")
        text = fragments[0].content.text or ""
        head = text.index("Основная надпись листа.")
        tail = text.index("Остальное, прочитанное на листе")
        assert "Бабанин" in text[head:tail]

    def test_unnamed_graph_not_invented(self, said):
        """Графа без имени имени не получает — уходит в последнюю часть."""
        fragments, _ = finalize_fragments([self.sheet()], doc_id="d1")
        text = fragments[0].content.text or ""
        tail = text[text.index("Остальное, прочитанное на листе"):]
        assert "Фарфурат" in tail

    def test_caution_still_once(self, said):
        fragments, _ = finalize_fragments([self.sheet()], doc_id="d1")
        text = fragments[0].content.text or ""
        assert text.count("нужно сверить с самим чертежом") == 1
        assert "(требует сверки)" not in text

    def test_parts_markup_contiguous(self, said):
        """Разметка частей — подряд и без зазоров, иначе чанкер её отвергнет."""
        fragments, _ = finalize_fragments([self.sheet()], doc_id="d1")
        text = fragments[0].content.text or ""
        parts = (fragments[0].content.structured_payload or {}).get("record_parts") or []
        assert parts and parts[0]["start"] == 0
        assert parts[-1]["end"] == len(text)
        for before, after in zip(parts, parts[1:]):
            assert before["end"] == after["start"]

    def test_idempotent(self, said):
        fragments, _ = finalize_fragments([self.sheet()], doc_id="d1")
        once = fragments[0].content.text
        again, _ = finalize_fragments(fragments, doc_id="d1")
        assert again[0].content.text == once
