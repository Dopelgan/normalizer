"""Извлечение фактов и значений (раздел 4 изменений контракта)."""

from core.facts.extractor import (
    extract_facts,
    facts_from_drawing_fields,
    facts_from_table,
    facts_from_text,
    to_number,
)
from core.models.contract import Content, DrawingField, Fragment, Position, Provenance


def fragment(**kwargs) -> Fragment:
    defaults = dict(
        fragment_id="cluster-a/document.pdf:page-4:block-2",
        type="text",
        content=Content(text=""),
        position=Position(page=4, bbox=[0.1, 0.2, 0.3, 0.4], order=12),
        confidence=0.96,
        completeness=1.0,
        provenance=Provenance(method="pdf_text", strategy_level=1, source="vector_pdf"),
    )
    defaults.update(kwargs)
    return Fragment(**defaults)


class TestNumbers:
    def test_number_with_unit_and_operator(self):
        facts = facts_from_text("Максимальная температура — 80 °C.", "f")
        assert len(facts) == 1
        fact = facts[0]
        assert fact.label == "максимальная температура"
        assert fact.value.kind == "number"
        assert fact.value.raw == "80 °C"
        assert fact.value.number == 80
        assert fact.value.unit == "°C"
        assert fact.value.operator == "<="

    def test_range_keeps_both_bounds(self):
        fact = facts_from_text("Допустимый диапазон давления — 0,4–0,6 МПа.", "f")[0]
        assert fact.value.kind == "range"
        assert (fact.value.min, fact.value.max) == (0.4, 0.6)
        assert fact.value.unit == "MPa"
        assert fact.value.operator == "range"

    def test_from_to_is_one_range(self):
        facts = facts_from_text("Напряжение питания: от 210 до 240 В.", "f")
        assert len(facts) == 1
        assert (facts[0].value.min, facts[0].value.max, facts[0].value.unit) == (210, 240, "V")

    def test_tolerance_becomes_range(self):
        fact = facts_from_text("Толщина стенки 3 мм ± 0,2 мм.", "f")[0]
        assert fact.value.kind == "range"
        assert (round(fact.value.min, 3), round(fact.value.max, 3)) == (2.8, 3.2)

    def test_not_more_than_gives_le(self):
        fact = facts_from_text("Масса не более 12,5 кг.", "f")[0]
        assert fact.value.operator == "<="
        assert fact.value.number == 12.5
        assert fact.value.unit == "kg"

    def test_thousands_separator(self):
        assert to_number("1 200,5") == 1200.5

    def test_unit_is_not_invented(self):
        """Единицы нет в тексте — нет её и в факте."""
        fact = facts_from_text("Количество отверстий — 4.", "f")[0]
        assert fact.value.unit is None

    def test_raw_keeps_original_spelling(self):
        fact = facts_from_text("Давление 0,6 МПа.", "f")[0]
        assert fact.value.raw == "0,6 МПа"


class TestOtherKinds:
    def test_duration(self):
        fact = facts_from_text("Срок хранения — не более 30 суток.", "f")[0]
        assert fact.value.kind == "duration"
        assert fact.value.unit == "d"
        assert fact.label == "срок хранения"

    def test_date(self):
        kinds = {f.value.kind for f in facts_from_text("Утверждено 12.03.2026.", "f")}
        assert "date" in kinds

    def test_identifier(self):
        facts = facts_from_text("Изготовить по ГОСТ 2.109-73.", "f")
        assert facts[0].value.kind == "identifier"
        assert facts[0].value.raw == "ГОСТ 2.109-73"


class TestFactIdentity:
    def test_fact_id_built_from_fragment_id(self):
        facts = facts_from_text("Температура 80 °C.", "cluster-a/doc.pdf:page-4:block-2")
        assert facts[0].fact_id == "cluster-a/doc.pdf:page-4:block-2:fact-1"
        assert facts[0].fragment_id == "cluster-a/doc.pdf:page-4:block-2"

    def test_key_is_extensible_not_fixed(self):
        fact = facts_from_text("Температура 80 °C.", "f", key="drawing.dimension")[0]
        assert fact.key == "drawing.dimension"


class TestTables:
    def test_cell_keeps_row_and_column(self):
        facts = facts_from_table(
            {"headers": ["Параметр", "Значение"],
             "rows": [["Давление", "0,4–0,6 МПа"], ["Температура", "80 °C"]]},
            "f",
        )
        assert [f.label for f in facts] == ["Давление", "Температура"]
        assert facts[0].provenance == {"row": 0, "column": 1, "header": "Значение"}
        assert facts[1].value.number == 80

    def test_empty_table_gives_nothing(self):
        assert facts_from_table({"headers": [], "rows": []}, "f") == []


class TestDrawings:
    def test_field_becomes_fact_with_category_key(self):
        fields = [DrawingField(category="diameter", value="20", unit="мм", confidence=0.9)]
        fact = facts_from_drawing_fields(fields, "f")[0]
        assert fact.key == "drawing.diameter"
        assert fact.value.number == 20
        assert fact.value.unit == "mm"

    def test_tolerance_becomes_range(self):
        fields = [DrawingField(category="diameter", value="20", tolerance="±0,1", unit="мм")]
        fact = facts_from_drawing_fields(fields, "f")[0]
        assert (round(fact.value.min, 3), round(fact.value.max, 3)) == (19.9, 20.1)


class TestDispatch:
    def test_text_fragment(self):
        facts = extract_facts(fragment(content=Content(text="Температура 80 °C.")))
        assert facts and facts[0].value.number == 80

    def test_image_fragment_has_no_facts(self):
        assert extract_facts(
            fragment(type="image", content=Content(image_ref="assets/doc/1.png"))
        ) == []
