"""
Ветка чертежей: разметка зрением, чтение внешней моделью, сборка фрагмента.

Модель здесь поддельная — проверяется договор с ней (что спрашивается, как
разбирается ответ, что попадает в контракт), а не качество чтения.
"""

import pytest

from core.drawing_processor import (
    MODE_NONE, MODE_SHEET_ONLY, MODE_VISION_VLM, DrawingProcessor,
)
from core.models.contract import Content, Fragment, Position, Provenance
from core.providers import drawing_vision
from core.providers.qwen_client import SheetReading

pytest.importorskip("numpy")
pytest.importorskip("PIL")

from tests.fixtures import synthetic_sheet as sheet  # noqa: E402

SHEET_URI = "assets/doc-1/sheet.png"


class FakeQwen:
    """Модель, которая отвечает заготовленным и запоминает, о чём спросили."""

    def __init__(self, reading=None, title_block=None, table=None, model="fake-vl"):
        self.available = True
        self.model = model
        self.reading = reading
        self.title_block_answer = title_block
        self.table_answer = table
        self.asked = []

    def read_sheet(self, image, hint=""):
        self.asked.append(("sheet", image.size, hint))
        return self.reading

    def read_title_block(self, image):
        self.asked.append(("title_block", image.size, ""))
        return self.title_block_answer

    def read_table(self, image):
        self.asked.append(("table", image.size, ""))
        return self.table_answer


def reading(*annotations, sheet_type="detail"):
    return SheetReading(sheet_type=sheet_type, annotations=list(annotations))


def annotation(category, value, bbox=None):
    return {"category": category, "value": value, "bbox": bbox or [0.1, 0.1, 0.2, 0.2]}


@pytest.fixture
def storage(temp_storage):
    temp_storage.write_file(SHEET_URI, sheet.png_bytes())
    return temp_storage


@pytest.fixture
def processor(storage):
    return DrawingProcessor(storage, qwen=FakeQwen(reading=reading()))


def image_fragment(image_ref=SHEET_URI) -> Fragment:
    return Fragment(
        fragment_id="doc-1-frag-001",
        type="image",
        content=Content(image_ref=image_ref),
        position=Position(page=1, bbox=[0.0, 0.0, 1.0, 1.0], order=1),
        confidence=0.9,
        completeness=1.0,
        provenance=Provenance(method="mineru_image", strategy_level=2, source="vector_pdf"),
    )


class TestSheetReadingFields:
    def test_annotations_become_fields(self, processor):
        processor.qwen.reading = reading(
            annotation("size", "⌀44H7", [0.2, 0.3, 0.3, 0.35]),
            annotation("roughness", "Ra 3.2"),
        )
        result = processor.process(SHEET_URI, {})

        assert result["mode"] == MODE_VISION_VLM
        size, roughness = result["parsed_fields"]
        assert size["category"] == "size"
        assert size["value"] == "⌀44H7"
        assert size["tolerance"] == "H7"
        assert size["source_of_tolerance"] == "fit_notation"
        assert size["unit"] == "мм"
        assert size["bbox"] == [0.2, 0.3, 0.3, 0.35]
        assert roughness["category"] == "roughness"
        assert all(f["provenance"] == "detected_from_annotation"
                   for f in result["parsed_fields"])

    def test_model_text_is_normalized_to_gost(self, processor):
        processor.qwen.reading = reading(annotation("size", "Ф44Н7"))
        field = processor.process(SHEET_URI, {})["parsed_fields"][0]
        assert field["value"] == "⌀44H7"

    def test_agreement_with_rules_raises_confidence(self, processor):
        processor.qwen.reading = reading(
            annotation("size", "⌀44"),               # правила согласны
            annotation("note", "⌀44 другое"),        # правила говорят size
            annotation("title_block", "Вилка"),      # правилам сказать нечего
        )
        agreed, disagreed, model_only = processor.process(SHEET_URI, {})["parsed_fields"]

        assert agreed["confidence"] > model_only["confidence"] > disagreed["confidence"]
        assert disagreed["category"] == "note", "спор решается в пользу модели"
        assert model_only["category"] == "title_block"

    def test_position_number_stays_a_position(self, processor):
        """Номер позиции правила считают размером — верить тут надо модели."""
        processor.qwen.reading = reading(annotation("position", "5"))
        field = processor.process(SHEET_URI, {})["parsed_fields"][0]
        assert field["category"] == "position"
        assert field["confidence"] < 0.75

    def test_unknown_category_falls_back_to_rules(self, processor):
        processor.qwen.reading = reading(annotation("выдуманная", "Ra 3.2"))
        field = processor.process(SHEET_URI, {})["parsed_fields"][0]
        assert field["category"] == "roughness"

    def test_graphics_leftovers_are_dropped(self, processor):
        processor.qwen.reading = reading(annotation("note", "|| --"), annotation("size", "20"))
        assert [f["value"] for f in processor.process(SHEET_URI, {})["parsed_fields"]] == ["20"]

    @pytest.mark.parametrize("bad", [
        [0.1, 0.2],                    # не четыре числа
        [700, 500, 900, 700],          # пиксели вместо долей — не угадываем
        [0.5, 0.5, 0.1, 0.1],          # вывернутый прямоугольник
        ["a", "b", "c", "d"],
    ])
    def test_broken_bbox_becomes_the_whole_sheet(self, processor, bad):
        processor.qwen.reading = reading(annotation("size", "20", bad))
        field = processor.process(SHEET_URI, {})["parsed_fields"][0]
        assert field["bbox"] == [0.0, 0.0, 1.0, 1.0]

    def test_missing_bbox_becomes_the_whole_sheet(self, processor):
        processor.qwen.reading = reading({"category": "size", "value": "20"})
        assert processor.process(SHEET_URI, {})["parsed_fields"][0]["bbox"] == [0, 0, 1, 1]

    def test_completeness_is_the_share_of_classified_fields(self, processor):
        processor.qwen.reading = reading(
            annotation("size", "20"), annotation("note", "Технические требования"),
        )
        assert processor.process(SHEET_URI, {})["completeness"] == pytest.approx(0.5)


class TestRegions:
    def test_vision_markup_reaches_the_model(self, processor):
        processor.process(SHEET_URI, {})
        kinds = [what for what, _, _ in processor.qwen.asked]
        assert kinds[0] == "sheet"
        assert "title_block" in kinds          # штамп найден геометрией
        assert "штамп" in processor.qwen.asked[0][2]

    def test_title_block_fields_become_fields(self, processor):
        processor.qwen.title_block_answer = {
            "fields": {"Наименование": "Вилка", "Материал": "Сталь 45"},
            "rows": [["Разраб.", "Иванов"]],
        }
        result = processor.process(SHEET_URI, {})
        by_category = {f["category"]: f for f in result["parsed_fields"]}

        assert "Вилка" in by_category["title_block"]["value"]
        assert by_category["material"]["value"].endswith("Сталь 45")
        assert result["payload"]["title_block"]["Наименование"] == "Вилка"

    def test_table_goes_to_payload_and_not_to_fields(self, processor):
        processor.qwen.table_answer = {
            "title": "Спецификация",
            "headers": ["Поз.", "Наименование", "Кол."],
            "rows": [["1", "Корпус", "1"], ["2", "Гайка M4", "2"]],
        }
        result = processor.process(SHEET_URI, {})
        tables = [t for t in result["payload"]["tables"] if t["kind"] == "table"]

        assert tables and tables[0]["rows"][1] == ["2", "Гайка M4", "2"]
        assert all("Корпус" not in f["value"] for f in result["parsed_fields"])

    def test_silent_region_does_not_break_the_sheet(self, processor):
        processor.qwen.reading = reading(annotation("size", "20"))
        processor.qwen.title_block_answer = None
        result = processor.process(SHEET_URI, {})
        assert result["parsed_fields"][0]["value"] == "20"

    def test_vision_can_be_switched_off(self, processor):
        processor.vision_enabled = False
        processor.qwen.reading = reading(annotation("size", "20"))
        result = processor.process(SHEET_URI, {})

        assert result["mode"] == MODE_SHEET_ONLY
        assert [what for what, _, _ in processor.qwen.asked] == ["sheet"]

    def test_same_inscription_read_twice_gives_one_field(self, processor):
        processor.qwen.reading = reading(annotation("title_block", "Вилка"))
        processor.qwen.title_block_answer = {"fields": {"Наименование": "Вилка"}, "rows": []}
        values = [f["value"] for f in processor.process(SHEET_URI, {})["parsed_fields"]]
        assert values.count("Вилка") == 1


class TestRefusals:
    def test_without_endpoint_nothing_is_parsed(self, storage):
        qwen = FakeQwen(reading=reading(annotation("size", "20")))
        qwen.available = False
        result = DrawingProcessor(storage, qwen=qwen).process(SHEET_URI, {})

        assert result["mode"] == MODE_NONE
        assert result["parsed_fields"] == []
        assert qwen.asked == []

    def test_silent_model_does_not_invent_fields(self, processor):
        processor.qwen.reading = None
        result = processor.process(SHEET_URI, {})
        assert result["mode"] == MODE_NONE
        assert result["parsed_fields"] == []

    def test_missing_file_is_not_a_crash(self, processor):
        assert processor.process("assets/нет-такого.png", {})["mode"] == MODE_NONE

    def test_exception_does_not_break_the_document(self, processor, mocker):
        mocker.patch.object(processor.qwen, "read_sheet", side_effect=RuntimeError("хост упал"))
        fragments = processor.enrich_fragments([image_fragment()], {})
        assert [f.type for f in fragments] == ["image"]


class TestFragment:
    def test_image_fragment_becomes_drawing(self, processor):
        processor.qwen.reading = reading(annotation("size", "⌀44H7"))
        fragment = processor.enrich_fragments([image_fragment()], {})[0]

        assert fragment.type == "drawing"
        assert fragment.fragment_id == "doc-1-frag-001"
        assert fragment.content.image_ref == SHEET_URI
        assert fragment.content.structured_drawing_fields[0].category == "size"
        assert fragment.provenance.method == "vision_plus_vlm"
        assert fragment.provenance.strategy_level == 7
        assert fragment.provenance.source == "drawing"
        assert fragment.content.structured_payload["recognition_mode"] == MODE_VISION_VLM
        assert fragment.content.structured_payload["model"] == "fake-vl"

    def test_nothing_read_leaves_the_image_as_is(self, processor):
        processor.qwen.reading = reading()
        assert processor.enrich_fragments([image_fragment()], {})[0].type == "image"

    def test_table_alone_is_enough_for_a_fragment(self, processor):
        processor.qwen.reading = reading()
        processor.qwen.table_answer = {"title": "", "headers": [], "rows": [["1", "Корпус"]]}
        fragment = processor.enrich_fragments([image_fragment()], {})[0]
        assert fragment.type == "drawing"
        assert fragment.content.structured_payload["tables"]

    def test_non_image_fragments_are_untouched(self, processor):
        text = Fragment(
            fragment_id="f1", type="text", content=Content(text="Текст"),
            position=Position(page=1, bbox=[0, 0, 1, 1]), confidence=1.0, completeness=1.0,
            provenance=Provenance(method="mineru", strategy_level=2, source="vector_pdf"),
        )
        assert processor.enrich_fragments([text], {})[0] is text

    def test_already_parsed_vector_drawing_is_left_alone(self, processor):
        vector = Fragment(
            fragment_id="f1", type="drawing", content=Content(image_ref=SHEET_URI),
            position=Position(page=1, bbox=[0, 0, 1, 1]), confidence=0.9, completeness=0.9,
            provenance=Provenance(method="text_layer_extraction", strategy_level=2,
                                  source="vector_pdf"),
        )
        assert processor.enrich_fragments([vector], {})[0] is vector

    def test_scanned_sheet_becomes_a_fragment_of_its_own(self, processor):
        processor.qwen.reading = reading(annotation("size", "20"))
        metadata = {
            "doc_id": "doc-test", "source_path": SHEET_URI, "file_type": "png",
            "classification": {"genre": "drawing", "source": "raster_only"},
        }
        fragments = processor.enrich_fragments([], metadata)

        assert [f.type for f in fragments] == ["drawing"]
        assert fragments[0].fragment_id == "doc-test-frag-000"
        assert fragments[0].content.structured_drawing_fields

    def test_ordinary_scan_is_not_treated_as_a_drawing(self, processor):
        metadata = {
            "doc_id": "doc-test", "source_path": SHEET_URI, "file_type": "png",
            "classification": {"genre": "text", "source": "raster_only"},
        }
        assert processor.enrich_fragments([], metadata) == []


class TestLayoutInPayload:
    def test_layout_is_reported(self, processor):
        processor.qwen.reading = reading(annotation("size", "20"))
        payload = processor.process(SHEET_URI, {})["payload"]
        layout = payload["layout"]

        assert layout["frame"] is not None
        assert any(r["kind"] == drawing_vision.KIND_TITLE_BLOCK for r in layout["regions"])
        assert payload["sheet_type"] == "detail"


class TestTimeBudget:
    def test_regions_are_skipped_when_the_budget_is_spent(self, processor, mocker):
        mocker.patch.object(
            __import__("core.drawing_processor", fromlist=["settings"]).settings,
            "DRAWING_TIME_BUDGET", 0.001,
        )
        processor.qwen.reading = reading(annotation("size", "20"))
        result = processor.process(SHEET_URI, {})

        assert [what for what, _, _ in processor.qwen.asked] == ["sheet"]
        assert result["payload"]["regions_skipped"] >= 1
        assert result["parsed_fields"], "прочитанное до исчерпания бюджета остаётся"

    def test_title_block_is_read_first(self, processor):
        processor.qwen.reading = reading(annotation("size", "20"))
        processor.process(SHEET_URI, {})
        kinds = [what for what, _, _ in processor.qwen.asked]
        assert kinds.index("title_block") < kinds.index("table")
