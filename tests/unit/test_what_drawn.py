"""
«Что изображено» на листе (DRAWING_WHAT_DRAWN).

Один вызов модели на лист детали или сборки целиком. Ответ проверяется
правилами: смысл, не сходящийся со знаком величины, два смысла у одной
величины и неизвестный номер величины отбрасываются. Модель поддельная,
лист и величины выдуманные.
"""

import copy
from io import BytesIO

import pytest

from core.config import settings
from core.providers import what_drawn
from core.providers.qwen_client import SheetReading

pytest.importorskip("PIL")
from PIL import Image  # noqa: E402

FIELDS = [
    {"category": "size", "value": "⌀40", "bbox": [0.30, 0.40, 0.34, 0.42]},
    {"category": "size", "value": "120", "bbox": [0.50, 0.70, 0.54, 0.72]},
    {"category": "radius", "value": "R5", "bbox": [0.60, 0.30, 0.62, 0.32]},
    {"category": "title_block", "value": "Наименование: Втулка опытная", "bbox": [0.7, 0.9, 0.9, 0.95]},
    {"category": "note", "value": "1. Острые кромки притупить.", "bbox": [0.1, 0.8, 0.4, 0.82]},
]

ANSWER = {
    "object": "Втулка опытная: короткая труба с отверстием.",
    "how_parts_fit": "Длина 120 при диаметре 40.",
    "views": [{"name": "главный вид", "shows": "втулка сбоку"}, {"name": "", "shows": "пусто"}],
    "features": [{"id": "f1", "kind": "hole", "where": "сквозное отверстие", "count": 1}],
    "dimensions": [
        {"value_id": "v1", "measures": "наружный диаметр втулки", "feature_id": "f1", "confidence": "high"},
        {"value_id": "v2", "measures": "длина втулки на главном виде", "feature_id": None, "confidence": "high"},
        {"value_id": "v3", "measures": "радиус скругления края", "feature_id": "f9", "confidence": "high"},
        {"value_id": "v9", "measures": "выдуманная величина", "confidence": "high"},
    ],
}


def parsed(fields=FIELDS, answer=ANSWER):
    _, dims = what_drawn.build_prompt(fields)
    return what_drawn.parse(copy.deepcopy(answer), dims)


def test_prompt_numbers_dimensions_and_passes_context():
    prompt, dims = what_drawn.build_prompt(FIELDS)
    assert [d["value"] for d in dims] == ["⌀40", "120", "R5"]
    assert "v1: ⌀40  рамка [0.3, 0.4, 0.34, 0.42]" in prompt
    assert "Основная надпись: Втулка опытная" in prompt
    assert "Примечания листа: 1. Острые кромки притупить." in prompt
    assert "ГОСТ 2.307" in prompt and '"how_parts_fit"' in prompt


def test_prompt_without_dimensions_is_empty():
    assert what_drawn.build_prompt([FIELDS[3]]) == ("", [])


def test_payload_shape():
    result = parsed()
    assert result == {
        "version": 1,
        "object": "Втулка опытная: короткая труба с отверстием.",
        "views": [{"name": "главный вид", "shows": "втулка сбоку"}],
        "dimensions": [
            {"value": "⌀40", "measures": "наружный диаметр втулки", "feature": "сквозное отверстие"},
            {"value": "120", "measures": "длина втулки на главном виде", "feature": None},
            {"value": "R5", "measures": "радиус скругления края", "feature": None},
        ],
        "parts_fit": "Длина 120 при диаметре 40.",
    }


def test_parts_fit_dropped_on_wrong_sum_or_invented_number():
    wrong = dict(ANSWER, how_parts_fit="Общая длина 120 = 40 + 5 + 60.")
    assert parsed(answer=wrong)["parts_fit"] is None
    good = [*FIELDS, {"category": "size", "value": "80", "bbox": [0.5, 0.5, 0.52, 0.52]}]
    right = dict(ANSWER, how_parts_fit="Общая длина 120 = 40 + 80.")
    assert parsed(good, right)["parts_fit"] == "Общая длина 120 = 40 + 80."
    invented = dict(ANSWER, how_parts_fit="Длина 130 при диаметре 40.")
    assert parsed(answer=invented)["parts_fit"] is None


def test_diameter_called_length_is_dropped():
    answer = copy.deepcopy(ANSWER)
    answer["dimensions"][0]["measures"] = "длина втулки"
    answer["dimensions"][1]["measures"] = "диаметр отверстия"     # линейный — не диаметр
    values = [row["value"] for row in parsed(answer=answer)["dimensions"]]
    assert values == ["R5"]


def test_same_value_two_meanings_gets_none():
    fields = [*FIELDS, {"category": "size", "value": "120", "bbox": [0.1, 0.2, 0.12, 0.22]}]
    answer = copy.deepcopy(ANSWER)
    answer["dimensions"].append({"value_id": "v4", "measures": "ширина фланца", "confidence": "high"})
    values = [row["value"] for row in parsed(fields, answer)["dimensions"]]
    assert "120" not in values and values == ["⌀40", "R5"]


def test_unknown_value_and_low_confidence_dropped():
    answer = copy.deepcopy(ANSWER)
    answer["dimensions"][2]["confidence"] = "low"
    values = [row["value"] for row in parsed(answer=answer)["dimensions"]]
    assert values == ["⌀40", "120"]     # v9 неизвестна, R5 неуверенно


def test_invented_number_in_meaning_dropped():
    answer = copy.deepcopy(ANSWER)
    answer["dimensions"][1]["measures"] = "длина 125 мм"
    assert [r["value"] for r in parsed(answer=answer)["dimensions"]] == ["⌀40", "R5"]


def test_link_label_preferred_over_model():
    fields = copy.deepcopy(FIELDS)
    fields[1]["link"] = {"target_label": "длина втулки", "target_bbox": [0, 0, 1, 1], "method": "vector"}
    prompt, dims = what_drawn.build_prompt(fields)
    assert "выноска ведёт к: длина втулки" in prompt
    answer = copy.deepcopy(ANSWER)
    answer["dimensions"][1]["measures"] = "высота"
    rows = what_drawn.parse(answer, dims)["dimensions"]
    assert {"value": "120", "measures": "длина втулки", "feature": None} in rows


@pytest.mark.parametrize("answer", [None, "текст", {}, {"object": "", "views": [], "dimensions": []}])
def test_empty_answer_gives_none(answer):
    _, dims = what_drawn.build_prompt(FIELDS)
    assert what_drawn.parse(answer, dims) is None


@pytest.mark.parametrize("value, meaning, bad", [
    ("⌀12", "диаметр отверстия", False),
    ("⌀12", "окружность расположения отверстий", False),
    ("⌀12", "длина", True),
    ("R3", "радиус гиба", False),
    ("R3", "отверстие", True),
    ("S2", "толщина листа", False),
    ("S2", "ширина", True),
    ("45", "высота", False),
    ("45", "радиус", True),
    ("Ra 3,2", "гладкость поверхности", False),
    ("30°", "угол фаски", False),
    ("45", "не видно", True),
])
def test_sign_rules(value, meaning, bad):
    assert what_drawn.contradicts(value, meaning) is bad


# ------------------------------------------------------------ обработчик

class WhatQwen:
    available = True
    model = "fake-vl"

    def __init__(self, answer, sheet_type="detail", fail=False):
        self.answer = answer
        self.sheet_type = sheet_type
        self.fail = fail
        self.calls = []

    def read_sheet(self, image, hint="", **_):
        fields = [{k: v for k, v in f.items()} for f in FIELDS]
        return SheetReading(sheet_type=self.sheet_type, annotations=fields)

    def read_title_block(self, image):
        return None

    def read_table(self, image):
        return None

    def read_what_drawn(self, image, prompt):
        self.calls.append(prompt)
        if self.fail:
            raise RuntimeError("модель недоступна")
        return copy.deepcopy(self.answer)


def run(temp_storage, monkeypatch, qwen, flag=True, budget_left=True):
    from core.drawing_processor import DrawingProcessor
    from core.models.contract import Content, Fragment, Position, Provenance

    monkeypatch.setattr(settings, "DRAWING_WHAT_DRAWN", flag)
    monkeypatch.setattr(settings, "DRAWING_VISION_ENABLED", False)
    if not budget_left:
        monkeypatch.setattr(settings, "DRAWING_TIME_BUDGET", 10)
        monkeypatch.setattr(settings, "VLM_TIMEOUT", 30)
    data = BytesIO()
    Image.new("RGB", (1654, 1169), "white").save(data, format="PNG")
    temp_storage.write_file("assets/doc/sheet.png", data.getvalue())
    fragment = Fragment(
        fragment_id="doc-frag-1", type="image", content=Content(image_ref="assets/doc/sheet.png"),
        position=Position(page=1, bbox=[0.0, 0.0, 1.0, 1.0], order=1), confidence=0.9, completeness=1.0,
        provenance=Provenance(method="mineru_image", strategy_level=2, source="vector_pdf"),
    )
    out = DrawingProcessor(temp_storage, qwen=qwen).enrich_fragments([fragment], {})
    return next(f for f in out if f.content.image_ref == "assets/doc/sheet.png")


def test_sheet_gets_what_drawn(temp_storage, monkeypatch):
    qwen = WhatQwen(ANSWER)
    sheet = run(temp_storage, monkeypatch, qwen)
    assert len(qwen.calls) == 1 and "v1: ⌀40" in qwen.calls[0]
    result = sheet.content.structured_payload["what_drawn"]
    assert result["version"] == 1 and result["object"].startswith("Втулка")
    values = {f.value for f in sheet.content.structured_drawing_fields}
    assert {row["value"] for row in result["dimensions"]} <= values


def test_scheme_sheet_makes_no_call(temp_storage, monkeypatch):
    qwen = WhatQwen(ANSWER, sheet_type="scheme")
    sheet = run(temp_storage, monkeypatch, qwen)
    assert qwen.calls == [] and "what_drawn" not in sheet.content.structured_payload


def test_budget_exhausted_makes_no_call(temp_storage, monkeypatch):
    qwen = WhatQwen(ANSWER)
    sheet = run(temp_storage, monkeypatch, qwen, budget_left=False)
    assert qwen.calls == [] and "what_drawn" not in sheet.content.structured_payload


def test_vlm_exception_gives_no_key(temp_storage, monkeypatch):
    qwen = WhatQwen(ANSWER, fail=True)
    sheet = run(temp_storage, monkeypatch, qwen)
    assert len(qwen.calls) == 1 and "what_drawn" not in sheet.content.structured_payload
    assert {f.value for f in sheet.content.structured_drawing_fields} >= {"⌀40", "120"}


def test_flag_off_output_identical(temp_storage, monkeypatch):
    assert settings.DRAWING_WHAT_DRAWN is False
    off = run(temp_storage, monkeypatch, WhatQwen(ANSWER), flag=False)
    empty = run(temp_storage, monkeypatch, WhatQwen(None), flag=True)
    assert "what_drawn" not in off.content.structured_payload
    # Пустой ответ при включённом флаге даёт тот же фрагмент, что и выключенный флаг.
    assert off.model_dump() == empty.model_dump()
