"""
Третье свидетельство (D4): дочитывание спорных чисел вырезками, сведение
нескольких чтений листа и плитки крупного листа сразу.

Модель поддельная: она «читает» картинку вырезок по заранее заданным
ответам, а номера строк берёт из числа присланных вырезок, — так видно,
что ответ по номеру возвращается своему полю.
"""

import pytest

from core.config import settings
from core.providers import disputed_reread as dr
from core.providers.qwen_client import QwenVisionClient, SheetReading

pytest.importorskip("PIL")
from PIL import Image  # noqa: E402


def field(category, value, bbox, **extra):
    return {"category": category, "value": value, "bbox": bbox, "confidence": 0.8,
            "tolerance": None, "nature": "executive", "unit": "", **extra}


class FakeReader:
    """Отвечает по номеру вырезки из `answers` (список по порядку вырезок)."""

    def __init__(self, answers, fail=False):
        self.answers = list(answers)
        self.fail = fail
        self.calls = []

    def read_values(self, image, count):
        self.calls.append((image.size, count))
        if self.fail:
            return None
        batch, self.answers = self.answers[:count], self.answers[count:]
        return {n: text for n, text in enumerate(batch, start=1) if text}


@pytest.fixture
def sheet():
    return Image.new("RGB", (1654, 2339), "white")


@pytest.fixture(autouse=True)
def cap(monkeypatch):
    monkeypatch.setattr(settings, "DRAWING_REREAD_MAX_CROPS", 48)


def test_decide_accepts_only_errors_of_one_sign():
    assert dr.decide("⌀4,5−0,2", "⌀4,5+0,2") == "⌀4,5+0,2"
    assert dr.decide("15*", "1,5*") == "1,5*"
    assert dr.decide("M18x15-8g", "M18x1,5-8g") == "M18x1,5-8g"
    assert dr.decide("ТАДУ 421451.003 34", "ТАДУ 421451.003 Э4") == "ТАДУ 421451.003 Э4"
    assert dr.decide("n.2", "п.2") == "п.2"
    # Другое число или не та надпись — поле остаётся.
    assert dr.decide("⌀280±0,3", "⌀28") is None
    assert dr.decide("⌀18", "R5") is None
    # То же прочтение — менять нечего.
    assert dr.decide("⌀4,5+0,2", "⌀4,5 +0,2") is None


def test_targets_own_rule_picks_conflicts_and_risky_values():
    fields = [
        field("size", "⌀280±0,3", [0.65, 0.34, 0.68, 0.40]),
        field("size", "⌀280+0,3", [0.65, 0.35, 0.68, 0.41]),   # то же число рядом
        field("size", "⌀315", [0.42, 0.35, 0.45, 0.41]),        # не спорное
        field("thread", "M20x15*", [0.10, 0.10, 0.20, 0.12]),   # резьба
        field("note", "1. * Размер для справок.", [0.1, 0.8, 0.5, 0.82]),
        field("title_block", "Обозначение: ТАДУ 421451.003 34", [0.6, 0.9, 0.9, 0.93]),
        field("size", "⌀4,5-0,2", [0.0, 0.0, 1.0, 1.0]),         # места нет — вырезать нечего
    ]
    chosen = dr.targets(fields)
    assert [members for members, _ in chosen] == [[0, 1], [3], [5]]
    assert chosen[0][1] == [0.65, 0.34, 0.68, 0.41]


def test_targets_prefer_d3_agreement_when_present():
    fields = [
        field("size", "⌀315", [0.42, 0.35, 0.45, 0.41], agreement="single"),
        field("size", "⌀4,5+0,2", [0.5, 0.5, 0.52, 0.55], agreement="agree"),
        field("size", "⌀280±0,3", [0.65, 0.34, 0.68, 0.40], agreement="conflict"),
    ]
    assert [members for members, _ in dr.targets(fields)] == [[2], [0]]


def test_targets_capped(monkeypatch):
    monkeypatch.setattr(settings, "DRAWING_REREAD_MAX_CROPS", 2)
    fields = [field("size", f"{n},5", [0.05 * n, 0.1, 0.05 * n + 0.03, 0.12]) for n in range(1, 6)]
    assert len(dr.targets(fields)) == 2


def test_grid_of_crops_round_trips_numbering(sheet):
    # 14 спорных полей: две картинки (12 + 2), ответ каждой строки — своему полю.
    fields = [field("size", f"{n}-0,{n}", [0.05 * (n % 10), 0.04 * n, 0.05 * (n % 10) + 0.04, 0.04 * n + 0.02])
              for n in range(1, 15)]
    answers = [f"{n}+0,{n}" for n in range(1, 15)]
    reader = FakeReader(answers)
    result, stats = dr.apply(reader, sheet, fields)
    assert [count for _, count in reader.calls] == [12, 2]
    assert reader.calls[0][0][1] == 12 * dr._ROW
    assert [f["value"] for f in result] == answers
    assert stats["corrected"] == 14 and stats["calls"] == 2


def test_conflict_group_keeps_the_reading_the_crop_confirms(sheet):
    fields = [
        field("size", "⌀280±0,3", [0.65, 0.34, 0.68, 0.40]),
        field("size", "⌀280+0,3", [0.65, 0.35, 0.68, 0.41]),
    ]
    result, stats = dr.apply(FakeReader(["⌀280±0,3"]), sheet, fields)
    assert [f["value"] for f in result] == ["⌀280±0,3"]
    assert stats["agreed"] == 1 and stats["dropped"] == 1


def test_failed_reread_leaves_fields_as_they_were(sheet):
    fields = [field("size", "⌀4,5-0,2", [0.5, 0.5, 0.52, 0.55])]
    result, stats = dr.apply(FakeReader([], fail=True), sheet, fields)
    assert result == fields and stats["failed_calls"] == 1


def test_unrelated_answer_does_not_rewrite(sheet):
    fields = [field("size", "⌀4,5-0,2", [0.5, 0.5, 0.52, 0.55])]
    result, _ = dr.apply(FakeReader(["R12"]), sheet, fields)
    assert result == fields


def test_title_block_label_kept_and_d3_readings_extended(sheet):
    fields = [field("title_block", "Обозначение: ТАДУ 421451.003 34", [0.6, 0.9, 0.9, 0.93],
                    readings=[{"source": "sheet", "value": "ТАДУ 421451.003 34"}])]
    result, _ = dr.apply(FakeReader(["ТАДУ 421451.003 Э4"]), sheet, fields)
    assert result[0]["value"] == "Обозначение: ТАДУ 421451.003 Э4"
    assert result[0]["readings"][-1] == {"source": "crop_reread", "value": "ТАДУ 421451.003 Э4"}


def test_no_call_when_out_of_sheet_budget(sheet, monkeypatch):
    monkeypatch.setattr(dr.gpu_gate, "budget_now", lambda: 1000.0)
    reader = FakeReader(["⌀4,5+0,2"])
    fields = [field("size", "⌀4,5-0,2", [0.5, 0.5, 0.52, 0.55])]
    result, stats = dr.apply(reader, sheet, fields, until=1010.0)
    assert not reader.calls and result == fields and stats["skipped_time"] == 1


def test_parse_values_forms():
    assert dr.parse_values({"values": [{"n": 2, "text": "⌀8"}, {"n": 9, "text": "x"}]}, 3) == {2: "⌀8"}
    assert dr.parse_values({"values": ["⌀8", "", "R5"]}, 3) == {1: "⌀8", 3: "R5"}
    assert dr.parse_values({"1": "M20x1,5"}, 1) == {1: "M20x1,5"}
    assert dr.parse_values(None, 1) == {}


def test_tall_crop_turned_to_a_line(sheet):
    piece = dr.crop(sheet, [0.65, 0.34, 0.66, 0.40])
    assert piece.width > piece.height


def test_consensus_of_three_runs():
    runs = [
        [{"value": "⌀280±0,3", "bbox": [0.65, 0.34, 0.68, 0.40]}, {"value": "R5", "bbox": [0.1, 0.1, 0.12, 0.12]}],
        [{"value": "⌀280±0,3", "bbox": [0.66, 0.34, 0.69, 0.40]}, {"value": "⌀280+0,3", "bbox": [0.65, 0.34, 0.68, 0.40]}],
        [{"value": "R5", "bbox": [0.8, 0.8, 0.82, 0.82]}],                      # то же значение, другое место
    ]
    merged = dr.consensus(runs)
    assert [(a["value"], a["agreement"]) for a in merged] == [
        ("⌀280±0,3", "agree"), ("R5", "single"), ("⌀280+0,3", "single"), ("R5", "single"),
    ]


def test_native_tiles_only_below_the_scale(monkeypatch):
    monkeypatch.setattr(settings, "QWEN_SHEET_MAX_SIDE", 2000)
    monkeypatch.setattr(settings, "DRAWING_SHEET_NATIVE_TILES", True)
    a4, a1 = Image.new("RGB", (1654, 2339)), Image.new("RGB", (6622, 4677))
    monkeypatch.setattr(settings, "DRAWING_NATIVE_MIN_SCALE", None)
    assert not dr.needs_native_tiles(a1)                      # порога нет — флаг молчит
    monkeypatch.setattr(settings, "DRAWING_NATIVE_MIN_SCALE", 0.5)
    assert dr.needs_native_tiles(a1) and not dr.needs_native_tiles(a4)
    monkeypatch.setattr(settings, "DRAWING_SHEET_NATIVE_TILES", False)
    assert not dr.needs_native_tiles(a1)


def test_read_sheet_goes_to_tiles_without_whole_call(monkeypatch):
    monkeypatch.setattr(settings, "QWEN_SHEET_MAX_SIDE", 2000)
    monkeypatch.setattr(settings, "DRAWING_SHEET_NATIVE_TILES", True)
    monkeypatch.setattr(settings, "DRAWING_NATIVE_MIN_SCALE", 0.5)
    monkeypatch.setattr(settings, "DRAWING_SMALL_PRINT_CARE", False)
    client = QwenVisionClient(endpoint="http://vlm")
    pieces = []

    def piece(image, prompt):
        pieces.append(image.size)
        return SheetReading(sheet_type="detail", annotations=[{"value": f"⌀{len(pieces)}", "bbox": [0.1, 0.1, 0.2, 0.2]}])

    monkeypatch.setattr(client, "_read_sheet_piece", piece)
    reading = client.read_sheet(Image.new("RGB", (6622, 4677)))
    assert len(pieces) == settings.DRAWING_SHEET_TILE_GRID ** 2
    assert all(size[0] < 6622 for size in pieces)
    assert reading.sheet_type == "detail" and len(reading.annotations) == len(pieces)


def test_reread_flag_off_makes_no_call(monkeypatch):
    monkeypatch.setattr(settings, "DRAWING_DISPUTED_REREAD", False)
    assert settings.DRAWING_CONSENSUS_READS == 1
    assert not dr.needs_native_tiles(Image.new("RGB", (6622, 4677)))


def test_decide_does_not_accept_a_lost_sign():
    assert dr.decide("40−0,25", "400,25") is None
    assert dr.decide("30±10", "3010") is None


def test_sign_variants_from_different_reads_become_one_field(sheet):
    runs = [[{"value": "⌀4,5+0,2", "bbox": [0.5, 0.5, 0.52, 0.55]}],
            [{"value": "⌀4,5-0,2", "bbox": [0.5, 0.5, 0.52, 0.55]}]]
    fields = [field("size", a["value"], a["bbox"], agreement=a["agreement"]) for a in dr.consensus(runs)]
    assert [members for members, _ in dr.targets(fields)] == [[0, 1]]
    result, _ = dr.apply(FakeReader(["⌀4,5+0,2"]), sheet, fields)
    assert [f["value"] for f in result] == ["⌀4,5+0,2"]


def test_fields_without_agreement_use_own_rule():
    fields = [field("size", "⌀315", [0.42, 0.35, 0.45, 0.41], agreement="agree"),
              field("size", "⌀4,5-0,2", [0.5, 0.5, 0.52, 0.55])]
    assert [members for members, _ in dr.targets(fields)] == [[1]]
