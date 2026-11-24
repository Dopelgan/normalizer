"""
OCR-подсказки листа (DRAWING_OCR_HINTS): строки PP-OCRv5 свидетельствуют, но в
поля сами не попадают. Строки и значения — с листа фланца ТАДУ 302389.001.01.02.001
и пробки ЭК150; модель поддельная.
"""

import json

import pytest

from core.providers import ocr_hints

pytest.importorskip("PIL")
from PIL import Image  # noqa: E402


def field(value, bbox, category="size", **extra):
    return {"category": category, "value": value, "bbox": bbox, "confidence": 0.9, **extra}


def line(text, box, region="field", score=0.98):
    return {"text": text, "box": box, "region": region, "score": score}


class FakeQwen:
    def __init__(self, answers):
        self.answers = answers
        self.calls = []

    def read_values(self, image, count):
        self.calls.append(count)
        return self.answers


def test_confirmed_value_gets_ocr_reading():
    fields = [field("⌀315", [0.41, 0.34, 0.45, 0.39])]
    stats = ocr_hints.new_stats()
    ocr_hints.match(fields, [line("Ø315", [0.415, 0.348, 0.446, 0.385])], stats)
    assert stats["confirmed"] == 1
    assert fields[0]["readings"][-1] == {"source": "ocr", "value": "Ø315"}
    assert not fields[0].get("needs_review")


def test_labelled_stamp_value_is_confirmed_by_its_cell():
    fields = [field("Инв. № подл.: 1012-05 (рукописно)", [0.06, 0.883, 0.114, 0.964], category="title_block")]
    stats = ocr_hints.new_stats()
    ocr_hints.match(fields, [line("1012-05", [0.085, 0.899, 0.109, 0.946], region="margin")], stats)
    assert stats["confirmed"] == 1


@pytest.mark.parametrize("model, ocr", [("50,5*", "S0,5*"), ("2M3", "2№3")])
def test_one_char_conflict_goes_to_review(model, ocr):
    category = "thread" if "M" in model else "size"
    fields = [field(model, [0.80, 0.08, 0.84, 0.11], category=category)]
    stats = ocr_hints.new_stats()
    ocr_hints.match(fields, [line(ocr, [0.80, 0.08, 0.84, 0.11])], stats)
    assert fields[0]["needs_review"] and stats["one_char"] == 1
    if category == "thread":
        # «№» у OCR: метка сварного шва, а не резьба.
        assert fields[0]["category"] == "callout" and stats["not_thread"] == 1


def test_region_fields_are_not_matched():
    fields = [field("Масса: 3,74", [0.06, 0.5, 0.96, 0.96], category="title_block")]
    stats = ocr_hints.new_stats()
    ocr_hints.match(fields, [line("3,74", [0.82, 0.86, 0.86, 0.88], region="stamp")], stats)
    assert stats["confirmed"] == 0 and "readings" not in fields[0]


def test_ocr_only_value_enters_only_when_model_reads_the_same():
    fields = [field("⌀280±0,3", [0.67, 0.4, 0.69, 0.5])]
    lines = [line("18*", [0.5, 0.82, 0.52, 0.84]), line("⌀245", [0.6, 0.4, 0.62, 0.5]),
             line("Острые кромки притупить.", [0.1, 0.7, 0.4, 0.72]),
             line("⌀280±0,3", [0.67, 0.4, 0.69, 0.5])]
    found = ocr_hints.candidates(fields, lines)
    # Фраза — не величина; строка поверх поля модели — не кандидат.
    assert [c["text"] for c in found] == ["18*", "⌀245"]
    qwen = FakeQwen({1: "18*", 2: "⌀248"})
    stats = ocr_hints.new_stats()
    added = ocr_hints.reread(qwen, Image.new("RGB", (1000, 1400), "white"), found, stats)
    assert qwen.calls == [2]
    assert [a["value"] for a in added] == ["18*"]
    assert added[0]["category"] == "size" and added[0]["readings"][0]["source"] == "ocr"
    assert stats["added"] == 1 and stats["reread"] == 2


def test_low_score_stamp_and_bare_numbers_are_not_candidates():
    lines = [line("315", [0.4, 0.3, 0.42, 0.32], score=0.5), line("3,74", [0.82, 0.86, 0.86, 0.88], region="stamp"),
             line("16", [0.5, 0.5, 0.52, 0.52]), line("1221", [0.6, 0.6, 0.63, 0.62])]
    assert ocr_hints.candidates([], lines) == []


def test_empty_inventory_and_designation_filled_for_review():
    fields = [field("Масштаб: 1:2", [0.88, 0.86, 0.95, 0.89], category="title_block")]
    lines = [line("1012-05", [0.085, 0.899, 0.109, 0.946], region="margin"),
             line("ТАДУ 302389.001.01.02.001", [0.48, 0.79, 0.89, 0.82], region="stamp"),
             line("10.02.2022", [0.08, 0.77, 0.11, 0.83], region="margin")]
    stats = ocr_hints.new_stats()
    ocr_hints.fill_empty(fields, lines, stats)
    values = [f["value"] for f in fields]
    assert "Инв. № подл.: 1012-05 (из OCR, требует сверки)" in values
    assert "Обозначение: ТАДУ 302389.001.01.02.001 (из OCR, требует сверки)" in values
    assert all(f.get("needs_review") for f in fields[1:]) and stats["filled"] == 2


def test_filled_graph_is_not_overwritten():
    fields = [field("Инв. № подл.: 1012-05", [0.06, 0.88, 0.11, 0.96], category="title_block")]
    stats = ocr_hints.new_stats()
    ocr_hints.fill_empty(fields, [line("1012-06", [0.085, 0.899, 0.109, 0.946], region="margin")], stats)
    assert len(fields) == 1 and stats["filled"] == 0


def test_rotated_margin_box_maps_back_to_the_strip():
    # Кусок 100×400 px, повёрнутый по часовой, стал 400×100; строка в левой
    # части повёрнутого — внизу исходной полосы.
    box = ocr_hints.to_page([0, 0, 100, 50], [0.0, 0.5, 0.1, 1.0], (400, 100), rotated=True)
    assert box[1] >= 0.87 and box[3] == pytest.approx(1.0) and box[0] == pytest.approx(0.0)


def test_lines_from_dir_by_file_and_page(tmp_path):
    (tmp_path / "S01.json").write_text(json.dumps({
        "file": "ТАДУ 302389.001.01.02.001.pdf", "page": 1, "lines": [line("315", [0.4, 0.3, 0.42, 0.32])]},
        ensure_ascii=False), encoding="utf-8")
    meta = {"s3_fileid": "s3://kb/ed0a/ТАДУ 302389.001.01.02.001.pdf"}
    assert ocr_hints.from_dir(str(tmp_path), meta, 1)[0]["text"] == "315"
    assert ocr_hints.from_dir(str(tmp_path), meta, 2) is None


def test_gost_pieces_cover_stamp_margin_and_tiles():
    pieces = ocr_hints.gost_pieces((3508, 2480), 300)   # A4 альбомный 300 dpi
    stamp = next(b for r, b, _ in pieces if r == "stamp")
    margin = next((b, rot) for r, b, rot in pieces if r == "margin")
    assert stamp[0] == pytest.approx(1 - 200 / 297, abs=0.01) and stamp[1] == pytest.approx(1 - 75 / 210, abs=0.01)
    assert margin[1] is True and margin[0][2] == pytest.approx(32 / 297, abs=0.01)
    assert sum(1 for r, _, _ in pieces if r == "field") == 4


def test_flag_off_makes_no_ocr_call(temp_storage, monkeypatch):
    from tests.unit.test_margin_columns import run

    called = []
    monkeypatch.setattr(ocr_hints, "from_dir", lambda *a: called.append(a))
    from core.config import settings

    monkeypatch.setattr(settings, "DRAWING_OCR_HINTS", False)
    monkeypatch.setattr(settings, "DRAWING_OCR_HINTS_DIR", "/nonexistent")
    run(temp_storage, monkeypatch, {"columns": {}, "top_designation": ""}, flag=False)
    assert called == []


def test_flag_on_reads_hints_for_whole_sheet(temp_storage, monkeypatch):
    from io import BytesIO

    from core.config import settings
    from core.drawing_processor import DrawingProcessor
    from tests.unit.test_margin_columns import DIM, MarginQwen

    monkeypatch.setattr(settings, "DRAWING_OCR_HINTS", True)
    monkeypatch.setattr(settings, "DRAWING_OCR_HINTS_DIR", "/hints")
    monkeypatch.setattr(settings, "DRAWING_VISION_ENABLED", False)
    monkeypatch.setattr(ocr_hints, "from_dir", lambda d, m, p: [line("1012-05", [0.03, 0.9, 0.06, 0.95], "margin")])
    data = BytesIO()
    Image.new("RGB", (840, 1188), "white").save(data, format="PNG")
    temp_storage.write_file("assets/doc/sheet.png", data.getvalue())
    result = DrawingProcessor(temp_storage, qwen=MarginQwen({}, [DIM])).process("assets/doc/sheet.png", {})
    stats = result["payload"]["ocr_hints"]
    assert stats["status"] == "read" and stats["filled"] == 1
    assert any(f["value"] == "Инв. № подл.: 1012-05 (из OCR, требует сверки)" for f in result["parsed_fields"])


def test_endpoint_lines_come_back_in_sheet_fractions(requests_mock, monkeypatch):
    from core.config import settings

    monkeypatch.setattr(settings, "DRAWING_OCR_HINTS_ENDPOINT", "http://ocr-hints:8090/")
    requests_mock.post("http://ocr-hints:8090/", json={"lines": [{"text": "315", "score": 0.99, "box": [0, 0, 10, 10]}]})
    lines = ocr_hints.from_endpoint(Image.new("RGB", (2480, 3508), "white"), 300)
    regions = {line["region"] for line in lines}
    assert regions == {"stamp", "margin", "field"}
    assert all(0.0 <= v <= 1.0 for line in lines for v in line["box"])


def test_endpoint_failure_gives_no_hints(requests_mock, monkeypatch):
    from core.config import settings

    monkeypatch.setattr(settings, "DRAWING_OCR_HINTS_ENDPOINT", "http://ocr-hints:8090/")
    requests_mock.post("http://ocr-hints:8090/", status_code=500)
    assert ocr_hints.from_endpoint(Image.new("RGB", (800, 600), "white"), 200) is None
