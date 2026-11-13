"""
Потери, найденные трассировкой десяти реальных чертежей (13.11).

1. Скан СБ в PDF на 4 листа: MinerU нарезал картинки на листах 1, 2 и 4, а
   повёрнутый лист 3 принял за таблицу. Один фрагмент drawing на любом листе
   отменял чтение листов целиком, и лист 3 в Qwen-VL не попадал вовсе
   (DRAWING_SHEET_MISSING_PAGES).
2. Лист JPG 4960x3507 с бледной рамкой: при уменьшении до рабочего размера
   линия рамки растворялась в фоне, детектор звал лист страницей текста, и
   документ уходил в плоское распознавание без Qwen-VL (RASTER_FRAME_THIN_LINES).
3. Материал «12Х18Н10Т» получал допуск «H10» (DRAWING_MATERIAL_NO_TOLERANCE).

Синтетика повторяет форму реальных случаев; сами листы в тесты не кладутся.
"""

import io

import pytest

from core.config import settings
from core.drawing_processor import DrawingProcessor, _field
from core.models.contract import Content, DrawingField, Fragment, Position, Provenance
from core.providers import raster_drawing

numpy = pytest.importorskip("numpy")
pytest.importorskip("PIL")


# --------------------------------------------------------------------------
# 1. Лист без фрагмента drawing в скане PDF
# --------------------------------------------------------------------------

def drawn(page):
    return Fragment(
        fragment_id=f"doc-1-frag-{page:03d}", type="drawing",
        content=Content(image_ref=f"assets/doc-1/crop-{page}.png", structured_drawing_fields=[
            DrawingField(category="size", value="20", confidence=0.6)]),
        position=Position(page=page, bbox=[0.2, 0.2, 0.6, 0.6], order=1),
        confidence=0.6, completeness=None,
        provenance=Provenance(method="vision_plus_vlm", strategy_level=7, source="drawing"),
    )


def junk_table(page):
    return Fragment(
        fragment_id=f"doc-1-frag-t{page}", type="table", content=Content(text="0 0 00 Φ"),
        position=Position(page=page, bbox=[0.1, 0.1, 0.9, 0.9], order=2),
        confidence=0.6, completeness=1.0,
        provenance=Provenance(method="mineru_table", strategy_level=5, source="raster_only"),
    )


def scan_metadata(pages=(1, 2, 3, 4)):
    return {
        "doc_id": "doc-1", "source_path": "src/scan.pdf", "file_type": "pdf",
        "classification": {"genre": "drawing", "source": "raster_only",
                           "signals": {"raster_drawing_pages": list(pages)}},
    }


class FakeQwen:
    available = True
    model = "fake-vl"


@pytest.fixture
def processor(temp_storage, monkeypatch):
    proc = DrawingProcessor(temp_storage, qwen=FakeQwen())
    asked = []

    def process(image_ref, metadata, page=1, transcribe=False):
        asked.append(page)
        return {"detected_count": 1, "avg_confidence": 0.75, "mode": "sheet_only",
                "parsed_fields": [_field("size", "120*", [0.1, 0.1, 0.2, 0.2], 0.75)],
                "payload": {"tables": []}, "warnings": ["DRAWING_COMPLETENESS_UNKNOWN"]}

    monkeypatch.setattr(proc, "process", process)
    proc.asked = asked
    return proc


class TestSheetWithoutDrawingFragment:
    def fragments(self):
        return [drawn(1), drawn(2), junk_table(3), drawn(4)]

    def test_flag_off_one_crop_cancels_all_sheets(self, processor):
        out = processor.enrich_fragments(self.fragments(), scan_metadata())
        assert processor.asked == []
        assert {f.position.page for f in out if f.type == "drawing"} == {1, 2, 4}

    def test_flag_on_undrawn_sheet_is_read_whole(self, processor, monkeypatch):
        monkeypatch.setattr(settings, "DRAWING_SHEET_MISSING_PAGES", True)
        out = processor.enrich_fragments(self.fragments(), scan_metadata())
        # Читается только лист 3: на остальных фрагменты drawing уже есть.
        assert processor.asked == [3]
        sheet = [f for f in out if f.type == "drawing" and f.position.page == 3]
        assert len(sheet) == 1
        assert sheet[0].content.structured_drawing_fields[0].value == "120*"
        # Фрагменты других листов не тронуты и не задвоены.
        assert sum(1 for f in out if f.type == "drawing") == 4

    def test_flag_on_all_sheets_drawn_reads_nothing(self, processor, monkeypatch):
        monkeypatch.setattr(settings, "DRAWING_SHEET_MISSING_PAGES", True)
        processor.enrich_fragments([drawn(1), drawn(2)], scan_metadata(pages=(1, 2)))
        assert processor.asked == []

    def test_flag_on_not_a_raster_drawing_is_left_alone(self, processor, monkeypatch):
        monkeypatch.setattr(settings, "DRAWING_SHEET_MISSING_PAGES", True)
        metadata = scan_metadata()
        metadata["classification"]["genre"] = "mixed"
        processor.enrich_fragments(self.fragments(), metadata)
        assert processor.asked == []


# --------------------------------------------------------------------------
# 2. Бледная тонкая рамка большого листа
# --------------------------------------------------------------------------

def faint_frame_sheet(width=4960, height=3507, line=90, thick=2):
    """Лист A3 при 300 dpi: рамка в 2 px средне-серым и семь строк текста."""
    from PIL import Image

    page = numpy.full((height, width), 255, dtype=numpy.uint8)
    top, bottom, left, right = 120, height - 120, 300, width - 100
    page[top:top + thick, left:right] = line
    page[bottom:bottom + thick, left:right] = line
    page[top:bottom, left:left + thick] = line
    page[top:bottom, right:right + thick] = line
    # Верхняя линия штампа — ещё одна длинная линия.
    page[height - 700:height - 700 + thick, width // 2:right] = line
    for index in range(7):
        y = 400 + index * 300
        page[y:y + 40, 800:1400] = 0
    buffer = io.BytesIO()
    Image.fromarray(page).save(buffer, format="PNG")
    return buffer.getvalue()


class TestThinFrame:
    def test_flag_off_faint_frame_is_lost(self):
        verdict = raster_drawing.analyse(faint_frame_sheet())
        assert not verdict.signals["frame"]
        assert not verdict.is_drawing

    def test_flag_on_faint_frame_makes_a_drawing(self, monkeypatch):
        monkeypatch.setattr(settings, "RASTER_FRAME_THIN_LINES", True)
        verdict = raster_drawing.analyse(faint_frame_sheet())
        assert verdict.kind == raster_drawing.KIND_DRAWING
        assert verdict.signals["thin_line_frame"] is True

    def test_flag_on_small_sheet_is_not_rechecked(self, monkeypatch):
        # Лист до ~3000 px уменьшается слабо: второй проход не нужен.
        monkeypatch.setattr(settings, "RASTER_FRAME_THIN_LINES", True)
        verdict = raster_drawing.analyse(faint_frame_sheet(width=2480, height=1754))
        assert "thin_line_frame" not in verdict.signals

    def test_flag_on_dense_text_page_stays_text(self, monkeypatch):
        from PIL import Image

        monkeypatch.setattr(settings, "RASTER_FRAME_THIN_LINES", True)
        page = numpy.asarray(Image.open(io.BytesIO(faint_frame_sheet()))).copy()
        # Тридцать строк текста короче половины листа: это страница текста.
        for index in range(30):
            y = 300 + index * 100
            page[y:y + 30, 800:2200] = 0
        buffer = io.BytesIO()
        Image.fromarray(page).save(buffer, format="PNG")
        verdict = raster_drawing.analyse(buffer.getvalue())
        assert not verdict.is_drawing


# --------------------------------------------------------------------------
# 3. Марка стали не посадка
# --------------------------------------------------------------------------

class TestMaterialTolerance:
    def test_flag_off_steel_grade_reads_as_fit(self):
        # Так модель и пишет марку: латиницей вперемешку с кириллицей.
        assert _field("material", "12X18H10Т ГОСТ 7350-77", [0, 0, 1, 1], 0.6)["tolerance"] == "H10"

    def test_flag_on_material_has_no_tolerance(self, monkeypatch):
        monkeypatch.setattr(settings, "DRAWING_MATERIAL_NO_TOLERANCE", True)
        for text in ("12X18H10Т ГОСТ 7350-77", "08X18H10Т", "материал: 12X18H10T ГОСТ 5582-75"):
            field = _field("material", text, [0, 0, 1, 1], 0.6)
            assert field["tolerance"] is None
            assert field["source_of_tolerance"] == "unknown"

    def test_flag_on_fit_on_size_is_kept(self, monkeypatch):
        monkeypatch.setattr(settings, "DRAWING_MATERIAL_NO_TOLERANCE", True)
        assert _field("size", "⌀20H7", [0, 0, 1, 1], 0.9)["tolerance"] == "H7"
