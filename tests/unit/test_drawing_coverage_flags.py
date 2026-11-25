"""
Покрытие чертежей и скорость ветки чертежей (аудит 25.11, N-04…N-23).

Каждая правка — за своим флагом, выключенным по умолчанию: с выключенным
флагом поведение и настройки в ключе кэша лестницы прежние. Картинки и PDF
здесь синтетические, модель подменена; обозначения и тексты придуманы.
"""

import threading
import time
from io import BytesIO

import pytest

from core import deadline
from core.config import settings
from core.drawing_processor import DrawingProcessor, _crop_lines, _field, _fields_from_text
from core.ladder import drawing_route, result_cache
from core.ladder.base import Strategy
from core.ladder.context import (
    GENRE_DRAWING,
    GENRE_UNKNOWN,
    SOURCE_RASTER,
    Classification,
    DocumentContext,
    _frame_sheet_pages,
    _raster_sheet_pages,
)
from core.ladder.router import LadderRouter
from core.models.contract import Content, Fragment, Position, Provenance, TableData
from core.models.parse_result import ParsedBlock
from core.providers import drawing_vision, raster_drawing
from core.providers.qwen_client import SheetReading

Image = pytest.importorskip("PIL.Image")
ImageDraw = pytest.importorskip("PIL.ImageDraw")
fitz = pytest.importorskip("fitz")


# --------------------------------------------------------------- синтетика
def drawn_sheet(size=(1400, 990), background=(255, 255, 255), ink=(0, 0, 0)):
    """Лист с рамкой формата, длинными линиями и штампом в углу."""
    image = Image.new("RGB", size, background)
    draw = ImageDraw.Draw(image)
    w, h = size
    draw.rectangle([int(w * .05), int(h * .03), int(w * .97), int(h * .97)], outline=ink, width=3)
    for y in (0.3, 0.5, 0.7):
        draw.line([int(w * .1), int(h * y), int(w * .9), int(h * y)], fill=ink, width=2)
    draw.line([int(w * .5), int(h * .1), int(w * .5), int(h * .9)], fill=ink, width=2)
    draw.rectangle([int(w * .6), int(h * .82), int(w * .97), int(h * .97)], outline=ink, width=2)
    return image


def png(image) -> bytes:
    data = BytesIO()
    image.save(data, format="PNG")
    return data.getvalue()


def colour_photo(size=(800, 600)):
    image = Image.new("RGB", size)
    draw = ImageDraw.Draw(image)
    for y in range(size[1]):
        draw.line([0, y, size[0], y], fill=(y % 256, 100, 200 - y // 3))
    draw.ellipse([200, 150, 600, 450], fill=(200, 50, 50))
    return image


def scan_pdf(drawing_pages, total):
    """Скан без текстового слоя: на `drawing_pages` — рамка и линии, прочие пусты."""
    document = fitz.open()
    sheet_png = png(drawn_sheet(size=(1169, 827)))
    for number in range(1, total + 1):
        page = document.new_page(width=842, height=595)
        if number in drawing_pages:
            page.insert_image(page.rect, stream=sheet_png)
    data = document.tobytes()
    document.close()
    return data


def crop(page, bbox=(0.2, 0.2, 0.6, 0.6), ref=None, text=None):
    return Fragment(
        fragment_id=f"doc-1-img-{page}-{bbox[0]}", type="image",
        content=Content(image_ref=ref or f"assets/doc-1/crop-{page}.png", text=text),
        position=Position(page=page, bbox=list(bbox), order=1),
        confidence=0.9, completeness=None,
        provenance=Provenance(method="mineru_ocr", strategy_level=5, source="scanned_pdf"),
    )


def result_of(fields=None, payload=None):
    return {"detected_count": len(fields or []), "avg_confidence": 0.75, "mode": "sheet_only",
            "parsed_fields": fields or [_field("size", "120", [0.1, 0.1, 0.2, 0.2], 0.75)],
            "payload": payload or {"tables": []}, "warnings": []}


class FakeQwen:
    available = True
    model = "fake-vl"


@pytest.fixture
def recording(temp_storage, monkeypatch):
    """Обработчик, у которого `process` записывает вызовы и ничего не читает."""
    proc = DrawingProcessor(temp_storage, qwen=FakeQwen())
    calls = []

    def process(image_ref, metadata, page=1, transcribe=False, **kwargs):
        calls.append({"ref": image_ref, "page": page, "transcribe": transcribe, **kwargs})
        return result_of()

    monkeypatch.setattr(proc, "process", process)
    proc.calls = calls
    return proc


# ------------------------------------------------------------ ключ кэша
COVERAGE_FLAGS = (
    "DRAWING_SHEETS_PER_PAGE", "DRAWING_KEEP_TITLE_BLOCK_REGION", "DRAWING_STAMP_FALLBACK_GOST",
    "RASTER_ALL_FRAMES", "RASTER_TINTED_LINE_CHECK", "LADDER_SKIP_OCR_ON_RASTER_DRAWING",
)


class TestCacheKey:
    def test_flags_off_leave_snapshot_unchanged(self):
        snapshot = result_cache.settings_snapshot()
        assert not set(COVERAGE_FLAGS) & set(snapshot)
        assert "DRAWING_PDF_FULL_READ" not in snapshot

    @pytest.mark.parametrize("flag", COVERAGE_FLAGS)
    def test_flag_on_changes_key(self, flag, monkeypatch):
        before = result_cache.cache_key("0" * 64, "pdf", {})
        monkeypatch.setattr(settings, flag, True)
        assert result_cache.cache_key("0" * 64, "pdf", {}) != before

    def test_defaults_are_off(self):
        for flag in COVERAGE_FLAGS + ("DRAWING_FIELDS_FROM_TEXT_WIDE", "DRAWING_PASSES_BY_GEOMETRY",
                                      "DRAWING_CROP_GATE", "DRAWING_SOURCE_CACHE"):
            assert getattr(type(settings)(), flag) is False
        assert type(settings)().DRAWING_CROPS_PARALLEL == 1
        assert type(settings)().DRAWING_STAMP_MAX_SIDE == 0


# ------------------------------------------------------- N-04 поля из текста
ELEMENT_TABLE = {
    "kind": drawing_vision.KIND_TABLE,
    "headers": ["Поз. обозначение", "Наименование", "Кол."],
    "rows": [["XT21", "Клемма пружинная КП-4", "12"], ["HL4, HL6...HL18", "Лампа сигнальная", "8"],
             ["R1", "", "1"]],
}


class TestFieldsFromTextWide:
    def test_off_element_rows_designations_short_notes_give_nothing(self, monkeypatch):
        monkeypatch.setattr(settings, "DRAWING_FIELDS_FROM_TEXT_WIDE", False)
        lines = ["ЖКЛМ.301245.007 СБ", "3) Окрасить", "DA3 Микросхема усилителя"]
        assert _fields_from_text([], lines, [ELEMENT_TABLE]) == []

    def test_on_element_rows_become_positions(self, monkeypatch):
        monkeypatch.setattr(settings, "DRAWING_FIELDS_FROM_TEXT_WIDE", True)
        values = {(f["category"], f["value"]) for f in _fields_from_text([], [], [ELEMENT_TABLE])}
        assert ("position", "XT21 Клемма пружинная КП-4") in values
        assert ("position", "HL4, HL6...HL18 Лампа сигнальная") in values
        # Голое обозначение без наименования полем не становится.
        assert not any(v.startswith("R1") for _, v in values)

    def test_on_lines_designation_note_and_element(self, monkeypatch):
        monkeypatch.setattr(settings, "DRAWING_FIELDS_FROM_TEXT_WIDE", True)
        lines = ["ЖКЛМ.301245.007 СБ", "3) Окрасить", "DA3 Микросхема усилителя",
                 "Вид А", "120"]
        values = {(f["category"], f["value"]) for f in _fields_from_text([], lines, [])}
        assert ("callout", "ЖКЛМ.301245.007 СБ") in values
        assert ("note", "3) Окрасить") in values
        assert ("position", "DA3 Микросхема усилителя") in values
        assert len(values) == 3

    def test_crop_lines_take_page_text_inside_the_crop(self):
        image = crop(1, bbox=(0.1, 0.1, 0.5, 0.5), text="подпись картинки")
        inside = Fragment(
            fragment_id="t1", type="text", content=Content(text="SA1 Выключатель кнопочный"),
            position=Position(page=1, bbox=[0.2, 0.2, 0.3, 0.25]), confidence=0.9, completeness=1.0,
            provenance=Provenance(method="mineru_ocr", strategy_level=5, source="scanned_pdf"))
        table = Fragment(
            fragment_id="t2", type="table",
            content=Content(table_data=TableData(rows=[["C12", "Конденсатор", "2"]])),
            position=Position(page=1, bbox=[0.1, 0.3, 0.5, 0.45]), confidence=0.9, completeness=1.0,
            provenance=Provenance(method="mineru_ocr", strategy_level=5, source="scanned_pdf"))
        outside = inside.model_copy(update={"fragment_id": "t3",
                                            "position": Position(page=1, bbox=[0.6, 0.6, 0.9, 0.7])})
        other_page = inside.model_copy(update={"fragment_id": "t4",
                                               "position": Position(page=2, bbox=[0.2, 0.2, 0.3, 0.25])})
        lines = _crop_lines(image, [image, inside, table, outside, other_page])
        assert lines == ["подпись картинки", "SA1 Выключатель кнопочный", "C12 | Конденсатор | 2"]

    def test_on_crop_gets_page_text_off_call_unchanged(self, recording, monkeypatch):
        image = crop(1, bbox=(0.1, 0.1, 0.5, 0.5))
        inside = Fragment(
            fragment_id="t1", type="text", content=Content(text="SA1 Выключатель кнопочный"),
            position=Position(page=1, bbox=[0.2, 0.2, 0.3, 0.25]), confidence=0.9, completeness=1.0,
            provenance=Provenance(method="mineru_ocr", strategy_level=5, source="scanned_pdf"))
        recording.enrich_fragments([image, inside], {"doc_id": "doc-1"})
        assert "crop_lines" not in recording.calls[0]
        recording.calls.clear()
        monkeypatch.setattr(settings, "DRAWING_FIELDS_FROM_TEXT_WIDE", True)
        recording.enrich_fragments([image, inside], {"doc_id": "doc-1"})
        assert recording.calls[0]["crop_lines"] == ["SA1 Выключатель кнопочный"]

    def test_process_uses_crop_lines_without_transcript(self, temp_storage, monkeypatch):
        class Qwen(FakeQwen):
            def read_sheet(self, image, hint="", **_):
                return SheetReading(sheet_type="scheme", annotations=[
                    {"category": "size", "value": "⌀20", "bbox": [0.1, 0.1, 0.2, 0.2]}])

        monkeypatch.setattr(settings, "DRAWING_VISION_ENABLED", False)
        monkeypatch.setattr(settings, "DRAWING_SHEET_GUARDED", False)
        temp_storage.write_file("assets/crop.png", png(Image.new("RGB", (600, 400), "white")))
        proc = DrawingProcessor(temp_storage, qwen=Qwen())

        def values(**kwargs):
            out = proc.process("assets/crop.png", {}, 1, log_page=1, **kwargs)
            return {f["value"] for f in out["parsed_fields"]}

        lines = ["SA1 Выключатель кнопочный"]
        assert values(crop_lines=lines) == {"⌀20"}
        monkeypatch.setattr(settings, "DRAWING_FIELDS_FROM_TEXT_WIDE", True)
        assert values(crop_lines=lines) == {"⌀20", "SA1 Выключатель кнопочный"}


# ---------------------------------------------- N-08 листы по каждой странице
def per_page_metadata(pages, genre=GENRE_UNKNOWN, file_type="pdf"):
    return {
        "doc_id": "doc-1", "source_path": "src/scan.pdf", "file_type": file_type,
        "classification": {"genre": genre, "source": SOURCE_RASTER,
                           "signals": {"raster_drawing_pages": [], "raster_sheet_pages": list(pages)}},
    }


class TestSheetsPerPage:
    def test_all_pages_are_checked(self):
        data = scan_pdf(drawing_pages={2, 10, 11}, total=12)
        assert _raster_sheet_pages(data) == [2, 10, 11]

    def test_classification_genre_unchanged_and_pages_recorded(self, monkeypatch):
        data = scan_pdf(drawing_pages={2, 10, 11}, total=12)
        context = DocumentContext(uri="src/scan.pdf", file_type="pdf", data=data)
        before = context.classification
        assert "raster_sheet_pages" not in before.signals
        monkeypatch.setattr(settings, "DRAWING_SHEETS_PER_PAGE", True)
        after = DocumentContext(uri="src/scan.pdf", file_type="pdf", data=data).classification
        assert after.genre == before.genre
        assert after.signals["raster_drawing_pages"] == before.signals["raster_drawing_pages"]
        assert after.signals["raster_sheet_pages"] == [2, 10, 11]

    def test_off_minority_scan_gets_no_sheet_read(self, recording):
        recording.enrich_fragments([crop(2)], per_page_metadata([2, 10, 11]))
        assert [c["page"] for c in recording.calls if c["ref"] == "src/scan.pdf"] == []

    def test_on_every_sheet_page_is_read_without_cap(self, recording, monkeypatch):
        monkeypatch.setattr(settings, "DRAWING_SHEETS_PER_PAGE", True)
        monkeypatch.setattr(settings, "DRAWING_PDF_FULL_READ", True)
        monkeypatch.setattr(settings, "DRAWING_MAX_SHEETS", 2)
        out = recording.enrich_fragments([crop(2)], per_page_metadata([2, 10, 11]))
        sheet_calls = [c for c in recording.calls if c["ref"] == "src/scan.pdf"]
        assert [(c["page"], c["transcribe"]) for c in sheet_calls] == [(2, True), (10, True), (11, True)]
        pages = [f.position.page for f in out]
        assert pages == sorted(pages)
        assert not any("DRAWING_SHEETS_SKIPPED" in w for f in out for w in f.warnings)

    def test_on_out_of_time_reports_skipped(self, recording, monkeypatch):
        # Срок кончается после первого листа: остальные — в DRAWING_SHEETS_SKIPPED.
        monkeypatch.setattr(settings, "DRAWING_SHEETS_PER_PAGE", True)
        left = iter([1000.0])
        monkeypatch.setattr(deadline, "remaining", lambda: next(left, 0.0))
        out = recording.enrich_fragments([], per_page_metadata([2, 10, 11]))
        assert [f.position.page for f in out] == [2]
        assert "DRAWING_SHEETS_SKIPPED:10,11" in out[0].warnings


# --------------------------------------------- N-09 штамп и потолок областей
class TestTitleBlockSurvivesSlice:
    def _layout(self, monkeypatch):
        views = [drawing_vision.Region(kind=drawing_vision.KIND_TABLE,
                                       bbox=[0.05 + 0.2 * i, 0.1, 0.2 + 0.2 * i, 0.5]) for i in range(4)]
        stamp = drawing_vision.Region(kind=drawing_vision.KIND_TABLE, bbox=[0.75, 0.88, 0.97, 0.97])
        monkeypatch.setattr(drawing_vision, "_find_regions",
                            lambda image: (None, [*views, stamp], {}))
        monkeypatch.setattr(drawing_vision, "_orientation_turn", lambda *a, **k: (0, None))
        monkeypatch.setattr(settings, "DRAWING_MAX_REGIONS", 4)
        return drawing_vision.analyse(Image.new("RGB", (1400, 990), "white"), deskew_limit=0)

    def test_off_stamp_is_cut(self, monkeypatch):
        assert self._layout(monkeypatch).title_block is None

    def test_on_stamp_kept_first(self, monkeypatch):
        monkeypatch.setattr(settings, "DRAWING_KEEP_TITLE_BLOCK_REGION", True)
        layout = self._layout(monkeypatch)
        assert layout.regions[0].kind == drawing_vision.KIND_TITLE_BLOCK
        assert len(layout.regions) == 4


# -------------------------------------------------- N-15 вырезка штампа
class TestStampCrop:
    def test_gost_box_on_a1_landscape(self):
        # A1 альбомный при 200 dpi: 841×594 мм.
        size = (round(841 / 25.4 * 200), round(594 / 25.4 * 200))
        box = drawing_vision.gost_fallback_box(size, 200)
        assert box[0] == pytest.approx(1 - 200 / 841, abs=1e-3)
        assert box[1] == pytest.approx(1 - 70 / 594, abs=1e-3)
        assert drawing_vision.gost_fallback_box(size, None) is None

    def test_gost_box_never_wider_than_old_fallback(self):
        # A4 книжный: штамп во всю ширину рамки — прежняя граница.
        box = drawing_vision.gost_fallback_box((1654, 2339), 200)
        assert box[0] == pytest.approx(0.05)
        assert box[1] > 0.74

    def test_fallback_uses_gost_only_with_flag(self, monkeypatch):
        layout = drawing_vision.SheetLayout(image=Image.new("RGB", (6622, 4677), "white"))
        assert drawing_vision.fallback_title_block(layout, dpi=200).bbox == [0.5, 0.74, 1.0, 1.0]
        monkeypatch.setattr(settings, "DRAWING_STAMP_FALLBACK_GOST", True)
        assert drawing_vision.fallback_title_block(layout, dpi=200).bbox[0] > 0.7
        assert drawing_vision.fallback_title_block(layout).bbox == [0.5, 0.74, 1.0, 1.0]

    def test_stamp_cap_is_a_setting(self, temp_storage, monkeypatch):
        seen = []

        class Qwen(FakeQwen):
            def read_title_block(self, image, **kwargs):
                seen.append((max(image.size), kwargs))
                return {"fields": {}, "rows": []}

        proc = DrawingProcessor(temp_storage, qwen=Qwen())
        layout = drawing_vision.SheetLayout(image=Image.new("RGB", (6000, 4000), "white"))
        region = drawing_vision.Region(kind=drawing_vision.KIND_TITLE_BLOCK, bbox=[0.5, 0.74, 1.0, 1.0])
        monkeypatch.setattr(settings, "DRAWING_TITLE_BLOCK_PASS", False)
        monkeypatch.setattr(settings, "DRAWING_NATIVE_TILES", False)
        proc._read_title_block(layout, region, [], {"tables": []})
        monkeypatch.setattr(settings, "DRAWING_STAMP_MAX_SIDE", 2600)
        proc._read_title_block(layout, region, [], {"tables": []})
        assert seen[0] == (settings.QWEN_REGION_MAX_SIDE, {})
        assert seen[1] == (2600, {"max_side": 2600})


# -------------------------------------------- N-16 проходы по геометрии листа
class TestPassesByGeometry:
    def _run(self, temp_storage, monkeypatch, sheet_type, frame):
        class Qwen(FakeQwen):
            def read_sheet(self, image, hint="", **_):
                return SheetReading(sheet_type=sheet_type, annotations=[
                    {"category": "size", "value": "⌀20", "bbox": [0.1, 0.1, 0.2, 0.2]}])

        temp_storage.write_file("src/sheet.png", png(Image.new("RGB", (1400, 990), "white")))
        proc = DrawingProcessor(temp_storage, qwen=Qwen())
        monkeypatch.setattr(settings, "DRAWING_MARGIN_COLUMNS", True)
        monkeypatch.setattr(settings, "DRAWING_SHEET_GUARDED", False)
        monkeypatch.setattr(drawing_vision, "analyse", lambda image: drawing_vision.SheetLayout(
            image=image, frame=[0.05, 0.03, 0.97, 0.97] if frame else None))
        called = []
        monkeypatch.setattr(proc, "_margin_columns", lambda *a, **k: called.append(True))
        proc.process("src/sheet.png", {}, 1)
        return bool(called)

    def test_off_scheme_with_frame_has_no_margin_pass(self, temp_storage, monkeypatch):
        assert not self._run(temp_storage, monkeypatch, "scheme", frame=True)

    def test_on_scheme_with_frame_gets_margin_pass(self, temp_storage, monkeypatch):
        monkeypatch.setattr(settings, "DRAWING_PASSES_BY_GEOMETRY", True)
        assert self._run(temp_storage, monkeypatch, "scheme", frame=True)

    def test_on_unknown_type_without_frame_gets_margin_pass(self, temp_storage, monkeypatch):
        monkeypatch.setattr(settings, "DRAWING_PASSES_BY_GEOMETRY", True)
        assert self._run(temp_storage, monkeypatch, "unknown", frame=False)

    def test_on_scheme_without_frame_has_no_margin_pass(self, temp_storage, monkeypatch):
        monkeypatch.setattr(settings, "DRAWING_PASSES_BY_GEOMETRY", True)
        assert not self._run(temp_storage, monkeypatch, "scheme", frame=False)


# ----------------------------------------- N-17 многокадровый TIFF и EXIF
def tiff(frames) -> bytes:
    data = BytesIO()
    frames[0].save(data, format="TIFF", save_all=True, append_images=frames[1:])
    return data.getvalue()


def rotated_jpeg() -> bytes:
    """Снимок 300×200, EXIF говорит «повернуть на 90°» (ориентация 6)."""
    image = Image.new("RGB", (300, 200), "white")
    exif = Image.Exif()
    exif[0x0112] = 6
    data = BytesIO()
    image.save(data, format="JPEG", exif=exif.tobytes())
    return data.getvalue()


class TestFramesAndExif:
    def test_off_first_frame_and_raw_orientation(self):
        data = tiff([drawn_sheet(), Image.new("RGB", (1400, 990), "white")])
        assert raster_drawing.open_image(data, frame=1).getpixel((5, 5)) == raster_drawing.open_image(
            data).getpixel((5, 5))
        assert raster_drawing.open_image(rotated_jpeg()).size == (300, 200)

    def test_on_frames_and_exif(self, monkeypatch):
        monkeypatch.setattr(settings, "RASTER_ALL_FRAMES", True)
        blank = Image.new("RGB", (1400, 990), (250, 250, 250))
        data = tiff([drawn_sheet(), blank, drawn_sheet()])
        assert raster_drawing.open_image(data, frame=1).getpixel((700, 300)) == (250, 250, 250)
        assert raster_drawing.open_image(data, frame=3) is None
        assert raster_drawing.open_image(rotated_jpeg()).size == (200, 300)
        signals = {}
        _frame_sheet_pages(data, signals)
        assert signals["raster_drawing_pages"] == [1, 3]
        assert signals["frame_count"] == 3

    def test_on_ocr_reads_every_frame(self, tmp_path, monkeypatch):
        from core.providers.tesseract_fallback import TesseractFallbackProvider

        path = tmp_path / "sheets.tif"
        path.write_bytes(tiff([drawn_sheet(), drawn_sheet(), drawn_sheet()]))
        provider = TesseractFallbackProvider.__new__(TesseractFallbackProvider)
        assert [n for n, _ in provider._images_of(str(path), "tif")] == [1]
        monkeypatch.setattr(settings, "RASTER_ALL_FRAMES", True)
        assert [n for n, _ in provider._images_of(str(path), "tif")] == [1, 2, 3]

    def test_on_sheet_branch_opens_the_frame(self, temp_storage, monkeypatch):
        blank = Image.new("RGB", (1400, 990), (250, 250, 250))
        temp_storage.write_file("src/sheets.tif", tiff([drawn_sheet(), blank]))
        proc = DrawingProcessor(temp_storage, qwen=FakeQwen())
        assert proc._open("src/sheets.tif", 2).getpixel((700, 300)) != (250, 250, 250)
        monkeypatch.setattr(settings, "RASTER_ALL_FRAMES", True)
        assert proc._open("src/sheets.tif", 2).getpixel((700, 300)) == (250, 250, 250)


# --------------------------------------------- N-23 тонированный скан
class TestTintedScan:
    @pytest.mark.parametrize("background, ink", [
        ((232, 208, 140), (60, 40, 20)),    # пожелтевшая бумага
        ((30, 60, 150), (230, 230, 255)),   # синька
        ((120, 120, 120), (20, 20, 20)),    # тёмный серый скан
    ])
    def test_line_drawing_on_tinted_paper(self, background, ink, monkeypatch):
        data = png(drawn_sheet(background=background, ink=ink))
        assert not raster_drawing.analyse(data).is_drawing
        monkeypatch.setattr(settings, "RASTER_TINTED_LINE_CHECK", True)
        assert raster_drawing.analyse(data).kind == raster_drawing.KIND_DRAWING

    def test_colour_photo_stays_photo(self, monkeypatch):
        monkeypatch.setattr(settings, "RASTER_TINTED_LINE_CHECK", True)
        assert raster_drawing.analyse(png(colour_photo())).kind == raster_drawing.KIND_PHOTO


# ------------------------------------------------ N-06 отбор вырезок
class TestCropGate:
    def _fragments(self, temp_storage):
        temp_storage.write_file("assets/drawing.png", png(drawn_sheet(size=(900, 640))))
        temp_storage.write_file("assets/photo.png", png(colour_photo()))
        return [crop(1, ref="assets/drawing.png"), crop(1, ref="assets/photo.png"),
                crop(2, bbox=(0.1, 0.1, 0.15, 0.15), ref="assets/drawing.png"),
                crop(3, ref="assets/drawing.png"), crop(4, ref="assets/drawing.png")]

    def test_off_every_crop_goes_to_model(self, recording, temp_storage):
        recording.enrich_fragments(self._fragments(temp_storage), {"doc_id": "doc-1"})
        assert len(recording.calls) == 5

    def test_on_small_and_photo_skipped_and_capped(self, recording, temp_storage, monkeypatch):
        monkeypatch.setattr(settings, "DRAWING_CROP_GATE", True)
        monkeypatch.setattr(settings, "DRAWING_CROP_MAX_PER_DOC", 2)
        out = recording.enrich_fragments(self._fragments(temp_storage), {"doc_id": "doc-1"})
        assert [(c["ref"], c["log_page"]) for c in recording.calls] == [
            ("assets/drawing.png", 1), ("assets/drawing.png", 3)]
        assert [f.type for f in out] == ["drawing", "image", "image", "drawing", "image"]
        assert "DRAWING_CROP_SKIPPED_CAP" in out[4].warnings
        assert "DRAWING_CROP_SKIPPED_CAP" not in out[1].warnings


# ------------------------------------------- N-18 исходник на время документа
class CountingStorage:
    def __init__(self, files):
        self.files, self.reads = files, 0

    def read_bytes(self, uri):
        self.reads += 1
        return self.files[uri]


class TestSourceCache:
    def _run(self, monkeypatch):
        storage = CountingStorage({"src/scan.pdf": scan_pdf({1, 2}, 2)})
        proc = DrawingProcessor(storage, qwen=FakeQwen())
        opened = []
        real_open = fitz.open
        monkeypatch.setattr(fitz, "open", lambda *a, **k: opened.append(1) or real_open(*a, **k))

        def process(image_ref, metadata, page=1, **kwargs):
            # Лист и два прохода по нему — как сам лист, геометрия и штамп.
            for _ in range(3):
                assert proc._open(image_ref, page) is not None
            return result_of()

        monkeypatch.setattr(proc, "process", process)
        metadata = {"doc_id": "doc-1", "source_path": "src/scan.pdf", "file_type": "pdf",
                    "classification": {"genre": GENRE_DRAWING, "source": SOURCE_RASTER,
                                       "signals": {"raster_drawing_pages": [1, 2]}}}
        out = proc.enrich_fragments([], metadata)
        assert [f.position.page for f in out] == [1, 2]
        return storage.reads, len(opened), proc

    def test_off_reads_every_time(self, monkeypatch):
        reads, opened, _ = self._run(monkeypatch)
        assert reads == 6 and opened == 6

    def test_on_reads_once_and_releases(self, monkeypatch):
        monkeypatch.setattr(settings, "DRAWING_SOURCE_CACHE", True)
        reads, opened, proc = self._run(monkeypatch)
        assert reads == 1 and opened == 1
        assert proc._source_cache is None and proc._pdf_cache is None


# ------------------------------------- N-19 уровни 4 и 6 у растрового чертежа
class Level(Strategy):
    def __init__(self, level, ran):
        self.level, self.name, self.method, self.ran = level, f"level{level}", "stub", ran

    def applicable(self, context):
        return True

    def run(self, context):
        self.ran.append(self.level)
        block = ParsedBlock(type="image", image_ref="assets/doc/sheet.jpg", page=1,
                            bbox=[0.02, 0.02, 0.98, 0.98], confidence=0.9)
        if self.level == 4:
            block = ParsedBlock(type="text", text="а" * 40, page=1, confidence=0.6, is_fallback=True)
        return self.build_result([block], source_kind="image", page_count=1,
                                 is_fallback=self.level == 4)


def sheet_context(genre=GENRE_DRAWING):
    context = DocumentContext(uri="documents/sheet", file_type="jpg", data=b"\xff\xd8\xff" + b"\0" * 64)
    context._classification = Classification(source=SOURCE_RASTER, genre=genre, raster_quality=0.8)
    return context


class TestSkipOcrOnRasterDrawing:
    def _levels(self, genre=GENRE_DRAWING, levels=(4, 5, 6)):
        ran = []
        LadderRouter([Level(n, ran) for n in levels], threshold=0.99).parse(sheet_context(genre))
        return ran

    def test_off_all_levels_run(self, monkeypatch):
        monkeypatch.setattr(settings, "DRAWING_IMAGE_FULL_READ", True)
        assert self._levels() == [4, 5, 6]

    def test_on_ocr_levels_skipped(self, monkeypatch):
        monkeypatch.setattr(settings, "LADDER_SKIP_OCR_ON_RASTER_DRAWING", True)
        monkeypatch.setattr(settings, "DRAWING_IMAGE_FULL_READ", True)
        assert self._levels() == [5]

    def test_on_needs_full_read_drawing_and_layout(self, monkeypatch):
        monkeypatch.setattr(settings, "LADDER_SKIP_OCR_ON_RASTER_DRAWING", True)
        assert self._levels() == [4, 5, 6]          # без полного чтения листа
        monkeypatch.setattr(settings, "DRAWING_IMAGE_FULL_READ", True)
        assert self._levels(genre=GENRE_UNKNOWN) == [4, 5, 6]
        assert self._levels(levels=(4, 6)) == [4, 6]

    def test_pdf_needs_pdf_full_read(self, monkeypatch):
        monkeypatch.setattr(settings, "LADDER_SKIP_OCR_ON_RASTER_DRAWING", True)
        context = DocumentContext(uri="documents/scan", file_type="pdf", data=b"%PDF-1.4\n")
        context._classification = Classification(source=SOURCE_RASTER, genre=GENRE_DRAWING)
        assert not drawing_route.skip_ocr_levels(context, [4, 5, 6])
        monkeypatch.setattr(settings, "DRAWING_PDF_FULL_READ", True)
        assert drawing_route.skip_ocr_levels(context, [4, 5, 6])


# ------------------------------------------- N-20 вырезки одновременно
class TestCropsParallel:
    def _run(self, temp_storage, monkeypatch, workers):
        proc = DrawingProcessor(temp_storage, qwen=FakeQwen())
        lock = threading.Lock()
        state = {"now": 0, "max": 0, "order": []}

        def process(image_ref, metadata, page=1, log_page=None, page_fields=None, **kwargs):
            with lock:
                state["now"] += 1
                state["max"] = max(state["max"], state["now"])
            time.sleep(0.05)
            with lock:
                state["now"] -= 1
                state["order"].append(image_ref)
            page_fields.append({"value": image_ref})
            return result_of(fields=[_field("size", image_ref[-6:-4], [0.1, 0.1, 0.2, 0.2], 0.75)])

        monkeypatch.setattr(proc, "process", process)
        monkeypatch.setattr(settings, "DRAWING_CROPS_PARALLEL", workers)
        fragments = [crop(page, bbox=(0.1 * n, 0.2, 0.1 * n + 0.3, 0.6), ref=f"assets/c-{page}{n}.png")
                     for page in (1, 2, 3) for n in (1, 2)]
        out = proc.enrich_fragments(fragments, {"doc_id": "doc-1"})
        return out, state

    def test_off_one_at_a_time(self, temp_storage, monkeypatch):
        out, state = self._run(temp_storage, monkeypatch, 1)
        assert state["max"] == 1
        assert [f.content.structured_drawing_fields[0].value for f in out] == [
            "11", "12", "21", "22", "31", "32"]

    def test_on_pages_overlap_same_page_in_order(self, temp_storage, monkeypatch):
        out, state = self._run(temp_storage, monkeypatch, 3)
        assert state["max"] > 1
        assert [f.content.structured_drawing_fields[0].value for f in out] == [
            "11", "12", "21", "22", "31", "32"]
        order = state["order"]
        for page in (1, 2, 3):
            assert order.index(f"assets/c-{page}1.png") < order.index(f"assets/c-{page}2.png")
