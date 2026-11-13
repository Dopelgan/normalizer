"""
Лист-чертёж картинкой: чертёжный разбор против сплошного распознавания
(LADDER_DRAWING_OVER_OCR).

Числа в тестах сняты с настоящих листов (13.11.2026, одиночные JPG): MinerU
возвращает лист одной картинкой без текста, сплошное распознавание — строки
Tesseract. Лестница сравнивала их по числу символов, непрочитанный лист
проигрывал всегда, и документ уходил в индекс как разобранный фоллбэком.
"""

import pytest

from core.config import settings
from core.ladder import drawing_route
from core.ladder.base import Strategy
from core.ladder.context import (
    GENRE_DRAWING,
    GENRE_TEXT,
    SOURCE_RASTER,
    Classification,
    DocumentContext,
)
from core.ladder.router import LadderRouter
from core.models.contract import Content, DrawingField, Fragment, Position, Provenance
from core.models.parse_result import ParsedBlock, ParseResult
from core.normalizer import TextNormalizer
from tests.unit.test_tasks import StubParser, task_db  # noqa: F401 — фикстура задачи

JPG = b"\xff\xd8\xff\xe0" + b"\x00" * 64
PDF = b"%PDF-1.4\n"

# Лист: (символов у plain_ocr, символов у layout) — из журнала разбора.
REAL_SHEETS = {
    "ТАДУ 421451.003 Э4, лист 3": (1692, 0),
    "ТАДУ 421451.003-03 Э4, лист 2": (2985, 17),
    "ТАДУ 421451.003.05.004": (722, 10),
    "гильза 408721.917.00 СБ, лист 2": (202, 20),
    "Рис. к ТЗ изм. 1": (66, 0),
}


def text_of(chars: int) -> str:
    """Текст ровно с `chars` символами без пробелов, словами по семь букв."""
    words = ["а" * 7] * (chars // 7) + (["б" * (chars % 7)] if chars % 7 else [])
    return " ".join(words)


class OcrLevel(Strategy):
    level, name, method = 4, "plain_ocr", "tesseract"

    def __init__(self, chars):
        self.chars = chars

    def applicable(self, context):
        return True

    def run(self, context):
        block = ParsedBlock(type="text", text=text_of(self.chars), page=1,
                            confidence=0.62, is_fallback=True)
        return self.build_result([block], source_kind="image", page_count=1, is_fallback=True)


class LayoutLevel(Strategy):
    """MinerU на листе-чертеже: лист одной картинкой и обрывок подписи."""

    level, name, method = 5, "layout", "mineru_ocr"

    def __init__(self, chars):
        self.chars = chars

    def applicable(self, context):
        return True

    def run(self, context):
        blocks = [ParsedBlock(type="image", image_ref="assets/doc/sheet.jpg", page=1,
                              bbox=[0.02, 0.02, 0.98, 0.98], confidence=0.94)]
        if self.chars:
            blocks.append(ParsedBlock(type="text", text=text_of(self.chars), page=1,
                                      bbox=[0.4, 0.9, 0.6, 0.95], confidence=0.5))
        return self.build_result(blocks, source_kind="image", page_count=1)


def sheet_context(genre=GENRE_DRAWING, data=JPG, file_type="jpg"):
    context = DocumentContext(uri="documents/sheet", file_type=file_type, data=data)
    context._classification = Classification(
        source=SOURCE_RASTER, genre=genre, raster_quality=0.77,
    )
    return context


def route(ocr_chars, layout_chars, **kwargs):
    router = LadderRouter([OcrLevel(ocr_chars), LayoutLevel(layout_chars)], threshold=0.75)
    return router.parse(sheet_context(**kwargs))


@pytest.fixture
def flag_on(monkeypatch):
    monkeypatch.setattr(settings, "LADDER_DRAWING_OVER_OCR", True)
    monkeypatch.setattr(settings, "DRAWING_IMAGE_FULL_READ", True)


class TestLadderOnDrawingSheet:
    @pytest.mark.parametrize("sheet", sorted(REAL_SHEETS))
    def test_without_flag_plain_ocr_wins_by_characters(self, sheet):
        """Прежнее поведение: непрочитанный лист проигрывает счёт символов."""
        result = route(*REAL_SHEETS[sheet])
        assert result.ladder["chosen_strategy"] == "plain_ocr"
        assert result.ladder["fallback_won"] is True
        assert result.fallback_reserve is None

    @pytest.mark.parametrize("sheet", sorted(REAL_SHEETS))
    def test_drawing_sheet_goes_to_the_drawing_route(self, flag_on, sheet):
        ocr_chars, layout_chars = REAL_SHEETS[sheet]
        result = route(ocr_chars, layout_chars)
        assert result.ladder["chosen_strategy"] == "layout"
        assert result.ladder["fallback_won"] is False
        assert result.is_fallback is False
        assert result.degraded == []
        # Распознавание не выброшено: оно ждёт сверки после чтения листа.
        assert result.fallback_reserve is not None
        assert result.fallback_reserve.parser_name == "plain_ocr"
        assert result.ladder["drawing_deferred"] == {
            "reserve_strategy": "plain_ocr", "reserve_chars": ocr_chars,
        }

    def test_text_scan_keeps_the_character_check(self, flag_on):
        """Страница текста: разбор, потерявший текст, по-прежнему проигрывает."""
        result = route(1692, 0, genre=GENRE_TEXT)
        assert result.ladder["chosen_strategy"] == "plain_ocr"
        assert result.fallback_reserve is None

    def test_pdf_scan_keeps_the_character_check(self, flag_on):
        """У скана PDF листов много и среди них бывает текст: правило не трогает."""
        result = route(1692, 0, data=PDF, file_type="pdf")
        assert result.ladder["chosen_strategy"] == "plain_ocr"
        assert result.fallback_reserve is None

    def test_no_reserve_when_layout_kept_the_text(self, flag_on):
        """Разбор сам принёс текст листа — запас не нужен, сверять нечего."""
        result = route(1000, 900)
        assert result.ladder["chosen_strategy"] == "layout"
        assert result.fallback_reserve is None
        assert "drawing_deferred" not in result.ladder

    def test_needs_full_sheet_read(self, monkeypatch):
        """Без полного чтения листа текста у модели нет: правило не включается."""
        monkeypatch.setattr(settings, "LADDER_DRAWING_OVER_OCR", True)
        monkeypatch.setattr(settings, "DRAWING_IMAGE_FULL_READ", False)
        result = route(1692, 0)
        assert result.ladder["chosen_strategy"] == "plain_ocr"
        assert result.fallback_reserve is None

    def test_advance_seniority_does_not_stop_the_ladder(self, flag_on):
        """
        Разбор дотянул до порога, но текст потерял: старшинство авансом не
        останавливает лестницу, следующий уровень всё равно пробуется.
        """

        class NextLevel(LayoutLevel):
            level, name = 7, "vlm_escalation"
            ran = False

            def run(self, context):
                type(self).ran = True
                return super().run(context)

        router = LadderRouter(
            [OcrLevel(2985), LayoutLevel(140), NextLevel(0)], threshold=0.6,
        )
        result = router.parse(sheet_context())
        assert NextLevel.ran is True
        assert result.is_fallback is False
        assert result.fallback_reserve is not None

    def test_reserve_is_not_written_to_the_raw_result(self, flag_on):
        result = route(1692, 0)
        assert "fallback_reserve" not in result.model_dump(mode="json")


def drawing_sheet(lines=(), fields=(), rows=(), page=1, fragment_id="doc-frag-000"):
    payload = {"tables": [{"kind": "table", "rows": [list(r) for r in rows]}] if rows else []}
    if lines:
        payload["transcript_line_count"] = len(lines)
    return Fragment(
        fragment_id=fragment_id, type="drawing",
        content=Content(
            text="\n".join(lines) or None, image_ref="documents/sheet",
            structured_drawing_fields=[
                DrawingField(category="size", value=value, confidence=0.9) for value in fields
            ],
            structured_payload=payload,
        ),
        position=Position(page=page, bbox=[0, 0, 1, 1], order=0),
        confidence=0.9, completeness=None,
        provenance=Provenance(method="vision_plus_vlm", strategy_level=7, source="drawing"),
    )


def unread_picture():
    return Fragment(
        fragment_id="doc-frag-001", type="image",
        content=Content(image_ref="assets/doc/sheet.jpg"),
        position=Position(page=1, bbox=[0.02, 0.02, 0.98, 0.98], order=1),
        confidence=0.9, completeness=1.0, warnings=["DRAWING_NOT_RECOGNIZED"],
        provenance=Provenance(method="mineru_image", strategy_level=5, source="raster_only"),
    )


def chosen_with_reserve(ocr_chars):
    """Результат лестницы при включённом флаге: layout выбран, OCR в запасе."""
    before = settings.LADDER_DRAWING_OVER_OCR, settings.DRAWING_IMAGE_FULL_READ
    settings.LADDER_DRAWING_OVER_OCR = settings.DRAWING_IMAGE_FULL_READ = True
    try:
        return route(ocr_chars, 0)
    finally:
        settings.LADDER_DRAWING_OVER_OCR, settings.DRAWING_IMAGE_FULL_READ = before


METADATA = {"doc_id": "doc", "source_path": "documents/sheet", "file_type": "jpg"}


class TestVolume:
    def test_counts_fields_tables_and_transcript_not_only_text(self):
        got = drawing_route.volume([
            drawing_sheet(lines=["ТАДУ 421451.003 Э4", "XT1 XT2"], fields=["Ø12", "45"],
                          rows=[["1", "Гильза", "1"]]),
        ])
        assert got.fields == 2
        assert got.table_rows == 1
        assert got.transcript_lines == 2
        assert got.read == 5
        assert got.chars == len("ТАДУ421451.003Э4XT1XT2")

    def test_sheet_read_without_transcript_is_measured_by_fields(self):
        """Лист без переписанного текста не пуст: размеры и таблица — содержимое."""
        got = drawing_route.volume([
            drawing_sheet(fields=["Ø12H7", "45±0,1"], rows=[["1", "Гильза"]]),
        ])
        assert got.chars == len("Ø12H7") + len("45±0,1") + len("1Гильза")

    def test_same_sheet_read_twice_is_not_counted_twice(self):
        """Вырезка вида и лист целиком — одно содержимое: берётся лучшее чтение."""
        sheet = drawing_sheet(lines=["Ø12H7 45 М8"], fields=["Ø12H7"])
        crop = drawing_sheet(fields=["Ø12H7", "45"], fragment_id="doc-frag-001")
        assert drawing_route.volume([sheet, crop]).chars == len("Ø12H745М8")

    def test_title_block_table_is_not_counted_on_top_of_its_fields(self):
        sheet = drawing_sheet(fields=["Гильза"])
        sheet.content.structured_payload["tables"] = [
            {"kind": "title_block", "rows": [["Гильза", "ТАДУ 408721.917.00"]]},
        ]
        got = drawing_route.volume([sheet])
        assert got.tables == 0 and got.table_rows == 0
        assert got.chars == len("Гильза")


class TestSettleAfterSheetRead:
    def test_nothing_to_settle_without_reserve(self):
        result = ParseResult(blocks=[ParsedBlock(type="text", text="текст", page=1)])
        fragments = [drawing_sheet(lines=["строка"])]
        settled, flags = drawing_route.settle(fragments, result, METADATA, TextNormalizer())
        assert settled is fragments and flags == {}
        assert "drawing_route" not in result.ladder
        assert drawing_route.lost(result) is False

    def test_read_sheet_wins_over_recognition(self):
        """Э4, лист 3: 1692 символа распознавания против переписанного листа."""
        result = chosen_with_reserve(1692)
        lines = [f"XT{i} A{i} B{i} 0,75" for i in range(200)]
        fragments = [drawing_sheet(lines=lines, fields=["XT1", "XT2"])]
        settled, flags = drawing_route.settle(fragments, result, METADATA, TextNormalizer())
        assert [f.type for f in settled] == ["drawing"]
        assert flags == {}
        assert result.ladder["drawing_route"]["won"] is True
        assert result.ladder["drawing_route"]["reserve_chars"] == 1692
        assert result.ladder["fallback_won"] is False
        assert result.degraded == []
        assert result.fallback_reserve is None
        assert drawing_route.lost(result) is False

    def test_sheet_with_few_words_wins_by_what_was_read(self):
        """Рис. к ТЗ: 66 символов распознавания, у модели — 15 строк и 11 полей."""
        result = chosen_with_reserve(66)
        lines = ["Рис. 1", "Вид А", "Б-Б (2:1)", "Ø12", "45", "М8", "R3", "2 фаски",
                 "1×45°", "Ra 6,3", "120", "80", "поз. 1", "поз. 2", "поз. 3"]
        fragments = [drawing_sheet(lines=lines, fields=lines[3:14])]
        settled, _ = drawing_route.settle(fragments, result, METADATA, TextNormalizer())
        assert [f.type for f in settled] == ["drawing"]
        assert result.ladder["drawing_route"]["won"] is True

    def test_unread_sheet_gets_recognition_text_back(self):
        """Модель лист не прочитала: текст распознавания не теряется."""
        result = chosen_with_reserve(1692)
        fragments = [unread_picture()]
        settled, flags = drawing_route.settle(fragments, result, METADATA, TextNormalizer())
        restored = [f for f in settled if f.type == "text"]
        assert settled[0].type == "image"
        assert sum(len("".join((f.content.text or "").split())) for f in restored) == 1692
        assert all(f.review_required for f in restored)
        assert all(f.provenance.method == "tesseract" for f in restored)
        assert set(flags) == {f.fragment_id for f in restored}
        assert result.ladder["drawing_route"]["won"] is False
        assert result.ladder["drawing_route"]["restored_fragments"] == len(restored)
        assert result.ladder["fallback_won"] is True
        assert any("текст распознавания сохранён" in note for note in result.degraded)
        assert drawing_route.lost(result) is True

    def test_kept_but_thinner_read_is_not_passed_off_as_whole(self):
        """
        Разбор уложился в допуск `LADDER_CONTENT_KEEP_RATIO`, но прочитал
        вдвое меньше распознавания. Текст распознавания не возвращается —
        иначе допуск не имел бы смысла, — зато недостача видна: без неё
        полулист уходил победителем и молча.
        """
        result = chosen_with_reserve(2985)
        lines = [f"XT{i} A{i} B{i} 0,75" for i in range(160)]
        fragments = [drawing_sheet(lines=lines)]
        settled, flags = drawing_route.settle(fragments, result, METADATA, TextNormalizer())
        route_outcome = result.ladder["drawing_route"]
        assert [f.type for f in settled] == ["drawing"] and flags == {}
        assert route_outcome["won"] is True
        assert route_outcome["thinner_kept"] is True
        assert route_outcome["volume"]["chars"] < route_outcome["reserve_chars"]
        assert any("короче распознавания" in note for note in result.degraded)
        assert drawing_route.thinner_kept(result) is True
        assert drawing_route.lost(result) is False

    def test_full_read_is_not_marked_thinner(self):
        """Разбор не беднее распознавания — ни пометки, ни деградации."""
        result = chosen_with_reserve(1692)
        lines = [f"XT{i} A{i} B{i} 0,75" for i in range(200)]
        fragments = [drawing_sheet(lines=lines, fields=["XT1", "XT2"])]
        drawing_route.settle(fragments, result, METADATA, TextNormalizer())
        # Ключи исхода пишутся всегда, в том числе ложными значениями: иначе
        # в запросе к БД «ключа нет» неотличимо от «сверки не было».
        assert result.ladder["drawing_route"]["thinner_kept"] is False
        assert result.degraded == []
        assert drawing_route.thinner_kept(result) is False

    def test_thin_read_does_not_replace_richer_recognition(self):
        """Три размера против 2985 символов распознавания: остаётся и то и другое."""
        result = chosen_with_reserve(2985)
        fragments = [drawing_sheet(fields=["Ø12", "45", "М8"])]
        settled, _ = drawing_route.settle(fragments, result, METADATA, TextNormalizer())
        assert settled[0].type == "drawing"
        assert any(f.type == "text" for f in settled)
        assert result.ladder["drawing_route"]["won"] is False
        assert drawing_route.lost(result) is True


class TestDocumentAfterSettle:
    """Задача целиком: что уезжает потребителю и в БД после сверки."""

    @pytest.fixture
    def run(self, monkeypatch, temp_storage, task_db, fake_redis):
        from core.providers.file_locator import LocatedFile
        from core.repositories import ChunkRepository, DocumentRepository
        from core.result_aggregator import get_request_state, init_request_state
        from ingest import tasks

        def run(ocr_chars, enrich):
            result = chosen_with_reserve(ocr_chars)
            result.ladder["classification"] = {"source": SOURCE_RASTER, "genre": GENRE_DRAWING}
            monkeypatch.setattr(
                tasks.StorageProviderFactory, "default", staticmethod(lambda: temp_storage))
            monkeypatch.setattr(
                tasks.FileLocator, "locate",
                lambda self, fileid: LocatedFile(fileid, f"documents/{fileid}.jpg", "jpg"))
            monkeypatch.setattr(
                tasks.DocumentParserFactory, "get_parser",
                staticmethod(lambda storage=None: StubParser(result)))
            monkeypatch.setattr(tasks.DrawingProcessor, "enrich_fragments", enrich)
            init_request_state("req-sheet", ["file-sheet"])
            tasks.process_document(request_id="req-sheet", s3_fileid="file-sheet")
            document = get_request_state("req-sheet")["documents"][0]
            session = task_db()
            try:
                doc_id = tasks.generate_doc_id("file-sheet")
                stored = DocumentRepository(session).get(doc_id)
                extra = dict(stored.extra_data or {})
                rows = [(row.chunk_type, dict(row.extra_data or {}))
                        for row in ChunkRepository(session).get_by_doc_id(doc_id)]
            finally:
                session.close()
            return document, extra, rows

        return run

    def test_read_sheet_is_delivered_without_recognition_text(self, run):
        lines = [f"XT{i} A{i} B{i} 0,75" for i in range(200)]

        def enrich(self, fragments, metadata):
            return [drawing_sheet(lines=lines, fields=["XT1"], fragment_id="sheet")]

        document, extra, rows = run(1692, enrich)
        assert [f["type"] for f in document["fragments"]] == ["drawing"]
        assert drawing_route.WARNING_THIN not in document["warnings"]
        assert extra["chosen_strategy"] == "layout"
        assert extra["fallback_won"] is False
        assert extra["degraded"] == []
        assert extra["drawing_route"]["won"] is True

    def test_unread_sheet_keeps_recognition_text_and_is_not_completed(self, run):
        def enrich(self, fragments, metadata):
            return [unread_picture()]

        document, extra, rows = run(1692, enrich)
        types = [f["type"] for f in document["fragments"]]
        assert "text" in types
        text = "".join(f["content"]["text"] or "" for f in document["fragments"])
        assert len("".join(text.split())) == 1692
        assert document["status"] == "partial"
        assert document["review_required"] is True
        assert drawing_route.WARNING_THIN in document["warnings"]
        assert extra["fallback_won"] is True
        assert extra["drawing_route"]["won"] is False
        # Служебные пометки возвращённого текста доехали до БД.
        assert all(flags["is_fallback"] and flags["needs_review"]
                   for kind, flags in rows if kind == "text")

    def test_kept_but_thinner_sheet_goes_to_review(self, run):
        """Недостача в допуске доезжает до документа предупреждением."""
        lines = [f"XT{i} A{i} B{i} 0,75" for i in range(160)]

        def enrich(self, fragments, metadata):
            return [drawing_sheet(lines=lines, fragment_id="sheet")]

        document, extra, rows = run(2985, enrich)
        assert [f["type"] for f in document["fragments"]] == ["drawing"]
        assert document["status"] == "partial"
        assert document["review_required"] is True
        assert drawing_route.WARNING_THIN_KEPT in document["warnings"]
        assert drawing_route.WARNING_THIN not in document["warnings"]
        assert extra["drawing_route"]["won"] is True
        assert extra["drawing_route"]["thinner_kept"] is True
