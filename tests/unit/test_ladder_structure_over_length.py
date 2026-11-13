"""
Спор чертёжной ветки и распознавания решается составом, а не объёмом
(LADDER_STRUCTURE_OVER_LENGTH).

Числа сняты с настоящих листов и документов прода (13.11.2026). Опорный
случай — СМ1638.11.05.00.00.00 СБ.jpg: растровый детектор назвал лист
страницей текста (рамка формата есть, но длинных линий 2 вместо трёх, а
полос строк 7), поэтому прежнее правило `LADDER_DRAWING_OVER_OCR` на него
не включалось. Дальше лестница сравнила уровни по числу символов, и
`plain_ocr` с 98 символами и оценкой 0.295 выиграл у `layout` с 0 символами
и оценкой 0.465: решал не балл, а мерка сохранности. Документ уехал как
`completed`, человек его не увидел.

Тесты здесь проверяют три вещи: лист с рамкой достаётся чертёжной ветке,
победитель определяется составом прочитанного, а документ без структуры
возвращается человеку. И отдельно — что обычный документ (письмо, протокол,
скан страницы текста) по-прежнему идёт плоским путём.
"""

import pytest

from core.config import settings
from core.ladder import drawing_route
from core.ladder.context import (
    GENRE_DRAWING,
    GENRE_TEXT,
    GENRE_UNKNOWN,
    SOURCE_RASTER,
    Classification,
    DocumentContext,
)
from core.ladder.router import LadderRouter
from core.ladder.structure import drawing_structure
from core.models.contract import Content, DrawingField, Fragment, Position, Provenance
from core.models.parse_result import ParsedBlock, ParseResult
from core.normalizer import TextNormalizer
from tests.unit.test_ladder_drawing_over_ocr import (
    JPG,
    PDF,
    LayoutLevel,
    OcrLevel,
    text_of,
    unread_picture,
)
from tests.unit.test_tasks import StubParser, task_db  # noqa: F401 — фикстура задачи

# Сигналы растрового детектора по настоящему листу СМ1638.11.05.00.00.00 СБ:
# рамка формата видна, длинных линий 2, полос строк 7 — род листа `text_scan`.
SM1638_SIGNALS = {
    "width": 4960, "height": 3507, "megapixels": 17.395, "ink": 0.0439,
    "long_rules": 2, "frame": True, "text_bands": 7, "thin_line_frame": True,
    "thin_line_rules": 2, "raster_kind": "text_scan",
}

# Сигналы по настоящему скану письма прода («Письмо О непредъявлении
# требований.pdf», страница 1, 200 dpi): полос строк 36, длинных линий нет,
# рамки формата нет. Такой лист правило не трогает.
LETTER_SIGNALS = {
    "width": 1654, "height": 2339, "ink": 0.076,
    "long_rules": 0, "frame": False, "text_bands": 36, "raster_kind": "text_scan",
}

# Настоящая страница с таблицами: «УПД №10541 от 16.09.2024.pdf», страница 2.
# Сетка таблицы даёт 10 длинных линий при 38 полосах строк, но рамки формата
# у неё нет — ни передаточный документ, ни страница паспорта чертежом не
# становятся.
TABLE_PAGE_SIGNALS = {
    "width": 1654, "height": 2339, "ink": 0.0636,
    "long_rules": 10, "frame": False, "text_bands": 38, "raster_kind": "text_scan",
}

# Лист СМ1638: символов у plain_ocr и у layout из журнала разбора.
SM1638_CHARS = (98, 0)

METADATA = {"doc_id": "doc", "source_path": "documents/sheet", "file_type": "jpg"}


@pytest.fixture
def flag_on(monkeypatch):
    monkeypatch.setattr(settings, "LADDER_DRAWING_OVER_OCR", True)
    monkeypatch.setattr(settings, "DRAWING_IMAGE_FULL_READ", True)
    monkeypatch.setattr(settings, "LADDER_STRUCTURE_OVER_LENGTH", True)
    # Значение прода: неизмеренная полнота чтения листа — повод проверить
    # фрагмент, но не потеря страницы. Иначе статус документа всегда
    # `partial`, и проверять исход сравнения по нему нельзя.
    monkeypatch.setattr(settings, "DRAWING_UNMEASURED_NOT_PARTIAL", True)


@pytest.fixture
def flag_off(monkeypatch):
    """Прежняя установка прода: старое правило включено, новое — нет."""
    monkeypatch.setattr(settings, "LADDER_DRAWING_OVER_OCR", True)
    monkeypatch.setattr(settings, "DRAWING_IMAGE_FULL_READ", True)
    monkeypatch.setattr(settings, "LADDER_STRUCTURE_OVER_LENGTH", False)
    monkeypatch.setattr(settings, "DRAWING_UNMEASURED_NOT_PARTIAL", True)


def sheet_context(genre=GENRE_TEXT, signals=None, data=JPG, file_type="jpg"):
    context = DocumentContext(uri="documents/sheet", file_type=file_type, data=data)
    context._classification = Classification(
        source=SOURCE_RASTER, genre=genre, raster_quality=0.77,
        signals=dict(SM1638_SIGNALS if signals is None else signals),
    )
    return context


def route(ocr_chars, layout_chars, **kwargs):
    router = LadderRouter([OcrLevel(ocr_chars), LayoutLevel(layout_chars)], threshold=0.75)
    return router.parse(sheet_context(**kwargs))


def sheet_fragment(text="", fields=(), payload=None, fragment_id="doc-sheet"):
    """Лист, прочитанный чертёжной веткой: поля и переписанный текст."""
    return Fragment(
        fragment_id=fragment_id, type="drawing",
        content=Content(
            text=text or None, image_ref="documents/sheet",
            structured_drawing_fields=[
                DrawingField(category=category, value=value, confidence=0.9)
                for category, value in fields
            ],
            structured_payload=dict(payload or {}),
        ),
        position=Position(page=1, bbox=[0, 0, 1, 1], order=0),
        confidence=0.9, completeness=None,
        provenance=Provenance(method="vision_plus_vlm", strategy_level=7, source="drawing"),
    )


# Что чертёжная ветка действительно прочитала с листа СМ1638 (28 полей,
# 41 строка переписанного текста). Текст сокращён до строк, которые в споре
# и участвуют: обозначение, надписи видов, размеры, графы штампа.
SM1638_SHEET_TEXT = "\n".join([
    "СМ1638.11.05.00.00.00 СБ", "Г-Г(1:1)", "2 места", "φ8,5", "2 отв.",
    "В-В(1:1)", "110", "65", "98", "φ10", "Инв. № подл.", "Подп. и дата",
    "Взам. инв. №", "Изм. /Лист", "№ докум.", "Лист", "Копировал", "Формат", "А3",
])
SM1638_FIELDS = (
    ("title_block", "СМ1638.11.05.00.00.00 СБ"), ("callout", "Г-Г(1:1)"),
    ("size", "φ8,5"), ("size", "φ10"), ("size", "110"), ("size", "65"),
)


def sm1638_sheet():
    return sheet_fragment(
        text=SM1638_SHEET_TEXT, fields=SM1638_FIELDS,
        payload={"transcript_line_count": 41, "title_block": {"Обозначение": "СМ1638.11.05.00.00.00 СБ"}},
    )


def chosen_with_reserve(ocr_chars=98, layout_chars=0, **kwargs):
    """Результат лестницы на листе с рамкой: layout выбран, OCR в запасе."""
    result = route(ocr_chars, layout_chars, **kwargs)
    assert result.fallback_reserve is not None, "запас не заведён — флаг выключен?"
    return result


class TestStructureSignals:
    """Признаки считаются по составу листа, а не по объёму текста."""

    def test_sheet_with_designation_and_title_block_is_structured(self):
        found = drawing_structure([sm1638_sheet()])
        assert found.designation is True
        assert found.title_block is True
        assert found.view_titles >= 2        # Г-Г и В-В
        assert found.dimensions >= 4         # φ8,5, φ10, 2 отв., 8,5
        assert found.structured is True
        assert found.complete is True

    def test_title_block_is_recognised_by_its_rows_alone(self):
        """Штамп виден по графам и тогда, когда модель не вернула его полем."""
        sheet = sheet_fragment(text="Инв. № подл.\nПодп. и дата\nВзам. инв. №\nФормат А3")
        assert drawing_structure([sheet]).title_block is True

    def test_flat_recognition_of_a_sheet_has_no_structure(self):
        """Строки Tesseract по чертежу: символы есть, состава нет."""
        garble = "rt №\n23 2 места\n7 7 2 3 1\na\nAL 10 SF Хх\nт 2 места > yas Г \\ Fr"
        sheet = sheet_fragment(text=garble)
        found = drawing_structure([sheet])
        assert found.fields == 0
        assert found.title_block is False
        assert found.designation is False
        assert found.structured is False

    def test_nothing_read_is_not_structured(self):
        assert drawing_structure([unread_picture()]).structured is False

    def test_a_few_dimensions_without_a_stamp_are_structured_but_not_complete(self):
        """Лист почти без текста: размеры есть, лист не опознан — на проверку."""
        sheet = sheet_fragment(fields=(("size", "⌀12"), ("size", "45")))
        found = drawing_structure([sheet])
        assert found.structured is True
        assert found.complete is False

    def test_a_date_is_not_a_designation(self):
        """Дата и номер пункта под обозначение по ЕСКД не подходят."""
        sheet = sheet_fragment(text="13.11.2026\nпункт 1.2.3\nверсия 2.10.4")
        assert drawing_structure([sheet]).designation is False

    @pytest.mark.parametrize("code", [
        "СМ1638.11.05.00.00.00 СБ", "405529.001.01", "301421.007.00",
        "061360.001.00", "302318.207.00", "ТАДУ 421451.003.05.004",
    ])
    def test_real_designations_are_recognised(self, code):
        assert drawing_structure([sheet_fragment(text=code)]).designation is True


class TestRouteOnASheetTheDetectorCalledText:
    """СМ1638: лист с рамкой формата, названный страницей текста."""

    def test_flag_off_keeps_todays_loss(self, flag_off):
        """Прежнее поведение воспроизводится полностью: выигрывает plain_ocr."""
        result = route(*SM1638_CHARS)
        assert result.ladder["chosen_strategy"] == "plain_ocr"
        assert result.ladder["fallback_won"] is True
        assert result.fallback_reserve is None
        assert drawing_route.unstructured_fallback(result) is False

    def test_flag_off_scores_are_the_ones_from_the_log(self, flag_off):
        """Решал не балл: у проигравшего уровня он был выше."""
        attempts = {a["strategy"]: a for a in route(*SM1638_CHARS).ladder["attempts"]}
        assert attempts["plain_ocr"]["score"] < attempts["layout"]["score"]
        assert attempts["plain_ocr"]["chars"] > attempts["layout"]["chars"]

    def test_sheet_with_a_frame_goes_to_the_drawing_route(self, flag_on):
        result = route(*SM1638_CHARS)
        assert result.ladder["chosen_strategy"] == "layout"
        assert result.ladder["fallback_won"] is False
        assert result.is_fallback is False
        assert result.degraded == []
        # Распознавание не выброшено: оно ждёт сверки после чтения листа.
        assert result.fallback_reserve is not None
        assert result.ladder["drawing_deferred"]["reserve_chars"] == 98

    def test_drawing_genre_still_goes_to_the_drawing_route(self, flag_on):
        result = route(*SM1638_CHARS, genre=GENRE_DRAWING)
        assert result.ladder["chosen_strategy"] == "layout"

    def test_sheet_with_almost_no_text_goes_to_the_drawing_route(self, flag_on):
        """
        Настоящий лист прода, у которого текста почти нет: распознавание
        дало 66 символов, рамки детектор не нашёл, но род листа — чертёж.
        Чертёжная ветка прочитала с него 11 полей и штамп.
        """
        signals = dict(SM1638_SIGNALS, frame=False, thin_line_frame=False)
        result = route(66, 0, genre=GENRE_DRAWING, signals=signals)
        assert result.ladder["chosen_strategy"] == "layout"
        settled, flags = drawing_route.settle(
            [sheet_fragment(
                text="Инв. № подл.\nПодп. и дата\nФормат А4",
                fields=tuple(("size", v) for v in ("⌀12", "45", "М8", "R3")),
                payload={"transcript_line_count": 15},
            )],
            result, METADATA, TextNormalizer())
        assert [f.type for f in settled] == ["drawing"] and flags == {}
        assert result.ladder["drawing_route"]["won"] is True

    def test_structure_beats_length_by_any_margin(self, flag_on):
        """Самый тяжёлый случай прода: 3228 символов распознавания против нуля."""
        result = route(3228, 0)
        assert result.ladder["chosen_strategy"] == "layout"
        assert result.fallback_reserve is not None


class TestOrdinaryDocumentsKeepThePlainRoute:
    """
    Жёсткое требование: документ, который и есть сплошной текст, обязан
    остаться на плоском пути. Переусердствовать здесь дороже, чем недобрать:
    письмо, протокол, страница паспорта — это не чертежи.
    """

    def test_scanned_letter_without_a_frame_keeps_plain_ocr(self, flag_on):
        """Скан письма: 36 полос строк, рамки нет — правило не включается."""
        result = route(1240, 0, signals=LETTER_SIGNALS)
        assert result.ladder["chosen_strategy"] == "plain_ocr"
        assert result.fallback_reserve is None

    def test_letter_text_is_not_promoted_even_as_unknown_genre(self, flag_on):
        result = route(1240, 0, genre=GENRE_UNKNOWN, signals=LETTER_SIGNALS)
        assert result.ladder["chosen_strategy"] == "plain_ocr"

    def test_page_the_layout_level_read_itself_is_not_touched(self, flag_on):
        """Уровень 5 сам прочитал страницу — спорить не о чем, запас не нужен."""
        result = route(1240, 1180, signals=LETTER_SIGNALS)
        assert result.ladder["chosen_strategy"] == "layout"
        assert result.fallback_reserve is None

    def test_pdf_scan_keeps_the_character_check(self, flag_on):
        """У скана PDF страниц много, среди них проза: счёт символов честен."""
        result = route(1692, 0, data=PDF, file_type="pdf", genre=GENRE_DRAWING)
        assert result.ladder["chosen_strategy"] == "plain_ocr"
        assert result.fallback_reserve is None

    def test_mixed_pdf_of_drawings_and_prose_is_not_rerouted(self, flag_on):
        """
        «305179.001.01.01.011.pdf» прода: 720 символов распознавания против
        8 у разбора с детекцией областей. Маршрут остаётся прежним — в таком
        файле есть и чертежи, и проза, и символы на нём меряют сохранность
        честно. Меняется только статус: потеря структуры перестаёт молчать.
        """
        result = route(720, 8, data=PDF, file_type="pdf", genre=GENRE_DRAWING)
        assert result.ladder["chosen_strategy"] == "plain_ocr"
        assert drawing_route.unstructured_fallback(result) is True

    def test_table_grid_and_stamps_do_not_make_a_drawing(self, flag_on):
        """
        Сетка таблицы и печати дают детектору длинные линии, но рамки формата
        у передаточного документа нет, и лист остаётся страницей текста. То же
        у страницы паспорта и у протокола с таблицами.
        """
        result = route(2104, 0, signals=TABLE_PAGE_SIGNALS)
        assert result.ladder["chosen_strategy"] == "plain_ocr"
        assert result.fallback_reserve is None

    def test_real_scanned_contract_keeps_the_plain_route(self, flag_on):
        """«Договор поставки №16940/337.pdf», страница 1: 52 полосы, 6 линеек."""
        signals = dict(LETTER_SIGNALS, long_rules=6, text_bands=52, ink=0.1088)
        assert route(1900, 0, signals=signals).ladder["chosen_strategy"] == "plain_ocr"


@pytest.mark.usefixtures("flag_on")
class TestSettleDecidedByStructure:
    def test_read_sheet_wins_with_fewer_characters(self):
        """СМ1638: 98 символов распознавания против состава чертёжного листа."""
        result = chosen_with_reserve(98)
        settled, flags = drawing_route.settle(
            [sm1638_sheet()], result, METADATA, TextNormalizer())
        outcome = result.ladder["drawing_route"]
        assert [f.type for f in settled] == ["drawing"] and flags == {}
        assert outcome["won"] is True
        assert outcome["structure"]["designation"] is True
        assert outcome["structure"]["title_block"] is True
        assert result.ladder["fallback_won"] is False
        assert drawing_route.lost(result) is False
        assert drawing_route.thin_structure(result) is False

    def test_wins_even_when_recognition_brought_far_more_characters(self):
        """
        Состав важнее объёма: 3228 символов распознавания правоты не дают,
        маршрут остаётся чертёжным. Но недостача объёма видна — лист,
        прочитанный короче распознавания, за полный не выдаётся.
        """
        result = chosen_with_reserve(3228)
        settled, flags = drawing_route.settle(
            [sm1638_sheet()], result, METADATA, TextNormalizer())
        assert [f.type for f in settled] == ["drawing"] and flags == {}
        assert result.ladder["drawing_route"]["won"] is True
        assert drawing_route.thinner_kept(result) is True

    def test_sheet_without_structure_returns_the_recognition_text(self):
        """Ветка листа не прочитала: текст распознавания возвращается."""
        result = chosen_with_reserve(1692)
        settled, flags = drawing_route.settle(
            [unread_picture()], result, METADATA, TextNormalizer())
        restored = [f for f in settled if f.type == "text"]
        assert sum(len("".join((f.content.text or "").split())) for f in restored) == 1692
        assert all(f.review_required for f in restored)
        assert result.ladder["drawing_route"]["won"] is False
        assert result.ladder["fallback_won"] is True
        assert any("без структуры" in note for note in result.degraded)
        assert drawing_route.lost(result) is True

    def test_garbled_sheet_text_without_structure_loses(self):
        """Переписанный текст есть, состава нет: за чертёжный разбор не сходит."""
        result = chosen_with_reserve(1692)
        garble = sheet_fragment(
            text="AL 10 SF Хх\nrt №\nт yas Г \\ Fr",
            payload={"transcript_line_count": 3},
        )
        drawing_route.settle([garble], result, METADATA, TextNormalizer())
        assert result.ladder["drawing_route"]["won"] is False

    def test_structured_but_unidentified_sheet_is_not_passed_off_as_whole(self):
        """Размеры есть, штампа и обозначения нет: разбор остаётся, лист — на проверку."""
        result = chosen_with_reserve(1692)
        sheet = sheet_fragment(fields=(("size", "⌀12"), ("size", "45")))
        settled, flags = drawing_route.settle(
            [sheet], result, METADATA, TextNormalizer())
        assert [f.type for f in settled] == ["drawing"] and flags == {}
        assert result.ladder["drawing_route"]["won"] is True
        assert result.ladder["drawing_route"]["thin_structure"] is True
        assert drawing_route.thin_structure(result) is True
        assert any("без штампа и обозначения" in note for note in result.degraded)


class TestDocumentIsReturnedToAHuman:
    """Молчаливый `completed` при победе плоского текста больше не выдаётся."""

    @pytest.fixture
    def run(self, monkeypatch, temp_storage, task_db, fake_redis):
        from core.providers.file_locator import LocatedFile
        from core.repositories import DocumentRepository
        from core.result_aggregator import get_request_state, init_request_state
        from ingest import tasks

        def run(parse_result, enrich):
            parse_result.ladder["classification"] = {
                "source": SOURCE_RASTER, "genre": GENRE_TEXT,
                "signals": dict(SM1638_SIGNALS),
            }
            # Лист один и он разобран: иначе статус `partial` ставила бы
            # неизвестная граница документа, а не исход сравнения.
            parse_result.source_page_count = 1
            monkeypatch.setattr(
                tasks.StorageProviderFactory, "default", staticmethod(lambda: temp_storage))
            monkeypatch.setattr(
                tasks.FileLocator, "locate",
                lambda self, fileid: LocatedFile(fileid, f"documents/{fileid}.jpg", "jpg"))
            monkeypatch.setattr(
                tasks.DocumentParserFactory, "get_parser",
                staticmethod(lambda storage=None: StubParser(parse_result)))
            monkeypatch.setattr(tasks.DrawingProcessor, "enrich_fragments", enrich)
            init_request_state("req-sheet", ["file-sheet"])
            tasks.process_document(request_id="req-sheet", s3_fileid="file-sheet")
            document = get_request_state("req-sheet")["documents"][0]
            session = task_db()
            try:
                stored = DocumentRepository(session).get(tasks.generate_doc_id("file-sheet"))
                extra = dict(stored.extra_data or {})
            finally:
                session.close()
            return document, extra

        return run

    def test_sheet_read_by_the_drawing_route_is_delivered_whole(self, flag_on, run):
        document, extra = run(
            chosen_with_reserve(98),
            lambda self, fragments, metadata: [sm1638_sheet()],
        )
        assert [f["type"] for f in document["fragments"]] == ["drawing"]
        assert extra["chosen_strategy"] == "layout"
        assert extra["drawing_route"]["won"] is True
        assert document["status"] == "completed"
        assert drawing_route.WARNING_FALLBACK_WON not in document["warnings"]

    def test_fallback_victory_is_no_longer_silent(self, flag_on, run):
        """
        Лист без рамки, разобранный плоским текстом при живом уровне 5:
        маршрут остаётся прежним, но документ уходит человеку, а не в индекс
        как полный.
        """
        parse_result = route(1240, 0, signals=LETTER_SIGNALS)
        assert parse_result.ladder["fallback_won"] is True
        document, extra = run(
            parse_result, lambda self, fragments, metadata: fragments)
        assert extra["chosen_strategy"] == "plain_ocr"
        assert document["status"] == "partial"
        assert document["review_required"] is True
        assert drawing_route.WARNING_FALLBACK_WON in document["warnings"]

    def test_flag_off_publishes_the_same_document_as_today(self, flag_off, run):
        """Поведение при выключенном флаге не меняется — включая молчание."""
        parse_result = route(1240, 0, signals=LETTER_SIGNALS)
        document, extra = run(
            parse_result, lambda self, fragments, metadata: fragments)
        assert extra["chosen_strategy"] == "plain_ocr"
        assert extra["fallback_won"] is True
        assert document["status"] == "completed"
        assert document["review_required"] is False
        assert drawing_route.WARNING_FALLBACK_WON not in document["warnings"]

    def test_degraded_document_carries_a_warning_rag_cannot_treat_as_advisory(
        self, flag_on, run):
        """
        Статуса `partial` для потребителя мало. RAG считает `partial` полным
        текстом, если все страницы разобраны и среди предупреждений нет ни
        одного, кроме совещательных — `DRAWING_COMPLETENESS_UNKNOWN`,
        `DRAWING_NOT_RECOGNIZED` и `parser_contract_dropped_invalid_extracted_facts:`
        (`super_rag/parser/completeness.py`), — и тогда ставит событию
        `indexed` без причины. Поэтому деградировавший документ обязан
        уезжать с предупреждением вне этого списка, иначе он снова
        проваливается в тишину.
        """
        advisory = {"DRAWING_COMPLETENESS_UNKNOWN", "DRAWING_NOT_RECOGNIZED"}
        document, _ = run(
            route(1240, 0, signals=LETTER_SIGNALS),
            lambda self, fragments, metadata: fragments,
        )
        assert document["status"] == "partial"
        assert [w for w in document["warnings"]
                if w not in advisory
                and not w.startswith("parser_contract_dropped_invalid_extracted_facts:")]

    def test_unread_sheet_goes_back_for_review_with_the_recognition_text(self, flag_on, run):
        document, extra = run(
            chosen_with_reserve(1692),
            lambda self, fragments, metadata: [unread_picture()],
        )
        text = "".join(f["content"]["text"] or "" for f in document["fragments"])
        assert len("".join(text.split())) == 1692
        assert document["status"] == "partial"
        assert document["review_required"] is True
        assert drawing_route.WARNING_THIN in document["warnings"]


class TestNothingChangesWithoutTheFlag:
    """Выключенный флаг обязан оставить разбор ровно таким, как сегодня."""

    def test_settle_still_decides_by_volume(self, flag_off):
        """Три размера против 2985 символов распознавания — прежний исход."""
        result = chosen_with_reserve(2985, genre=GENRE_DRAWING)
        sheet = sheet_fragment(
            text="СМ1638.11.05.00.00.00 СБ",
            fields=(("size", "⌀12"), ("size", "45"), ("size", "М8")),
        )
        drawing_route.settle([sheet], result, METADATA, TextNormalizer())
        assert result.ladder["drawing_route"]["won"] is False
        assert drawing_route.lost(result) is True

    def test_deferred_needs_the_flag_for_a_text_scan_with_a_frame(self, flag_off):
        assert drawing_route.deferred(sheet_context()) is False

    def test_deferred_does_not_need_the_flag_for_a_drawing_genre(self, flag_off):
        assert drawing_route.deferred(sheet_context(genre=GENRE_DRAWING)) is True

    def test_empty_result_is_not_mistaken_for_a_route_outcome(self, flag_off):
        result = ParseResult(blocks=[ParsedBlock(type="text", text=text_of(50), page=1)])
        assert drawing_route.lost(result) is False
        assert drawing_route.thin_structure(result) is False
        assert drawing_route.unstructured_fallback(result) is False
