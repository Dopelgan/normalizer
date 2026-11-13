"""
Устаревшие записи листа, который модель прочитала целиком.

Рядом с верным текстом модели в документе оставались записи распознавания того
же листа: таблица со штампом, где масштаб «7:1» вместо «1:1», таблица сальника
с «M24×15» вместо «M24x1,5», абзацы из одних обрывков. Правка абзацев
переписывала только текст абзаца, таблиц не трогала и ничего не убирала.

DRAWING_SUPPRESS_STALE_ON_FULL_READ убирает повтор и шум, при расхождении в
цифрах оставляет оба чтения, а запись с содержимым, которого у модели нет,
оставляет на месте всегда.

Строки взяты с разобранных листов; сами листы в тесты не кладутся.
"""

import pytest

from core.config import settings
from core.drawing_stale_sources import (
    PAYLOAD_KEY,
    REASON_DUPLICATE,
    REASON_NOISE,
    VARIANTS_KEY,
    WARNING,
    suppress_stale_sources,
)
from core.drawing_transcript import VARIANT_WARNING
from core.models.contract import (
    Content,
    DrawingField,
    Fragment,
    Position,
    Provenance,
    TableData,
)

pytest.importorskip("PIL")

# Переписанный моделью текст листа: штамп, резьба сальника, требования.
SHEET_LINES = [
    "Сальник ТАДУ 421451.001",
    "Резьба M24x1,5 под кабельный ввод",
    "Шероховатость неуказанных поверхностей Ra 6,3",
    "Лит. Масса Масштаб",
    "0,42 1:1",
    "Разраб. Бажко 26.08.22",
    "ТАДУ 421451.001 СБ",
]


def sheet(lines=SHEET_LINES, page=1, complete=True, fields=(("scale", "1:1"),), tables=True):
    payload = {"transcript_line_count": len(lines)}
    if not complete:
        payload["transcript_complete"] = False
    if tables:
        payload["tables"] = [{"kind": "title_block", "bbox": [0.6, 0.8, 1.0, 1.0],
                              "rows": [["Масса", "Масштаб"], ["0,42", "1:1"]]}]
    return Fragment(
        fragment_id=f"doc-1-frag-000-{page}", type="drawing",
        content=Content(
            text="\n".join(lines),
            structured_drawing_fields=[
                DrawingField(category=category, value=value, confidence=0.85)
                for category, value in fields
            ],
            structured_payload=payload,
        ),
        position=Position(page=page, bbox=[0, 0, 1, 1], order=0),
        confidence=0.75, completeness=None,
        provenance=Provenance(method="vlm_full_page", strategy_level=7, source="drawing"),
    )


def paragraph(text, page=1, method="ocr_layer", order=5):
    return Fragment(
        fragment_id=f"doc-1-p{page}-{order}", type="text", content=Content(text=text),
        position=Position(page=page, bbox=[0.1, 0.5, 0.9, 0.6], order=order),
        confidence=0.6, completeness=1.0,
        provenance=Provenance(method=method, strategy_level=5, source="scanned_pdf"),
    )


def table(rows, headers=(), page=1, method="mineru_table", order=6):
    return Fragment(
        fragment_id=f"doc-1-t{page}-{order}", type="table",
        content=Content(table_data=TableData(headers=list(headers), rows=[list(r) for r in rows])),
        position=Position(page=page, bbox=[0.6, 0.8, 1.0, 1.0], order=order),
        confidence=0.7, completeness=1.0,
        provenance=Provenance(method=method, strategy_level=5, source="scanned_pdf"),
    )


# Абзац распознавания из одних обрывков: Tesseract по чертёжному шрифту.
NOISE = "< - o a ni aera = pian HM 2XIBHIOT [OCT"


def tokens_of(fragments):
    """Лексемы всех фрагментов (написание для сравнения -> как в документе)."""
    from core.drawing_stale_sources import _block_text, _key, _tokens
    found = {}
    for fragment in fragments:
        for token in _tokens(_block_text(fragment)):
            found.setdefault(_key(token), token)
    return found


class TestDigitsDiffer:
    def test_wrong_scale_in_the_stamp_is_kept_and_marked(self):
        # «7:1» против «1:1» модели: кто прав — неизвестно, остаются оба чтения.
        stale = table([["Масса", "Масштаб"], ["0,42", "7:1"]])
        out = suppress_stale_sources([sheet(), stale])
        assert len(out) == 2 and out[1] is stale
        variants = stale.content.structured_payload[VARIANTS_KEY]
        assert {"block": "7:1", "model": "1:1"} in variants
        assert VARIANT_WARNING in stale.warnings and stale.review_required

    def test_wrong_thread_pitch_is_kept_and_marked(self):
        stale = table([["Сальник", "M24×15"]])
        out = suppress_stale_sources([sheet(), stale])
        assert len(out) == 2
        variants = stale.content.structured_payload[VARIANTS_KEY]
        assert variants == [{"block": "M24×15", "model": "M24x1,5"}]
        assert VARIANT_WARNING in stale.warnings

    def test_decimal_comma_lost_by_recognition_is_marked(self):
        stale = paragraph("Шероховатость неуказанных поверхностей Ra 63")
        out = suppress_stale_sources([sheet(), stale])
        assert len(out) == 2
        assert stale.content.structured_payload[VARIANTS_KEY] == [{"block": "63", "model": "6,3"}]

    def test_the_sheet_fragment_tells_what_was_marked(self):
        stale = table([["Масса", "Масштаб"], ["0,42", "7:1"]])
        model = sheet()
        suppress_stale_sources([model, stale])
        marked = model.content.structured_payload[PAYLOAD_KEY]["marked"]
        assert marked[0]["fragment_id"] == stale.fragment_id
        assert marked[0]["reason"] == "digits_differ"
        assert WARNING in model.warnings


class TestDropped:
    def test_noise_paragraph_is_dropped(self):
        out = suppress_stale_sources([sheet(), paragraph(NOISE)])
        assert len(out) == 1 and out[0].type == "drawing"
        dropped = out[0].content.structured_payload[PAYLOAD_KEY]["dropped"]
        assert dropped[0]["reason"] == REASON_NOISE

    def test_repeated_paragraph_is_dropped(self):
        out = suppress_stale_sources(
            [sheet(), paragraph("Шероховатость неуказанных поверхностей Ra 6,3")])
        assert len(out) == 1
        assert out[0].content.structured_payload[PAYLOAD_KEY]["dropped"][0]["reason"] == REASON_DUPLICATE

    def test_repeated_table_is_dropped_only_with_a_table_of_the_model(self):
        rows = [["Масса", "Масштаб"], ["0,42", "1:1"]]
        assert len(suppress_stale_sources([sheet(), table(rows)])) == 1
        # Своей таблицы у модели нет: строение таблицы строками не передаётся.
        assert len(suppress_stale_sources([sheet(tables=False), table(rows)])) == 2

    def test_noise_stays_on_a_sheet_read_in_parts(self):
        # Лист дочитан не целиком: шум мог стоять в непрочитанной части.
        out = suppress_stale_sources([sheet(complete=False), paragraph(NOISE)])
        assert len(out) == 2

    def test_noise_stays_on_an_english_sheet(self):
        english = ["Gland thread M24x1,5", "Roughness Ra 6,3 all over", "Sheet 1 of 2"]
        out = suppress_stale_sources([sheet(lines=english), paragraph(NOISE)])
        assert len(out) == 2

    def test_short_block_is_not_dropped_as_noise(self):
        out = suppress_stale_sources([sheet(), paragraph("[OCT 2246")])
        assert len(out) == 2


class TestKept:
    def test_table_with_content_absent_from_the_model_is_kept(self):
        # Ведомость с позициями, которых в переписанном тексте нет.
        stale = table(
            [["1", "Прокладка ТАДУ 754152.003", "2"],
             ["2", "Шайба пружинная 12 ГОСТ 6402-70", "4"]],
            headers=("Поз.", "Наименование", "Кол."),
        )
        out = suppress_stale_sources([sheet(), stale])
        assert len(out) == 2 and out[1] is stale
        assert not stale.warnings and not (stale.content.structured_payload or {})

    def test_paragraph_with_its_own_requirement_is_kept(self):
        text = "7. Сварные швы зачистить, покрытие Хим. Окс. прм по ГОСТ 9.306-85"
        out = suppress_stale_sources([sheet(), paragraph(text)])
        assert len(out) == 2 and out[1].content.text == text

    def test_other_page_is_not_touched(self):
        out = suppress_stale_sources([sheet(page=1), paragraph(NOISE, page=2)])
        assert len(out) == 2

    def test_page_without_a_whole_sheet_read_is_not_touched(self):
        bare = sheet()
        bare.content.structured_payload = {"tables": []}
        bare.content.text = None
        out = suppress_stale_sources([bare, paragraph(NOISE)])
        assert len(out) == 2

    def test_text_layer_and_model_tables_are_not_stale(self):
        vector = paragraph("Шероховатость неуказанных поверхностей Ra 6,3", method="text_layer")
        model_table = table([["Масса", "Масштаб"], ["0,42", "1:1"]], method="vlm_escalation")
        out = suppress_stale_sources([sheet(), vector, model_table])
        assert len(out) == 3

    def test_nothing_unique_is_lost(self):
        stale = [paragraph(NOISE, order=5),
                 paragraph("Шероховатость неуказанных поверхностей Ra 6,3", order=6),
                 table([["Масса", "Масштаб"], ["0,42", "7:1"]], order=7),
                 table([["1", "Прокладка ТАДУ 754152.003", "2"]], order=8)]
        before = tokens_of([sheet(), *stale])
        after = tokens_of(suppress_stale_sources([sheet(), *stale]))
        # Из набора лексем ушли только обрывки распознавания: ни слова,
        # ни значения не потеряно.
        from core.drawing_stale_sources import _kind
        lost = [before[key] for key in set(before) - set(after)]
        assert lost and all(_kind(token) == "garbage" for token in lost)
        assert {"таду", "754152.003", "6.3", "1:1", "7:1", "прокладка"} <= set(after)


class TestFlag:
    def test_flag_off_keeps_everything(self, temp_storage, monkeypatch):
        from core.drawing_processor import DrawingProcessor

        class FakeQwen:
            available = True
            model = "fake-vl"

        monkeypatch.setattr(settings, "DRAWING_SUPPRESS_STALE_ON_FULL_READ", False)
        processor = DrawingProcessor(temp_storage, qwen=FakeQwen())
        monkeypatch.setattr(processor, "_sheet_drawings", lambda fragments, metadata: [])
        fragments = [sheet(), paragraph(NOISE)]
        assert len(processor.enrich_fragments(fragments, {"doc_id": "doc-1"})) == 2

    def test_flag_on_suppresses_in_the_pipeline(self, temp_storage, monkeypatch):
        from core.drawing_processor import DrawingProcessor

        class FakeQwen:
            available = True
            model = "fake-vl"

        monkeypatch.setattr(settings, "DRAWING_SUPPRESS_STALE_ON_FULL_READ", True)
        processor = DrawingProcessor(temp_storage, qwen=FakeQwen())
        monkeypatch.setattr(processor, "_sheet_drawings", lambda fragments, metadata: [])
        fragments = [sheet(), paragraph(NOISE)]
        assert len(processor.enrich_fragments(fragments, {"doc_id": "doc-1"})) == 1
