"""
Пустая рамка таблицы дочитывается по рендеру страницы.

На проде 1180 таких рамок в 376 документах: ни содержимого у рамки, ни
строк текстового слоя внутри, ни вырезки — то есть всё, что в рамке было,
пропадало молча. На замере 24 настоящих рамок 18 читаются: штампы «СТО
04.18 03 Лист 56 из 56», шапки «№ Наименование Дата Уведомление План
качества Договор», надписи «АО «Проектный институт» МПИ АЭП АЭС с ВВЭР B05».

Строки и уверенности в тестах затвора — из того же замера, не выдуманные.
"""

import pytest

from core.config import settings
from core.providers import blank_table_ocr
from core.providers.text_layer import (
    OCR_LAYER,
    TextLayer,
    TextLine,
    reconcile_with_text_layer,
)

# Три реальных случая замера: повёрнутый мусор, короткий штамп и шапка.
NOISE = "виодлнох BELTON -oLodu eet и (dew -он) эинэвено9о №541-2 УП"
NOISE_CONFIDENCE = 62.6
STAMP = "СТО 04.18 03 Лист 56 из 56"
STAMP_CONFIDENCE = 95.1
HEADER = "№ Наименование pers Уведомление План качества Договор | изго"
HEADER_CONFIDENCE = 76.9


@pytest.fixture(autouse=True)
def blank_ocr_on(monkeypatch):
    """Ветка выключена по умолчанию — в тестах включаем явно."""
    monkeypatch.setattr(settings, "BLANK_TABLE_OCR_ENABLED", True)
    monkeypatch.setattr(settings, "BLANK_TABLE_OCR_MAX_REGIONS", 8)


def frame(page=1, bbox=None):
    """Пустая рамка таблицы: ни `table_data`, ни HTML, ни вырезки."""
    from core.models.parse_result import ParsedBlock

    return ParsedBlock(
        type="table", page=page,
        bbox=list(bbox or [0.08, 0.60, 0.92, 0.90]),
        page_size=[600.0, 800.0], confidence=0.8,
    )


def empty_layer(page_count=1):
    """Слой, в котором на странице рамки нет ни одной строки."""
    return TextLayer(pages={}, page_count=page_count)


def reader_of(text, confidence=0.9):
    """Поддельный читатель: отдаёт содержимое по новому контракту."""
    calls = []

    def read(block):
        calls.append(block.page)
        return {"text": text, "confidence": confidence}

    return read, calls


class TestAccepted:
    def test_text_lands_in_exactly_one_block(self):
        read, calls = reader_of(STAMP)

        blocks, stats = reconcile_with_text_layer(
            [frame()], empty_layer(), source=OCR_LAYER, table_reader=read
        )

        assert calls == [1]
        carriers = [b for b in blocks if STAMP in (b.text or "")]
        assert len(carriers) == 1
        assert carriers[0].type == "text"
        assert carriers[0].method == "tesseract"
        assert carriers[0].confidence == pytest.approx(0.9)
        # Пустая рамка не уезжает в индекс фрагментом без текста.
        assert not any(b.type == "table" for b in blocks)
        assert stats.blank_tables_read == 1
        assert stats.blank_tables_rejected == 0
        assert stats.blank_table_unread_pages == set()

    def test_frame_on_page_with_lines_elsewhere(self):
        # Вторая точка входа: строки слоя на странице есть, но не в рамке.
        read, calls = reader_of(HEADER)
        line = TextLine(text="Приложение № 15", bbox=[0.1, 0.04, 0.9, 0.09],
                        page=1, confidence=0.9)
        layer = TextLayer(pages={1: [line]}, page_count=1)

        blocks, stats = reconcile_with_text_layer(
            [frame()], layer, source=OCR_LAYER, table_reader=read
        )

        assert calls == [1]
        assert stats.blank_tables_read == 1
        carriers = [b for b in blocks if HEADER in (b.text or "")]
        assert len(carriers) == 1
        # Строка вне рамки по-прежнему возвращается своим блоком.
        assert any("Приложение" in (b.text or "") for b in blocks)

    def test_frame_with_crop_is_left_alone(self):
        # У рамки есть вырезка — содержимое уезжает своей дорогой.
        read, calls = reader_of(STAMP)
        block = frame()
        block.image_ref = "images/fig1.jpg"

        blocks, stats = reconcile_with_text_layer(
            [block], empty_layer(), source=OCR_LAYER, table_reader=read
        )

        assert calls == []
        assert stats.blank_tables_read == 0 and stats.blank_tables_rejected == 0
        assert [b.type for b in blocks] == ["table"]


class TestRejected:
    def test_nothing_indexed_and_page_stays_unread(self):
        calls = []

        def read(block):
            calls.append(block.page)
            return None   # затвор уже отклонил прочитанное

        blocks, stats = reconcile_with_text_layer(
            [frame(page=4)], empty_layer(page_count=4), source=OCR_LAYER, table_reader=read
        )

        assert calls == [4]
        assert not any((b.text or "").strip() for b in blocks)
        assert stats.blank_tables_read == 0
        assert stats.blank_tables_rejected == 1
        assert stats.blank_table_unread_pages == {4}
        assert stats.as_dict()["blank_table_unread_pages"] == [4]


class TestRegionCap:
    def test_only_capped_number_of_frames_is_read(self, monkeypatch):
        monkeypatch.setattr(settings, "BLANK_TABLE_OCR_MAX_REGIONS", 2)
        pages = [1, 2, 3]
        frames = [frame(page=page) for page in pages]
        # Настоящий читатель документа: предел рамок живёт в нём.
        monkeypatch.setattr(blank_table_ocr.table_vlm, "_render_page",
                            lambda *a, **k: _Image(200, 60))
        monkeypatch.setattr(blank_table_ocr.table_vlm, "_crop", lambda image, bbox: image)
        monkeypatch.setattr(blank_table_ocr.table_vlm, "_upright", lambda image: image)
        monkeypatch.setattr(blank_table_ocr, "read_region",
                            lambda image: (STAMP, STAMP_CONFIDENCE))
        read = blank_table_ocr.table_reader(b"%PDF-1.4", "pdf")

        blocks, stats = reconcile_with_text_layer(
            frames, empty_layer(page_count=3), source=OCR_LAYER, table_reader=read
        )

        assert stats.blank_tables_read == 2
        assert stats.blank_tables_rejected == 1
        # Рамка, до которой предел не дошёл, — непрочитанная страница, а не
        # тихая потеря: предел экономит время, а не содержимое.
        assert stats.blank_table_unread_pages == {3}
        assert sum(1 for b in blocks if STAMP in (b.text or "")) == 2

    def test_oversized_region_is_not_read(self, monkeypatch):
        monkeypatch.setattr(settings, "BLANK_TABLE_OCR_MAX_PIXELS", 1000)
        monkeypatch.setattr(blank_table_ocr.table_vlm, "_render_page",
                            lambda *a, **k: _Image(2000, 3000))
        monkeypatch.setattr(blank_table_ocr.table_vlm, "_crop", lambda image, bbox: image)
        monkeypatch.setattr(blank_table_ocr.table_vlm, "_upright", lambda image: image)
        monkeypatch.setattr(blank_table_ocr, "read_region",
                            lambda image: pytest.fail("вырезка выше предела не читается"))
        read = blank_table_ocr.table_reader(b"%PDF-1.4", "pdf")

        assert read(frame()) is None


class _Image:
    """Вырезка: от неё читателю нужен только размер."""

    def __init__(self, width, height):
        self.width, self.height = width, height


class TestDisabled:
    def test_flag_off_changes_nothing(self, monkeypatch):
        monkeypatch.setattr(settings, "BLANK_TABLE_OCR_ENABLED", False)
        read, calls = reader_of(STAMP)

        blocks, stats = reconcile_with_text_layer(
            [frame()], empty_layer(), source=OCR_LAYER, table_reader=read
        )
        plain, plain_stats = reconcile_with_text_layer(
            [frame()], empty_layer(), source=OCR_LAYER
        )

        assert calls == []
        assert [(b.type, b.text, b.bbox) for b in blocks] == [
            (b.type, b.text, b.bbox) for b in plain
        ]
        assert stats.as_dict() == plain_stats.as_dict()
        assert stats.blank_tables_read == 0 and stats.blank_tables_rejected == 0
        assert stats.blank_table_unread_pages == set()

    def test_reader_without_blank_branch_is_only_vlm(self, monkeypatch):
        # Флаг выключен — читатель документа не берётся дочитывать рамки, а
        # ветка модели зрения живёт своей жизнью.
        monkeypatch.setattr(settings, "BLANK_TABLE_OCR_ENABLED", False)
        monkeypatch.setattr(settings, "TABLE_VLM_RETRY_ENABLED", False)
        assert blank_table_ocr.table_reader(b"%PDF-1.4", "pdf") is None


class TestGate:
    """Затвор на реальных строках замера."""

    def test_rotated_garbage_is_rejected(self):
        # Повёрнутый мусор: доля односкриптовых 0.73 и доля букв 0.66 — выше,
        # чем у вполне читаемой рамки. Отличает только уверенность.
        assert blank_table_ocr.reject_reason(NOISE, NOISE_CONFIDENCE) is not None

    def test_short_stamp_with_one_token_is_accepted(self):
        # Один односкриптовый токен — «Лист». Правило «не меньше трёх
        # токенов» выбросило бы настоящий штамп.
        assert blank_table_ocr.reject_reason(STAMP, STAMP_CONFIDENCE) is None

    def test_table_header_is_accepted(self):
        assert blank_table_ocr.reject_reason(HEADER, HEADER_CONFIDENCE) is None

    def test_truly_blank_frame_is_rejected(self):
        # 4 рамки замера из 24 пусты по-настоящему: Tesseract не отдаёт ни
        # символа. Это шелуха, а не потеря.
        assert blank_table_ocr.reject_reason("", 0.0) is not None
        assert blank_table_ocr.reject_reason("| |", 90.0) is not None

    def test_confidence_floor_is_the_measured_gap(self, monkeypatch):
        # Полушум замера — 66.1, самая слабая читаемая рамка — 75.9.
        assert blank_table_ocr.reject_reason(HEADER, 66.1) is not None
        assert blank_table_ocr.reject_reason(HEADER, 75.9) is None

    def test_long_text_without_whole_words_is_rejected(self):
        # Каша из обрывков на уверенном чтении сетки: ни одного слова из
        # четырёх букв одного алфавита.
        assert blank_table_ocr.reject_reason("Пp0 1a 2б 3в 4г 5д 6е 7ж", 90.0) is not None

    def test_alpha_share_floor(self):
        assert blank_table_ocr.reject_reason("12.5 300 1 2 3 45,0 7", 90.0) is not None


class TestReadRegion:
    def test_mean_confidence_over_words_only(self, monkeypatch):
        # Среднее — по словам с непустым текстом и уверенностью >= 0: служебные
        # строки `image_to_data` с -1 иначе утянули бы среднее вниз.
        data = {
            "text": ["", "Лист", "56", "из", "56", " "],
            "conf": [-1, 96.0, 94.0, 95.0, 95.4, -1],
        }

        class _Pytesseract:
            class Output:
                DICT = "dict"

            @staticmethod
            def image_to_data(image, lang=None, config=None, output_type=None):
                return data

        monkeypatch.setitem(__import__("sys").modules, "pytesseract", _Pytesseract)
        text, confidence = blank_table_ocr.read_region(_Image(200, 60))

        assert text == "Лист 56 из 56"
        assert confidence == pytest.approx(95.1, abs=0.05)

    def test_no_words_is_zero_confidence(self, monkeypatch):
        class _Pytesseract:
            class Output:
                DICT = "dict"

            @staticmethod
            def image_to_data(image, lang=None, config=None, output_type=None):
                return {"text": ["", " "], "conf": [-1, -1]}

        monkeypatch.setitem(__import__("sys").modules, "pytesseract", _Pytesseract)
        assert blank_table_ocr.read_region(_Image(200, 60)) == ("", 0.0)


class TestUnreadPagesReachTheResult:
    """
    Отказ доезжает до `unread_text_pages` парсера.

    Это и есть причина молчания 1180 рамок: `_unread_pages` возвращает
    пустой список, пока проход слоя не обрывался, а уборка рамку снимает —
    и без объединения потеря становится невидимой.
    """

    def _response(self):
        import json

        middle = {"pdf_info": [{
            "page_idx": 0,
            "page_size": [600, 800],
            # Рамка без HTML, без вырезки и без содержимого.
            "para_blocks": [{"type": "table", "bbox": [50, 480, 550, 720], "blocks": []}],
        }]}
        return {"results": {"test.pdf": {"middle_json": json.dumps(middle)}}}

    def test_rejected_frame_marks_the_page_in_parse_result(self, temp_storage, monkeypatch):
        import requests_mock

        from core.providers import blank_table_ocr as module
        from core.providers.mineru_parser import MinerUParserProvider
        from ingest.tasks import build_warnings

        monkeypatch.setattr(settings, "OCR_LAYER_ENABLED", True)
        # Читатель документа отказывает: затвор не пропустил прочитанное.
        monkeypatch.setattr(module, "table_reader", lambda *a, **k: lambda block: None)

        temp_storage.write_file("documents/test.pdf", b"%PDF-1.4 fake")
        parser = MinerUParserProvider(temp_storage)
        # Слой распознавания: строка на странице есть, но не в рамке.
        line = TextLine(text="Приложение № 15", bbox=[0.1, 0.04, 0.9, 0.09],
                        page=1, confidence=0.9)
        monkeypatch.setattr(parser.tesseract, "text_layer",
                            lambda uri, file_type=None, first_page=1:
                            TextLayer(pages={1: [line]}, page_count=1))
        monkeypatch.setattr(parser.tesseract, "truncated_at", None, raising=False)

        with requests_mock.Mocker() as m:
            m.post(f"{settings.MINERU_ENDPOINT.rstrip('/')}/file_parse", json=self._response())
            result = parser.parse("documents/test.pdf", "pdf")

        assert result.text_layer_stats["blank_tables_rejected"] == 1
        assert result.unread_text_pages == [1]
        # Предупреждение собирает `build_warnings` — в контрактном виде.
        assert any(w.startswith("TEXT_NOT_READ: pages 1") for w in build_warnings(result, []))
