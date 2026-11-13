"""
Неразобранная таблица скана перечитывается моделью зрения.

Случай из СППИ (паспорт 230390, стр. 88 и 139): альбомная таблица «Данные о
сварных соединениях» на книжном листе. MinerU отдал ячейки разметкой, OCR
прочитал лист поперёк, и таблица разбиралась на обрывки строк — номера
протоколов, объём контроля и результаты из индекса пропадали.
"""

import io

import pytest
from PIL import Image

from core.config import settings
from core.providers import table_vlm
from core.providers.text_layer import OCR_LAYER, TextLayer, reconcile_with_text_layer
from tests.unit.test_mangled_tables import ocr_lines, table_block

QWEN_TABLE = {
    "title": "Table D.9. Data on welded joints",
    "headers": ["Test method", "Testing scope", "Test results", "Protocol"],
    "rows": [
        ["Visual check", "100 %", "SAT", "№541-3 УПП С ВИК dd 14.04.2025"],
        ["Radiographic inspection *", "25%", "SAT", "№541-230390 УПП-КВЭ-ИХ РКС dd 18.04.2025"],
    ],
}


def degrading_layer():
    # Строк меньше, чем строк таблицы, — собрать её по слою нельзя.
    return TextLayer(pages={1: ocr_lines(rows=1, with_header=False)}, page_count=1)


class TestReconcileHook:
    def test_reader_table_replaces_degradation(self):
        block = table_block()
        reader_calls = []

        def reader(b):
            reader_calls.append(b.page)
            return {"headers": QWEN_TABLE["headers"], "rows": QWEN_TABLE["rows"], "total_row": None}

        blocks, stats = reconcile_with_text_layer(
            [block], degrading_layer(), source=OCR_LAYER, table_reader=reader
        )

        assert reader_calls == [1]
        assert stats.degraded_tables == 0
        assert stats.vlm_tables == 1
        table = next(b for b in blocks if b.type == "table")
        assert table.table_data["rows"] == QWEN_TABLE["rows"]
        assert table.method == "vlm_escalation"
        assert table.table_html is None
        # Разметка формул из ячеек MinerU не уходит в индекс второй копией.
        assert not any(b.type == "formula" for b in blocks)

    def test_lines_missing_from_model_table_are_kept(self):
        def reader(b):
            return {"headers": ["Формула"], "rows": [["x = x0 + vt"]], "total_row": None}

        blocks, stats = reconcile_with_text_layer(
            [table_block()], degrading_layer(), source=OCR_LAYER, table_reader=reader
        )

        texts = " ".join(b.text or "" for b in blocks if b.type == "text")
        assert "Прямолинейное равномерное движение" in texts
        assert stats.table_lines_kept >= 1

    def test_reader_refusal_keeps_old_degradation(self):
        blocks, stats = reconcile_with_text_layer(
            [table_block()], degrading_layer(), source=OCR_LAYER, table_reader=lambda b: None
        )

        assert stats.degraded_tables == 1 and stats.vlm_tables == 0
        assert any(b.type == "text" for b in blocks)
        assert next(b for b in blocks if b.type == "table").table_data is None

    def test_reader_not_called_for_repairable_table(self):
        called = []
        layer = TextLayer(pages={1: ocr_lines()}, page_count=1)

        _, stats = reconcile_with_text_layer(
            [table_block()], layer, source=OCR_LAYER, table_reader=lambda b: called.append(b)
        )

        assert called == []
        assert stats.repaired_tables == 1


class FakeClient:
    available = True

    def __init__(self, answer):
        self.answer = answer
        self.images = []

    def read_table(self, image):
        self.images.append(image)
        return self.answer


def png_bytes(size=(400, 300)):
    buffer = io.BytesIO()
    Image.new("RGB", size, "white").save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def enabled(monkeypatch):
    monkeypatch.setattr(settings, "TABLE_VLM_RETRY_ENABLED", True)
    monkeypatch.setattr(settings, "TABLE_VLM_MAX_TABLES", 8)
    monkeypatch.setattr(table_vlm, "_upright", lambda image: image)


class TestTableReader:
    def test_disabled_by_default(self, monkeypatch):
        from core.config import Settings

        assert Settings.model_fields["TABLE_VLM_RETRY_ENABLED"].default is False
        monkeypatch.setattr(settings, "TABLE_VLM_RETRY_ENABLED", False)
        assert table_vlm.table_reader(png_bytes(), "png", client=FakeClient(QWEN_TABLE)) is None

    def test_unavailable_model_gives_no_reader(self, enabled):
        client = FakeClient(QWEN_TABLE)
        client.available = False
        assert table_vlm.table_reader(png_bytes(), "png", client=client) is None

    def test_reads_crop_of_image(self, enabled):
        client = FakeClient(QWEN_TABLE)
        reader = table_vlm.table_reader(png_bytes(), "png", client=client)

        data = reader(table_block())

        assert data["rows"] == QWEN_TABLE["rows"]
        assert data["headers"] == QWEN_TABLE["headers"]
        assert data["title"] == QWEN_TABLE["title"]
        assert len(client.images) == 1

    def test_reads_pdf_page(self, enabled):
        fitz = pytest.importorskip("fitz")
        document = fitz.open()
        document.new_page(width=595, height=842)
        document.new_page(width=595, height=842)
        pdf = document.tobytes()
        client = FakeClient(QWEN_TABLE)
        block = table_block()
        block.page = 2

        assert table_vlm.table_reader(pdf, "pdf", client=client)(block)["rows"]
        assert len(client.images) == 1

    @pytest.mark.parametrize("answer", [None, {"headers": ["a"], "rows": []},
                                        {"headers": [], "rows": [["", " "]]}])
    def test_empty_answer_is_none(self, enabled, answer):
        reader = table_vlm.table_reader(png_bytes(), "png", client=FakeClient(answer))
        assert reader(table_block()) is None

    def test_budget_per_document(self, enabled, monkeypatch):
        monkeypatch.setattr(settings, "TABLE_VLM_MAX_TABLES", 1)
        client = FakeClient(QWEN_TABLE)
        reader = table_vlm.table_reader(png_bytes(), "png", client=client)

        assert reader(table_block()) is not None
        assert reader(table_block()) is None
        assert len(client.images) == 1

    def test_model_failure_is_none(self, enabled):
        class Broken(FakeClient):
            def read_table(self, image):
                raise RuntimeError("boom")

        reader = table_vlm.table_reader(png_bytes(), "png", client=Broken(QWEN_TABLE))
        assert reader(table_block()) is None

    def test_soft_time_limit_is_not_swallowed(self, enabled):
        from celery.exceptions import SoftTimeLimitExceeded

        class Slow(FakeClient):
            def read_table(self, image):
                raise SoftTimeLimitExceeded()

        reader = table_vlm.table_reader(png_bytes(), "png", client=Slow(QWEN_TABLE))
        with pytest.raises(SoftTimeLimitExceeded):
            reader(table_block())

    def test_no_call_when_task_deadline_is_near(self, enabled, monkeypatch):
        # До срока задачи меньше одного вызова модели: таблица остаётся
        # прежнему исходу, а не роняет документ мягким лимитом.
        from core import deadline

        monkeypatch.setattr(deadline, "remaining", lambda: settings.VLM_TIMEOUT - 1)
        client = FakeClient(QWEN_TABLE)
        reader = table_vlm.table_reader(png_bytes(), "png", client=client)

        assert reader(table_block()) is None
        assert client.images == []


class TestUpright:
    def test_rotates_by_confident_osd(self, monkeypatch):
        from core.providers import ocr_prepare

        monkeypatch.setattr(ocr_prepare, "osd_rotation", lambda gray: (90, 5.0))
        image = Image.new("RGB", (400, 300), "white")
        image.putpixel((0, 0), (255, 0, 0))  # левый верхний угол
        upright = table_vlm._upright(image)
        # Поворот на 90 по часовой: левый верхний угол уходит в правый верхний.
        assert upright.size == (300, 400)
        assert upright.getpixel((299, 0)) == (255, 0, 0)

    def test_keeps_image_on_unsure_osd(self, monkeypatch):
        from core.providers import ocr_prepare

        monkeypatch.setattr(ocr_prepare, "osd_rotation", lambda gray: (90, 0.5))
        assert table_vlm._upright(Image.new("RGB", (400, 300))).size == (400, 300)
