"""
Неразобранная таблица скана перечитывается моделью зрения.

Типичный случай — альбомная таблица «Данные о сварных соединениях» на
книжном листе паспорта. MinerU отдаёт ячейки разметкой, OCR читает лист
поперёк, и таблица разбирается на обрывки строк — номера протоколов, объём
контроля и результаты из индекса пропадают.
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
        # До срока задачи меньше одного вызова модели: таблица разбирается
        # на текст, а не роняет документ мягким лимитом.
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


# ---------------------------------------------------------------------------
# Спецификация ЕСКД скана (TABLE_VLM_ESKD_SPEC)
# ---------------------------------------------------------------------------

# Строки MinerU со скана ТАДУ 687000.001 (изм. 2), лист 2: номер позиции
# склеен с обозначением, надпись поля подшивки стала первой ячейкой строки.
SPEC_MINERU = {
    "headers": ["", "шowdoф", "DHOE", "290", "Обозначение", "Наименование", "", "Приме-чание"],
    "rows": [
        ["A4", "", "1", "& 1 TARY 305364.00200", "Компенсатор перепадов давления", "$*1", "Заим.", ""],
        ["p upo", "", "A4", "", "8", "ТАДУ 765200.082.00", "Ввод кабельный типа", "2"],
        ["A3", "", "17", "17 TAAY 687000001001", "Крышка", "1", "", ""],
        ["A4", "", "18", "18 TAY 687000001002", "Уплотнение 7", "1", "", ""],
    ],
}
SPEC_QWEN = {
    "headers": ["Формат", "Зона", "Поз.", "Обозначение", "Наименование", "Кол.", "Примечание"],
    "rows": [
        ["A4", "", "1", "ТАДУ 305364.002.00", "Компенсатор перепадов давления", "1*1", "Заим."],
        ["A4", "", "8", "ТАДУ 765200.082.00", "Ввод кабельный типа \"a\"", "*2", "Заим."],
        ["A4", "", "9", "ТАДУ 765200.083.00", "Ввод кабельный типа \"k\"", "*2", "Заим."],
        ["A3", "", "17", "ТАДУ 687000.001.001", "Крышка", "1", ""],
        ["A4", "", "18", "ТАДУ 687000.001.002", "Уплотнение", "1", ""],
    ],
    "total_row": None,
}


def spec_block():
    from core.models.parse_result import ParsedBlock

    return ParsedBlock(type="table", page=1, order=0, bbox=[0.05, 0.05, 0.95, 0.9], confidence=0.9,
                       method="mineru_table", table_data={k: [list(r) for r in v] if k == "rows" else list(v)
                                                          for k, v in SPEC_MINERU.items()})


def spec_layer():
    from tests.unit.test_mangled_tables import line

    return TextLayer(pages={1: [line("Крышка", 0.5, 0.6), line("Уплотнение", 0.5, 0.65)]}, page_count=1)


class TestEskdSpec:
    @pytest.mark.parametrize("flag", [True, False])
    def test_scan_spec_is_read_by_vision(self, monkeypatch, flag):
        monkeypatch.setattr(settings, "TABLE_VLM_ESKD_SPEC", flag)
        called = []

        def reader(block):
            called.append(block.page)
            return SPEC_QWEN

        blocks, stats = reconcile_with_text_layer([spec_block()], spec_layer(), source=OCR_LAYER,
                                                  table_reader=reader)
        table = next(b for b in blocks if b.type == "table")
        rows = table.table_data["rows"]
        if flag:
            assert called == [1] and stats.vlm_tables == 1
            assert ["A3", "", "17", "ТАДУ 687000.001.001", "Крышка", "1", ""] in rows
        else:
            assert called == [] and stats.vlm_tables == 0
            assert not any(r[3:5] == ["ТАДУ 687000.001.001", "Крышка"] for r in rows)

    def test_short_answer_keeps_mineru_table(self, monkeypatch):
        monkeypatch.setattr(settings, "TABLE_VLM_ESKD_SPEC", True)
        short = dict(SPEC_QWEN, rows=SPEC_QWEN["rows"][:1])
        blocks, stats = reconcile_with_text_layer([spec_block()], spec_layer(), source=OCR_LAYER,
                                                  table_reader=lambda b: short)
        assert stats.vlm_tables == 0

    @pytest.mark.parametrize("data, expected", [
        (SPEC_MINERU, True),
        ({"headers": ["Поз.", "Обозначение", "Наименование", "Кол."], "rows": []}, True),
        ({"headers": ["Формула", "Физические величины"], "rows": [["x", "y"]]}, False),
        ({"headers": ["Обозначение", "Наименование"], "rows": []}, False),
        (None, False),
    ])
    def test_spec_detection(self, data, expected):
        assert table_vlm.eskd_spec(data) is expected

    def test_spec_crop_goes_with_its_own_ceiling(self, monkeypatch):
        monkeypatch.setattr(settings, "TABLE_VLM_RETRY_ENABLED", True)
        monkeypatch.setattr(settings, "TABLE_VLM_ESKD_SPEC", True)
        seen = {}

        class Client:
            available = True

            def read_table(self, image, max_side=None):
                seen["max_side"] = max_side
                return {"headers": [], "rows": [["1"]]}

        def render(data, ftype, page, dpi=None):
            seen["dpi"] = dpi
            return Image.new("RGB", (400, 600), "white")

        monkeypatch.setattr(table_vlm, "_render_page", render)
        monkeypatch.setattr(table_vlm, "_upright", lambda image: image)
        read = table_vlm.table_reader(b"%PDF", "pdf", client=Client())
        read(spec_block())
        assert seen == {"dpi": settings.TABLE_VLM_SPEC_DPI, "max_side": settings.TABLE_VLM_SPEC_MAX_SIDE}

    def test_continuation_from_next_sheet_is_kept(self, monkeypatch):
        """
        Многостраничную спецификацию MinerU отдаёт одной таблицей: вырезка —
        только первая страница, строки со второй шапки остаются от MinerU.
        """
        monkeypatch.setattr(settings, "TABLE_VLM_ESKD_SPEC", True)
        block = spec_block()
        block.table_data["rows"] += [
            ["", "wowoф", "DH", "210", "Обозначение", "Наименование", "", "Приме-чание"],
            ["A", "", "19", "ТАДУ 687000.001.003", "Панель монтажная", "*2", "", ""],
        ]
        blocks, stats = reconcile_with_text_layer([block], spec_layer(), source=OCR_LAYER,
                                                  table_reader=lambda b: SPEC_QWEN)
        rows = next(b for b in blocks if b.type == "table").table_data["rows"]
        assert stats.vlm_tables == 1
        assert rows[:5] == SPEC_QWEN["rows"]
        # Строки MinerU, прочитанные моделью, не повторяются.
        assert not any("TAAY 687000001001" in " ".join(r) for r in rows)
        assert rows[-1] == ["A", "", "19", "ТАДУ 687000.001.003", "Панель монтажная", "*2", "", ""]
