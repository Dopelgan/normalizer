"""
Доказательство чтения чертежа моделью — в логе.

На показе заказчику нужно видеть прямо в логе, что Qwen-VL действительно
читала каждый лист: по строке на каждый вызов модели (лист целиком, весь
текст листа, штамп, таблица) и строка-итог на лист. Содержимое листа,
ключи и токены в лог не попадают.
"""

import logging

import pytest

from core.drawing_processor import DrawingProcessor
from core.providers.qwen_client import SheetReading, Transcript

pytest.importorskip("numpy")
pytest.importorskip("PIL")

from tests.fixtures import synthetic_sheet as sheet  # noqa: E402

SHEET_URI = "assets/doc-1/sheet.png"
SECRET_TEXT = "⌀44H7"


class FakeQwen:
    """Модель отвечает заготовленным: проверяется лог, а не чтение."""

    def __init__(self):
        self.available = True
        self.model = "qwen3-vl-test"

    def read_sheet(self, image, hint=""):
        return SheetReading(
            sheet_type="detail",
            annotations=[
                {"category": "size", "value": SECRET_TEXT, "bbox": [0.2, 0.3, 0.3, 0.35]},
                {"category": "roughness", "value": "Ra 3.2", "bbox": [0.4, 0.4, 0.5, 0.45]},
            ],
        )

    def transcribe(self, image):
        return ["Технические требования", "1. Общие допуски по ГОСТ 30893.1-m"]

    def transcribe_sheet(self, image, until=None):
        return Transcript(lines=self.transcribe(image), tiles=2)

    def read_title_block(self, image):
        return {"fields": {"обозначение": "АБВГ.123456.001", "масса": "12,35"}, "rows": []}

    def read_title_block_fields(self, image):
        return {"fields": {"обозначение": "АБВГ.123456.001"}, "handwritten": {}, "rows": []}

    def read_side_column(self, image):
        return None

    def read_table(self, image):
        return {"title": "", "headers": ["Поз."], "rows": [["1"], ["2"], ["3"]]}


@pytest.fixture
def processor(temp_storage):
    temp_storage.write_file(SHEET_URI, sheet.png_bytes())
    return DrawingProcessor(temp_storage, qwen=FakeQwen())


def lines_of(caplog) -> list:
    return [r.getMessage() for r in caplog.records
            if r.levelno == logging.INFO and r.getMessage().startswith("Qwen-VL")]


class TestCallLines:
    def test_each_model_pass_gets_its_own_line(self, processor, caplog):
        with caplog.at_level(logging.INFO, logger="core.drawing_processor"):
            processor.process(SHEET_URI, {}, page=3, transcribe=True)
        messages = lines_of(caplog)

        sheet_line = next(m for m in messages if "проход «лист целиком»" in m)
        assert "стр. 3" in sheet_line
        assert "надписей 2" in sheet_line
        assert "модель qwen3-vl-test" in sheet_line
        assert sheet_line.endswith(" с")

        text_line = next(m for m in messages if "проход «весь текст листа»" in m)
        assert "строк текста 2" in text_line
        assert "стр. 3" in text_line

        assert any("проход «штамп»" in m and "граф " in m for m in messages)

    def test_sheet_summary_names_page_mode_model_and_counts(self, processor, caplog):
        with caplog.at_level(logging.INFO, logger="core.drawing_processor"):
            result = processor.process(SHEET_URI, {}, page=3, transcribe=True)
        summary = [m for m in lines_of(caplog) if m.startswith("Qwen-VL итог:")]

        assert len(summary) == 1, "на лист — одна строка-итог"
        line = summary[0]
        assert SHEET_URI in line
        assert "стр. 3" in line
        assert f"режим {result['mode']}" in line
        assert f"полей {result['detected_count']}" in line
        assert "строк текста 2" in line
        regions = result["completeness_mechanisms"]["geometry_regions"]
        assert f"областей {regions['read']} из {regions['expected']}" in line
        assert "модель qwen3-vl-test" in line

    def test_page_of_a_cutout_comes_from_the_fragment(self, processor, caplog):
        """У вырезки MinerU номер страницы берётся из фрагмента, а не из `page`."""
        from core.models.contract import Content, Fragment, Position, Provenance

        fragment = Fragment(
            fragment_id="doc-1-frag-001",
            type="image",
            content=Content(image_ref=SHEET_URI),
            position=Position(page=7, bbox=[0.0, 0.0, 1.0, 1.0], order=1),
            confidence=0.9,
            completeness=1.0,
            provenance=Provenance(method="mineru_image", strategy_level=2, source="vector_pdf"),
        )
        with caplog.at_level(logging.INFO, logger="core.drawing_processor"):
            processor.enrich_fragments([fragment], {})

        assert all("стр. 7" in m for m in lines_of(caplog))

    def test_log_carries_no_sheet_content(self, processor, caplog):
        with caplog.at_level(logging.INFO, logger="core.drawing_processor"):
            processor.process(SHEET_URI, {}, page=3, transcribe=True)
        joined = "\n".join(lines_of(caplog))

        assert SECRET_TEXT not in joined
        assert "Технические требования" not in joined
        assert "АБВГ.123456.001" not in joined
