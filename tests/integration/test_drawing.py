"""
Приёмка ветки чертежей на живой модели.

Требуется хост с Qwen3-VL (vLLM, OpenAI-совместимый) и `QWEN_ENDPOINT` в
окружении. Без него все тесты пропускаются: поднимать модель ради юнит-тестов
незачем, а без неё ветка чертежей не работает по определению.

    QWEN_ENDPOINT=http://gpu-host:8000 pytest tests/integration/test_drawing.py -m integration

Детектор YOLO и Florence из проекта убраны: в разборе чертежа
они не задействованы (см. шапку core/drawing_processor.py).
"""

import pytest

from core.config import settings
from core.drawing_processor import MODE_SHEET_ONLY, MODE_VISION_VLM, DrawingProcessor
from core.providers import drawing_vision
from core.providers.qwen_client import QwenVisionClient
from core.providers.storage import LocalStorageProvider

DRAWING = "fixtures/sample_drawing.png"

pytestmark = pytest.mark.integration


@pytest.fixture(scope="session")
def client():
    qwen = QwenVisionClient()
    if not qwen.available:
        pytest.skip("QWEN_ENDPOINT не задан")
    if not qwen.health():
        pytest.skip(f"Хост модели не отвечает: {settings.QWEN_ENDPOINT}")
    return qwen


@pytest.fixture(scope="session")
def storage():
    return LocalStorageProvider(base_path="/data/shared")


@pytest.fixture(scope="session")
def sheet():
    pytest.importorskip("PIL")
    from tests.fixtures import synthetic_sheet

    return synthetic_sheet.build()


class TestModel:
    def test_sheet_is_read_as_json(self, client, sheet):
        reading = client.read_sheet(sheet, hint="рамка формата найдена")
        assert reading is not None, "модель не вернула разбираемый JSON"
        assert reading.sheet_type
        for annotation in reading.annotations:
            assert "value" in annotation

    def test_region_is_read_as_a_table(self, client, sheet):
        layout = drawing_vision.analyse(sheet)
        assert layout.title_block is not None
        piece = drawing_vision.crop(layout.image, layout.title_block.bbox)

        block = client.read_title_block(piece)
        assert block is not None
        assert isinstance(block["fields"], dict)


class TestProcessor:
    def test_contract_shape(self, client, storage):
        if not storage.exists(DRAWING):
            pytest.skip("Фикстура чертежа не смонтирована")

        result = DrawingProcessor(storage, qwen=client).process(DRAWING, {})
        assert set(result) == {
            "detected_count", "parsed_fields", "avg_confidence",
            "completeness", "mode", "payload",
        }
        assert result["mode"] in (MODE_VISION_VLM, MODE_SHEET_ONLY, "none")
        for field in result["parsed_fields"]:
            assert field["category"]
            assert 0 <= field["confidence"] <= 1
            assert all(0.0 <= c <= 1.0 for c in field["bbox"])

    def test_nothing_is_invented_on_a_blank_sheet(self, client, storage):
        """Пустой лист должен остаться пустым, а не обрасти выдуманными полями."""
        pytest.importorskip("PIL")
        from PIL import Image

        blank = "fixtures/blank_sheet.png"
        import io

        buffer = io.BytesIO()
        Image.new("RGB", (1600, 1131), "white").save(buffer, format="PNG")
        storage.write_file(blank, buffer.getvalue())

        result = DrawingProcessor(storage, qwen=client).process(blank, {})
        assert len(result["parsed_fields"]) <= 1
