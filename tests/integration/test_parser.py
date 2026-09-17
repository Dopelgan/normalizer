"""
Парсер с реальным MinerU.

Требуется: docker compose up -d mineru-api
Фикстуры монтируются в /data/shared/fixtures (см. docker-compose.yml).
"""

import logging
import time

import pytest
import requests

from core.config import settings
from core.models.parse_result import ParseResult
from core.providers.document_parser_factory import DocumentParserFactory
from core.providers.mineru_parser import MinerUParserProvider
from core.providers.storage import LocalStorageProvider

logger = logging.getLogger(__name__)

SHARED_ROOT = "/data/shared"
PDF = "fixtures/sample.pdf"
TABLE_PDF = "fixtures/sample_table.pdf"
IMAGE = "fixtures/sample_image.png"


@pytest.fixture(scope="session")
def storage():
    return LocalStorageProvider(base_path=SHARED_ROOT)


@pytest.fixture(scope="session")
def parser(storage):
    return MinerUParserProvider(storage)


def require_mineru():
    url = f"{settings.MINERU_ENDPOINT.rstrip('/')}/health"
    try:
        started = time.time()
        response = requests.get(url, timeout=5)
        logger.info("MinerU health %s: %s за %.2f с", url, response.status_code, time.time() - started)
        if response.status_code != 200:
            pytest.skip(f"MinerU отвечает {response.status_code}")
    except requests.RequestException as exc:
        pytest.skip(f"MinerU недоступен: {exc}")


def require_fixture(storage, uri):
    if not storage.exists(uri):
        pytest.skip(f"Фикстура {uri} не смонтирована в {SHARED_ROOT}")


def describe(result: ParseResult, name: str) -> None:
    by_type = {}
    for block in result.blocks:
        by_type[block.type] = by_type.get(block.type, 0) + 1
    logger.info("%s: parser=%s fallback=%s блоков=%d %s",
                name, result.parser_name, result.is_fallback, len(result.blocks), by_type)


@pytest.mark.integration
class TestMinerURealFiles:
    def test_pdf_text_layer(self, parser, storage):
        require_mineru()
        require_fixture(storage, PDF)

        result = parser.parse(PDF, "pdf")
        describe(result, "sample.pdf")

        assert isinstance(result, ParseResult)
        assert result.blocks, "Блоки должны быть извлечены"

        text_blocks = [b for b in result.blocks if b.type == "text"]
        assert text_blocks, "Должны быть текстовые блоки"
        all_text = " ".join(b.text for b in text_blocks if b.text)
        assert "Technical" in all_text or "Specification" in all_text

    def test_bbox_is_normalized(self, parser, storage):
        require_mineru()
        require_fixture(storage, PDF)

        result = parser.parse(PDF, "pdf")
        for block in result.blocks:
            assert len(block.bbox) == 4
            assert all(0.0 <= c <= 1.0 for c in block.bbox), f"bbox вне [0,1]: {block.bbox}"
            assert block.bbox[0] <= block.bbox[2] and block.bbox[1] <= block.bbox[3]

    def test_pages_start_from_one(self, parser, storage):
        require_mineru()
        require_fixture(storage, PDF)
        result = parser.parse(PDF, "pdf")
        assert min(b.page for b in result.blocks) >= 1

    def test_table_pdf(self, parser, storage):
        require_mineru()
        require_fixture(storage, TABLE_PDF)

        result = parser.parse(TABLE_PDF, "pdf")
        describe(result, "sample_table.pdf")
        assert result.blocks

        tables = [b for b in result.blocks if b.type == "table"]
        for table in tables:
            # Если MinerU распознал таблицу, структура обязана разобраться.
            assert table.table_data is not None or table.table_html is not None

    def test_image_falls_back_to_ocr_if_needed(self, parser, storage):
        require_mineru()
        require_fixture(storage, IMAGE)

        result = parser.parse(IMAGE, "png")
        describe(result, "sample_image.png")
        assert result.parser_name in ("mineru", "tesseract_full")

    def test_factory(self, storage):
        """
        Фабрика отдаёт лестницу, а не парсер напрямую: уровень MinerU —
        один из уровней, а не единственный путь разбора.
        """
        from core.ladder.router import LadderParserProvider
        from core.ladder.strategies.recognition import LayoutStrategy

        parser = DocumentParserFactory.get_parser(storage)
        assert isinstance(parser, LadderParserProvider)

        layout = [s for s in parser.router.strategies if isinstance(s, LayoutStrategy)]
        assert layout and layout[0].level == 5
        assert isinstance(layout[0].provider, MinerUParserProvider)


@pytest.mark.integration
class TestTesseractFallback:
    def test_ocr_reads_pdf(self, storage):
        from core.providers.tesseract_fallback import TesseractFallbackProvider

        require_fixture(storage, PDF)
        blocks = TesseractFallbackProvider(storage=storage).parse_all_pages(PDF)
        if not blocks:
            pytest.skip("Tesseract не установлен или ничего не распознал")
        assert all(b.is_fallback for b in blocks)
        assert all(b.type == "text" for b in blocks)
