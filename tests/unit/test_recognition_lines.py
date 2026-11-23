"""
Строки распознавания с координатами в положении фрагмента
(`RECOGNITION_LINES`). Все данные синтетические.
"""

import json

import pytest
from PIL import Image

from core import document_parts, recognition_lines
from core.config import settings
from core.ladder.context import DocumentContext
from core.ladder.strategies.native_text import PdfTextLayerStrategy
from core.models.contract import Position
from core.models.parse_result import ParsedBlock, ParseResult
from core.normalizer.normalizer import TextNormalizer
from core.providers.mineru_parser import MinerUParserProvider
from core.providers.tesseract_fallback import OcrLine, TesseractFallbackProvider
from core.providers.text_layer import OCR_LAYER, TextLayer, TextLine, reconcile_with_text_layer

NEW_KEYS = {"lines", "page_size", "page_rotation"}


@pytest.fixture
def lines_on(monkeypatch):
    monkeypatch.setattr(settings, "RECOGNITION_LINES", True)


@pytest.fixture
def lines_off(monkeypatch):
    monkeypatch.setattr(settings, "RECOGNITION_LINES", False)


@pytest.fixture
def text_pdf_bytes():
    pymupdf = pytest.importorskip("pymupdf")
    document = pymupdf.open()
    page = document.new_page(width=600, height=800)
    for row in range(5):
        page.insert_text((60, 60 + row * 24), "plain running text line here", fontsize=11)
    data = document.tobytes()
    document.close()
    return data


def _normalize(blocks):
    result = ParseResult(blocks=blocks, parser_name="test", source_kind="scanned_pdf")
    return TextNormalizer().normalize(result, {"doc_id": "doc-1"})


def _middle_json():
    # Страница 1000×2000, один текстовый блок из двух строк MinerU.
    return {"pdf_info": [{
        "page_idx": 0, "page_size": [1000, 2000],
        "para_blocks": [{
            "type": "text", "bbox": [100, 200, 900, 300], "score": 0.9,
            "lines": [
                {"bbox": [100, 200, 900, 240],
                 "spans": [{"content": "Первая", "score": 0.8},
                           {"content": "строка", "score": 1.0}]},
                {"bbox": [100, 260, 500, 300], "spans": [{"content": "Вторая"}]},
            ],
        }],
    }]}


class TestContractShape:
    def test_flag_off_position_json_unchanged(self):
        dumped = Position(page=1).model_dump(mode="json")
        assert not NEW_KEYS & set(dumped)

    def test_optional_line_keys_are_omitted(self):
        position = Position(page=1, page_size=[595, 842], page_rotation=0, lines=[
            {"text": "a", "bbox": [0, 0, 0.5, 0.1], "source": "text_layer"},
        ])
        dumped = position.model_dump(mode="json")
        assert dumped["lines"] == [{"text": "a", "bbox": [0.0, 0.0, 0.5, 0.1],
                                    "source": "text_layer"}]
        assert dumped["page_size"] == [595.0, 842.0]
        assert dumped["page_rotation"] == 0
        # Старый потребитель читает положение обратно без потерь.
        assert Position(**dumped).lines[0].text == "a"

    def test_fields_empty_when_flag_off(self, lines_off):
        line = TextLine(text="x", bbox=[0, 0, 1, 1], page=1)
        assert recognition_lines.fields([line], "text_layer", page_size=(1, 1)) == {}


class TestMinerU:
    def test_lines_from_middle_json(self, lines_on, temp_storage):
        provider = MinerUParserProvider(temp_storage)
        blocks, _ = provider._blocks_from_middle_json(_middle_json(), {})
        [block] = blocks
        assert block.page_size == [1000.0, 2000.0]
        assert block.lines == [
            {"text": "Первая строка", "bbox": [0.1, 0.1, 0.9, 0.12],
             "source": "mineru", "confidence": 0.9},
            {"text": "Вторая", "bbox": [0.1, 0.13, 0.5, 0.15], "source": "mineru"},
        ]
        [fragment] = _normalize(blocks)
        dumped = fragment.position.model_dump(mode="json")
        assert [l["text"] for l in dumped["lines"]] == ["Первая строка", "Вторая"]
        assert dumped["page_size"] == [1000.0, 2000.0]

    def test_flag_off_keeps_old_block_and_position(self, lines_off, temp_storage):
        provider = MinerUParserProvider(temp_storage)
        blocks, _ = provider._blocks_from_middle_json(_middle_json(), {})
        assert blocks[0].lines is None
        [fragment] = _normalize(blocks)
        assert not NEW_KEYS & set(fragment.position.model_dump(mode="json"))


class TestTextLayer:
    def test_vector_layer_lines_without_confidence(self, lines_on, text_pdf_bytes):
        ctx = DocumentContext(uri="documents/x", file_type="pdf", metadata={},
                              data=text_pdf_bytes)
        result = PdfTextLayerStrategy().run(ctx)
        block = result.blocks[0]
        assert block.lines and all(l["source"] == "text_layer" for l in block.lines)
        assert all("confidence" not in l for l in block.lines)
        assert block.page_rotation == 0 and len(block.page_size) == 2
        # Рамка блока — объединение рамок его строк.
        xs0 = min(l["bbox"][0] for l in block.lines)
        assert xs0 == pytest.approx(block.bbox[0], abs=1e-3)

    def test_rebuilt_block_takes_layer_lines(self, lines_on):
        # Блок MinerU пересобирается из строк слоя распознавания: строки
        # блока должны стать строками слоя, а не остаться строками MinerU.
        block = ParsedBlock(type="text", text="Перв строк", page=1,
                            bbox=[0.1, 0.1, 0.9, 0.2], method="mineru_ocr",
                            lines=[{"text": "Перв строк", "bbox": [0.1, 0.1, 0.9, 0.2],
                                    "source": "mineru"}])
        layer = TextLayer(pages={1: [
            TextLine(text="Первая строка", bbox=[0.1, 0.1, 0.9, 0.14], page=1, confidence=0.7),
            TextLine(text="Вторая строка", bbox=[0.1, 0.15, 0.9, 0.19], page=1, confidence=0.9),
        ]}, page_count=1)
        blocks, _ = reconcile_with_text_layer([block], layer, OCR_LAYER)
        [rebuilt] = [b for b in blocks if b.type == "text"]
        assert [l["text"] for l in rebuilt.lines] == ["Первая строка", "Вторая строка"]
        assert {l["source"] for l in rebuilt.lines} == {"ocr_layer"}
        assert [l["confidence"] for l in rebuilt.lines] == [0.7, 0.9]


class TestTesseract:
    def test_lines_kept_before_merge(self, lines_on, monkeypatch):
        provider = TesseractFallbackProvider(storage=object())
        monkeypatch.setattr(provider, "ocr_lines", lambda _image, _psm=None: [
            OcrLine(text="Строка один", bbox=[0.1, 0.1, 0.6, 0.13], confidence=0.91),
            OcrLine(text="Строка два", bbox=[0.1, 0.14, 0.5, 0.17], confidence=0.62),
        ])
        blocks = provider._blocks_from_image(Image.new("L", (400, 600), 255), page=1)
        [block] = blocks
        assert block.page_size == [400.0, 600.0]
        assert block.page_rotation == 0
        assert [(l["text"], l["confidence"], l["source"]) for l in block.lines] == [
            ("Строка один", 0.91, "tesseract"), ("Строка два", 0.62, "tesseract"),
        ]

    def test_turned_lines_carry_rotation(self, lines_on, monkeypatch):
        provider = TesseractFallbackProvider(storage=object())
        monkeypatch.setattr(provider, "ocr_lines", lambda _image, _psm=None: [
            OcrLine(text="Боковая надпись листа", bbox=[0.1, 0.1, 0.13, 0.6],
                    confidence=0.8, rotation=90.0),
        ])
        [block] = provider._blocks_from_image(Image.new("L", (400, 600), 255), page=1)
        assert block.page_rotation == 90
        assert block.lines[0]["rotation"] == 90.0


class TestMergeAndSize:
    def test_merged_fragment_keeps_both_line_sets(self, lines_on, monkeypatch):
        monkeypatch.setattr(settings, "DOCUMENT_PARTS_MARKUP", True)
        head = ParsedBlock(type="text", text="Имеется:", page=1, order=0,
                           bbox=[0.1, 0.1, 0.3, 0.12],
                           lines=[{"text": "Имеется:", "bbox": [0.1, 0.1, 0.3, 0.12],
                                   "source": "tesseract", "confidence": 0.9}])
        tail = ParsedBlock(type="text", text="паспорт", page=1, order=1,
                           bbox=[0.1, 0.13, 0.3, 0.15],
                           lines=[{"text": "паспорт", "bbox": [0.1, 0.13, 0.3, 0.15],
                                   "source": "tesseract", "confidence": 0.8}])
        fragments = document_parts.apply(_normalize([head, tail]))
        joined = [f for f in fragments if f.position.lines and len(f.position.lines) == 2]
        assert joined, "строки второго участника потеряны при соединении"

    def test_payload_size_per_line(self, lines_on):
        # Оценка прироста выдачи: строка средней длины с рамкой и уверенностью.
        line = recognition_lines.record("Калибратор температуры КТ-500 зав. номер",
                                        [0.1234, 0.2345, 0.8765, 0.2567], "tesseract", 0.913)
        size = len(json.dumps(line, ensure_ascii=False).encode("utf-8"))
        text = len(line["text"].encode("utf-8"))
        # Сверх самого текста строка стоит меньше 100 байт.
        assert size - text < 100
