"""
Парсер MinerU: разбор реального формата ответа `/file_parse`.

Старая версия этих тестов мокала выдуманную ручку `/parse` с полями
`blocks`/`failed_pages` — такого ответа MinerU не отдаёт никогда.
"""

import base64
import json

import pytest
import requests
import requests_mock

from core.config import settings
from core.providers.document_parser import ParserFailed
from core.providers import mineru_parser
from core.providers.mineru_parser import MinerUParserProvider, parse_table_html

PARSE_URL = f"{settings.MINERU_ENDPOINT.rstrip('/')}/file_parse"
PNG_1PX = base64.b64encode(base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)).decode()


@pytest.fixture
def parser(temp_storage):
    temp_storage.write_file("documents/test.pdf", b"%PDF-1.4 fake")
    return MinerUParserProvider(temp_storage)


def middle_json_response(**overrides):
    middle = {
        "pdf_info": [
            {
                "page_idx": 0,
                "page_size": [600, 800],
                "para_blocks": [
                    {
                        "type": "title",
                        "bbox": [60, 40, 540, 80],
                        "lines": [{"spans": [{"content": "Расчетный бланк"}]}],
                    },
                    {
                        "type": "text",
                        "bbox": [60, 100, 540, 300],
                        "lines": [
                            {"spans": [{"content": "Первая строка абзаца."}]},
                            {"spans": [{"content": "Вторая строка абзаца."}]},
                        ],
                    },
                    {
                        "type": "table",
                        "bbox": [60, 320, 540, 500],
                        "blocks": [{"lines": [{"spans": [{
                            "html": "<table><tr><th>Параметр</th><th>Значение</th></tr>"
                                    "<tr><td>Ra</td><td>3.2</td></tr></table>"
                        }]}]}],
                    },
                    {
                        "type": "image",
                        "bbox": [60, 520, 300, 700],
                        "blocks": [{"lines": [{"spans": [{"img_path": "images/fig1.jpg"}]}]}],
                    },
                    {
                        "type": "interline_equation",
                        "bbox": [60, 720, 300, 760],
                        "lines": [{"spans": [{"content": "S = a \\cdot b"}]}],
                    },
                ],
            }
        ]
    }
    payload = {
        "results": {
            "test.pdf": {
                "middle_json": json.dumps(middle),
                "images": {"images/fig1.jpg": PNG_1PX},
            }
        }
    }
    payload["results"]["test.pdf"].update(overrides)
    return payload


class TestMiddleJson:
    def test_block_types_extracted(self, parser):
        with requests_mock.Mocker() as m:
            m.post(PARSE_URL, json=middle_json_response())
            result = parser.parse("documents/test.pdf", "pdf")

        types = [b.type for b in result.blocks]
        assert types.count("text") == 2      # заголовок + абзац
        assert "table" in types
        assert "image" in types
        assert "formula" in types
        assert result.parser_name == "mineru"
        assert result.is_fallback is False

    def test_bbox_normalized_to_unit_range(self, parser):
        with requests_mock.Mocker() as m:
            m.post(PARSE_URL, json=middle_json_response())
            result = parser.parse("documents/test.pdf", "pdf")

        title = result.blocks[0]
        assert title.bbox == pytest.approx([0.1, 0.05, 0.9, 0.1], abs=1e-6)
        assert all(0.0 <= c <= 1.0 for b in result.blocks for c in b.bbox)

    def test_pages_numbered_from_one(self, parser):
        with requests_mock.Mocker() as m:
            m.post(PARSE_URL, json=middle_json_response())
            result = parser.parse("documents/test.pdf", "pdf")
        assert {b.page for b in result.blocks} == {1}

    def test_table_html_parsed(self, parser):
        with requests_mock.Mocker() as m:
            m.post(PARSE_URL, json=middle_json_response())
            result = parser.parse("documents/test.pdf", "pdf")

        table = next(b for b in result.blocks if b.type == "table")
        assert table.table_data["headers"] == ["Параметр", "Значение"]
        assert table.table_data["rows"] == [["Ra", "3.2"]]

    def test_images_saved_to_storage(self, parser, temp_storage):
        with requests_mock.Mocker() as m:
            m.post(PARSE_URL, json=middle_json_response())
            result = parser.parse(
                "documents/test.pdf", "pdf", {"asset_prefix": "assets/doc-1/"}
            )

        image = next(b for b in result.blocks if b.type == "image")
        assert image.image_ref == "assets/doc-1/fig1.jpg"
        assert temp_storage.exists(image.image_ref)

    def test_title_becomes_section_title(self, parser):
        with requests_mock.Mocker() as m:
            m.post(PARSE_URL, json=middle_json_response())
            result = parser.parse("documents/test.pdf", "pdf")
        assert result.blocks[0].section_title == "Расчетный бланк"


class TestContentListFallback:
    def test_used_when_middle_json_absent(self, parser):
        payload = {
            "results": {
                "test.pdf": {
                    "content_list": json.dumps([
                        {"type": "text", "text": "Абзац из content_list", "page_idx": 0},
                        {"type": "table", "table_body": "<table><tr><td>1</td></tr></table>",
                         "page_idx": 1},
                    ])
                }
            }
        }
        with requests_mock.Mocker() as m:
            m.post(PARSE_URL, json=payload)
            result = parser.parse("documents/test.pdf", "pdf")

        assert [b.page for b in result.blocks] == [1, 2]
        assert result.blocks[0].text == "Абзац из content_list"

    def test_md_content_is_last_resort(self, parser):
        with requests_mock.Mocker() as m:
            m.post(PARSE_URL, json={"results": {"test.pdf": {"md_content": "# Заголовок"}}})
            result = parser.parse("documents/test.pdf", "pdf")
        assert result.blocks[0].text == "# Заголовок"


class TestErrorHandling:
    def test_connection_error_falls_back_to_ocr(self, parser, mocker):
        mocked = mocker.patch.object(parser.tesseract, "parse_all_pages", return_value=[])
        with requests_mock.Mocker() as m:
            m.post(PARSE_URL, exc=requests.exceptions.ConnectTimeout("нет связи"))
            result = parser.parse("documents/test.pdf", "pdf")
        assert result.parser_name == "tesseract_full"
        assert result.is_fallback is True
        mocked.assert_called_once()

    def test_5xx_falls_back_to_ocr(self, parser, mocker):
        mocker.patch.object(parser.tesseract, "parse_all_pages", return_value=[])
        with requests_mock.Mocker() as m:
            m.post(PARSE_URL, status_code=503, text="unavailable")
            result = parser.parse("documents/test.pdf", "pdf")
        assert result.parser_name == "tesseract_full"
        assert result.is_fallback is True

    def test_4xx_is_a_defect_not_a_fallback(self, parser):
        """Ошибка в запросе не должна маскироваться под «MinerU недоступен»."""
        with requests_mock.Mocker() as m:
            m.post(PARSE_URL, status_code=422, text="bad params")
            with pytest.raises(ParserFailed, match="422"):
                parser.parse("documents/test.pdf", "pdf")

    def test_empty_text_triggers_ocr_and_keeps_images(self, parser, mocker):
        from core.models.parse_result import ParsedBlock

        mocker.patch.object(
            parser.tesseract, "parse_all_pages",
            return_value=[ParsedBlock(type="text", text="Текст со скана", page=1,
                                      bbox=[0, 0, 1, 1], is_fallback=True)],
        )
        payload = {
            "results": {
                "test.pdf": {
                    "content_list": json.dumps(
                        [{"type": "image", "img_path": "images/fig1.jpg", "page_idx": 0}]
                    ),
                    "images": {"images/fig1.jpg": PNG_1PX},
                }
            }
        }
        with requests_mock.Mocker() as m:
            m.post(PARSE_URL, json=payload)
            result = parser.parse("documents/test.pdf", "pdf")

        assert any(b.type == "text" and b.is_fallback for b in result.blocks)
        assert any(b.type == "image" for b in result.blocks)


class TestSourceFileIsReadOnce:
    def test_pdf_is_opened_once_per_parse(self, parser, mocker):
        """
        Текстовый слой и поиск чертёжных листов спрашивают один и тот же
        документ. Раньше каждый открывал файл сам, и альбом чертежей
        пролистывался дважды за один разбор.
        """
        opened = mocker.patch.object(mineru_parser, "open_pdf", return_value=None)
        with requests_mock.Mocker() as m:
            m.post(PARSE_URL, json=middle_json_response())
            parser.parse("documents/test.pdf", "pdf")
        assert opened.call_count == 1

    def test_not_a_pdf_is_not_opened_at_all(self, parser, mocker):
        opened = mocker.patch.object(mineru_parser, "open_pdf", return_value=None)
        with requests_mock.Mocker() as m:
            m.post(PARSE_URL, json=middle_json_response())
            parser.parse("documents/test.pdf", "png")
        opened.assert_not_called()


class TestTableHtmlParser:
    def test_headers_and_rows(self):
        parsed = parse_table_html(
            "<table><tr><th>A</th><th>B</th></tr><tr><td>1</td><td>2</td></tr></table>"
        )
        assert parsed == {"headers": ["A", "B"], "rows": [["1", "2"]], "total_row": None}

    def test_without_th_first_row_is_header(self):
        parsed = parse_table_html("<table><tr><td>A</td></tr><tr><td>1</td></tr></table>")
        assert parsed["headers"] == ["A"]
        assert parsed["rows"] == [["1"]]

    def test_ragged_rows_padded(self):
        parsed = parse_table_html(
            "<table><tr><th>A</th><th>B</th></tr><tr><td>1</td></tr></table>"
        )
        assert parsed["rows"] == [["1", ""]]

    def test_garbage_returns_none(self):
        assert parse_table_html("") is None
        assert parse_table_html("просто текст") is None


class TestOcrLayerForRaster:
    """
    У растра текстового слоя в файле нет, но его можно собрать
    распознаванием. Схема та же, что у вектора: структура от MinerU, текст
    со стороны. Без этого русская проза в ячейках таблиц так и остаётся
    побуквенной LaTeX-разметкой.
    """

    PROSE = (
        r"\mathsf { v } [ \mathsf { M } / \mathsf { c } ] - \mathsf { c } "
        r"\mathsf { K } \mathsf { O } \mathsf { p } \mathsf { o } \mathsf { c } "
        r"\mathsf { T } \mathsf { b }"
    )

    def _payload(self):
        return {"results": {"formulas.png": {"middle_json": json.dumps({"pdf_info": [{
            "page_idx": 0,
            "page_size": [600, 800],
            "para_blocks": [{
                "type": "table",
                "bbox": [12, 40, 588, 760],
                "blocks": [{"lines": [{"spans": [{
                    "html": "<table><tr><th>Формула</th><th>Величины</th></tr>"
                            "<tr><td>x = x_0 + vt</td><td>" + self.PROSE + "</td></tr>"
                            "<tr><td>v = wR</td><td>" + self.PROSE + "</td></tr></table>"
                }]}]}],
            }],
        }]})}}}

    def _layer(self):
        from core.providers.text_layer import TextLayer, TextLine

        def line(text, x0, y0):
            return TextLine(text=text, bbox=[x0, y0, x0 + 0.3, y0 + 0.02], page=1,
                            confidence=0.88)

        lines = [
            line("Формула", 0.05, 0.08), line("Величины", 0.45, 0.08),
            line("x = x0 + vt", 0.05, 0.35), line("Прямолинейное движение", 0.45, 0.35),
            line("v = wR", 0.05, 0.70), line("Движение по окружности", 0.45, 0.70),
        ]
        return TextLayer(pages={1: lines}, page_count=1)

    def test_prose_cells_become_readable(self, parser, mocker, temp_storage):
        temp_storage.write_file("documents/formulas.png", b"\x89PNG fake")
        mocker.patch.object(parser.tesseract, "text_layer", return_value=self._layer())
        with requests_mock.Mocker() as m:
            m.post(PARSE_URL, json=self._payload())
            result = parser.parse("documents/formulas.png", "png")

        table = next(b for b in result.blocks if b.type == "table")
        cells = [c for row in table.table_data["rows"] for c in row]
        assert not any("mathsf" in c for c in cells)
        assert any("Прямолинейное движение" in c for c in cells)
        assert result.text_layer_stats["layer"] == "ocr"
        assert result.text_layer_stats["repaired_tables"] == 1

    def test_source_kind_tells_the_truth_about_a_scan(self, parser, mocker, temp_storage):
        """
        Скан без текстового слоя — не векторный PDF. Раньше `source_kind`
        ставился по расширению файла ещё до того, как выяснялось, что слоя
        в файле нет, и в провенансе фрагментов скана стояло `vector_pdf`.
        """
        temp_storage.write_file("documents/scan.pdf", b"%PDF-1.4 fake")
        mocker.patch.object(parser, "_read_pdf", return_value=(None, {}))
        mocker.patch.object(parser.tesseract, "text_layer", return_value=self._layer())
        payload = {"results": {"scan.pdf": self._payload()["results"]["formulas.png"]}}
        with requests_mock.Mocker() as m:
            m.post(PARSE_URL, json=payload)
            result = parser.parse("documents/scan.pdf", "pdf")

        assert result.source_kind == "scanned_pdf"

    def test_source_kind_of_an_image_stays_an_image(self, parser, mocker, temp_storage):
        temp_storage.write_file("documents/formulas.png", b"\x89PNG fake")
        mocker.patch.object(parser.tesseract, "text_layer", return_value=self._layer())
        with requests_mock.Mocker() as m:
            m.post(PARSE_URL, json=self._payload())
            result = parser.parse("documents/formulas.png", "png")
        assert result.source_kind == "image"

    def test_missing_ocr_layer_changes_nothing(self, parser, mocker, temp_storage):
        """Слой не обязателен: без него разбор остаётся прежним."""
        temp_storage.write_file("documents/formulas.png", b"\x89PNG fake")
        mocker.patch.object(parser.tesseract, "text_layer", return_value=None)
        with requests_mock.Mocker() as m:
            m.post(PARSE_URL, json=self._payload())
            result = parser.parse("documents/formulas.png", "png")
        table = next(b for b in result.blocks if b.type == "table")
        assert any("mathsf" in c for row in table.table_data["rows"] for c in row)

    def test_ocr_failure_is_not_fatal(self, parser, mocker, temp_storage):
        from core.providers.tesseract_fallback import OcrUnavailable

        temp_storage.write_file("documents/formulas.png", b"\x89PNG fake")
        mocker.patch.object(parser.tesseract, "text_layer", side_effect=OcrUnavailable("нет"))
        with requests_mock.Mocker() as m:
            m.post(PARSE_URL, json=self._payload())
            result = parser.parse("documents/formulas.png", "png")
        assert result.blocks

    def test_unavailable_mineru_is_reported_as_degradation(self, parser, mocker):
        """Лежащий сервис не должен быть неотличим от плохого документа."""
        mocker.patch.object(parser.tesseract, "parse_all_pages", return_value=[])
        with requests_mock.Mocker() as m:
            m.post(PARSE_URL, status_code=503, text="unavailable")
            result = parser.parse("documents/test.pdf", "pdf")
        assert any("mineru_unavailable" in note for note in result.degraded)

