"""Векторные чертежи: определение листа и разбор полей из текстового слоя."""

import pytest

from core.providers.text_layer import TextLayer, TextLine
from core.providers.vector_drawing import (
    PROVENANCE_VECTOR,
    classify,
    detect_drawing_pages,
    determine_nature,
    extract_drawing_fields,
    extract_tolerance,
    extract_unit,
    structure_ratio,
    tolerance_source,
)


def line(text, x=0.3, y=0.3, page=1, **kwargs):
    return TextLine(text=text, page=page, bbox=[x, y, x + 0.08, y + 0.02], **kwargs)


class TestPageDetection:
    @pytest.fixture
    def pdf_bytes(self):
        pymupdf = pytest.importorskip("pymupdf")
        document = pymupdf.open()

        # Лист 1 — сплошной текст.
        text_page = document.new_page(width=600, height=800)
        for row in range(30):
            text_page.insert_text((60, 60 + row * 20), "plain running text line", fontsize=11)

        # Лист 2 — чертёж: рамка формата, много геометрии, сетка штампа.
        sheet = document.new_page(width=1190, height=842)
        sheet.draw_rect(pymupdf.Rect(20, 20, 1170, 822))
        for i in range(160):
            sheet.draw_line(pymupdf.Point(100 + i, 200), pymupdf.Point(140 + i, 320))
        for row in range(9):
            sheet.draw_line(pymupdf.Point(820, 700 + row * 14),
                            pymupdf.Point(1170, 700 + row * 14))

        data = document.tobytes()
        document.close()
        return data

    def test_text_page_is_not_a_drawing(self, pdf_bytes):
        assert detect_drawing_pages(pdf_bytes)[1].is_drawing is False

    def test_cad_sheet_is_detected(self, pdf_bytes):
        verdict = detect_drawing_pages(pdf_bytes)[2]
        assert verdict.is_drawing is True
        assert verdict.signals["format_frame"] is True
        assert verdict.signals["title_block_lines"] >= 6

    def test_broken_file_does_not_raise(self):
        assert detect_drawing_pages(b"not a pdf") == {}


class TestClassification:
    @pytest.mark.parametrize("text,category", [
        ("Ø20h7", "size"),
        ("R15", "radius"),
        ("M12x1.5", "thread"),
        ("Ra 3.2", "roughness"),
        ("45°", "size"),
        ("120", "size"),
        ("±0,5", "tolerance"),
        ("А-А", "callout"),
        ("Сталь 45 ГОСТ 1050-88", "material"),
        ("Неуказанные предельные отклонения размеров", "note"),
    ])
    def test_category(self, text, category):
        assert classify(text)[0] == category

    def test_title_block_zone_wins(self):
        """Правый нижний угол листа — штамп по ГОСТ 2.104."""
        stamp = line("АБВГ.301261.005", x=0.80, y=0.90)
        assert classify(stamp.text, stamp)[0] == "title_block"

    def test_unclassified_text_is_honest_about_it(self):
        category, confidence = classify("произвольная подпись на поле")
        assert category == "note"
        assert confidence < 0.5


class TestHonestFields:
    @pytest.mark.parametrize("text,category,expected", [
        ("Ra 3.2 мкм", "roughness", "мкм"),
        ("20 мм", "size", "мм"),
        ("45°", "size", "°"),
        ("Ø20", "size", "мм"),                 # линейный размер на чертеже
        ("Технические требования", "note", ""),  # единицы нет — и выдумывать нечего
    ])
    def test_unit_is_not_invented(self, text, category, expected):
        assert extract_unit(text, category) == expected

    @pytest.mark.parametrize("text,nature", [
        ("120*", "reference"),                   # пометка справочного по ГОСТ 2.307
        ("размер для справок", "reference"),
        ("Ø20 h7", "executive"),
        ("* см. п. 3 технических требований", "executive"),  # звёздочка не у числа
    ])
    def test_nature(self, text, nature):
        assert determine_nature(text) == nature

    @pytest.mark.parametrize("text,tolerance,source", [
        ("20±0,1", "±0,1", "explicit"),
        ("Ø20h7", "h7", "fit_notation"),
        ("Ø20", None, "unknown"),
    ])
    def test_tolerance(self, text, tolerance, source):
        assert extract_tolerance(text) == tolerance
        assert tolerance_source(text) == source


class TestFieldExtraction:
    @pytest.fixture
    def layer(self):
        lines = [
            line("Ø20h7", x=0.30, y=0.30),
            line("R15", x=0.45, y=0.40),
            # Размер подписан повёрнутым текстом — в растре это и теряется.
            line("85±0,2", x=0.20, y=0.55, rotation=90.0),
            line("Ra 6.3", x=0.60, y=0.25),
            line("Сталь 45", x=0.82, y=0.88),
            line("произвольное примечание", x=0.15, y=0.70),
        ]
        return TextLayer(pages={1: lines}, page_count=1)

    def test_every_line_becomes_a_field(self, layer):
        assert len(extract_drawing_fields(layer, 1)) == 6

    def test_values_are_taken_verbatim(self, layer):
        values = {f["value"] for f in extract_drawing_fields(layer, 1)}
        assert "85±0,2" in values
        assert "Ø20h7" in values

    def test_rotated_text_is_not_lost(self, layer):
        rotated = [f for f in extract_drawing_fields(layer, 1) if f["value"] == "85±0,2"]
        assert rotated and rotated[0]["tolerance"] == "±0,2"

    def test_provenance_says_where_it_came_from(self, layer):
        assert all(f["provenance"] == PROVENANCE_VECTOR
                   for f in extract_drawing_fields(layer, 1))

    def test_structure_ratio_counts_only_classified(self, layer):
        fields = extract_drawing_fields(layer, 1)
        # Пять полей разобраны, одно осталось примечанием.
        assert structure_ratio(fields) == pytest.approx(5 / 6, abs=0.01)

    def test_no_fields_means_zero_ratio(self):
        assert structure_ratio([]) == 0.0
