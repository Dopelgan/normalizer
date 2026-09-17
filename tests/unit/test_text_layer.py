"""Текстовый слой: извлечение, сшивка с рамками парсера, починка ячеек."""

import pytest

from core.models.parse_result import ParsedBlock
from core.providers.text_layer import (
    METHOD_PARSER_OCR,
    METHOD_TEXT_LAYER,
    METHOD_TEXT_LAYER_RECOVERED,
    PYMUPDF_AVAILABLE,
    TextLayer,
    TextLine,
    extract_text_layer,
    fold_homoglyphs,
    group_lines,
    lines_to_text,
    reconcile_with_text_layer,
)


def line(text, y, x=0.1, width=0.5, height=0.02, page=1, **kwargs):
    return TextLine(text=text, page=page,
                    bbox=[x, y, x + width, y + height], **kwargs)


def layer_of(*lines, page_count=1):
    pages = {}
    for item in lines:
        pages.setdefault(item.page, []).append(item)
    return TextLayer(pages=pages, page_count=page_count)


class TestExtraction:
    @pytest.fixture
    def pdf_bytes(self):
        pymupdf = pytest.importorskip("pymupdf")
        document = pymupdf.open()
        page = document.new_page(width=600, height=800)
        page.insert_text((60, 100), "Normal black body text", fontsize=12)
        # Мелкий светло-серый текст — ровно то, что теряет распознавание.
        page.insert_text((60, 140), "tiny grey caption", fontsize=6,
                         color=(0.72, 0.72, 0.72))
        page.insert_text((60, 300), "rotated dimension", fontsize=10, rotate=90)
        data = document.tobytes()
        document.close()
        return data

    def test_gray_and_small_text_is_not_lost(self, pdf_bytes):
        layer = extract_text_layer(pdf_bytes)
        texts = [l.text for l in layer.lines(1)]
        assert "tiny grey caption" in texts
        assert "Normal black body text" in texts

    def test_rotation_is_preserved(self, pdf_bytes):
        layer = extract_text_layer(pdf_bytes)
        rotated = [l for l in layer.lines(1) if l.is_rotated]
        assert len(rotated) == 1
        assert rotated[0].text == "rotated dimension"

    def test_bbox_is_normalized(self, pdf_bytes):
        layer = extract_text_layer(pdf_bytes)
        for item in layer.lines(1):
            assert all(0.0 <= c <= 1.0 for c in item.bbox)

    def test_broken_file_does_not_raise(self):
        assert extract_text_layer(b"not a pdf at all") is None


class TestUsability:
    def test_empty_layer_is_not_usable(self):
        assert layer_of(page_count=3).is_usable() is False

    def test_layer_with_text_on_most_pages_is_usable(self):
        layer = layer_of(
            line("Достаточно длинный текст первой страницы", 0.1, page=1),
            line("Достаточно длинный текст второй страницы", 0.1, page=2),
            page_count=2,
        )
        assert layer.is_usable() is True


class TestGrouping:
    def test_close_lines_form_one_paragraph(self):
        groups = group_lines([line("Первая строка", 0.10), line("Вторая строка", 0.125)])
        assert len(groups) == 1

    def test_distant_lines_split(self):
        groups = group_lines([line("Первая строка", 0.10), line("Далеко внизу", 0.70)])
        assert len(groups) == 2

    def test_monospace_keeps_line_breaks(self):
        code = [line("func main() {", 0.10, mono=True),
                line("    return nil", 0.125, mono=True)]
        assert lines_to_text(code) == "func main() {\n    return nil"

    def test_prose_is_joined_with_spaces(self):
        prose = [line("Начало фразы", 0.10), line("и её продолжение", 0.125)]
        assert lines_to_text(prose) == "Начало фразы и её продолжение"


class TestReconcile:
    def test_block_text_is_replaced_by_layer(self):
        block = ParsedBlock(type="text", text="2000 грs", page=1,
                            bbox=[0.05, 0.08, 0.70, 0.14], method=METHOD_PARSER_OCR)
        layer = layer_of(line("2000 rps", 0.10))

        blocks, stats = reconcile_with_text_layer([block], layer)

        assert blocks[0].text == "2000 rps"
        assert blocks[0].method == METHOD_TEXT_LAYER
        assert stats.rebuilt_blocks == 1

    def test_text_outside_every_frame_is_recovered(self):
        """Серый курсив и код в рамке не попадают в layout-блоки MinerU."""
        block = ParsedBlock(type="text", text="Заголовок", page=1,
                            bbox=[0.05, 0.08, 0.70, 0.14])
        layer = layer_of(line("Заголовок", 0.10),
                         line("Потерянный серый абзац", 0.50))

        blocks, stats = reconcile_with_text_layer([block], layer)

        assert stats.recovered_blocks == 1
        recovered = [b for b in blocks if b.method == METHOD_TEXT_LAYER_RECOVERED]
        assert recovered[0].text == "Потерянный серый абзац"

    def test_page_without_any_block_is_recovered_whole(self):
        """Страница, которую парсер потерял целиком, возвращается из слоя."""
        layer = layer_of(line("Стартовый код", 0.10, page=2),
                         line("package main", 0.20, page=2, mono=True),
                         page_count=2)

        blocks, stats = reconcile_with_text_layer([], layer)

        assert {b.page for b in blocks} == {2}
        assert stats.recovered_blocks == 2

    def test_empty_table_frame_gives_its_text_back(self):
        """MinerU нашёл таблицу, но не разобрал её — текст терять нельзя."""
        block = ParsedBlock(type="table", table_data=None, page=1,
                            bbox=[0.05, 0.05, 0.95, 0.60])
        layer = layer_of(line("Навык Проверяется ли", 0.10))

        blocks, stats = reconcile_with_text_layer([block], layer)

        assert stats.recovered_blocks == 1
        assert any(b.type == "text" and "Навык" in (b.text or "") for b in blocks)

    def test_blocks_are_renumbered_in_reading_order(self):
        upper = ParsedBlock(type="text", text="Верхний", page=1, bbox=[0.05, 0.05, 0.9, 0.12])
        lower = ParsedBlock(type="text", text="Нижний", page=1, bbox=[0.05, 0.60, 0.9, 0.68])
        layer = layer_of(line("Верхний", 0.07), line("Нижний", 0.62))

        blocks, _ = reconcile_with_text_layer([lower, upper], layer)

        assert [b.text for b in blocks] == ["Верхний", "Нижний"]
        assert [b.order for b in blocks] == [0, 1]

    def test_no_duplication_when_parser_has_no_coordinates(self):
        """content_list без middle_json: bbox на весь лист, сопоставлять нечем."""
        block = ParsedBlock(type="text", text="Абзац из распознавания", page=1,
                            bbox=[0.0, 0.0, 1.0, 1.0], method=METHOD_PARSER_OCR)
        layer = layer_of(line("Абзац из распознавания", 0.20))

        blocks, _ = reconcile_with_text_layer([block], layer)

        assert len(blocks) == 1
        assert blocks[0].method == METHOD_TEXT_LAYER_RECOVERED
        assert blocks[0].bbox != [0.0, 0.0, 1.0, 1.0]

    def test_tables_survive_when_text_has_no_coordinates(self):
        table = ParsedBlock(type="table", page=1, bbox=[0.0, 0.0, 1.0, 1.0],
                            table_data={"headers": ["А"], "rows": [["1"]], "total_row": None})
        text = ParsedBlock(type="text", text="Абзац", page=1, bbox=[0.0, 0.0, 1.0, 1.0])
        layer = layer_of(line("Абзац", 0.20))

        blocks, _ = reconcile_with_text_layer([table, text], layer)

        assert [b.type for b in blocks].count("table") == 1

    def test_ocr_text_survives_when_layer_gives_much_less(self):
        """Съехавшая рамка не должна обрезать уже распознанный текст."""
        block = ParsedBlock(type="text", page=1, bbox=[0.05, 0.05, 0.95, 0.95],
                            text="Очень длинный распознанный абзац. " * 5)
        layer = layer_of(line("обрывок", 0.5))

        blocks, stats = reconcile_with_text_layer([block], layer)

        assert blocks[0].text.startswith("Очень длинный")
        assert blocks[0].method == METHOD_PARSER_OCR
        assert stats.rebuilt_blocks == 0


class TestHomoglyphs:
    @pytest.mark.parametrize("cyrillic,latin", [
        ("БД", "БD"), ("через", "чepeз"), ("expires_at", "exрires_at"),
    ])
    def test_confusable_pairs_fold_together(self, cyrillic, latin):
        assert fold_homoglyphs(cyrillic) == fold_homoglyphs(latin)

    def test_different_words_do_not_fold_together(self):
        assert fold_homoglyphs("таблица") != fold_homoglyphs("формула")

    def test_table_cells_are_repaired_from_layer(self):
        block = ParsedBlock(
            type="table", page=1, bbox=[0.05, 0.05, 0.95, 0.40],
            table_data={"headers": ["Критерий", "Признак"],
                        "rows": [["БD", "чepeз DB constraint"]], "total_row": None},
        )
        layer = layer_of(
            line("Критерий", 0.08, x=0.05), line("Признак", 0.08, x=0.50),
            line("БД", 0.15, x=0.05), line("через DB constraint", 0.15, x=0.50),
        )

        blocks, stats = reconcile_with_text_layer([block], layer)

        assert blocks[0].table_data["rows"][0] == ["БД", "через DB constraint"]
        assert stats.repaired_cells == 2

    def test_unrelated_cell_is_left_alone(self):
        block = ParsedBlock(
            type="table", page=1, bbox=[0.05, 0.05, 0.95, 0.40],
            table_data={"headers": [], "rows": [["Совершенно другое значение"]],
                        "total_row": None},
        )
        layer = layer_of(line("Ничего похожего рядом", 0.15))

        blocks, stats = reconcile_with_text_layer([block], layer)

        assert blocks[0].table_data["rows"][0] == ["Совершенно другое значение"]
        assert stats.repaired_cells == 0


@pytest.mark.skipif(PYMUPDF_AVAILABLE, reason="проверяем поведение без PyMuPDF")
def test_missing_pymupdf_degrades_quietly():
    assert extract_text_layer(b"%PDF-1.4") is None
