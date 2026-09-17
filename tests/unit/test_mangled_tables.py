"""
Русский текст, который модель таблиц MinerU отдаёт LaTeX-разметкой.

Случай из выдачи по `physical_formulas.png`: таблица «формула | физические
величины». Колонка формул разобрана верно, колонка описаний превращена в
побуквенную разметку вида `\mathsf { c } \mathsf { K } \mathsf { O } ...`,
из которой исходную строку не восстановить. Здесь проверяется вся цепочка:
узнать такую ячейку, заменить её текстом построчного распознавания, а если
геометрия не сошлась — разобрать таблицу на текст и формулы, но не отдавать
разметку в индекс.
"""

import pytest

from core.models.parse_result import ParsedBlock
from core.providers import latex_text, table_from_lines
from core.providers.text_layer import (
    OCR_LAYER,
    VECTOR_LAYER,
    TextLayer,
    TextLine,
    reconcile_with_text_layer,
)

# Строки взяты из настоящего ответа парсера, а не придуманы.
FORMULA_SIMPLE = r"\boldsymbol { x } = \boldsymbol { x } _ { o } + \boldsymbol { v } t"
FORMULA_GLUED = (
    r"\scriptstyle a = { \frac { v - v _ { o } } { t } }"
    r"S = v _ { o } t + \frac { a t ^ { 2 } } { 2 } = \frac { v ^ { 2 } - v _ { o } ^ { 2 } } { 2 a }"
)
PROSE = (
    r"\mathsf { v } [ \mathsf { M } / \mathsf { c } ] - \mathsf { c } \mathsf { K } "
    r"\mathsf { O } \mathsf { p } \mathsf { o } \mathsf { c } \mathsf { T } \mathsf { b }"
)
PROSE_RUN = r"\mathsf { X } _ { 0 } \left[ \mathsf { M } \right] - \mathsf { K O O p a m H a r c a }"


class TestDetector:
    @pytest.mark.parametrize("latex", [FORMULA_SIMPLE, FORMULA_GLUED, "x^2 + y^2 = z^2", ""])
    def test_real_formulas_are_not_touched(self, latex):
        """
        `\\boldsymbol` в формуле помечает вектор, и три вектора в строке —
        это всё ещё формула. Признаком прозы служит только переключение
        текстовой гарнитуры на каждой букве.
        """
        assert latex_text.is_mangled_text(latex) is False

    @pytest.mark.parametrize("latex", [PROSE, PROSE_RUN])
    def test_prose_under_markup_is_recognised(self, latex):
        assert latex_text.is_mangled_text(latex) is True

    def test_plain_text_is_not_latex(self):
        assert latex_text.is_mangled_text("Прямолинейное равномерное движение") is False

    def test_decode_strips_markup(self):
        assert latex_text.decode(FORMULA_SIMPLE) == "x = x _ o + v t"
        assert "mathsf" not in latex_text.decode(PROSE)
        # Одиночные буквы, разнесённые разметкой, склеиваются обратно.
        assert "KOOpamHarca" in latex_text.decode(PROSE_RUN)

    def test_mangled_cells_lists_only_prose(self):
        table = {"headers": ["Формула", PROSE], "rows": [[FORMULA_SIMPLE, PROSE_RUN]]}
        assert latex_text.mangled_cells(table) == [PROSE, PROSE_RUN]


def line(text, x0, y0, confidence=0.9):
    return TextLine(text=text, bbox=[x0, y0, x0 + 0.3, y0 + 0.02], page=1,
                    confidence=confidence)


def table_block(rows_of_prose=3):
    rows = [[FORMULA_SIMPLE, PROSE] for _ in range(rows_of_prose)]
    return ParsedBlock(
        type="table", page=1, order=0, bbox=[0.02, 0.05, 0.98, 0.95],
        confidence=0.96, method="mineru_table", image_ref="assets/table.jpg",
        table_data={"headers": ["Формула", "Физические величины"], "rows": rows},
    )


def ocr_lines(rows=3, with_header=True):
    """Левая колонка — формулы, правая — русские описания."""
    lines = []
    y = 0.08
    if with_header:
        lines += [line("Формула", 0.05, y), line("Физические величины", 0.45, y)]
        y += 0.15
    texts = [
        ("x = x0 + vt", ["Прямолинейное равномерное движение", "x [м] – координата"]),
        ("a = (v - v0)/t", ["Прямолинейное равноускоренное движение", "a [м/с2] – ускорение"]),
        ("v = wR", ["Движение по окружности", "v [м/с] – линейная скорость"]),
    ][:rows]
    for formula, description in texts:
        lines.append(line(formula, 0.05, y))
        for index, text in enumerate(description):
            lines.append(line(text, 0.45, y + index * 0.025))
        y += 0.25
    return lines


class TestRepairByGeometry:
    def test_prose_cells_are_replaced_with_recognised_text(self):
        block = table_block()
        layer = TextLayer(pages={1: ocr_lines()}, page_count=1)

        blocks, stats = reconcile_with_text_layer([block], layer, source=OCR_LAYER)

        assert stats.mangled_tables == 1
        assert stats.repaired_tables == 1
        assert stats.degraded_tables == 0
        table = blocks[0].table_data
        # Формулы остались формулами — LaTeX в них настоящий.
        assert table["rows"][0][0] == FORMULA_SIMPLE
        # А проза стала прозой.
        assert "Прямолинейное равномерное движение" in table["rows"][0][1]
        assert "mathsf" not in table["rows"][1][1]

    def test_nothing_happens_to_a_healthy_table(self):
        block = ParsedBlock(
            type="table", page=1, bbox=[0.02, 0.05, 0.98, 0.95],
            table_data={"headers": ["Параметр"], "rows": [["Ra"]]},
        )
        layer = TextLayer(pages={1: [line("Параметр", 0.05, 0.1)]}, page_count=1)
        blocks, stats = reconcile_with_text_layer([block], layer, source=OCR_LAYER)
        assert stats.mangled_tables == 0
        assert blocks[0].table_data["headers"] == ["Параметр"]


class TestDegradation:
    def test_table_falls_apart_when_geometry_does_not_match(self):
        """
        Строк распознавания меньше, чем строк таблицы, — сопоставить нечем.
        Тогда честнее отдать содержимое текстом и формулами, чем заполнить
        ячейки наугад или оставить в индексе побуквенную разметку.
        """
        block = table_block()
        layer = TextLayer(pages={1: ocr_lines(rows=1, with_header=False)}, page_count=1)

        blocks, stats = reconcile_with_text_layer([block], layer, source=OCR_LAYER)

        assert stats.degraded_tables == 1
        kinds = [b.type for b in blocks]
        assert "formula" in kinds and "text" in kinds
        table = next(b for b in blocks if b.type == "table")
        # Разбора нет — но картинка области осталась, и полнота честно нулевая.
        assert table.table_data is None
        assert table.image_ref == "assets/table.jpg"
        assert not any(
            "mathsf" in (b.text or "") for b in blocks if b.type == "text"
        )

    def test_formulas_survive_degradation(self):
        block = table_block()
        layer = TextLayer(pages={1: ocr_lines(rows=1, with_header=False)}, page_count=1)
        blocks, _ = reconcile_with_text_layer([block], layer, source=OCR_LAYER)
        formulas = [b.text for b in blocks if b.type == "formula"]
        assert FORMULA_SIMPLE in formulas


class TestLayerSource:
    def test_ocr_layer_does_not_pretend_to_be_the_file(self):
        """
        Текст из слоя векторного PDF взят из файла — уверенность 1.0 честна.
        Текст, собранный распознаванием, — нет, и выдавать его за точный
        нельзя: на этой уверенности стоит выбор уровня лестницы.
        """
        block = ParsedBlock(type="text", text="старое", page=1, bbox=[0.0, 0.05, 0.9, 0.12])
        lines = [line("Прямолинейное равномерное движение", 0.05, 0.06, confidence=0.6)]
        layer = TextLayer(pages={1: lines}, page_count=1)

        blocks, _ = reconcile_with_text_layer([block], layer, source=OCR_LAYER)
        assert blocks[0].method == "ocr_layer"
        assert blocks[0].confidence == pytest.approx(0.6, abs=0.01)

    def test_vector_layer_keeps_its_certainty(self):
        block = ParsedBlock(type="text", text="старое", page=1, bbox=[0.0, 0.05, 0.9, 0.12])
        lines = [line("Прямолинейное равномерное движение", 0.05, 0.06, confidence=0.6)]
        layer = TextLayer(pages={1: lines}, page_count=1)

        blocks, _ = reconcile_with_text_layer([block], layer, source=VECTOR_LAYER)
        assert blocks[0].method == "text_layer"
        assert blocks[0].confidence == 1.0


class TestColumnsAndRows:
    def test_columns_need_a_real_gap(self):
        """Неровный левый край — это одна колонка, а не две."""
        lines = [line("а", 0.10, 0.1), line("б", 0.105, 0.2), line("в", 0.11, 0.3)]
        assert table_from_lines._column_bands(lines, 2) is None

    def test_rows_need_a_real_gap(self):
        """Строки, идущие вплотную, не образуют разных строк таблицы."""
        lines = [line("а", 0.1, 0.10), line("б", 0.1, 0.121), line("в", 0.1, 0.142)]
        assert table_from_lines._row_bands(lines, 3) is None

    def test_single_column_is_taken_as_is(self):
        lines = [line("а", 0.1, 0.1)]
        assert table_from_lines._column_bands(lines, 1) == [lines]
