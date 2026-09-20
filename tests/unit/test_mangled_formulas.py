"""
Русский текст, который модель формул MinerU отдаёт LaTeX-разметкой.

Случай из выдачи по `scanned_inspection_report_ocr_only.pdf`: скан акта без
текстового слоя. Из 312 фрагментов 302 приехали типом `formula`, и в них
лежит не формула, а заголовок листа: `\\mathsf { A K T T E X H M 4 E C K O
P O O C M O T P A }`. Собрать строку обратно из разметки нельзя — буквы
потеряны при распознавании. Зато лист прочитан построчно, и текст можно
взять оттуда, ровно как для ячеек таблиц.

Строки в тестах взяты из настоящего ответа парсера, а не придуманы.
"""

import pytest

from core.models.parse_result import ParsedBlock
from core.providers import latex_text
from core.providers.text_layer import (
    OCR_LAYER,
    TextLayer,
    TextLine,
    reconcile_with_text_layer,
)

# Проза под разметкой: заголовки и строки таблицы значений со скана.
MANGLED = [
    r"\mathsf { A K T T E X H M 4 E C K O P O O C M O T P A } / \mathsf { C T P A H } M L \vert \mathsf { A 1 }",
    r"\mathsf { O G b e K T : K C - 1 7 / A r p e r a T C M P - 0 0 1 }",
    r"0 1 ~ { \mathsf { C H } } { \mathsf { - } } 0 1 { \mathsf { - } } 0 1 ~ { \mathsf { p a r a m e t e r } } { \mathsf { = } } { \mathsf { C } } \qquad { \mathsf { v a l u e } } { \mathsf { = } } ~ 1 2 . 2 1 9",
    r"\begin{array} { r l } { 0 3 \textsf { C H - 0 1 - 0 3 } \textsf { p a r a m e t e r } = \textsf { N m 3 / h } \textsf { v a l u e } = } & { { } 8 . 8 8 6 } \end{array}",
    r"3 \mathsf { a } \mathsf { K } \mathsf { I } \mathsf { I } \mathsf { O } \mathsf { \Psi } \mathsf { e } \mathsf { H } \mathsf { M } \mathsf { e } \colon \mathsf { B } \mathsf { I } 3 \mathsf { y } \mathsf { a } \mathsf { J }",
    r"N S - 2 0 2 6 \mathrm { - } 0 0 0 1 / \mathsf { c h e c k s u m } = S \mathsf { Y N T H - } 0 1",
]

# Та же строка, но разметки в ней нет вовсе — только цепочка одиночных
# букв. До правки она проходила насквозь: цепочка засчитывалась лишь
# вместе с переключением гарнитуры.
MANGLED_WITHOUT_FONTS = (
    r"A K T T E X H M \in C K O \Gamma 0 O C M O T P A / C T P A H M L A 2"
)

# Настоящие формулы из выдачи по `formulas.png` — их трогать нельзя.
REAL_FORMULAS = [
    r"\begin{array} { r c l } { x } & { = } & { \sigma ( y - x ) } \\ { y } & { = } & { \rho x - y - x z } \end{array}",
    r"\left( \sum _ { k = 1 } ^ { n } a _ { k } b _ { k } \right) ^ { 2 } \leq \left( \sum _ { k = 1 } ^ { n } a _ { k } ^ { 2 } \right)",
    r"\mathbf { V } _ { 1 } \times \mathbf { V } _ { 2 } = \left| { \frac { \partial X } { \partial u } } \right|",
    r"P ( E ) = { \binom { n } { k } } p ^ { k } ( 1 - p ) ^ { n - k }",
    r"{ \frac { 1 } { ( { \sqrt { \phi { \sqrt { 5 } } } } - \phi ) e ^ { \frac { 2 } { 5 } \pi } } } = 1 + { \frac { e ^ { - 2 \pi } } { 1 } }",
]


class TestDetectorOnRealOutput:
    @pytest.mark.parametrize("latex", MANGLED)
    def test_prose_under_markup_is_recognised(self, latex):
        assert latex_text.is_mangled_text(latex) is True

    def test_prose_without_font_switches_is_recognised(self):
        """Цепочка одиночных букв — признак сама по себе."""
        assert latex_text.font_switches(MANGLED_WITHOUT_FONTS) == 0
        assert latex_text.is_mangled_text(MANGLED_WITHOUT_FONTS) is True

    @pytest.mark.parametrize("latex", REAL_FORMULAS)
    def test_real_formulas_are_not_touched(self, latex):
        assert latex_text.is_mangled_text(latex) is False


def formula_block(latex, bbox=(0.04, 0.03, 0.42, 0.06)):
    return ParsedBlock(
        type="formula", text=latex, page=1, order=1, bbox=list(bbox),
        confidence=0.547, method="mineru_formula",
    )


def line(text, bbox=(0.05, 0.035, 0.40, 0.052), confidence=0.82):
    return TextLine(text=text, bbox=list(bbox), page=1, confidence=confidence)


class TestFormulaBranch:
    def test_prose_is_replaced_with_recognised_text(self):
        """
        Текст берётся со стороны — из построчного распознавания того же
        куска листа. Блок перестаёт быть формулой: он ею и не был.
        """
        block = formula_block(MANGLED[0])
        layer = TextLayer(
            pages={1: [line("АКТ ТЕХНИЧЕСКОГО ОСМОТРА / СТРАНИЦА 1")]}, page_count=1
        )

        blocks, stats = reconcile_with_text_layer([block], layer, source=OCR_LAYER)

        assert stats.mangled_formulas == 1
        assert stats.repaired_formulas == 1
        assert blocks[0].type == "text"
        assert blocks[0].text == "АКТ ТЕХНИЧЕСКОГО ОСМОТРА / СТРАНИЦА 1"
        assert blocks[0].method == "ocr_layer"
        assert blocks[0].confidence == pytest.approx(0.82, abs=0.01)

    def test_real_formula_keeps_its_latex(self):
        """Строки, попавшие в рамку настоящей формулы, её не переписывают."""
        block = formula_block(REAL_FORMULAS[3])
        layer = TextLayer(pages={1: [line("P(E) = C(n,k) p^k")]}, page_count=1)

        blocks, stats = reconcile_with_text_layer([block], layer, source=OCR_LAYER)

        assert stats.mangled_formulas == 0
        assert blocks[0].type == "formula"
        assert blocks[0].text == REAL_FORMULAS[3]

    def test_nothing_to_replace_with_leaves_a_flag(self):
        """
        Строк распознавания в рамке нет. Разметка снимается — голый текст
        хотя бы читается и ищется, — но блок помечается фоллбэком и уезжает
        на проверку человеку.
        """
        block = formula_block(MANGLED[1])
        layer = TextLayer(pages={1: [line("другая строка", bbox=(0.5, 0.5, 0.9, 0.52))]},
                          page_count=1)

        blocks, stats = reconcile_with_text_layer([block], layer, source=OCR_LAYER)

        assert stats.decoded_formulas == 1
        assert blocks[0].type == "text"
        assert blocks[0].method == "latex_decoded"
        assert blocks[0].is_fallback is True
        assert "mathsf" not in blocks[0].text
        assert "\\" not in blocks[0].text


class TestDuplicateTableContinuation:
    """
    Многостраничную таблицу MinerU отдаёт целиком в рамке первой страницы,
    а дальше присылает пустые рамки. Случай из выдачи по
    `procurement_and_acceptance_dossier_2026.pdf`: 73 позиции из 101 уехали
    в индекс дважды — один раз колонками, второй плоским текстом.
    """

    @staticmethod
    def rows(count):
        return [
            [str(n), f"NC-{n:05d}", f"Комплект изделия / модуль / кабельный узел {n:02d}",
             "шт", "158,750", "Визуальный"]
            for n in range(1, count + 1)
        ]

    def make(self):
        parsed = ParsedBlock(
            type="table", page=1, order=1, bbox=[0.05, 0.04, 0.94, 0.94],
            confidence=0.99, method="mineru_table", image_ref="assets/t.jpg",
            table_data={
                "headers": ["Поз.", "Артикул", "Наименование", "Ед.", "Цена", "Контроль"],
                "rows": self.rows(4),
            },
        )
        empty = ParsedBlock(
            type="table", page=2, order=2, bbox=[0.05, 0.04, 0.94, 0.94],
            confidence=0.988, method="mineru_table",
        )
        return parsed, empty

    def test_rows_do_not_arrive_a_second_time_as_text(self):
        parsed, empty = self.make()
        lines = [
            TextLine(
                text=f"3 NC-00003 Комплект изделия / модуль / кабельный узел 03 шт 158,750 Визуальный",
                bbox=[0.06, 0.10 + index * 0.03, 0.93, 0.12 + index * 0.03], page=2,
            )
            for index in range(2)
        ]
        layer = TextLayer(pages={2: lines}, page_count=2)

        blocks, stats = reconcile_with_text_layer([parsed, empty], layer)

        assert stats.duplicate_lines == 2
        assert stats.dropped_empty_tables == 1
        # Пустой рамки без разбора и без картинки в выдаче не осталось.
        assert [b.page for b in blocks] == [1]
        assert not any(b.type == "text" for b in blocks)

    def test_new_rows_still_arrive(self):
        """Гасятся только повторы: строка, которой в таблице нет, остаётся."""
        parsed, empty = self.make()
        lines = [TextLine(
            text="AT-041 Расход SCADA 154.7 PASS Фактическое значение заполняется",
            bbox=[0.06, 0.10, 0.93, 0.12], page=2,
        )]
        layer = TextLayer(pages={2: lines}, page_count=2)

        blocks, stats = reconcile_with_text_layer([parsed, empty], layer)

        assert stats.duplicate_lines == 0
        assert stats.dropped_empty_tables == 0
        assert any(b.type == "text" and "AT-041" in (b.text or "") for b in blocks)
