"""
Склеенные числа в ячейке таблицы (TABLE_SPLIT_GLUED_NUMBERS).

Модель таблиц отдаёт многострочную ячейку с номерами одним числом: список
заводских номеров партии приходит как «4585445855…». Строки слоя внутри рамки
той же таблицы знают каждый номер отдельной строкой; по ним число и режется.
Доказательство — только строки самой таблицы, куски — из разных строк, а
похожие на счёт (20 цифр) и дату (ДДММГГГГ) числа не режутся. Данные
синтетические.
"""

import time

import pytest

from core.config import settings
from core.models.parse_result import ParsedBlock
from core.providers.text_layer import (
    OCR_LAYER,
    TextLayer,
    TextLine,
    layer_vocabulary,
    reconcile_with_text_layer,
    split_glued_number,
    split_glued_text,
)

SERIALS = ["45854", "45855", "45856", "45857", "45858"]


def line(text, y, x=0.5, width=0.1, height=0.02, page=1):
    return TextLine(text=text, page=page, bbox=[x, y, x + width, y + height],
                    confidence=0.9)


def lines_of(*texts):
    return [line(text, 0.1 + i * 0.03) for i, text in enumerate(texts)]


@pytest.fixture
def enabled(monkeypatch):
    monkeypatch.setattr(settings, "TABLE_SPLIT_GLUED_NUMBERS", True)


class TestSplitGluedNumber:
    def test_serials_on_separate_lines_are_split(self):
        assert split_glued_number("".join(SERIALS), lines_of(*SERIALS)) == SERIALS

    def test_pieces_must_come_from_different_lines(self):
        six = ["458541", "458542", "458543"]
        assert split_glued_number("".join(six), lines_of(" ".join(six))) is None
        assert split_glued_number("458541458541458541", lines_of("458541")) is None

    def test_number_present_in_layer_even_with_stray_space_is_kept(self):
        assert split_glued_number("".join(SERIALS), lines_of(" ".join(SERIALS[:2]) + "x",
                                                             "".join(SERIALS[:3]) + " " + "".join(SERIALS[3:]),
                                                             *SERIALS)) is None

    def test_account_like_twenty_digits_never_split(self):
        assert split_glued_number("40702810912345678901",
                                  lines_of("4070281091", "2345678901")) is None

    def test_date_like_eight_digits_never_split(self):
        assert split_glued_number("16112026", lines_of("1611", "2026")) is None
        assert split_glued_number("20261116", lines_of("2026", "1116")) is None

    @pytest.mark.parametrize("core, parts", [
        ("1027700132195", ["1027700", "132195"]),     # ОГРН, 13 цифр
        ("770912345601", ["77091234", "5601"]),       # ИИН/ИНН, 12 цифр
        ("87272345678", ["8727", "2345678"]),         # телефон, 11 цифр
        ("045678123", ["0456", "78123"]),             # БИК, 9 цифр
        ("312345678901234", ["31234567", "8901234"]),  # ОГРНИП, 15 цифр
    ])
    def test_requisites_wrapped_in_narrow_cell_are_kept(self, core, parts):
        assert split_glued_number(core, lines_of(*parts)) is None

    def test_wrapped_long_number_is_not_a_stack(self):
        # Перенос: продолжение с того же края, первая строка заполняет ячейку.
        first = line("12345678", 0.10, x=0.5, width=0.12)
        rest = line("1234", 0.13, x=0.5, width=0.06)
        assert split_glued_number("123456781234", [first, rest]) is None
        stack = [line(n, 0.10 + i * 0.03, x=0.5, width=0.06) for i, n in enumerate(
            ["4581", "4582", "4583", "4584", "4585"])]
        assert split_glued_number("45814582458345844585", stack) is None  # 20 цифр — счёт
        stack = stack[:3] + [line("4586", 0.22, x=0.5, width=0.06), line("4587", 0.25, x=0.5, width=0.06),
                             line("4588", 0.28, x=0.5, width=0.06)]
        assert split_glued_number("458145824583458645874588", stack) == [
            "4581", "4582", "4583", "4586", "4587", "4588"]

    def test_rows_must_not_overlap_vertically(self):
        same_row = [line(n, 0.10, x=0.1 + i * 0.15, width=0.1) for i, n in enumerate(SERIALS)]
        assert split_glued_number("".join(SERIALS), same_row) is None

    def test_two_long_halves_are_not_split(self):
        assert split_glued_number("12345671234567", lines_of("1234567", "1234567")) is None

    def test_piece_missing_from_layer_keeps_number(self):
        assert split_glued_number("".join(SERIALS), lines_of(*SERIALS[:-1])) is None

    def test_short_layer_numbers_do_not_cut(self):
        assert split_glued_number("25300000000", lines_of("253", "0000", "25", "300")) is None

    def test_long_run_is_fast(self):
        # Ревью: перебор разбиений без памяти зависал на длинной серии цифр.
        lines = lines_of(*["1111", "11111", "111111", "1111111"] * 3)
        started = time.perf_counter()
        for size in (61, 120, 200):
            split_glued_number("1" * size, lines)
        assert time.perf_counter() - started < 0.5


class TestScope:
    def test_prose_outside_tables_is_never_split(self, enabled):
        # Словарь всей страницы больше не доказывает границу внутри числа.
        vocab = layer_vocabulary(lines_of(*SERIALS))
        glued = "".join(SERIALS)
        assert split_glued_text(glued, vocab)[0] == glued


class TestReconcile:
    def _table(self, cell):
        return ParsedBlock(
            type="table", page=1, bbox=[0.05, 0.05, 0.95, 0.60],
            table_data={"headers": ["№", "Заводские номера"],
                        "rows": [["8", cell]], "total_row": None},
            table_html=f"<table><tr><td>8</td><td>{cell}</td></tr></table>",
            text=f"<table><tr><td>8</td><td>{cell}</td></tr></table>",
        )

    def _layer(self, serials=SERIALS):
        lines = [line("№ Заводские номера", 0.08, x=0.1, width=0.5)]
        lines += [line(serial, 0.12 + i * 0.03) for i, serial in enumerate(serials)]
        return TextLayer(pages={1: lines}, page_count=1)

    def test_scan_table_cell_is_split_in_data_and_html(self, enabled):
        glued = "".join(SERIALS)
        blocks, stats = reconcile_with_text_layer([self._table(glued)], self._layer(), source=OCR_LAYER)
        table = next(b for b in blocks if b.type == "table")
        assert table.table_data["rows"][0][1] == " ".join(SERIALS)
        assert " ".join(SERIALS) in table.table_html
        assert glued not in table.text
        assert stats.split_tokens >= 1

    def test_vector_table_cell_is_split(self, enabled):
        glued = "".join(SERIALS)
        blocks, _ = reconcile_with_text_layer([self._table(glued)], self._layer())
        table = next(b for b in blocks if b.type == "table")
        assert table.table_data["rows"][0][1] == " ".join(SERIALS)

    def test_number_outside_table_frame_does_not_prove_split(self, enabled):
        # Номера стоят на странице, но вне рамки таблицы: доказательства нет.
        glued = "".join(SERIALS)
        far = [line(serial, 0.80 + i * 0.03) for i, serial in enumerate(SERIALS)]
        layer = TextLayer(pages={1: [line("№ Заводские номера", 0.08, x=0.1, width=0.5)] + far},
                          page_count=1)
        blocks, _ = reconcile_with_text_layer([self._table(glued)], layer, source=OCR_LAYER)
        table = next(b for b in blocks if b.type == "table")
        assert table.table_data["rows"][0][1] == glued

    def test_flag_off_keeps_cell(self):
        assert settings.TABLE_SPLIT_GLUED_NUMBERS is False
        glued = "".join(SERIALS)
        blocks, _ = reconcile_with_text_layer([self._table(glued)], self._layer(), source=OCR_LAYER)
        table = next(b for b in blocks if b.type == "table")
        assert table.table_data["rows"][0][1] == glued
