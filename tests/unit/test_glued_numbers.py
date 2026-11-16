"""
Склеенные числа в ячейке таблицы (TABLE_SPLIT_GLUED_NUMBERS).

Модель таблиц отдаёт многострочную ячейку с номерами одним числом: список
заводских номеров партии приходит как «4585445855…», а в платёжке «р/с»,
«БИК» и «к/с» — как одно число из 49 цифр. Слой той же страницы знает каждый
номер отдельным словом; по нему число и режется. Данные синтетические.
"""

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
ACCOUNT, BIK, CORR = "40702810338290030371", "044525225", "30101810400000000225"


def line(text, y, x=0.1, width=0.5, height=0.02, page=1):
    return TextLine(text=text, page=page, bbox=[x, y, x + width, y + height],
                    confidence=0.9)


@pytest.fixture
def enabled(monkeypatch):
    monkeypatch.setattr(settings, "TABLE_SPLIT_GLUED_NUMBERS", True)


def vocab_of(*texts):
    return layer_vocabulary([line(text, 0.1 + i * 0.03) for i, text in enumerate(texts)])


class TestSplitGluedNumber:
    def test_serials_are_split_by_layer_numbers(self):
        vocab = vocab_of(*SERIALS)
        assert split_glued_number("".join(SERIALS), vocab) == SERIALS

    def test_account_bik_corr_are_split(self):
        vocab = vocab_of(f"Сч. № {ACCOUNT}", f"БИК {BIK}", f"Сч. № {CORR}")
        assert split_glued_number(ACCOUNT + BIK + CORR, vocab) == [ACCOUNT, BIK, CORR]

    def test_number_known_whole_is_not_split(self):
        vocab = vocab_of("4585445855", "45854", "45855")
        assert split_glued_number("4585445855", vocab) is None

    def test_piece_missing_from_layer_keeps_number(self):
        # Последний номер слой не прочитал: доказать границы нечем.
        vocab = vocab_of(*SERIALS[:-1])
        assert split_glued_number("".join(SERIALS), vocab) is None

    def test_short_layer_numbers_do_not_cut(self):
        # Короткие числа слоя (позиции, количества) не режут длинное число.
        vocab = vocab_of("253", "0000", "25", "30000")
        assert split_glued_number("2530000", vocab) is None
        assert split_glued_number("25300000000", vocab) is None

    def test_not_digits_or_short_is_ignored(self):
        vocab = vocab_of(*SERIALS)
        assert split_glued_number("45854А45855", vocab) is None
        assert split_glued_number("4585", vocab) is None

    def test_fewest_pieces_win(self):
        vocab = vocab_of("1234", "5678", "12345678", "9012")
        assert split_glued_number("123456789012", vocab) == ["12345678", "9012"]


class TestFlag:
    def test_off_by_default_number_stays_glued(self):
        assert settings.TABLE_SPLIT_GLUED_NUMBERS is False
        vocab = vocab_of(*SERIALS)
        glued = "".join(SERIALS)
        assert split_glued_text(glued, vocab)[0] == glued

    def test_on_splits_in_text_with_edges(self, enabled):
        vocab = vocab_of(*SERIALS)
        text, count = split_glued_text("(" + "".join(SERIALS) + ")", vocab)
        assert text == "(" + " ".join(SERIALS) + ")"
        assert count == 1


class TestReconcile:
    def _table(self, cell):
        return ParsedBlock(
            type="table", page=1, bbox=[0.05, 0.05, 0.95, 0.60],
            table_data={"headers": ["№", "Заводские номера"],
                        "rows": [["8", cell]], "total_row": None},
            table_html=f"<table><tr><td>8</td><td>{cell}</td></tr></table>",
            text=f"<table><tr><td>8</td><td>{cell}</td></tr></table>",
        )

    def _layer(self):
        lines = [line("№ Заводские номера", 0.08)]
        lines += [line(serial, 0.12 + i * 0.03, x=0.5, width=0.1) for i, serial in enumerate(SERIALS)]
        return TextLayer(pages={1: lines}, page_count=1)

    def test_scan_table_cell_is_split_in_data_and_html(self, enabled):
        glued = "".join(SERIALS)
        blocks, _ = reconcile_with_text_layer([self._table(glued)], self._layer(), source=OCR_LAYER)
        table = next(b for b in blocks if b.type == "table")
        assert table.table_data["rows"][0][1] == " ".join(SERIALS)
        assert " ".join(SERIALS) in table.table_html
        assert glued not in table.text

    def test_vector_table_cell_is_split(self, enabled):
        glued = "".join(SERIALS)
        blocks, _ = reconcile_with_text_layer([self._table(glued)], self._layer())
        table = next(b for b in blocks if b.type == "table")
        assert table.table_data["rows"][0][1] == " ".join(SERIALS)

    def test_flag_off_keeps_cell(self):
        glued = "".join(SERIALS)
        blocks, _ = reconcile_with_text_layer([self._table(glued)], self._layer(), source=OCR_LAYER)
        table = next(b for b in blocks if b.type == "table")
        assert table.table_data["rows"][0][1] == glued
