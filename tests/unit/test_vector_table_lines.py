"""
Строки векторного слоя в рамке таблицы, потерянные моделью таблиц
(TABLE_VECTOR_LINES_KEPT).

Счёт на оплату с текстовым слоем: модель таблиц оставила ячейки «БИК» и
«Сч. №» пустыми, а строки «Итого», «Всего к оплате» и сумму прописью внутри
рамки товарной таблицы не взяла вовсе. У слоя распознавания такие строки
возвращаются текстом, у векторного пропадали. Данные синтетические.
"""

import pytest

from core.config import settings
from core.models.parse_result import ParsedBlock
from core.providers.text_layer import (
    OCR_LAYER,
    TextLayer,
    TextLine,
    reconcile_with_text_layer,
)


def line(text, y, x=0.1, width=0.3, height=0.012, page=1):
    return TextLine(text=text, page=page, bbox=[x, y, x + width, y + height],
                    confidence=1.0)


@pytest.fixture
def enabled(monkeypatch):
    monkeypatch.setattr(settings, "TABLE_VECTOR_LINES_KEPT", True)


def goods_table():
    rows = [["1", "Ресертификационный аудит", "1", "усл", "30 000,00", "30 000,00"]]
    html = "<table><tr>" + "".join(f"<td>{c}</td>" for c in rows[0]) + "</tr></table>"
    return ParsedBlock(
        type="table", page=1, bbox=[0.05, 0.44, 0.87, 0.58],
        table_data={"headers": ["№", "Товары (работы, услуги)", "Кол-во", "Ед.", "Цена", "Сумма"],
                    "rows": rows, "total_row": None},
        table_html=html, text=html,
    )


def bank_table():
    rows = [["БИК", "", "", ""], ["Сч. №", "", "", ""]]
    return ParsedBlock(
        type="table", page=1, bbox=[0.05, 0.14, 0.87, 0.25],
        table_data={"headers": ["Банк получателя", "", "", ""], "rows": rows, "total_row": None},
        table_html="<table></table>", text="<table></table>",
    )


def layer():
    lines = [
        line("Банк получателя", 0.15, x=0.06),
        line("БИК", 0.155, x=0.51, width=0.03),
        line("045678123", 0.155, x=0.59, width=0.08),
        line("Сч. №", 0.17, x=0.51, width=0.05),
        line("30101810500000000314", 0.17, x=0.59, width=0.19),
        line("№ Товары (работы, услуги) Кол-во Ед. Цена Сумма", 0.45),
        line("1 Ресертификационный аудит 1 усл 30 000,00 30 000,00", 0.47),
        line("Итого:", 0.50, x=0.69, width=0.05),
        line("Всего к оплате:", 0.535, x=0.61, width=0.13),
        line("Всего наименований 1, на сумму 30 000,00 руб", 0.55, x=0.06),
        line("Тридцать тысяч рублей 00 копеек", 0.566, x=0.06),
    ]
    return TextLayer(pages={1: lines}, page_count=1)


def texts(blocks):
    return [b.text for b in blocks if b.type == "text"]


def test_flag_off_vector_lines_in_table_frame_are_lost():
    assert settings.TABLE_VECTOR_LINES_KEPT is False
    blocks, stats = reconcile_with_text_layer([bank_table(), goods_table()], layer())
    joined = " ".join(texts(blocks))
    assert "Тридцать тысяч" not in joined
    assert stats.table_lines_kept == 0


def test_lost_words_and_numbers_come_back_as_text(enabled):
    blocks, stats = reconcile_with_text_layer([bank_table(), goods_table()], layer())
    joined = " ".join(texts(blocks))
    for expected in ("045678123", "30101810500000000314", "Итого", "Всего к оплате",
                     "Тридцать тысяч рублей 00 копеек"):
        assert expected in joined
    assert stats.table_lines_kept >= 5


def test_lines_already_in_cells_are_not_repeated(enabled):
    blocks, _ = reconcile_with_text_layer([goods_table()], layer())
    joined = " ".join(texts(blocks))
    # Строка шапки и строка позиции целиком есть в ячейках — текстом не дублируются.
    assert "Ресертификационный" not in joined
    assert "Товары" not in joined


def test_short_numbers_alone_do_not_come_back(enabled):
    table = goods_table()
    lay = TextLayer(pages={1: [line("12 15", 0.50, x=0.69, width=0.05)]}, page_count=1)
    blocks, stats = reconcile_with_text_layer([table], lay)
    assert stats.table_lines_kept == 0


def test_ocr_layer_rules_unchanged(enabled):
    # У слоя распознавания порог прежний: одно слово «Итого:» не возвращается.
    table = goods_table()
    lay = TextLayer(pages={1: [line("Итого:", 0.50, x=0.69, width=0.05)]}, page_count=1)
    blocks, stats = reconcile_with_text_layer([table], lay, source=OCR_LAYER)
    assert "Итого" not in " ".join(texts(blocks))


def test_number_formatted_differently_is_not_new(enabled):
    # Ревью 16.11: в ячейке «30 000,00», в слое «30000,00» — строка позиции не
    # должна возвращаться текстом повтором всей строки таблицы.
    table = goods_table()
    lay = TextLayer(pages={1: [line("1 Ресертификационный аудит 1 усл 30000,00 30000,00", 0.47)]},
                    page_count=1)
    blocks, stats = reconcile_with_text_layer([table], lay)
    assert "Ресертификационный" not in " ".join(texts(blocks))
    assert stats.table_lines_kept == 0
