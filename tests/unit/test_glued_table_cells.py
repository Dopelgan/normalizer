"""
Склеенные строки ячеек таблицы на скане.

Модель таблиц MinerU собирает многострочную ячейку без разделителя:
«Крепление / биозащиты / на фильтрах» приходит как «Креплениебиозащитына
фильтрах», и ни одно из слов в поиске уже не находится. Слой распознавания
той же страницы эти слова знает — по нему склейка и режется. Данные
синтетические.
"""

import pytest

from core.models.parse_result import ParsedBlock
from core.providers.text_layer import (
    METHOD_OCR_LAYER_RECOVERED,
    OCR_LAYER,
    TextLayer,
    TextLine,
    layer_vocabulary,
    reconcile_with_text_layer,
    split_glued_text,
    steel_grade,
    unify_script,
)


def line(text, y, x=0.1, width=0.5, height=0.02, page=1):
    return TextLine(text=text, page=page, bbox=[x, y, x + width, y + height],
                    confidence=0.9)


def layer_of(*lines):
    return TextLayer(pages={1: list(lines)}, page_count=1)


VOCAB = layer_vocabulary([
    line("Крепление кожуха на фильтрах", 0.1),
    line("биозащиты Гайка М8 с внутренним", 0.2),
    line("АБВГ 123456.007-19 См. АБВГ 123456.007.01", 0.3),
    line("установлены в мо- дуле", 0.4),
])


class TestSplitGluedText:
    def test_glued_words_are_split_by_layer_words(self):
        text, count = split_glued_text("Креплениебиозащитына фильтрах", VOCAB)
        assert text == "Крепление биозащиты на фильтрах"
        assert count == 1

    def test_code_glued_to_word_is_split_and_kept_whole(self):
        text, _ = split_glued_text("АБВГ123456.007-19", VOCAB)
        assert text == "АБВГ 123456.007-19"

    def test_split_after_punctuation_and_homoglyphs(self):
        # «См.» распознано латиницей, следующее слово приклеено: токен со
        # смешанным алфавитом берёт написание слоя.
        text, _ = split_glued_text("Cm.АБВГ 123456.007.01", VOCAB)
        assert text == "См. АБВГ 123456.007.01"

    def test_known_word_is_not_touched(self):
        assert split_glued_text("биозащиты", VOCAB) == ("биозащиты", 0)

    def test_partial_segmentation_is_rejected(self):
        # «неизвестное» в слое нет — разрезать наугад нельзя.
        assert split_glued_text("Креплениенеизвестное", VOCAB)[1] == 0

    def test_numbers_are_never_cut_between_digits(self):
        vocab = layer_vocabulary([line("123456 007", 0.1)])
        assert split_glued_text("123456.007", vocab) == ("123456.007", 0)

    def test_hyphenated_word_is_not_cut(self):
        assert split_glued_text("мо-дуле", VOCAB) == ("мо-дуле", 0)

    def test_markup_is_not_touched(self):
        vocab = layer_vocabulary([line("row span col Крепление кожуха", 0.1)])
        html = "<td rowspan=2 colspan=1>Креплениекожуха</td>"
        assert split_glued_text(html, vocab)[0] == (
            "<td rowspan=2 colspan=1>Крепление кожуха</td>"
        )


class TestSplitGuards:
    """Замечания ревью: шифры, мусор словаря OCR, буквы с цифрами."""

    def test_designation_code_is_not_cut_at_punctuation(self):
        vocab = layer_vocabulary([line("АБВГ 301611.002", 0.1)])
        assert split_glued_text("АБВГ.301611.002", vocab) == ("АБВГ.301611.002", 0)

    def test_slash_is_not_a_cut(self):
        vocab = layer_vocabulary([line("кгс см2 давление", 0.1)])
        assert split_glued_text("давление/см2", vocab)[1] == 0

    def test_junk_short_pieces_do_not_cut_real_words(self):
        vocab = layer_vocabulary([line("ре дуктор ный", 0.1)])
        assert split_glued_text("редукторный", vocab) == ("редукторный", 0)

    def test_short_letter_digit_mix_is_not_cut(self):
        vocab = layer_vocabulary([line("М12 х50 Ст3 сп", 0.1)])
        assert split_glued_text("М12х50", vocab)[1] == 0
        assert split_glued_text("Ст3сп20", layer_vocabulary([line("Ст3 сп20", 0.1)]))[1] == 0

    def test_word_glued_to_number_is_cut(self):
        vocab = layer_vocabulary([line("ГОСТ 7798-70 Сталь 45", 0.1)])
        assert split_glued_text("ГОСТ7798-70", vocab)[0] == "ГОСТ 7798-70"
        assert split_glued_text("Сталь45", vocab)[0] == "Сталь 45"

    def test_function_word_piece_is_allowed(self):
        assert split_glued_text("Креплениебиозащитына", VOCAB)[0] == "Крепление биозащиты на"


class TestReconcileOcrTable:
    def _table(self):
        cell = "Креплениебиозащитына фильтрах"
        return ParsedBlock(
            type="table", page=1, bbox=[0.05, 0.05, 0.95, 0.60],
            table_data={"headers": ["Обозначение", "Примечание"],
                        "rows": [["АБВГ123456.007-19", cell]], "total_row": None},
            table_html=f"<table><tr><td>АБВГ123456.007-19</td><td>{cell}</td></tr></table>",
            text=f"<table><tr><td>АБВГ123456.007-19</td><td>{cell}</td></tr></table>",
        )

    def _layer(self):
        return layer_of(
            line("Обозначение Примечание", 0.08),
            line("АБВГ Крепление", 0.15),
            line("123456.007-19 биозащиты", 0.18),
            line("на фильтрах", 0.21),
            # Подпись к столбцу, которую модель таблиц потеряла целиком.
            line("Количество деталей изделия", 0.30),
            # Обрывки линий сетки, прочитанные как буквы.
            line("О ООО ОИО | i ОИ И", 0.35),
            # Длинные обрывки без гласных — тоже не слова.
            line("ВКЛМ ПРСТ ||||", 0.40),
        )

    def test_glued_cells_are_split_in_data_and_html(self):
        blocks, stats = reconcile_with_text_layer([self._table()], self._layer(), source=OCR_LAYER)
        table = next(b for b in blocks if b.type == "table")
        assert table.table_data["rows"][0] == ["АБВГ 123456.007-19", "Крепление биозащиты на фильтрах"]
        assert "Крепление биозащиты на фильтрах" in table.table_html
        assert "Крепление биозащиты на фильтрах" in table.text
        assert stats.split_tokens == 2

    def test_lost_table_words_come_back_without_duplicates(self):
        blocks, stats = reconcile_with_text_layer([self._table()], self._layer(), source=OCR_LAYER)
        recovered = [b.text for b in blocks if b.method == METHOD_OCR_LAYER_RECOVERED]
        # Строки, чьи слова уже есть в таблице, повторно не выдаются.
        assert recovered == ["Количество деталей изделия"]
        assert stats.table_lines_kept == 1

    def test_vector_layer_splits_cells_but_returns_no_lines(self):
        # Модель таблиц склеивает ячейки и на векторном листе; строки слоя
        # из рамки там не возвращаются — это не измерялось.
        blocks, stats = reconcile_with_text_layer([self._table()], self._layer())
        table = next(b for b in blocks if b.type == "table")
        assert table.table_data["rows"][0][1] == "Крепление биозащиты на фильтрах"
        assert [b for b in blocks if b.type == "text"] == []


class TestMixedScript:
    """Латиница и греческий внутри русских слов ячейки (синтетика)."""

    @pytest.mark.parametrize("word, expected", [
        ("HOCТЬ", "НОСТЬ"),            # есть буква, которой нет в латинице
        ("Пοпроекту", "Пοпроекту"),    # греческое — только по словарю слоя
        ("Πoпроекту", "Πoпроекту"),    # три алфавита — только по словарю слоя
        ("exрires", "expires"),        # латинское слово с кириллицей
        ("OTK", "OTK"),                # один алфавит — не трогаем
        ("priёmki", "priёmki"),        # каша из обоих — не угадываем
    ])
    def test_unify_script(self, word, expected):
        assert unify_script(word) == expected

    def test_transliterated_piece_takes_layer_spelling(self):
        vocab = layer_vocabulary([line("На стадии приёмки изделий", 0.1)])
        assert split_glued_text("Haстадииpriёmkiизделий", vocab)[0] == (
            "На стадии приёмки изделий"
        )

    def test_three_scripts_are_resolved_by_layer(self):
        vocab = layer_vocabulary([line("По проекту", 0.1)])
        assert split_glued_text("Πoпроекту", vocab)[0] == "По проекту"

    def test_vector_spelling_wins_single_script_word(self):
        vocab = layer_vocabulary([line("СТО 05.18 СТП", 0.1)])
        assert split_glued_text("CTO 05.18СТП", vocab, exact=True)[0] == "СТО 05.18 СТП"
        # Слою распознавания в токене одного алфавита не верим.
        assert split_glued_text("CTO", vocab)[0] == "CTO"

    def test_vector_table_cells_are_split(self):
        table = ParsedBlock(
            type="table", page=1, bbox=[0.05, 0.05, 0.95, 0.60],
            table_data={"headers": ["Буквенныйшифр", "Нoмepизменений"],
                        "rows": [["АБВГ", "1"]], "total_row": None},
        )
        layer = layer_of(
            line("Буквенный", 0.10, x=0.06, width=0.2), line("шифр", 0.13, x=0.06, width=0.1),
            line("Номер", 0.10, x=0.50, width=0.2), line("изменений", 0.13, x=0.50, width=0.2),
        )
        blocks, _ = reconcile_with_text_layer([table], layer)
        assert blocks[0].table_data["headers"] == ["Буквенный шифр", "Номер изменений"]

    def test_latin_lookalike_piece_in_russian_token_becomes_cyrillic(self):
        vocab = layer_vocabulary([line("Ha стадии приёмки", 0.1)])
        # В токене есть и латиница без кириллической пары (r, i), поэтому
        # алфавит целиком не выправить — решает кусок.
        assert split_glued_text("Haстадииpriёmki", vocab)[0] == "На стадии приёмки"

    def test_cut_after_comma_before_word_with_suffix_in_brackets(self):
        vocab = layer_vocabulary([line("главный аудитор, аудитор(ы)", 0.1)], )
        assert split_glued_text("главныйаудитор,аудитор(ы)", vocab, exact=True)[0] == (
            "главный аудитор, аудитор(ы)"
        )


class TestLegitMixedTokens:
    """Ревью bd7b651: законные смеси латиницы и греческого не трогаются."""

    TOKENS = ["kΩ", "μm", "Δp", "τmax"]

    @pytest.mark.parametrize("token", TOKENS)
    def test_unify_script_leaves_greek_latin(self, token):
        assert unify_script(token) == token

    @pytest.mark.parametrize("exact", [True, False])
    def test_tokens_from_layer_are_kept(self, exact):
        text = "R = 10 kΩ, 5 μm, Δp, τmax"
        vocab = layer_vocabulary([line(text, 0.1)])
        assert split_glued_text(text, vocab, exact)[0] == text

    @pytest.mark.parametrize("token, expected", [
        ("кBт", "кВт"), ("MПa", "МПа"), ("exрires", "expires"),
        ("IP65", "IP65"), ("M8", "M8"), ("Rev.1", "Rev.1"), ("Ø20", "Ø20"), ("SiO2", "SiO2"),
    ])
    def test_substitutions_still_fixed(self, token, expected):
        assert split_glued_text(token, {})[0] == expected


class TestGostSymbols:
    """Ревью e501bde: греческая буква с русским индексом — законное обозначение."""

    @pytest.mark.parametrize("token", ["τср", "αт", "ρв", "κп", "εр", "ТипA1"])
    def test_kept_without_vocabulary(self, token):
        assert split_glued_text(token, {})[0] == token
        assert unify_script(token) == token

    def test_greek_fixed_only_by_layer(self):
        vocab = layer_vocabulary([line("Попроекту", 0.1)])
        assert split_glued_text("Пοпроекту", vocab)[0] == "Попроекту"

    def test_layer_spelling_never_turns_digit_into_letter(self):
        # OCR прочитал «3» как «з»; свёртка их не различает.
        vocab = layer_vocabulary([line("АБВГ 681821.001ПМз", 0.1)])
        assert split_glued_text("681821.001ПM3", vocab)[0] == "681821.001ПM3"


class TestDrawingStamp:
    """Основная надпись чертежа, прочитанная с курсивными подменами (синтетика)."""

    def _stamp(self, *rows):
        return ParsedBlock(
            type="table", page=1, bbox=[0.05, 0.80, 0.95, 0.99],
            table_data={"headers": ["Изм.", "Лист", "№ докум.", "Подп.", "Дата"],
                        "rows": [list(r) for r in rows] + [["Пров.", "Сидоров", "", "", ""]],
                        "total_row": None},
        )

    def _reconcile(self, block, *texts):
        lines = [line(t, 0.82 + 0.02 * i, x=0.06, width=0.85) for i, t in enumerate(texts)]
        blocks, _ = reconcile_with_text_layer([block], layer_of(*lines), source=OCR_LAYER)
        return next(b for b in blocks if b.type == "table").table_data["rows"]

    def test_cursive_labels_are_fixed_in_stamp(self):
        rows = self._reconcile(
            self._stamp(["Разрад.", "Иванов", "Macca", "Macwmad", "Лum."],
                        ["Проб.", "Петров", "Дама", "FOCT", "Лустов 1"],
                        ["Ра3pаб.", "И3м.", "", "", ""]),
            # Слой прочитал подпись той же латиницей — штамп всё равно прав.
            "Иванов Петров Macca",
        )
        assert rows[0] == ["Разраб.", "Иванов", "Масса", "Масштаб", "Лит."]
        assert rows[2][:2] == ["Разраб.", "Изм."]
        assert rows[1] == ["Пров.", "Петров", "Дата", "ГОСТ", "Листов 1"]

    def test_material_with_steel_grade(self):
        rows = self._reconcile(
            self._stamp(["Луcт12X18H10T ГОСТ 5582-75", "", "", "", ""]), "ГОСТ 5582-75",
        )
        assert rows[0][0] == "Лист 12Х18Н10Т ГОСТ 5582-75"

    def test_part_name_glued_to_digit_and_known_word(self):
        rows = self._reconcile(
            self._stamp(["Крышка2", "Продольный усилителькороткий", "", "", ""]),
            "Продольный усилитель",
        )
        assert rows[0][:2] == ["Крышка 2", "Продольный усилитель короткий"]

    def test_labels_are_not_guessed_outside_stamp(self):
        # «отбор проб» — не «Пров.»: без штампа закрытый словарь не действует.
        block = ParsedBlock(
            type="table", page=1, bbox=[0.05, 0.05, 0.95, 0.40],
            table_data={"headers": ["Операция", "Объём"],
                        "rows": [["Отбор проб.", "Дама"]], "total_row": None},
        )
        layer = layer_of(line("Операция Объём", 0.1))
        blocks, _ = reconcile_with_text_layer([block], layer, source=OCR_LAYER)
        assert blocks[0].table_data["rows"][0] == ["Отбор проб.", "Дама"]

    @pytest.mark.parametrize("token, expected", [
        ("12X18H10T", "12Х18Н10Т"), ("08X18H10T", "08Х18Н10Т"),
        ("770X520X320", None), ("10X20", None), ("IP65", None), ("M8", None),
        ("10AT", None), ("25MBA", None), ("1K5M", None), ("100KT", None), ("10TA", None),
    ])
    def test_steel_grade(self, token, expected):
        assert steel_grade(token) == expected

    def test_compound_with_short_known_stem_is_not_cut(self):
        vocab = layer_vocabulary([line("тепло", 0.1)])
        assert split_glued_text("теплообменник", vocab)[0] == "теплообменник"


class TestOcrTextBlocks:
    def test_mixed_script_fixed_in_parser_text_block(self):
        block = ParsedBlock(type="text", page=1, bbox=[0.05, 0.05, 0.95, 0.10],
                            text="HOCТЬ размеры УхХхZ, мм")
        layer = layer_of(line("Прочий текст", 0.5))
        blocks, _ = reconcile_with_text_layer([block], layer, source=OCR_LAYER)
        assert blocks[0].text == "НОСТЬ размеры YxXxZ, мм"


class TestStampDetection:
    """Ревью d54e808: обычная таблица не штамп, коды — не марки стали."""

    def _reconcile(self, headers, rows, bbox=(0.05, 0.80, 0.95, 0.99)):
        block = ParsedBlock(type="table", page=1, bbox=list(bbox),
                            table_data={"headers": headers, "rows": rows, "total_row": None})
        # Строка слоя внутри рамки: без неё таблица сшивкой не обрабатывается.
        layer = layer_of(line("Прочий текст", bbox[1] + 0.02, x=bbox[0] + 0.01, width=0.3))
        blocks, _ = reconcile_with_text_layer([block], layer, source=OCR_LAYER)
        return next(b for b in blocks if b.type == "table").table_data["rows"]

    def test_revision_table_is_not_a_stamp(self):
        rows = self._reconcile(["Изм.", "Лист", "№ докум.", "Подп.", "Дата"],
                               [["прав.", "Инд", "Дота", "", ""]])
        assert rows[0][:3] == ["прав.", "Инд", "Дота"]

    def test_spec_with_gost_is_not_a_stamp(self):
        rows = self._reconcile(["Обозначение", "Материал", "Масса"],
                               [["ГОСТ 5582-75", "ГОСТ 19904-90", "Дама"]])
        assert rows[0][2] == "Дама"

    def test_stamp_must_sit_at_the_bottom(self):
        rows = self._reconcile(["Изм.", "Лист", "Подп.", "Дата", "Масса"],
                               [["Пров.", "Разрад.", "", "", ""]], bbox=(0.05, 0.05, 0.95, 0.30))
        assert rows[0][1] == "Разрад."

    def test_prav_is_not_prov_even_in_stamp(self):
        rows = self._reconcile(["Изм.", "Лист", "Подп.", "Дата", "Масса"],
                               [["Пров.", "прав.", "", "", ""]])
        assert rows[0][1] == "прав."

    @pytest.mark.parametrize("text", ["10AT", "25MBA", "1K5M", "100KT", "10TA",
                                      "type 10AT", "12X18H10T"])
    def test_codes_outside_stamp_are_kept(self, text):
        rows = self._reconcile(["Code", "Value"], [[text, "1"]], bbox=(0.05, 0.1, 0.9, 0.3))
        assert rows[0][0] == text

    def test_stamp_recognised_by_control_signature(self):
        # Курсивные «Разрад.»/«Проб.» не совпадают буква в букву, но
        # «Т.контр.» штамп выдаёт.
        rows = self._reconcile(["Изм.", "Лист", "Подп.", "Дама", "Т.контр."],
                               [["Разрад.", "Проб.", "Macca", "", ""]])
        assert rows[0][:3] == ["Разраб.", "Пров.", "Масса"]
