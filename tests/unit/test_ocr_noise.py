"""Шум распознавания растра: выправка слов, таблица-спецификация, строки сетки."""

import pytest

from core.models.parse_result import ParsedBlock
from core.providers import ocr_cleanup
from core.providers.ocr_cleanup import Vocabulary, clean_blocks, clean_layer, clean_text
from core.providers.text_layer import OCR_LAYER, TextLayer, TextLine, reconcile_with_text_layer

# Словарь вместо моделей Tesseract: в тестах утилит tesseract нет.
VOCAB = Vocabulary(
    rus=frozenset(
        "ввод разъемный винт тип типов рейка шайба кабельный давления сборочный чертеж "
        "обозначение условия технические по от оценки нормы объеме тост панель монтажная "
        "гайка компенсатор перепадов клемма протокол указывается схеме в "
        "дирка оп цро ог ап поп бубна бета то".split()
    ),
    eng=frozenset("no the drawing assembly din".split()),
)


@pytest.mark.parametrize("raw, expected", [
    ("Bbod разъемный типов", "Ввод разъемный типов"),
    ("Винт BuHT", "Винт Винт"),
    ("Рейка Peika", "Рейка Рейка"),
    ("Зам. 3aum.", "Зам. Заим."),
    ("TY 27.33.13-024 Клемма", "ТУ 27.33.13-024 Клемма"),
    ("TADY 687000.050-XX*2 Клемма", "ТАДУ 687000.050-XX*2 Клемма"),
    # «ТОСТ» есть в общем словаре, но обозначение документации старше.
    ("TOCT P 50.05.08-2018 в объеме", "ГОСТ P 50.05.08-2018 в объеме"),
    ("по ГОСm 5632-2014", "по ГОСТ 5632-2014"),
    ("по ГОСт 15150-69", "по ГОСТ 15150-69"),
    ("Нормы оценки no ГОСТ", "Нормы оценки по ГОСТ"),
    ("Протокол No 651", "Протокол № 651"),
    ("Ввод кадельный типа", "Ввод кабельный типа"),
    ("Компенсатор перепадов дабления", "Компенсатор перепадов давления"),
    ("Сдорочный чертеж", "Сборочный чертеж"),
    ("Технические цсловия", "Технические условия"),
    ("указыдается в схеме", "указывается в схеме"),
])
def test_words_fixed_in_russian_line(raw, expected):
    assert clean_text(raw, VOCAB)[0] == expected


@pytest.mark.parametrize("raw", [
    # Обозначения и коды с цифрами не трогаются никогда.
    "Болт DIN 933 M6x14-A2 по ТУ",
    "СП-01 CM-02 ТАДУ 405210.001TY",
    # Английский текст остаётся английским.
    "the drawing no assembly",
    # Слово без словарного прочтения остаётся как есть.
    "Hopmer оценки по ГОСТ",
    # Уровень качества ISO 5817 — латиница по делу.
    "уровень качества по ГОСТ - B",
    # Короткие обрывки не становятся «словами» из словаря.
    "Клемма oN upo or an",
    # Курсив не даёт «б» вместо «д»: настоящее слово вне словаря остаётся.
    "Бирка маркировочная",
    # Половина слова до переноса не выправляется.
    "требования к изде- лиям",
    # Заглавная N — не курсивная «п».
    "Протокол NoN ПКN",
])
def test_codes_english_and_unknown_kept(raw):
    assert clean_text(raw, VOCAB)[0] == raw


def test_grid_junk_removed_dashes_kept():
    text, stats = clean_text("Ввод — разъемный | типов __ —_", VOCAB)
    assert text == "Ввод — разъемный типов"
    assert stats.junk_tokens == 3


@pytest.mark.parametrize("raw", [
    "напряжение ~ 220 В по ГОСТ",
    "Подпись ______ дата",
    "формула $\\beta$ и $a \\to b$ по ГОСТ",
    "двойной  пробел по ГОСТ",
    # Имя собственное посреди текста не «выправляется».
    "город Дубна по ГОСТ",
])
def test_meaningful_tokens_kept(raw):
    assert clean_text(raw, VOCAB)[0] == raw


def test_empty_vocabulary_only_strips_grid():
    text, stats = clean_text("Bbod | разъемный", Vocabulary())
    assert text == "Bbod разъемный"
    assert stats.latin_words == 0


def test_cells_use_table_context():
    # Одна ячейка «TY» сама по себе латиница; по таблице видно, что русская.
    table = ParsedBlock(type="table", text="", page=1, bbox=[0, 0, 1, 1],
                        table_data={"headers": ["Обозначение", "Наименование"],
                                    "rows": [["TY", "Винт BuHT"], ["DIN 933", "Шайда"]]},
                        table_html="<table><tr><td>TY</td><td>Шайда</td></tr></table>")
    stats = clean_blocks([table], VOCAB)
    assert table.table_data["rows"] == [["ТУ", "Винт Винт"], ["DIN 933", "Шайба"]]
    assert table.table_html == "<table><tr><td>ТУ</td><td>Шайба</td></tr></table>"
    assert stats.total >= 3


def test_clean_layer_drops_lines_left_empty():
    layer = TextLayer(pages={1: [
        TextLine(text="| ||", page=1, bbox=[0.1, 0.1, 0.2, 0.12]),
        TextLine(text="Bbod разъемный", page=1, bbox=[0.1, 0.2, 0.5, 0.22]),
    ]}, page_count=1)
    cleaned, _ = clean_layer(layer, VOCAB)
    assert [l.text for l in cleaned.pages[1]] == ["Ввод разъемный"]


def test_vocabulary_without_tesseract_tools_is_empty(monkeypatch):
    ocr_cleanup._load_vocabulary.cache_clear()
    monkeypatch.setattr(ocr_cleanup.shutil, "which", lambda name: None)
    try:
        assert ocr_cleanup.load_vocabulary().empty
    finally:
        ocr_cleanup._load_vocabulary.cache_clear()


# ------------------------------------- спецификация ЕСКД во весь лист
NOTES = [
    "Устанавливается только для 2 и 3 класса по НП-001-15 или по требованию заказчика.",
    "Изображено в схеме сборки, соответствующей условному обозначению коробки.",
    "Наличие, количество и тип в соответствии с условиями договора поставки,",
    "указывается в схеме сборки, соответствующей условному обозначению коробки.",
    "Допускается замена в соответствии с ведомостью замены клемм и принадлежностей.",
    "Маркировку наносить краской контрастного цвета на табличке корпуса коробки.",
    "Резьбовые соединения затянуть и застопорить краской по всей длине соединения.",
    "Коробку испытать на прочность и плотность пробным давлением воздуха в течение минуты.",
    "Комплект монтажных частей поставляется по отдельному заказу потребителя изделия.",
    "Остальные технические требования по техническим условиям на коробку соединительную.",
]


def _spec_table():
    rows = [["A4", "", str(i), f"ТАДУ 765200.0{80 + i}.00", "Ввод кабельный", "*2"] for i in range(1, 9)]
    return ParsedBlock(type="table", text="<table></table>", page=1, bbox=[0.03, 0.02, 0.97, 0.98],
                       table_data={"headers": ["Формат", "Зона", "Поз.", "Обозначение", "Наименование", "Кол."],
                                   "rows": rows})


def _notes_layer(conf=0.9):
    return TextLayer(pages={1: [
        TextLine(text=t, page=1, bbox=[0.2, 0.55 + i * 0.03, 0.8, 0.57 + i * 0.03], confidence=conf)
        for i, t in enumerate(NOTES)
    ]}, page_count=1)


def test_frame_guard_drops_spec_table_without_flag():
    result, stats = reconcile_with_text_layer(
        [_spec_table()], _notes_layer(), source=OCR_LAYER, frame_table_guard=True)
    assert stats.frame_tables == 1
    assert not [b for b in result if b.type == "table"]


def test_structured_spec_table_kept_on_scan():
    result, stats = reconcile_with_text_layer(
        [_spec_table()], _notes_layer(), source=OCR_LAYER, frame_table_guard=True,
        keep_structured_tables=True)
    assert stats.frame_tables == 0
    assert [b for b in result if b.type == "table"]


def test_prose_frame_table_still_dropped_with_flag():
    # Текст в рамке: одна длинная ячейка на строку — это не таблица.
    prose = ParsedBlock(type="table", text="", page=1, bbox=[0.03, 0.02, 0.97, 0.98],
                        table_data={"rows": [[t] for t in NOTES]})
    _, stats = reconcile_with_text_layer(
        [prose], _notes_layer(), source=OCR_LAYER, frame_table_guard=True,
        keep_structured_tables=True)
    assert stats.frame_tables == 1


# ------------------------------ повтор таблицы через линии сетки
def _grid_lines(conf):
    # Строки слоя внутри таблицы, слов которых в ячейках нет: прочтение
    # той же таблицы через линии сетки.
    return TextLayer(pages={1: [
        TextLine(text="Kownewcamop перепадов dabrewn монтажыя", page=1,
                 bbox=[0.1, 0.3, 0.9, 0.32], confidence=conf),
    ]}, page_count=1)


def _small_table():
    return ParsedBlock(type="table", text="<table></table>", page=1, bbox=[0.05, 0.1, 0.95, 0.5],
                       table_data={"headers": ["Поз.", "Наименование"],
                                   "rows": [["1", "Компенсатор перепадов давления"]]})


@pytest.mark.parametrize("conf, floor, kept", [(0.45, 0.7, 0), (0.45, 0.0, 1), (0.9, 0.7, 1)])
def test_low_confidence_table_lines_dropped(conf, floor, kept):
    result, stats = reconcile_with_text_layer(
        [_small_table()], _grid_lines(conf), source=OCR_LAYER, table_line_min_confidence=floor)
    texts = [b for b in result if b.type == "text"]
    assert len(texts) == kept
    assert stats.low_confidence_table_lines == ((1 - kept) if floor else 0)


# ---------------------------------------- разворот в две колонки
def test_wide_gap_splits_tesseract_line():
    from core.providers.tesseract_fallback import _split_wide_gaps

    # (left, top, right, bottom, слово, уверенность); лист 1000 px.
    words = [(10, 0, 60, 20, "Термопреобразователи", 0.9), (70, 0, 120, 20, "сопротивления", 0.9),
             (560, 0, 600, 20, "Сбор", 0.9), (610, 0, 680, 20, "металлов", 0.9)]
    assert [[w[4] for w in s] for s in _split_wide_gaps(words, 1000)] == [
        ["Термопреобразователи", "сопротивления"], ["Сбор", "металлов"]]
    # Обычный пробел строку не режет.
    assert len(_split_wide_gaps(words[:2], 1000)) == 1


def _column_lines(x, texts):
    return [TextLine(text=t, page=1, bbox=[x, 0.1 + i * 0.03, x + 0.4, 0.12 + i * 0.03])
            for i, t in enumerate(texts)]


def test_prose_columns_grouped_separately():
    from core.providers.text_layer import group_columns, group_lines

    left = _column_lines(0.05, [f"Левая колонка, строка прозы номер {i}" for i in range(5)])
    right = _column_lines(0.55, [f"Правая колонка, строка прозы номер {i}" for i in range(5)])
    groups = group_columns(left + right)
    texts = [" ".join(l.text for l in g) for g in groups]
    assert len(groups) == 2
    assert "Левая" in texts[0] and "Правая" not in texts[0]
    # Без деления по колонкам строки перемешаны.
    assert len(group_lines(left + right)) == 1


def test_table_columns_stay_together():
    from core.providers.text_layer import group_columns, group_lines

    pos = _column_lines(0.05, ["1", "2", "3", "4", "5"])
    names = _column_lines(0.55, ["Винт", "Гайка", "Шайба", "Ввод", "Рейка"])
    assert group_columns(pos + names) == group_lines(pos + names)


def test_spread_lines_replaced_by_parts_only_for_prose_columns():
    from core.providers.text_layer import column_lines

    def spread(i, left_text, right_text):
        left = TextLine(text=left_text, page=1, bbox=[0.05, 0.1 + i * 0.03, 0.45, 0.12 + i * 0.03])
        right = TextLine(text=right_text, page=1, bbox=[0.55, 0.1 + i * 0.03, 0.95, 0.12 + i * 0.03])
        return TextLine(text=f"{left_text} {right_text}", page=1,
                        bbox=[0.05, 0.1 + i * 0.03, 0.95, 0.12 + i * 0.03], parts=(left, right))

    prose = [spread(i, f"Левая колонка, строка прозы {i}", f"Правая колонка, строка прозы {i}")
             for i in range(5)]
    assert len(column_lines(prose)) == 10
    # Ряд таблицы: куски — ячейки, строка остаётся целой.
    rows = [spread(i, str(i), f"Ввод кабельный {i}") for i in range(5)]
    assert column_lines(rows) == rows


def test_clean_layer_cleans_parts():
    part = TextLine(text="Bbod разъемный |", page=1, bbox=[0.1, 0.2, 0.3, 0.22])
    line = TextLine(text="Bbod разъемный | Клемма", page=1, bbox=[0.1, 0.2, 0.9, 0.22], parts=(part,))
    cleaned, _ = clean_layer(TextLayer(pages={1: [line]}, page_count=1), VOCAB)
    assert cleaned.pages[1][0].parts[0].text == "Ввод разъемный"


def test_capital_word_after_sentence_end_is_fixed():
    assert clean_text("Клемма. Сдорочный чертеж", VOCAB)[0] == "Клемма. Сборочный чертеж"
    assert clean_text("Клемма\nСдорочный чертеж", VOCAB)[0] == "Клемма\nСборочный чертеж"


def test_empty_frame_keeps_lines_without_parsed_tables():
    empty = ParsedBlock(type="table", text="", page=1, bbox=[0.05, 0.1, 0.95, 0.5])
    result, _ = reconcile_with_text_layer(
        [empty], _grid_lines(0.45), source=OCR_LAYER, table_line_min_confidence=0.7)
    assert [b for b in result if b.type == "text"]


def test_continuation_frame_drops_unsure_lines():
    empty = ParsedBlock(type="table", text="", page=2, bbox=[0.05, 0.1, 0.95, 0.5])
    head = _small_table()
    layer = TextLayer(pages={2: [TextLine(text="Kownewcamop перепадов dabrewn монтажыя", page=2,
                                          bbox=[0.1, 0.3, 0.9, 0.32], confidence=0.45)]},
                      page_count=2)
    result, stats = reconcile_with_text_layer(
        [head, empty], layer, source=OCR_LAYER, table_line_min_confidence=0.7)
    assert not [b for b in result if b.type == "text"]
    assert stats.low_confidence_table_lines == 1


# ---------------------------------------------------- примечания чертежа
DRAWING_VOCAB = Vocabulary(
    rus=frozenset("договора поставки масса изделия технологических операций".split()),
    eng=frozenset("the".split()),
)


@pytest.mark.parametrize("raw, expected", [
    ("условиям Cozobopa поставки", "условиям Договора поставки"),
    ("6. Macca usdenua указана в схеме", "6. Масса изделия указана в схеме"),
    ("при проведении MEXHONOZUYECKUX операций", "при проведении ТЕХНОЛОГИЧЕСКИХ операций"),
])
def test_drawing_note_italic_latin_fixed(raw, expected):
    assert clean_text(raw, DRAWING_VOCAB)[0] == expected


def _block(text, method, page=1, type_="text"):
    return ParsedBlock(type=type_, text=text, page=page, bbox=[0, 0, 1, 1], order=0,
                       confidence=0.9, method=method)


MINERU_NOTES = (
    "4. Размеры Х и У - габаритные размеры конечного изделия определяются условиями "
    "договора поставки и указаны в условном обозначении коробки. 5. Координаты отверстий "
    "и их количество определяется схемой размеров. 6. Масса изделия указана в схеме размеров. "
    "7. Допускается замена материала и сортамента в соответствии с ведомостью замены."
)


def test_recovered_repeat_of_mineru_text_dropped():
    from core.providers.mineru_parser import drop_repeated_recovered_blocks
    from core.providers.text_layer import METHOD_OCR_LAYER_RECOVERED as REC

    blocks = [
        _block(MINERU_NOTES, "mineru_ocr"),
        # Повтор примечаний строками через графику.
        _block("$ \ < 4. Размеры Х u У - габаритные размеры конечного изделия "
               "определятся условиям Cozobopa поставки и указаны в условнам обозначении", REC),
        # Графика без единого русского слова.
        _block("a С» 125) \/ R263 1\/) G", REC),
        # Текст, которого у MinerU нет, остаётся.
        _block("Маркировать шрифтом лазерным способом на технологической бирке", REC),
        # Другая страница без текста MinerU — слой единственный источник.
        _block("a С» 125) \/ R263", REC, page=2),
    ]
    kept, counts = drop_repeated_recovered_blocks(blocks, 0.9)
    assert counts["total"] == 2
    assert [b.text[:10] for b in kept] == [
        MINERU_NOTES[:10], "Маркироват", "a С» 125) ",
    ]


def test_recovered_english_text_kept_on_english_page():
    # Английский скан: в восстановленном блоке нет русских слов, но это
    # текст, которого у MinerU нет, а не графика чертежа.
    from core.providers.mineru_parser import drop_repeated_recovered_blocks
    from core.providers.text_layer import METHOD_OCR_LAYER_RECOVERED as REC

    english = (
        "The pump shall be installed on a rigid foundation and aligned before start-up. "
        "Check the rotation direction and the oil level in the bearing housing. "
        "Operating temperature must not exceed the limits given in the data sheet. "
        "Replace the mechanical seal if leakage is visible after the first hour of operation."
    )
    assert len("".join(english.split())) >= 200
    blocks = [
        _block(english, "mineru_ocr"),
        _block("Warranty does not cover damage caused by improper storage", REC),
    ]
    kept, counts = drop_repeated_recovered_blocks(blocks, 0.9)
    assert counts["total"] == 0
    assert len(kept) == 2


@pytest.mark.parametrize("recovered", [
    "10JEB10CT001",
    "⌀14 R1,5",
    "Ø 8",
    # Русские слова — повтор MinerU, но кода у MinerU нет.
    "4. Размеры Х и У - габаритные размеры конечного изделия 10LAB20AA101",
])
def test_recovered_designation_absent_from_mineru_kept(recovered):
    # Обозначение без русских слов похоже на графику, но это данные: если
    # у MinerU на странице его нет, блок не отбрасывается.
    from core.providers.mineru_parser import drop_repeated_recovered_blocks
    from core.providers.text_layer import METHOD_OCR_LAYER_RECOVERED as REC

    blocks = [_block(MINERU_NOTES, "mineru_ocr"), _block(recovered, REC)]
    kept, counts = drop_repeated_recovered_blocks(blocks, 0.9)
    assert counts["total"] == 0
    assert kept[-1].text == recovered


def test_recovered_designation_known_to_mineru_dropped():
    # Код уже есть у MinerU (кириллические «СТ», пробел внутри) — блок повтор.
    from core.providers.mineru_parser import drop_repeated_recovered_blocks
    from core.providers.text_layer import METHOD_OCR_LAYER_RECOVERED as REC

    blocks = [
        _block(MINERU_NOTES + " Датчик 10JEB10 СТ001, отверстие ⌀ 14.", "mineru_ocr"),
        _block("10JEB10CT001 ⌀14", REC),
    ]
    kept, counts = drop_repeated_recovered_blocks(blocks, 0.9)
    assert counts["total"] == 1
    assert len(kept) == 1
