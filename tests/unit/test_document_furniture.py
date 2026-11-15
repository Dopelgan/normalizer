"""
Рамка бланка и переносы (`SCAN_DROP_FRAME_FURNITURE`,
`SCAN_JOIN_HYPHENATED_WORDS`), заголовок раздела
(`DOCUMENT_HEADING_WITH_BODY`).

Все строки здесь настоящие — из выгрузки индексированного текста 50
документов с прода (ТАДУ 681821.005 РЭ, ТАДУ 681821.001 РЭ,
ТАДУ 681821.003 И1, ТАДУ 681821.005 ПС, СТП ТАДУ 15.01-22ТУ, свидетельства
о поверке СБ210). Придуманных нет, кроме одной фразы, где «Подпись и
дата» — содержание, а не рамка: такой случай в выборке не встретился, и
именно его правило обязано не испортить.

Флаги ставятся через `monkeypatch.setattr` на объявленное поле `Settings`.
"""

import pytest

from core.config import settings
from core import document_furniture, document_parts
from core.contract_finalize import finalize_fragments
from core.models.contract import Content, Fragment, Position, Provenance

# Чанки, которые в индексе есть и которых быть не должно: рамка и больше
# ничего. «Формат А4» встречается так 72 раза, «Формат Ad» — 10.
FURNITURE = [
    "Формат А4",
    "Формат Ad",
    "Формат A4",
    "Подпись и дата",
    "Подпись и дата Ионова 06.2023",
    "Анцупова 11.2021 Подпись и дата",
    "Инв. № подл. 952",
    "Подпись и дата Инв. № дубл. Взам. инв № Анцупова 11.2021 Подпись и дата Инв. № подл.",
    "[TABLE] Изм Л ист № документа Подпись Дата",
    "[TABLE] ТАДУ 681821.001РЭ Лист 35 Изм Лист№ документа Подпись Дата",
    "Лист ТАДУ 681821.003И1 13 Изм Лист № документа Подпись Дата Формат А4",
    "Ионова 06.2023 Подпись и дата Инв. № подл. 1405 Изм [Tver [М документа "
    "[Подпись [Дата (ТАДУ 681821.005P3) Формат Ad",
]
# Содержание, которое правило обязано оставить целиком.
CONTENT = [
    "4. Общие указания",
    "Таблица 1",
    "- для кабелей, кабельных муфт и клеммных коробок условия Ж по ГОСТ 23216.",
    "Ошиновка– изделие III категории сейсмостойкости по НП-031-01.",
    "СВИДЕТЕЛЬСТВО О ПОВЕРКЕ СРЕДСТВА ИЗМЕРЕНИЙ",
    "- лак КО-921 ГОСТ 16508 – 1кг - отвердитель ТБТ – 0,004кг - толуол "
    "ГОСТ 14710 -0,08кг. Взам. ин",
    # Запись о ревизии документа: извещение и дата — это содержание, и
    # похожесть на графу изменений её не отменяет.
    "Изм.: 1 Зам. ТАДУ.655-25 2.09.19",
    # Придуманная фраза: в паспорте графа «Подпись и дата» бывает частью
    # формы, то есть содержанием.
    "Подпись и дата монтажника заносится в формуляр изделия при приёмке.",
]
# Настоящий абзац ТАДУ 681821.005 РЭ: боковая графа вставлена в середину
# слова «поверхностей», и чанк при этом длинный — 1673 символа в индексе.
SPLICED = (
    "Настоящее «Руководство по эксплуатации» предназначено для ознакомления "
    "обслуживающего персонала АЭС с устройством и конструкцией ошиновки "
    "компенсатора давления реакторной установки. В приложении Г приведены "
    "растворы для дезактивации наружных поверхно- Подпись и дата стей. "
    "В приложении Д приведены инструкции по монтажу муфт. Перед началом "
    "работы ознакомиться с настоящим руководством по эксплуата- ции."
)


# Страница 1 свидетельства о поверке (doc-3200eb32143f9a398e00ef83c91b31aa):
# подпись графы бланка по приказу 2510 напечатана мелким шрифтом под
# названием поля, и распознавание ставит её между названием «Средство
# измерений» и его значением «Преобразователь термоэлектрический ...».
CERTIFICATE = (
    "СВИДЕТЕЛЬСТВО О ПОВЕРКЕ № С-ВЯ/15-11-2026/260024 "
    "Средство измерений "
    "наименование и обозначение типа, модификация (при наличии), "
    "регистрационный номер в Федеральном информационном фонде по "
    "обеспечению единства измерений, "
    "Преобразователь термоэлектрический; ТХА Метран-2700, модификация "
    "2700-01, рег. № 75763-19; Зав. номер 260024 "
    "поверено в полном объёме и признано пригодным к применению."
)


@pytest.fixture
def furniture(monkeypatch):
    monkeypatch.setattr(settings, "SCAN_DROP_FRAME_FURNITURE", True)


@pytest.fixture
def furniture_off(monkeypatch):
    monkeypatch.setattr(settings, "SCAN_DROP_FRAME_FURNITURE", False)
    monkeypatch.setattr(settings, "SCAN_JOIN_HYPHENATED_WORDS", False)


@pytest.fixture
def joining(monkeypatch):
    monkeypatch.setattr(settings, "SCAN_JOIN_HYPHENATED_WORDS", True)


def text_fragment(text: str) -> Fragment:
    return Fragment(
        fragment_id="", type="text",
        content=Content(text=text),
        position=Position(page=1, bbox=[0, 0, 1, 1]),
        confidence=0.9, completeness=None,
        provenance=Provenance(method="ocr_layer", strategy_level=5, source="pdf"),
    )


class TestIsFurniture:
    @pytest.mark.parametrize("text", FURNITURE)
    def test_frame_only(self, text):
        assert document_furniture.is_furniture(text) is True

    @pytest.mark.parametrize("text", CONTENT)
    def test_content_kept(self, text):
        assert document_furniture.is_furniture(text) is False

    def test_empty(self):
        assert document_furniture.is_furniture("") is False
        assert document_furniture.is_furniture(None) is False

    def test_digits_alone_are_not_furniture(self):
        """Фрагмент из одних цифр — не рамка: это не её признак."""
        assert document_furniture.is_furniture("189260196") is False


class TestSplices:
    def test_graph_cut_from_sentence(self, furniture):
        fixed, cut = document_furniture.without_splices(SPLICED)
        assert cut == 1
        assert "наружных поверхностей." in fixed
        assert "Подпись и дата" not in fixed

    def test_short_text_untouched(self, furniture):
        """На коротком фрагменте вырезать нечего: он либо рамка, либо текст."""
        short = "Подпись и дата Ионова"
        assert document_furniture.without_splices(short) == (short, 0)

    def test_label_at_paragraph_edge_untouched(self, furniture):
        """Графа в начале абзаца может быть содержанием — не трогается."""
        text = ("Подпись и дата заполняются монтажником. " + "Текст далее. " * 20)
        fixed, cut = document_furniture.without_splices(text)
        assert cut == 0 and fixed == text

    @pytest.mark.parametrize("text", [
        "В этот лист вносятся все изменения конструкторской документации, "
        "и каждая запись подтверждается извещением установленного образца "
        "в порядке, который предусмотрен стандартом предприятия.",
        "Дата приёмки и подпись заносятся в формуляр, а формат А4 здесь "
        "указан как требование к бумаге, на которой печатается документ, "
        "и к оформлению самого листа требований не предъявляется.",
    ])
    def test_common_word_not_cut_from_prose(self, furniture, text):
        """Одно частое слово («лист», «дата») из фразы не вырезается."""
        fixed, cut = document_furniture.without_splices(text)
        assert cut == 0 and fixed == text


class TestFormCaptions:
    """Мелкий шрифт бланка свидетельства о поверке, стоящий между ярлыком
    и значением.

    На проде из 274 чанков со словом «СВИДЕТЕЛЬСТВО О ПОВЕРКЕ» в 89 чанках
    55 документов подпись графы бланка (по приказу 2510) напечатана
    физически между названием поля и его значением, поэтому ярлык и
    значение попадают в разные чанки, и на вопрос о значении возвращается
    ярлык без значения. Рамка ЕСКД такую подпись не знает, а `_SPLICE`
    на стыке «буква — строчная буква» здесь не срабатывает: справа от
    подписи стоит заглавная буква значения.
    """

    def test_label_and_value_become_adjacent(self, furniture):
        fixed, cut = document_furniture.without_splices(CERTIFICATE)
        assert cut >= 1
        assert "наименование и обозначение типа" not in fixed
        assert "единства измерений" not in fixed
        assert "Средство измерений Преобразователь термоэлектрический" in fixed
        assert "Зав. номер 260024" in fixed

    @pytest.mark.parametrize("caption", [
        "полное и краткое наименование организации-поверителя",
        "заводской (серийный) номер или буквенно-цифровое обозначение",
        "наименование или обозначение документа, на основании которого "
        "выполнена поверка",
        "должность руководителя или другого уполномоченного лица",
        "фамилия, инициалы",
        "перечень влияющих факторов с указанием их значений",
    ])
    def test_each_caption_dropped(self, furniture, caption):
        text = ("Знак поверки нанесён на свидетельство. " + caption
                + " Общество с ограниченной ответственностью «Поверка-Сервис». "
                + "Прочий текст страницы. " * 8)
        fixed, cut = document_furniture.without_splices(text)
        assert cut >= 1
        assert caption not in fixed

    def test_content_not_touched(self, furniture):
        """Фраза содержания, похожая на подпись бланка, остаётся целиком."""
        text = ("В формуляр заносятся фамилию, инициалы и должность лица, "
                "принявшего изделие, а также наименование организации, "
                "выполнившей монтаж, и обозначение документа, по которому "
                "выполнены работы на объекте заказчика в полном объёме.")
        fixed, cut = document_furniture.without_splices(text)
        assert cut == 0 and fixed == text


class TestHyphenBreaks:
    @pytest.mark.parametrize("text,expected", [
        ("эксплуата- ции", "эксплуатации"),
        ("поверхно- стей", "поверхностей"),
        ("электронагрева- телей", "электронагревателей"),
        ("ОСНОВ- ными", "ОСНОВными"),
    ])
    def test_joined(self, text, expected):
        assert document_furniture.without_hyphen_breaks(text)[0] == expected

    @pytest.mark.parametrize("text", [
        "научно- технический центр",
        "контрольно- измерительный прибор",
    ])
    def test_compound_kept(self, text):
        """Составное слово разорвано по своему дефису — сращивать нельзя."""
        assert document_furniture.without_hyphen_breaks(text)[0] == text

    def test_dash_between_words_kept(self):
        text = "завода-изготовителя — 24 месяца"
        assert document_furniture.without_hyphen_breaks(text)[0] == text


class TestHyphenBreaksAcrossNewline:
    """Перенос, отданный распознаванием дефисом и переводом строки.

    Такой разрыв на проде встречается 1083 раза в 32 документах в
    текстовых фрагментах и ещё 840 раз в 204 фрагментах 100 документов —
    в чертёжных. Прежний образец требовал пробела или табуляции между
    половинками, поэтому ни один из этих случаев не сращивался.
    """

    def test_joined_when_head_is_cyrillic(self):
        text = ("Указанный ресурс действителен при соблюдении потребите-\n"
                "лем требований документации.")
        fixed, joined = document_furniture.without_hyphen_breaks(text)
        assert joined == 1
        assert "соблюдении потребителем требований" in fixed
        assert "-" not in fixed

    def test_compound_head_kept(self):
        """Составное слово с переводом строки — тоже не сращивается."""
        text = "пуско-\nналадочных работ"
        fixed, joined = document_furniture.without_hyphen_breaks(text)
        assert joined == 0
        assert fixed == text

    @pytest.mark.parametrize("text", [
        "Hand-\nover",
        "ship-\narrived",
        "manufac-\nturing",
    ])
    def test_latin_head_left_alone(self, text):
        """Латинская голова не сращивается: защиты для неё ещё нет."""
        fixed, joined = document_furniture.without_hyphen_breaks(text)
        assert joined == 0
        assert fixed == text

    def test_windows_newline(self):
        fixed, joined = document_furniture.without_hyphen_breaks("поверхно-\r\nстей")
        assert (fixed, joined) == ("поверхностей", 1)

    def test_idempotent(self):
        text = "манометр измери-\nтельный поверен"
        once, first = document_furniture.without_hyphen_breaks(text)
        twice, second = document_furniture.without_hyphen_breaks(once)
        assert first == 1 and second == 0 and twice == once


class TestApply:
    def test_flag_off_changes_nothing(self, furniture_off):
        fragments = [text_fragment(t) for t in FURNITURE[:3]] + [text_fragment(SPLICED)]
        out, _ = finalize_fragments(fragments, doc_id="d1")
        assert len(out) == 4
        assert out[-1].content.text == SPLICED

    def test_furniture_dropped(self, furniture):
        fragments = [text_fragment(t) for t in FURNITURE] + [text_fragment(CONTENT[3])]
        out, _ = finalize_fragments(fragments, doc_id="d1")
        assert len(out) == 1
        assert out[0].content.text == CONTENT[3]

    def test_body_repaired_in_place(self, furniture, joining):
        out, _ = finalize_fragments([text_fragment(SPLICED)], doc_id="d1")
        text = out[0].content.text
        assert "наружных поверхностей." in text
        assert "руководством по эксплуатации." in text
        assert "Подпись и дата" not in text

    def test_no_content_word_lost(self, furniture, joining):
        """Ни одно слово содержания не исчезает — только графы и половинки."""
        import re

        out, _ = finalize_fragments([text_fragment(SPLICED)], doc_id="d1")
        after = out[0].content.text or ""
        words = set(re.findall(r"[А-Яа-яЁё]{3,}", after.lower()))
        for word in re.findall(r"[А-Яа-яЁё]{3,}", SPLICED.lower()):
            if word in ("подпись", "дата"):
                continue  # подпись графы — её и снимали
            if any(word in whole for whole in words if len(whole) > len(word)):
                continue  # половина слова, разорванного переносом: сращена
            assert word in words, word

    def test_table_fragment_untouched(self, furniture):
        from core.models.contract import TableData

        fragment = text_fragment("Формат А4")
        fragment.content.table_data = TableData(headers=["Формат"], rows=[["А4"]])
        out, _ = finalize_fragments([fragment], doc_id="d1")
        assert len(out) == 1

    def test_drawing_fragment_untouched(self, furniture):
        fragment = text_fragment("Формат А4")
        fragment.type = "drawing"
        out, _ = finalize_fragments([fragment], doc_id="d1")
        assert len(out) == 1

    def test_drawing_hyphen_break_joined(self, furniture, joining):
        """У чертежа переносы сращиваются: 840 случаев в 204 фрагментах прода."""
        fragment = text_fragment("Трубопровод из полиэтилена высокого давле-\nния")
        fragment.type = "drawing"
        out, _ = finalize_fragments([fragment], doc_id="d1")
        assert len(out) == 1
        assert out[0].content.text == "Трубопровод из полиэтилена высокого давления"

    def test_drawing_frame_still_not_dropped(self, furniture, joining):
        """Расширение не добавило чертежу ни снятия рамки, ни вырезания граф."""
        frame = text_fragment("Формат А4")
        frame.type = "drawing"
        spliced = text_fragment(SPLICED)
        spliced.type = "drawing"
        out, _ = finalize_fragments([frame, spliced], doc_id="d1")
        assert len(out) == 2
        assert out[0].content.text == "Формат А4"
        assert "Подпись и дата" in (out[1].content.text or "")

    def test_drawing_hyphen_break_stops_promising_exact_offsets(self, furniture, joining):
        from core.models.contract import SourceSpan

        raw = "Трубопровод из полиэтилена высокого давле-\nния"
        fragment = text_fragment(raw)
        fragment.type = "drawing"
        fragment.position.source_spans = [SourceSpan(
            source_unit_id="unit-4", page=1, char_start=0, char_end=len(raw))]
        fragment.position.text_offsets_exact = True
        out, _ = finalize_fragments([fragment], doc_id="d1")
        assert out[0].position.text_offsets_exact is False

    def test_idempotent(self, furniture, joining):
        out, _ = finalize_fragments([text_fragment(SPLICED)], doc_id="d1")
        once = out[0].content.text
        again, _ = finalize_fragments(out, doc_id="d1")
        assert again[0].content.text == once

    def test_rewritten_text_stops_promising_exact_offsets(self, furniture, joining):
        """Переписанный текст снимает обещание точных смещений.

        Смещения измеряются по тексту исходного блока (`source_contract`), и
        снятие графы или сращивание переноса делает их неверными: `char_end`
        указывает за конец текста. Цитата на скане бланка подсвечивалась со
        сдвигом или вылезала за свой текст.
        """
        from core.models.contract import SourceSpan

        fragment = text_fragment(SPLICED)
        fragment.position.source_spans = [SourceSpan(
            source_unit_id="unit-1", page=1, char_start=0, char_end=len(SPLICED))]
        fragment.position.text_offsets_exact = True
        out, _ = finalize_fragments([fragment], doc_id="d1")
        assert out[0].content.text != SPLICED
        assert out[0].position.text_offsets_exact is False
        # Спаны остаются: из какой единицы источника взят фрагмент, по-прежнему известно.
        assert [span.source_unit_id for span in out[0].position.source_spans] == ["unit-1"]

    def test_length_changing_repair_also_stops_promising_exact_offsets(self, monkeypatch):
        """Починка, изменившая длину текста, снимает обещание так же, как рамка.

        Сведение разорванного слова укорачивает текст, а смещения измерены по
        исходному блоку: тот же устаревший `char_end`, что и у рамки.
        """
        from core.models.contract import SourceSpan

        monkeypatch.setattr(settings, "SCAN_NO_MIXED_SCRIPT_WORDS", True)
        monkeypatch.setattr(settings, "SCAN_DESIGNATION_REPAIR", True)
        monkeypatch.setattr(settings, "SCAN_DROP_FRAME_FURNITURE", False)
        monkeypatch.setattr(settings, "SCAN_JOIN_HYPHENATED_WORDS", False)

        fragment = text_fragment("начало")
        fragment.position.source_spans = [SourceSpan(
            source_unit_id="unit-3", page=1, char_start=0, char_end=6)]
        fragment.position.text_offsets_exact = True

        def shorten(target):
            target.content.text = "начал"
            return [object()]

        monkeypatch.setattr("core.providers.mixed_script.repair_fragment", shorten)
        out, _ = finalize_fragments([fragment], doc_id="d1")
        assert out[0].content.text == "начал"
        assert out[0].position.text_offsets_exact is False

    def test_untouched_text_keeps_exact_offsets(self, furniture, joining):
        """Текст, который правило не меняло, обещание сохраняет."""
        from core.models.contract import SourceSpan

        clean = CONTENT[3]
        fragment = text_fragment(clean)
        fragment.position.source_spans = [SourceSpan(
            source_unit_id="unit-2", page=1, char_start=0, char_end=len(clean))]
        fragment.position.text_offsets_exact = True
        out, _ = finalize_fragments([fragment], doc_id="d1")
        assert out[0].content.text == clean
        assert out[0].position.text_offsets_exact is True



class TestHeadingWithBody:
    HEADINGS = [
        "1.1 НАЗНАЧЕНИЕ И КЛАССИФИКАЦИЯ",
        "2.5 МОНТАЖ",
        "9. Производственная деятельность",
        "4. Общие указания",
        "3.6 Управление конфигурацией",
        "4 ГАРАНТИЙНЫЕ ОБЯЗАТЕЛЬСТВА 2 ОСНОВНЫЕ ТЕХНИЧЕСКИЕ ХАРАКТЕРИСТИКИ",
        "Таблица 5 - Нормы расхода лакокрасочных материалов",
        "Таблица 3б – Комплект ЗИП ТАДУ 681821.001.01.70.00",
        "Приложение В",
        "Рис. В.1",
    ]
    NOT_HEADINGS = [
        "4. При монтаже убедиться, что кабель обесточен.",
        "6.3 Гарантийный СРОК хранения в упаковке завода-изготовителя — 24 месяца "
        "с момента отгрузки оборудования.",
        "Ошиновка– изделие III категории сейсмостойкости по НП-031-01.",
        "- для кабелей условия Ж по ГОСТ 23216.",
        "ОКПД2 26.51.51.110",
        "189260196",
    ]

    @pytest.fixture
    def heading(self, monkeypatch):
        monkeypatch.setattr(settings, "DOCUMENT_PARTS_MARKUP", True)
        monkeypatch.setattr(settings, "DOCUMENT_HEADING_WITH_BODY", True)

    @pytest.mark.parametrize("line", HEADINGS)
    def test_heading(self, heading, line):
        assert document_parts.is_heading_line(line) is True
        assert document_parts.is_label_line(line) is True

    @pytest.mark.parametrize("line", NOT_HEADINGS)
    def test_not_heading(self, heading, line):
        assert document_parts.is_heading_line(line) is False

    @pytest.mark.parametrize("line", HEADINGS)
    def test_flag_off_keeps_old_behaviour(self, monkeypatch, line):
        monkeypatch.setattr(settings, "DOCUMENT_HEADING_WITH_BODY", False)
        # Прежний образец ярлыка заголовка не узнавал — в этом и был изъян.
        assert document_parts.is_label_line(line) is False

    def test_label_line_still_works(self, heading):
        assert document_parts.is_label_line("Знак поверки:") is True
        assert document_parts.is_label_line("4.") is True
