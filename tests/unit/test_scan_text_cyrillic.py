"""
Испорченная путаницей алфавитов надпись скана чертежа.

Строки взяты с разобранных листов ТАДУ 061360.001.00 СБ и
ТАДУ 301129.001.00 СБ и из корпуса прода — ничего не придумано. Сами листы
в тесты не кладутся.

Проверяется три вещи:

* обозначение и стандарт восстанавливаются по строгой форме
  (`SCAN_DESIGNATION_REPAIR`), а испорченное слово обычного текста — нет:
  правдоподобно неверное требование хуже видимо испорченного;
* страница, у которой распознавание испорчено, опознаётся по измеримому
  свойству самой страницы (`DRAWING_RESCAN_POOR_OCR_PAGES`), а чистая
  страница и английская страница — нет;
* слово из двух алфавитов сводится к одному только с опознанным прочтением
  (`SCAN_NO_MIXED_SCRIPT_WORDS`), а язык выбирается по строке, так что
  английская строка остаётся английской (`OCR_PER_LINE_LANGUAGE`).
"""

from dataclasses import dataclass, field
from typing import List

from core.models.contract import Content, Fragment, Position, Provenance
from core.providers import designation_repair, mixed_script, per_line_language
from core.scan_text_rescan import PageText, page_verdict, poor_pages

# ------------------------------------------------- строки с разобранных листов
# Текст технических требований ТАДУ 061360.001.00 СБ, как его отдало
# распознавание: испорчено и обозначение, и сами слова.
REQUIREMENTS_061360 = (
    "7 * Размеры для справок 2 Гайки поз, 6 затянуть 00 упора дез применения "
    "инструмента, 3 Момент затяжки винтов поз. 7 - 527 Не. 4. Маркировать "
    "шрифтом 7-lp3 ГОСТ 26.008-85 на крышке поз. № = - условное обозначение "
    "изделия: 8 ad ‘ky’ при поставках по территории Россиц, ресл. Беларусь и "
    "за рубеж, s кроме AI \"Зль-Дабаа’. 9) ‘kU-1\" при поставках на АЭС "
    "“Эль-Дабаа“. — 5 Маркуровать на wmyyepax корпуса поз. 2 стрелки "
    "направления подачи рабочей сребы. 9 Остальные требования-по TARY "
    "06196000119,"
)
# ТАДУ 301129.001.00 СБ: классификация по НП и прихватка сваркой.
REQUIREMENTS_301129 = (
    "2. Классификация по НП-001-15/нП-о01-97 3. Деталь поз. 10 прихватить "
    "оргонодугобой сваркой неплавищимся электродом"
)


def _text_fragment(page, confidence, text, method="ocr_layer"):
    return Fragment(
        fragment_id=f"f{page}-{confidence}", type="text",
        content=Content(text=text),
        position=Position(page=page, bbox=[0.0, 0.0, 1.0, 1.0], order=1),
        confidence=confidence, completeness=1.0,
        provenance=Provenance(method=method, strategy_level=5, source="scanned_pdf"),
    )


# ------------------------------------------------- правка обозначений: правит
def test_standard_number_restored_from_lowercase_and_letter_zero():
    """«нП-о01-97»: регистр и «о» на месте нуля заданы формой обозначения."""
    repaired, changes = designation_repair.repair_text(REQUIREMENTS_301129)
    assert "НП-001-15/НП-001-97" in repaired
    assert [(c.before, c.after) for c in changes] == [("нП-о01-97", "НП-001-97")]


def test_eskd_document_kind_restored_from_latin():
    """«ТАДУ 405210.001TY»: код документа — из закрытого списка ГОСТ 2.102."""
    repaired, _ = designation_repair.repair_text(
        "Остальные требования по ТАДУ 405210.001TY"
    )
    assert repaired.endswith("ТАДУ 405210.001ТУ")


def test_eskd_code_restored_only_to_known_organisation():
    """Латинская «T» в «ТАДУ» правится, а «TAAY» остаётся как есть."""
    repaired, _ = designation_repair.repair_text("Чертёж TАДУ 061360.001.00 СБ")
    assert repaired == "Чертёж ТАДУ 061360.001.00 СБ"
    kept, changes = designation_repair.repair_text("Чертёж TAAY 421451.003.05")
    assert kept == "Чертёж TAAY 421451.003.05" and not changes


def test_steel_and_wire_grades_restored_by_reference():
    """«12X18H10T», «CB-04X19H11M3», «12Х18Н1ОТ», «Св-04Х19Н11МЗ»."""
    for before, after in (
        ("Материал 12X18H10T", "Материал 12Х18Н10Т"),
        ("Материал 12Х18Н1ОТ", "Материал 12Х18Н10Т"),
        ("Проволока CB-04X19H11M3", "Проволока Св-04Х19Н11М3"),
        ("Проволока Св-04Х19Н11МЗ", "Проволока Св-04Х19Н11М3"),
        ("Сталь 08Х18н10Т", "Сталь 08Х18Н10Т"),
    ):
        assert designation_repair.repair_text(before)[0] == after


def test_font_designation_restored():
    """«5-ПрЗ» по ГОСТ 26.008-85: «З» после «Пр» — это цифра."""
    repaired, _ = designation_repair.repair_text(
        "Маркировать шрифтом 5-ПрЗ ГОСТ 26.008-85"
    )
    assert "шрифтом 5-Пр3 " in repaired


def test_standard_prefix_case_restored():
    """«гОСт 23170-78», «Гост 19903-2015», «ГОСТ P 50.06.01-2017»."""
    for before, after in (
        ("по гОСт 23170-78", "по ГОСТ 23170-78"),
        ("по Гост 19903-2015", "по ГОСТ 19903-2015"),
        ("по ГОСТ P 50.06.01-2017", "по ГОСТ Р 50.06.01-2017"),
        ("по СанПин 2.6.1.24-03", "по СанПиН 2.6.1.24-03"),
        ("по ТУЗ580-388-00217053-2008", "по ТУ3580-388-00217053-2008"),
    ):
        assert designation_repair.repair_text(before)[0] == after


# --------------------------------------------- правка обозначений: не правит
def test_broken_words_of_prose_left_as_they_are():
    """
    Слова требований остаются испорченными: у слова нет строгой формы, и
    правка по похожести подменила бы смысл требования.
    """
    repaired, changes = designation_repair.repair_text(REQUIREMENTS_061360)
    for broken in ("затянуть 00 упора дез применения", "Маркуровать",
                   "wmyyepax", "рабочей сребы", "Россиц, ресл."):
        assert broken in repaired
    assert "оргонодугобой" in designation_repair.repair_text(
        REQUIREMENTS_301129)[0]
    assert all(change.pattern != "mixed_script" for change in changes)


def test_numbers_are_never_guessed():
    """«527 Не» не становится моментом затяжки, «06196000119» — номером."""
    repaired, _ = designation_repair.repair_text(REQUIREMENTS_061360)
    assert "527 Не." in repaired
    assert "TARY 06196000119" in repaired


def test_designation_too_broken_for_homoglyphs_left_alone():
    """«TARY»: латинской «R» не соответствует «Д», прочтения нет."""
    text = "Остальные требования-по TARY 06196000119"
    assert designation_repair.repair_text(text) == (text, [])


def test_font_designation_with_lost_letter_left_alone():
    """«7-lp3»: «l» и «p» — это не «Пр», а две буквы вместо одной."""
    repaired, _ = designation_repair.repair_text(
        "Маркировать шрифтом 7-lp3 ГОСТ 26.008-85"
    )
    assert "шрифтом 7-lp3 " in repaired


def test_latin_ost_not_turned_into_cyrillic_ost():
    """
    «OCT 5582-75» читается и как «ОСТ», и как «ГОСТ» с потерянной буквой.
    Прочтений два — значит участок остаётся как был: «ОСТ 5582-75» вместо
    «ГОСТ 5582-75» было бы правдоподобной подменой стандарта.
    """
    text = "Прокат по OCT 5582-75"
    assert designation_repair.repair_text(text) == (text, [])


def test_unknown_grade_left_alone():
    """«12X1BH10T» и «12Х48Н1ОТ» — не марки: в справочнике их нет."""
    for text in ("Материал 12X1BH10T", "Материал 12Х48Н1ОТ",
                 "Проволока Св-04х19н1им3"):
        assert designation_repair.repair_text(text) == (text, [])


def test_ordinary_words_and_measurements_left_alone():
    """«в зоне», «10 мм», «20 мА», «сто 58» — не обозначения."""
    for text in ("Контроль в зоне сварного шва", "Размер 10мм по чертежу",
                 "Выход 20мА, предел 16Мпа", "НПО «Турбоатом»",
                 "Формат A3 Лист 2 Копировал", "Шероховатость Ra 1,6"):
        assert designation_repair.repair_text(text) == (text, [])


def test_lowercase_prefix_before_a_range_is_ordinary_speech():
    """
    «сто 50-60 деталей» — это русская речь, а не «СТО 50-60»: префикс
    целиком строчными ссылкой не делает и диапазон в номере. Прописная
    буква в префиксе нужна всегда, иначе правится то, что было верным.
    """
    for text in ("Партия сто 50-60 деталей", "Нужно ту 100-5 проверить",
                 "Передать рд 12-3 заказчику", "Отмерить сто 100 мм"):
        assert designation_repair.repair_text(text) == (text, [])


def test_english_text_left_alone():
    """В английской строке латинская «OCT» — это месяц, а не «ОСТ»."""
    text = "Tested to ISO 9001 and IEC 61010 in OCT 2020"
    assert designation_repair.repair_text(text) == (text, [])


def test_repair_is_recorded_in_payload_and_not_as_a_warning():
    """
    Правка — не повод считать текст неполным. Для RAG любая пометка
    фрагмента означает «текст неполный», и документ с восстановленным
    обозначением терял бы полноту за то, что стал вернее. Поэтому замены
    пишутся в payload, а `warnings` не трогаются.
    """
    fragment = _text_fragment(1, 0.7, "Чертёж ТАДУ 405210.001TY, 12X18H10T")
    changes = designation_repair.repair_fragment(fragment)
    assert len(changes) == 2 and not fragment.warnings
    recorded = fragment.content.structured_payload[designation_repair.PAYLOAD_KEY]
    assert {item["after"] for item in recorded} == {
        "ТАДУ 405210.001ТУ", "12Х18Н10Т"}

    mixed = _text_fragment(1, 0.7, "Код ККS изделия")
    assert mixed_script.repair_fragment(mixed) and not mixed.warnings
    assert mixed.content.structured_payload[mixed_script.PAYLOAD_KEY] == [
        {"before": "ККS", "after": "KKS"}]


# ------------------------------------------------------ порог перечитывания
def test_corrupted_page_crosses_the_threshold():
    """
    Страница требований 061360 (оценка распознавания 0,709) за порогом по
    оценке. Признак путаницы алфавитов её не ловит: испорчено тут не
    алфавитом, а похожестью букв («Маркуровать», «сребы»), целиком
    латинских слов всего 4%. Поэтому признаков два, и достаточно любого.
    """
    verdict = page_verdict([PageText(0.709, REQUIREMENTS_061360)])
    assert verdict.rescan and verdict.reasons == ["ocr_score"]
    assert verdict.mixed_share < 0.05


def test_garbled_page_crosses_the_threshold_by_script_alone():
    """
    Лист, который распознавание прочитало латиницей целиком, но собой
    доволен: порог по алфавиту нужен ровно для него.
    """
    verdict = page_verdict([PageText(0.93, (
        "Cozobopa MEXHONOZUYECKUX npouеccob. Mapxupobame na wmyyepax "
        "kopnyca nos. 2 cmpenku nanpabnenuя noдaчu paбoчeй cpeды."
    ))])
    assert verdict.rescan and verdict.reasons == ["script_mix"]


def test_clean_russian_page_does_not_cross_the_threshold():
    """Чистая страница требований не перечитывается."""
    verdict = page_verdict([PageText(0.95, (
        "1. Размеры для справок. 2. Гайки поз. 6 затянуть до упора без "
        "применения инструмента. 3. Маркировать шрифтом 7-Пр3 по "
        "ГОСТ 26.008-85. 4. Остальные требования по ТАДУ 061360.001ТУ."
    ))])
    assert not verdict.rescan and not verdict.reasons


def test_english_page_does_not_cross_the_threshold_by_script():
    """
    Английская страница — не порча: латиница там целыми строками, а признак
    смотрит на долю латиницы только в русской странице.
    """
    verdict = page_verdict([PageText(0.93, (
        "Switching device with cold junction compensation. The device shall "
        "be tested in accordance with ISO 9001 and marked For NPP."
    ))])
    assert not verdict.rescan


def test_short_page_is_not_judged():
    """На двух словах доли ничего не значат: страница не перечитывается."""
    assert not page_verdict([PageText(0.09, "Формат A3")]).rescan


def test_poor_pages_collects_only_recognition_fragments():
    """Берутся страницы записей распознавания, одна страница — один вердикт."""
    pages = poor_pages([
        _text_fragment(1, 0.709, REQUIREMENTS_061360),
        _text_fragment(2, 0.97, (
            "Технические характеристики. Расчётное давление среды "
            "избыточное 0,5 МПа. Класс защиты IP 67 по ГОСТ 14254-2015."
        )),
    ])
    assert list(pages) == [1]
    assert pages[1].as_dict()["reasons"]


# --------------------------------------------- слово из двух алфавитов
def test_mixed_script_word_resolved_only_when_recognised():
    """«ТAДУ», «HТЛ», «ККS» сводятся к одному алфавиту по справочнику."""
    for before, after in (("Чертёж ТAДУ 061360", "Чертёж ТАДУ 061360"),
                          ("Завод HТЛ-ПРИБОР", "Завод НТЛ-ПРИБОР"),
                          ("Код ККS изделия", "Код KKS изделия")):
        assert mixed_script.repair_text(before)[0] == after


def test_legitimately_mixed_notation_left_alone():
    """«Rизм», «Pисп», «KMnO» смешаны по самой записи: прочтения нет."""
    for text in ("Сопротивление Rизм при 20 °C", "Давление Pисп 1,5 МПа",
                 "Раствор КMnO 4", "Подпись Аndreev"):
        assert mixed_script.repair_text(text) == (text, [])


def test_english_and_russian_words_left_alone():
    """Чистое слово одного алфавита не трогается ни в какой строке."""
    for text in ("Switching device with cold junction compensation",
                 "Маркировать на штуцерах корпуса",
                 "Для АЭС For NPP NTL-PRIBOR"):
        assert mixed_script.repair_text(text) == (text, [])


# --------------------------------------------- выбор языка по строке
@dataclass
class _Line:
    """Строка распознавания: столько, сколько нужно выбору языка."""

    text: str
    bbox: List[float] = field(default_factory=lambda: [0.0, 0.0, 1.0, 1.0])
    confidence: float = 0.0
    rotation: float = 0.0
    parts: List["_Line"] = field(default_factory=list)


def test_english_line_keeps_the_more_confident_english_reading():
    """Английская строка двуязычного листа остаётся английской."""
    russian = [_Line("Switchinq dеviсе with соld junсtiоn", [0, 0.1, 1, 0.2], 0.41),
               _Line("Устройство коммутационное", [0, 0.3, 1, 0.4], 0.94)]
    english = [_Line("Switching device with cold junction", [0, 0.1, 1, 0.2], 0.92),
               _Line("YcTpoucTBo KoMMyTaunoHHoe", [0, 0.3, 1, 0.4], 0.52)]
    chosen = per_line_language.choose([russian, english])
    assert [line.text for line in chosen] == [
        "Switching device with cold junction", "Устройство коммутационное"]


def test_line_place_on_the_sheet_comes_from_the_primary_reading():
    """Рамка и угол письма — из основного чтения, текст — из уверенного."""
    russian = [_Line("Mapkupobamb", [0.1, 0.1, 0.9, 0.2], 0.40, rotation=90.0)]
    english = [_Line("Marking", [0.11, 0.1, 0.91, 0.2], 0.80)]
    line = per_line_language.choose([russian, english])[0]
    assert line.text == "Marking"
    assert line.bbox == [0.1, 0.1, 0.9, 0.2] and line.rotation == 90.0


def test_reading_without_a_counterpart_adds_no_line():
    """Чужое чтение не может добавить страницу строк из перекрытий."""
    russian = [_Line("Технические требования", [0, 0.1, 1, 0.2], 0.90)]
    english = [_Line("Technical requirements", [0, 0.1, 1, 0.2], 0.95),
               _Line("Sheet 1", [0, 0.8, 1, 0.9], 0.99)]
    chosen = per_line_language.choose([russian, english])
    assert len(chosen) == 1 and chosen[0].text == "Technical requirements"


def test_equal_confidence_keeps_the_primary_reading():
    """На равных побеждает первая модель: выбор не дёргается от шума."""
    russian = [_Line("Лист 2", [0, 0.1, 1, 0.2], 0.90)]
    english = [_Line("Jlucm 2", [0, 0.1, 1, 0.2], 0.91)]
    assert per_line_language.choose([russian, english])[0].text == "Лист 2"
