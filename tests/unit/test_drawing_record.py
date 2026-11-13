"""
Запись чертёжного листа, собранная по частям (`DRAWING_RECORD_COMPOSED`).

Все образцы здесь — настоящие: тексты записей и графы штампа взяты из
разбора листов СМ1638.11.05.00.00.00 СБ и ТАДУ 302318.207.00 СБ на
рабочей системе (очередь `platform_parser_ingest_queue.parser_payload`).
Придуманных листов в этом файле нет: сборка проверяется ровно на том, на
чём заказчик её и забраковал.
"""

import pytest

from core import drawing_record
from core.config import settings
from core.contract_finalize import finalize_fragments
from core.ladder import drawing_route
from core.models.parse_result import ParsedBlock, ParseResult
from core.normalizer import TextNormalizer
from core.providers.title_block import label
from core.models.contract import Content, DrawingField, Fragment, Position, Provenance

# Настоящий текст записи листа СМ1638.11.05.00.00.00 СБ — тот самый
# «перемешанный» вид, с которого задача и началась.
SM1638_TEXT = "\n".join([
    "СМ1638.11.05.00.00.00 СБ",
    "Г-Г(1:1)",
    "2 месга",
    "23", "1", "2", "3", "1", "10", "20", "21",
    "⌀8,5",
    "2 отв.",
    "25",
    "В-В(1:1)",
    "2 месга",
    "24", "11", "9", "110", "65", "98", "22",
    "⌀10",
    "2 отв.",
    "Инв. № подл.",
    "Подп. и дата",
])

# Настоящий текст записи листа ТАДУ 302318.207.00 (начало).
TADU_TEXT = "\n".join([
    "ТАДУ 302318.207.00 СБ",
    "Справ. N",
    "Перв. применение",
    "A", "6", "N°1", "N°1", "5", "R65", "N°1", "8", "2", "B", "A", "7",
    "A (2:1)",
    "Разрез клапанов поз.6 и 7",
    "1. Размеры с неуказанными предельными отклонениями для справок.",
    "629",
    "03.2020",
])

# Графы штампа того же листа, как их принёс разбор.
# Одна строка листа. Сборка идёт только у фрагмента с текстом листа,
# поэтому в тестах частей текст листа всегда непустой.
SHEET_LINE = "25"

TADU_STAMP = {
    "mass": "26,02",
    "name": "Трубная обвязка",
    "checked": "Постнов",
    "approved": "Глинко",
    "developed": "Прохоров",
    "designation": "ТАДУ 302318.207.00 СБ",
    "norm_control": "Домрачева",
    "tech_control": "Юханов",
}


def _fragment(text, fields=(), payload=None):
    """Фрагмент чертежа, как его строит `drawing_processor.build_fragment`."""
    body = {"transcript_line_count": len(text.split("\n")) if text else 0}
    body.update(payload or {})
    return Fragment(
        fragment_id="f1",
        type="drawing",
        content=Content(
            text=text or None,
            image_ref="s3://bucket/sheet.png",
            structured_drawing_fields=[DrawingField(**f) for f in fields],
            structured_payload=body,
        ),
        position=Position(page=1, bbox=[0, 0, 1, 1]),
        confidence=0.9,
        completeness=1.0,
        provenance=Provenance(method="vlm_full_page", strategy_level=7,
                              source="drawing"),
    )


@pytest.fixture
def composed(monkeypatch):
    monkeypatch.setattr(settings, "DRAWING_RECORD_COMPOSED", True, raising=False)


@pytest.fixture
def plain(monkeypatch):
    monkeypatch.setattr(settings, "DRAWING_RECORD_COMPOSED", False, raising=False)


# --------------------------------------------------------------- флаг выключен
class TestFlagOff:
    def test_text_is_byte_identical(self, plain):
        fragment = _fragment(SM1638_TEXT, payload={"title_block": TADU_STAMP})
        drawing_record.apply([fragment])
        assert fragment.content.text == SM1638_TEXT

    def test_payload_is_untouched(self, plain):
        fragment = _fragment(TADU_TEXT, payload={"title_block": TADU_STAMP})
        before = dict(fragment.content.structured_payload)
        drawing_record.apply([fragment])
        assert fragment.content.structured_payload == before
        assert drawing_record.COMPOSED_KEY not in fragment.content.structured_payload

    def test_finalize_keeps_text_byte_identical(self, plain):
        """Через настоящий вызов в конвейере, а не только через `apply`."""
        fragment = _fragment(TADU_TEXT, payload={"title_block": TADU_STAMP})
        finalize_fragments([fragment], "doc-1", extract=False)
        assert fragment.content.text == TADU_TEXT


# ------------------------------------------------------------------ без потерь
class TestNoLoss:
    @pytest.mark.parametrize("text", [SM1638_TEXT, TADU_TEXT])
    def test_every_line_still_reaches_the_record(self, composed, text):
        fragment = _fragment(text, payload={"title_block": TADU_STAMP})
        drawing_record.apply([fragment])
        record = fragment.content.text
        for line in text.split("\n"):
            line = line.strip()
            if not line or drawing_record._fold(line) in drawing_record._FURNITURE_FOLDED:
                continue
            assert line in record, line

    @pytest.mark.parametrize("text", [SM1638_TEXT, TADU_TEXT])
    def test_lines_stay_in_reading_order_as_one_block(self, composed, text):
        """
        Строки листа остаются одним куском и в прежнем порядке: рядом
        стоящие «⌀8,5» и «2 отв.» так и остаются рядом.
        """
        fragment = _fragment(text, payload={"title_block": TADU_STAMP})
        drawing_record.apply([fragment])
        kept, _ = drawing_record._without_furniture(text.split("\n"))
        block = "\n".join(line for line in kept if line.strip())
        assert block in fragment.content.text

    def test_diameter_and_its_count_stay_adjacent(self, composed):
        fragment = _fragment(SM1638_TEXT)
        drawing_record.apply([fragment])
        assert "⌀8,5\n2 отв." in fragment.content.text
        assert "⌀10\n2 отв." in fragment.content.text

    def test_misread_word_is_not_corrected_away(self, composed):
        """«2 месга» — ошибка чтения модели, но терять её нельзя."""
        fragment = _fragment(SM1638_TEXT)
        drawing_record.apply([fragment])
        assert "2 месга" in fragment.content.text

    def test_fragment_without_sheet_text_is_left_alone(self, composed):
        """
        Фрагмент без текста листа не собирается: его запись отдаёт в индекс
        адаптер RAG, и он помечает неподтверждённые значения словами
        «требует сверки». Проза этих пометок не несёт.
        """
        fragment = _fragment("", fields=[
            {"category": "callout", "value": "Шток", "confidence": 0.4},
        ], payload={"title_block": TADU_STAMP})
        drawing_record.apply([fragment])
        assert fragment.content.text is None
        assert drawing_record.COMPOSED_KEY not in fragment.content.structured_payload

    def test_whitespace_only_text_counts_as_no_text(self, composed):
        fragment = _fragment("   \n  \n", fields=[
            {"category": "callout", "value": "Шток"},
        ])
        drawing_record.apply([fragment])
        assert drawing_record.COMPOSED_KEY not in fragment.content.structured_payload

    def test_nothing_to_compose_leaves_fragment_alone(self, composed):
        fragment = _fragment("")
        assert drawing_record.compose_fragment(fragment) is None
        assert fragment.content.text is None


# ------------------------------------------------- числа и обозначения как есть
class TestValuesVerbatim:
    @pytest.mark.parametrize("value", [
        "⌀8,5", "⌀10", "110", "65", "98", "1087±3", "⌀18x2,5", "26,02",
        "ТАДУ 302318.207.00 СБ", "СМ1638.11.05.00.00.00 СБ",
    ])
    def test_value_appears_character_for_character(self, composed, value):
        text = SM1638_TEXT + "\n" + TADU_TEXT + "\n1087±3\n⌀18x2,5"
        fragment = _fragment(text, payload={"title_block": TADU_STAMP})
        drawing_record.apply([fragment])
        assert value in fragment.content.text

    def test_mass_keeps_its_comma_and_gains_no_unit(self, composed):
        fragment = _fragment(TADU_TEXT, payload={"title_block": TADU_STAMP})
        drawing_record.apply([fragment])
        assert "Масса — 26,02." in fragment.content.text
        assert "26,02 кг" not in fragment.content.text
        assert "26.02" not in fragment.content.text

    def test_tolerance_is_named_not_glued_to_the_number(self, composed):
        fragment = _fragment(SHEET_LINE, fields=[
            {"category": "size", "value": "⌀200", "tolerance": "±2"},
        ])
        drawing_record.apply([fragment])
        assert "⌀200 (допуск ±2)" in fragment.content.text

    def test_tolerance_already_inside_the_value_is_not_repeated(self, composed):
        fragment = _fragment(SHEET_LINE, fields=[
            {"category": "size", "value": "⌀200±2", "tolerance": "±2"},
        ])
        drawing_record.apply([fragment])
        assert "допуск" not in fragment.content.text

    def test_bare_digit_tolerance_is_not_swallowed_by_the_value(self, composed):
        """Допуск «2» входит в «⌀200» как подстрока, но назван не был."""
        fragment = _fragment(SHEET_LINE, fields=[
            {"category": "size", "value": "⌀200", "tolerance": "2"},
        ])
        drawing_record.apply([fragment])
        assert "⌀200 (допуск 2)" in fragment.content.text

    def test_second_graph_reading_is_not_swallowed_as_a_substring(self, composed):
        """«2,7» не считается сказанным потому, что входит в «12,75»."""
        fragment = _fragment(SHEET_LINE, payload={"title_block": {
            "mass": "12,75", "масса": "2,7",
        }})
        drawing_record.apply([fragment])
        assert "Масса — 12,75; 2,7." in fragment.content.text


# ------------------------------------------------------------- основная надпись
class TestStamp:
    def test_mass_reaches_the_record_with_its_label(self, composed):
        """
        Главное, чего в записи не было: масса из штампа. По прежней записи
        на «какая масса» ответить было нечем — значения в ней не было вовсе.
        """
        assert "26,02" not in TADU_TEXT
        fragment = _fragment(TADU_TEXT, payload={"title_block": TADU_STAMP})
        drawing_record.apply([fragment])
        assert "Масса — 26,02" in fragment.content.text

    def test_designation_and_name_open_the_record(self, composed):
        fragment = _fragment(TADU_TEXT, payload={"title_block": TADU_STAMP})
        drawing_record.apply([fragment])
        assert fragment.content.text.startswith("Основная надпись листа.")
        assert "Обозначение — ТАДУ 302318.207.00 СБ." in fragment.content.text
        assert "Наименование — Трубная обвязка." in fragment.content.text

    def test_signers_are_named_by_what_they_did(self, composed):
        fragment = _fragment(TADU_TEXT, payload={"title_block": TADU_STAMP})
        drawing_record.apply([fragment])
        assert ("Лист подписали: разработал Прохоров, проверил Постнов, "
                "технический контроль Юханов, нормоконтроль Домрачева, "
                "утвердил Глинко." in fragment.content.text)

    def test_graph_named_in_russian_is_the_same_graph(self, composed):
        """
        Настоящий штамп штанги АЕГН.01.00017.111.0015 СБ: разбор назвал
        графы по-русски («масса»), а не внутренними ключами. Без приведения
        имени графа уходила в запись строчной и не в своём порядке.
        """
        fragment = _fragment(SHEET_LINE, payload={"title_block": {
            "масса": "2,7", "листов": "1", "литера": "И", "масштаб": "1:2",
            "организация": 'ООО "ФИНПРОМАТОМ"',
        }})
        drawing_record.apply([fragment])
        record = fragment.content.text
        assert ("Масса — 2,7. Масштаб — 1:2. Литера — И. Листов — 1. "
                'Организация — ООО "ФИНПРОМАТОМ".' in record)

    def test_two_namings_of_one_graph_are_both_kept(self, composed):
        fragment = _fragment(SHEET_LINE, payload={"title_block": {
            "mass": "2,7", "масса": "2,75",
        }})
        drawing_record.apply([fragment])
        assert "Масса — 2,7; 2,75." in fragment.content.text

    def test_corrected_value_is_preferred_over_the_raw_one(self, composed):
        """
        Разбор кладёт в payload ответ модели как есть, а `drawing_text`
        применяет только к значениям полей («Обозначение: …»). Заказчику
        надо показать исправленную копию, а не ту, что рядом с ней.
        """
        fragment = _fragment(SHEET_LINE, fields=[
            {"category": "title_block", "value": "Обозначение: ТАДУ 421451.001"},
        ], payload={"title_block": {"designation": "ТАДУ 421451001"}})
        drawing_record.apply([fragment])
        assert "Обозначение — ТАДУ 421451.001." in fragment.content.text
        assert "ТАДУ 421451001" not in fragment.content.text

    def test_graph_without_a_corrected_copy_falls_back_to_payload(self, composed):
        fragment = _fragment(SHEET_LINE, fields=[], payload={
            "title_block": {"mass": "26,02"}})
        drawing_record.apply([fragment])
        assert "Масса — 26,02." in fragment.content.text

    def test_handwritten_takes_the_handwritten_correction(self, composed):
        """
        Одна графа бывает прочитана и печатной в штампе, и рукописной в
        боковой колонке — это два разных значения, и путать их нельзя.
        """
        fragment = _fragment(SHEET_LINE, fields=[
            {"category": "title_block", "value": "Инв. № подл.: 1012-02 (рукописно)"},
        ], payload={"title_block_handwritten": {"inventory": "1012-O2"}})
        drawing_record.apply([fragment])
        assert "Инв. № подл. — 1012-02." in fragment.content.text

    def test_unknown_graph_also_gets_its_corrected_value(self, composed):
        """
        Непонятая графа ищется по свёрнутой записи и у поля, и у payload —
        иначе исправленная копия такой графы не находилась бы никогда, а
        запасной путь срабатывал бы по случайности.
        """
        fragment = _fragment(SHEET_LINE, fields=[
            {"category": "title_block", "value": "Код заказа: ВД-005"},
        ], payload={"title_block": {"Код заказа": "ВД-ОО5"}})
        drawing_record.apply([fragment])
        assert "Код заказа — ВД-005." in fragment.content.text
        assert "ВД-ОО5" not in fragment.content.text

    @pytest.mark.parametrize("key", ["mass", "scale", "designation", "format"])
    def test_internal_key_keeps_its_gost_label(self, composed, key):
        """
        Внутренний ключ разбора — ключ `LABELS`, а не повод звать графу
        «mass». Подписанты сюда не входят: их запись называет глаголом.
        """
        fragment = _fragment(SHEET_LINE, payload={"title_block": {key: "X"}})
        drawing_record.apply([fragment])
        assert f"{label(key)} — X." in fragment.content.text
        assert f"{key} — X." not in fragment.content.text

    def test_internal_signer_key_is_named_by_the_verb(self, composed):
        fragment = _fragment(SHEET_LINE,
                             payload={"title_block": {"norm_control": "X"}})
        drawing_record.apply([fragment])
        assert "Лист подписали: нормоконтроль X." in fragment.content.text

    def test_unknown_graph_is_not_dropped(self, composed):
        fragment = _fragment(SHEET_LINE, payload={"title_block": {"custom_graph": "ВД005"}})
        drawing_record.apply([fragment])
        assert "ВД005" in fragment.content.text

    def test_handwritten_graphs_are_marked_as_such(self, composed):
        fragment = _fragment(TADU_TEXT, payload={
            "title_block": TADU_STAMP,
            "title_block_handwritten": {"inventory": "1012-02",
                                        "signed_date": "10.02.2022"},
        })
        drawing_record.apply([fragment])
        record = fragment.content.text
        assert "Записи от руки" in record
        assert "Инв. № подл. — 1012-02." in record
        assert "Подп. и дата — 10.02.2022." in record


# ---------------------------------------------------------------- рамка-мебель
class TestFurniture:
    def test_side_column_captions_are_dropped_and_counted(self, composed):
        fragment = _fragment(SM1638_TEXT)
        drawing_record.apply([fragment])
        record = fragment.content.text
        assert "Инв. № подл." not in record
        assert "Подп. и дата" not in record
        assert fragment.content.structured_payload[drawing_record.DROPPED_KEY] == 2

    def test_caption_with_a_value_on_the_same_line_stays(self, composed):
        fragment = _fragment("Инв. № подл. 1085-02")
        drawing_record.apply([fragment])
        assert "Инв. № подл. 1085-02" in fragment.content.text
        assert drawing_record.DROPPED_KEY not in fragment.content.structured_payload

    @pytest.mark.parametrize("line", ["Масса", "Масштаб", "Лист", "Изм.",
                                      "Формат", "Разраб.", "Лит."])
    def test_graphs_that_may_label_a_nearby_value_are_kept(self, composed, line):
        """
        Подпись «Масса» рядом со строкой «15,7» — единственное, что об этом
        числе говорит. Такие подписи не выбрасываются никогда.
        """
        fragment = _fragment(line + "\n15,7")
        drawing_record.apply([fragment])
        assert line in fragment.content.text

    def test_handwritten_graph_carries_what_the_caption_lacked(self, composed):
        """Подпись выброшена, но значение той же графы в записи есть."""
        fragment = _fragment(SM1638_TEXT, payload={
            "title_block_handwritten": {"inventory": "1085-02"},
        })
        drawing_record.apply([fragment])
        assert "Инв. № подл. — 1085-02." in fragment.content.text


# ---------------------------------------------------------------------- виды
class TestViews:
    def test_named_views_are_listed(self, composed):
        fragment = _fragment(SM1638_TEXT)
        drawing_record.apply([fragment])
        record = fragment.content.text
        assert "Г-Г(1:1)" in record
        assert "В-В(1:1)" in record
        assert "Виды, разрезы и сечения, названные на листе:" in record

    def test_section_title_in_words_is_a_view(self, composed):
        fragment = _fragment(TADU_TEXT)
        drawing_record.apply([fragment])
        assert "Разрез клапанов поз.6 и 7" in fragment.content.text.split(
            "Надписи листа в порядке чтения.")[0]

    @pytest.mark.parametrize("line", ["Б(4)", "Т(5)", "А(4)", "Д(4)", "Ц(4)"])
    def test_letter_with_a_bare_number_is_not_a_view(self, composed, line):
        """
        Масштаб по ГОСТ 2.302 — отношение, с двоеточием. «Б(4)» — номер
        исполнения или позиции; на настоящих листах таких «видов»
        набиралось по восемнадцать на один лист.
        """
        fragment = _fragment(line + "\n25")
        drawing_record.apply([fragment])
        assert "Виды, разрезы и сечения" not in fragment.content.text

    @pytest.mark.parametrize("line", ["Б(2:1)", "A(1:4)", "Г(1:2,5)"])
    def test_letter_with_a_ratio_is_a_view(self, composed, line):
        fragment = _fragment(line + "\n25")
        drawing_record.apply([fragment])
        assert line in fragment.content.text.split(
            "Надписи листа в порядке чтения.")[0]

    def test_paired_letter_is_a_view_with_or_without_scale(self, composed):
        fragment = _fragment("А-А\nГ-Г(1:1)\nВ-В (2)")
        drawing_record.apply([fragment])
        head = fragment.content.text.split("Надписи листа в порядке чтения.")[0]
        assert "А-А" in head and "Г-Г(1:1)" in head and "В-В (2)" in head

    def test_the_hedge_is_said_once(self, composed):
        fragment = _fragment(SM1638_TEXT, fields=[
            {"category": "size", "value": "⌀8,5"},
        ])
        drawing_record.apply([fragment])
        assert fragment.content.text.count(drawing_record.NO_VIEW_LINK) == 1

    def test_the_hedge_is_said_when_there_are_no_named_views(self, composed):
        fragment = _fragment("25", fields=[
            {"category": "size", "value": "⌀8,5"},
        ])
        drawing_record.apply([fragment])
        assert fragment.content.text.count(drawing_record.NO_VIEW_LINK) == 1

    def test_bare_letter_is_not_called_a_view(self, composed):
        """
        «A» на листе — и место разреза, и номер исполнения. Назвать её видом
        было бы догадкой, поэтому в перечень видов она не попадает.
        """
        fragment = _fragment(TADU_TEXT)
        drawing_record.apply([fragment])
        head = fragment.content.text.split("Надписи листа в порядке чтения.")[0]
        assert "\nA\n" not in head
        assert "; A;" not in head

    def test_attachment_is_not_invented(self, composed):
        """Запись честно говорит, что размер к виду не привязан."""
        fragment = _fragment(SM1638_TEXT)
        drawing_record.apply([fragment])
        assert drawing_record.NO_VIEW_LINK in fragment.content.text


# ------------------------------------------------------------------- размеры
class TestDimensions:
    def test_values_are_grouped_by_what_they_are(self, composed):
        fragment = _fragment(SHEET_LINE, fields=[
            {"category": "size", "value": "⌀57"},
            {"category": "size", "value": "100±10"},
            {"category": "size", "value": "35°"},
            {"category": "radius", "value": "R65"},
            {"category": "thread", "value": "M24x1,5"},
            {"category": "roughness", "value": "Ra 1,6"},
            {"category": "tolerance", "value": "20H7"},
        ])
        drawing_record.apply([fragment])
        record = fragment.content.text
        assert "Диаметры — ⌀57." in record
        assert "Линейные размеры — 100±10." in record
        assert "Углы — 35°." in record
        assert "Радиусы — R65." in record
        assert "Резьба — M24x1,5." in record
        assert "Шероховатость — Ra 1,6." in record
        assert "Допуски и посадки — 20H7." in record

    @pytest.mark.parametrize("value", ["3", "1", "7", "10", "25"])
    def test_bare_number_is_not_announced_as_a_dimension(self, composed, value):
        """
        Категорию `size` разбор ставит уже за то, что надпись похожа на
        число: так в неё попадают номера выносок и зон. Назвать их
        линейными размерами — сказать неправду о том, что они измеряют.

        `unit` здесь заполнен так, как его заполняет разбор: «мм» он
        выводит из категории любому числу, с листа не читая. Признаком
        он поэтому не считается.
        """
        fragment = _fragment(SHEET_LINE, fields=[
            {"category": "size", "value": value, "unit": "мм"}])
        drawing_record.apply([fragment])
        assert "Линейные размеры" not in (fragment.content.text or "")

    @pytest.mark.parametrize("value", ["L", "h", "d", "Dp", "Dn"])
    def test_letter_placeholder_is_not_announced_as_a_dimension(self, composed, value):
        fragment = _fragment(SHEET_LINE, fields=[{"category": "size", "value": value}])
        drawing_record.apply([fragment])
        assert "Линейные размеры" not in (fragment.content.text or "")

    @pytest.mark.parametrize("field,expected", [
        ({"category": "size", "value": "3±0,5"}, "3±0,5"),
        ({"category": "size", "value": "100±10"}, "100±10"),
        ({"category": "size", "value": "18x2,5"}, "18x2,5"),
        ({"category": "size", "value": "1737 мм", "unit": "мм"}, "1737 мм"),
        ({"category": "size", "value": "250", "tolerance": "±2"}, "250"),
    ])
    def test_dimension_that_says_what_it_is_stays(self, composed, field, expected):
        fragment = _fragment(SHEET_LINE, fields=[field])
        drawing_record.apply([fragment])
        assert expected in fragment.content.text
        assert "Линейные размеры" in fragment.content.text

    def test_bare_number_still_reaches_the_record(self, composed):
        """Из перечня родов оно ушло, из записи — нет."""
        fragment = _fragment("3\n1\n7", fields=[
            {"category": "size", "value": "3"},
        ])
        drawing_record.apply([fragment])
        assert "\n3\n1\n7" in fragment.content.text

    @pytest.mark.parametrize("value", ["Ø57", "ø44", "Φ20", "φ8,5", "⌀10", "Ф20"])
    def test_every_diameter_glyph_is_a_diameter(self, composed, value):
        """
        Векторный путь кладёт в поля то, что прочитал, знак к одному виду
        не приводя: «Ø57» уходило в линейные размеры.
        """
        fragment = _fragment(SHEET_LINE, fields=[{"category": "size", "value": value}])
        drawing_record.apply([fragment])
        assert f"Диаметры — {value}." in fragment.content.text

    @pytest.mark.parametrize("value", ["Фаска 2х45°", "Фланец"])
    def test_cyrillic_ef_in_a_word_is_not_a_diameter(self, composed, value):
        """
        «Ф» считается диаметром только вплотную к числу, как и в
        `drawing_text`: иначе «Фаска 2х45°» ушла бы в диаметры, потому что
        знак диаметра проверяется прежде градуса.
        """
        fragment = _fragment(SHEET_LINE, fields=[
            {"category": "size", "value": value},
        ])
        drawing_record.apply([fragment])
        assert "Диаметры" not in fragment.content.text

    def test_repeated_reading_is_said_once(self, composed):
        fragment = _fragment(SHEET_LINE, fields=[
            {"category": "size", "value": "⌀57"},
            {"category": "size", "value": "⌀57"},
        ])
        drawing_record.apply([fragment])
        assert "Диаметры — ⌀57." in fragment.content.text

    def test_positions_point_at_the_specification(self, composed):
        fragment = _fragment(SHEET_LINE, fields=[
            {"category": "position", "value": "17"},
            {"category": "position", "value": "18"},
            {"category": "position", "value": "19"},
        ])
        drawing_record.apply([fragment])
        record = fragment.content.text
        assert "Номера позиций на листе: 17, 18, 19." in record
        assert "в спецификации" in record

    def test_callouts_are_their_own_part(self, composed):
        fragment = _fragment(SHEET_LINE, fields=[
            {"category": "callout", "value": "Т1-ИНп-3"},
        ])
        drawing_record.apply([fragment])
        assert "Обозначения и выноски на листе: Т1-ИНп-3." in fragment.content.text

    def test_requirements_are_kept_as_written(self, composed):
        fragment = _fragment(SHEET_LINE, fields=[
            {"category": "note",
             "value": "1. Сварка аргонодуговая по ГОСТ 14771-76."},
        ])
        drawing_record.apply([fragment])
        record = fragment.content.text
        assert "Текст и примечания листа." in record
        assert "1. Сварка аргонодуговая по ГОСТ 14771-76." in record

    def test_notes_are_not_called_technical_requirements(self, composed):
        """
        Категория `note` — это «не классифицировано» с уверенностью 0,4. В
        настоящих записях под ней лежали строки спецификации и целый текст
        документа СМК; заголовок «технические требования листа» был бы
        неправдой.
        """
        fragment = _fragment(SHEET_LINE, fields=[
            {"category": "note",
             "value": "2 Выходные данные анализа СМК включают в себя"},
        ])
        drawing_record.apply([fragment])
        assert "Технические требования" not in fragment.content.text


# ------------------------------------------------------------------- таблицы
class TestTables:
    def test_every_cell_gets_its_column_name(self, composed):
        fragment = _fragment(SHEET_LINE, payload={"tables": [{
            "kind": "table",
            "title": "Таблица исполнений",
            "headers": ["Обозначение", "Наименование", "Кол-во"],
            "rows": [["ТАДУ 405210.001ПС", "Паспорт", "1 экз."]],
        }]})
        drawing_record.apply([fragment])
        record = fragment.content.text
        assert "Таблица на листе «Таблица исполнений»." in record
        assert ("Обозначение — ТАДУ 405210.001ПС; Наименование — Паспорт; "
                "Кол-во — 1 экз.." in record)

    def test_headerless_row_keeps_its_values(self, composed):
        fragment = _fragment(SHEET_LINE, payload={"tables": [{
            "kind": "table", "rows": [["Консервация В3-0", "2024"]],
        }]})
        drawing_record.apply([fragment])
        assert "Консервация В3-0; 2024." in fragment.content.text

    def test_stamp_grid_is_not_repeated_as_a_table(self, composed):
        """
        Сетка штампа — те же подписи и значения, что запись уже назвала
        графами. Повтор вернул бы перемешивание подписей с числами.
        """
        fragment = _fragment(SHEET_LINE, payload={
            "title_block": TADU_STAMP,
            "tables": [{
                "kind": "title_block",
                "rows": [
                    ["Изм", "Лист", "N докум.", "Подп.", "Дата"],
                    ["Разраб.", "Прохоров", "17.01.20"],
                    ["Проверил", "Постнов", "28.01.20"],
                    ["Лит.", "Масса", "Масштаб"],
                ],
            }],
        })
        drawing_record.apply([fragment])
        assert "Таблица в основной надписи листа." not in fragment.content.text

    def test_specification_with_the_same_kind_is_kept(self, composed):
        """
        Та же пометка `title_block`, но это «Комплектность»: наименования и
        количества, а не подписи граф. Такая таблица остаётся.
        """
        fragment = _fragment(SHEET_LINE, payload={
            "title_block": TADU_STAMP,
            "tables": [{
                "kind": "title_block",
                "rows": [
                    ["Наименование", "Кол-во", "Примечание"],
                    ["Термопреобразователь сопротивления", "1 шт.", ""],
                    ["Паспорт", "1 экз.", ""],
                    ["Кабель соединительный", "1 шт.", ""],
                ],
            }],
        })
        drawing_record.apply([fragment])
        record = fragment.content.text
        assert "Таблица в основной надписи листа." in record
        assert "Термопреобразователь сопротивления" in record

    def test_stamp_grid_is_kept_when_graphs_were_not_parsed(self, composed):
        """Без разобранных граф сетка — всё, что о штампе известно."""
        fragment = _fragment(SHEET_LINE, payload={"tables": [{
            "kind": "title_block",
            "rows": [["Изм", "Лист", "N докум."], ["Разраб.", "Прохоров"]],
        }]})
        drawing_record.apply([fragment])
        assert "Прохоров" in fragment.content.text


# -------------------------------------------------------------------- конвейер
class TestCaution:
    """
    Фраза о сверке с чертежом. До сборки неподтверждённые значения помечал
    адаптер RAG; единственным, кто записывает лист, стала сборка, и
    осторожность обязана перейти вместе с содержанием.
    """

    def test_unconfirmed_values_are_admitted_once(self, composed):
        fragment = _fragment(SHEET_LINE, fields=[
            {"category": "size", "value": "⌀8,5", "confidence": 0.8},
        ], payload={"title_block": TADU_STAMP})
        drawing_record.apply([fragment])
        record = fragment.content.text
        assert record.startswith(drawing_record.CAUTION_ALL)
        assert record.count(drawing_record.CAUTION_ALL) == 1

    def test_part_unconfirmed_says_part(self, composed):
        fragment = _fragment(SHEET_LINE, fields=[
            {"category": "size", "value": "⌀8,5", "confidence": 0.99,
             "confirmed": True},
            {"category": "radius", "value": "R65", "confidence": 0.4},
        ])
        drawing_record.apply([fragment])
        assert fragment.content.text.startswith(drawing_record.CAUTION_PART)

    def test_everything_confirmed_says_nothing(self, composed):
        fragment = _fragment(SHEET_LINE, fields=[
            {"category": "size", "value": "⌀8,5", "confidence": 0.99,
             "confirmed": True},
        ])
        drawing_record.apply([fragment])
        record = fragment.content.text
        assert drawing_record.CAUTION_ALL not in record
        assert drawing_record.CAUTION_PART not in record

    def test_an_unconfirmed_table_also_counts(self, composed):
        fragment = _fragment(SHEET_LINE, payload={"tables": [{
            "kind": "table", "rows": [["Сдувка", "С-22-2"]],
        }]})
        drawing_record.apply([fragment])
        assert fragment.content.text.startswith(drawing_record.CAUTION_ALL)

    def test_the_old_parenthetical_is_not_used(self, composed):
        """Приписку «(требует сверки)» из индексируемого текста убрали."""
        fragment = _fragment(SHEET_LINE, fields=[
            {"category": "size", "value": "⌀8,5", "confidence": 0.8},
        ])
        drawing_record.apply([fragment])
        assert "(требует сверки)" not in fragment.content.text

    def test_the_caution_is_plain_language(self, composed):
        """Без «уверенности», «модели» и прочего служебного словаря."""
        for sentence in (drawing_record.CAUTION_ALL, drawing_record.CAUTION_PART):
            lowered = sentence.lower()
            for word in ("модель", "уверенност", "confidence", "vlm",
                         "подтвержд", "confirmed"):
                assert word not in lowered


class TestChunkability:
    """
    Разметка частей для чанкера RAG. Он режет запись по пределу в токенах и
    ищет границы «только по явным пометкам источника» (заголовок Markdown,
    номер пункта) — прозаический заголовок «Диаметры — …» для него невидим.
    Поэтому границы называются явно, в payload.
    """

    def _composed(self):
        fragment = _fragment(SM1638_TEXT + "\n" + TADU_TEXT, fields=[
            {"category": "size", "value": "⌀8,5"},
            {"category": "radius", "value": "R65"},
            {"category": "position", "value": "17"},
            {"category": "note", "value": "1. Размеры для справок."},
        ], payload={"title_block": TADU_STAMP})
        drawing_record.apply([fragment])
        return fragment

    def test_offsets_reconstruct_the_record_exactly(self, composed):
        fragment = self._composed()
        text = fragment.content.text
        parts = fragment.content.structured_payload[drawing_record.PARTS_KEY]
        assert "\n\n".join(text[p["start"]:p["end"]] for p in parts) == text

    def test_parts_do_not_overlap_and_run_in_order(self, composed):
        fragment = self._composed()
        parts = fragment.content.structured_payload[drawing_record.PARTS_KEY]
        assert parts[0]["start"] == 0
        for earlier, later in zip(parts, parts[1:]):
            assert earlier["end"] < later["start"]

    def test_title_is_the_first_line_of_its_part(self, composed):
        fragment = self._composed()
        text = fragment.content.text
        for part in fragment.content.structured_payload[drawing_record.PARTS_KEY]:
            body = text[part["start"]:part["end"]]
            assert body.startswith(part["title"])
            assert "\n" not in part["title"]

    def test_breaks_are_line_starts_inside_their_part(self, composed):
        fragment = self._composed()
        text = fragment.content.text
        for part in fragment.content.structured_payload[drawing_record.PARTS_KEY]:
            for offset in part["breaks"]:
                assert part["start"] < offset <= part["end"]
                assert text[offset - 1] == "\n"

    def test_a_group_of_values_is_never_cut_from_its_heading(self, composed):
        """
        Между началами строк — значит, перечень «Диаметры — …» режется только
        целиком. Внутри строки чанкер резать не обязан: самая большая часть
        «Размеры» на 250 листах с прода — 411 символов из 1280 предела.
        """
        fragment = self._composed()
        text = fragment.content.text
        part = next(p for p in fragment.content.structured_payload[
            drawing_record.PARTS_KEY] if p["kind"] == "dimensions")
        for offset in part["breaks"]:
            assert text[offset:].startswith(("Диаметры", "Радиусы", "Резьба",
                                             "Шероховатость", "Углы",
                                             "Допуски", "Линейные"))

    def test_the_stamp_opens_the_record_after_the_caution(self, composed):
        """
        Если запись разрежут и найдётся только первый кусок, в нём должен
        быть ответ на «какая масса». Это свойство порядка, а не случайность.
        Фраза об осторожности идёт прежде значений, но коротка и основную
        надпись из первого куска не вытесняет: предел — 512 токенов, около
        1280 символов.
        """
        fragment = self._composed()
        parts = fragment.content.structured_payload[drawing_record.PARTS_KEY]
        assert [part["kind"] for part in parts[:2]] == ["caution", "stamp"]
        assert "Масса — 26,02" in fragment.content.text[:1280]

    def test_parts_are_separated_by_a_blank_line(self, composed):
        """Чанкер, который разметку не читает, а абзацы различает, тоже попадёт."""
        fragment = self._composed()
        text = fragment.content.text
        parts = fragment.content.structured_payload[drawing_record.PARTS_KEY]
        for earlier, later in zip(parts, parts[1:]):
            assert text[earlier["end"]:later["start"]] == "\n\n"

    def test_context_names_the_sheet_and_is_not_repeated_in_the_prose(self, composed):
        """
        Обозначение в каждый кусок ставит чанкер своим заголовком, который в
        поиск уходит, а заказчику не показывается. В прозе его повторять не
        надо — это был бы шум в каждой части.
        """
        fragment = self._composed()
        payload = fragment.content.structured_payload
        context = payload[drawing_record.CONTEXT_KEY]
        assert "ТАДУ 302318.207.00 СБ" in context
        assert "Трубная обвязка" in context
        assert fragment.content.text.count("ТАДУ 302318.207.00 СБ") == 2

    def test_no_markup_is_added_to_the_prose(self, composed):
        """Границы названы в payload, а не решёткой Markdown в тексте."""
        fragment = self._composed()
        assert "#" not in fragment.content.text

    def test_nothing_emitted_when_nothing_is_composed(self, composed):
        fragment = _fragment("")
        drawing_record.apply([fragment])
        assert drawing_record.PARTS_KEY not in fragment.content.structured_payload


class TestPipeline:
    def test_other_fragment_types_are_untouched(self, composed):
        fragment = Fragment(
            fragment_id="t1", type="text",
            content=Content(text=SM1638_TEXT),
            position=Position(page=1, bbox=[0, 0, 1, 1]),
            confidence=0.9, completeness=1.0,
            provenance=Provenance(method="ocr_layer", strategy_level=4,
                                  source="ocr"),
        )
        drawing_record.apply([fragment])
        assert fragment.content.text == SM1638_TEXT

    def test_fragment_id_does_not_move_when_the_record_is_composed(self, composed):
        """
        Идентификатор чертежа считается по полям и таблицам, а не по
        тексту: сборка записи не создаёт в RAG дубль вместо обновления.
        """
        fields = [{"category": "size", "value": "⌀57"}]
        plain_fragment = _fragment(TADU_TEXT, fields=fields,
                                   payload={"title_block": TADU_STAMP})
        composed_fragment = _fragment(TADU_TEXT, fields=fields,
                                      payload={"title_block": TADU_STAMP})
        drawing_record.apply([composed_fragment])
        first, _ = finalize_fragments([plain_fragment], "doc-1", extract=False)
        second, _ = finalize_fragments([composed_fragment], "doc-1", extract=False)
        assert first[0].fragment_id == second[0].fragment_id

    def test_payload_says_the_record_was_composed(self, composed):
        fragment = _fragment(TADU_TEXT, payload={"title_block": TADU_STAMP})
        finalize_fragments([fragment], "doc-1")
        assert fragment.content.structured_payload[drawing_record.COMPOSED_KEY] is True

    def _run_pipeline(self):
        """
        Настоящий порядок стадий оркестратора: нормализация (она сама
        финализирует с `extract=False`) -> сверка маршрута -> финализация
        с фактами. Чертёж здесь векторный: у такого фрагмента `content.text`
        нет, и именно на нём видно, собралась ли запись раньше времени.
        """
        block = ParsedBlock(
            type="drawing", page=1, bbox=[0.0, 0.0, 1.0, 1.0], order=1,
            image_ref="s3://bucket/sheet.pdf",
            drawing_fields=[
                {"category": "size", "value": "⌀57", "confidence": 0.9},
                {"category": "position", "value": "17", "confidence": 0.9},
            ],
        )
        parse_result = ParseResult(blocks=[block], parser_name="vector_drawing")
        normalizer = TextNormalizer()
        fragments, _ = normalizer.normalize_with_flags(
            parse_result, {"doc_id": "doc-1"})
        return normalizer, parse_result, fragments

    def test_record_is_not_composed_at_the_normalize_stage(self, composed):
        """
        `finalize_fragments` вызывается дважды: нормализатор зовёт её с
        `extract=False` ещё до ветки чертежей и до сверки маршрута. Там
        собирать нельзя — иначе собранная запись попала бы в сверку.
        """
        _, _, fragments = self._run_pipeline()
        drawing = next(f for f in fragments if f.type == "drawing")
        assert drawing.content.text is None
        assert drawing_record.COMPOSED_KEY not in (
            drawing.content.structured_payload or {})

    def test_route_volume_sees_the_text_the_parse_produced(self, composed):
        """
        Объём для сверки «чертёж против OCR» считается по `content.text`.
        На стадии сверки текста у векторного чертежа нет, и сборка не
        должна была его создать: иначе флаг меняет исход сверки.
        """
        _, parse_result, fragments = self._run_pipeline()
        drawing = next(f for f in fragments if f.type == "drawing")
        before = drawing_route.volume(fragments).chars
        # Сверка на этом месте и происходит (ingest/tasks.py).
        drawing_route.settle(fragments, parse_result, {}, TextNormalizer())
        # Объём тот же, и текста у векторного чертежа по-прежнему нет:
        # сборка его не создала. Ноль здесь не ждём — объём считает и поля.
        assert drawing_route.volume(fragments).chars == before
        assert drawing.content.text is None

    def test_record_is_composed_once_after_the_drawing_branch(self, composed):
        """
        Сборка идёт на финализации с фактами — после ветки чертежей,
        которая к этому времени положила в запись текст листа.
        """
        _, _, fragments = self._run_pipeline()
        drawing = next(f for f in fragments if f.type == "drawing")
        drawing.content.text = TADU_TEXT          # что делает ветка чертежей
        drawing.content.structured_payload["title_block"] = TADU_STAMP
        finalize_fragments(fragments, "doc-1")
        record = drawing.content.text
        assert "Масса — 26,02" in record
        assert record.count("Надписи листа в порядке чтения.") == 1

    def test_second_finalisation_changes_nothing(self, composed):
        """
        Идемпотентность. Второй проход вкладывал сборку внутрь её же части
        «Надписи листа в порядке чтения»: заголовок переставал отвечать
        тому, что под ним, а запись почти удваивалась.
        """
        _, _, fragments = self._run_pipeline()
        drawing = next(f for f in fragments if f.type == "drawing")
        drawing.content.text = TADU_TEXT
        drawing.content.structured_payload["title_block"] = TADU_STAMP
        finalize_fragments(fragments, "doc-1")
        once = drawing.content.text
        finalize_fragments(fragments, "doc-1")
        assert drawing.content.text == once

    def test_reading_order_part_holds_no_composed_part(self, composed):
        """Под заголовком «Надписи листа…» стоят строки листа, и только они."""
        fragment = _fragment(TADU_TEXT, payload={"title_block": TADU_STAMP})
        drawing_record.apply([fragment])
        drawing_record.apply([fragment])
        tail = fragment.content.text.split("Надписи листа в порядке чтения.")[1]
        assert "Основная надпись листа." not in tail
        assert drawing_record.NO_VIEW_LINK not in tail

    def test_composition_would_change_the_volume_if_it_ran_too_early(self, composed):
        """
        Обратная сторона предыдущей проверки: раздувание настоящее. Если
        сборку перенесут раньше сверки, объём изменится — и тест выше
        упадёт не зря.
        """
        fields = [{"category": "size", "value": "⌀57"}]
        fragment = _fragment(TADU_TEXT, fields=fields,
                             payload={"title_block": TADU_STAMP})
        before = drawing_route.volume([fragment]).chars
        drawing_record.apply([fragment])
        assert drawing_route.volume([fragment]).chars > before

    def test_transcript_line_count_is_left_as_it_was(self, composed):
        """Число строк листа — вход правки абзацев OCR; сборка его не меняет."""
        fragment = _fragment(TADU_TEXT, payload={"title_block": TADU_STAMP})
        before = fragment.content.structured_payload["transcript_line_count"]
        drawing_record.apply([fragment])
        assert fragment.content.structured_payload["transcript_line_count"] == before
