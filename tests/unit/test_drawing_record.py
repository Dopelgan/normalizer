"""
Запись чертёжного листа, собранная по частям (`DRAWING_RECORD_COMPOSED`).

Все образцы здесь — настоящие: тексты записей и графы штампа взяты из
разбора листов СМ1638.11.05.00.00.00 СБ и ТАДУ 302318.207.00 СБ на
рабочей системе (очередь `platform_parser_ingest_queue.parser_payload`).
Придуманных листов в этом файле нет: сборка проверяется ровно на том, на
чём заказчик её и забраковал.
"""

from pathlib import Path

import pytest

from core import drawing_record
from core.config import settings
from core.contract_finalize import finalize_fragments
from core.ladder import drawing_route
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
        fragment = _fragment("", fields=[
            {"category": "size", "value": "⌀200", "tolerance": "±2"},
        ])
        drawing_record.apply([fragment])
        assert "⌀200 (допуск ±2)" in fragment.content.text


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
        fragment = _fragment("", payload={"title_block": {
            "масса": "2,7", "листов": "1", "литера": "И", "масштаб": "1:2",
            "организация": 'ООО "ФИНПРОМАТОМ"',
        }})
        drawing_record.apply([fragment])
        record = fragment.content.text
        assert ("Масса — 2,7. Масштаб — 1:2. Литера — И. Листов — 1. "
                'Организация — ООО "ФИНПРОМАТОМ".' in record)

    def test_two_namings_of_one_graph_are_both_kept(self, composed):
        fragment = _fragment("", payload={"title_block": {
            "mass": "2,7", "масса": "2,75",
        }})
        drawing_record.apply([fragment])
        assert "Масса — 2,7; 2,75." in fragment.content.text

    def test_unknown_graph_is_not_dropped(self, composed):
        fragment = _fragment("", payload={"title_block": {"custom_graph": "ВД005"}})
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
        assert ("К какому из них относится каждый размер, запись чертежа "
                "не указывает." in fragment.content.text)


# ------------------------------------------------------------------- размеры
class TestDimensions:
    def test_values_are_grouped_by_what_they_are(self, composed):
        fragment = _fragment("", fields=[
            {"category": "size", "value": "⌀57"},
            {"category": "size", "value": "100"},
            {"category": "size", "value": "35°"},
            {"category": "radius", "value": "R65"},
            {"category": "thread", "value": "M24x1,5"},
            {"category": "roughness", "value": "Ra 1,6"},
            {"category": "tolerance", "value": "20H7"},
        ])
        drawing_record.apply([fragment])
        record = fragment.content.text
        assert "Диаметры — ⌀57." in record
        assert "Линейные размеры — 100." in record
        assert "Углы — 35°." in record
        assert "Радиусы — R65." in record
        assert "Резьба — M24x1,5." in record
        assert "Шероховатость — Ra 1,6." in record
        assert "Допуски и посадки — 20H7." in record

    def test_repeated_reading_is_said_once(self, composed):
        fragment = _fragment("", fields=[
            {"category": "size", "value": "⌀57"},
            {"category": "size", "value": "⌀57"},
        ])
        drawing_record.apply([fragment])
        assert "Диаметры — ⌀57." in fragment.content.text

    def test_positions_point_at_the_specification(self, composed):
        fragment = _fragment("", fields=[
            {"category": "position", "value": "17"},
            {"category": "position", "value": "18"},
            {"category": "position", "value": "19"},
        ])
        drawing_record.apply([fragment])
        record = fragment.content.text
        assert "Номера позиций на листе: 17, 18, 19." in record
        assert "в спецификации" in record

    def test_callouts_are_their_own_part(self, composed):
        fragment = _fragment("", fields=[
            {"category": "callout", "value": "Т1-ИНп-3"},
        ])
        drawing_record.apply([fragment])
        assert "Обозначения и выноски на листе: Т1-ИНп-3." in fragment.content.text

    def test_requirements_are_kept_as_written(self, composed):
        fragment = _fragment("", fields=[
            {"category": "note",
             "value": "1. Сварка аргонодуговая по ГОСТ 14771-76."},
        ])
        drawing_record.apply([fragment])
        record = fragment.content.text
        assert "Технические требования и примечания листа." in record
        assert "1. Сварка аргонодуговая по ГОСТ 14771-76." in record


# ------------------------------------------------------------------- таблицы
class TestTables:
    def test_every_cell_gets_its_column_name(self, composed):
        fragment = _fragment("", payload={"tables": [{
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
        fragment = _fragment("", payload={"tables": [{
            "kind": "table", "rows": [["Консервация В3-0", "2024"]],
        }]})
        drawing_record.apply([fragment])
        assert "Консервация В3-0; 2024." in fragment.content.text

    def test_stamp_grid_is_not_repeated_as_a_table(self, composed):
        """
        Сетка штампа — те же подписи и значения, что запись уже назвала
        графами. Повтор вернул бы перемешивание подписей с числами.
        """
        fragment = _fragment("", payload={
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
        fragment = _fragment("", payload={
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
        fragment = _fragment("", payload={"tables": [{
            "kind": "title_block",
            "rows": [["Изм", "Лист", "N докум."], ["Разраб.", "Прохоров"]],
        }]})
        drawing_record.apply([fragment])
        assert "Прохоров" in fragment.content.text


# -------------------------------------------------------------------- конвейер
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
        finalize_fragments([fragment], "doc-1", extract=False)
        assert fragment.content.structured_payload[drawing_record.COMPOSED_KEY] is True

    def test_route_decision_is_made_before_the_record_is_composed(self):
        """
        Сборка раздувает число символов `content.text`, а по нему
        `core/ladder/drawing_route.volume` сравнивает чертёжный разбор с
        распознаванием. Сравнение обязано идти по прежнему тексту, иначе
        флаг начнёт менять выбор маршрута на реальных листах. Защита —
        только порядок вызовов в оркестраторе, поэтому порядок проверяется.
        """
        source = (Path(__file__).resolve().parents[2]
                  / "ingest" / "tasks.py").read_text(encoding="utf-8")
        settle = source.index("drawing_route.settle(")
        finalize = source.index("finalize_fragments(fragments")
        assert settle < finalize

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
