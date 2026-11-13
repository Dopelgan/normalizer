"""
Числовая достоверность размеров и знаков на чертеже.

Все строки в этих проверках — настоящие: взяты из сверки полей по пяти
демонстрационным чертежам (ТАДУ 687000.001.001 «Крышка», ТАДУ 687000.001 СБ
«Коробка соединительная», «Пробка ЭК150», «Расширитель») и из разборов,
сохранённых на сервере.
"""

import pytest

from core.config import settings
from core.providers import dimension_fidelity as df
from core.providers.drawing_text import normalize
from core.providers.vector_drawing import classify, extract_tolerance, tolerance_source


@pytest.fixture(autouse=True)
def fidelity_on(monkeypatch):
    """Флаг выключен по умолчанию; проверки идут при включённом."""
    monkeypatch.setattr(settings, "DRAWING_DIMENSION_FIDELITY", True)


def test_flag_off_changes_nothing(monkeypatch):
    """Без флага поведение то же, что и до правки: ошибки остаются на месте."""
    monkeypatch.setattr(settings, "DRAWING_DIMENSION_FIDELITY", False)
    # Гомоглиф знака шероховатости правился и до флага, запятая — нет.
    assert normalize("Ра 16") == "Ра 16"
    assert normalize("M28x15") == "M28x15"
    # Латинизация марки и была источником ложного допуска.
    assert extract_tolerance(normalize("12Х18Н10Т")) == "H10"


class TestThreadPitch:
    """«M28x1,5» прочитано как «M28x15»: шага 15 мм у метрической резьбы нет."""

    @pytest.mark.parametrize("text,expected", [
        ("M28x15", "M28x1,5"),
        ("M24×15", "M24×1,5"),
        ("M33x15", "M33x1,5"),
        ("M24×15 | рис. 4 | вилка СНЦ22-10/14В", "M24×1,5 | рис. 4 | вилка СНЦ22-10/14В"),
        ("M14x125", "M14x1,25"),
        ("M12x175", "M12x1,75"),
    ])
    def test_pitch_comma_restored(self, text, expected):
        assert df.repair(text) == expected

    @pytest.mark.parametrize("text", [
        "M42x2",            # шаг 2 мм в ряду ГОСТ 8724 есть
        "M50x2",
        "M58x2",
        "M33x1,5",          # запятая на месте
        "M12x1.5",
        "M24",              # без шага
        "Tr38x3",           # трапецеидальная резьба — не метрическая
        "11, 2 паза",
        "M6x20", "M8x25", "M10x40", "M12x45", "M20x80", "M12x50-6g",
    ])
    def test_valid_pitch_is_left_alone(self, text):
        assert df.repair(text) == text

    @pytest.mark.parametrize("text,expected", [
        # Длина крепежа, а не шаг: 5 мм на резьбе М12 невозможны, и 50 —
        # из ряда длин по ГОСТ 7798. Путь настоящий: кириллическую «М»
        # латинской делает `normalize`, и только после этого правило видит
        # резьбу.
        ("Болт М12х50 ГОСТ 7798-70", "Болт M12x50 ГОСТ 7798-70"),
        ("Винт М8х30 ГОСТ 17475-80", "Винт M8x30 ГОСТ 17475-80"),
        ("Шпилька М16х60", "Шпилька M16x60"),
        ("М12х50-6g", "M12x50-6g"),
        ("М28х15", "M28x1,5"),
    ])
    def test_fastener_length_is_not_a_pitch(self, text, expected):
        assert normalize(text) == expected


class TestRoughness:
    """«Ra 1,6» прочитано как «Ра 16»: кириллическая «Р» и потерянная запятая."""

    @pytest.mark.parametrize("text,expected", [
        ("Ра 16", "Ra 1,6"),
        ("Ра16", "Ra1,6"),
        ("Ra 63", "Ra 6,3"),
        ("Ra 125", "Ra 12,5"),
    ])
    def test_roughness_comma_restored(self, text, expected):
        assert normalize(text) == expected

    @pytest.mark.parametrize("text", [
        "Ra 1,6",
        "Ra 6,3",
        "Ra 25",            # предпочтительное значение ряда ГОСТ 2789
        "Ra 50",
        "Ra 100",
        "Ra 20",            # не предпочтительное, но «2» тоже не из ряда
        # Ряд Rz по ГОСТ 2789 другой: эти значения в нём настоящие.
        "Rz 16", "Rz 32", "Rz 63", "Rz 125",
    ])
    def test_valid_roughness_is_left_alone(self, text):
        assert df.repair(text) == text

    def test_whole_note_from_kryshka(self):
        """Пункт 8 технических требований чертежа «Крышка»."""
        text = "8. Лицевую сторону шлифовать, шерохов. не хуже, чем Ра 16"
        assert normalize(text).endswith("Ra 1,6")

    def test_glued_note_from_kryshka(self):
        """В разборе 13.11 слова склеены: «чемРа 16 (опероцию…»."""
        text = "8.Лицевую сторону шлифовать, шерохов. не хуже, чемРа 16 (опероцию"
        assert "чемRa 1,6 (" in df.repair(text)

    @pytest.mark.parametrize("text", [
        "unit-c5b88120cca7607eff94ba04",
        "assets/3c2f630c1c3334165c2408ba47d0cab1.jpg",
        "j4:p001:text:648ba444250a",
        "Шайба 5", "Скоба 2", "Кабель 5", "Труба 20",
        # Обозначения по ГОСТ 2.710 и серии изделий: вторая буква заглавная,
        # и это не знак шероховатости.
        "ток КЗ 31 кА", "Выключатель ВА 47-29", "реле КА 1", "КА1", "РЗ 2",
    ])
    def test_identifiers_and_words_are_left_alone(self, text):
        assert df.repair(text) == text


class TestDeviation:
    """«2⁺¹˙⁵» прочитано как «2+15»: надстрочные цифры и потерянная запятая."""

    @pytest.mark.parametrize("text,expected", [
        ("2⁺¹˙⁵", "2+1,5"),
        ("3,5⁺¹˙⁵", "3,5+1,5"),
        ("⌀16⁺⁰˙⁰¹⁸", "⌀16+0,018"),
        ("2⁺¹·⁵", "2+1,5"),
    ])
    def test_superscripts_expanded(self, text, expected):
        assert df.repair(text) == expected

    @pytest.mark.parametrize("text", [
        "• Примечание 1",   # точка списка — не разделитель дроби
        "2 шт. • ⌀10",
        "Площадь 10 м²",    # степень единицы — не отклонение
        "x² + y²",
    ])
    def test_other_superscripts_are_left_alone(self, text):
        assert df.repair(text) == text

    @pytest.mark.parametrize("text,expected", [
        ("2+15", "2+1,5"),          # отклонение не бывает больше размера
        ("3,5+15", "3,5+1,5"),
        ("40−025", "40−0,25"),
    ])
    def test_impossible_deviation_repaired(self, text, expected):
        assert df.repair(text) == expected

    @pytest.mark.parametrize("text", [
        "30±10",            # отклонение меньше размера — запись возможна
        "15±5",
        "20±5",
        "100+25",
        "12+0,8",
        "⌀218−0,2",
        "4-16",             # диапазон диаметра кабеля, а не отклонение
        "15- 20",
        "19 - 25",
        "1 Зам. ТАДУ.655-25",
        "ГОСТ 14771-76",
        "по НП-001-15",
        "ГОСТ Р 50.05.08-2018",
        # Настоящий минус в обозначении: цепочка номеров, а не отклонение.
        "ТУ 3185−001−12345678−2015",
        "ГОСТ 2.307−2011",
        "4−16",             # диапазон с настоящим минусом
    ])
    def test_possible_records_are_left_alone(self, text):
        assert df.repair(text) == text


class TestScale:
    """Масштаб «1:1» прочитан как «7:1»: курсивная «1» похожа на «7»."""

    def test_scale_field_out_of_gost_series_repaired(self):
        assert df.repair_scale("7:1") == "1:1"
        assert df.repair_scale("7:5") == "1:5"

    def test_view_heading_in_brackets_repaired(self):
        assert df.repair("Б(7:1)") == "Б(1:1)"

    @pytest.mark.parametrize("text", [
        "Поз. 7:1",         # не графа масштаба и не заголовок вида
        "лист 7:15",
        "2:7",
        "7:5",
        "(7:5)",            # скобка без буквы вида — не масштаб
        "(2:7)",
        "(7:15)",           # в скобках бывает и время
    ])
    def test_ratio_outside_scale_context_is_left_alone(self, text):
        assert df.repair(text) == text

    @pytest.mark.parametrize("text", [
        "1:1", "1:2", "1:4", "А (1:2)", "Б(2:1)", "2:1", "1:2,5",
        "9:19:15Z", "соотношение 3:7",
    ])
    def test_valid_scale_is_left_alone(self, text):
        assert df.repair(text) == text


class TestSteelGrade:
    """«12Х18Н10Т» получала допуск «H10», которого на чертеже нет."""

    @pytest.mark.parametrize("text", [
        "12Х18Н10Т",
        "08Х18Н10Т",
        "12Х18Н10Т, 08Х18Н10Т",
        "Св-04Х19Н11М3 ГОСТ 2246-70",
    ])
    def test_grade_gives_no_tolerance(self, text):
        assert extract_tolerance(normalize(text)) is None
        assert tolerance_source(normalize(text)) == "unknown"

    @pytest.mark.parametrize("text", [
        "12Х18Н10Т",
        "08Х18Н10Т",
    ])
    def test_grade_is_material_not_size(self, text):
        assert classify(text)[0] == "material"

    @pytest.mark.parametrize("text", [
        "12Х18Н10Т",
        "08Х18Н10Т ГОСТ 5632-2014",
        "Св-04Х19Н11М3 ГОСТ 2246-70",
    ])
    def test_grade_letters_are_not_latinised(self, text):
        assert normalize(text) == text

    @pytest.mark.parametrize("text", [
        "10Х17Н13М2Т",      # хвост из двух легирующих элементов
        "12Х18Н10Т, 08Х18Н10Т",
        # Без хвоста — только когда рядом сказано о материале.
        "Сталь 08Х18Н10",
        "08Х18Н10 ГОСТ 5632-2014",
        "Лист 2 ГОСТ 19904-90 / 12Х18Н9 ГОСТ 5582-75",
    ])
    def test_grade_with_long_tail_is_a_grade(self, text):
        assert df.is_steel_grade(text)
        assert extract_tolerance(normalize(text)) is None

    def test_grade_without_tail_and_without_context_stays_ambiguous(self):
        """«12х18Н9» без слова о материале — это и размер паза с посадкой."""
        assert not df.is_steel_grade("08Х18Н10")

    @pytest.mark.parametrize("text", [
        "18х11Н9",          # размер паза с посадкой, а не марка
        "20x4H7",
        "40x5h11",
    ])
    def test_size_with_fit_is_not_taken_for_a_grade(self, text):
        assert not df.is_steel_grade(text)
        assert extract_tolerance(normalize(text)) is not None

    def test_real_fit_still_found(self):
        assert extract_tolerance("⌀44H7") == "H7"
        assert tolerance_source("⌀44H7") == "fit_notation"
        assert extract_tolerance("20±0,1") == "±0,1"


class TestSignAndReferenceMark:
    """
    Транскрипция листа подменяет «±» на «+», а вырезка вида читает знак
    верно. В индекс должно попасть прочтение со знаком и со звёздочкой.
    """

    @pytest.mark.parametrize("first,second,expected", [
        ("30+10*", "30±10", "30±10*"),
        ("30±10", "30+10*", "30±10*"),
        ("⌀4,5+1*", "⌀4,5±1*", "⌀4,5±1*"),
        ("218+0,2", "218−0,2", "218−0,2"),
        ("15+5", "15±5*", "15±5*"),
        ("20±5*", "20+5", "20±5*"),
        ("30+10", "30-10", "30-10"),    # обычный дефис — тоже минус
    ])
    def test_signed_reading_wins(self, first, second, expected):
        assert df.prefer(first, second) == expected

    def test_same_dimension_merged_once(self):
        fields = [
            {"category": "size", "value": "30+10*", "confidence": 0.5,
             "bbox": [0.1, 0.1, 0.2, 0.15]},
            {"category": "size", "value": "30±10", "confidence": 0.9,
             "bbox": [0.1, 0.1, 0.2, 0.15]},
        ]
        merged = df.merge_readings(fields)
        assert [f["value"] for f in merged] == ["30±10*"]
        assert merged[0]["confidence"] == 0.9

    def test_two_separate_annotations_are_kept(self):
        """«+0,2» и «−0,2», написанные на листе в разных местах, — две надписи."""
        fields = [
            {"category": "size", "value": "⌀218+0,2", "confidence": 0.9,
             "bbox": [0.1, 0.1, 0.2, 0.15]},
            {"category": "size", "value": "⌀218−0,2", "confidence": 0.9,
             "bbox": [0.6, 0.7, 0.7, 0.75]},
        ]
        assert len(df.merge_readings(fields)) == 2

    def test_reading_without_bbox_still_merges_plus_and_pm(self):
        """«+» вместо «±» — известная ошибка модели, её сводим и без рамки."""
        fields = [
            {"category": "size", "value": "30+10", "confidence": 0.4, "bbox": None},
            {"category": "size", "value": "30±10", "confidence": 0.9,
             "bbox": [0.1, 0.1, 0.2, 0.15]},
        ]
        assert [f["value"] for f in df.merge_readings(fields)] == ["30±10"]

    @pytest.mark.parametrize("bbox", [None, [0.0, 0.0, 1.0, 1.0]])
    def test_plus_and_minus_without_known_place_are_both_kept(self, bbox):
        """«+0,2» и «−0,2» — законная пара; без рамки сводить их нельзя."""
        fields = [
            {"category": "size", "value": "⌀218+0,2", "confidence": 0.9, "bbox": bbox},
            {"category": "size", "value": "⌀218−0,2", "confidence": 0.9,
             "bbox": [0.1, 0.1, 0.2, 0.15]},
        ]
        assert len(df.merge_readings(fields)) == 2


class TestPrompts:
    """Там, где неверный знак возвращает сама модель, правится запрос."""

    def test_sign_rule_is_added_only_with_the_flag(self, monkeypatch):
        from core.providers import qwen_client

        assert "это «±»" in qwen_client._transcribe_prompt()
        assert "Звёздочку" in qwen_client._transcribe_prompt()
        monkeypatch.setattr(settings, "DRAWING_DIMENSION_FIDELITY", False)
        assert "это «±»" not in qwen_client._transcribe_prompt()


class TestBlocks:
    """Ячейка таблицы MinerU правится на месте: неверный токен не доживает до индекса."""

    def test_table_cell_repaired(self):
        class Block:
            type = "table"
            table_html = "<td>M28x15</td>"

            def __init__(self):
                self.table_data = {
                    "headers": ["Тип", "Резьба"],
                    "rows": [["b", "M28x15"], ["e", "M33x1.5"]],
                }

        block = Block()
        # Одна исправленная ячейка — одна правка, хотя та же запись
        # поправлена и в исходном HTML таблицы.
        assert df.repair_blocks([block]) == 1
        assert block.table_data["rows"][0][1] == "M28x1,5"
        assert block.table_data["rows"][1][1] == "M33x1.5"
        assert block.table_html == "<td>M28x1,5</td>"

    def test_text_block_repaired(self):
        class Block:
            type = "text"
            table_data = None
            table_html = None
            text = "5. Основной материал 12Х18Н10Т, шерохов. Ра 16"

        block = Block()
        df.repair_blocks([block])
        assert block.text.endswith("Ra 1,6")
        assert "12Х18Н10Т" in block.text


class TestTranscriptLines:
    def test_lines_repaired(self):
        lines = ["Тип b | 15-20 | 41 | 32 | M28x15 | рис. 2", "шерохов. Ра 16"]
        assert df.repair_lines(lines) == [
            "Тип b | 15-20 | 41 | 32 | M28x1,5 | рис. 2", "шерохов. Ra 1,6",
        ]
