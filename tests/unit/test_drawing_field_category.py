"""
Строгая категория поля чертежа (`DRAWING_FIELD_CATEGORY_STRICT`) и знак
отклонения, восстановленный по транскрипции листа.

Все значения в тестах — настоящие: выборка из прода
(`platform_parser_ingest_queue.parser_payload`, 300 фрагментов чертежей в
порядке md5, 10 174 поля). Придуманных записей здесь нет.

Что в замере измерено и что посчитано. Точность заголовка 87,2 % (64 неверных
из 501) и счёт спора модели с решающей нотацией — 27 спорных записей, правила
правы 26 из 26, модель 0 из 26 — **измерены** на этой выборке. Итог «неверных
заголовков 87 → 22» вместе с правкой `drawing_processor._reconcile`
**смоделирован**: заявка модели до `_reconcile` в контракте не сохраняется, и
за неё взята прод-категория. Измеренная часть того же сравнения — 87 → 60,
правкой одного `classify`.

Флаги здесь ставятся только через `monkeypatch.setattr` на поле, которое в
`Settings` объявлено: прямое присваивание оставило бы флаг включённым до конца
прогона, а неизвестное поле на синглтоне — ещё и в чужих тестах.
"""

import pytest

from core.config import settings
from core.providers import dimension_fidelity
from core.providers.vector_drawing import classify, decisive_category


@pytest.fixture
def strict(monkeypatch):
    monkeypatch.setattr(settings, "DRAWING_FIELD_CATEGORY_STRICT", True)


@pytest.fixture
def strict_off(monkeypatch):
    monkeypatch.setattr(settings, "DRAWING_FIELD_CATEGORY_STRICT", False)


class TestFlagOff:
    """Флаг выключен — разбор прежний, до знака."""

    @pytest.mark.parametrize("text,category", [
        # Именно эти разборы флаг и правит: при выключенном флаге они
        # остаются такими, какими пришли из прода.
        ("0,1 m3/h", "thread"),
        ("Measurement range, m3/h", "thread"),
        ("No more than 60°С", "size"),
        ("fittings K4", "tolerance"),
        ("-1300", "tolerance"),
        ("2№3", "note"),
        # Верные разборы — тоже без изменений.
        ("⌀57", "size"),
        ("R15", "radius"),
        ("M20x1,5", "thread"),
        ("Ra 6,3", "roughness"),
        ("12", "size"),
        ("A", "callout"),
    ])
    def test_unchanged(self, strict_off, text, category):
        assert classify(text)[0] == category


class TestDecisiveNotation:
    """Записи, у которых другого прочтения по ЕСКД нет."""

    @pytest.mark.parametrize("text,category", [
        ("⌀57", "size"),
        ("⌀32x2,5", "size"),
        ("⌀50±1", "size"),
        ("ГОСТ 5264-80-Н1-⌀ 8", "size"),
        ("R15", "radius"),
        ("R3,25*", "radius"),
        ("M20x1,5", "thread"),
        ("M8*", "thread"),
        ("2 M 2", "thread"),
        ("Ra 6,3", "roughness"),
        ("Ra 1,6", "roughness"),
        # Сварной шов и номер позиции: «№» с цифрой вплотную.
        ("2№3", "position"),
        ("2№2", "position"),
        ("№1", "position"),
        ("4№1", "position"),
        ("4,№9", "position"),
    ])
    def test_decisive(self, strict, text, category):
        assert decisive_category(text) == category
        assert classify(text)[0] == category

    @pytest.mark.parametrize("text", [
        # Номер документа и графа штампа — не номер позиции.
        "Зав. № 220483",
        "План качества № 3.10.22",
        "№1-СМК",
        "Приказ №8-СМК от",
        "Р.П. №261.2017",
        # Спорные записи: нотация не решает, и приоритета у правил нет.
        "0,1 m3/h",
        "fittings K4",
        "F1",
        "S27",
        "-1300",
        "12",
        "A",
    ])
    def test_not_decisive(self, strict, text):
        assert decisive_category(text) is None


class TestDecisiveWinsOverClassify:
    """
    Решающая нотация отвечает одно и то же независимо от флага.

    `drawing_processor._reconcile` на решающей нотации возвращает
    `decisive_category(text) or by_rules`. При включённом флаге
    `decisive_category` и `classify` совпадают, и выражение ничего не
    меняет; при выключенном — `classify` склеенную пометку «№» с цифрой не
    знает и отдаёт «note», а `decisive_category` знает. Разойтись они могут
    только здесь, и побеждает тот, кого и спрашивали: сварной шов,
    названный примечанием, вреда не наносит, но остаётся неверным.
    """

    #                 (заявка модели, надпись, категория по нотации)
    ROWS = [
        ("thread", "⌀32x2,5", "size"),
        ("size", "Ra 6,3", "roughness"),
        ("size", "M24x1,5", "thread"),
        ("note", "R12", "radius"),
        ("thread", "2№3", "position"),
        ("callout", "№1", "position"),
    ]

    @pytest.mark.parametrize("claimed,text,category", ROWS)
    def test_resolution_is_flag_independent(self, strict_off, claimed, text, category):
        # Ровно то выражение, которое стоит в _reconcile.
        resolved = decisive_category(text) or classify(text)[0]
        assert resolved == category
        assert resolved != claimed

    @pytest.mark.parametrize("claimed,text,category", ROWS)
    def test_strict_classify_agrees(self, strict, claimed, text, category):
        assert decisive_category(text) == category
        assert classify(text)[0] == category

    def test_numero_is_the_only_divergence(self, strict_off):
        # При выключенном флаге «№» с цифрой — единственное расхождение
        # между двумя помощниками на решающей нотации.
        assert decisive_category("2№3") == "position"
        assert classify("2№3")[0] == "note"
        for text in ("⌀32x2,5", "Ra 6,3", "M24x1,5", "R12"):
            assert decisive_category(text) == classify(text)[0]


class TestLatinMetreIsNotThread:
    """Строчная «m» перед числом — единица, а не резьба по ГОСТ 8724."""

    @pytest.mark.parametrize("text", [
        "0,1 m3/h",
        "0,2 m3/h",
        "3,6 m3/h",
        "m3/h",
        "Measurement range, m3/h",
        "Not less than 0,25 m2",
        "No more than 3,6 m3/h.",
        "from 0,0040 to 0,04 m3/h",
    ])
    def test_not_thread(self, strict, text):
        assert classify(text)[0] != "thread"

    @pytest.mark.parametrize("text", ["M20x1,5", "M14x1,5*", "M33x1,5*", "M5*", "M8*"])
    def test_real_thread_kept(self, strict, text):
        assert classify(text)[0] == "thread"


class TestTemperatureIsNotSize:
    """«°С» — температура; угловой размер — «45°»."""

    @pytest.mark.parametrize("text", [
        "No more than 60°С",
        "No more than 45°С",
        "not above 45°С",
        "from 5 to 40°С",
        "from 6 to 38,2 °С***",
    ])
    def test_not_size(self, strict, text):
        assert classify(text)[0] != "size"

    def test_angle_still_size(self, strict):
        assert classify("45°")[0] == "size"


class TestToleranceNeedsSize:
    """Посадка и знак отклонения существуют только при самом размере."""

    @pytest.mark.parametrize("text", [
        # Отметка уровня и номер позиции по КЖ — не допуск.
        "-1300", "-2100", "-15530", "-896",
        # Обозначение по системе кодирования — не допуск.
        "10(20)KUA70CQ003-P01",
        # Буква с цифрой сама по себе: посадки на размере нет.
        "fittings K4", "F1", "S27", "n4",
    ])
    def test_not_tolerance(self, strict, text):
        assert classify(text)[0] != "tolerance"

    @pytest.mark.parametrize("text", [
        "2. Неуказанные предельные отклонения размеров: H14; h14, ±IT14/2.",
        "3. Неуказанные пред. откл. размеров: H14, h14, ±IT14/2.",
    ])
    def test_it_grade_is_tolerance(self, strict, text):
        assert classify(text)[0] == "tolerance"

    def test_fit_on_size_is_tolerance(self, strict):
        assert classify("20H7")[0] == "tolerance"


class TestProseGetsNoHeading:
    """Фраза размерной категории не получает: пустой заголовок дешевле лжи."""

    @pytest.mark.parametrize("text", [
        "(±5 % +3) µg/dm3",
        "not above ±4 % from the measured value",
        "19,7 MPa (up to fittings K4",
        "Weighing error no more than ± 30 mg",
        "temperature of at least 5°С, MPa",
        "urement at a sample temperature of 25 °С.",
    ])
    def test_prose_is_note(self, strict, text):
        assert classify(text)[0] == "note"

    def test_grade_is_material_not_a_size(self, strict, monkeypatch):
        # Марку материала узнаёт `dimension_fidelity.is_steel_grade`, то есть
        # только вместе с DRAWING_DIMENSION_FIDELITY. Строгая категория этого
        # не меняет: без того флага марка остаётся без категории, но
        # размерного заголовка не получает ни в одном случае.
        assert classify("12X18H10Т ГОСТ 5582-75")[0] == "note"
        monkeypatch.setattr(settings, "DRAWING_DIMENSION_FIDELITY", True)
        assert classify("12X18H10Т ГОСТ 5582-75")[0] == "material"


class TestTranscriptSigns:
    """Знак «±», потерянный проходом по надписям, но целый в транскрипции."""

    def test_plus_becomes_plus_minus(self):
        # ТАДУ 302318.207.00: транскрипция «1087±3», поле «1087+3».
        fields = [{"category": "size", "value": "1087+3"}]
        fixed = dimension_fidelity.adopt_transcript_signs(
            fields, ["Корпус в сборе", "1087±3", "⌀57"]
        )
        assert fixed == 1
        assert fields[0]["value"] == "1087±3"

    def test_silent_without_a_match(self):
        # Ни одного знака в транскрипции — ни правки, ни трогания полей.
        fields = [
            {"category": "size", "value": "⌀57"},
            {"category": "size", "value": "1087+3"},
        ]
        before = [dict(f) for f in fields]
        assert dimension_fidelity.adopt_transcript_signs(
            fields, ["Корпус в сборе", "⌀57", "Ra 6,3"]
        ) == 0
        assert fields == before

    def test_reference_star_restored(self):
        fields = [{"category": "size", "value": "30+10"}]
        dimension_fidelity.adopt_transcript_signs(fields, ["30±10*"])
        assert fields[0]["value"] == "30±10*"

    def test_no_sign_in_transcript_keeps_value(self):
        fields = [{"category": "size", "value": "1087+3"}]
        assert dimension_fidelity.adopt_transcript_signs(fields, ["1087+3"]) == 0
        assert fields[0]["value"] == "1087+3"

    def test_other_size_not_touched(self):
        # Другой размер на листе знак не отдаёт.
        fields = [{"category": "size", "value": "1087+3"}]
        assert dimension_fidelity.adopt_transcript_signs(fields, ["20±0,1"]) == 0
        assert fields[0]["value"] == "1087+3"

    def test_plus_minus_is_not_downgraded(self):
        # Обратной подмены не бывает: «±» на «+» не меняется.
        fields = [{"category": "size", "value": "9,5±0,5*"}]
        assert dimension_fidelity.adopt_transcript_signs(fields, ["9,5+0,5"]) == 0
        assert fields[0]["value"] == "9,5±0,5*"

    def test_no_transcript(self):
        fields = [{"category": "size", "value": "1087+3"}]
        assert dimension_fidelity.adopt_transcript_signs(fields, None) == 0
        assert dimension_fidelity.adopt_transcript_signs(fields, []) == 0
