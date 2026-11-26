"""
Правка чертёжного текста по закрытым словарям (DRAWING_LEXICON_FIX, C15, F5).

Все фамилии придуманы (`tests/fixtures/lexicon/`): настоящие подписанты
заказчика в тесты не кладутся.

Проверяется:

* фамилия в графе подписанта ставится к ближайшей из списка, если
  расхождение мало и ближайшая одна; две близкие записи — отказ и пометка
  `unverified_signer`, прочтение остаётся;
* прочтение подписи, которое ни на кого не похоже, фамилией не становится;
* номер ГОСТ: точка внутри номера снимается по форме или по списку, номер
  системы с точкой («ГОСТ 2.104-2006») не трогается;
* термины в графах штампа; цифра марки не меняется никогда;
* с выключенным флагом вывод и ключ кэша прежние; со включённым ключ
  зависит от содержимого списков;
* сборщик списка подписантов по синтетическим результатам.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from core.config import settings
from core.models.contract import Content, DrawingField, Fragment, Position, Provenance
from core.providers import designation_repair, lexicon_fix

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures" / "lexicon"
SIGNERS = str(FIXTURES / "signers.txt")
TERMS = str(FIXTURES / "terms.txt")
GOST = str(FIXTURES / "gost.txt")


@pytest.fixture
def lexicon():
    designation_repair.known_names.cache_clear()
    return lexicon_fix.load(SIGNERS, TERMS, GOST)


def field(value, category="title_block"):
    return {"category": category, "value": value, "bbox": [0.7, 0.9, 0.95, 0.95],
            "confidence": 0.8}


def drawing(fields, text=None, block=None, hand=None):
    payload = {"title_block": dict(block or {})}
    if hand is not None:
        payload["title_block_handwritten"] = dict(hand)
    return Fragment(
        fragment_id="d-1", type="drawing",
        content=Content(text=text, structured_drawing_fields=[DrawingField(**f) for f in fields],
                        structured_payload=payload),
        position=Position(page=1, bbox=[0, 0, 1, 1]),
        provenance=Provenance(method="vision_plus_vlm", strategy_level=7, source="drawing"),
        confidence=0.8, completeness=0.8,
    )


def notes_of(fragment):
    return (fragment.completeness_mechanisms.get("drawing_diagnostics") or {}).get(
        lexicon_fix.NOTES_KEY, [])


def values(fragment):
    return [f.value for f in fragment.content.structured_drawing_fields]


# ------------------------------------------------------------------ список

def test_list_is_read_with_initials_and_comments(lexicon):
    assert lexicon.signers == ("Пеструхин", "Зарубякин", "Колбина", "Колбино", "Мельтюков")
    assert "Сдвижка" in lexicon.terms
    assert "ГОСТ 14771-76" in lexicon.gost


def test_distance_glyph_confusions_are_cheap():
    # «а»/«о», «б»/«в», «Г»/«Е» — дешевле обычной замены.
    assert lexicon_fix.distance("Зорубякин", "Зарубякин") == pytest.approx(0.4)
    assert lexicon_fix.distance("Зерубякин", "Зарубякин") == pytest.approx(1.0)
    # Латиница-двойник и «ё» — без цены.
    assert lexicon_fix.distance("Мeльтюкoв", "Мельтюков") == 0
    # Цифра не становится другой цифрой.
    assert lexicon_fix.distance("12Х18Н9Т", "12Х18Н10Т") >= 1e6


# --------------------------------------------------------- подписанты

def test_role_surname_snaps_to_signer(lexicon):
    sheet = drawing([field("Разраб.: Пестрякин П.П."), field("Пров.: Зорубякин")],
                    text="Разраб. Пестрякин П.П.\nПров. Зорубякин",
                    block={"developed": "Пестрякин П.П.", "checked": "Зорубякин"})
    stats = lexicon_fix.apply([sheet], lexicon)
    assert values(sheet) == ["Разраб.: Пеструхин П.П.", "Пров.: Зарубякин"]
    assert sheet.content.structured_payload["title_block"] == {
        "developed": "Пеструхин П.П.", "checked": "Зарубякин"}
    assert sheet.content.text == "Разраб. Пеструхин П.П.\nПров. Зарубякин"
    assert stats["changed"] == 2
    # Прочтение до правки — в учёте.
    befores = {n["before"] for n in notes_of(sheet)}
    assert {"Пестрякин", "Зорубякин"} <= befores
    assert all(f.unverified_signer is None for f in sheet.content.structured_drawing_fields)


def test_exact_and_uppercase_signer_untouched(lexicon):
    sheet = drawing([field("Утв.: МЕЛЬТЮКОВ"), field("Н.контр.: Колбина")])
    lexicon_fix.apply([sheet], lexicon)
    assert values(sheet) == ["Утв.: МЕЛЬТЮКОВ", "Н.контр.: Колбина"]
    assert notes_of(sheet) == []
    assert "lexicon_fixes" not in sheet.content.structured_payload


def test_ambiguous_surname_is_kept_and_marked(lexicon):
    # «Колбину» одинаково близко к «Колбина» и к «Колбино»: какая — неизвестно.
    sheet = drawing([field("Т.контр.: Колбину")])
    lexicon_fix.apply([sheet], lexicon)
    (only,) = sheet.content.structured_drawing_fields
    assert only.value == "Т.контр.: Колбину"
    assert only.unverified_signer is True
    note = notes_of(sheet)[0]
    assert note["kind"] == lexicon_fix.UNVERIFIED and note["reason"] == "ambiguous"


def test_unknown_printed_surname_is_kept_and_marked(lexicon):
    sheet = drawing([field("Пров.: Шварценбах")])
    lexicon_fix.apply([sheet], lexicon)
    (only,) = sheet.content.structured_drawing_fields
    assert only.value == "Пров.: Шварценбах" and only.unverified_signer is True
    assert only.model_dump()["unverified_signer"] is True


def test_scribble_reading_is_not_a_name(lexicon):
    # Подпись в поле подшивки: модель прочитала закорючку фамилией.
    sheet = drawing([field("Подп. и дата: Шварценбах Ш.Ш. 08.22 (рукописно)"),
                     field("Подп.: Хрюндиков")],
                    hand={"signed_date": "Разраб. Шварценбах 26.08.22"})
    stats = lexicon_fix.apply([sheet], lexicon)
    assert values(sheet) == ["Подп. и дата: 08.22 (рукописно)"]
    assert stats["dropped"] == 1
    assert sheet.content.structured_payload["title_block_handwritten"] == {
        "signed_date": "Разраб. 26.08.22"}
    notes = notes_of(sheet)
    # Учёт не в payload: оттуда снятая фамилия вернулась бы в индекс.
    assert "lexicon_fixes" not in sheet.content.structured_payload
    assert {n["before"] for n in notes if n["kind"] == lexicon_fix.DROPPED} == {
        "Шварценбах Ш.Ш.", "Хрюндиков", "Шварценбах"}


def test_scribble_close_to_signer_snaps(lexicon):
    sheet = drawing([field("Подп. и дата: Пестрякин 08.22 (рукописно)")])
    lexicon_fix.apply([sheet], lexicon)
    assert values(sheet) == ["Подп. и дата: Пеструхин 08.22 (рукописно)"]


def test_no_signers_list_no_signer_step():
    sheet = drawing([field("Пров.: Шварценбах")])
    lexicon_fix.apply([sheet], lexicon_fix.Lexicon())
    (only,) = sheet.content.structured_drawing_fields
    assert only.value == "Пров.: Шварценбах" and only.unverified_signer is None


# ------------------------------------------------------------------ ГОСТ

def test_gost_dot_removed_by_list(lexicon):
    text, notes = lexicon_fix.fix_gost("Шов ГОСТ 14.771-76", lexicon.gost)
    assert text == "Шов ГОСТ 14771-76" and notes[0]["before"] == "ГОСТ 14.771-76"


def test_gost_dot_removed_by_form_without_list():
    text, _ = lexicon_fix.fix_gost("Сварка по ГОСТ 16.037-80, ГОСТ Р 50.632-93")
    assert text == "Сварка по ГОСТ 16037-80, ГОСТ Р 50632-93"


def test_gost_system_numbers_untouched():
    source = "ГОСТ 2.104-2006; ГОСТ 14.004-83; ГОСТ 12.1.005-88; ГОСТ Р 52857.1-2007; ОСТ 26.260-81"
    assert lexicon_fix.fix_gost(source) == (source, [])
    # Без списка «14.771» — номер системы ЕСТПП: по форме не правится.
    assert lexicon_fix.fix_gost("ГОСТ 14.771-76")[0] == "ГОСТ 14.771-76"


def test_gost_list_keeps_listed_dotted(lexicon):
    assert lexicon_fix.fix_gost("ГОСТ 2.104-2006", lexicon.gost)[0] == "ГОСТ 2.104-2006"
    # Склейки нет в списке — не правится.
    assert lexicon_fix.fix_gost("ГОСТ 16.037-80", lexicon.gost)[0] == "ГОСТ 16.037-80"


def test_gost_in_fields_and_text(lexicon):
    sheet = drawing([field("Материал: Лист 2 ГОСТ 14.771-76", category="material")],
                    text="Сварные швы по ГОСТ 14.771-76")
    lexicon_fix.apply([sheet], lexicon)
    assert values(sheet) == ["Материал: Лист 2 ГОСТ 14771-76"]
    assert sheet.content.text == "Сварные швы по ГОСТ 14771-76"


# --------------------------------------------------------------- термины

def test_term_snaps_in_title_block(lexicon):
    sheet = drawing([field("Наименование: Сдувка"), field("Материал: Сталь 12Х18Н1ОТ", "material")],
                    block={"name": "Сдувка"})
    lexicon_fix.apply([sheet], lexicon)
    assert values(sheet) == ["Наименование: Сдвижка", "Материал: Сталь 12Х18Н10Т"]
    assert sheet.content.structured_payload["title_block"]["name"] == "Сдвижка"


def test_term_far_or_digit_change_refused(lexicon):
    # «Сдвиг» короче на два знака; «12Х18Н9Т» — другая марка (цифра).
    for source in ("Наименование: Сдвиг", "Материал: 12Х18Н9Т", "Наименование: Корпус"):
        text, _ = lexicon_fix.fix_terms(source.split(": ")[1], lexicon.terms)
        assert text == source.split(": ")[1]


def test_terms_not_applied_to_sizes(lexicon):
    sheet = drawing([field("Сдувка 12", category="size")])
    lexicon_fix.apply([sheet], lexicon)
    assert values(sheet) == ["Сдувка 12"]


# ------------------------------------------------------ флаг и ключ кэша

def test_flag_is_off_by_default():
    assert settings.DRAWING_LEXICON_FIX is False
    assert settings.DRAWING_LEXICON_SIGNERS_PATH == ""
    assert settings.DRAWING_LEXICON_TERMS_PATH == ""
    assert settings.DRAWING_LEXICON_GOST_PATH == ""


def test_flag_off_processor_does_not_call_fix(temp_storage, monkeypatch):
    from core.drawing_processor import DrawingProcessor

    def boom(*_a, **_k):
        raise AssertionError("правка по словарям вызвана при выключенном флаге")

    monkeypatch.setattr(lexicon_fix, "apply", boom)
    sheet = drawing([field("Разраб.: Пестрякин")])
    processor = DrawingProcessor(temp_storage, qwen=type("Q", (), {"available": False, "model": "x"})())
    out = processor.enrich_fragments([sheet], {})
    assert [f.value for f in out[0].content.structured_drawing_fields] == ["Разраб.: Пестрякин"]
    assert "unverified_signer" not in out[0].content.structured_drawing_fields[0].model_dump()


def test_flag_on_processor_applies_fix(temp_storage, monkeypatch):
    from core.drawing_processor import DrawingProcessor

    designation_repair.known_names.cache_clear()
    monkeypatch.setattr(settings, "DRAWING_LEXICON_FIX", True)
    monkeypatch.setattr(settings, "DRAWING_LEXICON_SIGNERS_PATH", SIGNERS)
    sheet = drawing([field("Разраб.: Пестрякин")])
    processor = DrawingProcessor(temp_storage, qwen=type("Q", (), {"available": False, "model": "x"})())
    out = processor.enrich_fragments([sheet], {})
    assert [f.value for f in out[0].content.structured_drawing_fields] == ["Разраб.: Пеструхин"]


def test_cache_key_unchanged_with_flag_off(monkeypatch, tmp_path):
    from core.ladder import result_cache

    before = result_cache.cache_key("abc", "pdf", {})
    # Пути без флага ключ не меняют.
    monkeypatch.setattr(settings, "DRAWING_LEXICON_SIGNERS_PATH", SIGNERS)
    assert result_cache.cache_key("abc", "pdf", {}) == before
    monkeypatch.setattr(settings, "DRAWING_LEXICON_FIX", True)
    with_list = result_cache.cache_key("abc", "pdf", {})
    assert with_list != before
    # Другое содержимое списка — другой ключ.
    other = tmp_path / "signers.txt"
    other.write_text("Зарубякин\n", encoding="utf-8")
    monkeypatch.setattr(settings, "DRAWING_LEXICON_SIGNERS_PATH", str(other))
    assert result_cache.cache_key("abc", "pdf", {}) not in (before, with_list)


def test_digest_ignores_line_endings(tmp_path):
    a, b = tmp_path / "a.txt", tmp_path / "b.txt"
    a.write_bytes("Пеструхин\nЗарубякин\n".encode("utf-8"))
    b.write_bytes("Пеструхин\r\nЗарубякин\r\n".encode("utf-8"))
    assert lexicon_fix.digest([str(a)]) == lexicon_fix.digest([str(b)])


# ---------------------------------------------------------------- сборщик

def _result(doc_id, developed, checked=None, extra_field=None):
    block = {"developed": developed}
    if checked:
        block["checked"] = checked
    fields = [{"category": "title_block", "value": f"Разраб.: {developed}"}]
    if extra_field:
        fields.append({"category": "title_block", "value": extra_field})
    return {
        "s3_fileid": f"file-{doc_id}",
        "document_metadata": {"doc_id": doc_id},
        "fragments": [{"type": "drawing", "content": {
            "structured_payload": {"title_block": block},
            "structured_drawing_fields": fields}}],
    }


def test_builder_on_synthetic_results(tmp_path):
    source = tmp_path / "results"
    (source / "sub").mkdir(parents=True)
    docs = [
        _result("d1", "Пеструхин П.П.", "Зарубякин"),
        _result("d2", "Пеструхин", "Зарубякин"),
        _result("d3", "ПЕСТРУХИН", "Зарубякин"),
        _result("d4", "Пестрякин"),  # порча: в той же группе, в меньшинстве
        _result("d5", "Мельтюков"),  # одиночка: меньше --min-docs
        # Подпись (рукописное) не голосует.
        _result("d6", "Зарубякин", extra_field="Подп. и дата: Хрюндиков (рукописно)"),
    ]
    for index, doc in enumerate(docs):
        folder = source / "sub" if index % 2 else source
        (folder / f"r{index}.json").write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    # Ответ со списком документов и тот же документ повторно — голос один.
    (source / "batch.json").write_text(json.dumps({"documents": [docs[0]]}, ensure_ascii=False),
                                       encoding="utf-8")
    (source / "broken.json").write_text("{", encoding="utf-8")
    output = tmp_path / "signers.txt"
    script = ROOT / "tools" / "build_signer_lexicon.py"
    done = subprocess.run(
        [sys.executable, "-I", "-X", "utf8", str(script), str(source), "--output", str(output), "--min-docs", "3"],
        capture_output=True, text=True, encoding="utf-8"
    )
    assert done.returncode == 0, done.stderr
    lines = [l for l in output.read_text(encoding="utf-8").splitlines() if not l.startswith("#")]
    assert lines == ["Зарубякин", "Пеструхин"]
    # Собранный список читается правкой.
    designation_repair.known_names.cache_clear()
    built = lexicon_fix.load(str(output))
    assert built.signers == ("Зарубякин", "Пеструхин")


def test_builder_tie_takes_nothing():
    sys.path.insert(0, str(ROOT / "tools"))
    try:
        import build_signer_lexicon as builder
    finally:
        sys.path.pop(0)
    import collections

    names, disputes = builder.cluster(collections.Counter({"Колбина": 3, "Колбино": 3}), 3, 0.5)
    # Близкие написания поровну: каждое — отдельный человек или порча? Не берётся.
    assert names == [] and disputes
    names, _ = builder.cluster(collections.Counter({"Колбина": 6, "Колбино": 4}), 3, 0.5)
    # Второе написание частое — остаётся отдельной записью.
    assert names == ["Колбина", "Колбино"]


# ------------------------------------------- текст листа и прочтения поля

def test_dropped_scribble_leaves_text_and_readings(lexicon):
    # Снятая «фамилия» по закорючке уходит и из текста листа, и из прочтений
    # поля; сырое прочтение остаётся только в учёте.
    sheet = drawing([field("Подп. и дата: Шварценбах Ш.Ш. 08.22 (рукописно)")],
                    text="Изм. Лист\nПодп. и дата Шварценбах Ш.Ш. 08.22\nМасса 3,4")
    only = sheet.content.structured_drawing_fields[0]
    only.readings = [{"source": "sheet", "value": "Шварценбах Ш.Ш. 08.22"},
                     {"source": "stamp", "value": "Шварценбах"}]
    only.alternatives = ["Шварценбах Ш.Ш. 08.22", "08.22"]
    lexicon_fix.apply([sheet], lexicon)
    assert "Шварценбах" not in sheet.content.text
    assert sheet.content.text == "Изм. Лист\nПодп. и дата 08.22\nМасса 3,4"
    assert only.readings == [{"source": "sheet", "value": "08.22"}]
    assert only.alternatives == ["08.22"]
    assert "Шварценбах" not in json.dumps(sheet.content.model_dump(), ensure_ascii=False)
    assert any(n["before"] == "Шварценбах Ш.Ш." for n in notes_of(sheet))


def test_dropped_whole_field_leaves_text(lexicon):
    sheet = drawing([field("Подп.: Хрюндиков")], text="Подп. Хрюндиков\nДата 08.22")
    lexicon_fix.apply([sheet], lexicon)
    assert values(sheet) == []
    assert sheet.content.text == "Подп.\nДата 08.22"


def test_snapped_name_replaces_misreading_in_readings(lexicon):
    sheet = drawing([field("Пров.: Зорубякин")])
    only = sheet.content.structured_drawing_fields[0]
    only.readings = [{"source": "stamp", "value": "Зорубякин"}, {"source": "sheet", "value": "Пров. Зорубякин"}]
    only.alternatives = ["Зорубякин"]
    lexicon_fix.apply([sheet], lexicon)
    assert [r["value"] for r in only.readings] == ["Зарубякин", "Пров. Зарубякин"]
    assert only.alternatives == ["Зарубякин"]


def test_text_changed_only_next_to_role_label(lexicon):
    # Та же запись в другом месте листа — не графа подписанта: не трогается.
    text = "Пров. Зорубякин\nСогласовано: Зорубякин\nТ. контр. Зорубякин"
    sheet = drawing([field("Пров.: Зорубякин")], text=text)
    lexicon_fix.apply([sheet], lexicon)
    assert sheet.content.text == "Пров. Зарубякин\nСогласовано: Зорубякин\nТ. контр. Зорубякин"


def test_text_label_synonym_and_block_key(lexicon):
    sheet = drawing([], text="Проверил: Зорубякин\nЗорубякин", block={"checked": "Зорубякин"})
    lexicon_fix.apply([sheet], lexicon)
    assert sheet.content.text == "Проверил: Зарубякин\nЗорубякин"


def test_text_unlocated_is_unchanged(lexicon):
    sheet = drawing([field("Пров.: Зорубякин")], text="Зорубякин 12.03 Масса")
    lexicon_fix.apply([sheet], lexicon)
    assert values(sheet) == ["Пров.: Зарубякин"]
    assert sheet.content.text == "Зорубякин 12.03 Масса"


def test_term_in_text_only_on_field_line(lexicon):
    sheet = drawing([field("Сдувка")], text="Сдувка\nСдувка 12 шт. по месту")
    lexicon_fix.apply([sheet], lexicon)
    assert values(sheet) == ["Сдвижка"]
    assert sheet.content.text == "Сдвижка\nСдувка 12 шт. по месту"


# --- Квалитеты в записи неуказанных отклонений ------------------------------

@pytest.mark.parametrize("read, fixed", [
    ("3. Неуказанные пред. откл. размеров: Н14, н14, ±Т14/2.", "3. Неуказанные пред. откл. размеров: H14, h14, ±IT14/2."),
    ("2. Неуказанные предельные отклонения размеров: H14, h14, ±1Т14/2", "2. Неуказанные предельные отклонения размеров: H14, h14, ±IT14/2"),
    ("Неуказанные предельные отклонения ±IT14/2", "Неуказанные предельные отклонения ±IT14/2"),
    ("Т14 Труба", "Т14 Труба"),
    ("откл. Поз. Т15 Сталь Т8", "откл. Поз. Т15 Сталь Т8"),
    ("откл. Н10 гайка", "откл. Н10 гайка"),
    ("Н14 позиция на виде", "Н14 позиция на виде"),
])
def test_fix_grades(read, fixed):
    assert lexicon_fix.fix_grades(read)[0] == fixed
