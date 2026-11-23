"""
Выправка ошибок распознавания по словарю корпуса (`OCR_CORRECTION_DICTIONARY`).

Слова в тестах — общие слова языка и выдуманные сокращения: словарь
заказчика живёт только на его сервере, и в репозиторий его слова не
попадают.
"""

import json

import pytest

from core.config import settings
from core.contract_finalize import finalize_fragments
from core.models.contract import Content, Fragment, Position, Provenance, TableData
from core.providers import ocr_correction
from core.providers.ocr_correction import build_dictionary, correct_text

DICTIONARY = build_dictionary(
    {
        "дорота": "дорога",        # курсивная подмена т/г
        "3ал": "зал",              # цифра вместо буквы
        "mолоко": "молоко",        # латинская буква в русском слове
        "moлoko": "молоко",        # латиница почти целиком
        "ипапка": "и папка",       # слипшийся предлог
        "работаc": "работа с",     # слипшийся предлог в конце, латинская «c»
        "kpы": "кры",              # цепочка: «кры» само ключ — пара отброшена
        "кры": "крыша",
        "столь": "Столь",          # пара без изменения — отброшена
    },
    ["дорога", "молоко", "папка", "работа", "термометр", "кошка", "окно", "стол"],
)


def _fragment(text, method="ocr_layer", type_="text", table=None):
    return Fragment(
        fragment_id="", type=type_,
        content=Content(text=text, table_data=table),
        position=Position(page=1, bbox=[0, 0, 1, 1]),
        confidence=0.9, completeness=None,
        provenance=Provenance(method=method, strategy_level=5, source="pdf"),
    )


class TestDictionary:
    def test_chains_and_noop_pairs_dropped(self):
        assert "kpы" not in DICTIONARY.pairs
        assert "столь" not in DICTIONARY.pairs
        assert DICTIONARY.pairs["кры"] == "крыша"

    def test_missing_or_broken_file_is_empty(self, tmp_path):
        assert ocr_correction.load_dictionary("").empty
        assert ocr_correction.load_dictionary(str(tmp_path / "нет.json")).empty
        broken = tmp_path / "broken.json"
        broken.write_text("{не json", encoding="utf-8")
        assert ocr_correction.load_dictionary(str(broken)).empty

    def test_file_loaded(self, tmp_path):
        path = tmp_path / "dict.json"
        path.write_text(json.dumps({"pairs": {"Дорота": "дорога"}, "words": ["Окно"]},
                                   ensure_ascii=False), encoding="utf-8")
        loaded = ocr_correction.load_dictionary(str(path))
        assert loaded.pairs == {"дорота": "дорога"}
        assert loaded.words == frozenset({"окно"})


class TestCorrectText:
    @pytest.mark.parametrize("before, after", [
        ("Длинная дорота домой.", "Длинная дорога домой."),
        ("ДОРОТА", "ДОРОГА"),
        ("Дорота", "Дорога"),
        ("3ал заседаний", "Зал заседаний"),
        ("свежее mолоко", "свежее молоко"),
        ("ручка ипапка", "ручка и папка"),
        ("Ипапка", "И папка"),
        ("работаc детьми", "работа с детьми"),
        ("MOЛOKO", "МОЛОКО"),
        ("MOЛOKo свежее", "МОЛОКО свежее"),
    ])
    def test_pairs(self, before, after):
        assert correct_text(before, DICTIONARY)[0] == after

    def test_offsets_point_into_original(self):
        text = "Тут дорота и mолоко."
        fixed, edits = correct_text(text, DICTIONARY)
        assert fixed == "Тут дорога и молоко."
        for edit in edits:
            assert text[edit.offset:edit.offset + len(edit.before)] == edit.before
        assert [e.rule for e in edits] == ["pair", "pair"]

    def test_mixed_script_word_resolved_by_words(self):
        # «кошка» с латинскими «o» и «k»: в парах нет, прочтение одно.
        fixed, edits = correct_text("Серая koшка спит", DICTIONARY)
        assert fixed == "Серая кошка спит"
        assert edits[0].rule == "mixed_script"

    def test_mixed_script_with_unknown_reading_left(self):
        assert correct_text("Rизм и Pисп", DICTIONARY)[0] == "Rизм и Pисп"

    def test_english_and_digit_words_untouched(self):
        text = "Model X-200 for the stol АБВГ 405 ГОСТ 1234-56"
        assert correct_text(text, DICTIONARY)[0] == text

    def test_hyphen_break_inside_line_joined(self):
        fixed, edits = correct_text("Это термо- метр в окне", DICTIONARY)
        assert fixed == "Это термометр в окне"
        assert edits[0].rule == "hyphen_join"
        assert edits[0].before == "термо- метр"

    def test_hyphen_break_over_newline_joined(self):
        assert correct_text("термо-\nметр", DICTIONARY)[0] == "термометр"

    def test_hyphen_not_joined_when_both_halves_are_words(self):
        assert correct_text("окно- стол", build_dictionary({}, ["окно", "стол", "окностол"]))[0] \
            == "окно- стол"

    def test_hyphen_not_joined_without_space(self):
        assert correct_text("термо-метр", DICTIONARY)[0] == "термо-метр"

    def test_half_word_at_hyphen_not_replaced_by_pair(self):
        # «дорота-» — половина слова у переноса: пара её не трогает.
        assert correct_text("дорота- ми", DICTIONARY)[0] == "дорота- ми"

    def test_idempotent(self):
        text = "дорота, ипапка, koшка, термо- метр, 3ал"
        once, _ = correct_text(text, DICTIONARY)
        twice, edits = correct_text(once, DICTIONARY)
        assert twice == once and edits == []

    def test_empty_dictionary_changes_nothing(self):
        text = "дорота koшка"
        assert correct_text(text, build_dictionary({}, []))[0] == text


class TestNumberGlyphs:
    @pytest.mark.parametrize("before, after", [
        ("длина 1Омм", "длина 10мм"),
        ("длина 2О мм", "длина 20 мм"),
        ("зазор О,4 мм", "зазор 0,4 мм"),
        ("ход 1О,5", "ход 10,5"),
        ("ток l5 мА", "ток 15 мА"),
        ("нагрев до 1ОО °C", "нагрев до 100 °C"),
        ("размер 1З мм", "размер 13 мм"),
        ("глубина O.15 m", "глубина 0.15 m"),
        ("давление О.О9", "давление 0.09"),
        ("ток 2О А", "ток 20 А"),
        ("неустойка 0,о5%", "неустойка 0,05%"),
    ])
    def test_value_with_unit_or_decimal(self, before, after):
        assert correct_text(before, DICTIONARY, numeric=True)[0] == after

    @pytest.mark.parametrize("text", [
        "раздел 1О Требования",       # номер раздела, единицы нет
        "система 18О 9001",            # «ISO 9001», прочитанное цифрами
        "сталь 12Х18Н1ОТ",             # марка: буквы рядом
        "ООО «Ромашка»",               # слово без цифр
        "позиция О1 мм",               # буква в начале без запятой
        "+0,1 и -0,2",                 # отклонения не трогаются
        "пункт О.7 формы",             # точка и одна цифра — номер пункта
        "код 9ОА",                     # слитная однобуквенная «единица»
        "АБВГ 42145100З.04",           # номер обозначения: длинная целая часть
        "деталь О5.01.35.01",          # код продолжается через точку
        "класс 1,II",                  # римское число
    ])
    def test_not_a_value_untouched(self, text):
        assert correct_text(text, DICTIONARY, numeric=True)[0] == text

    def test_off_without_flag(self):
        assert correct_text("длина 1Омм", DICTIONARY)[0] == "длина 1Омм"


class TestFragment:
    def test_text_layer_never_touched(self):
        for method in ("text_layer_extraction", "office_extraction", "vector_text_layer",
                       "неизвестный"):
            fragment = _fragment("дорота", method=method)
            assert ocr_correction.correct_fragment(fragment, DICTIONARY) == []
            assert fragment.content.text == "дорота"
            assert fragment.content.structured_payload is None

    def test_drawing_fragment_not_touched(self):
        fragment = _fragment("дорота", method="vision_plus_vlm", type_="drawing")
        assert ocr_correction.correct_fragment(fragment, DICTIONARY) == []

    @pytest.mark.parametrize("method", sorted(ocr_correction.OCR_METHODS))
    def test_recognition_text_corrected_with_provenance(self, method):
        fragment = _fragment("Наша дорота", method=method)
        ocr_correction.correct_fragment(fragment, DICTIONARY)
        assert fragment.content.text == "Наша дорога"
        assert fragment.content.structured_payload["ocr_corrections"] == [
            {"rule": "pair", "from": "дорота", "to": "дорога", "offset": 5},
        ]

    def test_table_cells_corrected(self):
        table = TableData(headers=["3ал", "Дата"], rows=[["mолоко", 5], ["", None]])
        fragment = _fragment(None, method="mineru_table", type_="table", table=table)
        records = ocr_correction.correct_fragment(fragment, DICTIONARY)
        assert fragment.content.table_data.headers == ["Зал", "Дата"]
        assert fragment.content.table_data.rows == [["молоко", 5], ["", None]]
        assert [r["cell"] for r in records] == [["header", 0], [0, 0]]


class TestFinalize:
    @pytest.fixture
    def dictionary_file(self, tmp_path, monkeypatch):
        path = tmp_path / "ocr_corrections.json"
        path.write_text(json.dumps({"pairs": {"дорота": "дорога"}, "words": ["дорога"]},
                                   ensure_ascii=False), encoding="utf-8")
        ocr_correction.load_dictionary.cache_clear()
        monkeypatch.setattr(settings, "OCR_CORRECTION_DICTIONARY_PATH", str(path))
        yield path
        ocr_correction.load_dictionary.cache_clear()

    def test_flag_off_by_default(self, dictionary_file):
        assert settings.OCR_CORRECTION_DICTIONARY is False
        fragments, _ = finalize_fragments([_fragment("Наша дорота")], doc_id="d1", extract=False)
        assert fragments[0].content.text == "Наша дорота"

    def test_flag_on_corrects_before_ids_and_twice_safe(self, dictionary_file, monkeypatch):
        monkeypatch.setattr(settings, "OCR_CORRECTION_DICTIONARY", True)
        fragments, _ = finalize_fragments([_fragment("Наша дорота")], doc_id="d1", extract=False)
        fragments, _ = finalize_fragments(fragments, doc_id="d1", extract=True)
        fragment = fragments[0]
        assert fragment.content.text == "Наша дорога"
        assert fragment.content.structured_payload["ocr_corrections"] == [
            {"rule": "pair", "from": "дорота", "to": "дорога", "offset": 5},
        ]
        plain, _ = finalize_fragments([_fragment("Наша дорога")], doc_id="d1", extract=True)
        assert fragment.fragment_id == plain[0].fragment_id

    def test_text_layer_untouched_with_flag(self, dictionary_file, monkeypatch):
        monkeypatch.setattr(settings, "OCR_CORRECTION_DICTIONARY", True)
        fragments, _ = finalize_fragments(
            [_fragment("Наша дорота", method="text_layer_extraction")], doc_id="d1", extract=True)
        assert fragments[0].content.text == "Наша дорота"
