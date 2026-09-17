"""Приведение прочитанного текста чертежа к записи по ГОСТ."""

import pytest

from core.providers.drawing_text import looks_like_text, normalize


class TestSymbols:
    @pytest.mark.parametrize("text,expected", [
        ("Ø110", "⌀110"),
        ("ø44", "⌀44"),
        ("Φ20", "⌀20"),
        ("Ф20", "⌀20"),          # кириллическая «Ф» вплотную к числу
        ("+-0,1", "±0,1"),
        ("+/- 0.5", "± 0.5"),
    ])
    def test_signs(self, text, expected):
        assert normalize(text) == expected

    @pytest.mark.parametrize("text,expected", [
        ("20Н7", "20H7"),        # посадка кириллицей
        ("⌀44Н7", "⌀44H7"),
        ("М12", "M12"),          # резьба кириллицей
        ("М12х1,5", "M12x1,5"),
        ("M12-6Н", "M12-6H"),
        ("Ка 3.2", "Ra 3.2"),
        ("Rа 1,6", "Ra 1,6"),
    ])
    def test_homoglyphs_in_notation(self, text, expected):
        assert normalize(text) == expected

    @pytest.mark.parametrize("text", [
        "Гайка M4",              # свёртка сломала бы «M4» в «МЧ»
        "Фаска 1x45°",
        "Наименование",
        "Сталь 45 ГОСТ 1050-2013",
        "Не более 3 шт",
    ])
    def test_words_are_left_alone(self, text):
        assert normalize(text) == text

    def test_spaces_are_collapsed(self):
        assert normalize("  Вилка   в  сборе \n") == "Вилка в сборе"

    def test_empty(self):
        assert normalize("") == ""
        assert normalize(None) == ""


class TestReadable:
    @pytest.mark.parametrize("text", ["20", "⌀44H7", "Сталь 45", "Ra 3.2"])
    def test_inscriptions_pass(self, text):
        assert looks_like_text(text)[0]

    @pytest.mark.parametrize("text", ["", "|| --", "—-—", "   "])
    def test_graphics_leftovers_are_rejected(self, text):
        passed, reason = looks_like_text(text)
        assert not passed and reason
