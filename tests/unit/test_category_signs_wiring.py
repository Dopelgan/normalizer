"""
Две правки разбора под `DRAWING_FIELD_CATEGORY_STRICT`: знак отклонения из
транскрипции и решающая нотация против заявки модели.

Сами `dimension_fidelity.adopt_transcript_signs` и
`vector_drawing.decisive_category` живут на ветке точности категорий
(`fix/field-category-accuracy-20261114`) — здесь проверяется не они, а
проводка: доходят ли до них строки транскрипции и текст надписи, и что
происходит, когда ни той ветки, ни флага в дереве нет. Поведение самих
функций на настоящих значениях проверено в дереве с обеими ветками, и об
этом сказано в примечании к ветке.
"""

import pytest

pytest.importorskip("PIL")
from PIL import Image  # noqa: E402

from core.config import settings
from core import drawing_processor
from core.drawing_processor import (
    DrawingProcessor, _category_strict, _reconcile,
)
from core.providers import dimension_fidelity, vector_drawing

# Настоящая пара с ТАДУ 302318.207.00: транскрипция сохранила «±»,
# проход по надписям подменил его на «+».
FIELD_TEXT = "1087+3"
TRANSCRIPT_TEXT = "1087±3"


@pytest.fixture
def strict(monkeypatch):
    """
    Флаг приносит чужая ветка, и в этом дереве его в `Settings` нет: pydantic
    отказывается заводить неизвестное поле даже на запись, то есть включить
    флаг здесь нельзя и случайно. Поэтому подменяется сам читатель.
    """
    monkeypatch.setattr(drawing_processor, "_category_strict", lambda: True)


# Настройку и обе функции приносит ветка точности категорий. До её слияния
# их здесь нет, после — есть, и проверки на отсутствие верны только до.
PAIRED = hasattr(settings, "DRAWING_FIELD_CATEGORY_STRICT")


class TestSafeWithoutItsOwnBranch:
    """
    Пока чужой ветки в дереве нет, обе правки обязаны быть мертвы и ничего
    не ронять. Прямое обращение к настройке подняло бы AttributeError на
    каждом листе, а широкий `except Exception` в обогащении превратил бы
    каждый чертёж в `DRAWING_NOT_RECOGNIZED`.
    """

    def test_reading_the_setting_never_raises(self):
        """Главное свойство: читатель отвечает, а не падает, в любом дереве."""
        assert _category_strict() in (True, False)

    @pytest.mark.skipif(PAIRED, reason="ветка точности категорий уже в дереве")
    def test_the_setting_is_absent_here(self):
        assert not hasattr(settings, "DRAWING_FIELD_CATEGORY_STRICT")

    @pytest.mark.skipif(PAIRED, reason="ветка точности категорий уже в дереве")
    def test_the_helpers_are_absent_here_too(self):
        assert not hasattr(dimension_fidelity, "adopt_transcript_signs")
        assert not hasattr(vector_drawing, "decisive_category")

    @pytest.mark.skipif(PAIRED, reason="ветка точности категорий уже в дереве")
    def test_both_patches_are_dead_without_it(self):
        assert _category_strict() is False

    def test_reconcile_works_as_before_with_the_flag_off(self, monkeypatch):
        """Старое поведение: спор отдан модели, понижена уверенность."""
        monkeypatch.setattr(drawing_processor, "_category_strict", lambda: False)
        assert _reconcile("thread", "size", "⌀32x2,5") == ("thread", 0.6)
        assert _reconcile("size", "size", "⌀32x2,5") == ("size", 0.9)

    def test_the_new_argument_is_optional(self, monkeypatch):
        monkeypatch.setattr(drawing_processor, "_category_strict", lambda: False)
        assert _reconcile("size", "size") == ("size", 0.9)


class TestDecisiveNotationWiring:
    """Текст надписи доходит до `_reconcile`, и спор решается правилами."""

    @staticmethod
    def decisive(monkeypatch, settled):
        monkeypatch.setattr(vector_drawing, "decisive_category",
                            lambda text: settled.get(" ".join(text.split())),
                            raising=False)

    @pytest.mark.parametrize("text,settled,claimed", [
        ("⌀32x2,5", "size", "thread"),
        ("Ra 6,3", "roughness", "size"),
        ("M24x1,5", "thread", "size"),
        ("R12", "radius", "note"),
    ])
    def test_notation_wins_wherever_it_settles_the_question(
        self, monkeypatch, strict, text, settled, claimed,
    ):
        """На этих четырёх записях оба помощника отвечают одинаково."""
        self.decisive(monkeypatch, {text: settled})
        assert _reconcile(claimed, settled, text) == (settled, 0.6)

    def test_agreement_on_decisive_notation_keeps_full_confidence(
        self, monkeypatch, strict,
    ):
        self.decisive(monkeypatch, {"Ra 6,3": "roughness"})
        assert _reconcile("roughness", "roughness", "Ra 6,3") == ("roughness", 0.9)

    def test_the_decisive_answer_wins_not_the_rules_one(
        self, monkeypatch, strict,
    ):
        """
        В эту ветку входят потому, что нотация ответила, поэтому
        возвращается её ответ. Разойтись с `classify` она может только на
        склеенном «№» с цифрой: для «2№3» нотация говорит `position`, а
        старый `classify` — `note`. Раньше правка отдавала `note`, и «2№3»
        уходил из «Резьба» под «Примечание»; теперь — под «Позиция».
        """
        self.decisive(monkeypatch, {"2№3": "position"})
        assert _reconcile("thread", "note", "2№3") == ("position", 0.6)

    def test_the_model_agreeing_with_the_notation_keeps_full_confidence(
        self, monkeypatch, strict,
    ):
        """Уверенность сверяется с тем, что возвращаем, а не с `by_rules`."""
        self.decisive(monkeypatch, {"№1": "position"})
        assert _reconcile("position", "note", "№1") == ("position", 0.9)

    def test_a_judgement_call_is_left_to_the_model(self, monkeypatch, strict):
        """Там, где нотация молчит, спор остаётся спором — как и раньше."""
        self.decisive(monkeypatch, {})
        assert _reconcile("position", "note", "2№3") == ("position", 0.75)

    def test_the_text_actually_reaches_reconcile(self, monkeypatch, strict):
        """Проводка: без текста исключение не сработало бы вовсе."""
        seen = []
        monkeypatch.setattr(vector_drawing, "decisive_category",
                            lambda text: seen.append(text) or None, raising=False)
        worker = DrawingProcessor(storage=object(), qwen=object())
        worker._fields_from_annotations([
            {"category": "thread", "value": "⌀32x2,5", "bbox": [0.1, 0.1, 0.2, 0.2]},
        ])
        assert "⌀32x2,5" in seen

    def test_flag_off_does_not_consult_the_notation(self, monkeypatch):
        monkeypatch.setattr(drawing_processor, "_category_strict", lambda: False)
        called = []
        monkeypatch.setattr(vector_drawing, "decisive_category",
                            lambda text: called.append(text) or "size", raising=False)
        assert _reconcile("thread", "size", "⌀32x2,5") == ("thread", 0.6)
        assert called == []


class TestTranscriptSignsWiring:
    """Строки транскрипции доходят до `adopt_transcript_signs`."""

    @staticmethod
    def worker(monkeypatch, calls):
        monkeypatch.setattr(dimension_fidelity, "adopt_transcript_signs",
                            lambda fields, lines: calls.append((fields, lines)),
                            raising=False)
        monkeypatch.setattr(settings, "DRAWING_DIMENSION_FIDELITY", True)
        monkeypatch.setattr(settings, "DRAWING_IMAGE_FULL_READ", True)
        # Со включённым флагом `_reconcile` спрашивает и вторую функцию
        # чужой ветки: без неё разбор надписи упал бы ещё до правки знака.
        monkeypatch.setattr(vector_drawing, "decisive_category",
                            lambda text: None, raising=False)

        class Sheet:
            model = "fake-vl"
            available = True

            def read_sheet(self, image, hint=""):
                from core.providers.qwen_client import SheetReading
                return SheetReading(sheet_type="detail", annotations=[
                    {"category": "size", "value": FIELD_TEXT,
                     "bbox": [0.1, 0.1, 0.2, 0.2]}])

            def transcribe(self, image):
                return [TRANSCRIPT_TEXT]

        worker = DrawingProcessor(storage=object(), qwen=Sheet())
        worker.vision_enabled = False
        monkeypatch.setattr(worker, "_open",
                            lambda ref, page=1: Image.new("RGB", (400, 300), "white"))
        return worker

    def test_the_transcript_lines_are_handed_over(self, monkeypatch, strict):
        calls = []
        worker = self.worker(monkeypatch, calls)
        worker.process("sheet.jpg", {}, page=1, transcribe=True)
        assert len(calls) == 1
        fields, lines = calls[0]
        assert lines == [TRANSCRIPT_TEXT]
        assert [f["value"] for f in fields] == [FIELD_TEXT]

    def test_the_fields_are_handed_over_after_deduplication(self, monkeypatch, strict):
        """Правка идёт по уже сведённым полям, а не по сырым надписям."""
        calls = []
        worker = self.worker(monkeypatch, calls)
        result = worker.process("sheet.jpg", {}, page=1, transcribe=True)
        assert calls[0][0] is result["parsed_fields"]

    def test_flag_off_hands_nothing_over(self, monkeypatch):
        monkeypatch.setattr(drawing_processor, "_category_strict", lambda: False)
        calls = []
        worker = self.worker(monkeypatch, calls)
        worker.process("sheet.jpg", {}, page=1, transcribe=True)
        assert calls == []

    def test_without_the_fidelity_flag_nothing_is_handed_over(
        self, monkeypatch, strict,
    ):
        """Правка живёт внутри DRAWING_DIMENSION_FIDELITY, как и merge_readings."""
        calls = []
        worker = self.worker(monkeypatch, calls)
        monkeypatch.setattr(settings, "DRAWING_DIMENSION_FIDELITY", False)
        worker.process("sheet.jpg", {}, page=1, transcribe=True)
        assert calls == []
