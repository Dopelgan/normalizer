"""
Ярлык обычного документа и его значения: один фрагмент и разметка частей.

Строки взяты из корпуса сохранённых разборов: «Знак поверки:» (на 92
документах — 48 чанков-заголовков без значений), «Имеется:» / «Должно
быть:», «Гл.метролог», номера пунктов СТО. Проверяется то, ради чего
модуль написан: подпись не остаётся без значений, разметка выполняет
контракт чанкера, и при выключенном флаге не меняется ничего.
"""

import pytest

from core import document_parts
from core.config import settings
from core.models.contract import (
    Content,
    DrawingField,
    ExtractedFact,
    Fragment,
    Position,
    Provenance,
    TableData,
)


@pytest.fixture(autouse=True)
def _flag_on(monkeypatch):
    monkeypatch.setattr(settings, "DOCUMENT_PARTS_MARKUP", True)


def fragment(text, *, page=1, order=1, kind="text", section=None,
             table=None, fields=None, confidence=0.9, completeness=0.9):
    return Fragment(
        fragment_id=f"f{order}",
        type=kind,
        content=Content(
            text=text,
            table_data=TableData(**table) if table else None,
            structured_drawing_fields=[DrawingField(**f) for f in (fields or [])] or None,
        ),
        position=Position(page=page, order=order),
        section_title=section,
        confidence=confidence,
        completeness=completeness,
        provenance=Provenance(method="text_layer_extraction", strategy_level=2,
                              source="vector_pdf"),
    )


def texts(fragments):
    return [f.content.text for f in fragments]


# --------------------------------------------------------------- соединение

def test_label_fragment_and_its_values_become_one_fragment():
    out = document_parts.apply([
        fragment("Знак поверки:", order=1),
        fragment("Гл.метролог", order=2),
    ])
    assert texts(out) == ["Знак поверки:\nГл.метролог"]


def test_a_chain_of_labels_travels_to_the_values_at_its_end():
    """«Имеется:» и «Должно быть:» идут подряд, значения — за ними."""
    out = document_parts.apply([
        fragment("Имеется:", order=1),
        fragment("Должно быть:", order=2),
        fragment("Отклонение 0,2 мм", order=3),
    ])
    assert texts(out) == ["Имеется:\nДолжно быть:\nОтклонение 0,2 мм"]


def test_a_bare_clause_number_keeps_the_clause_body():
    out = document_parts.apply([
        fragment("4.8.3", order=1),
        fragment("Несоответствия могут быть обнаружены при поверке", order=2),
    ])
    assert len(out) == 2  # без точки это не номер пункта, а просто число
    out = document_parts.apply([
        fragment("4.8.3.", order=1),
        fragment("Несоответствия могут быть обнаружены при поверке", order=2),
    ])
    assert texts(out) == ["4.8.3.\nНесоответствия могут быть обнаружены при поверке"]


def test_a_trailing_label_takes_its_values_with_the_text_before_it():
    out = document_parts.apply([
        fragment("Поверка проведена в объёме методики.\nЗнак поверки:", order=1),
        fragment("Гл.метролог\nМихалина И.А.", order=2),
    ])
    assert texts(out) == [
        "Поверка проведена в объёме методики.\nЗнак поверки:\nГл.метролог\nМихалина И.А."
    ]


def test_an_empty_fragment_between_the_label_and_its_values_is_stepped_over():
    """Вырезка пустого поля формы в индекс не попадает и не разделяет их."""
    out = document_parts.apply([
        fragment("Знак поверки:", order=1),
        fragment("", kind="image", order=2),
        fragment("Гл.метролог", order=3, section="Гл.метролог"),
    ])
    assert texts(out) == ["Знак поверки:\nГл.метролог", ""]


def test_nothing_is_lost_when_the_label_takes_its_values():
    head, tail = "Знак поверки:", "Гл.метролог\nМихалина И.А."
    out = document_parts.apply([fragment(head, order=1), fragment(tail, order=2)])
    joined = "".join(f.content.text or "" for f in out)
    assert head in joined and tail in joined
    assert len(joined) == len(head) + len(tail) + 1  # добавлен один перевод строки


# ----------------------------------------------------------------- отказы

def test_a_label_before_a_table_is_left_alone():
    out = document_parts.apply([
        fragment("Внешние условия:", order=1),
        fragment(None, kind="table", order=2,
                 table={"headers": ["Параметр"], "rows": [["Температура"]]}),
    ])
    assert texts(out) == ["Внешние условия:", None]


def test_a_label_before_a_drawing_is_left_alone():
    out = document_parts.apply([
        fragment("Размер:", order=1),
        fragment("Основная надпись: 12,3", kind="drawing", order=2,
                 fields=[{"category": "linear", "value": "12,3", "confidence": 0.8}]),
    ])
    assert texts(out) == ["Размер:", "Основная надпись: 12,3"]


def test_values_on_another_page_are_not_claimed():
    out = document_parts.apply([
        fragment("Примечания:", page=3, order=1),
        fragment("1. Допускается в заказных спецификациях…", page=4, order=2),
    ])
    assert len(out) == 2


def test_two_named_sections_are_not_joined():
    """«Работодатель:» и «Работник:» — заголовки двух разделов, не подпись."""
    out = document_parts.apply([
        fragment("Работодатель:", order=1, section="Работодатель:"),
        fragment("Работник:", order=2, section="Работник:"),
    ])
    assert len(out) == 2


def test_a_long_sentence_ending_in_a_colon_is_not_a_label():
    long_line = ("В настоящем стандарте организации использованы ссылки на "
                 "следующие нормативные документы, перечень которых приведён "
                 "в разделе 2 и уточняется ежегодно службой стандартизации "
                 "общества по состоянию на первое января текущего года:")
    out = document_parts.apply([
        fragment(long_line, order=1),
        fragment("ГОСТ Р 8.568-2017", order=2),
    ])
    assert len(out) == 2


def test_a_fragment_that_is_not_a_label_keeps_its_neighbour():
    out = document_parts.apply([
        fragment("Поверка проведена в объёме методики.", order=1),
        fragment("Гл.метролог", order=2),
    ])
    assert len(out) == 2


# ------------------------------------------------------------ разметка частей

def _holds_the_chunker_contract(text, parts):
    """Та же проверка, что делает чанкер RAG, написанная здесь заново.

    Части идут по порядку, касаются друг друга, покрывают текст от `0` до
    `len(text)`; каждый `breaks` — строго внутри своей части и на начале
    строки.
    """
    cursor = 0
    for part in parts:
        assert part["start"] == cursor, parts
        assert part["end"] > part["start"]
        assert isinstance(part["kind"], str) and part["kind"].strip()
        assert isinstance(part["title"], str)
        previous = None
        for offset in part["breaks"]:
            assert part["start"] < offset < part["end"]
            assert text[offset - 1] == "\n"
            assert previous is None or offset > previous
            previous = offset
        cursor = part["end"]
    assert cursor == len(text)
    return True


def test_the_joined_fragment_says_where_its_label_starts():
    out = document_parts.apply([
        fragment("Поверка проведена в объёме методики.\nЗнак поверки:", order=1),
        fragment("Гл.метролог\nМихалина И.А.", order=2),
    ])
    parts = out[0].content.structured_payload[document_parts.PARTS_KEY]
    text = out[0].content.text
    assert _holds_the_chunker_contract(text, parts)
    assert [part["kind"] for part in parts] == ["text", "labelled_values"]
    assert parts[-1]["title"] == "Знак поверки:"
    assert text[parts[-1]["start"]:].startswith("Знак поверки:")


def test_the_part_title_is_the_label_so_a_continuation_repeats_it():
    out = document_parts.apply([
        fragment("Диаметры —", order=1),
        fragment("\n".join(f"⌀{8 + index},5" for index in range(40)), order=2),
    ])
    parts = out[0].content.structured_payload[document_parts.PARTS_KEY]
    text = out[0].content.text
    assert _holds_the_chunker_contract(text, parts)
    assert len(parts) == 1 and parts[0]["title"] == "Диаметры —"
    # Резать внутри можно только по началам строк, и их столько, сколько
    # строк значений: ни одно значение не разрежется посередине.
    assert len(parts[0]["breaks"]) == text.count("\n")


def test_two_labels_with_their_own_values_stay_two_fragments():
    """«Имеется:» и «Должно быть:» получают каждая свои значения.

    Дальше этого соединение не идёт: сложить две подписи с их значениями в
    один фрагмент разбор не просил, а разметка нужна каждой своя.
    """
    out = document_parts.apply([
        fragment("Имеется:", order=1),
        fragment("Отклонение 0,2 мм\nПогрешность 0,05", order=2),
        fragment("Должно быть:", order=3),
        fragment("Отклонение 0,1 мм", order=4),
    ])
    assert texts(out) == ["Имеется:\nОтклонение 0,2 мм\nПогрешность 0,05",
                          "Должно быть:\nОтклонение 0,1 мм"]
    for item in out:
        text = item.content.text
        parts = item.content.structured_payload[document_parts.PARTS_KEY]
        assert _holds_the_chunker_contract(text, parts)
        assert parts[0]["title"] == text.split("\n")[0]


def test_a_fragment_that_was_not_joined_carries_no_markup():
    out = document_parts.apply([
        fragment("Поверка проведена в объёме методики.", order=1),
        fragment("Гл.метролог", order=2),
    ])
    for item in out:
        assert not (item.content.structured_payload or {}).get(document_parts.PARTS_KEY)


# --------------------------------------------------------------- флаг выключен

def test_the_flag_off_changes_nothing(monkeypatch):
    monkeypatch.setattr(settings, "DOCUMENT_PARTS_MARKUP", False)
    given = [
        fragment("Знак поверки:", order=1),
        fragment("Гл.метролог", order=2),
        fragment("Внешние условия:", order=3),
        fragment(None, kind="table", order=4,
                 table={"headers": ["Параметр"], "rows": [["Температура"]]}),
    ]
    before = [item.model_dump(mode="json") for item in given]
    out = document_parts.apply(given)
    assert out is given
    assert [item.model_dump(mode="json") for item in out] == before


def test_the_joined_fragment_is_not_sent_to_a_person():
    """Соединение не ставит review_required и не добавляет предупреждений."""
    out = document_parts.apply([
        fragment("Знак поверки:", order=1),
        fragment("Гл.метролог", order=2),
    ])
    assert out[0].review_required is False
    assert out[0].warnings == []


def test_the_worse_participant_decides_confidence_and_completeness():
    out = document_parts.apply([
        fragment("Знак поверки:", order=1, confidence=0.95, completeness=0.9),
        fragment("Гл.метролог", order=2, confidence=0.4, completeness=0.6),
    ])
    assert out[0].confidence == 0.4
    assert out[0].completeness == 0.6


def test_a_carried_fact_points_at_the_fragment_it_now_lives_in():
    """Модуль вызван после извлечения фактов: факт не теряется и не чужой."""
    label = fragment("Знак поверки:", order=1)
    values = fragment("Погрешность 0,05 мм", order=2)
    values.extracted_facts = [ExtractedFact.model_validate(
        {"fact_id": "fact-1", "fragment_id": values.fragment_id,
         "key": "tolerance", "label": "Погрешность",
         "value": {"kind": "number", "raw": "0,05", "number": 0.05, "unit": "мм"}},
    )]
    out = document_parts.apply([label, values])
    assert len(out) == 1
    assert [fact.fragment_id for fact in out[0].extracted_facts] == [out[0].fragment_id]
