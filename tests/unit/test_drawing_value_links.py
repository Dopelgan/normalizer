"""
Связь величины с видом (`core/providers/drawing_views.py`).

Проверка идёт по геометрии трёх настоящих листов
(`tests/fixtures/real_sheet_views.py`), а не по придуманным координатам:
главное число здесь — **ноль неверных привязок**, и выдумать его нельзя.
"""

from __future__ import annotations

import json

import pytest

from core.providers import drawing_views
from tests.fixtures import real_sheet_views as real


# ===========================================================================
# Разбор заголовка
# ===========================================================================

@pytest.mark.parametrize("text, name, kind, scale", [
    ("Б (4:1)", "Б", drawing_views.KIND_DETAIL, "4:1"),
    ("А(2:1)", "А", drawing_views.KIND_DETAIL, "2:1"),
    ("В–В", "В-В", drawing_views.KIND_SECTION, ""),
    ("А-А", "А-А", drawing_views.KIND_SECTION, ""),
    ("Разрез В-В", "В-В", drawing_views.KIND_SECTION, ""),
    ("Вид А", "А", drawing_views.KIND_VIEW, ""),
    ("Сечение Б-Б (1:2)", "Б-Б", drawing_views.KIND_SECTION, "1:2"),
    ("А", "А", drawing_views.KIND_VIEW, ""),
])
def test_title_parsed(text, name, kind, scale):
    assert drawing_views.parse_title(text) == (name, kind, scale)


@pytest.mark.parametrize("text", [
    "⌀8,5", "15±0,1", "М24х1,5", "Ra 6,3", "(X+38)±0,2", "12Х18Н10Т",
    "2 места", "п.4", "О", "Х", "С-22-2", "",
    "1. * Размеры для справок.",
])
def test_not_a_title(text):
    """Величина, марка стали и буквы, которыми ГОСТ 2.316 вид не обозначает."""
    assert drawing_views.parse_title(text) is None


def test_unnamed_view_with_scale():
    """Знак «развёрнуто» модель читает как попало — вид признаётся по масштабу."""
    assert drawing_views.parse_title("Q (1:2,5)") == ("", drawing_views.KIND_VIEW, "1:2,5")


# ===========================================================================
# Заголовок против буквы-указания
# ===========================================================================

def _title(name, scale="", bbox=(0.1, 0.1, 0.2, 0.2)):
    return drawing_views.ViewTitle(
        name=name, kind=drawing_views.KIND_VIEW, scale=scale, bbox=list(bbox), index=0,
    )


def test_pointer_letter_is_not_a_title():
    """«Б (4:1)» — заголовок, «Б» у выноски — указание, откуда он снят."""
    chosen, pointers, dropped = drawing_views._chosen(
        [_title("Б", "4:1"), _title("Б")]
    )
    assert [t.scale for t in chosen] == ["4:1"]
    assert len(pointers) == 1 and dropped == 0


def test_two_bare_letters_drop_both():
    """Без масштаба заголовок от стрелки не отличить — видом не признаётся ни одна."""
    chosen, pointers, dropped = drawing_views._chosen([_title("А"), _title("А")])
    assert chosen == [] and pointers == [] and dropped == 2


# ===========================================================================
# Настоящие листы
# ===========================================================================

def _views(data):
    found, signals = drawing_views.views_from(
        drawing_views._components(real.grid(data)),
        real.step(data),
        drawing_views.titles(real.fields(data)),
    )
    return found, signals


def _named(data, found):
    """Вид -> имя из эталона (у вида без буквы имени в выдаче нет)."""
    names = real.title_names(data)
    return {
        view.view_id: names.get(json.dumps(view.title_bbox), view.name or view.view_id)
        for view in found
    }


@pytest.mark.parametrize("name, titles_found", [
    ("kryshka_p1", ["", "А", "Б"]),
    ("rasshir_p1", ["Б", "В"]),
    ("probka_p1", ["А", "Б", "В-В"]),
])
def test_views_of_real_sheet(name, titles_found):
    """
    Какие виды находятся на настоящем листе.

    У крышки это развёртка (знак «развёрнуто», имени нет), «А (1:2)» и
    «Б (4:1)»; главный вид заголовка не имеет и видом не объявляется. У
    расширителя — «Б(1:2)» и «В(1:2)»; вид «А» теряется осознанно: буква у
    стрелки и буква заголовка на этом листе обе без масштаба.
    """
    data = real.sheet(name)
    found, _signals = _views(data)
    assert sorted(view.name for view in found) == sorted(titles_found)


def test_detail_view_knows_where_it_is_taken_from():
    """Выносной элемент «Б (4:1)» снят с развёртки — это видно по букве-указанию."""
    data = real.sheet("kryshka_p1")
    found, _signals = _views(data)
    by_name = {view.name: view for view in found}
    assert by_name["Б"].taken_from == by_name[""].view_id


@pytest.mark.parametrize("name, right, unattributed", [
    ("kryshka_p1", 19, 1),
    ("rasshir_p1", 11, 3),
    ("probka_p1", 23, 0),
])
def test_no_wrong_links_on_real_sheets(name, right, unattributed):
    """
    Главное число: неверных привязок — ноль.

    `right` — величины, привязанные к тому виду, которому они принадлежат на
    листе; `unattributed` — величины вида, оставшиеся без привязки (честный
    недобор). Неверная привязка — это и величина, отданная чужому виду, и
    величина, у которой вида нет, но привязка появилась.
    """
    data = real.sheet(name)
    fields = real.fields(data)
    found, _signals = _views(data)
    records, _counts = drawing_views.links(found, fields, real.step(data))
    names = _named(data, found)
    want = real.expected(data)

    ok, wrong, missed = 0, [], 0
    for record in records:
        expected_name = want.get(record["value"])
        if expected_name is None:
            continue
        got = names.get(record.get("view") or "")
        if record.get("view") is None:
            missed += int(bool(expected_name))
        elif not expected_name:
            wrong.append((record["value"], got, "вида у надписи нет"))
        elif got != expected_name:
            wrong.append((record["value"], got, expected_name))
        else:
            ok += 1
    assert wrong == []
    assert (ok, missed) == (right, unattributed)


def test_hole_diameter_belongs_to_the_unfolded_view():
    """
    «⌀8,5 N отв.» стоит за пределами развёртки и связано с ней выноской —
    привязка берётся из связности, отдельного поиска выносок для этого нет.
    """
    data = real.sheet("kryshka_p1")
    fields = real.fields(data)
    found, _signals = _views(data)
    records, _counts = drawing_views.links(found, fields, real.step(data))
    unfolded = next(view for view in found if view.name == "")
    hole = next(r for r in records if r["value"].startswith("⌀8,5"))
    assert hole["view"] == unfolded.view_id


def test_chamfer_angle_belongs_to_its_detail_view():
    """Угол фаски 35° — на виде Б расширителя, 20° — на виде В."""
    data = real.sheet("rasshir_p1")
    fields = real.fields(data)
    found, _signals = _views(data)
    records, _counts = drawing_views.links(found, fields, real.step(data))
    names = _named(data, found)
    by_value = {r["value"]: names.get(r.get("view") or "") for r in records}
    assert by_value["35°"] == "Б"
    assert by_value["20°"] == "В"


def test_sheet_wide_value_stays_unattributed():
    """Общая шероховатость в углу листа виду не принадлежит и привязки не получает."""
    data = real.sheet("kryshka_p1")
    fields = real.fields(data)
    found, _signals = _views(data)
    records, _counts = drawing_views.links(found, fields, real.step(data))
    roughness = next(r for r in records if r["value"] == "Ra 6,3")
    assert roughness["view"] is None
    assert roughness["basis"] == drawing_views.BASIS_NONE


def test_prose_is_not_a_value_of_a_view():
    """Пункт требований — текст листа, а не величина вида, даже вплотную к нему."""
    data = real.sheet("probka_p1")
    fields = real.fields(data)
    found, _signals = _views(data)
    records, _counts = drawing_views.links(found, fields, real.step(data))
    linked = {r["value"] for r in records}
    assert not any(value.startswith("Технические требования") for value in linked)


# ===========================================================================
# Флаг
# ===========================================================================

def test_disabled_by_default(monkeypatch):
    """Флаг-выключатель: без него `attach` не возвращает ничего."""
    from core.config import settings

    monkeypatch.setattr(settings, "DRAWING_VALUE_VIEW_LINKS", False, raising=False)
    assert drawing_views.enabled() is False
    assert drawing_views.attach(_layout_stub(), []) is None


def _layout_stub():
    class _Layout:
        image = None
        frame = None
        regions: list = []
        signals: dict = {}
        turn = 0
        angle = 0.0

    return _Layout()
