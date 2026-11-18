"""
Названия исполнений на чертеже и наименование на листах 2+.

Все названия придуманы: настоящие названия заказчика в тесты не кладутся.

Проверяется:

* «З»/«3» вместо «Э» (и обратно) в названии исполнения правится только
  до точного совпадения с названием из справочника
  (`DESIGNATION_KNOWN_NAMES_PATH`), и только у чертёжных фрагментов под
  `DRAWING_DESIGNATION_REPAIR`; настоящее название с «З» не меняется;
* лист 2+ с сокращённым штампом получает наименование листа 1 с пометкой
  «sheet1» (`DRAWING_SHEET_NAME_CARRY`); другое обозначение — нет;
* при выключенных флагах вывод тот же, что и раньше.
"""

from core.config import settings
from core.contract_finalize import finalize_fragments
from core.drawing_sheet_name import CARRIED_FROM, SOURCES_KEY
from core.models.contract import Content, DrawingField, Fragment, Position, Provenance
from core.providers import designation_repair

# Придуманный справочник: исполнения с «Э», одно настоящее с «З» и одно
# короткое (короче трёх знаков — не правится).
NAMES = ("ВОЛНА-ПЛЭ-КР", "ПЛЭ", "ГЭ-КР", "МЗК-ДР", "ЛИА", "ЭЖ")


def _names_file(tmp_path, names=NAMES):
    path = tmp_path / "known_designations.txt"
    path.write_text("# справочник\n" + "\n".join(names) + "\n\n", encoding="utf-8")
    designation_repair.known_names.cache_clear()
    return str(path)


def _drawing(text, page=1, block=None, fields=None):
    return Fragment(
        fragment_id=f"d-{page}",
        type="drawing",
        content=Content(
            text=text,
            structured_drawing_fields=[DrawingField(**f) for f in (fields or [])],
            structured_payload={"title_block": dict(block or {})},
        ),
        position=Position(page=page, bbox=[0, 0, 1, 1]),
        provenance=Provenance(method="vision_plus_vlm", strategy_level=7, source="drawing"),
        confidence=0.8, completeness=0.8,
    )


def _text(text):
    return Fragment(
        fragment_id="t-1", type="text", content=Content(text=text),
        position=Position(page=1, bbox=[0, 0, 1, 1]),
        provenance=Provenance(method="ocr", strategy_level=3, source="scan"),
        confidence=0.8, completeness=0.8,
    )


# ------------------------------------------------------------- справочник

def test_known_name_ze_to_e(tmp_path):
    names = designation_repair.known_names(_names_file(tmp_path))
    text, changes = designation_repair.repair_text(
        "Исполнение ВОЛНА-ПЛЗ-КР и ПЛ3, вариант ГЗ-КР.", names)
    assert text == "Исполнение ВОЛНА-ПЛЭ-КР и ПЛЭ, вариант ГЭ-КР."
    assert {c.pattern for c in changes} == {"known_name"}


def test_known_name_latin_lookalikes(tmp_path):
    names = designation_repair.known_names(_names_file(tmp_path))
    text, _ = designation_repair.repair_text("Исполнение ЛUA по заказу", names)
    assert text == "Исполнение ЛИА по заказу"


def test_real_ze_name_unchanged_and_reverse(tmp_path):
    names = designation_repair.known_names(_names_file(tmp_path))
    # «МЗК-ДР» — настоящее название с «З»: остаётся; «МЭК-ДР» — обратная порча.
    text, _ = designation_repair.repair_text("Исполнение МЗК-ДР и МЭК-ДР", names)
    assert text == "Исполнение МЗК-ДР и МЗК-ДР"


def test_unknown_and_short_and_digits_unchanged(tmp_path):
    names = designation_repair.known_names(_names_file(tmp_path))
    source = "Режим КЗ, позиция 3Ж и ЗЖ, исполнение СОСНА-ПЛК, 333."
    text, changes = designation_repair.repair_text(source, names)
    assert text == source and changes == []


def test_without_names_no_known_name_pass():
    source = "Исполнение ВОЛНА-ПЛЗ-КР"
    assert designation_repair.repair_text(source) == (source, [])


def test_missing_file_is_empty():
    designation_repair.known_names.cache_clear()
    assert designation_repair.known_names("/nonexistent/known.txt") == ()
    assert designation_repair.known_names("") == ()


# ------------------------------------------------------------- финализация

def _finalize(fragments):
    return finalize_fragments(fragments, "doc-1", "f.pdf", extract=False)[0]


def test_drawing_repair_under_new_flag(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "DESIGNATION_KNOWN_NAMES_PATH", _names_file(tmp_path))
    monkeypatch.setattr(settings, "DRAWING_DESIGNATION_REPAIR", True)
    monkeypatch.setattr(settings, "SCAN_DESIGNATION_REPAIR", False)
    sheet = _drawing(
        "Клапан ВОЛНА-ПЛЗ-КР", block={"designation": "АБВГ 123456.001", "name": "Клапан ВОЛНА-ПЛЗ-КР"},
        fields=[{"category": "title_block", "value": "Наименование: Клапан ВОЛНА-ПЛЗ-КР"}])
    plain = _text("Исполнение ВОЛНА-ПЛЗ-КР")
    out = _finalize([sheet, plain])
    assert out[0].content.text == "Клапан ВОЛНА-ПЛЭ-КР"
    assert out[0].content.structured_drawing_fields[0].value == "Наименование: Клапан ВОЛНА-ПЛЭ-КР"
    assert out[0].content.structured_payload["title_block"]["name"] == "Клапан ВОЛНА-ПЛЭ-КР"
    assert out[0].content.structured_payload[designation_repair.PAYLOAD_KEY]
    # Текстовый фрагмент — по-прежнему только под SCAN_DESIGNATION_REPAIR.
    assert out[1].content.text == "Исполнение ВОЛНА-ПЛЗ-КР"


def test_scan_flag_alone_unchanged(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "DESIGNATION_KNOWN_NAMES_PATH", _names_file(tmp_path))
    monkeypatch.setattr(settings, "DRAWING_DESIGNATION_REPAIR", False)
    monkeypatch.setattr(settings, "SCAN_DESIGNATION_REPAIR", True)
    out = _finalize([_text("Исполнение ВОЛНА-ПЛЗ-КР"), _drawing("Клапан ВОЛНА-ПЛЗ-КР")])
    assert out[0].content.text == "Исполнение ВОЛНА-ПЛЗ-КР"
    assert out[1].content.text == "Клапан ВОЛНА-ПЛЗ-КР"


# ------------------------------------------------------------- лист 2+

def _sheets(second_designation="АБВГ 123456.001"):
    first = _drawing(
        "лист 1", page=1, block={"designation": "АБВГ 123456.001", "name": "Корпус клапана", "sheet": "1"},
        fields=[{"category": "title_block", "value": "Обозначение: АБВГ 123456.001"},
                {"category": "title_block", "value": "Наименование: Корпус клапана", "confidence": 0.7}])
    second = _drawing(
        "лист 2", page=2, block={"designation": second_designation, "sheet": "2"},
        fields=[{"category": "title_block", "value": f"Обозначение: {second_designation}"}])
    return [first, second]


def test_sheet_name_carried(monkeypatch):
    monkeypatch.setattr(settings, "DRAWING_SHEET_NAME_CARRY", True)
    out = _finalize(_sheets())
    payload = out[1].content.structured_payload
    assert payload["title_block"]["name"] == "Корпус клапана"
    assert payload[SOURCES_KEY] == {"name": CARRIED_FROM} and CARRIED_FROM == "sheet1"
    values = [f.value for f in out[1].content.structured_drawing_fields]
    assert "Наименование: Корпус клапана" in values
    # Лист 1 не меняется.
    assert SOURCES_KEY not in out[0].content.structured_payload


def test_sheet_name_not_carried_other_designation(monkeypatch):
    monkeypatch.setattr(settings, "DRAWING_SHEET_NAME_CARRY", True)
    out = _finalize(_sheets("АБВГ 654321.002"))
    assert "name" not in out[1].content.structured_payload["title_block"]
    assert SOURCES_KEY not in out[1].content.structured_payload


def test_flags_off_identical(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "DESIGNATION_KNOWN_NAMES_PATH", _names_file(tmp_path))
    for flag in ("DRAWING_SHEET_NAME_CARRY", "DRAWING_DESIGNATION_REPAIR", "SCAN_DESIGNATION_REPAIR"):
        monkeypatch.setattr(settings, flag, False)
    fragments = _sheets() + [_drawing("Клапан ВОЛНА-ПЛЗ-КР", page=3)]
    expected = [f.model_copy(deep=True) for f in fragments]
    out = _finalize(fragments)
    for got, before in zip(out, expected):
        assert got.content.model_dump() == before.content.model_dump()


# ------------------------------------------------------------- после ревью

def test_as_read_line_not_repaired(tmp_path):
    names = designation_repair.known_names(_names_file(tmp_path))
    sheet = _drawing(
        "Клапан ВОЛНА-ПЛЗ-КР\nОбозначение (как на листе) — ВОЛНА-ПЛЗ-КР 12.\nЕщё ВОЛНА-ПЛЗ-КР 12",
        block={"designation": "ВОЛНА-ПЛЭ-КР 12", "designation_as_read": "ВОЛНА-ПЛЗ-КР 12"})
    designation_repair.repair_drawing_fragment(sheet, names)
    assert sheet.content.text == (
        "Клапан ВОЛНА-ПЛЭ-КР\nОбозначение (как на листе) — ВОЛНА-ПЛЗ-КР 12.\nЕщё ВОЛНА-ПЛЗ-КР 12")
    assert sheet.content.structured_payload["title_block"]["designation_as_read"] == "ВОЛНА-ПЛЗ-КР 12"


def test_lowercase_token_not_uppercased(tmp_path):
    names = designation_repair.known_names(_names_file(tmp_path))
    source = "исполнение волна-плз-кр и плэ"
    assert designation_repair.repair_text(source, names)[0] == source


def test_names_file_created_later_is_read(tmp_path):
    designation_repair.known_names.cache_clear()
    path = tmp_path / "later.txt"
    assert designation_repair.known_names(str(path)) == ()
    path.write_text("ПЛЭ\n", encoding="utf-8")
    assert designation_repair.known_names(str(path)) == ("ПЛЭ",)
