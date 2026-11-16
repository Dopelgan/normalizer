"""Одна запись номера свидетельства, заводского номера и обозначения (SCAN_CANONICAL_IDS)."""

import pytest

from core.config import settings
from core.contract_finalize import finalize_fragments
from core.gateway import profile as gateway_profile
from core.models.contract import (
    Content, DrawingField, Fragment, Position, Provenance, TableData,
)
from core.providers import canonical_ids


def scan(text: str, page: int = 1, source: str = "scanned_pdf") -> Fragment:
    return Fragment(
        fragment_id="", type="text",
        content=Content(text=text),
        position=Position(page=page, bbox=[0, 0, 1, 1]),
        confidence=0.9, completeness=None,
        provenance=Provenance(method="ocr_layer", strategy_level=5, source=source),
    )


@pytest.fixture
def enabled(monkeypatch):
    monkeypatch.setattr(settings, "SCAN_CANONICAL_IDS", True)


def _facts(fragment, key):
    return [f for f in fragment.extracted_facts if f.key == key]


# ------------------------------------------------------------ свидетельство
def test_document_vote_fixes_series():
    fragments = [
        scan("Свидетельство № С-ВЫТ/27-09-2022/189200853"),
        scan("№С-ВЫГ/27-09-2022/189200853", page=2),
        scan("№ С-ВЫГ/27-09-2022/189200853", page=3),
    ]
    canonical_ids.apply(fragments)
    assert fragments[0].content.text == "Свидетельство № С-ВЫГ/27-09-2022/189200853"
    record = fragments[0].completeness_mechanisms[canonical_ids.MECHANISMS_KEY][0]
    assert record["read_as"] == "С-ВЫТ/27-09-2022/189200853"
    assert record["evidence"] == "document_vote"


def test_profile_fixes_single_mention_and_keeps_raw():
    fragment = scan("№С-ВЫТ/27-09-2022/189200853")
    canonical_ids.apply([fragment], series=["С-ВЫГ"])
    assert fragment.content.text == "№С-ВЫГ/27-09-2022/189200853"
    fact = _facts(fragment, canonical_ids.FACT_CERTIFICATE)[0]
    assert fact.value.raw == "С-ВЫГ/27-09-2022/189200853"
    assert fact.provenance == {"read_as": "С-ВЫТ/27-09-2022/189200853", "evidence": "profile"}


def test_latin_reading_folds_to_profile_series():
    fragment = scan("WC-BbIT/27-09-2022/189200853")
    canonical_ids.apply([fragment], series=["С-ВЫГ"])
    assert fragment.content.text == "WС-ВЫГ/27-09-2022/189200853"
    assert fragment.position.text_offsets_exact is False


def test_unknown_series_stays():
    fragment = scan("№ С-ВИЯ/01-02-2023/123456789")
    canonical_ids.apply([fragment], series=["С-ВЫГ"])
    assert fragment.content.text == "№ С-ВИЯ/01-02-2023/123456789"
    assert not fragment.completeness_mechanisms
    # Номер всё равно отдан фактом — как прочитан, без доказательства.
    fact = _facts(fragment, canonical_ids.FACT_CERTIFICATE)[0]
    assert fact.value.raw == "С-ВИЯ/01-02-2023/123456789"
    assert fact.provenance["evidence"] is None


def test_tie_without_profile_stays():
    fragments = [scan("С-ВЫТ/27-09-2022/189200853"), scan("С-ВЫГ/27-09-2022/189200853")]
    assert canonical_ids.apply(fragments) == 0
    assert fragments[0].content.text == "С-ВЫТ/27-09-2022/189200853"


def test_vote_does_not_merge_different_series():
    # Две разные серии с одним хвостом не похожи друг на друга — голос
    # не переписывает ни одну.
    fragments = [scan("С-МА/27-09-2022/189200853"), scan("С-МА/27-09-2022/189200853"),
                 scan("С-ВЫГ/27-09-2022/189200853")]
    canonical_ids.apply(fragments)
    assert fragments[2].content.text == "С-ВЫГ/27-09-2022/189200853"


def test_table_cells_are_read_too():
    fragment = Fragment(
        fragment_id="", type="table",
        content=Content(table_data=TableData(headers=["Номер"], rows=[["С-ВЫТ/27-09-2022/189200853"]])),
        position=Position(page=1, bbox=[0, 0, 1, 1]),
        confidence=0.9, completeness=None,
        provenance=Provenance(method="mineru_table", strategy_level=5, source="scanned_pdf"),
    )
    canonical_ids.apply([fragment], series=["С-ВЫГ"])
    assert fragment.content.table_data.rows[0][0] == "С-ВЫГ/27-09-2022/189200853"


def test_vector_pdf_is_not_touched():
    fragment = scan("С-ВЫТ/27-09-2022/189200853", source="vector_pdf")
    canonical_ids.apply([fragment], series=["С-ВЫГ"])
    assert fragment.content.text == "С-ВЫТ/27-09-2022/189200853"
    assert not fragment.extracted_facts


@pytest.mark.parametrize("left,right,expected", [
    ("С-ВЫТ", "С-ВЫГ", 1), ("С-8ЫГ", "С-ВЫГ", 1), ("С-ВЫТГ", "С-ВЫГ", 1),
    ("С-ВИЯ", "С-ВЫГ", 2), ("С-8ЫТ", "С-ВЫГ", 2), ("С-ВЫГ1", "С-ВЫГ", 2),
])
def test_series_distance(left, right, expected):
    assert canonical_ids.series_distance(left, right) == expected


# --------------------------------------------------------- заводской номер
def test_serial_in_declared_range():
    fragment = scan("Заводской номер 22075о")
    canonical_ids.apply([fragment], source_name="СБ210 зав. №№ 220723-220772 свидетельства.pdf")
    assert fragment.content.text == "Заводской номер 220750"
    fact = _facts(fragment, canonical_ids.FACT_SERIAL)[0]
    assert (fact.value.raw, fact.provenance["read_as"]) == ("220750", "22075о")


def test_serial_outside_range_stays():
    fragment = scan("Заводской номер 22095о")
    canonical_ids.apply([fragment], source_name="СБ210 зав. №№ 220723-220772.pdf")
    assert fragment.content.text == "Заводской номер 22095о"


def test_serial_range_from_snapshot_key_digits():
    fragment = scan("Заводской номер 22075о")
    canonical_ids.apply([fragment], file_id="rag-parser-snapshots/ab/__210____.___220723-220772___.pdf")
    assert fragment.content.text == "Заводской номер 220750"


# -------------------------------------------------------------- обозначение
def test_designation_from_file_name():
    fragment = scan("ТАдУ 405210.001 ГУ")
    canonical_ids.apply([fragment], source_name="ТАДУ 405210.001 Корпус.pdf")
    assert fragment.content.text == "ТАДУ 405210.001 ГУ"


def test_designation_with_other_digits_stays():
    fragment = scan("ТАдУ 405210.002")
    canonical_ids.apply([fragment], source_name="ТАДУ 405210.001 Корпус.pdf")
    assert fragment.content.text == "ТАдУ 405210.002"


def test_drawing_title_block_designation():
    fragment = Fragment(
        fragment_id="d1", type="drawing",
        content=Content(
            text="Обозначение TAДУ 405210.001",
            structured_drawing_fields=[DrawingField(category="title_block",
                                                    value="Обозначение: TAДУ 405210.001")],
            structured_payload={"title_block": {"designation": "TAДУ 405210.001"}},
        ),
        position=Position(page=1, bbox=[0, 0, 1, 1]),
        confidence=0.9, completeness=None,
        provenance=Provenance(method="vlm_full_page", strategy_level=7, source="drawing"),
    )
    canonical_ids.apply([fragment], source_name="ТАДУ 405210.001 Корпус.jpg")
    assert fragment.content.structured_payload["title_block"]["designation"] == "ТАДУ 405210.001"
    assert fragment.content.structured_drawing_fields[0].value == "Обозначение: ТАДУ 405210.001"
    assert fragment.content.text == "Обозначение ТАДУ 405210.001"
    assert _facts(fragment, canonical_ids.FACT_DESIGNATION)[0].provenance["read_as"] == "TAДУ 405210.001"


# ----------------------------------------------------------- финализация
def test_flag_off_changes_nothing():
    fragments = [scan("№С-ВЫТ/27-09-2022/189200853"), scan("С-ВЫГ/27-09-2022/189200853"),
                 scan("С-ВЫГ/27-09-2022/189200853")]
    finalized, _ = finalize_fragments(fragments, "doc-1", "f.pdf", source_name="x 220723-220772.pdf")
    assert finalized[0].content.text == "№С-ВЫТ/27-09-2022/189200853"
    assert not any(f.key.startswith("identity.") for f in finalized[0].extracted_facts)


def test_finalize_keeps_fragment_ids(enabled, monkeypatch):
    def run():
        fragments = [scan("№С-ВЫТ/27-09-2022/189200853"), scan("С-ВЫГ/27-09-2022/189200853", 2),
                     scan("С-ВЫГ/27-09-2022/189200853", 3)]
        return finalize_fragments(fragments, "doc-1", "f.pdf")[0]

    monkeypatch.setattr(settings, "SCAN_CANONICAL_IDS", False)
    before = [f.fragment_id for f in run()]
    monkeypatch.setattr(settings, "SCAN_CANONICAL_IDS", True)
    after = run()
    assert [f.fragment_id for f in after] == before
    assert after[0].content.text == "№С-ВЫГ/27-09-2022/189200853"


def test_profile_series_from_profile_file(tmp_path, monkeypatch):
    path = tmp_path / "profile.yaml"
    path.write_text(
        "scan_identity:\n  certificate_series:\n    - С-ВЫГ\n"
        "    - {series: С-МА, note: другой поверитель}\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(settings, "GATEWAY_PROFILE_PATH", str(path))
    gateway_profile.reset_cache()
    try:
        assert canonical_ids.profile_series() == ["С-ВЫГ", "С-МА"]
    finally:
        gateway_profile.reset_cache()


def test_shipped_profile_has_no_series():
    gateway_profile.reset_cache()
    try:
        assert canonical_ids.profile_series() == []
    finally:
        gateway_profile.reset_cache()
