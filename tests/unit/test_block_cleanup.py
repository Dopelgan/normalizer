"""
Уборка мусора листа чертежа (DRAWING_BLOCK_CLEANUP).

Скан листа детали дал три вида мусора: таблицу MinerU во весь лист из
повёрнутых граф поля подшивки, поле с рамкой на пол-листа и подписи граф
штампа отдельными полями. Значения синтетические.
"""

from io import BytesIO

import pytest

from core.config import settings
from core.models.contract import Content, DrawingField, Fragment, Position, Provenance
from core.models.parse_result import ParsedBlock
from core.providers import block_cleanup, field_placement
from core.providers.qwen_client import SheetReading
from core.providers.text_layer import OCR_LAYER, TextLayer, TextLine, reconcile_with_text_layer

# Ячейки «таблицы», которую MinerU собрал из повёрнутого текста граф.
GIBBERISH = ("nad a 0000001006800006 oN godu vohP oN HM При НИ WD upo "
             "eo uN 00000010 oN HM WD xq ad").split()


def _gibberish_table():
    rows = [GIBBERISH[i:i + 3] for i in range(0, len(GIBBERISH), 3)]
    return ParsedBlock(type="table", text="<table></table>", page=1, bbox=[0.05, 0.04, 0.96, 0.97],
                       table_data={"rows": rows})


def _spec_table():
    rows = [["A4", "", str(i), f"АБВГ.123456.0{10 + i}", "Болт М6х20 ГОСТ 7798-70", "4"] for i in range(1, 9)]
    return ParsedBlock(type="table", text="<table></table>", page=1, bbox=[0.03, 0.02, 0.97, 0.98],
                       table_data={"headers": ["Формат", "Зона", "Поз.", "Обозначение", "Наименование", "Кол."],
                                   "rows": rows})


def _layer():
    return TextLayer(pages={1: [
        TextLine(text="Технические требования", page=1, bbox=[0.2, 0.1, 0.8, 0.12], confidence=0.9),
    ]}, page_count=1)


def _english_table():
    words = ("bolt washer nut cover spring gasket housing shaft bearing seal plate "
             "screw of the in a to pin").split()
    rows = [[w, "steel", "zinc plated", "4"] for w in words]
    return ParsedBlock(type="table", text="", page=1, bbox=[0.03, 0.02, 0.97, 0.98],
                       table_data={"rows": rows})


def _abbreviation_table():
    rows = [["Поз.", "Обозн.", "Наим.", "Кол.", "Прим."]] + [
        [str(i), "НИ", "ТУ", "Шт.", "Ст3 сп", "ОТК", "М8", "Ø6", "№п/п", "ГН"] for i in range(1, 8)]
    return ParsedBlock(type="table", text="", page=1, bbox=[0.03, 0.02, 0.97, 0.98],
                       table_data={"rows": rows})


def _reconcile(blocks, drop, drawing_page=None):
    return reconcile_with_text_layer(
        blocks, _layer(), source=OCR_LAYER, frame_table_guard=True,
        keep_structured_tables=True, drop_gibberish_tables=drop, drawing_page=drawing_page)


def _tables(result):
    return [b for b in result if b.type == "table"]


def test_gibberish_page_table_dropped_even_when_structured_tables_kept():
    assert block_cleanup.junk_share(" ".join(GIBBERISH)) >= block_cleanup.JUNK_SHARE
    result, stats = _reconcile([_gibberish_table()], drop=True, drawing_page=lambda page: True)
    assert not _tables(result)
    assert stats.as_dict()["gibberish_tables"] == 1


def test_prod_like_margin_gibberish_scores_as_junk():
    text = "nad a 0000001006800006 oN godu upo vohP oN HM При НИ WD upo"
    assert block_cleanup.junk_share(text) >= block_cleanup.JUNK_SHARE


def test_gibberish_table_kept_on_text_page():
    result, _ = _reconcile([_gibberish_table()], drop=True, drawing_page=lambda page: False)
    assert _tables(result)


def test_gibberish_table_kept_without_flag():
    result, stats = _reconcile([_gibberish_table()], drop=False)
    assert _tables(result)
    assert "gibberish_tables" not in stats.as_dict()


@pytest.mark.parametrize("make", [_spec_table, _english_table, _abbreviation_table])
def test_real_tables_kept_on_drawing_page(make):
    assert block_cleanup.junk_share(block_cleanup.table_text(make())) < block_cleanup.JUNK_SHARE
    result, _ = _reconcile([make()], drop=True, drawing_page=lambda page: True)
    assert _tables(result)


def test_small_table_is_not_judged():
    small = _gibberish_table()
    small.bbox = [0.1, 0.1, 0.4, 0.4]
    assert not block_cleanup.is_junk_page_table(small)


@pytest.mark.parametrize("token", ["1012-05", "АБВГ.123456.001", "М6х20", "3,74", "1:2", "12Х18Н10Т",
                                   "Сталь", "ГОСТ", "Лит.", "и", "Ra", "НИ", "bolt", "of", "a", "Ø6",
                                   "№п/п", "mm", "00000010"])
def test_valid_tokens(token):
    assert not block_cleanup.junk(token)


@pytest.mark.parametrize("token", ["oN", "vohP", "WD", "HM", "xq", "0000001006800006"])
def test_junk_tokens(token):
    assert block_cleanup.junk(token)


# ---------------------------------------------------------------- рамки полей

def _drawing(fields):
    return Fragment(
        fragment_id="f-1", type="drawing",
        content=Content(structured_drawing_fields=fields,
                        structured_payload={"layout": {"angle": 0.0, "turn": 0},
                                            "coordinate_reference": "deskewed_page"}),
        position=Position(page=1, bbox=[0.0, 0.0, 1.0, 1.0]),
        provenance=Provenance(method="vlm_full_page", strategy_level=7, source="drawing"),
        confidence=0.8, completeness=1.0,
    )


def _fields():
    return [
        DrawingField(category="title_block", value="Инв. № подл.: 05", bbox=[0.057, 0.496, 0.96, 0.966],
                     confidence=0.5),
        DrawingField(category="size", value="⌀18", bbox=[0.4, 0.4, 0.43, 0.42], confidence=0.9),
        DrawingField(category="note", value="1. Острые кромки притупить.", bbox=[0.1, 0.1, 0.7, 0.6],
                     confidence=0.9),
    ]


def test_oversized_field_loses_its_box(monkeypatch):
    monkeypatch.setattr(settings, "DRAWING_BLOCK_CLEANUP", True)
    (fragment,) = field_placement.apply([_drawing(_fields())])
    big, size, note = fragment.content.structured_drawing_fields
    assert big.placeable is False and big.bbox == [0.0, 0.0, 1.0, 1.0] and big.bbox_exact is False
    assert big.value == "Инв. № подл.: 05"
    assert size.placeable is None and size.bbox == [0.4, 0.4, 0.43, 0.42]
    # Технические требования законно занимают пол-листа.
    assert note.placeable is None and note.bbox == [0.1, 0.1, 0.7, 0.6]
    assert big.model_dump()["placeable"] is False


def test_oversized_field_kept_without_flag():
    (fragment,) = field_placement.apply([_drawing(_fields())])
    big = fragment.content.structured_drawing_fields[0]
    assert big.bbox == [0.057, 0.496, 0.96, 0.966]
    assert "placeable" not in big.model_dump()


# ---------------------------------------------------------------- подписи граф

@pytest.mark.parametrize("value", ["Лит.", "Масса", "Масштаб", "Лист", "Листов", "Формат", "Разраб.",
                                   "Пров.", "Т.контр.", "Н.контр.", "Утв.", "Изм.", "№ докум.", "Подп.",
                                   "Дата", "Основная надпись: Лит.", "ЛИТ", "изм. лист № докум. подп. дата"])
def test_caption_only_values_dropped(value):
    assert block_cleanup.drop_caption_fields([{"category": "title_block", "value": value}]) == []


@pytest.mark.parametrize("value", ["Масса: 3,74", "Масштаб: 1:2", "Лист 2", "Сталь 20 ГОСТ 1050-2013",
                                   "Подп. и дата: 10.02.2022"])
def test_real_values_kept(value):
    item = {"category": "title_block", "value": value}
    assert block_cleanup.drop_caption_fields([item]) == [item]


def test_caption_of_other_category_kept():
    item = {"category": "note", "value": "Масса"}
    assert block_cleanup.drop_caption_fields([item]) == [item]


class CaptionQwen:
    available = True
    model = "fake-vl"

    def __init__(self, annotations):
        self.annotations = annotations

    def read_sheet(self, image, hint="", **_):
        return SheetReading(sheet_type="detail", annotations=list(self.annotations))

    def read_title_block(self, image):
        return None

    def read_table(self, image):
        return None


SHEET = [
    {"category": "title_block", "value": "Масса: 3,74", "bbox": [0.8, 0.9, 0.85, 0.92]},
    {"category": "title_block", "value": "Лит.", "bbox": [0.7, 0.9, 0.72, 0.92]},
    {"category": "title_block", "value": "№ докум.", "bbox": [0.6, 0.9, 0.62, 0.92]},
    {"category": "size", "value": "⌀18", "bbox": [0.4, 0.4, 0.43, 0.42]},
]


def _sheet_values(temp_storage, monkeypatch, flag):
    from PIL import Image

    from core.drawing_processor import DrawingProcessor

    monkeypatch.setattr(settings, "DRAWING_BLOCK_CLEANUP", flag)
    monkeypatch.setattr(settings, "DRAWING_VISION_ENABLED", False)
    data = BytesIO()
    Image.new("RGB", (840, 1188), "white").save(data, format="PNG")
    temp_storage.write_file("assets/doc/sheet.png", data.getvalue())
    fragment = Fragment(
        fragment_id="doc-frag-1", type="image", content=Content(image_ref="assets/doc/sheet.png"),
        position=Position(page=1, bbox=[0.0, 0.0, 1.0, 1.0], order=1), confidence=0.9, completeness=1.0,
        provenance=Provenance(method="mineru_image", strategy_level=2, source="vector_pdf"),
    )
    out = DrawingProcessor(temp_storage, qwen=CaptionQwen(SHEET)).enrich_fragments([fragment], {})
    sheet = next(f for f in out if f.content.image_ref == "assets/doc/sheet.png")
    return [f.value for f in sheet.content.structured_drawing_fields]


def test_caption_fields_not_emitted_from_sheet(temp_storage, monkeypatch):
    pytest.importorskip("PIL")
    values = _sheet_values(temp_storage, monkeypatch, True)
    assert "Масса: 3,74" in values and "⌀18" in values
    assert "Лит." not in values and "№ докум." not in values


def test_caption_fields_kept_without_flag(temp_storage, monkeypatch):
    pytest.importorskip("PIL")
    values = _sheet_values(temp_storage, monkeypatch, False)
    assert "Лит." in values
