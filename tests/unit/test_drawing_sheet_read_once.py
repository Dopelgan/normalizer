"""
Лист скана PDF читается моделью один раз (DRAWING_SHEET_READ_ONCE).

При DRAWING_PDF_FULL_READ лист, нарезанный MinerU на N видов, стоил N+1
проходов «лист целиком»: каждая вырезка уходила в модель своим проходом, а
потом лист читался целиком ещё раз. По замеру прода за 36 часов — 137 лишних
проходов, четверть страниц читалась не по одному разу, одна страница восемь
раз.

Проверяется: число проходов модели, возврат к вырезкам при неудавшемся
чтении листа, неприкосновенность ветки DRAWING_SHEET_MISSING_PAGES и объём,
по которому `core/ladder/drawing_route.py` сравнивает чертёж с распознаванием.

Синтетика повторяет форму реальных случаев; сами листы в тесты не кладутся.
"""

import pytest

from core.config import settings
from core.drawing_processor import DrawingProcessor, _field
from core.ladder import drawing_route
from core.models.contract import Content, Fragment, Position, Provenance

pytest.importorskip("PIL")


SHEET_LINES = [
    "Ra 6,3 (√)",
    "8. Лицевую сторону шлифовать, шерохов. не хуже, чем Ra 1,6",
    "ТАДУ 687000.001.001",
]


def view_crop(page, order):
    """Вырезка вида, какую нарезает MinerU из листа скана."""
    return Fragment(
        fragment_id=f"doc-1-frag-{page:03d}-{order}", type="image",
        content=Content(image_ref=f"assets/doc-1/crop-{page}-{order}.png"),
        position=Position(page=page, bbox=[0.2, 0.2, 0.6, 0.6], order=order),
        confidence=0.6, completeness=None,
        provenance=Provenance(method="mineru", strategy_level=5, source="scanned_pdf"),
    )


def scan_metadata(pages=(1,)):
    return {
        "doc_id": "doc-1", "source_path": "src/scan.pdf", "file_type": "pdf",
        "classification": {"genre": "drawing", "source": "raster_only",
                           "signals": {"raster_drawing_pages": list(pages)}},
    }


class FakeQwen:
    available = True
    model = "fake-vl"


def make_processor(temp_storage, monkeypatch, sheet_reads=True):
    """
    Процессор, считающий проходы модели. `sheet_reads=False` — чтение листа
    целиком не удалось: переписанного текста у листа нет.
    """
    proc = DrawingProcessor(temp_storage, qwen=FakeQwen())
    calls = []

    def process(image_ref, metadata, page=1, transcribe=False,
                log_page=None, text_only=False):
        calls.append({"ref": image_ref, "page": page, "transcribe": transcribe,
                      "log_page": log_page})
        payload = {"tables": []}
        if transcribe and sheet_reads:
            payload["transcript_lines"] = list(SHEET_LINES)
        if transcribe and not sheet_reads:
            # Лист целиком прочитать не удалось: ни текста, ни полей.
            return {"detected_count": 0, "avg_confidence": 0.0, "mode": "sheet_only",
                    "parsed_fields": [], "payload": {"tables": []}, "warnings": []}
        return {"detected_count": 1, "avg_confidence": 0.75, "mode": "sheet_only",
                "parsed_fields": [_field("size", "120*", [0.1, 0.1, 0.2, 0.2], 0.75)],
                "payload": payload, "warnings": []}

    monkeypatch.setattr(proc, "process", process)
    proc.calls = calls
    return proc


def sheet_passes(proc):
    """Проходы «лист целиком» — по самому файлу, а не по вырезке."""
    return [call for call in proc.calls if call["ref"] == "src/scan.pdf"]


# --------------------------------------------------------------- число проходов
@pytest.mark.parametrize("views", [1, 2, 4, 8])
def test_sheet_read_once_instead_of_n_plus_one(temp_storage, monkeypatch, views):
    """Лист из N видов: было N+1 проходов модели, стало 1."""
    monkeypatch.setattr(settings, "DRAWING_PDF_FULL_READ", True)
    fragments = [view_crop(1, order) for order in range(1, views + 1)]

    monkeypatch.setattr(settings, "DRAWING_SHEET_READ_ONCE", False)
    before = make_processor(temp_storage, monkeypatch)
    before.enrich_fragments(list(fragments), scan_metadata())
    assert len(before.calls) == views + 1

    monkeypatch.setattr(settings, "DRAWING_SHEET_READ_ONCE", True)
    after = make_processor(temp_storage, monkeypatch)
    result = after.enrich_fragments(list(fragments), scan_metadata())
    assert len(after.calls) == 1
    assert len(sheet_passes(after)) == 1
    # Лист прочитан целиком и встал перед вырезками своей страницы.
    sheets = [f for f in result if f.type == "drawing"]
    assert len(sheets) == 1
    assert (sheets[0].content.structured_payload or {})["transcript_line_count"] == 3
    # Вырезки никуда не делись — они остались картинками на своих местах.
    assert len([f for f in result if f.type == "image"]) == views


def test_pages_counted_separately(temp_storage, monkeypatch):
    """Каждая страница читается своим проходом, лишних проходов нет."""
    monkeypatch.setattr(settings, "DRAWING_PDF_FULL_READ", True)
    monkeypatch.setattr(settings, "DRAWING_SHEET_READ_ONCE", True)
    fragments = [view_crop(1, 1), view_crop(1, 2), view_crop(2, 1), view_crop(3, 1)]
    proc = make_processor(temp_storage, monkeypatch)
    proc.enrich_fragments(fragments, scan_metadata(pages=(1, 2, 3)))
    assert len(proc.calls) == 3
    assert sorted(call["page"] for call in sheet_passes(proc)) == [1, 2, 3]


# ------------------------------------------------- неудавшееся чтение листа
def test_failed_sheet_read_falls_through_to_crops(temp_storage, monkeypatch):
    """
    Лист целиком прочитать не удалось — вырезки читаются как раньше. Иначе
    одна неудача молча теряла бы всю страницу.
    """
    monkeypatch.setattr(settings, "DRAWING_PDF_FULL_READ", True)
    monkeypatch.setattr(settings, "DRAWING_SHEET_READ_ONCE", True)
    fragments = [view_crop(1, 1), view_crop(1, 2)]
    proc = make_processor(temp_storage, monkeypatch, sheet_reads=False)
    result = proc.enrich_fragments(fragments, scan_metadata())
    # Проход листа + обе вырезки: страница прочитана хоть чем-то.
    assert len(proc.calls) == 3
    assert len([f for f in result if f.type == "drawing"]) == 2


def test_crops_on_other_pages_still_read(temp_storage, monkeypatch):
    """Страница без прочитанного листа отдаёт свои вырезки модели."""
    monkeypatch.setattr(settings, "DRAWING_PDF_FULL_READ", True)
    monkeypatch.setattr(settings, "DRAWING_SHEET_READ_ONCE", True)
    # Лестница назвала чертёжной только страницу 1; вырезка со страницы 2
    # листом целиком не прочитана и остаётся за вырезкой.
    fragments = [view_crop(1, 1), view_crop(2, 1)]
    proc = make_processor(temp_storage, monkeypatch)
    proc.enrich_fragments(fragments, scan_metadata(pages=(1,)))
    assert len(proc.calls) == 2
    assert [call["ref"] for call in proc.calls].count("assets/doc-1/crop-2-1.png") == 1


# ----------------------------------------------------- чужая ветка не тронута
def test_missing_pages_branch_untouched(temp_storage, monkeypatch):
    """
    DRAWING_SHEET_MISSING_PAGES: там признак «на листе есть чертёж» даёт сам
    обход вырезок, и порядок остаётся прежним даже при включённом флаге.
    """
    monkeypatch.setattr(settings, "DRAWING_PDF_FULL_READ", False)
    monkeypatch.setattr(settings, "DRAWING_IMAGE_FULL_READ", False)
    monkeypatch.setattr(settings, "DRAWING_SHEET_MISSING_PAGES", True)
    monkeypatch.setattr(settings, "DRAWING_SHEET_READ_ONCE", True)
    fragments = [view_crop(1, 1), view_crop(1, 2)]
    proc = make_processor(temp_storage, monkeypatch)
    proc.enrich_fragments(fragments, scan_metadata(pages=(1, 2)))
    # Вырезки читаются, лист страницы 2 дочитывается: ровно как без флага.
    refs = [call["ref"] for call in proc.calls]
    assert refs.count("assets/doc-1/crop-1-1.png") == 1
    assert refs.count("assets/doc-1/crop-1-2.png") == 1


def test_flag_off_keeps_every_pass(temp_storage, monkeypatch):
    """С выключенным флагом проходы в точности прежние."""
    monkeypatch.setattr(settings, "DRAWING_PDF_FULL_READ", True)
    monkeypatch.setattr(settings, "DRAWING_SHEET_READ_ONCE", False)
    fragments = [view_crop(1, 1), view_crop(1, 2)]
    proc = make_processor(temp_storage, monkeypatch)
    proc.enrich_fragments(fragments, scan_metadata())
    assert len(proc.calls) == 3


# ------------------------------------------------- объём для спора с OCR
def test_volume_drops_by_crop_fields(temp_storage, monkeypatch):
    """
    Счёт, по которому `drawing_route` спорит с распознаванием. Символы на
    странице берутся максимумом по чтениям листа, поэтому текст листа их
    держит; поля же суммируются, и поля вырезок из счёта уходят.
    """
    monkeypatch.setattr(settings, "DRAWING_PDF_FULL_READ", True)
    fragments = [view_crop(1, 1), view_crop(1, 2)]

    monkeypatch.setattr(settings, "DRAWING_SHEET_READ_ONCE", False)
    before = make_processor(temp_storage, monkeypatch)
    old = drawing_route.volume(before.enrich_fragments(list(fragments), scan_metadata()))

    monkeypatch.setattr(settings, "DRAWING_SHEET_READ_ONCE", True)
    after = make_processor(temp_storage, monkeypatch)
    new = drawing_route.volume(after.enrich_fragments(list(fragments), scan_metadata()))

    # Текст листа — тот же: символы страницы не просели.
    assert new.chars == old.chars
    assert new.transcript_lines == old.transcript_lines
    # Поля вырезок из счёта ушли: это и есть цена, за которой надо следить.
    assert new.fields == 1
    assert old.fields == 3


def test_verdict_holds_while_sheet_reads_the_title_block(temp_storage, monkeypatch):
    """
    Состав разбора решает штамп, а штамп несёт только лист целиком: вырезки
    видов его не видят. Поэтому `structured` от обхода вырезок не падает, и
    LADDER_DRAWING_OVER_OCR остаётся при своём исходе.
    """
    monkeypatch.setattr(settings, "DRAWING_PDF_FULL_READ", True)
    monkeypatch.setattr(settings, "DRAWING_SHEET_READ_ONCE", True)
    proc = make_processor(temp_storage, monkeypatch)
    result = proc.enrich_fragments([view_crop(1, 1), view_crop(1, 2)], scan_metadata())
    found = drawing_route.drawing_structure(result)
    # Обозначение по ЕСКД прочитано с листа — состав чертёжный.
    assert found.designation
    assert found.structured


def test_rich_crop_is_the_one_case_that_loses_chars(temp_storage, monkeypatch):
    """
    Единственный случай, где счёт символов и правда просаживается: вырезка
    прочитана богаче, чем лист целиком. Символы страницы — максимум по
    чтениям, так что лист беднее вырезки их и теряет. Такой лист стоит
    держать на виду: с него флаг снимает не только время.
    """
    monkeypatch.setattr(settings, "DRAWING_PDF_FULL_READ", True)
    proc = DrawingProcessor(temp_storage, qwen=FakeQwen())

    def process(image_ref, metadata, page=1, transcribe=False,
                log_page=None, text_only=False):
        if transcribe:
            # Лист прочитан скупо: одна строка текста и штамп.
            return {"detected_count": 1, "avg_confidence": 0.7, "mode": "sheet_only",
                    "parsed_fields": [_field("title_block", "ТАДУ 687000.001.001",
                                             [0.7, 0.9, 1.0, 1.0], 0.8)],
                    "payload": {"tables": [], "transcript_lines": ["Ra 6,3"]},
                    "warnings": []}
        # Вырезка вида: мелкие размеры внутри вида, которых на листе не видно.
        return {"detected_count": 6, "avg_confidence": 0.8, "mode": "sheet_only",
                "parsed_fields": [
                    _field("size", f"{10 + i},5 H7(+0,030)", [0.1, 0.1, 0.2, 0.2], 0.8)
                    for i in range(6)],
                "payload": {"tables": []}, "warnings": []}

    monkeypatch.setattr(proc, "process", process)
    fragments = [view_crop(1, 1), view_crop(1, 2)]

    monkeypatch.setattr(settings, "DRAWING_SHEET_READ_ONCE", False)
    old = drawing_route.volume(proc.enrich_fragments(list(fragments), scan_metadata()))
    monkeypatch.setattr(settings, "DRAWING_SHEET_READ_ONCE", True)
    new = drawing_route.volume(proc.enrich_fragments(list(fragments), scan_metadata()))

    # Символы просели — именно та просадка, которая может опустить объём под
    # нижнюю границу и отдать лист распознаванию.
    assert new.chars < old.chars
    # Но состав разбора держит штамп с листа: `structured` не падает, и
    # исход решает только граница объёма.
    assert drawing_route.drawing_structure(
        proc.enrich_fragments(list(fragments), scan_metadata())).structured
