"""
Связь размера с тем, что он измеряет (`DRAWING_DIMENSION_LINKS`).

Геометрия проверяется на синтетических векторных PDF, нарисованных здесь же
(fitz): связь у них известна заранее. Ответ модели — подменённый: проверяется
не модель, а то, что сомнительный ответ связи не даёт.
"""

import pytest

fitz = pytest.importorskip("fitz")
pytest.importorskip("PIL")
from PIL import Image  # noqa: E402

from core.models.contract import DrawingField  # noqa: E402
from core.providers import dimension_links as dl  # noqa: E402
from core.providers.qwen_client import SheetReading  # noqa: E402

PAGE_W, PAGE_H = 842.0, 595.0
SHEET = (2339, 1653)  # рендер 200 dpi той же страницы


def arrow(page, tip, toward):
    """Закрашенная стрелка: остриё в `tip`, основание — в сторону `toward`."""
    (tx, ty), (bx, by) = tip, toward
    length = ((bx - tx) ** 2 + (by - ty) ** 2) ** 0.5
    ux, uy = (bx - tx) / length, (by - ty) / length
    base = (tx + ux * 10, ty + uy * 10)
    left = (base[0] - uy * 2.5, base[1] + ux * 2.5)
    right = (base[0] + uy * 2.5, base[1] - ux * 2.5)
    page.draw_polyline([tip, left, right, tip], color=(0, 0, 0), fill=(0, 0, 0), closePath=True)


def frac(rect):
    x0, y0, x1, y1 = rect
    return [x0 / PAGE_W, y0 / PAGE_H, x1 / PAGE_W, y1 / PAGE_H]


LINEAR_TEXT = (232.0, 331.0, 268.0, 345.0)
HOLE_TEXT = (652.0, 168.0, 690.0, 182.0)
HOLE = ((600.0, 250.0), 20.0)
LONELY_TEXT = (500.0, 480.0, 540.0, 494.0)


def sheet_pdf(rotation=0):
    """Деталь, линейный размер «300» и выноска «⌀20» к отверстию."""
    document = fitz.open()
    page = document.new_page(width=PAGE_W, height=PAGE_H)
    page.draw_rect(fitz.Rect(100, 200, 400, 300), color=(0, 0, 0), width=1.4)
    # Выносные линии и размерная линия со стрелками внутрь.
    page.draw_line((100, 305), (100, 360), color=(0, 0, 0), width=0.5)
    page.draw_line((400, 305), (400, 360), color=(0, 0, 0), width=0.5)
    page.draw_line((100, 350), (400, 350), color=(0, 0, 0), width=0.5)
    arrow(page, (100, 350), (400, 350))
    arrow(page, (400, 350), (100, 350))
    # Отверстие и выноска: полка под надписью, наклонная к острию на контуре.
    (cx, cy), r = HOLE
    page.draw_rect(fitz.Rect(540, 190, 660, 310), color=(0, 0, 0), width=1.4)
    page.draw_circle((cx, cy), r, color=(0, 0, 0), width=1.4)
    tip = (cx + r * 0.7071, cy - r * 0.7071)
    page.draw_line((648, 185), (695, 185), color=(0, 0, 0), width=0.5)
    page.draw_line((648, 185), tip, color=(0, 0, 0), width=0.5)
    arrow(page, tip, (648, 185))
    if rotation:
        page.set_rotation(rotation)
    data = document.tobytes()
    document.close()
    return data


def fields():
    return [
        {"category": "size", "value": "300", "bbox": frac(LINEAR_TEXT)},
        {"category": "size", "value": "⌀20", "bbox": frac(HOLE_TEXT)},
        {"category": "size", "value": "15", "bbox": frac(LONELY_TEXT)},
        {"category": "title_block", "value": "Крышка", "bbox": [0.8, 0.9, 0.95, 0.95]},
    ]


def geometry_of(data, angle=0.0, turn=0, size=SHEET):
    document = fitz.open(stream=data, filetype="pdf")
    page = document[0]
    frame = dl.PageFrame.from_page(page, size, angle, turn)
    geometry = dl.geometry_from_page(page, frame)
    document.close()
    return geometry


def near(a, b, tolerance=0.01):
    return all(abs(x - y) <= tolerance for x, y in zip(a, b))


# ===========================================================================
# Координаты
# ===========================================================================

class TestPageFrame:
    @pytest.mark.parametrize("rotation,angle,turn", [(0, 0.0, 0), (90, 0.0, 0), (270, 1.5, 90), (0, -2.0, 180)])
    def test_point_lands_where_the_rendered_sheet_shows_it(self, rotation, angle, turn):
        """Точка страницы попадает туда, где её видно на выправленном рендере."""
        from core.providers import drawing_vision

        document = fitz.open()
        page = document.new_page(width=400, height=200)
        page.draw_rect(fitz.Rect(48, 18, 52, 22), color=(0, 0, 0), fill=(0, 0, 0))
        page.set_rotation(rotation)
        pixmap = page.get_pixmap(dpi=144)
        image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
        sheet = drawing_vision.reproduce(image, angle, turn)
        frame = dl.PageFrame.from_page(page, sheet.size, angle, turn)
        x, y = frame(50, 20)
        grey = sheet.convert("L")
        dark = [(i % sheet.size[0], i // sheet.size[0]) for i, v in enumerate(grey.getdata()) if v < 100]
        cx = sum(p[0] for p in dark) / len(dark)
        cy = sum(p[1] for p in dark) / len(dark)
        assert abs(cx - x) <= 2.0 and abs(cy - y) <= 2.0
        document.close()


# ===========================================================================
# Геометрия
# ===========================================================================

class TestVectorLinks:
    def test_arrows_are_found_and_not_counted_as_lines(self):
        geometry = geometry_of(sheet_pdf())
        assert len(geometry.arrows) == 3
        assert all(arrow.direction is not None for arrow in geometry.arrows)

    def test_linear_dimension_spans_the_extension_lines(self):
        geometry = geometry_of(sheet_pdf())
        link, why = dl.vector_link(fields()[0], geometry)
        assert why == ""
        assert link["method"] == "vector"
        assert link["confidence"] == dl.VECTOR_CONFIDENCE_BOTH
        assert link["target_label"] == "размер по горизонтали"
        x0, y0, x1, y1 = link["target_bbox"]
        assert abs(x0 - 100 / PAGE_W) < 0.01 and abs(x1 - 400 / PAGE_W) < 0.01
        assert y0 < 310 / PAGE_H < 350 / PAGE_H < y1
        tips = sorted(link["leader"])
        assert near(tips[0], [100 / PAGE_W, 350 / PAGE_H]) and near(tips[1], [400 / PAGE_W, 350 / PAGE_H])

    def test_overall_size_is_named_when_the_view_is_known(self):
        geometry = geometry_of(sheet_pdf())
        view = [95 / PAGE_W, 195 / PAGE_H, 405 / PAGE_W, 365 / PAGE_H]
        link, _ = dl.vector_link(fields()[0], geometry, views=[view])
        assert link["target_label"] == "габаритная длина"

    def test_leader_points_at_the_hole(self):
        geometry = geometry_of(sheet_pdf())
        link, why = dl.vector_link(fields()[1], geometry)
        assert why == ""
        assert link["target_label"] == "диаметр отверстия"
        (cx, cy), r = HOLE
        assert near(link["target_bbox"], frac((cx - r, cy - r, cx + r, cy + r)), 0.006)
        start, end = link["leader"][0], link["leader"][-1]
        assert near(end, [(cx + r * 0.7071) / PAGE_W, (cy - r * 0.7071) / PAGE_H])
        assert near(start, [648 / PAGE_W, 185 / PAGE_H])
        assert link["confidence"] == dl.VECTOR_CONFIDENCE_LEADER

    def test_text_without_lines_gets_no_link(self):
        geometry = geometry_of(sheet_pdf())
        link, why = dl.vector_link(fields()[2], geometry)
        assert link is None and why == "no_lines"

    def test_rotated_page_gives_the_same_links(self):
        """Страница с /Rotate, выправленная поворотом: связь та же в координатах листа."""
        from core.providers import drawing_vision

        data = sheet_pdf(rotation=90)
        document = fitz.open(stream=data, filetype="pdf")
        pixmap = document[0].get_pixmap(dpi=72)
        document.close()
        image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
        sheet = drawing_vision.reproduce(image, 0.0, 90)
        assert sheet.size == (842, 595)
        geometry = geometry_of(data, 0.0, 90, sheet.size)
        link, why = dl.vector_link(fields()[0], geometry)
        assert why == "" and link["target_label"] == "размер по горизонтали"
        assert abs(link["target_bbox"][0] - 100 / PAGE_W) < 0.01

    def test_two_equal_dimension_lines_are_ambiguous(self):
        document = fitz.open()
        page = document.new_page(width=PAGE_W, height=PAGE_H)
        for y, (a, b) in ((350, (100, 400)), (326, (150, 350))):
            page.draw_line((a, y), (b, y), color=(0, 0, 0), width=0.5)
            arrow(page, (a, y), (b, y))
            arrow(page, (b, y), (a, y))
        data = document.tobytes()
        document.close()
        item = {"category": "size", "value": "300", "bbox": frac((232, 332, 268, 344))}
        link, why = dl.vector_link(item, geometry_of(data))
        assert link is None and why == "ambiguous"

    def test_line_without_arrows_is_not_a_dimension(self):
        document = fitz.open()
        page = document.new_page(width=PAGE_W, height=PAGE_H)
        page.draw_line((100, 350), (400, 350), color=(0, 0, 0), width=0.5)
        data = document.tobytes()
        document.close()
        link, why = dl.vector_link(fields()[0], geometry_of(data))
        assert link is None and why == "no_arrows"

    def test_scan_in_pdf_has_no_geometry(self):
        document = fitz.open()
        document.new_page(width=PAGE_W, height=PAGE_H)
        data = document.tobytes()
        document.close()
        assert geometry_of(data) is None


# ===========================================================================
# Ответ модели
# ===========================================================================

ASKED = [(1, {"category": "size", "value": "⌀20", "bbox": [0.40, 0.30, 0.45, 0.33]}),
         (2, {"category": "size", "value": "505±10", "bbox": [0.20, 0.70, 0.28, 0.73]})]
SIZE = (2000, 1400)


def answer(*rows):
    return {"links": list(rows)}


def good(**over):
    row = {"id": 1, "target": "диаметр отверстия", "target_bbox": [300, 380, 360, 440],
           "leader": [[400, 330], [355, 385]]}
    row.update(over)
    return row


class TestVlmValidation:
    def test_good_answer_in_thousandths(self):
        found, dropped = dl.validate_vlm(answer(good()), ASKED, SIZE)
        assert dropped == {}
        link = found[1]
        assert link["method"] == "vlm" and link["target_label"] == "диаметр отверстия"
        assert link["target_bbox"] == [0.3, 0.38, 0.36, 0.44]
        assert link["leader"] == [[0.4, 0.33], [0.355, 0.385]]
        assert link["confidence"] == dl.VLM_CONFIDENCE_LEADER
        DrawingField(category="size", value="⌀20", link=link)

    def test_fractions_are_accepted_too(self):
        row = good(target_bbox=[0.3, 0.38, 0.36, 0.44], leader=[[0.4, 0.33], [0.355, 0.385]])
        found, _ = dl.validate_vlm(answer(row), ASKED, SIZE)
        assert found[1]["target_bbox"] == [0.3, 0.38, 0.36, 0.44]

    def test_span_target_gets_thickness(self):
        """Длина отрезком «от выносной до выносной» — рамка нулевой толщины."""
        row = good(target="габаритная длина", target_bbox=[150, 400, 660, 400], leader=None)
        found, dropped = dl.validate_vlm(answer(row), ASKED, SIZE)
        assert dropped == {}
        x0, y0, x1, y1 = found[1]["target_bbox"]
        assert (x0, x1) == (0.15, 0.66) and y1 - y0 == pytest.approx(dl._VLM_TARGET_MIN, abs=1e-4)

    def test_overall_size_may_cover_the_part(self):
        row = good(target="габаритная высота", target_bbox=[180, 160, 860, 870], leader=None)
        found, _ = dl.validate_vlm(answer(row), ASKED, SIZE)
        assert 1 in found
        found, dropped = dl.validate_vlm(answer(good(target_bbox=[180, 160, 860, 870], leader=None)), ASKED, SIZE)
        assert found == {} and dropped == {"bad_bbox": 1}

    def test_prompt_does_not_carry_label_coordinates(self):
        """Иначе модель возвращает рамку надписи как рамку цели (лист крышки, 12 из 12)."""
        prompt = dl.vlm_prompt(ASKED)
        assert "1: «⌀20»" in prompt and "400, 300" not in prompt and "[400" not in prompt

    def test_no_leader_is_kept_with_lower_confidence(self):
        found, _ = dl.validate_vlm(answer(good(leader=None)), ASKED, SIZE)
        assert found[1]["leader"] == [] and found[1]["confidence"] == dl.VLM_CONFIDENCE_BARE

    @pytest.mark.parametrize("row,reason", [
        (good(target_bbox=[300, 380, 1200, 440]), "bad_bbox"),          # за краем листа
        (good(target_bbox=[360, 380, 300, 440]), "bad_bbox"),           # вывернутая рамка
        (good(target_bbox=[0, 0, 900, 900]), "bad_bbox"),               # пол-листа
        (good(target_bbox=[401, 301, 449, 329]), "target_is_label"),    # сама надпись
        (good(target_bbox=[900, 900, 950, 950], leader=None), "too_far"),
        (good(leader=[[100, 100], [355, 385]]), "leader_mismatch"),     # начало не у надписи
        (good(leader=[[400, 330], [700, 700]]), "leader_mismatch"),     # конец не у элемента
        (good(leader=[[400, 330]]), "bad_leader"),
        (good(target="hole"), "bad_label"),
        (good(target=""), "bad_label"),
        (good(id=7), "unknown_id"),
        (good(id="x"), "malformed"),
    ])
    def test_doubtful_answer_gives_no_link(self, row, reason):
        found, dropped = dl.validate_vlm(answer(row), ASKED, SIZE)
        assert found == {} and dropped == {reason: 1}

    def test_target_in_title_block_is_dropped(self):
        title_block = [0.25, 0.35, 0.5, 0.5]
        found, dropped = dl.validate_vlm(answer(good()), ASKED, SIZE, forbidden=[title_block])
        assert found == {} and dropped == {"target_in_title_block": 1}

    def test_two_answers_for_one_number_drop_both(self):
        found, dropped = dl.validate_vlm(answer(good(), good(target="ось")), ASKED, SIZE)
        assert found == {} and dropped == {"duplicate_id": 1}

    @pytest.mark.parametrize("data", [None, {}, {"links": "нет"}, ["links"]])
    def test_no_answer(self, data):
        found, dropped = dl.validate_vlm(data, ASKED, SIZE)
        assert found == {} and dropped == {"no_answer": 1}


# ===========================================================================
# Применение к полям листа
# ===========================================================================

class TestApply:
    def test_vector_first_then_one_model_call_for_the_rest(self):
        items = fields()
        calls = []

        def ask(image, prompt, count):
            calls.append((image.size, prompt, count))
            # Номер 1 — первая несвязанная геометрией величина: «15».
            return answer({"id": 1, "target": "толщина стенки", "target_bbox": [600, 800, 660, 860],
                           "leader": [[600, 810], [620, 830]]})

        image = Image.new("RGB", SHEET, "white")
        stats = dl.apply(items, image=image, geometry=geometry_of(sheet_pdf()), ask=ask)
        assert len(calls) == 1 and calls[0][2] == 1 and "«15»" in calls[0][1]
        assert stats["dimensions"] == 3
        assert stats["linked_vector"] == 2 and stats["linked_vlm"] == 1 and stats["linked"] == 3
        assert items[0]["link"]["method"] == "vector" and items[2]["link"]["method"] == "vlm"
        assert "link" not in items[3]

    def test_scan_goes_to_the_model_only(self):
        items = fields()[:2]
        stats = dl.apply(items, image=Image.new("RGB", SHEET, "white"), ask=lambda *a: None)
        assert stats["asked_vlm"] == 2 and stats["linked"] == 0
        assert stats["dropped"] == {"no_answer": 1} and stats["vlm_unlinked"] == 2
        assert all("link" not in item for item in items)

    def test_no_time_no_call(self):
        calls = []
        stats = dl.apply(fields()[:2], image=Image.new("RGB", SHEET, "white"),
                         ask=lambda *a: calls.append(a), has_time=lambda: False)
        assert calls == [] and stats["dropped"] == {"no_time": 2}

    def test_cancelled_sheet_is_not_swallowed(self):
        from core import deadline

        def ask(*args):
            raise deadline.Cancelled()

        with pytest.raises(deadline.Cancelled):
            dl.apply(fields()[:2], image=Image.new("RGB", SHEET, "white"), ask=ask)

    def test_model_error_gives_no_links(self):
        def ask(*args):
            raise RuntimeError("сеть")

        items = fields()[:2]
        stats = dl.apply(items, image=Image.new("RGB", SHEET, "white"), ask=ask)
        assert stats["linked"] == 0 and all("link" not in item for item in items)

    def test_title_block_values_are_not_linked(self):
        item = {"category": "size", "value": "1:2", "bbox": [0.8, 0.9, 0.85, 0.95]}
        assert not dl.linkable(item, forbidden=[[0.6, 0.8, 1.0, 1.0]])
        assert dl.linkable(item)
        assert not dl.linkable({"category": "size", "value": "5", "bbox": [0, 0, 1, 1]})

    def test_stale_link_is_removed(self):
        items = [{"category": "note", "value": "x", "bbox": [0.1, 0.1, 0.2, 0.2], "link": {"old": 1}}]
        dl.apply(items)
        assert "link" not in items[0]

    def test_marked_image_keeps_size(self):
        image = Image.new("RGB", (800, 600), "white")
        marked = dl.marked_image(image, ASKED)
        assert marked.size == image.size and marked is not image
        assert image.getpixel((320, 180)) == (255, 255, 255)


# ===========================================================================
# Контракт и проводка
# ===========================================================================

class TestContract:
    def test_no_link_no_key(self):
        assert "link" not in DrawingField(category="size", value="20").model_dump()

    def test_link_is_serialised(self):
        link = {"target_label": "диаметр", "target_bbox": [0.1, 0.1, 0.2, 0.2],
                "leader": [[0.1, 0.1], [0.2, 0.2]], "method": "vector", "confidence": 0.9}
        dumped = DrawingField(category="size", value="⌀20", link=link).model_dump()
        assert dumped["link"] == link


class Storage:
    def __init__(self, data):
        self.data = data

    def read_bytes(self, uri):
        return self.data


class Model:
    model = "fake-vl"
    available = True

    def __init__(self):
        self.asked = []

    def read_sheet(self, image, hint=""):
        return SheetReading(sheet_type="detail", annotations=[
            {"category": "size", "value": "300", "bbox": frac(LINEAR_TEXT)},
            {"category": "size", "value": "⌀20", "bbox": frac(HOLE_TEXT)},
        ])

    def read_dimension_links(self, image, prompt, count):
        self.asked.append(count)


def worker(data, model):
    from core.drawing_processor import DrawingProcessor

    work = DrawingProcessor(storage=Storage(data), qwen=model)
    work.vision_enabled = False
    return work


class TestWiring:
    def test_flag_off_adds_nothing(self, monkeypatch):
        from core import drawing_processor

        monkeypatch.setattr(drawing_processor, "_dimension_links", lambda: False)
        model = Model()
        result = worker(sheet_pdf(), model).process("sheet.pdf", {}, page=1)
        assert "dimension_links" not in result["payload"]
        assert all("link" not in item for item in result["parsed_fields"])
        assert model.asked == []

    def test_flag_on_links_the_vector_sheet(self, monkeypatch):
        from core import drawing_processor

        monkeypatch.setattr(drawing_processor, "_dimension_links", lambda: True)
        model = Model()
        result = worker(sheet_pdf(), model).process("sheet.pdf", {}, page=1)
        stats = result["payload"]["dimension_links"]
        assert stats["geometry"] is True and stats["linked_vector"] == 2
        assert model.asked == []  # всё связала геометрия — модель не звали
        labels = {item["value"]: item["link"]["target_label"] for item in result["parsed_fields"]}
        assert labels == {"300": "размер по горизонтали", "⌀20": "диаметр отверстия"}

    def test_stats_stay_out_of_the_indexed_payload(self, monkeypatch):
        """Учёт — в служебном, а не в payload: адаптер RAG вывел бы его в индекс."""
        from core import drawing_processor
        from core.models.contract import Content, Fragment, Position, Provenance

        monkeypatch.setattr(drawing_processor, "_dimension_links", lambda: True)
        work = worker(sheet_pdf(), Model())
        result = work.process("sheet.pdf", {}, page=1)
        source = Fragment(fragment_id="f1", type="image", content=Content(image_ref="sheet.pdf"),
                          position=Position(page=1, bbox=[0, 0, 1, 1]), confidence=0.9, completeness=1.0,
                          provenance=Provenance(method="mineru_ocr", strategy_level=3, source="mineru"))
        fragment = work.build_fragment(result, source, {})
        assert "dimension_links" not in (fragment.content.structured_payload or {})
        stats = fragment.completeness_mechanisms["drawing_diagnostics"]["dimension_links"]
        assert stats["linked_vector"] == 2
        links = [field.link for field in fragment.content.structured_drawing_fields]
        assert all(link is not None and link.method == "vector" for link in links)
