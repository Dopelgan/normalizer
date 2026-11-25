"""
Размерные линии на растре листа (`DRAWING_DIMENSION_GEOMETRY`, карточка N-G1).

Листы рисуются здесь же (PIL): лист A3 при 200 dpi, деталь толстым контуром,
размеры тонкими линиями со стрелками и выносными, надписи — штрихами цифр
в рамке поля. Связь у них известна заранее.
"""

import copy
import time

import pytest

pytest.importorskip("numpy")
from PIL import Image, ImageDraw  # noqa: E402

from core.config import settings  # noqa: E402
from core.models.contract import DrawingField  # noqa: E402
from core.providers import dimension_geometry as dg  # noqa: E402

W, H = 3307, 2339        # A3 при 200 dpi
TEXT = 28                # высота цифр 3,5 мм
THIN, THICK = 2, 5       # размерные и контурные линии
ARROW_LEN, ARROW_HALF = 24, 4
OVER = 12                # выносная за размерной линией


@pytest.fixture(autouse=True)
def flag_on(monkeypatch):
    monkeypatch.setattr(settings, "DRAWING_DIMENSION_GEOMETRY", True, raising=False)


def sheet():
    image = Image.new("L", (W, H), 255)
    draw = ImageDraw.Draw(image)
    draw.rectangle([60, 60, W - 60, H - 60], outline=0, width=THICK)  # рамка формата
    return image, draw


def arrow(draw, tip, direction):
    """Закрашенная стрелка: остриё в `tip`, основание — против `direction`."""
    (tx, ty), (dx, dy) = tip, direction
    bx, by = tx - dx * ARROW_LEN, ty - dy * ARROW_LEN
    draw.polygon([(tx, ty), (bx - dy * ARROW_HALF, by + dx * ARROW_HALF),
                  (bx + dy * ARROW_HALF, by - dx * ARROW_HALF)], fill=0)


def glyphs(draw, box, vertical=False):
    """Штрихи цифр в рамке надписи: чернила текста рядом с линией."""
    x1, y1, x2, y2 = box
    if not vertical:
        x = x1 + 2
        while x + 10 < x2:
            draw.rectangle([x, y1 + 2, x + 2, y2 - 2], fill=0)
            draw.rectangle([x, y1 + 2, x + 10, y1 + 4], fill=0)
            x += 16
    else:
        y = y1 + 2
        while y + 10 < y2:
            draw.rectangle([x1 + 2, y, x2 - 2, y + 2], fill=0)
            draw.rectangle([x1 + 2, y, x1 + 4, y + 10], fill=0)
            y += 16


def h_dim(draw, x1, x2, y, feature_y1, feature_y2, label_x=None, value_len=4, arrows=True):
    """Горизонтальный размер: линия на `y`, выносные от кромок `feature_y*` за линию на OVER."""
    for x, fy in ((x1, feature_y1), (x2, feature_y2)):
        if fy > y:
            draw.line([(x, y - OVER), (x, fy)], fill=0, width=THIN)
        else:
            draw.line([(x, fy), (x, y + OVER)], fill=0, width=THIN)
    draw.line([(x1, y), (x2, y)], fill=0, width=THIN)
    if arrows:
        arrow(draw, (x1, y), (-1, 0))
        arrow(draw, (x2, y), (1, 0))
    cx = (x1 + x2) / 2 if label_x is None else label_x
    half = value_len * 8
    box = (cx - half, y - 10 - TEXT, cx + half, y - 10)
    glyphs(draw, box)
    return box


def v_dim(draw, y1, y2, x, feature_x1, feature_x2, value_len=4):
    """Вертикальный размер: линия на `x`, надпись повёрнута и стоит слева от линии."""
    for y, fx in ((y1, feature_x1), (y2, feature_x2)):
        if fx > x:
            draw.line([(x - OVER, y), (fx, y)], fill=0, width=THIN)
        else:
            draw.line([(fx, y), (x + OVER, y)], fill=0, width=THIN)
    draw.line([(x, y1), (x, y2)], fill=0, width=THIN)
    arrow(draw, (x, y1), (0, -1))
    arrow(draw, (x, y2), (0, 1))
    cy = (y1 + y2) / 2
    half = value_len * 8
    box = (x - 10 - TEXT, cy - half, x - 10, cy + half)
    glyphs(draw, box, vertical=True)
    return box


def field(value, box, category="size", **extra):
    out = {"category": category, "value": value, "confidence": 0.8,
           "bbox": [box[0] / W, box[1] / H, box[2] / W, box[3] / H]}
    out.update(extra)
    return out


def part_sheet():
    """
    Деталь 1200×600 px (масштаб 0,1 мм/px): габариты сверху и слева, цепочка
    из трёх размеров снизу, отверстие с радиусом, второй вид — цилиндр с ⌀.
    """
    image, draw = sheet()
    left, top, right, bottom = 800, 800, 2000, 1400
    draw.rectangle([left, top, right, bottom], outline=0, width=THICK)
    draw.ellipse([1100 - 80, 1100 - 80, 1100 + 80, 1100 + 80], outline=0, width=THICK)
    fields = [
        field("120", h_dim(draw, left, right, 650, top, top)),                       # 0 габарит
        field("60", v_dim(draw, top, bottom, 650, left, left, value_len=2)),          # 1 габарит
        field("30", h_dim(draw, left, 1100, 1550, bottom, bottom, value_len=2)),      # 2 от края
        field("40", h_dim(draw, 1100, 1500, 1550, bottom, bottom, value_len=2)),      # 3 внутри
        field("50", h_dim(draw, 1500, right, 1550, bottom, bottom, value_len=2)),     # 4 от края
    ]
    # Второй вид: цилиндр 400×300, диаметр по образующим справа.
    draw.rectangle([2300, 900, 2700, 1200], outline=0, width=THICK)
    fields.append(field("⌀30", v_dim(draw, 900, 1200, 2850, 2700, 2700, value_len=3)))  # 5
    # Радиус отверстия: выноска от надписи к окружности, стрелка на окружности.
    tip = (1100 + 80 * 0.7071, 1100 + 80 * 0.7071)
    end = (1255, 1255)
    draw.line([tip, end], fill=0, width=THIN)
    arrow(draw, tip, (-0.7071, -0.7071))
    label = (1262, 1241, 1330, 1269)
    glyphs(draw, label)
    fields.append(field("R8", label, category="radius"))                                # 6
    return image, fields


def test_flag_off_changes_nothing(monkeypatch):
    monkeypatch.setattr(settings, "DRAWING_DIMENSION_GEOMETRY", False)
    image, fields = part_sheet()
    before = copy.deepcopy(fields)
    assert dg.apply(fields, image) is None
    assert fields == before


def test_horizontal_overall():
    image, fields = part_sheet()
    stats = dg.apply(fields, image)
    link = fields[0]["link"]
    assert link["method"] == "raster"
    assert link["axis"] == "horizontal" and link["dimension_kind"] == "linear"
    assert abs(link["span_px"] - 1200) <= 4
    assert link["extent_role"] == "overall"
    assert link["target_label"] == "габаритный размер по горизонтали"
    (ax, ay), (bx, by) = link["leader"]
    assert abs(ax * W - 800) <= 3 and abs(bx * W - 2000) <= 3 and abs(ay * H - 650) <= 2
    # Выносные доходят до верхней кромки детали.
    assert all(abs(p[1] * H - 800) <= 4 for p in link["edges"])
    assert link["confidence"] == 0.8
    assert stats["linked"] >= 1


def test_vertical_overall():
    image, fields = part_sheet()
    dg.apply(fields, image)
    link = fields[1]["link"]
    assert link["axis"] == "vertical"
    assert abs(link["span_px"] - 600) <= 4
    assert link["extent_role"] == "overall"
    assert all(abs(p[0] * W - 800) <= 4 for p in link["edges"])


def test_chain_splits_by_extension_lines():
    image, fields = part_sheet()
    dg.apply(fields, image)
    spans = [fields[n]["link"]["span_px"] for n in (2, 3, 4)]
    assert [abs(s - e) <= 4 for s, e in zip(spans, (300, 400, 500))] == [True] * 3
    assert [fields[n]["link"]["extent_role"] for n in (2, 3, 4)] == ["edge", "inner", "edge"]


def test_diameter_on_side_view():
    image, fields = part_sheet()
    dg.apply(fields, image)
    link = fields[5]["link"]
    assert link["dimension_kind"] == "diameter" and link["axis"] == "vertical"
    assert abs(link["span_px"] - 300) <= 4
    assert link["extent_role"] == "overall"
    assert link["target_label"] == "диаметр"


def test_radius_leader():
    image, fields = part_sheet()
    dg.apply(fields, image)
    link = fields[6]["link"]
    assert link["dimension_kind"] == "radius" and "axis" not in link
    assert link["target_label"] == "радиус"
    end = link["edges"][0]
    tip = (1100 + 80 * 0.7071, 1100 + 80 * 0.7071)
    assert abs(end[0] * W - tip[0]) <= 6 and abs(end[1] * H - tip[1]) <= 6


def test_scale_check_marks_misread_value():
    image, fields = part_sheet()
    fields[3]["value"] = "70"   # на деле 40: отношение к длине выпадает
    dg.apply(fields, image)
    assert fields[3]["link"]["scale_ok"] is False
    assert fields[3]["needs_review"] is True
    assert all(fields[n]["link"]["scale_ok"] is True for n in (0, 1, 2, 4))
    assert "needs_review" not in fields[0]


def test_no_line_no_link():
    image, draw = sheet()
    box = (1500, 1000, 1560, 1028)
    glyphs(draw, box)
    fields = [field("25", box)]
    stats = dg.apply(fields, image)
    assert "link" not in fields[0]
    assert stats["linked"] == 0 and stats["missed"]


def test_contour_under_text_is_not_a_dimension_line():
    # Число над толстой кромкой детали без выносных — не размер этой кромки.
    image, draw = sheet()
    draw.rectangle([800, 800, 2000, 1400], outline=0, width=THICK)
    box = (1370, 762, 1430, 790)
    glyphs(draw, box)
    fields = [field("25", box)]
    dg.apply(fields, image)
    assert "link" not in fields[0]


def test_line_without_extension_lines_is_refused():
    image, draw = sheet()
    draw.line([(900, 1000), (1700, 1000)], fill=0, width=THIN)
    box = (1270, 962, 1330, 990)
    glyphs(draw, box)
    fields = [field("80", box)]
    stats = dg.apply(fields, image)
    assert "link" not in fields[0]
    assert stats["missed"] == {"no_extension": 1}


def test_centre_line_crossing_is_not_an_end():
    # Осевая пересекает размерную линию далеко в обе стороны: это не выносная.
    image, draw = sheet()
    draw.rectangle([800, 800, 2000, 1400], outline=0, width=THICK)
    box = h_dim(draw, 800, 2000, 650, 800, 800, label_x=1100)
    draw.line([(1400, 560), (1400, 1450)], fill=0, width=THIN)
    fields = [field("120", box)]
    dg.apply(fields, image)
    assert abs(fields[0]["link"]["span_px"] - 1200) <= 4


def test_scan_noise_and_loose_label_boxes():
    # Скан: размытие, серый фон, шум; рамки надписей модели сдвинуты и шире.
    import numpy as np
    from PIL import ImageFilter

    image, fields = part_sheet()
    rng = np.random.default_rng(7)
    grey = np.asarray(image.filter(ImageFilter.GaussianBlur(0.8)), dtype=np.float32)
    grey = np.clip(grey * 0.85 + 30 + rng.normal(0, 20, grey.shape), 0, 255).astype(np.uint8)
    for n, item in enumerate(fields):
        x1, y1, x2, y2 = item["bbox"]
        dy = (8 if n % 2 else -8) / H
        item["bbox"] = [x1 - 0.003, y1 + dy, x2 + 0.003, y2 + dy]
    dg.apply(fields, Image.fromarray(grey))
    roles = [fields[n]["link"].get("extent_role") for n in range(6)]
    assert roles == ["overall", "overall", "edge", "inner", "edge", "overall"]
    assert [fields[n]["link"]["span_px"] for n in range(6)] == pytest.approx([1200, 600, 300, 400, 500, 300], abs=4)


def test_arrow_of_linear_dimension_is_not_a_radius():
    # «R» рядом со стрелкой линейного размера: остриё упирается в тонкую выносную.
    image, draw = sheet()
    draw.rectangle([800, 800, 2000, 1400], outline=0, width=THICK)
    h_dim(draw, 800, 2000, 650, 800, 800)
    box = (2008, 618, 2068, 646)
    glyphs(draw, box)
    fields = [field("R5", box, category="radius")]
    stats = dg.apply(fields, image)
    assert "link" not in fields[0]
    assert stats["missed"] == {"no_arrow": 1}


def test_vector_link_is_kept_and_vlm_confidence_not_raised():
    image, fields = part_sheet()
    vector = {"target_label": "габаритная длина", "target_bbox": [0.1, 0.1, 0.2, 0.2],
              "leader": [], "method": "vector", "confidence": 0.9}
    vlm = {"target_label": "высота детали", "target_bbox": [0.1, 0.1, 0.2, 0.2],
           "leader": [], "method": "vlm", "confidence": 0.45}
    fields[0]["link"] = dict(vector)
    fields[1]["link"] = dict(vlm)
    stats = dg.apply(fields, image)
    assert fields[0]["link"] == vector
    assert stats["vector_kept"] == 1
    link = fields[1]["link"]
    assert link["method"] == "raster" and link["target_label"] == "высота детали"
    assert link["confidence"] == 0.45


def test_not_dimensions_are_skipped():
    for value in ("2×45°", "M8", "S0,5", "30°", "Ra 3,2"):
        assert dg.kind_of(value) is None
    assert dg.kind_of("⌀20H7") == "diameter"
    assert dg.kind_of("R5") == "radius"
    assert dg.kind_of("(120)") == "linear"
    assert dg.number_of("⌀20H7") == 20.0 and dg.number_of("50,5*") == 50.5


def test_contract_keeps_old_link_shape_and_carries_raster_keys():
    old = {"target_label": "диаметр отверстия", "target_bbox": [0.1, 0.1, 0.2, 0.2],
           "leader": [], "method": "vlm", "confidence": 0.6}
    dumped = DrawingField(category="size", value="⌀8", link=old).model_dump()
    assert dumped["link"] == old
    image, fields = part_sheet()
    dg.apply(fields, image)
    raster = DrawingField(**fields[0]).model_dump()["link"]
    assert raster["extent_role"] == "overall" and raster["axis"] == "horizontal"
    assert len(raster["edges"]) == 2 and "scale_ok" in raster


def test_mask_failure_is_soft(monkeypatch):
    image, fields = part_sheet()
    before = copy.deepcopy(fields)

    def broken(*_args, **_kwargs):
        raise RuntimeError("нет памяти")

    monkeypatch.setattr(dg, "_Sheet", broken)
    stats = dg.apply(fields, image)
    assert fields == before and stats["error"] == "RuntimeError"


def busy_sheet(count=40):
    """A3 с `count` размерами в сетке деталей: замер времени на лист."""
    image, draw = sheet()
    fields = []
    for k in range(count // 4):
        col, row = k % 5, k // 5
        x0, y0 = 300 + col * 580, 300 + row * 900
        x1, y1 = x0 + 300, y0 + 400
        draw.rectangle([x0, y0, x1, y1], outline=0, width=THICK)
        fields.append(field("30", h_dim(draw, x0, x1, y0 - 120, y0, y0, value_len=2)))
        fields.append(field("40", v_dim(draw, y0, y1, x0 - 120, x0, x0, value_len=2)))
        fields.append(field("15", h_dim(draw, x0, x0 + 150, y1 + 120, y1, y1, value_len=2)))
        fields.append(field("15", h_dim(draw, x0 + 150, x1, y1 + 120, y1, y1, value_len=2)))
    return image, fields


def test_cpu_time_per_sheet_is_small():
    image, fields = busy_sheet()
    started = time.perf_counter()
    stats = dg.apply(fields, image)
    seconds = time.perf_counter() - started
    assert stats["linked"] == len(fields)
    assert seconds < 3.0   # цель — меньше 1 с; запас на медленный CI
