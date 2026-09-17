"""
Синтетический лист чертежа для тестов разметки зрением.

Настоящий лист в репозиторий класть незачем: проверяется геометрия — рамка
формата, сетка штампа и сетка таблицы, — а не содержание надписей. Лист
строится из линий, поэтому ожидаемые координаты известны точно.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

WIDTH, HEIGHT = 1600, 1131


def build(with_table: bool = True, angle: float = 0.0):
    """Лист А3 с рамкой, штампом и (по желанию) таблицей спецификации."""
    from PIL import Image, ImageDraw

    image = Image.new("L", (WIDTH, HEIGHT), 255)
    draw = ImageDraw.Draw(image)

    # Рамка формата: внешняя по краю листа и внутренняя с полем подшивки.
    draw.rectangle([20, 20, WIDTH - 20, HEIGHT - 20], outline=0, width=3)
    draw.rectangle([60, 30, WIDTH - 30, HEIGHT - 30], outline=0, width=3)

    # Штамп по ГОСТ 2.104 — в правом нижнем углу, сетка граф.
    x0, y0 = WIDTH - 30 - 420, HEIGHT - 30 - 160
    x1, y1 = WIDTH - 30, HEIGHT - 30
    draw.rectangle([x0, y0, x1, y1], outline=0, width=2)
    for index in range(1, 5):
        draw.line([x0, y0 + index * 32, x1, y0 + index * 32], fill=0, width=2)
    for offset in (80, 170, 250, 330):
        draw.line([x0 + offset, y0, x0 + offset, y1], fill=0, width=2)

    if with_table:
        sx0, sy0, sx1, sy1 = 120, 80, 620, 320
        draw.rectangle([sx0, sy0, sx1, sy1], outline=0, width=2)
        for index in range(1, 6):
            draw.line([sx0, sy0 + index * 40, sx1, sy0 + index * 40], fill=0, width=2)
        for offset in (60, 260, 380):
            draw.line([sx0 + offset, sy0, sx0 + offset, sy1], fill=0, width=2)

    # Немного графики, чтобы лист не состоял из одних таблиц.
    draw.ellipse([700, 500, 900, 700], outline=0, width=3)
    draw.line([650, 600, 950, 600], fill=0, width=2)

    if angle:
        image = image.rotate(angle, fillcolor=255)
    return image


def title_block_bbox() -> List[float]:
    """Где на самом деле нарисован штамп, в долях листа."""
    x0, y0 = WIDTH - 30 - 420, HEIGHT - 30 - 160
    return [x0 / WIDTH, y0 / HEIGHT, (WIDTH - 30) / WIDTH, (HEIGHT - 30) / HEIGHT]


def table_bbox() -> List[float]:
    return [120 / WIDTH, 80 / HEIGHT, 620 / WIDTH, 320 / HEIGHT]


def png_bytes(**kwargs) -> bytes:
    import io

    buffer = io.BytesIO()
    build(**kwargs).save(buffer, format="PNG")
    return buffer.getvalue()


def overlap(first: List[float], second: List[float]) -> float:
    """Доля площади пересечения от меньшей из двух областей."""
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    intersection = (x2 - x1) * (y2 - y1)
    areas: Tuple[float, float] = (
        (first[2] - first[0]) * (first[3] - first[1]),
        (second[2] - second[0]) * (second[3] - second[1]),
    )
    return intersection / min(areas)


def expected() -> Dict[str, List[float]]:
    return {"title_block": title_block_bbox(), "table": table_bbox()}
