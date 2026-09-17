"""
Генератор тестовых файлов (без внешних зависимостей).

Фикстуры лежат в репозитории, но пересобрать их можно в любой момент:

    python tests/fixtures/generate_fixtures.py

Текст в PDF — латиница: кириллица потребовала бы встроенного Type0-шрифта,
а для проверки конвейера достаточно извлекаемого текстового слоя.
"""

import struct
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent


# ===========================================================================
# PDF
# ===========================================================================

def build_pdf(lines, path: Path) -> None:
    """Одностраничный PDF с текстовым слоем (Helvetica, WinAnsi)."""
    content_lines = ["BT", "/F1 14 Tf", "72 760 Td", "18 TL"]
    for line in lines:
        escaped = line.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")
        content_lines.append(f"({escaped}) Tj T*")
    content_lines.append("ET")
    stream = "\n".join(content_lines).encode("latin-1")

    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] "
        b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>",
    ]

    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"

    xref_offset = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_offset}\n%%EOF\n"
    ).encode()

    path.write_bytes(bytes(out))


# ===========================================================================
# PNG
# ===========================================================================

def build_png(width: int, height: int, draw, path: Path) -> None:
    """8-битный серый PNG. `draw(x, y)` возвращает яркость 0..255."""
    raw = bytearray()
    for y in range(height):
        raw.append(0)  # фильтр строки: None
        for x in range(width):
            raw.append(draw(x, y) & 0xFF)

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data)) + tag + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(bytes(raw), 9))
    png += chunk(b"IEND", b"")
    path.write_bytes(png)


def drawing_pixel(x: int, y: int) -> int:
    """Схематичный «чертёж»: рамка, основная надпись, контур детали."""
    if x < 4 or y < 4 or x > 795 or y > 595:
        return 0                                   # рамка листа
    if 560 <= x <= 790 and 480 <= y <= 590:
        if x in (560, 790) or y in (480, 590) or (y - 480) % 22 == 0:
            return 0                               # основная надпись
    if 200 <= x <= 500 and 150 <= y <= 350:
        if x in (200, 500) or y in (150, 350):
            return 0                               # контур детали
    if 150 <= x <= 550 and y == 400 and x % 12 < 8:
        return 0                                   # размерная линия
    return 255


def main() -> None:
    build_pdf(
        [
            "Technical Specification",
            "",
            "1. Scope",
            "This document defines requirements for the shaft assembly.",
            "",
            "2. Requirements",
            "Surface roughness shall not exceed Ra 3.2 um.",
            "Material: steel 45, GOST 1050-2013.",
            "Tolerance: h7 for all mating diameters.",
        ],
        HERE / "sample.pdf",
    )
    build_pdf(
        [
            "Calculation Sheet",
            "",
            "Parameter    Value    Unit",
            "Diameter     20.0     mm",
            "Length       145.5    mm",
            "Mass         0.36     kg",
        ],
        HERE / "sample_table.pdf",
    )
    build_png(800, 600, drawing_pixel, HERE / "sample_drawing.png")
    build_png(
        400, 200,
        lambda x, y: 0 if (40 <= x <= 360 and 80 <= y <= 120 and (x // 10) % 2 == 0) else 255,
        HERE / "sample_image.png",
    )
    for name in ("sample.pdf", "sample_table.pdf", "sample_drawing.png", "sample_image.png"):
        print(f"{name}: {(HERE / name).stat().st_size} байт")


if __name__ == "__main__":
    main()
