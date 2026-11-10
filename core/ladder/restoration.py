"""
Восстановление изображений низкого качества (этапы В-1..В-5).

Отдельный этап перед разбором, а не часть его. Порядок важен: сначала
геометрия, потом фотометрия, и только затем — избирательное повышение
разрешения. Обратный порядок увеличивает вместе с картинкой и её дефекты.

Модуль ничего не решает о судьбе документа: он возвращает выправленный
растр и перечень применённого. Стало ли лучше, проверяет вызывающая
сторона повторной оценкой (В-5).
"""

from __future__ import annotations

import io
import logging
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

# Ниже этого числа пикселей по меньшей стороне мелкие обозначения на
# чертеже нечитаемы даже человеком — тогда включается увеличение.
_MIN_SHORT_SIDE = 1400
_MAX_UPSCALE_PIXELS = 40_000_000

# Наклон меньше этого не выправляем: поворот сам по себе размывает растр.
_MIN_SKEW = 0.6

# Сколько секунд срока задачи оставить на разбор уже восстановленных страниц.
_PAGE_RESERVE = 120


def restore(data: bytes, pdf: bool = False) -> Tuple[Optional[bytes], List[str]]:
    """Выправленный PNG (изображение или первая страница PDF) и список шагов."""
    from core.ladder.context import _load_image

    image = _load_image(data, pdf=pdf)
    if image is None:
        return None, []
    image, applied = _restore_image(image)
    return _png(image), applied


def restore_pages(
    data: bytes, max_pages: int, dpi: int = 200
) -> Tuple[Optional[bytes], List[str], Optional[bytes]]:
    """
    Многостраничный PDF целиком: каждая страница выправляется отдельно, и
    собирается PDF той же длины. Раньше восстанавливалась только первая
    страница — и при победе этого уровня остальные страницы скана пропадали.

    Третий элемент — PNG первой страницы для повторной оценки качества (В-5).
    """
    try:
        import fitz
        from PIL import Image
    except ImportError:  # pragma: no cover
        return None, [], None

    try:
        source = fitz.open(stream=data, filetype="pdf")
    except Exception as exc:  # noqa: BLE001
        logger.debug("PDF не открылся для восстановления: %s", exc)
        return None, [], None

    output = fitz.open()
    applied: List[str] = []
    first_png: Optional[bytes] = None
    try:
        from core import deadline

        for index, page in enumerate(source):
            if index >= max_pages:
                logger.warning("Восстановление усечено: %d страниц из %d", max_pages, source.page_count)
                break
            left = deadline.remaining()
            if left is not None and left < _PAGE_RESERVE:
                logger.warning("Восстановление остановлено по сроку задачи на странице %d", index + 1)
                applied.append(f"усечено по времени: {index} из {source.page_count} стр.")
                break
            pixmap = page.get_pixmap(dpi=dpi)
            image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
            restored, steps = _restore_image(image)
            png = _png(restored)
            if first_png is None:
                first_png = png
            for step in steps:
                if step not in applied:
                    applied.append(step)
            target = output.new_page(width=page.rect.width, height=page.rect.height)
            target.insert_image(target.rect, stream=png)
        if not output.page_count:
            return None, [], None
        return output.tobytes(), applied, first_png
    finally:
        output.close()
        source.close()


def _png(image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def _restore_image(image):
    """Этапы В-2..В-4 для одного листа: выправленный растр и список шагов."""
    from PIL import Image, ImageFilter, ImageOps

    from core.ladder.context import _skew_degrees

    applied: List[str] = []
    image = image.convert("L")

    # В-2. Геометрическое выправление.
    try:
        import numpy
        skew = _skew_degrees(numpy.asarray(image, dtype=numpy.float32), numpy)
        if abs(skew) >= _MIN_SKEW:
            image = image.rotate(
                -skew, resample=Image.BICUBIC, expand=True, fillcolor=255
            )
            applied.append(f"наклон {skew:+.1f}°")
    except ImportError:  # pragma: no cover
        logger.debug("numpy недоступен — наклон не выправляем")

    # В-3. Фотометрическое выправление.
    image = ImageOps.autocontrast(image, cutoff=1)
    applied.append("контраст")
    image = image.filter(ImageFilter.MedianFilter(size=3))
    applied.append("шум")

    # В-4. Повышение разрешения — только когда мелкое действительно не читается.
    short_side = min(image.size)
    if short_side < _MIN_SHORT_SIDE:
        factor = min(2.0, _MIN_SHORT_SIDE / max(1, short_side))
        target = (int(image.width * factor), int(image.height * factor))
        if target[0] * target[1] <= _MAX_UPSCALE_PIXELS:
            image = image.resize(target, Image.LANCZOS)
            applied.append(f"увеличение ×{factor:.1f}")
    return image, applied
