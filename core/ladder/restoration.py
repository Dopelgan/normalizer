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


def restore(data: bytes, pdf: bool = False) -> Tuple[Optional[bytes], List[str]]:
    """Выправленный PNG и список применённых шагов."""
    try:
        from PIL import Image, ImageFilter, ImageOps
    except ImportError:  # pragma: no cover
        logger.warning("Pillow недоступен — восстановление невозможно")
        return None, []

    from core.ladder.context import _load_image, _skew_degrees

    image = _load_image(data, pdf=pdf)
    if image is None:
        return None, []

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

    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue(), applied
