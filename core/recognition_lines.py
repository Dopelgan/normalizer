"""
Строки распознавания с координатами (флаг `RECOGNITION_LINES`).

Блок разбора несёт одну рамку — объединение своих строк. Для просмотра в
админке этого мало: чтобы показать каждую распознанную строку на её месте,
нужны рамки самих строк, размер страницы и поворот текста на ней. Всё это
у источников есть (текстовый слой, MinerU, Tesseract), но до сих пор
выбрасывалось при склейке строк в блок.

Модуль только собирает эти данные в `ParsedBlock.lines`,
`ParsedBlock.page_size` и `ParsedBlock.page_rotation`; в контракт
(`position.lines`, `position.page_size`, `position.page_rotation`) их
переносит `core/source_contract.attach_source`. При выключенном флаге
функции возвращают пустой словарь и ничего не меняют — выдача байт в байт
прежняя.
"""

from typing import Any, Dict, Iterable, List, Optional, Sequence

from core.config import settings

#: Источник строки: векторный текстовый слой PDF.
SOURCE_TEXT_LAYER = "text_layer"
#: Слой, собранный построчным распознаванием растра под MinerU.
SOURCE_OCR_LAYER = "ocr_layer"
#: Строки из `middle_json` MinerU.
SOURCE_MINERU = "mineru"
#: Строки Tesseract из фоллбэка.
SOURCE_TESSERACT = "tesseract"

# Наклон строки меньше этого считается горизонтальным и не пишется.
_ROTATION_EPS = 5.0
# Знаков после запятой у координат: 1e-4 страницы — доли миллиметра на A4,
# точнее для подсветки не нужно, а выдача короче.
_DIGITS = 4


def enabled() -> bool:
    # Прямое чтение атрибута: по нему тест кэша лестницы находит флаг и требует его в ключе.
    return bool(settings.RECOGNITION_LINES)


def _bbox(raw: Sequence[float]) -> Optional[List[float]]:
    if not raw or len(raw) != 4:
        return None
    try:
        coords = [max(0.0, min(1.0, float(c))) for c in raw]
    except (TypeError, ValueError):
        return None
    x1, y1, x2, y2 = coords
    return [round(min(x1, x2), _DIGITS), round(min(y1, y2), _DIGITS),
            round(max(x1, x2), _DIGITS), round(max(y1, y2), _DIGITS)]


def record(
    text: str,
    bbox: Sequence[float],
    source: str,
    confidence: Optional[float] = None,
    rotation: float = 0.0,
    page: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """Одна строка в форме контракта; None, если строки или рамки нет."""
    text = (text or "").strip()
    box = _bbox(bbox)
    if not text or box is None:
        return None
    item: Dict[str, Any] = {"text": text, "bbox": box, "source": source}
    if confidence is not None:
        item["confidence"] = round(max(0.0, min(1.0, float(confidence))), 3)
    if abs(float(rotation or 0.0)) > _ROTATION_EPS:
        item["rotation"] = round(float(rotation), 1)
    if page is not None:
        item["page"] = int(page)
    return item


def from_text_lines(
    lines: Iterable[Any], source: str, exact: bool = False,
) -> List[Dict[str, Any]]:
    """
    Строки `TextLine` (слой PDF или строки Tesseract). У векторного слоя
    уверенность — единица по построению, её не пишем (`exact=True`).
    Куски строки (`parts`) не раскрываются: строка показывается целиком.
    """
    result = []
    for line in lines:
        item = record(
            line.text, line.bbox, source,
            confidence=None if exact else getattr(line, "confidence", None),
            rotation=getattr(line, "rotation", 0.0),
        )
        if item is not None:
            result.append(item)
    return result


def fields(
    lines: Iterable[Any],
    source: str,
    exact: bool = False,
    page_size: Optional[Sequence[float]] = None,
    page_rotation: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Поля `ParsedBlock` для строк блока: `lines`, `page_size`,
    `page_rotation`. Пусто, если флаг выключен, — тогда блок строится
    ровно как раньше.
    """
    if not enabled():
        return {}
    out: Dict[str, Any] = {"lines": from_text_lines(lines, source, exact=exact)}
    out.update(page_fields(page_size, page_rotation))
    return out


def page_fields(
    page_size: Optional[Sequence[float]], page_rotation: Optional[int],
) -> Dict[str, Any]:
    """Размер страницы и поворот текста на ней; пусто при выключенном флаге."""
    if not enabled():
        return {}
    out: Dict[str, Any] = {}
    if page_size and len(page_size) == 2 and page_size[0] and page_size[1]:
        out["page_size"] = [round(float(page_size[0]), 2), round(float(page_size[1]), 2)]
    if page_rotation is not None:
        out["page_rotation"] = int(page_rotation) % 360
    return out


def apply(
    block: Any, lines: Iterable[Any], source: str, exact: bool = False,
    page_rotation: Optional[int] = None,
) -> None:
    """
    Заменяет строки уже построенного блока — когда его текст пересобран из
    других строк (слой вместо распознавания MinerU).
    """
    if not enabled():
        return
    block.lines = from_text_lines(lines, source, exact=exact)
    if page_rotation is not None:
        block.page_rotation = int(page_rotation) % 360


def mineru_lines(raw: Dict[str, Any], width: float, height: float) -> List[Dict[str, Any]]:
    """
    Строки блока `middle_json` MinerU: `lines[].bbox` и текст из
    `spans[].content`. Уверенность — средняя `score` спанов, если она есть.
    Вложенные блоки (`blocks[].lines[]`) обходятся тоже.
    """
    result: List[Dict[str, Any]] = []
    if width <= 0 or height <= 0:
        return result

    def norm(box: Any) -> Optional[List[float]]:
        coords = [float(c) for c in (box or []) if isinstance(c, (int, float))]
        if len(coords) == 8:
            xs, ys = coords[0::2], coords[1::2]
            coords = [min(xs), min(ys), max(xs), max(ys)]
        if len(coords) != 4:
            return None
        return [coords[0] / width, coords[1] / height, coords[2] / width, coords[3] / height]

    def walk(node: Dict[str, Any]) -> None:
        for line in node.get("lines") or []:
            if not isinstance(line, dict):
                continue
            spans = [s for s in line.get("spans") or [] if isinstance(s, dict)]
            text = " ".join(
                str(s.get("content") or s.get("text") or "") for s in spans
            )
            text = " ".join(text.split())
            box = norm(line.get("bbox") or line.get("poly"))
            scores = [float(s["score"]) for s in spans
                      if isinstance(s.get("score"), (int, float))]
            item = record(text, box or [], SOURCE_MINERU,
                          confidence=sum(scores) / len(scores) if scores else None)
            if item is not None:
                result.append(item)
        for child in node.get("blocks") or []:
            if isinstance(child, dict):
                walk(child)

    walk(raw)
    return result
