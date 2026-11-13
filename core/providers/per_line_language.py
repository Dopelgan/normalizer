"""
Язык выбирается по строке, а не по документу (`OCR_PER_LINE_LANGUAGE`).

Зачем. Tesseract с совмещённой моделью «rus+eng» на одной странице решает
за всю страницу сразу и потому ошибается в обе стороны: русскую строку
чертёжного шрифта отдаёт латиницей («Mapxupobame», «Konupoban»), а на
английском листе — наоборот. А в корпусе есть и полностью английские
документы, и двуязычные листы, где «Устройство коммутационное с компенсацией
холодных спаев» стоит рядом с «Switching device with cold junction
compensation», и экспортная маркировка «For NPP». Запрет на латиницу всю эту
бумагу сломал бы.

Что делается. Страница читается каждой языковой моделью по отдельности, и
для каждой строки остаётся то чтение, в котором распознавание уверено
больше. Уверенность по строке Tesseract отдаёт сам (`OcrLine.confidence`,
собирается из уверенности по словам в `tesseract_fallback._read_lines`) —
здесь она только сравнивается, ничего нового не считается.

Строки сопоставляются по вертикальному перекрытию рамок: у одной и той же
строки листа рамки у разных моделей совпадают почти точно, а слова и
уверенность расходятся.

Осторожность. Рамкой служит чтение первой модели: её набор строк и остаётся
набором строк страницы. Чужое чтение может заменить текст строки, но не
может добавить строку, которой в первом чтении нет, и не может её убрать:
иначе страница получала бы то лишние строки из перекрытий, то потерянные.
Замена делается только при заметно большей уверенности (`MARGIN`): на
равных побеждает первая модель, и выбор не дёргается от шума.
"""

from __future__ import annotations

from typing import Any, List, Sequence

# Насколько чужое чтение должно быть увереннее, чтобы заменить строку.
MARGIN = 0.02
# Рамки считаются одной строкой листа с такой долей вертикального перекрытия.
_MIN_OVERLAP = 0.5


def _overlap(first: Sequence[float], second: Sequence[float]) -> float:
    """Доля вертикального перекрытия двух рамок от меньшей высоты."""
    top, bottom = max(first[1], second[1]), min(first[3], second[3])
    if bottom <= top:
        return 0.0
    height = min(first[3] - first[1], second[3] - second[1])
    return (bottom - top) / height if height > 0 else 0.0


def _best_match(line: Any, candidates: Sequence[Any]) -> Any:
    """Строка другого чтения, стоящая на листе на том же месте."""
    best, best_overlap = None, _MIN_OVERLAP
    for candidate in candidates:
        share = _overlap(line.bbox, candidate.bbox)
        if share > best_overlap:
            best, best_overlap = candidate, share
    return best


def choose(readings: Sequence[Sequence[Any]]) -> List[Any]:
    """
    Строки страницы, у каждой — чтение самой уверенной языковой модели.

    На входе чтения одной и той же страницы разными моделями, первое —
    основное. На выходе строки основного чтения, у части из них текст и
    уверенность взяты у другой модели.
    """
    primary = list(readings[0]) if readings else []
    others = [list(reading) for reading in readings[1:] if reading]
    if not primary or not others:
        return primary
    result: List[Any] = []
    for line in primary:
        winner = line
        for reading in others:
            candidate = _best_match(line, reading)
            if candidate is None or not candidate.text.strip():
                continue
            if candidate.confidence > winner.confidence + MARGIN:
                winner = candidate
        if winner is line:
            result.append(line)
            continue
        # Место строки на листе — из основного чтения: рамки и угол письма
        # дальше по конвейеру сшивают колонки и абзацы, и они должны быть
        # из одного набора.
        result.append(type(line)(
            text=winner.text, bbox=list(line.bbox),
            confidence=winner.confidence, rotation=line.rotation,
            parts=list(winner.parts),
        ))
    return result
