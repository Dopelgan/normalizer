"""
Оценка результата уровня лестницы.

Правило подъёма из архитектуры: если стратегия дала уверенность ниже порога
либо структурную неполноту, документ передаётся на следующий уровень.
Здесь обе половины этого правила выражены числом, чтобы уровни можно было
сравнивать между собой и выбирать лучший, а не первый непустой.

Оценка намеренно наказывает структурную неполноту отдельно от уверенности:
текстовый слой даёт идеально точный текст (уверенность 1.0), но про таблицы
и чертежи не знает ничего, и без этой поправки лестница останавливалась бы
на нём для любого документа.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from core.ladder.base import has_structure, text_chars
from core.ladder.context import (
    GENRE_DRAWING,
    GENRE_MIXED,
    GENRE_SPREADSHEET,
    Classification,
    DocumentContext,
)
from core.models.parse_result import ParseResult

logger = logging.getLogger(__name__)

# Сколько символов на странице считается «текст на месте», когда эталона нет.
# Величина заведомо грубая: она нужна только там, где сравнивать не с чем.
# У векторного PDF эталон — текстовый слой, у растра — слой построчного
# распознавания, и в обоих случаях полнота измеряется, а не угадывается.
_EXPECTED_CHARS_PER_PAGE = 150

# Во сколько раз падает оценка, если структура ожидалась, но не получена.
_STRUCTURE_PENALTY = 0.55

# Жанры, в которых плоского текста заведомо недостаточно.
_STRUCTURED_GENRES = (GENRE_SPREADSHEET, GENRE_DRAWING, GENRE_MIXED)


@dataclass
class Score:
    """Разложенная оценка — в лог и в сырой результат, чтобы выбор был виден."""

    value: float
    confidence: float
    completeness: float
    structure_ok: bool
    reason: str = ""

    def as_dict(self) -> dict:
        return {
            "score": self.value,
            "confidence": self.confidence,
            "completeness": self.completeness,
            "structure_ok": self.structure_ok,
            "reason": self.reason,
        }


def score_result(
    result: ParseResult,
    context: Optional[DocumentContext] = None,
    exhaustive: bool = False,
    expect_structure: bool = False,
) -> Score:
    """
    Единая оценка результата уровня в [0, 1].

    `exhaustive` означает, что формат отдал всё своё содержимое и потерять
    его стратегия не могла — тогда полнота равна единице по построению, а не
    оценивается по объёму текста. Без этого идеально разобранная книга XLSX
    выглядела бы неполной только потому, что в ней мало символов.

    `expect_structure` ставит маршрутизатор, когда структуру на этом
    документе уже кто-то вернул. Жанр для растра всегда `text` или
    `unknown`, в список структурных он не входит, и проверка структурной
    полноты для сканов и картинок не срабатывала вовсе — а именно там
    плоское распознавание и подменяет таблицу с формулами сплошной кашей.
    """
    if not result.blocks:
        return Score(0.0, 0.0, 0.0, False, "уровень не дал ни одного блока")

    produced = text_chars(result)
    if produced == 0 and not has_structure(result):
        return Score(0.0, 0.0, 0.0, False, "уровень не дал содержимого")

    confidence = _weighted_confidence(result)
    completeness = 1.0 if exhaustive else _completeness(result, produced, context)
    classification = context.classification if context is not None else None
    structure_ok = _structure_ok(result, classification)
    reason = ""
    if structure_ok and expect_structure and not has_structure(result):
        structure_ok = False
        reason = "структуру вернул другой уровень, здесь её нет"
    elif not structure_ok:
        reason = (
            f"ожидалась структура (жанр {classification.genre}), "
            "получен плоский текст"
        )

    value = 0.5 * confidence + 0.5 * completeness
    if not structure_ok:
        value *= _STRUCTURE_PENALTY

    return Score(
        value=round(min(1.0, max(0.0, value)), 3),
        confidence=round(confidence, 3),
        completeness=round(completeness, 3),
        structure_ok=structure_ok,
        reason=reason,
    )


def _weighted_confidence(result: ParseResult) -> float:
    """Средняя уверенность, взвешенная по объёму текста блока."""
    weights = [max(1, len((b.text or "")) + _table_weight(b)) for b in result.blocks]
    total = sum(weights)
    if not total:
        return 0.0
    return sum(b.confidence * w for b, w in zip(result.blocks, weights)) / total


def _table_weight(block) -> int:
    if not block.table_data:
        return 0
    rows = block.table_data.get("rows") or []
    return sum(len(str(c)) for row in rows for c in row)


def _completeness(result: ParseResult, produced: int, context) -> float:
    """
    Доля содержимого, доехавшая до результата. Если у документа есть
    текстовый слой, это прямое измерение; иначе — оценка по объёму текста
    на страницу, которая честно остаётся оценкой.
    """
    expected = sum((result.text_layer_chars or {}).values())
    if not expected and context is not None:
        layer = context.text_layer
        if layer is not None:
            expected = layer.char_count()

    if expected:
        return min(1.0, produced / expected)

    pages = max(1, result.page_count or len({b.page for b in result.blocks}))
    return min(1.0, produced / (_EXPECTED_CHARS_PER_PAGE * pages))


def _structure_ok(result: ParseResult, classification: Optional[Classification]) -> bool:
    """В структурных жанрах плоский текст — это структурная неполнота."""
    if classification is None:
        return True
    if classification.genre not in _STRUCTURED_GENRES:
        return True
    return has_structure(result)
