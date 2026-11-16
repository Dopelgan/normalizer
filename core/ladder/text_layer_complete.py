"""
Полнота текстового слоя PDF: можно ли не запускать уровень 4 (Tesseract).

На векторном PDF текст уровня 5 берётся из слоя файла, а не из распознавания,
и полный проход Tesseract тратил около 2 с на страницу ради результата,
который выбрасывался (пример с прода 16.11: 80 страниц, 160 с). Проход нужен,
только если слой неполон: тогда распознавание — единственная мерка того,
сколько текста на странице есть.

Слой считается полным, когда текст есть на каждой странице файла (страница
со сканом и одной наложенной надписью — не текст) и медиана символов на
страницу не меньше `MIN_MEDIAN_CHARS`. Одна страница без текста — и уровень 4
работает как прежде (флаг `LADDER_SKIP_OCR_ON_TEXT_LAYER`).
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Optional

from core.ladder.context import SOURCE_VECTOR_TEXT, DocumentContext

MIN_MEDIAN_CHARS = 200


@dataclass(frozen=True)
class CompleteLayer:
    pages: int
    chars: int


def complete_text_layer(context: DocumentContext) -> Optional[CompleteLayer]:
    """Полный слой — его размер; иначе None (уровень 4 нужен)."""
    if context.classification.source != SOURCE_VECTOR_TEXT:
        return None
    layer = context.text_layer
    if layer is None or not layer.page_count:
        return None
    # Число страниц самого файла, а не слоя: страница, которой в слое нет
    # вовсе, — тоже страница без текста.
    pages = context.source_page_count or layer.page_count
    if pages != layer.page_count:
        return None
    counts = []
    for page in range(1, pages + 1):
        chars = layer.char_count(page)
        if chars <= 0 or layer.overlay_only(page):
            return None
        counts.append(chars)
    if statistics.median(counts) < MIN_MEDIAN_CHARS:
        return None
    return CompleteLayer(pages=pages, chars=sum(counts))
