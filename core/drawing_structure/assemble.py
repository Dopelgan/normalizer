"""
Сборщик структуры листа `drawing-sheet-v1` (этапы S0–S7 плана).

Детерминированный: только прочтения, которые проходы ветки чертежей уже
дали странице, — ни одного вызова модели. Порядок этапов:

S0–S1 `candidates.collect` — пул прочтений в координатах страницы;
штамп   `title.build`      — графы по ключу, значение сводной записи, рамка клетки;
S2      `merge.merge`      — одна надпись = один элемент, прочие прочтения в `alternatives`;
S3      `snap.snap`        — рамка по строке текстового слоя/распознавания;
S4      `typing.classify`  — тип надписи и разбор величины;
S5–S6   `links.build`      — виды, разнесение, связи, `id` и ссылки;
S7      `validate.finish`  — проверки, учёт, схема и предел размера.

Точка входа для интеграции (карточка N6) — `build_sheet`; флаг и место
вызова в `drawing_processor` здесь не задаются.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from core.drawing_structure import candidates as candidates_
from core.drawing_structure import links, merge, snap, title, validate
from core.drawing_structure import typing as typing_
from core.models.contract import Fragment
from core.models.drawing_sheet import DrawingSheet


def build_sheet(drawings: Sequence[Fragment], others: Sequence[Fragment] = ()
                ) -> Tuple[Optional[DrawingSheet], List[str]]:
    """
    Структура одной страницы. `drawings` — чертёжные фрагменты страницы в
    порядке ответа (по нему считаются `field_refs`), `others` — прочие
    фрагменты той же страницы (строки распознавания). Возвращает структуру
    (или `None`, если собирать не из чего или она не прошла проверку схемы)
    и предупреждения `DRAWING_STRUCTURE_*`.
    """
    pool = candidates_.collect(drawings, others)
    if not pool.candidates and not pool.transcript and not pool.tables:
        return None, []

    in_title = [c for c in pool.candidates if title.in_title(c, pool)]
    rest = [c for c in pool.candidates if not title.in_title(c, pool)]
    title_block, leftovers = title.build(in_title, pool)
    elements = merge.merge(rest + leftovers)
    snap.snap(elements, pool.lines)
    typed = [(element, typing_.classify(element.value, element.category, pool.sheet_type)) for element in elements]

    sections = links.build(pool, typed, title_block)
    links.identify(sections, title_block)
    warnings = links.resolve(sections)

    frame = dict(pool.frame)
    page_px = frame.get("page_px")
    frame["page_px"] = [int(v) for v in page_px] if isinstance(page_px, (list, tuple)) and len(page_px) == 2 else None
    fmt = (title_block.get("fields") or {}).get("format")
    if fmt is not None:
        frame["format"] = fmt.get("value")
        frame["format_confidence"] = fmt.get("confidence", 0.0)
    merged = sum(e.merged for e in elements) + sum(
        max(0, len(t.get("field_refs") or []) - 1) for t in (title_block.get("fields") or {}).values())
    return validate.finish(pool.page, pool.sheet_type, frame, title_block, sections, merged, warnings)
