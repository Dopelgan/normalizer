"""
Структура листа чертежа `drawing-sheet-v1` (флаг `DRAWING_STRUCTURE`).

`apply` — точка входа из `DrawingProcessor.enrich_fragments`: по каждой
странице с чертёжными фрагментами собирает структуру (`assemble.build_sheet`)
и кладёт её в `content.drawing_sheet` фрагмента «лист целиком» — ровно один
на страницу. Плоские поля, `structured_payload` и текст не меняются;
предупреждения сборщика и учёт листа — в
`completeness_mechanisms["drawing_diagnostics"]["structure"]`, а не в
`warnings` фрагмента: на проверку человеку структура лист не отправляет.

Сбой сборки одной страницы документ не роняет: у страницы просто нет структуры.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from core.models.contract import Fragment

logger = logging.getLogger(__name__)

# Предупреждения сборщика в учёте листа — не больше стольких.
MAX_WARNINGS = 20


def apply(fragments: List[Fragment]) -> List[Fragment]:
    """Структура каждой страницы с чертежами; фрагменты возвращаются те же, в том же порядке."""
    from core.drawing_structure import assemble, candidates

    pages: Dict[int, List[Fragment]] = {}
    for fragment in fragments:
        if fragment.position is None:
            continue
        pages.setdefault(int(fragment.position.page), []).append(fragment)
    for page, group in pages.items():
        drawings = [f for f in group if f.type == "drawing"]
        if not drawings:
            continue
        others = [f for f in group if f.type != "drawing"]
        carrier = candidates.whole_sheet(drawings)
        if carrier is None:
            continue
        try:
            sheet, warnings = assemble.build_sheet(drawings, others)
        except Exception as exc:  # noqa: BLE001 — структура не должна ронять документ
            logger.warning("Структура листа %s не собрана: %s", page, exc)
            _note(carrier, {"built": False, "error": type(exc).__name__})
            continue
        if sheet is not None:
            sheet = _without_nul(sheet)
        carrier.content.drawing_sheet = sheet
        note: Dict[str, Any] = {"built": sheet is not None,
                                "warnings": list(warnings)[:MAX_WARNINGS]}
        if sheet is not None:
            note["stats"] = sheet.stats.model_dump(mode="json")
        _note(carrier, note)
    return fragments


def _note(carrier: Fragment, note: Dict[str, Any]) -> None:
    mechanisms = dict(carrier.completeness_mechanisms or {})
    diagnostics = dict(mechanisms.get("drawing_diagnostics") or {})
    diagnostics["structure"] = note
    mechanisms["drawing_diagnostics"] = diagnostics
    carrier.completeness_mechanisms = mechanisms


def _without_nul(sheet):
    """
    `\\x00` из прочтения модели не уходит в выдачу: PostgreSQL не переводит
    его в jsonb (то же, что `contract_finalize._drop_nul` делает с полями).
    """
    from core.models.drawing_sheet import DrawingSheet, sheet_json

    if "\\u0000" not in sheet_json(sheet):
        return sheet
    cleaned = _strip(sheet.model_dump(mode="json"))
    try:
        return DrawingSheet.model_validate(cleaned)
    except Exception:  # noqa: BLE001 — без структуры лучше, чем с битой
        return None


def _strip(value: Any) -> Optional[Any]:
    if isinstance(value, str):
        return value.replace("\x00", "")
    if isinstance(value, dict):
        return {k: _strip(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_strip(v) for v in value]
    return value
