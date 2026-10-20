"""Признаки качества для отбора RAG, без повышения оценок распознавания."""

from core.processing_contract import DRAWING_MIN_COMPLETENESS, DRAWING_MIN_FACT_CONFIDENCE


def apply_drawing_quality(fragment):
    if not fragment.completeness_mechanisms:
        fragment.completeness = None
    fields = fragment.content.structured_drawing_fields or []
    tables = (fragment.content.structured_payload or {}).get("tables", [])
    suspicious = False
    for field in fields:
        synthetic = "(dev)" in field.value.lower()
        field.confirmed = field.confidence >= DRAWING_MIN_FACT_CONFIDENCE and not synthetic
        suspicious = suspicious or synthetic
    for table in tables:
        # Для чтения таблицы модель пока не отдаёт калиброванной оценки.
        # Отсутствующую оценку нельзя заменить уверенностью соседнего поля.
        confidence = table.get("confidence")
        table["confirmed"] = bool(
            isinstance(confidence, (int, float))
            and DRAWING_MIN_FACT_CONFIDENCE <= confidence <= 1
        )
    if suspicious:
        fragment.warnings = list(dict.fromkeys(fragment.warnings + ["SYNTHETIC_DRAWING_VALUE"]))
    fragment.review_required = bool(
        fragment.review_required or fragment.missing_regions or fragment.warnings
        or fragment.completeness is None
        or fragment.completeness < DRAWING_MIN_COMPLETENESS
        or any(not field.confirmed for field in fields)
        or any(not table["confirmed"] for table in tables)
    )
    if fragment.completeness is None:
        fragment.warnings = list(dict.fromkeys(fragment.warnings + ["DRAWING_COMPLETENESS_UNKNOWN"]))


def document_quality(fragments):
    """Один результат используется при первой выдаче и сохраняется в БД."""
    drawings = [fragment for fragment in fragments if fragment.type == "drawing"]
    warnings = list(dict.fromkeys(w for fragment in fragments for w in fragment.warnings))
    return bool(any(fragment.review_required for fragment in drawings)
                or any("DRAWING_NOT_RECOGNIZED" in f.warnings for f in fragments)), warnings
