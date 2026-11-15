"""Признаки качества для отбора RAG, без повышения оценок распознавания."""

from core.config import settings
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


# Пометки реальной потери листа: с ними документ partial при любом флаге.
_LOSS_WARNINGS = ("DRAWING_NOT_RECOGNIZED", "DRAWING_SKIPPED_NO_TIME", "DRAWING_SHEETS_SKIPPED")


def _drawing_lost(fragment, transcribed_pages=frozenset()):
    # Область, которую модель не прочитала отдельно, не потеряна, если весь
    # текст её листа переписан проходом полного чтения (DRAWING_IMAGE_FULL_READ,
    # DRAWING_PDF_FULL_READ).
    # Недочитанная часть самого переписанного текста (transcript_tile)
    # потеря всегда: её не покрывает другой фрагмент той же страницы.
    unread = [region for region in fragment.missing_regions
              if region.get("kind") == "transcript_tile"
              or region.get("page") not in transcribed_pages]
    return bool(unread) or any(
        warning.startswith(_LOSS_WARNINGS) for warning in fragment.warnings
    )


def _transcribed_pages(drawings):
    """Страницы, весь текст которых переписан моделью."""
    if not (settings.DRAWING_IMAGE_FULL_READ or settings.DRAWING_PDF_FULL_READ):
        return frozenset()
    return frozenset(
        fragment.position.page for fragment in drawings
        if fragment.position is not None
        and (fragment.content.structured_payload or {}).get("transcript_line_count")
        # Лист, часть которого не дочитана (DRAWING_TRANSCRIBE_TILES), не
        # переписан целиком: его непрочитанные области остаются потерей.
        and (fragment.content.structured_payload or {}).get("transcript_complete") is not False
        and (fragment.content.text or "").strip()
    )


def _from_rescan(fragment):
    """
    Лист перечитывания плохого скана (DRAWING_RESCAN_POOR_OCR_PAGES). Его
    страница уже покрыта распознаванием, и лист только добавляет текст
    модели: его недочитанная часть или неизмеренная полнота — повод
    проверить сам лист, но не потеря страницы документа.

    Признак — `rescan_reason`. С DRAWING_DIAGNOSTICS_NOT_INDEXED
    `finalize_fragments` (до `document_quality`) уносит его из payload в
    `completeness_mechanisms["drawing_diagnostics"]`, а идентификатор
    фрагмента меняет, поэтому смотрятся оба места. В payload для
    потребителя признак не добавляется: без флага он ушёл бы в
    индексируемый текст строкой «Метка: значение».
    """
    if "rescan_reason" in (fragment.content.structured_payload or {}):
        return True
    # Ключ — contract_finalize.DIAGNOSTICS_KEY; импорт оттуда был бы
    # циклическим (contract_finalize импортирует этот модуль).
    diagnostics = (fragment.completeness_mechanisms or {}).get("drawing_diagnostics")
    return isinstance(diagnostics, dict) and "rescan_reason" in diagnostics


def document_quality(fragments):
    """Один результат используется при первой выдаче и сохраняется в БД."""
    drawings = [fragment for fragment in fragments if fragment.type == "drawing"]
    warnings = list(dict.fromkeys(w for fragment in fragments for w in fragment.warnings))
    # Листы перечитывания документ частичным не делают (см. _from_rescan).
    judged = [fragment for fragment in drawings if not _from_rescan(fragment)]
    if settings.DRAWING_UNMEASURED_NOT_PARTIAL:
        # Неизмеренная полнота чтения — повод проверить фрагмент, но не
        # потеря страницы: документ partial только при реальной потере.
        covered = _transcribed_pages(drawings)
        drawing_partial = any(_drawing_lost(fragment, covered) for fragment in judged)
    else:
        drawing_partial = any(fragment.review_required for fragment in judged)
    return bool(drawing_partial
                or any("DRAWING_NOT_RECOGNIZED" in f.warnings for f in fragments)), warnings
