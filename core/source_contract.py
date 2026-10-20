"""Происхождение структурных единиц без потери исходного текста."""

import hashlib
import json
from collections import Counter

from core.models.contract import SourceSpan
from core.processing_contract import PROCESSING_PROFILE_VERSION


def source_unit_ids(blocks, doc_id):
    """Контентные ID; bbox и позиция в общей выдаче не влияют на них."""
    seen = Counter()
    result = {}
    for block in blocks:
        if block.source_unit_id:
            identity = block.source_unit_id
        else:
            table = None if block.table_data is None else {
                key: block.table_data.get(key) for key in ("headers", "rows", "total_row")
            }
            fields = None if block.drawing_fields is None else [
                {key: field.get(key) for key in ("category", "value", "tolerance", "unit", "nature")}
                for field in block.drawing_fields
            ]
            payload = [doc_id, block.type, block.page, block.sheet_name,
                       block.text, table, fields, block.image_ref]
            digest = hashlib.sha256(json.dumps(
                payload, ensure_ascii=False, sort_keys=True, default=str,
            ).encode("utf-8")).hexdigest()[:24]
            identity = f"unit-{digest}"
        seen[identity] += 1
        result[id(block)] = identity if seen[identity] == 1 else f"{identity}-{seen[identity]}"
    return result


def attach_source(fragment, block, source_unit_id, *, exact_text=False):
    fragment.provenance.source_unit_id = source_unit_id
    fragment.provenance.processing_profile_version = PROCESSING_PROFILE_VERSION
    fragment.section_path = list(block.section_path)
    if not fragment.section_path and fragment.section_title:
        fragment.section_path = [fragment.section_title]
    position = fragment.position
    if position:
        position.page_start = position.page_end = block.page
        position.bbox_exact = block.bbox_exact
        if exact_text and block.text is not None:
            position.source_spans = [SourceSpan(
                source_unit_id=source_unit_id, page=block.page, sheet=position.sheet,
                char_start=0, char_end=len(block.text),
            )]
            position.text_offsets_exact = True
    if fragment.content.table_data:
        fragment.content.table_data.table_id = source_unit_id
    for index, table in enumerate((fragment.content.structured_payload or {}).get("tables", [])):
        table.setdefault("table_id", f"{source_unit_id}:table-{index + 1}")
        table.setdefault("row_start", 0)
        table.setdefault("row_end", len(table.get("rows") or []))
    fragment.warnings = list(dict.fromkeys(fragment.warnings + block.warnings))
