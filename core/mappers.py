"""Преобразование между фрагментами контракта и строками БД."""

from typing import Any, Dict, Optional

from core.db.models import ChunkDB, DocumentDB
from core.models.contract import (
    Content,
    DocumentMetadata,
    ExtractedFact,
    Fragment,
    Position,
    Provenance,
)


def fragment_to_row(
    fragment: Fragment, doc_id: str, extra: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Фрагмент -> словарь для ChunkRepository.create_chunk."""
    position = fragment.position.model_dump() if fragment.position else None
    return {
        "id": fragment.fragment_id,
        "doc_id": doc_id,
        "chunk_type": fragment.type,
        "content": fragment.content.model_dump(),
        "position": position,
        "section_title": fragment.section_title,
        "confidence": float(fragment.confidence),
        # Колонка NOT NULL сохранена для существующих БД. Неизвестность
        # хранится явно в JSON, а при чтении восстанавливается как null.
        "completeness": float(fragment.completeness or 0.0),
        "provenance": fragment.provenance.model_dump(),
        "extracted_facts": [f.model_dump(mode="json") for f in fragment.extracted_facts],
        "graph_nodes": [n.model_dump() for n in fragment.graph_nodes],
        "relations": [r.model_dump() for r in fragment.relations],
        "order_index": (position or {}).get("order"),
        "extra_data": {
            **dict(extra or {}),
            "contract_quality": {
                "completeness_unknown": fragment.completeness is None,
                **fragment.model_dump(include={
                    "section_path", "review_required", "warnings", "missing_regions",
                    "completeness_mechanisms", "structure_score",
                }),
            },
        },
    }


def row_to_fragment(row: ChunkDB) -> Fragment:
    """Строка БД -> фрагмент контракта (для повторной отдачи без переобработки)."""
    quality = dict((row.extra_data or {}).get("contract_quality") or {})
    unknown = quality.pop("completeness_unknown", False)
    # Старые drawing-строки использовали structure_ratio вместо полноты.
    if row.chunk_type == "drawing" and not quality.get("completeness_mechanisms"):
        unknown = True
        quality["review_required"] = True
    return Fragment(
        fragment_id=row.id,
        type=row.chunk_type,
        content=Content(**(row.content or {})),
        position=Position(**row.position) if row.position else None,
        section_title=row.section_title,
        confidence=float(row.confidence or 0.0),
        completeness=None if unknown else float(row.completeness or 0.0),
        provenance=Provenance(**(row.provenance or {
            "method": "unknown", "strategy_level": 1, "source": "unknown",
        })),
        extracted_facts=[
            ExtractedFact(**fact) for fact in (row.extracted_facts or [])
        ],
        graph_nodes=row.graph_nodes or [],
        relations=row.relations or [],
        **quality,
    )


def document_to_metadata(document: DocumentDB) -> DocumentMetadata:
    """Строка documents -> document_metadata контракта."""
    return DocumentMetadata(
        doc_id=document.id,
        doc_version=document.doc_version,
        language=document.language or "ru",
        source_path=document.source_path or "",
        updated_at=document.updated_at or document.created_at,
        status=document.status if document.status in ("pending", "indexed", "error") else "pending",
    )
