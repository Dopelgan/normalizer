"""Преобразование между фрагментами контракта и строками БД."""

from typing import Any, Dict, Optional

from core.db.models import ChunkDB, DocumentDB
from core.models.contract import Content, DocumentMetadata, Fragment, Position, Provenance


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
        "completeness": float(fragment.completeness),
        "provenance": fragment.provenance.model_dump(),
        "graph_nodes": [n.model_dump() for n in fragment.graph_nodes],
        "relations": [r.model_dump() for r in fragment.relations],
        "order_index": (position or {}).get("order"),
        "extra_data": dict(extra or {}),
    }


def row_to_fragment(row: ChunkDB) -> Fragment:
    """Строка БД -> фрагмент контракта (для повторной отдачи без переобработки)."""
    return Fragment(
        fragment_id=row.id,
        type=row.chunk_type,
        content=Content(**(row.content or {})),
        position=Position(**row.position) if row.position else None,
        section_title=row.section_title,
        confidence=float(row.confidence or 0.0),
        completeness=float(row.completeness or 0.0),
        provenance=Provenance(**(row.provenance or {
            "method": "unknown", "strategy_level": 1, "source": "unknown",
        })),
        graph_nodes=row.graph_nodes or [],
        relations=row.relations or [],
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
