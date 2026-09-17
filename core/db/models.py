"""SQLAlchemy-модели хранилища документов и фрагментов."""

from datetime import datetime, timezone

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    Text,
)
from sqlalchemy.ext.mutable import MutableDict, MutableList
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


def utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class DocumentDB(Base):
    __tablename__ = "documents"

    id = Column(String(64), primary_key=True)              # стабильный doc_id
    s3_fileid = Column(String(256), nullable=True, index=True)
    source_path = Column(String(1024), nullable=False)     # URI файла (трассировка)
    doc_version = Column(String(50), nullable=True)
    language = Column(String(10), nullable=False, default="ru")
    title = Column(String(256), nullable=True)
    created_at = Column(DateTime, default=utcnow, nullable=False)
    updated_at = Column(DateTime, default=utcnow, onupdate=utcnow, nullable=False)
    file_hash = Column(String(64), nullable=True)          # SHA-256 содержимого
    status = Column(String(20), nullable=False, default="pending")  # pending|indexed|error
    extra_data = Column(MutableDict.as_mutable(JSON), default=dict)

    chunks = relationship("ChunkDB", back_populates="document", cascade="all, delete-orphan")

    __table_args__ = (
        Index("idx_documents_status", "status"),
        Index("idx_documents_file_hash", "file_hash"),
    )

    def __repr__(self) -> str:
        return f"<DocumentDB id={self.id} status={self.status}>"


class ChunkDB(Base):
    """Фрагмент документа. `id` совпадает с `fragment_id` контракта."""

    __tablename__ = "chunks"

    id = Column(String(80), primary_key=True)
    doc_id = Column(
        String(64), ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )

    chunk_type = Column(String(20), nullable=False)   # text|table|formula|drawing|image|structured
    content = Column(JSON, nullable=False)
    position = Column(JSON, nullable=True)            # {page, sheet, bbox, order}
    section_title = Column(String(256), nullable=True)
    confidence = Column(Float, nullable=False, default=0.0)
    completeness = Column(Float, nullable=False, default=0.0)
    provenance = Column(JSON, nullable=False)         # {method, strategy_level, source}
    graph_nodes = Column(MutableList.as_mutable(JSON), default=list)
    relations = Column(MutableList.as_mutable(JSON), default=list)
    order_index = Column(Integer, nullable=True)
    # Служебные данные, наружу по контракту не отдаются.
    extra_data = Column(MutableDict.as_mutable(JSON), default=dict)

    document = relationship("DocumentDB", back_populates="chunks")

    __table_args__ = (
        Index("idx_chunks_doc_type", "doc_id", "chunk_type"),
        Index("idx_chunks_doc_order", "doc_id", "order_index"),
    )

    def __repr__(self) -> str:
        return f"<ChunkDB id={self.id} doc_id={self.doc_id} type={self.chunk_type}>"


class IntakeDecisionDB(Base):
    """
    Решение по файлу на приёме: Data Gateway или Quality Gate.

    Нужна отдельная таблица, а не поле у документа: отклонённый и
    карантинный файл документом не становится, но и терять его нельзя —
    карантин попадает в очередь администратору, а отчёт по итогам массового
    приёма собирается именно отсюда.
    """

    __tablename__ = "intake_decisions"

    id = Column(String(80), primary_key=True)          # стабильный по s3_fileid и этапу
    s3_fileid = Column(String(256), nullable=False)
    source_path = Column(String(1024), nullable=True)
    stage = Column(String(20), nullable=False)         # data_gateway | quality_gate
    layer = Column(String(10), nullable=True)          # G-1..G-4, QG-1..QG-5
    outcome = Column(String(20), nullable=False)       # accept|quarantine|reject
    category = Column(String(40), nullable=True)
    reason = Column(Text, nullable=False)
    confidence = Column(Float, nullable=False, default=1.0)
    signals = Column(MutableDict.as_mutable(JSON), default=dict)
    file_hash = Column(String(64), nullable=True)

    created_at = Column(DateTime, default=utcnow, nullable=False)
    # Решение администратора по карантину: именно на этих записях система
    # учится корректировать правила отсева.
    resolved_outcome = Column(String(20), nullable=True)
    resolved_by = Column(String(128), nullable=True)
    resolved_at = Column(DateTime, nullable=True)

    # Имена индексов заданы явно и совпадают с миграцией
    # 0002_intake_decisions: иначе create_all() и alembic создают одно и то
    # же под разными именами, и следующий upgrade падает.
    __table_args__ = (
        Index("idx_intake_fileid", "s3_fileid"),
        Index("idx_intake_hash", "file_hash"),
        Index("idx_intake_outcome", "outcome"),
        Index("idx_intake_stage_created", "stage", "created_at"),
    )

    def __repr__(self) -> str:
        return f"<IntakeDecisionDB {self.s3_fileid} {self.stage}={self.outcome}>"
