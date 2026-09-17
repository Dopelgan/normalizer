"""Начальная схема: documents и chunks.

Revision ID: 0001
Revises:
"""

import sqlalchemy as sa
from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "documents",
        sa.Column("id", sa.String(length=64), primary_key=True),
        sa.Column("s3_fileid", sa.String(length=256), nullable=True),
        sa.Column("source_path", sa.String(length=1024), nullable=False),
        sa.Column("doc_version", sa.String(length=50), nullable=True),
        sa.Column("language", sa.String(length=10), nullable=False, server_default="ru"),
        sa.Column("title", sa.String(length=256), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.Column("file_hash", sa.String(length=64), nullable=True),
        sa.Column("status", sa.String(length=20), nullable=False, server_default="pending"),
        sa.Column("extra_data", sa.JSON(), nullable=True),
    )
    op.create_index("ix_documents_s3_fileid", "documents", ["s3_fileid"])
    op.create_index("idx_documents_status", "documents", ["status"])
    op.create_index("idx_documents_file_hash", "documents", ["file_hash"])

    op.create_table(
        "chunks",
        sa.Column("id", sa.String(length=80), primary_key=True),
        sa.Column("doc_id", sa.String(length=64), nullable=False),
        sa.Column("chunk_type", sa.String(length=20), nullable=False),
        sa.Column("content", sa.JSON(), nullable=False),
        sa.Column("position", sa.JSON(), nullable=True),
        sa.Column("section_title", sa.String(length=256), nullable=True),
        sa.Column("confidence", sa.Float(), nullable=False, server_default="0"),
        sa.Column("completeness", sa.Float(), nullable=False, server_default="0"),
        sa.Column("provenance", sa.JSON(), nullable=False),
        sa.Column("graph_nodes", sa.JSON(), nullable=True),
        sa.Column("relations", sa.JSON(), nullable=True),
        sa.Column("order_index", sa.Integer(), nullable=True),
        sa.Column("extra_data", sa.JSON(), nullable=True),
        sa.ForeignKeyConstraint(["doc_id"], ["documents.id"], ondelete="CASCADE"),
    )
    op.create_index("ix_chunks_doc_id", "chunks", ["doc_id"])
    op.create_index("idx_chunks_doc_type", "chunks", ["doc_id", "chunk_type"])
    op.create_index("idx_chunks_doc_order", "chunks", ["doc_id", "order_index"])


def downgrade() -> None:
    op.drop_table("chunks")
    op.drop_table("documents")
