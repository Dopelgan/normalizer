"""Решения приёма: Data Gateway и Quality Gate.

Отдельная таблица, а не поле у документа: отклонённый и карантинный файл
документом не становится, но и теряться не должен — карантин попадает в
очередь администратору, а отчёт по итогам массового приёма собирается
отсюда же.

Revision ID: 0002
Revises: 0001
"""

import sqlalchemy as sa
from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "intake_decisions",
        sa.Column("id", sa.String(length=80), primary_key=True),
        sa.Column("s3_fileid", sa.String(length=256), nullable=False),
        sa.Column("source_path", sa.String(length=1024), nullable=True),
        sa.Column("stage", sa.String(length=20), nullable=False),
        sa.Column("layer", sa.String(length=10), nullable=True),
        sa.Column("outcome", sa.String(length=20), nullable=False),
        sa.Column("category", sa.String(length=40), nullable=True),
        sa.Column("reason", sa.Text(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False, server_default="1.0"),
        sa.Column("signals", sa.JSON(), nullable=True),
        sa.Column("file_hash", sa.String(length=64), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("resolved_outcome", sa.String(length=20), nullable=True),
        sa.Column("resolved_by", sa.String(length=128), nullable=True),
        sa.Column("resolved_at", sa.DateTime(), nullable=True),
    )
    op.create_index("idx_intake_fileid", "intake_decisions", ["s3_fileid"])
    op.create_index("idx_intake_hash", "intake_decisions", ["file_hash"])
    op.create_index("idx_intake_outcome", "intake_decisions", ["outcome"])
    op.create_index("idx_intake_stage_created", "intake_decisions", ["stage", "created_at"])


def downgrade() -> None:
    op.drop_index("idx_intake_stage_created", table_name="intake_decisions")
    op.drop_index("idx_intake_outcome", table_name="intake_decisions")
    op.drop_index("idx_intake_hash", table_name="intake_decisions")
    op.drop_index("idx_intake_fileid", table_name="intake_decisions")
    op.drop_table("intake_decisions")
