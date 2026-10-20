"""Расширение s3_fileid до 2048 символов без потери данных.

Обычный B-tree индекс PostgreSQL имеет предел размера индексной записи.
Для длинного ``s3_fileid`` нужен hash-индекс: приложение ищет значение
только по точному равенству, для которого такой индекс подходит.

``source_path`` также расширяется, потому что содержит ``SOURCE_PREFIX``
вместе с ``s3_fileid`` и поэтому может быть длиннее самого идентификатора.

Revision ID: 0004
Revises: 0003
"""

import sqlalchemy as sa
from alembic import op

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_index("ix_documents_s3_fileid", table_name="documents")
    op.drop_index("idx_intake_fileid", table_name="intake_decisions")

    op.alter_column(
        "documents",
        "s3_fileid",
        existing_type=sa.String(length=256),
        type_=sa.String(length=2048),
        existing_nullable=True,
    )
    op.alter_column(
        "documents",
        "source_path",
        existing_type=sa.String(length=1024),
        type_=sa.String(length=4096),
        existing_nullable=False,
    )
    op.alter_column(
        "intake_decisions",
        "s3_fileid",
        existing_type=sa.String(length=256),
        type_=sa.String(length=2048),
        existing_nullable=False,
    )
    op.alter_column(
        "intake_decisions",
        "source_path",
        existing_type=sa.String(length=1024),
        type_=sa.String(length=4096),
        existing_nullable=True,
    )

    op.create_index(
        "ix_documents_s3_fileid",
        "documents",
        ["s3_fileid"],
        postgresql_using="hash",
    )
    op.create_index(
        "idx_intake_fileid",
        "intake_decisions",
        ["s3_fileid"],
        postgresql_using="hash",
    )


def downgrade() -> None:
    # Сужение намеренно не обрезает данные: downgrade завершится ошибкой,
    # если в базе уже есть значения длиннее старых лимитов.
    op.drop_index("ix_documents_s3_fileid", table_name="documents")
    op.drop_index("idx_intake_fileid", table_name="intake_decisions")

    op.alter_column(
        "documents",
        "s3_fileid",
        existing_type=sa.String(length=2048),
        type_=sa.String(length=256),
        existing_nullable=True,
    )
    op.alter_column(
        "documents",
        "source_path",
        existing_type=sa.String(length=4096),
        type_=sa.String(length=1024),
        existing_nullable=False,
    )
    op.alter_column(
        "intake_decisions",
        "s3_fileid",
        existing_type=sa.String(length=2048),
        type_=sa.String(length=256),
        existing_nullable=False,
    )
    op.alter_column(
        "intake_decisions",
        "source_path",
        existing_type=sa.String(length=4096),
        type_=sa.String(length=1024),
        existing_nullable=True,
    )

    op.create_index(
        "ix_documents_s3_fileid", "documents", ["s3_fileid"]
    )
    op.create_index(
        "idx_intake_fileid", "intake_decisions", ["s3_fileid"]
    )
