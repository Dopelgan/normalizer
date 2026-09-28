"""Изменения контракта Parser: извлечённые факты у фрагмента.

Факты лежат рядом с фрагментом, а не в служебном `extra_data`: это поле
контракта, и повторная выдача уже разобранного документа обязана отдавать
его так же, как выдача сразу после разбора.

Остальные поля изменённого контракта (`status`, `content_hash`,
`normalized_content_hash`, `mime_type`, `pages`, `processed_pages`,
`failed_pages`, `warnings`) хранятся в `documents.extra_data` и схемы не
меняют.

Revision ID: 0003
Revises: 0002
"""

import sqlalchemy as sa
from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "chunks",
        sa.Column("extracted_facts", sa.JSON(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("chunks", "extracted_facts")
