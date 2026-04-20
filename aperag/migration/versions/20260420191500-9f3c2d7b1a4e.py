"""add celery processing lease fields

Revision ID: 9f3c2d7b1a4e
Revises: a1b2c3d4e5f6
Create Date: 2026-04-20 19:15:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "9f3c2d7b1a4e"
down_revision: Union[str, None] = "a1b2c3d4e5f6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column("document_index", sa.Column("processing_token", sa.String(length=64), nullable=True))
    op.add_column("document_index", sa.Column("lease_expires_at", sa.DateTime(timezone=True), nullable=True))
    op.create_index(
        "idx_document_index_status_lease", "document_index", ["status", "lease_expires_at"], unique=False
    )

    op.add_column("collection_summary", sa.Column("processing_token", sa.String(length=64), nullable=True))
    op.add_column("collection_summary", sa.Column("lease_expires_at", sa.DateTime(timezone=True), nullable=True))
    op.create_index(
        "idx_collection_summary_status_lease",
        "collection_summary",
        ["status", "lease_expires_at"],
        unique=False,
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index("idx_collection_summary_status_lease", table_name="collection_summary")
    op.drop_column("collection_summary", "lease_expires_at")
    op.drop_column("collection_summary", "processing_token")

    op.drop_index("idx_document_index_status_lease", table_name="document_index")
    op.drop_column("document_index", "lease_expires_at")
    op.drop_column("document_index", "processing_token")
