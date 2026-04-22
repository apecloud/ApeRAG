"""graphindex v2 collection-state marker table

Revision ID: e6f7a8b9c0d1
Revises: d4e5f6a7b8c9
Create Date: 2026-04-22 17:00:00.000000

Adds ``graphindex_collection_state``, the explicit per-collection
"this collection is on v2" marker. A row is inserted the first time
``GraphIndexService.index_document`` completes for the collection;
its presence is the signal the business layer uses to route reads
to v2 vs. fall back to legacy LightRAG.

Having an explicit marker decouples rollout state from data content:
a collection that has been migrated to v2 but whose graph is
legitimately empty (zero-entity extraction, or all docs later
deleted) continues to read from v2 instead of silently falling
back to legacy data. See ``aperag/graphindex/models.py`` for the
full rationale.

No data migration: existing collections have no marker row, so
they correctly keep reading through the legacy fallback path until
the user runs a v2 re-index.
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "e6f7a8b9c0d1"
down_revision: Union[str, Sequence[str], None] = "d4e5f6a7b8c9"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "graphindex_collection_state",
        sa.Column("collection_id", sa.String(length=255), primary_key=True),
        sa.Column(
            "initialized_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )


def downgrade() -> None:
    op.drop_table("graphindex_collection_state")
