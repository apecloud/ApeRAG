"""indexing redesign — collection_id + source_path dispatch columns (T2.1)

Phase celery T2.1: per ``docs/modularization/indexing-redesign-design-pack.md``
§E.2 + §I.3, the orchestrator + reconciler need the dispatch payload to
self-contain ``collection_id`` (cleanup worker GC + tenant scoping) and
``source_path`` (orchestrator-to-modality input handoff) without
relying on the canonical ``collections/<cid>/documents/<did>/...`` path
parsing on every dispatch — vision modality reads non-canonical
synthetic JSON in T1 and could read PDF page extracts in T2.x without
breaking the layout assumption.

Both columns are added as NULL to keep T1.1 fixtures + early Wave 2
test rows working without backfill; the orchestrator gracefully skips
rows missing ``source_path`` (logs warning, leaves PENDING for
re-dispatch) so the schema upgrade is non-disruptive.

Revision ID: c2e8d5a1f3b9
Revises: f9c4d2a8e1b5
Create Date: 2026-04-27 00:00:00.000000
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "c2e8d5a1f3b9"
down_revision: Union[str, None] = "f9c4d2a8e1b5"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "document_index_v2",
        sa.Column("collection_id", sa.String(length=64), nullable=True),
    )
    op.add_column(
        "document_index_v2",
        sa.Column("source_path", sa.Text(), nullable=True),
    )
    op.create_index(
        "idx_document_index_v2_collection",
        "document_index_v2",
        ["collection_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("idx_document_index_v2_collection", table_name="document_index_v2")
    op.drop_column("document_index_v2", "source_path")
    op.drop_column("document_index_v2", "collection_id")
