"""add collection auto-description columns (Wave 10 W10-1)

Wave 10 §K.13: collection summary + description auto-generation.

Adds 5 nullable columns to ``collection``:

* ``summary`` (TEXT) — Stage 1 output: long-form canonical content
  produced by agent-runtime free-explore. 5000-10000 chars.
* ``summary_updated_at`` (TIMESTAMP) — last successful Stage 1 regen.
* ``description_updated_at`` (TIMESTAMP) — last successful Stage 2
  derive. ``description`` itself is a pre-existing column whose
  semantic shifts in Wave 10 (user-editable → auto-generated).
* ``regen_lease_owner`` (VARCHAR(64)) — cluster-level lease token
  (UUID hex) used by both Stage 1 and Stage 2 regen so the two
  stages cannot race against each other on the same collection.
* ``regen_lease_expires_at`` (TIMESTAMP) — lease expiry; the next
  reconciler reclaims after a crash.

Per spec §K.13 the columns are all nullable (additive migration);
existing rows backfill to NULL meaning "no summary/description yet".
The reconciler hook (Chunk E) will populate them on the next sweep
once docs are present.

The legacy ``collection_summary`` table drop is **deferred to a
later chunk** (Chunk B) so this migration is purely additive — it
ships independently without breaking the still-live
``collection_summary_service`` reads.

Revision ID: d8f1e2a5b9c3
Revises: c7e3a1b9f4d6
Create Date: 2026-04-28 05:00:00.000000
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "d8f1e2a5b9c3"
down_revision: Union[str, None] = "c7e3a1b9f4d6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "collection",
        sa.Column("summary", sa.Text(), nullable=True),
    )
    op.add_column(
        "collection",
        sa.Column("summary_updated_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "collection",
        sa.Column("description_updated_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "collection",
        sa.Column("regen_lease_owner", sa.String(length=64), nullable=True),
    )
    op.add_column(
        "collection",
        sa.Column("regen_lease_expires_at", sa.DateTime(timezone=True), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("collection", "regen_lease_expires_at")
    op.drop_column("collection", "regen_lease_owner")
    op.drop_column("collection", "description_updated_at")
    op.drop_column("collection", "summary_updated_at")
    op.drop_column("collection", "summary")
