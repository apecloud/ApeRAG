"""drop legacy ``collection_summary`` table (Wave 10 Chunk B)

Wave 10 §K.13 — collection-level summary state machine hard-cut.

The ``collection_summary`` table backed the legacy
``collection_summary_service.py`` orchestration (PENDING / GENERATING
/ COMPLETE / FAILED state machine + lease + processing_token +
version-driven reconciler). Wave 10 replaces this with:

* 5 nullable columns on ``collection`` itself (Chunk A migration
  ``d8f1e2a5b9c3``): ``summary`` / ``summary_updated_at`` /
  ``description_updated_at`` / ``regen_lease_owner`` /
  ``regen_lease_expires_at``.
* ``collection_regen_service`` (Chunk C) for two-stage regen
  (agent-runtime explore → ``summary`` → derive ``description``).
* ``reconcile_collection_descriptions_hook`` (Chunk E) wired into the
  existing 30s reconciler loop.

The Wave 10 design treats ``collection_summary`` as deprecated
infrastructure with no production data worth preserving (per
earayu2 msg=6a200cc4 hard-cut directive); the downgrade therefore
recreates an empty table shell with the original schema so a
rollback can replay subsequent migrations cleanly, but does NOT
attempt to backfill rows from the new ``Collection.summary`` column.

Revision ID: e1a2b3c4d5f6
Revises: d8f1e2a5b9c3
Create Date: 2026-04-28 06:00:00.000000
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "e1a2b3c4d5f6"
down_revision: Union[str, None] = "d8f1e2a5b9c3"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.drop_index("idx_collection_summary_status_lease", table_name="collection_summary")
    op.drop_index("ix_collection_summary_collection_id", table_name="collection_summary")
    op.drop_index("ix_collection_summary_status", table_name="collection_summary")
    op.drop_table("collection_summary")


def downgrade() -> None:
    # Recreate the table shell (empty); rollback replays subsequent
    # migrations from a known schema. No data backfill — Wave 10
    # hard-cut directive (no production data on the legacy table).
    op.create_table(
        "collection_summary",
        sa.Column("id", sa.String(length=24), primary_key=True),
        sa.Column("collection_id", sa.String(length=24), nullable=False),
        sa.Column("status", sa.String(length=50), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False, server_default=sa.text("1")),
        sa.Column("observed_version", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("summary", sa.Text(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("processing_token", sa.String(length=64), nullable=True),
        sa.Column("lease_expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("gmt_created", sa.DateTime(timezone=True), nullable=False),
        sa.Column("gmt_updated", sa.DateTime(timezone=True), nullable=False),
        sa.Column("gmt_last_reconciled", sa.DateTime(timezone=True), nullable=True),
        sa.UniqueConstraint("collection_id", name="uq_collection_summary"),
    )
    op.create_index(
        "idx_collection_summary_status_lease",
        "collection_summary",
        ["status", "lease_expires_at"],
    )
    op.create_index("ix_collection_summary_collection_id", "collection_summary", ["collection_id"])
    op.create_index("ix_collection_summary_status", "collection_summary", ["status"])
