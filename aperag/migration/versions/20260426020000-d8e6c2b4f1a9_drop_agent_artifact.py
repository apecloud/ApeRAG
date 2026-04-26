"""drop agent_artifact table + agent_turn artifact_id columns (D8.6 #80 chunk-2)

Phase 8 D8.6 (#80) chunk-2 hard-cut: the agent runtime now writes the
canonical ``UIMessage`` envelope to ``agent_message`` at end-of-turn
(D8.2 #74 store + #80 wire-in), so the legacy ``agent_artifact`` row
projection (``answer`` / ``reference_bundle`` / ``error_summary`` /
``tool_result_summary`` / ``search_result_summary``) is dead weight.
This migration drops the table and the two ``agent_turn`` FK columns
that pointed into it.

Pre-launch system has no users / no data, so the cutover is direct
delete (per earayu2 hard-cut acceptance) — no backfill / no row
migration. The downgrade restores the dropped surface so a rollback
can replay subsequent migrations cleanly.

The ``agent_timeline_event`` table is intentionally retained — its
removal is chunk-3 (replay/reload semantic change reviewed
separately).

Revision ID: d8e6c2b4f1a9
Revises: c8f2d34a51e7
Create Date: 2026-04-26 02:00:00.000000
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "d8e6c2b4f1a9"
down_revision: Union[str, None] = "c8f2d34a51e7"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.drop_index("ix_agent_turn_reference_bundle_artifact_id", table_name="agent_turn")
    op.drop_index("ix_agent_turn_answer_artifact_id", table_name="agent_turn")
    op.drop_column("agent_turn", "reference_bundle_artifact_id")
    op.drop_column("agent_turn", "answer_artifact_id")
    op.drop_index("ix_agent_artifact_turn_id", table_name="agent_artifact")
    op.drop_index("ix_agent_artifact_artifact_type", table_name="agent_artifact")
    op.drop_index("idx_agent_artifact_turn_type", table_name="agent_artifact")
    op.drop_table("agent_artifact")


def downgrade() -> None:
    op.create_table(
        "agent_artifact",
        sa.Column("id", sa.String(length=24), nullable=False),
        sa.Column("turn_id", sa.String(length=24), nullable=False),
        sa.Column("artifact_type", sa.String(length=50), nullable=False),
        sa.Column("summary", sa.Text(), nullable=True),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column("storage_ref", sa.Text(), nullable=True),
        sa.Column("gmt_created", sa.DateTime(timezone=True), nullable=False),
        sa.Column("gmt_updated", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_agent_artifact_turn_type", "agent_artifact", ["turn_id", "artifact_type"], unique=False)
    op.create_index("ix_agent_artifact_artifact_type", "agent_artifact", ["artifact_type"], unique=False)
    op.create_index("ix_agent_artifact_turn_id", "agent_artifact", ["turn_id"], unique=False)
    op.add_column("agent_turn", sa.Column("answer_artifact_id", sa.String(length=24), nullable=True))
    op.add_column("agent_turn", sa.Column("reference_bundle_artifact_id", sa.String(length=24), nullable=True))
    op.create_index("ix_agent_turn_answer_artifact_id", "agent_turn", ["answer_artifact_id"], unique=False)
    op.create_index(
        "ix_agent_turn_reference_bundle_artifact_id",
        "agent_turn",
        ["reference_bundle_artifact_id"],
        unique=False,
    )
