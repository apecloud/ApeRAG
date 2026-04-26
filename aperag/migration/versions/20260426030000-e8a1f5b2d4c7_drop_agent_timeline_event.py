"""drop agent_timeline_event table (D8.6 #80 chunk-3)

Phase 8 D8.6 (#80) chunk-3 hard-cut: wire-emitter envelopes are now
ephemeral wire telemetry — pushed to the Redis live cache only (live
SSE + reconnect-within-TTL). Reload outside the Redis TTL has no
historical events to replay; the canonical at-rest authority is
``agent_message`` (D8.2 #74).

Pre-launch system has no users / no data, so the cutover is direct
delete (per earayu2 hard-cut acceptance) — no backfill / no row
migration. The downgrade restores the dropped surface so a rollback
can replay subsequent migrations cleanly.

Revision ID: e8a1f5b2d4c7
Revises: d8e6c2b4f1a9
Create Date: 2026-04-26 03:00:00.000000
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "e8a1f5b2d4c7"
down_revision: Union[str, None] = "d8e6c2b4f1a9"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.drop_index("ix_agent_timeline_event_type", table_name="agent_timeline_event")
    op.drop_index("ix_agent_timeline_event_turn_id", table_name="agent_timeline_event")
    op.drop_index("ix_agent_timeline_event_timestamp", table_name="agent_timeline_event")
    op.drop_index("idx_agent_timeline_event_turn_timestamp", table_name="agent_timeline_event")
    op.drop_table("agent_timeline_event")


def downgrade() -> None:
    op.create_table(
        "agent_timeline_event",
        sa.Column("id", sa.String(length=24), nullable=False),
        sa.Column("turn_id", sa.String(length=24), nullable=False),
        sa.Column("sequence", sa.Integer(), nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("type", sa.String(length=128), nullable=False),
        sa.Column("label", sa.String(length=128), nullable=True),
        sa.Column("status", sa.String(length=64), nullable=True),
        sa.Column("actor", sa.String(length=50), nullable=False),
        sa.Column("data", sa.JSON(), nullable=False),
        sa.Column("gmt_created", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("turn_id", "sequence", name="uq_agent_timeline_event_turn_sequence"),
    )
    op.create_index(
        "idx_agent_timeline_event_turn_timestamp",
        "agent_timeline_event",
        ["turn_id", "timestamp"],
        unique=False,
    )
    op.create_index("ix_agent_timeline_event_timestamp", "agent_timeline_event", ["timestamp"], unique=False)
    op.create_index("ix_agent_timeline_event_turn_id", "agent_timeline_event", ["turn_id"], unique=False)
    op.create_index("ix_agent_timeline_event_type", "agent_timeline_event", ["type"], unique=False)
