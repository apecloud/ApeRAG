"""add agent_message table for D8.2 UIMessage at-rest storage

Phase 8 task #74 (D8.2) — introduce the ``agent_message`` table that
holds the UIMessage at-rest envelope per the canonical
``docs/modularization/agent-message-protocol-design.md``. One row per
assistant turn (1:1 with ``agent_turn`` via ``turn_id``); ``parts``
holds the JSON-serialised ``UIMessagePart`` array.

The legacy ``agent_artifact`` / ``agent_timeline_event`` tables are
retained during D8.x rollout — D8.6 (#80) will drop them once the FE
renderer is reading from ``agent_message`` exclusively. This keeps
the migration window safe for in-flight turns.

Revision ID: d8e2c4a17b91
Revises: 7c4e9e1f8b21
Create Date: 2026-04-25 20:00:00.000000
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "d8e2c4a17b91"
down_revision: Union[str, None] = "7c4e9e1f8b21"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "agent_message",
        sa.Column("id", sa.String(length=24), nullable=False),
        sa.Column("turn_id", sa.String(length=24), nullable=False),
        sa.Column("chat_id", sa.String(length=24), nullable=False),
        sa.Column("role", sa.String(length=16), nullable=False),
        sa.Column("schema_version", sa.String(length=64), nullable=False),
        sa.Column("parts", sa.JSON(), nullable=False),
        sa.Column(
            "gmt_created",
            sa.DateTime(timezone=True),
            nullable=False,
        ),
        sa.Column(
            "gmt_updated",
            sa.DateTime(timezone=True),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("turn_id", name="uq_agent_message_turn"),
    )
    op.create_index("ix_agent_message_turn_id", "agent_message", ["turn_id"], unique=False)
    op.create_index("ix_agent_message_chat_id", "agent_message", ["chat_id"], unique=False)
    op.create_index(
        "idx_agent_message_chat_created",
        "agent_message",
        ["chat_id", "gmt_created"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("idx_agent_message_chat_created", table_name="agent_message")
    op.drop_index("ix_agent_message_chat_id", table_name="agent_message")
    op.drop_index("ix_agent_message_turn_id", table_name="agent_message")
    op.drop_table("agent_message")
