"""add agent_message.runtime_kind discriminator (Phase 8 D8.5-BE / #92)

Phase 8 task #92 — adds the ``runtime_kind`` column to ``agent_message``
so a single canonical UIMessage table can host messages produced by
distinct runtimes (the agent reasoning loop today; future direct-LLM
and RAG-only chat paths). Per architect canonical lock msg=e01e9b4b
+ Weston msg=94dac98a, ``runtime_kind`` is a stable enum
(``agent_runtime`` / ``direct_chat`` / ``rag_chat``) and ``role``
retains its ChatML speaker semantics.

Existing rows (all produced by the agent runtime to date) are
backfilled to ``agent_runtime`` via the column ``server_default`` so
no data migration step is required. The column is non-null going
forward; SQLAlchemy ORM also defaults new rows to ``agent_runtime``
when the writer doesn't supply a value (D8.5 keeps the agent runtime
write path unchanged).

Per Phase 8 destructive philosophy + earayu2 msg=f20d5034 hard-cut
acceptance: no legacy chat-history table to drop in this migration —
the non-agent path's previous storage was Redis-only via
``RedisChatMessageHistory`` (kept until D8.6 / #80 cleanup post-soak
per Weston msg=5ec539c8).

Revision ID: c8f2d34a51e7
Revises: 84fac9e3d8c2
Create Date: 2026-04-26 01:20:00.000000
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "c8f2d34a51e7"
down_revision: Union[str, None] = "84fac9e3d8c2"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "agent_message",
        sa.Column(
            "runtime_kind",
            sa.String(length=24),
            nullable=False,
            server_default="agent_runtime",
        ),
    )


def downgrade() -> None:
    op.drop_column("agent_message", "runtime_kind")
