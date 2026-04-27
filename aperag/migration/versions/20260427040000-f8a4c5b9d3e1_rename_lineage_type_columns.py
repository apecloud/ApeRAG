"""rename ``type`` columns on ``aperag_lineage_*`` tables

Wave 6 #36 (per architect Pattern 3 ruling): the ``type`` columns on
``aperag_lineage_entity`` and ``aperag_lineage_relation`` are renamed
to ``entity_type`` and ``relation_type`` respectively. Frees the bare
``type`` name across the §D.3 ``LineageGraphStore`` Protocol surface
so callers can introduce Python type-hint metadata or
Cypher ``TYPE()`` keyword usage without shadow conflicts on either
the ORM attribute side or the SQL/Cypher property side.

Hard-cut per earayu2 msg=30c81418 — graph indexing was gated behind
``enable_knowledge_graph=False`` by default since Wave 4, so there is
no production data on the legacy ``type`` columns. We rename in place
without a backfill / dual-column transitional state.

Revision ID: f8a4c5b9d3e1
Revises: f1c8d2a5b6e3
Create Date: 2026-04-27 04:00:00.000000
"""

from typing import Sequence, Union

from alembic import op

revision: str = "f8a4c5b9d3e1"
down_revision: Union[str, None] = "f1c8d2a5b6e3"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.alter_column(
        "aperag_lineage_entity",
        "type",
        new_column_name="entity_type",
    )
    op.alter_column(
        "aperag_lineage_relation",
        "type",
        new_column_name="relation_type",
    )


def downgrade() -> None:
    op.alter_column(
        "aperag_lineage_entity",
        "entity_type",
        new_column_name="type",
    )
    op.alter_column(
        "aperag_lineage_relation",
        "relation_type",
        new_column_name="type",
    )
