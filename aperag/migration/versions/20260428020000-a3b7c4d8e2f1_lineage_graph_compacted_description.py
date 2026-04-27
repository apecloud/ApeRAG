"""add ``compacted_description`` column to ``aperag_lineage_*`` tables

Wave 7 W7-1 (spec §K.12.4): adds a nullable ``Text`` column to both
lineage tables so :class:`GraphIndexCompactor` (W7-2) can persist the
LLM-summarised unified description for each entity / relation.

The column is **derived**: ``description_parts`` (JSONB) remains the
per-doc source of truth; ``compacted_description`` is the cache fed to
the vector embedding step (W7-3) so embedded text stays bounded under
``max_description_chars`` (default 8000 — application-layer enforced
by Compactor, not a DB ``CHECK``). NULL means "not yet compacted" —
readers fall back to joining ``description_parts.text``.

PostgreSQL ``Text`` is unlimited; per spec §K.12.5 the length cap
lives in the application layer (Compactor + ``graph_extractor``)
because the same Protocol must run against Neo4j ``STRING`` and Nebula
``string`` properties which also have no fixed cap. Schema CHECK
constraints would have to be replicated cross-backend; the Compactor
truncate-not-fail policy is the canonical enforcement.

Hard-cut per spec §K.12 (legacy ``graphindex_*`` deployment was gated
behind ``enable_knowledge_graph=False`` and Wave 4 hard-cut already
locked-in no-production-data on the new ``aperag_lineage_*`` tables).
The downgrade drops the column so a rollback can replay subsequent
migrations cleanly.

Revision ID: a3b7c4d8e2f1
Revises: f8a4c5b9d3e1
Create Date: 2026-04-28 02:00:00.000000
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "a3b7c4d8e2f1"
down_revision: Union[str, None] = "f8a4c5b9d3e1"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "aperag_lineage_entity",
        sa.Column("compacted_description", sa.Text(), nullable=True),
    )
    op.add_column(
        "aperag_lineage_relation",
        sa.Column("compacted_description", sa.Text(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("aperag_lineage_relation", "compacted_description")
    op.drop_column("aperag_lineage_entity", "compacted_description")
