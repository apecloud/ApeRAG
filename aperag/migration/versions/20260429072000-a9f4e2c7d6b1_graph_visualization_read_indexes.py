"""add graph visualization read indexes

The hybrid graph UI reads lineage entities by collection/type/name and
expands relation neighbours in both source and target directions. The
original PostgreSQL schema had only JSONB lineage indexes plus the
source-leading relation primary key, so target-side expansion could
fall back to a broad per-collection scan.

Revision ID: a9f4e2c7d6b1
Revises: f2c3d4e5b6a8
Create Date: 2026-04-29 07:20:00.000000
"""

from typing import Sequence, Union

from alembic import op

revision: str = "a9f4e2c7d6b1"
down_revision: Union[str, None] = "f2c3d4e5b6a8"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_index(
        "idx_lineage_entity_collection_type_name",
        "aperag_lineage_entity",
        ["collection_id", "entity_type", "name"],
        unique=False,
    )
    op.create_index(
        "idx_lineage_relation_collection_target_source_type",
        "aperag_lineage_relation",
        ["collection_id", "target", "source", "relation_type"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        "idx_lineage_relation_collection_target_source_type",
        table_name="aperag_lineage_relation",
    )
    op.drop_index(
        "idx_lineage_entity_collection_type_name",
        table_name="aperag_lineage_entity",
    )
