"""add ``aperag_lineage_entity_alias`` table for user-driven entity merge

Wave 7 W7-6 (spec §K.12.7 / §K.12.10b): persists user-driven entity
merge intent so that subsequent indexer ``upsert_*_with_lineage`` calls
can transparently redirect a written ``record.name`` to the canonical
target when an alias has been recorded. Distinct from
``GraphCurationSuggestion`` (which records *suggested* merges pending
review); this table records *applied* merges.

Per spec §K.12.10b:

* ``(collection_id, alias_name)`` is the primary key — one alias maps
  to exactly one canonical at any time. Re-merging an alias to a new
  canonical UPDATEs the row in-place.
* ``canonical_name`` always points at the final (flattened) target —
  cycle detection in the service layer enforces transitive flattening,
  so a chain ``A → B → C`` is rewritten to ``A → C`` and ``B → C`` in
  one transaction.
* The table SURVIVES canonical entity GC (spec §K.12.7 decision X):
  if the canonical entity is later deleted, the alias rows stay so a
  future re-indexer write to the alias name still resolves correctly
  to the (now-empty) canonical, preserving user intent.

Pre-launch: no production users on the new lineage path → land
without backfill (per earayu2 hard-cut acceptance).

Revision ID: b5d2e8f1c9a4
Revises: a3b7c4d8e2f1
Create Date: 2026-04-28 03:00:00.000000
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "b5d2e8f1c9a4"
down_revision: Union[str, None] = "a3b7c4d8e2f1"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "aperag_lineage_entity_alias",
        sa.Column("collection_id", sa.String(length=64), nullable=False),
        sa.Column("alias_name", sa.String(length=512), nullable=False),
        sa.Column("canonical_name", sa.String(length=512), nullable=False),
        sa.Column("merged_by", sa.String(length=256), nullable=True),
        sa.Column(
            "gmt_created",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.Column(
            "gmt_updated",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.PrimaryKeyConstraint("collection_id", "alias_name", name="pk_aperag_lineage_entity_alias"),
    )
    op.create_index(
        "ix_aperag_lineage_entity_alias_canonical",
        "aperag_lineage_entity_alias",
        ["collection_id", "canonical_name"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_aperag_lineage_entity_alias_canonical",
        table_name="aperag_lineage_entity_alias",
    )
    op.drop_table("aperag_lineage_entity_alias")
