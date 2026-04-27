"""indexing redesign — lineage graph PostgreSQL tables (T8 chunk 4a)

Phase celery T8 chunk 4a per ``docs/modularization/indexing-redesign-design-pack.md``
§D.3.5 + chunk 4 acceptance lock item 1 + 2 (architect msg=baf6618e /
huangheng msg=b6f20096):

Adds the two PostgreSQL tables that back
:class:`aperag.indexing.graph_storage.postgres.PostgresLineageGraphStore`
(T8 chunk 1, msg=f0571f98), mirroring the adapter's private ORM 100%:

* ``aperag_lineage_entity`` — one row per ``(collection_id, name)``
  storing the lineage SET in ``source_lineage`` JSONB and the
  per-document description fragments in ``description_parts`` JSONB.
* ``aperag_lineage_relation`` — one row per
  ``(collection_id, source, target, type)`` quadruple storing
  ``evidence_lineage`` + ``description_parts`` JSONB.

The relation table intentionally does NOT carry a standalone
``description`` Text column — Wave 4 chunk 4 drops that legacy
residual across all 3 backends (Postgres / Neo4j / Nebula). The
canonical relation description is the per-document fragments in
``description_parts`` (§D.3.3 Option A "preserve every doc's
contribution verbatim"); the redundant overwritten-on-every-upsert
``description`` column was a v1 migration artefact with no consumer
on the §D.3 / §G.5 read path (``RelationWithLineage`` does not expose
it). Dropping it cross-backend keeps "ORM 100% mirror" honest.

Pre-launch system has no users / no data, so the two tables land
without backfill (per earayu2 hard-cut acceptance msg=9730bb6b).
The downgrade drops both tables so a rollback can replay subsequent
migrations cleanly.

PostgreSQL only — the lineage adapter SQL uses ``jsonb_*`` functions
(``jsonb_agg`` / ``jsonb_array_elements`` / ``@>`` containment) and
the dedup-key strip-then-append pattern in
``upsert_*_with_lineage`` cannot be expressed against SQLite TEXT
columns. Other backends (Neo4j / Nebula) realise the same
``LineageGraphStore`` Protocol via their native list-of-string
encoding (see ``aperag/indexing/graph_storage/neo4j.py`` /
``nebula.py``).

Revision ID: e7a3b9c2d1f6
Revises: d0f4c1b9a8e2
Create Date: 2026-04-27 02:30:00.000000
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "e7a3b9c2d1f6"
down_revision: Union[str, None] = "d0f4c1b9a8e2"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "aperag_lineage_entity",
        sa.Column("collection_id", sa.String(length=64), nullable=False),
        sa.Column("name", sa.String(length=512), nullable=False),
        sa.Column("type", sa.String(length=64), nullable=False),
        sa.Column(
            "source_lineage",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
        sa.Column(
            "description_parts",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
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
        sa.PrimaryKeyConstraint("collection_id", "name"),
    )
    # JSONB containment scan for ``find_entity_ids_with_lineage`` —
    # the adapter issues ``source_lineage @> [{"document_id": $d}]``
    # to enumerate entities touched by a document. The GIN index lets
    # the @> operator skip a sequential scan on a multi-tenant table.
    op.create_index(
        "idx_lineage_entity_source_lineage_gin",
        "aperag_lineage_entity",
        ["source_lineage"],
        unique=False,
        postgresql_using="gin",
    )

    op.create_table(
        "aperag_lineage_relation",
        sa.Column("collection_id", sa.String(length=64), nullable=False),
        sa.Column("source", sa.String(length=512), nullable=False),
        sa.Column("target", sa.String(length=512), nullable=False),
        sa.Column("type", sa.String(length=64), nullable=False),
        sa.Column(
            "evidence_lineage",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
        sa.Column(
            "description_parts",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
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
        sa.PrimaryKeyConstraint("collection_id", "source", "target", "type"),
    )
    op.create_index(
        "idx_lineage_relation_evidence_lineage_gin",
        "aperag_lineage_relation",
        ["evidence_lineage"],
        unique=False,
        postgresql_using="gin",
    )


def downgrade() -> None:
    op.drop_index(
        "idx_lineage_relation_evidence_lineage_gin",
        table_name="aperag_lineage_relation",
    )
    op.drop_table("aperag_lineage_relation")
    op.drop_index(
        "idx_lineage_entity_source_lineage_gin",
        table_name="aperag_lineage_entity",
    )
    op.drop_table("aperag_lineage_entity")
