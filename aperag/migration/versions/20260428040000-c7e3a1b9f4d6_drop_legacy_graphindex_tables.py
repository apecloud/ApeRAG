"""drop legacy ``graphindex_*`` tables — Wave 7 W7-10 close-out

Drops ``graphindex_chunks`` / ``graphindex_edges`` / ``graphindex_nodes``
plus their indexes / unique constraints. The legacy
``aperag/domains/knowledge_graph/graphindex/`` package was deleted in
the same PR (Wave 7 W7-10), and the new
``aperag_lineage_entity`` / ``aperag_lineage_relation`` tables (alembic
``e7a3b9c2d1f6``) own the graph-store role going forward.

Hard-cut policy per spec §K.12.12: legacy graph indexing was gated
behind ``enable_knowledge_graph=False`` until Wave 4, then never wired
into the new indexing pipeline (``run_index_document_sync`` had 0
production callers since Wave 4 hard-cut), so the legacy tables are
empty across every deployment. No backfill is needed; the downgrade
recreates the schema-only-empty tables so a rollback can replay
subsequent migrations cleanly.

Revision ID: c7e3a1b9f4d6
Revises: b5d2e8f1c9a4
Create Date: 2026-04-28 04:00:00.000000
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "c7e3a1b9f4d6"
down_revision: Union[str, None] = "b5d2e8f1c9a4"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Drop indexes first (Postgres tolerates DROP TABLE cascading them
    # but explicit drops keep the migration's behaviour identical
    # under ``alembic upgrade --sql`` offline mode).
    op.drop_index("idx_graphindex_nodes_collection_type", table_name="graphindex_nodes")
    op.drop_index("idx_graphindex_nodes_collection_name", table_name="graphindex_nodes")
    op.drop_index(
        "idx_graphindex_nodes_chunks", table_name="graphindex_nodes", postgresql_using="gin"
    )
    op.drop_table("graphindex_nodes")

    op.drop_index("idx_graphindex_edges_collection_tgt", table_name="graphindex_edges")
    op.drop_index("idx_graphindex_edges_collection_src", table_name="graphindex_edges")
    op.drop_index(
        "idx_graphindex_edges_chunks", table_name="graphindex_edges", postgresql_using="gin"
    )
    op.drop_table("graphindex_edges")

    op.drop_index("idx_graphindex_chunks_collection_doc", table_name="graphindex_chunks")
    op.drop_table("graphindex_chunks")


def downgrade() -> None:
    # Recreate the legacy tables empty — mirrors the original CREATE
    # statement in alembic ``6a8fc31427d9`` so a rollback puts the DB
    # back into the pre-W7-10 schema. No data restore is possible /
    # needed because the legacy tables were always empty (per
    # hard-cut policy above).
    op.create_table(
        "graphindex_chunks",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("collection_id", sa.String(length=255), nullable=False),
        sa.Column("chunk_id", sa.String(length=64), nullable=False),
        sa.Column("doc_id", sa.String(length=255), nullable=False),
        sa.Column("order_in_doc", sa.Integer(), server_default="0", nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("file_path", sa.Text(), server_default="", nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "collection_id", "chunk_id", name="uq_graphindex_chunks_collection_chunk"
        ),
    )
    op.create_index(
        "idx_graphindex_chunks_collection_doc",
        "graphindex_chunks",
        ["collection_id", "doc_id"],
        unique=False,
    )

    op.create_table(
        "graphindex_edges",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("collection_id", sa.String(length=255), nullable=False),
        sa.Column("source_id", sa.String(length=255), nullable=False),
        sa.Column("target_id", sa.String(length=255), nullable=False),
        sa.Column("description", sa.Text(), server_default="", nullable=False),
        sa.Column("weight", sa.Numeric(precision=6, scale=3), server_default="0", nullable=False),
        sa.Column("source_chunk_ids", sa.ARRAY(sa.String()), server_default="{}", nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "collection_id",
            "source_id",
            "target_id",
            name="uq_graphindex_edges_collection_src_tgt",
        ),
    )
    op.create_index(
        "idx_graphindex_edges_chunks",
        "graphindex_edges",
        ["source_chunk_ids"],
        unique=False,
        postgresql_using="gin",
    )
    op.create_index(
        "idx_graphindex_edges_collection_src",
        "graphindex_edges",
        ["collection_id", "source_id"],
        unique=False,
    )
    op.create_index(
        "idx_graphindex_edges_collection_tgt",
        "graphindex_edges",
        ["collection_id", "target_id"],
        unique=False,
    )

    op.create_table(
        "graphindex_nodes",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("collection_id", sa.String(length=255), nullable=False),
        sa.Column("entity_id", sa.String(length=255), nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("type", sa.String(length=64), nullable=False),
        sa.Column("description", sa.Text(), server_default="", nullable=False),
        sa.Column("source_chunk_ids", sa.ARRAY(sa.String()), server_default="{}", nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "collection_id", "entity_id", name="uq_graphindex_nodes_collection_entity"
        ),
    )
    op.create_index(
        "idx_graphindex_nodes_chunks",
        "graphindex_nodes",
        ["source_chunk_ids"],
        unique=False,
        postgresql_using="gin",
    )
    op.create_index(
        "idx_graphindex_nodes_collection_name",
        "graphindex_nodes",
        ["collection_id", "name"],
        unique=False,
    )
    op.create_index(
        "idx_graphindex_nodes_collection_type",
        "graphindex_nodes",
        ["collection_id", "type"],
        unique=False,
    )
