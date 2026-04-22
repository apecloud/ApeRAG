"""graphindex v2 tables (nodes / edges / chunks)

Revision ID: d4e5f6a7b8c9
Revises: c1d2e3f4a5b6
Create Date: 2026-04-22 08:00:00.000000

Introduces the three physical tables backing the rewritten graphindex
module (``aperag/graphindex/``):

* ``graphindex_nodes``  — entities
* ``graphindex_edges``  — directed relations
* ``graphindex_chunks`` — source chunks

Co-exists with the legacy ``lightrag_graph_*`` tables; those stay in the
database, untouched, until a follow-up PR ports the curation features
and drops them explicitly.

See ``docs/zh-CN/design/graphindex_rewrite.md`` §5 for the schema
rationale.
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "d4e5f6a7b8c9"
down_revision: Union[str, Sequence[str], None] = "c1d2e3f4a5b6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "graphindex_nodes",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("collection_id", sa.String(length=255), nullable=False),
        sa.Column("entity_id", sa.String(length=255), nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("type", sa.String(length=64), nullable=False),
        sa.Column("description", sa.Text(), nullable=False, server_default=""),
        sa.Column(
            "source_chunk_ids",
            sa.ARRAY(sa.String()),
            nullable=False,
            server_default="{}",
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.UniqueConstraint(
            "collection_id", "entity_id", name="uq_graphindex_nodes_collection_entity"
        ),
    )
    op.create_index(
        "idx_graphindex_nodes_collection_type",
        "graphindex_nodes",
        ["collection_id", "type"],
    )
    op.create_index(
        "idx_graphindex_nodes_collection_name",
        "graphindex_nodes",
        ["collection_id", "name"],
    )
    op.create_index(
        "idx_graphindex_nodes_chunks",
        "graphindex_nodes",
        ["source_chunk_ids"],
        postgresql_using="gin",
    )

    op.create_table(
        "graphindex_edges",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("collection_id", sa.String(length=255), nullable=False),
        sa.Column("source_id", sa.String(length=255), nullable=False),
        sa.Column("target_id", sa.String(length=255), nullable=False),
        sa.Column("description", sa.Text(), nullable=False, server_default=""),
        sa.Column("weight", sa.Numeric(precision=6, scale=3), nullable=False, server_default="0"),
        sa.Column(
            "source_chunk_ids",
            sa.ARRAY(sa.String()),
            nullable=False,
            server_default="{}",
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.UniqueConstraint(
            "collection_id",
            "source_id",
            "target_id",
            name="uq_graphindex_edges_collection_src_tgt",
        ),
    )
    op.create_index(
        "idx_graphindex_edges_collection_src",
        "graphindex_edges",
        ["collection_id", "source_id"],
    )
    op.create_index(
        "idx_graphindex_edges_collection_tgt",
        "graphindex_edges",
        ["collection_id", "target_id"],
    )
    op.create_index(
        "idx_graphindex_edges_chunks",
        "graphindex_edges",
        ["source_chunk_ids"],
        postgresql_using="gin",
    )

    op.create_table(
        "graphindex_chunks",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("collection_id", sa.String(length=255), nullable=False),
        sa.Column("chunk_id", sa.String(length=64), nullable=False),
        sa.Column("doc_id", sa.String(length=255), nullable=False),
        sa.Column("order_in_doc", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("file_path", sa.Text(), nullable=False, server_default=""),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.UniqueConstraint(
            "collection_id", "chunk_id", name="uq_graphindex_chunks_collection_chunk"
        ),
    )
    op.create_index(
        "idx_graphindex_chunks_collection_doc",
        "graphindex_chunks",
        ["collection_id", "doc_id"],
    )


def downgrade() -> None:
    op.drop_index("idx_graphindex_chunks_collection_doc", table_name="graphindex_chunks")
    op.drop_table("graphindex_chunks")

    op.drop_index("idx_graphindex_edges_chunks", table_name="graphindex_edges")
    op.drop_index("idx_graphindex_edges_collection_tgt", table_name="graphindex_edges")
    op.drop_index("idx_graphindex_edges_collection_src", table_name="graphindex_edges")
    op.drop_table("graphindex_edges")

    op.drop_index("idx_graphindex_nodes_chunks", table_name="graphindex_nodes")
    op.drop_index("idx_graphindex_nodes_collection_name", table_name="graphindex_nodes")
    op.drop_index("idx_graphindex_nodes_collection_type", table_name="graphindex_nodes")
    op.drop_table("graphindex_nodes")
