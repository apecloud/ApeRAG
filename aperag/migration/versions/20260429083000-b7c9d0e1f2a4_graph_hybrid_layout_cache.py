"""add graph hybrid layout cache

Revision ID: b7c9d0e1f2a4
Revises: c4d8e9f1a2b3
Create Date: 2026-04-29 08:30:00.000000
"""

from __future__ import annotations

from typing import Sequence

import sqlalchemy as sa
from alembic import op

revision = "b7c9d0e1f2a4"
down_revision = "c4d8e9f1a2b3"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "graph_hybrid_layout_cache",
        sa.Column("collection_id", sa.String(length=64), nullable=False),
        sa.Column("cache_key", sa.String(length=64), nullable=False),
        sa.Column("backend_type", sa.String(length=32), nullable=False),
        sa.Column("max_entities", sa.Integer(), nullable=False),
        sa.Column("entity_count", sa.Integer(), nullable=False),
        sa.Column("points_json", sa.JSON(), nullable=False),
        sa.Column("cluster_labels", sa.JSON(), nullable=False),
        sa.Column("gmt_created", sa.DateTime(timezone=True), nullable=False),
        sa.Column("gmt_updated", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("collection_id", "cache_key"),
    )
    op.create_index(
        "idx_graph_hybrid_layout_cache_collection_updated",
        "graph_hybrid_layout_cache",
        ["collection_id", "gmt_updated"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        "idx_graph_hybrid_layout_cache_collection_updated",
        table_name="graph_hybrid_layout_cache",
    )
    op.drop_table("graph_hybrid_layout_cache")
