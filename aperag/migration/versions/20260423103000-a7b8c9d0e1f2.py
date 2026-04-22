"""create graph curation run and suggestion tables

Revision ID: a7b8c9d0e1f2
Revises: f1e2d3c4b5a6
Create Date: 2026-04-23 10:30:00.000000

Graph merge suggestion discovery is reintroduced in graphindex v2 as a
dedicated ``graph_curation`` workflow instead of reviving the old
LightRAG-era pipeline. The new workflow persists:

* async analysis runs
* persisted suggestions for review / accept / reject

The graph truth path remains unchanged: accepting a suggestion reuses
the existing ``GraphIndexService.merge_entities`` primitive.
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "a7b8c9d0e1f2"
down_revision: Union[str, Sequence[str], None] = "f1e2d3c4b5a6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "graph_curation_runs",
        sa.Column("id", sa.String(length=32), nullable=False),
        sa.Column("user_id", sa.String(length=256), nullable=False),
        sa.Column("collection_id", sa.String(length=24), nullable=False),
        sa.Column("status", sa.String(length=50), nullable=False),
        sa.Column("config_json", sa.JSON(), nullable=True),
        sa.Column("stats", sa.JSON(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("gmt_created", sa.DateTime(timezone=True), nullable=False),
        sa.Column("gmt_updated", sa.DateTime(timezone=True), nullable=False),
        sa.Column("gmt_started", sa.DateTime(timezone=True), nullable=True),
        sa.Column("gmt_finished", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "idx_graph_curation_runs_collection_status",
        "graph_curation_runs",
        ["collection_id", "status"],
        unique=False,
    )
    op.create_index(
        "idx_graph_curation_runs_user_created",
        "graph_curation_runs",
        ["user_id", "gmt_created"],
        unique=False,
    )
    op.create_table(
        "graph_curation_suggestions",
        sa.Column("id", sa.String(length=32), nullable=False),
        sa.Column("run_id", sa.String(length=32), nullable=False),
        sa.Column("user_id", sa.String(length=256), nullable=False),
        sa.Column("collection_id", sa.String(length=24), nullable=False),
        sa.Column("status", sa.String(length=50), nullable=False),
        sa.Column("entity_ids", sa.JSON(), nullable=False),
        sa.Column("entity_snapshots", sa.JSON(), nullable=False),
        sa.Column("target_entity_id", sa.String(length=255), nullable=False),
        sa.Column("confidence_score", sa.Numeric(precision=6, scale=3), nullable=False),
        sa.Column("reason", sa.Text(), nullable=False),
        sa.Column("evidence", sa.JSON(), nullable=True),
        sa.Column("resolution_note", sa.Text(), nullable=True),
        sa.Column("operated_by", sa.String(length=256), nullable=True),
        sa.Column("gmt_created", sa.DateTime(timezone=True), nullable=False),
        sa.Column("gmt_updated", sa.DateTime(timezone=True), nullable=False),
        sa.Column("gmt_operated", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "idx_graph_curation_suggestions_run",
        "graph_curation_suggestions",
        ["run_id"],
        unique=False,
    )
    op.create_index(
        "idx_graph_curation_suggestions_collection_status",
        "graph_curation_suggestions",
        ["collection_id", "status"],
        unique=False,
    )
    op.create_index(
        "idx_graph_curation_suggestions_collection_created",
        "graph_curation_suggestions",
        ["collection_id", "gmt_created"],
        unique=False,
    )
def downgrade() -> None:
    op.drop_index(
        "idx_graph_curation_suggestions_collection_created",
        table_name="graph_curation_suggestions",
    )
    op.drop_index(
        "idx_graph_curation_suggestions_collection_status",
        table_name="graph_curation_suggestions",
    )
    op.drop_index("idx_graph_curation_suggestions_run", table_name="graph_curation_suggestions")
    op.drop_table("graph_curation_suggestions")

    op.drop_index("idx_graph_curation_runs_user_created", table_name="graph_curation_runs")
    op.drop_index("idx_graph_curation_runs_collection_status", table_name="graph_curation_runs")
    op.drop_table("graph_curation_runs")
