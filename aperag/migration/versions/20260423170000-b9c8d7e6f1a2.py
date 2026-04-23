"""evaluation v3 phase 1: add evaluation_datasets and evaluation_dataset_items

Purely additive foundation for the evaluation-v3 simplification. This
revision only creates the two new tables that the Phase 2 contract switch
will hook up to; it does not drop ``benchmark_*`` tables, does not touch
``evaluation_runs`` / ``evaluation_run_items``, and does not change any
public API. Old benchmark routes continue to run unchanged after this
migration applies.

Revision ID: b9c8d7e6f1a2
Revises: a7b8c9d0e1f2
Create Date: 2026-04-23 17:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "b9c8d7e6f1a2"
down_revision: Union[str, None] = "a7b8c9d0e1f2"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "evaluation_datasets",
        sa.Column("id", sa.String(length=32), primary_key=True),
        sa.Column("user_id", sa.String(length=256), nullable=False),
        sa.Column("collection_id", sa.String(length=24), nullable=True),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("source_type", sa.String(length=70), nullable=False),
        sa.Column("schema_hint", sa.JSON(), nullable=True),
        sa.Column("item_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("gmt_created", sa.DateTime(timezone=True), nullable=False),
        sa.Column("gmt_updated", sa.DateTime(timezone=True), nullable=False),
        sa.Column("gmt_deleted", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index(
        "idx_evaluation_datasets_user",
        "evaluation_datasets",
        ["user_id"],
    )
    op.create_index(
        "idx_evaluation_datasets_collection",
        "evaluation_datasets",
        ["collection_id"],
    )
    op.create_index(
        "ix_evaluation_datasets_gmt_deleted",
        "evaluation_datasets",
        ["gmt_deleted"],
    )

    op.create_table(
        "evaluation_dataset_items",
        sa.Column("id", sa.String(length=32), primary_key=True),
        sa.Column("dataset_id", sa.String(length=32), nullable=False),
        sa.Column("case_key", sa.String(length=128), nullable=False),
        sa.Column("input_message", sa.Text(), nullable=False),
        sa.Column("expected_answer", sa.Text(), nullable=True),
        sa.Column("reference_context", sa.Text(), nullable=True),
        sa.Column("tags", sa.JSON(), nullable=True),
        sa.Column("case_metadata", sa.JSON(), nullable=True),
        sa.Column("sort_key", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("gmt_created", sa.DateTime(timezone=True), nullable=False),
        sa.Column("gmt_updated", sa.DateTime(timezone=True), nullable=False),
        sa.Column("gmt_deleted", sa.DateTime(timezone=True), nullable=True),
        sa.UniqueConstraint("dataset_id", "case_key", name="uq_evaluation_dataset_item_case_key"),
    )
    op.create_index(
        "idx_evaluation_dataset_items_dataset",
        "evaluation_dataset_items",
        ["dataset_id"],
    )
    op.create_index(
        "ix_evaluation_dataset_items_gmt_deleted",
        "evaluation_dataset_items",
        ["gmt_deleted"],
    )


def downgrade() -> None:
    op.drop_index("ix_evaluation_dataset_items_gmt_deleted", table_name="evaluation_dataset_items")
    op.drop_index("idx_evaluation_dataset_items_dataset", table_name="evaluation_dataset_items")
    op.drop_table("evaluation_dataset_items")

    op.drop_index("ix_evaluation_datasets_gmt_deleted", table_name="evaluation_datasets")
    op.drop_index("idx_evaluation_datasets_collection", table_name="evaluation_datasets")
    op.drop_index("idx_evaluation_datasets_user", table_name="evaluation_datasets")
    op.drop_table("evaluation_datasets")
