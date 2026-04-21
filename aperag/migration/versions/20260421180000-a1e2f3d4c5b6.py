"""evaluation v2: benchmark datasets and evaluation runs

Revision ID: a1e2f3d4c5b6
Revises: 5f8a1c2d9b7e
Create Date: 2026-04-21 18:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "a1e2f3d4c5b6"
down_revision: Union[str, None] = "5f8a1c2d9b7e"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "benchmark_datasets",
        sa.Column("id", sa.String(length=32), primary_key=True),
        sa.Column("user_id", sa.String(length=256), nullable=False),
        sa.Column("collection_id", sa.String(length=24), nullable=True),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("source_type", sa.String(length=70), nullable=False),
        sa.Column("schema_hint", sa.JSON(), nullable=True),
        sa.Column("gmt_created", sa.DateTime(timezone=True), nullable=False),
        sa.Column("gmt_updated", sa.DateTime(timezone=True), nullable=False),
        sa.Column("gmt_deleted", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("idx_benchmark_datasets_user", "benchmark_datasets", ["user_id"])
    op.create_index("idx_benchmark_datasets_collection", "benchmark_datasets", ["collection_id"])
    op.create_index("ix_benchmark_datasets_gmt_deleted", "benchmark_datasets", ["gmt_deleted"])

    op.create_table(
        "benchmark_dataset_versions",
        sa.Column("id", sa.String(length=32), primary_key=True),
        sa.Column("dataset_id", sa.String(length=32), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("version_name", sa.String(length=255), nullable=True),
        sa.Column("status", sa.String(length=50), nullable=False),
        sa.Column("case_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("source_snapshot", sa.JSON(), nullable=True),
        sa.Column("gmt_created", sa.DateTime(timezone=True), nullable=False),
        sa.Column("gmt_updated", sa.DateTime(timezone=True), nullable=False),
        sa.Column("gmt_published", sa.DateTime(timezone=True), nullable=True),
        sa.UniqueConstraint("dataset_id", "version", name="uq_benchmark_dataset_version"),
    )
    op.create_index("idx_benchmark_dataset_versions_dataset", "benchmark_dataset_versions", ["dataset_id"])
    op.create_index("idx_benchmark_dataset_versions_status", "benchmark_dataset_versions", ["status"])

    op.create_table(
        "benchmark_cases",
        sa.Column("id", sa.String(length=32), primary_key=True),
        sa.Column("dataset_version_id", sa.String(length=32), nullable=False),
        sa.Column("case_key", sa.String(length=128), nullable=False),
        sa.Column("input_message", sa.Text(), nullable=False),
        sa.Column("expected_answer", sa.Text(), nullable=True),
        sa.Column("reference_context", sa.Text(), nullable=True),
        sa.Column("tags", sa.JSON(), nullable=True),
        sa.Column("case_metadata", sa.JSON(), nullable=True),
        sa.Column("sort_key", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("gmt_created", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("dataset_version_id", "case_key", name="uq_benchmark_case_version_key"),
    )
    op.create_index("idx_benchmark_cases_version", "benchmark_cases", ["dataset_version_id"])

    op.create_table(
        "evaluation_runs",
        sa.Column("id", sa.String(length=32), primary_key=True),
        sa.Column("user_id", sa.String(length=256), nullable=False),
        sa.Column("bot_id", sa.String(length=24), nullable=False),
        sa.Column("dataset_version_id", sa.String(length=32), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=True),
        sa.Column("bot_config_snapshot", sa.JSON(), nullable=True),
        sa.Column("model_config_snapshot", sa.JSON(), nullable=True),
        sa.Column("judge_config", sa.JSON(), nullable=True),
        sa.Column("status", sa.String(length=50), nullable=False),
        sa.Column("summary", sa.JSON(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("gmt_created", sa.DateTime(timezone=True), nullable=False),
        sa.Column("gmt_updated", sa.DateTime(timezone=True), nullable=False),
        sa.Column("gmt_started", sa.DateTime(timezone=True), nullable=True),
        sa.Column("gmt_finished", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("idx_evaluation_runs_user", "evaluation_runs", ["user_id"])
    op.create_index("idx_evaluation_runs_bot", "evaluation_runs", ["bot_id"])
    op.create_index("idx_evaluation_runs_status", "evaluation_runs", ["status"])
    op.create_index("idx_evaluation_runs_dataset_version", "evaluation_runs", ["dataset_version_id"])

    op.create_table(
        "evaluation_run_items",
        sa.Column("id", sa.String(length=32), primary_key=True),
        sa.Column("run_id", sa.String(length=32), nullable=False),
        sa.Column("case_id", sa.String(length=32), nullable=False),
        sa.Column("case_key", sa.String(length=128), nullable=False),
        sa.Column("status", sa.String(length=50), nullable=False),
        sa.Column("best_score", sa.Numeric(precision=6, scale=3), nullable=True),
        sa.Column("latest_attempt_id", sa.String(length=32), nullable=True),
        sa.Column("attempt_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("gmt_created", sa.DateTime(timezone=True), nullable=False),
        sa.Column("gmt_updated", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("run_id", "case_id", name="uq_evaluation_run_item_run_case"),
    )
    op.create_index("idx_evaluation_run_items_run", "evaluation_run_items", ["run_id"])
    op.create_index("idx_evaluation_run_items_status", "evaluation_run_items", ["status"])

    op.create_table(
        "evaluation_run_item_attempts",
        sa.Column("id", sa.String(length=32), primary_key=True),
        sa.Column("run_item_id", sa.String(length=32), nullable=False),
        sa.Column("run_id", sa.String(length=32), nullable=False),
        sa.Column("attempt_no", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(length=50), nullable=False),
        sa.Column("agent_chat_id", sa.String(length=24), nullable=True),
        sa.Column("agent_turn_id", sa.String(length=24), nullable=True),
        sa.Column("answer_text", sa.Text(), nullable=True),
        sa.Column("judge_result", sa.JSON(), nullable=True),
        sa.Column("score", sa.Numeric(precision=6, scale=3), nullable=True),
        sa.Column("latency_ms", sa.Integer(), nullable=True),
        sa.Column("token_usage", sa.JSON(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("retry_reason", sa.Text(), nullable=True),
        sa.Column("gmt_created", sa.DateTime(timezone=True), nullable=False),
        sa.Column("gmt_started", sa.DateTime(timezone=True), nullable=True),
        sa.Column("gmt_finished", sa.DateTime(timezone=True), nullable=True),
        sa.UniqueConstraint("run_item_id", "attempt_no", name="uq_evaluation_run_item_attempt_no"),
    )
    op.create_index("idx_evaluation_run_item_attempts_item", "evaluation_run_item_attempts", ["run_item_id"])
    op.create_index("idx_evaluation_run_item_attempts_run", "evaluation_run_item_attempts", ["run_id"])


def downgrade() -> None:
    op.drop_index("idx_evaluation_run_item_attempts_run", table_name="evaluation_run_item_attempts")
    op.drop_index("idx_evaluation_run_item_attempts_item", table_name="evaluation_run_item_attempts")
    op.drop_table("evaluation_run_item_attempts")

    op.drop_index("idx_evaluation_run_items_status", table_name="evaluation_run_items")
    op.drop_index("idx_evaluation_run_items_run", table_name="evaluation_run_items")
    op.drop_table("evaluation_run_items")

    op.drop_index("idx_evaluation_runs_dataset_version", table_name="evaluation_runs")
    op.drop_index("idx_evaluation_runs_status", table_name="evaluation_runs")
    op.drop_index("idx_evaluation_runs_bot", table_name="evaluation_runs")
    op.drop_index("idx_evaluation_runs_user", table_name="evaluation_runs")
    op.drop_table("evaluation_runs")

    op.drop_index("idx_benchmark_cases_version", table_name="benchmark_cases")
    op.drop_table("benchmark_cases")

    op.drop_index("idx_benchmark_dataset_versions_status", table_name="benchmark_dataset_versions")
    op.drop_index("idx_benchmark_dataset_versions_dataset", table_name="benchmark_dataset_versions")
    op.drop_table("benchmark_dataset_versions")

    op.drop_index("ix_benchmark_datasets_gmt_deleted", table_name="benchmark_datasets")
    op.drop_index("idx_benchmark_datasets_collection", table_name="benchmark_datasets")
    op.drop_index("idx_benchmark_datasets_user", table_name="benchmark_datasets")
    op.drop_table("benchmark_datasets")
