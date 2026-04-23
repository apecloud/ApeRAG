"""evaluation v3 phase 2: drop benchmark tables, switch evaluation runs to dataset snapshot

Destructive migration that switches the evaluation schema onto the simplified
Dataset + Run model. Prerequisites from the Phase 1 additive revision
(``b9c8d7e6f1a2``): ``evaluation_datasets`` / ``evaluation_dataset_items``
tables already exist.

Steps in ``upgrade()``:

1. Truncate the existing run tables. The public API was not formally
   released; per ``#evaluation #20`` design + @earayu2 16:25 口径 there are
   no production rows to preserve.
2. Drop ``evaluation_runs.dataset_version_id`` and the matching index.
3. Add the new ``evaluation_runs`` columns:
   ``dataset_id`` (NOT NULL), ``collection_id`` (nullable), ``dataset_name``
   (nullable snapshot). Indexes on ``dataset_id`` / ``collection_id``.
4. Rewrite ``evaluation_run_items``: drop the ``case_id`` FK-style column
   and its unique constraint; add ``source_dataset_item_id`` (audit, not an
   FK), ``sort_key`` (NOT NULL), ``input_message`` (NOT NULL), plus the
   nullable snapshot columns ``expected_answer``, ``reference_context``,
   ``tags``, ``case_metadata``.
5. Drop ``benchmark_cases``, ``benchmark_dataset_versions``,
   ``benchmark_datasets``.

``downgrade()`` is skeleton-only — it leaves the database at the cleared
evaluation schema (empty ``evaluation_datasets`` / ``evaluation_dataset_items``)
and does not resurrect the benchmark tables, because the simplified model is
canonical going forward.

Revision ID: c1d2e3f4a5b7
Revises: b9c8d7e6f1a2
Create Date: 2026-04-23 18:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "c1d2e3f4a5b7"
down_revision: Union[str, None] = "b9c8d7e6f1a2"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1) Clear out existing evaluation runtime data — no production rows exist.
    op.execute("TRUNCATE TABLE evaluation_run_item_attempts")
    op.execute("TRUNCATE TABLE evaluation_run_items")
    op.execute("TRUNCATE TABLE evaluation_runs")

    # 2) evaluation_runs: drop dataset_version_id and reshape FK surface.
    op.drop_index("idx_evaluation_runs_dataset_version", table_name="evaluation_runs")
    op.drop_column("evaluation_runs", "dataset_version_id")

    # 3) evaluation_runs: add the dataset-anchored columns.
    op.add_column(
        "evaluation_runs",
        sa.Column("dataset_id", sa.String(length=32), nullable=False),
    )
    op.add_column(
        "evaluation_runs",
        sa.Column("collection_id", sa.String(length=24), nullable=True),
    )
    op.add_column(
        "evaluation_runs",
        sa.Column("dataset_name", sa.String(length=255), nullable=True),
    )
    op.create_index("idx_evaluation_runs_dataset", "evaluation_runs", ["dataset_id"])
    op.create_index("idx_evaluation_runs_collection", "evaluation_runs", ["collection_id"])

    # 4) evaluation_run_items: drop legacy case_* columns, add snapshot columns.
    op.drop_constraint(
        "uq_evaluation_run_item_run_case", "evaluation_run_items", type_="unique"
    )
    op.drop_column("evaluation_run_items", "case_id")

    op.add_column(
        "evaluation_run_items",
        sa.Column("source_dataset_item_id", sa.String(length=32), nullable=True),
    )
    op.add_column(
        "evaluation_run_items",
        sa.Column("sort_key", sa.Integer(), nullable=False, server_default="0"),
    )
    op.add_column(
        "evaluation_run_items",
        sa.Column("input_message", sa.Text(), nullable=False, server_default=""),
    )
    # server_default seeded an empty string for the NOT NULL backfill; drop it
    # so application code is forced to populate input_message explicitly.
    op.alter_column("evaluation_run_items", "input_message", server_default=None)
    op.add_column(
        "evaluation_run_items",
        sa.Column("expected_answer", sa.Text(), nullable=True),
    )
    op.add_column(
        "evaluation_run_items",
        sa.Column("reference_context", sa.Text(), nullable=True),
    )
    op.add_column(
        "evaluation_run_items",
        sa.Column("tags", sa.JSON(), nullable=True),
    )
    op.add_column(
        "evaluation_run_items",
        sa.Column("case_metadata", sa.JSON(), nullable=True),
    )

    # 5) Drop the legacy benchmark tables.
    op.drop_index("idx_benchmark_cases_version", table_name="benchmark_cases")
    op.drop_table("benchmark_cases")

    op.drop_index(
        "idx_benchmark_dataset_versions_status", table_name="benchmark_dataset_versions"
    )
    op.drop_index(
        "idx_benchmark_dataset_versions_dataset", table_name="benchmark_dataset_versions"
    )
    op.drop_table("benchmark_dataset_versions")

    op.drop_index(
        "ix_benchmark_datasets_gmt_deleted", table_name="benchmark_datasets"
    )
    op.drop_index("idx_benchmark_datasets_collection", table_name="benchmark_datasets")
    op.drop_index("idx_benchmark_datasets_user", table_name="benchmark_datasets")
    op.drop_table("benchmark_datasets")


def downgrade() -> None:
    # Skeleton-only downgrade: revert the evaluation_runs / run_items column
    # reshape so the schema matches the Phase 1 state. We intentionally do NOT
    # re-create the benchmark_* tables — the simplified model is canonical and
    # the old layout was never released to production.
    op.execute("TRUNCATE TABLE evaluation_run_item_attempts")
    op.execute("TRUNCATE TABLE evaluation_run_items")
    op.execute("TRUNCATE TABLE evaluation_runs")

    op.drop_column("evaluation_run_items", "case_metadata")
    op.drop_column("evaluation_run_items", "tags")
    op.drop_column("evaluation_run_items", "reference_context")
    op.drop_column("evaluation_run_items", "expected_answer")
    op.drop_column("evaluation_run_items", "input_message")
    op.drop_column("evaluation_run_items", "sort_key")
    op.drop_column("evaluation_run_items", "source_dataset_item_id")
    op.add_column(
        "evaluation_run_items",
        sa.Column("case_id", sa.String(length=32), nullable=False, server_default=""),
    )
    op.alter_column("evaluation_run_items", "case_id", server_default=None)
    op.create_unique_constraint(
        "uq_evaluation_run_item_run_case",
        "evaluation_run_items",
        ["run_id", "case_id"],
    )

    op.drop_index("idx_evaluation_runs_collection", table_name="evaluation_runs")
    op.drop_index("idx_evaluation_runs_dataset", table_name="evaluation_runs")
    op.drop_column("evaluation_runs", "dataset_name")
    op.drop_column("evaluation_runs", "collection_id")
    op.drop_column("evaluation_runs", "dataset_id")

    op.add_column(
        "evaluation_runs",
        sa.Column(
            "dataset_version_id", sa.String(length=32), nullable=False, server_default=""
        ),
    )
    op.alter_column("evaluation_runs", "dataset_version_id", server_default=None)
    op.create_index(
        "idx_evaluation_runs_dataset_version", "evaluation_runs", ["dataset_version_id"]
    )
