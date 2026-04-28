"""add evaluation answer_model + judge_model + judge_breakdown columns

Per architect spec ``#evaluation msg=2424afe2`` (2026-04-29) +
@earayu2 ratify ``msg=e6a534a8``: the evaluation pipeline is upgrading
from string-match judges to a real LLM-as-judge. The functional MVP
this round only computes a single Correctness dimension, but we land
all forward-compatible schema in one migration so a Phase 5 expansion
to Completeness / Faithfulness / Relevance does not need another
DDL pass.

* ``evaluation_runs.answer_model: VARCHAR(64)`` — model_id used for
  the answer phase. ``NULL`` means "fall back to
  ``Collection.config.completion``".
* ``evaluation_runs.judge_model: VARCHAR(64)`` — model_id used for
  the LLM-as-judge call. Same fallback semantics.
* ``evaluation_run_items.judge_breakdown: JSONB`` — per-item judge
  output breakdown reserved for the Phase 5 multi-dimensional
  scoring. The Correctness MVP leaves it ``NULL``; future scoring
  passes write
  ``{"correctness": int, "completeness": int|null,
    "faithfulness": int|null, "relevance": int|null}``.

Revision ID: a1b2c3d4e5f7
Revises: f2c3d4e5b6a8
Create Date: 2026-04-29 01:00:00.000000
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "a1b2c3d4e5f7"
down_revision = "f2c3d4e5b6a8"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "evaluation_runs",
        sa.Column("answer_model", sa.String(length=64), nullable=True),
    )
    op.add_column(
        "evaluation_runs",
        sa.Column("judge_model", sa.String(length=64), nullable=True),
    )
    op.add_column(
        "evaluation_run_items",
        sa.Column("judge_breakdown", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("evaluation_run_items", "judge_breakdown")
    op.drop_column("evaluation_runs", "judge_model")
    op.drop_column("evaluation_runs", "answer_model")
