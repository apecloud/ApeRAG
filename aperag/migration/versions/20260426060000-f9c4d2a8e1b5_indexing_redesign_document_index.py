"""indexing redesign — document_index table (T1.1 Foundation)

Phase celery T1.1: per ``docs/modularization/indexing-redesign-design-pack.md``
§F.1, create the ``document_index`` table that replaces the legacy Celery
orchestration state spread across ``aperag/tasks/`` + ``aperag/domains/
indexing/``.

Schema is the per-(document, parse_version, modality) triple with an
explicit ``is_serving`` flag. The partial unique index
``uniq_document_index_v2_serving`` enforces the §F.1 invariant — at most
one row per ``(document_id, modality)`` may have ``is_serving=TRUE`` —
at the database layer rather than the orchestrator (Bryce v2 review
msg=7ccb176f #2). Postgres native; SQLite 3.8+ supports the same
syntax (Tier 1 §L deploy stays consistent).

Pre-launch system has no users / no data, so this lands directly
without any backfill or migration of the existing Celery state (per
earayu2 hard-cut acceptance msg=9730bb6b). The downgrade drops the
table so a rollback can replay subsequent migrations cleanly.

Revision ID: f9c4d2a8e1b5
Revises: e8a1f5b2d4c7
Create Date: 2026-04-26 06:00:00.000000
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "f9c4d2a8e1b5"
down_revision: Union[str, None] = "e8a1f5b2d4c7"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "document_index_v2",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("document_id", sa.String(length=64), nullable=False),
        sa.Column("parse_version", sa.String(length=16), nullable=False),
        sa.Column("modality", sa.String(length=32), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("retry_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("retry_after", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_heartbeat", sa.DateTime(timezone=True), nullable=True),
        sa.Column("derived_artifact_path", sa.Text(), nullable=True),
        # §H.2 multi-tenant isolation: tenant_scope_key is the
        # rate-limit / quota / bulkhead partition key for the T2.2
        # quota lane. Carried per row so quota lookups need no JOIN.
        sa.Column("tenant_scope_key", sa.String(length=64), nullable=False),
        sa.Column(
            "is_serving",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("FALSE"),
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "document_id",
            "parse_version",
            "modality",
            name="uq_document_index_v2_triple",
        ),
    )
    # Lookup index for the cleanup worker + reconciler (status filter +
    # parse_version GC scan).
    op.create_index(
        "idx_document_index_v2_status_modality",
        "document_index_v2",
        ["status", "modality"],
        unique=False,
    )
    op.create_index(
        "idx_document_index_v2_document_modality",
        "document_index_v2",
        ["document_id", "modality"],
        unique=False,
    )
    # §H.2 tenant scope index for T2.2 quota lookup.
    op.create_index(
        "idx_document_index_v2_tenant_scope",
        "document_index_v2",
        ["tenant_scope_key"],
        unique=False,
    )
    # §F.1 partial unique invariant: at most one serving row per
    # (document_id, modality). DB-enforced so an orchestrator bug
    # cannot leave two rows with is_serving=TRUE for the same pair.
    op.create_index(
        "uniq_document_index_v2_serving",
        "document_index_v2",
        ["document_id", "modality"],
        unique=True,
        postgresql_where=sa.text("is_serving = TRUE"),
        sqlite_where=sa.text("is_serving = TRUE"),
    )


def downgrade() -> None:
    op.drop_index("uniq_document_index_v2_serving", table_name="document_index_v2")
    op.drop_index("idx_document_index_v2_tenant_scope", table_name="document_index_v2")
    op.drop_index("idx_document_index_v2_document_modality", table_name="document_index_v2")
    op.drop_index("idx_document_index_v2_status_modality", table_name="document_index_v2")
    op.drop_table("document_index_v2")
