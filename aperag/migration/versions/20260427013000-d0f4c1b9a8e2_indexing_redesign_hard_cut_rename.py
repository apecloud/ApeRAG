"""indexing redesign — hard-cut + rename to canonical (T3.1)

Phase celery T3.1 per ``docs/modularization/indexing-redesign-design-pack.md``
§F.1 + §K Wave 3 + architect amendments msg=4a801b2b / msg=498b12f0:

This migration completes the Wave 1 → Wave 3 schema cutover:

1. ``DROP TABLE document_index`` — the legacy Celery-era table (per
   ``aperag/domains/indexing/db/models.py:DocumentIndex`` Wave 1 code,
   which Wave 3 hard-deletes alongside this migration).
2. ``ALTER TABLE document_index_v2`` — set the two T2.1 dispatch
   columns (``collection_id``, ``source_path``) to ``NOT NULL``. The
   Wave 1 fixture back-compat that justified ``NULL`` is gone in
   Wave 3 (the orchestrator + reconciler always populate them at
   INSERT time per architect msg=498b12f0).
3. ``RENAME TABLE document_index_v2 → document_index`` — back to
   the §F.1 canonical name. The "v2" suffix was a temporary measure
   (architect msg=4a801b2b) to avoid the SQLAlchemy table-name
   collision with the legacy class while both lived in the codebase.
4. Rename every index from ``*_v2_*`` → ``*_*`` to match the new
   table name (PG + SQLite both support ``ALTER INDEX RENAME``;
   the partial unique index is dropped + re-created since
   Postgres ALTER INDEX cannot relocate ``WHERE`` predicates and
   SQLite would silently keep the old reference).

Pre-launch system has no users / no data, so the schema rewrite
lands without backfill (per earayu2 hard-cut acceptance msg=9730bb6b).
The downgrade reverses every step so a rollback can replay subsequent
migrations cleanly.

Revision ID: d0f4c1b9a8e2
Revises: c2e8d5a1f3b9
Create Date: 2026-04-27 01:30:00.000000
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "d0f4c1b9a8e2"
down_revision: Union[str, None] = "c2e8d5a1f3b9"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. Drop the legacy Celery-era ``document_index`` table. Pre-
    # launch + no callers in Wave 3 (the dependent code is hard-
    # deleted in the same PR).
    op.execute("DROP TABLE IF EXISTS document_index CASCADE")

    # 2. Promote the two dispatch columns to NOT NULL. Wave 1 fixtures
    # used NULL for back-compat; Wave 3 orchestrator + reconciler
    # always populate them.
    op.alter_column(
        "document_index_v2",
        "collection_id",
        existing_type=sa.String(length=64),
        nullable=False,
    )
    op.alter_column(
        "document_index_v2",
        "source_path",
        existing_type=sa.Text(),
        nullable=False,
    )

    # 3. Rename indexes from *_v2_* → *_* before we rename the table
    # (PG / SQLite both fine with this order, and it keeps the index
    # symbol changes visible in the alembic diff). The partial-unique
    # index is dropped + re-created because the WHERE predicate must
    # be re-emitted for the new index name (PG quirk: ALTER INDEX
    # RENAME does not regenerate the predicate symbol map).
    op.drop_index(
        "uniq_document_index_v2_serving",
        table_name="document_index_v2",
    )
    op.execute(
        "ALTER INDEX uq_document_index_v2_triple "
        "RENAME TO uq_document_index_triple"
    )
    op.execute(
        "ALTER INDEX idx_document_index_v2_status_modality "
        "RENAME TO idx_document_index_status_modality"
    )
    op.execute(
        "ALTER INDEX idx_document_index_v2_document_modality "
        "RENAME TO idx_document_index_document_modality"
    )
    op.execute(
        "ALTER INDEX idx_document_index_v2_tenant_scope "
        "RENAME TO idx_document_index_tenant_scope"
    )
    op.execute(
        "ALTER INDEX idx_document_index_v2_collection "
        "RENAME TO idx_document_index_collection"
    )

    # 4. Rename the table back to the §F.1 canonical name.
    op.rename_table("document_index_v2", "document_index")

    # 5. Re-create the partial unique index against the final table
    # name. PG + SQLite 3.8+ both support the same syntax.
    op.create_index(
        "uniq_document_index_serving",
        "document_index",
        ["document_id", "modality"],
        unique=True,
        postgresql_where=sa.text("is_serving = TRUE"),
        sqlite_where=sa.text("is_serving = TRUE"),
    )


def downgrade() -> None:
    # Reverse the upgrade, mirroring its order in reverse.
    op.drop_index("uniq_document_index_serving", table_name="document_index")
    op.rename_table("document_index", "document_index_v2")
    op.execute(
        "ALTER INDEX idx_document_index_collection "
        "RENAME TO idx_document_index_v2_collection"
    )
    op.execute(
        "ALTER INDEX idx_document_index_tenant_scope "
        "RENAME TO idx_document_index_v2_tenant_scope"
    )
    op.execute(
        "ALTER INDEX idx_document_index_document_modality "
        "RENAME TO idx_document_index_v2_document_modality"
    )
    op.execute(
        "ALTER INDEX idx_document_index_status_modality "
        "RENAME TO idx_document_index_v2_status_modality"
    )
    op.execute(
        "ALTER INDEX uq_document_index_triple "
        "RENAME TO uq_document_index_v2_triple"
    )
    op.create_index(
        "uniq_document_index_v2_serving",
        "document_index_v2",
        ["document_id", "modality"],
        unique=True,
        postgresql_where=sa.text("is_serving = TRUE"),
        sqlite_where=sa.text("is_serving = TRUE"),
    )
    op.alter_column(
        "document_index_v2",
        "source_path",
        existing_type=sa.Text(),
        nullable=True,
    )
    op.alter_column(
        "document_index_v2",
        "collection_id",
        existing_type=sa.String(length=64),
        nullable=True,
    )
    # The legacy ``document_index`` table is recreated minimally so
    # the f9c4d2a8e1b5 → c2e8d5a1f3b9 chain can replay cleanly, but
    # it is intentionally schema-less because the legacy class was
    # also deleted in this Wave 3 migration. Operators rolling back
    # past this migration must restore the legacy class file before
    # re-running upgrades — there is no production scenario for it.
    op.execute(
        "CREATE TABLE document_index (id INTEGER PRIMARY KEY)"
    )
