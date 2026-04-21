"""add lightrag schema indexes for chunk lookups

Revision ID: b7c3d4e5f6a7
Revises: a1e2f3d4c5b6
Create Date: 2026-04-22 00:20:00.000000

"""

from typing import Sequence, Union

from alembic import op

revision: str = "b7c3d4e5f6a7"
down_revision: Union[str, None] = "a1e2f3d4c5b6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _execute_concurrent_ddl(sql: str) -> None:
    # These indexes land on live LightRAG tables, so build/drop them without
    # taking the write-blocking locks of transactional CREATE/DROP INDEX.
    with op.get_context().autocommit_block():
        op.execute(sql)


def upgrade() -> None:
    _execute_concurrent_ddl(
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS "
        "idx_lightrag_doc_chunks_workspace_doc "
        "ON lightrag_doc_chunks (workspace, full_doc_id)"
    )
    _execute_concurrent_ddl(
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS "
        "idx_lightrag_vdb_entity_chunk_ids_gin "
        "ON lightrag_vdb_entity USING gin (chunk_ids)"
    )
    _execute_concurrent_ddl(
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS "
        "idx_lightrag_vdb_relation_chunk_ids_gin "
        "ON lightrag_vdb_relation USING gin (chunk_ids)"
    )


def downgrade() -> None:
    _execute_concurrent_ddl("DROP INDEX CONCURRENTLY IF EXISTS idx_lightrag_vdb_relation_chunk_ids_gin")
    _execute_concurrent_ddl("DROP INDEX CONCURRENTLY IF EXISTS idx_lightrag_vdb_entity_chunk_ids_gin")
    _execute_concurrent_ddl("DROP INDEX CONCURRENTLY IF EXISTS idx_lightrag_doc_chunks_workspace_doc")
