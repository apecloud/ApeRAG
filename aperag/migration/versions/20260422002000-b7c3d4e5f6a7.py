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


def upgrade() -> None:
    op.create_index(
        "idx_lightrag_doc_chunks_workspace_doc",
        "lightrag_doc_chunks",
        ["workspace", "full_doc_id"],
    )
    op.create_index(
        "idx_lightrag_vdb_entity_chunk_ids_gin",
        "lightrag_vdb_entity",
        ["chunk_ids"],
        postgresql_using="gin",
    )
    op.create_index(
        "idx_lightrag_vdb_relation_chunk_ids_gin",
        "lightrag_vdb_relation",
        ["chunk_ids"],
        postgresql_using="gin",
    )


def downgrade() -> None:
    op.drop_index("idx_lightrag_vdb_relation_chunk_ids_gin", table_name="lightrag_vdb_relation")
    op.drop_index("idx_lightrag_vdb_entity_chunk_ids_gin", table_name="lightrag_vdb_entity")
    op.drop_index("idx_lightrag_doc_chunks_workspace_doc", table_name="lightrag_doc_chunks")
