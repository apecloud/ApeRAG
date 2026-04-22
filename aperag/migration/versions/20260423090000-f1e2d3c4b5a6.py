"""drop LightRAG legacy tables (graph, kv, vdb, merge suggestions)

Revision ID: f1e2d3c4b5a6
Revises: d4e5f6a7b8c9
Create Date: 2026-04-23 09:00:00.000000

LightRAG (the vendored fork under ``aperag/graph/``) has been removed
from the codebase. Every table it owned is now orphaned: this migration
drops them in one shot so the database schema stops advertising
storage the application no longer writes to.

Tables dropped:

* ``lightrag_graph_nodes``        — graph node storage
* ``lightrag_graph_edges``        — graph edge storage
* ``lightrag_doc_chunks``         — chunked doc storage w/ vector
* ``lightrag_vdb_entity``         — entity vector storage
* ``lightrag_vdb_relation``       — relation vector storage
* ``graph_index_merge_suggestions``         — curation suggestions (active)
* ``graph_index_merge_suggestions_history`` — curation suggestions (history)

**No data migration**. The product decision documented in
``docs/zh-CN/design/graphindex_rewrite.md`` is a hard cutover: users
re-index each collection into the graphindex v2 tables on demand. Any
row still living in the tables dropped here is discarded.

``downgrade`` is intentionally unimplemented. Recreating these tables
without recreating the LightRAG code that wrote them would give a
false sense of roll-back ability; restoring v1 means reverting the
code change alongside this migration.
"""

from typing import Sequence, Union

from alembic import op

revision: str = "f1e2d3c4b5a6"
down_revision: Union[str, Sequence[str], None] = "d4e5f6a7b8c9"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


_TABLES_TO_DROP = (
    # Graph
    "lightrag_graph_edges",
    "lightrag_graph_nodes",
    # KV / vector / chunk
    "lightrag_vdb_relation",
    "lightrag_vdb_entity",
    "lightrag_doc_chunks",
    # Curation
    "graph_index_merge_suggestions_history",
    "graph_index_merge_suggestions",
)


def upgrade() -> None:
    for table in _TABLES_TO_DROP:
        # ``IF EXISTS`` because fresh installs after v2 never had
        # these tables created (the install path runs ``alembic upgrade
        # head`` straight through), yet we still want this migration
        # to be a no-op success there rather than crashing boot.
        op.execute(f"DROP TABLE IF EXISTS {table} CASCADE")


def downgrade() -> None:
    raise NotImplementedError(
        "Cannot recreate LightRAG tables: the code that wrote to them has been "
        "deleted as well. Roll back the application to a pre-v2 release before "
        "attempting to downgrade the schema."
    )
