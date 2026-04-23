# Copyright 2025 ApeCloud, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""SQLAlchemy models for the graphindex v2 tables.

Three physical tables, all keyed on ``collection_id`` as the tenant
discriminator:

* ``graphindex_nodes``  — entities
* ``graphindex_edges``  — relations (directed)
* ``graphindex_chunks`` — source chunks, for citation and re-ingest

Table names share the ``graphindex_`` prefix so any cross-tenant
"purge everything for collection X" operation can target them as a
family (see ``PostgresGraphStore.drop_collection``).

**Private module**: these classes are only imported by
``storage/postgres.py`` and Alembic migration scripts. Business code
must not ``from aperag.domains.knowledge_graph.graphindex.models import ...``. The
``__all__`` whitelist is there to make that conscious.
"""

from __future__ import annotations

from sqlalchemy import (
    ARRAY,
    BigInteger,
    Column,
    DateTime,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy import text as sql_text  # aliased: ``text`` collides with column name

from aperag.db.base import Base

# Shared table-name constants so Alembic migrations and SQL code paths
# can reference them without drift.
NODES_TABLE = "graphindex_nodes"
EDGES_TABLE = "graphindex_edges"
CHUNKS_TABLE = "graphindex_chunks"


class GraphIndexNode(Base):
    """An entity (node) in the knowledge graph."""

    __tablename__ = NODES_TABLE
    __table_args__ = (
        UniqueConstraint("collection_id", "entity_id", name="uq_graphindex_nodes_collection_entity"),
        Index("idx_graphindex_nodes_collection_type", "collection_id", "type"),
        Index("idx_graphindex_nodes_collection_name", "collection_id", "name"),
        # GIN index on ``source_chunk_ids`` lets "find all nodes that
        # reference this chunk" be fast during delete_document.
        Index(
            "idx_graphindex_nodes_chunks",
            "source_chunk_ids",
            postgresql_using="gin",
        ),
    )

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    collection_id = Column(String(255), nullable=False)
    entity_id = Column(String(255), nullable=False)
    name = Column(Text, nullable=False)
    type = Column(String(64), nullable=False)
    description = Column(Text, nullable=False, server_default="")
    # TEXT[] so we can update set-semantically and GIN-index it.
    source_chunk_ids = Column(ARRAY(String), nullable=False, server_default="{}", default=list)
    # ``server_default`` (not Python-side ``default``) so raw-SQL INSERTs
    # in ``storage/postgres.py`` get correct timestamps without having to
    # supply them column-by-column.
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=sql_text("now()"))
    updated_at = Column(DateTime(timezone=True), nullable=False, server_default=sql_text("now()"))


class GraphIndexEdge(Base):
    """A directed relation between two entities."""

    __tablename__ = EDGES_TABLE
    __table_args__ = (
        UniqueConstraint(
            "collection_id",
            "source_id",
            "target_id",
            name="uq_graphindex_edges_collection_src_tgt",
        ),
        Index("idx_graphindex_edges_collection_src", "collection_id", "source_id"),
        Index("idx_graphindex_edges_collection_tgt", "collection_id", "target_id"),
        Index(
            "idx_graphindex_edges_chunks",
            "source_chunk_ids",
            postgresql_using="gin",
        ),
    )

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    collection_id = Column(String(255), nullable=False)
    source_id = Column(String(255), nullable=False)
    target_id = Column(String(255), nullable=False)
    description = Column(Text, nullable=False, server_default="")
    weight = Column(Numeric(6, 3), nullable=False, server_default="0")
    source_chunk_ids = Column(ARRAY(String), nullable=False, server_default="{}", default=list)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=sql_text("now()"))
    updated_at = Column(DateTime(timezone=True), nullable=False, server_default=sql_text("now()"))


class GraphIndexChunk(Base):
    """A source chunk referenced by entities and relations."""

    __tablename__ = CHUNKS_TABLE
    __table_args__ = (
        UniqueConstraint("collection_id", "chunk_id", name="uq_graphindex_chunks_collection_chunk"),
        Index("idx_graphindex_chunks_collection_doc", "collection_id", "doc_id"),
    )

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    collection_id = Column(String(255), nullable=False)
    chunk_id = Column(String(64), nullable=False)
    doc_id = Column(String(255), nullable=False)
    order_in_doc = Column(Integer, nullable=False, server_default="0")
    text = Column(Text, nullable=False)
    file_path = Column(Text, nullable=False, server_default="")
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=sql_text("now()"))


__all__ = [
    "NODES_TABLE",
    "EDGES_TABLE",
    "CHUNKS_TABLE",
    "GraphIndexNode",
    "GraphIndexEdge",
    "GraphIndexChunk",
]
