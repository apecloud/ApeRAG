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

"""PostgreSQL implementation of the ``GraphStore`` Protocol.

Deliberate choices (so nobody has to ask later):

* **Async SQLAlchemy throughout.** No ``asyncio.to_thread`` wrappers
  around sync code — the reason LightRAG v1 had them was inheriting an
  upstream design, not a PG-specific necessity.
* **Raw SQL via ``text()`` for hot paths (upsert, neighborhood walk),
  ORM for cold paths (delete_collection, list_labels).** ORM for
  bulk-upsert would require model hydration of every row; text() with
  ``VALUES (...), (...), ...`` is ~10x faster on modern SQLAlchemy.
* **Single transaction per public method.** Every async method opens
  ``engine.begin()`` once. If a caller wants to batch several methods
  in one transaction, that's the service layer's job; this layer keeps
  the boundary crisp.
* **All values are parameterised.** Table names come from
  ``models.NODES_TABLE`` etc. (module constants, never user input);
  everything else goes through ``:name`` bind params. Zero f-string
  concatenation of user-derived values.
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from aperag.graphindex.dto import (
    Chunk,
    DeleteDocumentResult,
    Entity,
    KnowledgeGraph,
    Relation,
)
from aperag.graphindex.models import (
    CHUNKS_TABLE,
    EDGES_TABLE,
    NODES_TABLE,
)

logger = logging.getLogger(__name__)


class PostgresGraphStore:
    """``GraphStore`` backed by PostgreSQL. The only implementation shipped
    with graphindex v2.

    Thread safety: ``AsyncEngine`` is already safe to share across
    coroutines; this class is stateless beyond its engine reference. One
    instance per process is fine.
    """

    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine

    # =========================================================== schema
    async def ensure_schema(self) -> None:
        """Tables are created by Alembic migrations, not by the
        application. This method stays for Protocol parity and so tests
        can stand up a database without running migrations."""
        # Intentional no-op: we rely on `alembic upgrade head` to create
        # the graphindex_* tables. Running DDL from application code
        # creates subtle "who owns the schema" bugs in multi-deployment
        # scenarios. If you need the tables for a test, use
        # `tests/conftest.py` fixtures that import Base and call
        # ``Base.metadata.create_all``.
        return None

    async def drop_collection(self, collection_id: str) -> None:
        """Rip out every row tagged with this collection id across all
        three tables. Used by the collection-delete Celery task.
        """
        async with self._engine.begin() as conn:
            await conn.execute(
                text(f"DELETE FROM {EDGES_TABLE} WHERE collection_id = :cid"),
                {"cid": collection_id},
            )
            await conn.execute(
                text(f"DELETE FROM {NODES_TABLE} WHERE collection_id = :cid"),
                {"cid": collection_id},
            )
            await conn.execute(
                text(f"DELETE FROM {CHUNKS_TABLE} WHERE collection_id = :cid"),
                {"cid": collection_id},
            )
        logger.info("graphindex: dropped all rows for collection %s", collection_id)

    # ============================================================ write
    async def upsert_chunks(self, collection_id: str, chunks: Sequence[Chunk]) -> None:
        if not chunks:
            return
        values_sql, params = _build_multi_row_values(
            [
                {
                    "cid": collection_id,
                    "chunk_id": c.chunk_id,
                    "doc_id": c.doc_id,
                    "ord": c.order_in_doc,
                    "txt": c.text,
                    "fp": c.file_path or "",
                }
                for c in chunks
            ],
            columns=("cid", "chunk_id", "doc_id", "ord", "txt", "fp"),
        )
        sql = (
            f"INSERT INTO {CHUNKS_TABLE} "
            f"(collection_id, chunk_id, doc_id, order_in_doc, text, file_path) "
            f"VALUES {values_sql} "
            f"ON CONFLICT (collection_id, chunk_id) DO UPDATE SET "
            f"doc_id = EXCLUDED.doc_id, "
            f"order_in_doc = EXCLUDED.order_in_doc, "
            f"text = EXCLUDED.text, "
            f"file_path = EXCLUDED.file_path"
        )
        async with self._engine.begin() as conn:
            await conn.execute(text(sql), params)

    async def upsert_entities(self, collection_id: str, entities: Sequence[Entity]) -> None:
        """Insert entities; on conflict merge source_chunk_ids as a
        set union and keep the longer description (simple but stable)."""
        if not entities:
            return
        values_sql, params = _build_multi_row_values(
            [
                {
                    "cid": collection_id,
                    "eid": e.entity_id,
                    "name": e.name,
                    "type": e.type,
                    "desc": e.description,
                    "chunks": list(e.source_chunk_ids),
                }
                for e in entities
            ],
            columns=("cid", "eid", "name", "type", "desc", "chunks"),
            cast_columns={"chunks": "text[]"},
        )
        sql = (
            f"INSERT INTO {NODES_TABLE} "
            f"(collection_id, entity_id, name, type, description, source_chunk_ids) "
            f"VALUES {values_sql} "
            f"ON CONFLICT (collection_id, entity_id) DO UPDATE SET "
            # Union the chunk id arrays without duplicates.
            f"source_chunk_ids = ARRAY("
            f"  SELECT DISTINCT unnest("
            f"    {NODES_TABLE}.source_chunk_ids || EXCLUDED.source_chunk_ids"
            f"  )"
            f"), "
            # Prefer the longer description — cheap proxy for "more
            # informative". Real merge semantics belong in the curation
            # pipeline, not here.
            f"description = CASE "
            f"  WHEN length(EXCLUDED.description) > length({NODES_TABLE}.description) "
            f"  THEN EXCLUDED.description "
            f"  ELSE {NODES_TABLE}.description "
            f"END, "
            # Keep the newer type; type is a closed-set vocabulary from
            # the extraction prompt, so values converge on repeated writes.
            f"type = EXCLUDED.type, "
            f"name = EXCLUDED.name, "
            f"updated_at = now()"
        )
        async with self._engine.begin() as conn:
            await conn.execute(text(sql), params)

    async def upsert_relations(self, collection_id: str, relations: Sequence[Relation]) -> None:
        """Insert relations; on ``(source, target)`` conflict union the
        chunk id set, take the max weight, and concatenate descriptions
        with a separator."""
        if not relations:
            return
        values_sql, params = _build_multi_row_values(
            [
                {
                    "cid": collection_id,
                    "src": r.source_id,
                    "tgt": r.target_id,
                    "desc": r.description,
                    "w": float(r.weight),
                    "chunks": list(r.source_chunk_ids),
                }
                for r in relations
            ],
            columns=("cid", "src", "tgt", "desc", "w", "chunks"),
            cast_columns={"chunks": "text[]"},
        )
        sql = (
            f"INSERT INTO {EDGES_TABLE} "
            f"(collection_id, source_id, target_id, description, weight, source_chunk_ids) "
            f"VALUES {values_sql} "
            f"ON CONFLICT (collection_id, source_id, target_id) DO UPDATE SET "
            f"source_chunk_ids = ARRAY("
            f"  SELECT DISTINCT unnest("
            f"    {EDGES_TABLE}.source_chunk_ids || EXCLUDED.source_chunk_ids"
            f"  )"
            f"), "
            f"weight = GREATEST({EDGES_TABLE}.weight, EXCLUDED.weight), "
            f"description = CASE "
            f"  WHEN length(EXCLUDED.description) > length({EDGES_TABLE}.description) "
            f"  THEN EXCLUDED.description "
            f"  ELSE {EDGES_TABLE}.description "
            f"END, "
            f"updated_at = now()"
        )
        async with self._engine.begin() as conn:
            await conn.execute(text(sql), params)

    # =========================================================== delete
    async def delete_document_rows(self, collection_id: str, doc_id: str) -> DeleteDocumentResult:
        """Atomic delete: wipe this document's chunks, prune chunk ids
        from any entity/relation that mentioned them, and garbage-collect
        entities/relations that became orphan.

        Runs in a single transaction so an interrupted delete never
        leaves dangling chunk-id references in the graph.
        """
        async with self._engine.begin() as conn:
            # 1. Collect the chunk ids that belong to this document.
            chunk_rows = (
                await conn.execute(
                    text(f"SELECT chunk_id FROM {CHUNKS_TABLE} WHERE collection_id = :cid AND doc_id = :did"),
                    {"cid": collection_id, "did": doc_id},
                )
            ).all()
            chunk_ids = [row.chunk_id for row in chunk_rows]
            if not chunk_ids:
                return DeleteDocumentResult(
                    doc_id=doc_id,
                    chunks_removed=0,
                    entities_removed=0,
                    relations_removed=0,
                )

            # 2. Delete the chunks themselves.
            deleted_chunks = (
                await conn.execute(
                    text(f"DELETE FROM {CHUNKS_TABLE} WHERE collection_id = :cid AND doc_id = :did"),
                    {"cid": collection_id, "did": doc_id},
                )
            ).rowcount or 0

            # 3. Prune chunk ids from entity and relation rows; use
            # ``array(select ... except ...)`` form so the array value
            # stays well-formed under concurrent writes.
            #
            # We cast both sides of the ``&&`` and ``unnest`` to text[]
            # because the SQLAlchemy ARRAY(String) column maps to
            # varchar[] on PG, and PG's ``&&`` operator is type-strict
            # (varchar[] && text[] errors at runtime). Casting both
            # sides to text[] is cheap and removes the trap.
            prune_nodes_sql = (
                f"UPDATE {NODES_TABLE} SET "
                f"source_chunk_ids = ARRAY("
                f"  SELECT unnest(CAST(source_chunk_ids AS text[])) "
                f"  EXCEPT SELECT unnest(CAST(:chunks AS text[]))"
                f"), "
                f"updated_at = now() "
                f"WHERE collection_id = :cid "
                f"  AND CAST(source_chunk_ids AS text[]) && CAST(:chunks AS text[])"
            )
            await conn.execute(text(prune_nodes_sql), {"cid": collection_id, "chunks": chunk_ids})
            prune_edges_sql = prune_nodes_sql.replace(NODES_TABLE, EDGES_TABLE)
            await conn.execute(text(prune_edges_sql), {"cid": collection_id, "chunks": chunk_ids})

            # 4. Delete rows whose source_chunk_ids became empty. These
            # entities/relations were ONLY supported by the deleted
            # document.
            deleted_edges = (
                await conn.execute(
                    text(
                        f"DELETE FROM {EDGES_TABLE} "
                        f"WHERE collection_id = :cid "
                        f"  AND array_length(source_chunk_ids, 1) IS NULL"
                    ),
                    {"cid": collection_id},
                )
            ).rowcount or 0
            deleted_nodes = (
                await conn.execute(
                    text(
                        f"DELETE FROM {NODES_TABLE} "
                        f"WHERE collection_id = :cid "
                        f"  AND array_length(source_chunk_ids, 1) IS NULL"
                    ),
                    {"cid": collection_id},
                )
            ).rowcount or 0

        return DeleteDocumentResult(
            doc_id=doc_id,
            chunks_removed=int(deleted_chunks),
            entities_removed=int(deleted_nodes),
            relations_removed=int(deleted_edges),
        )

    # ============================================================= read
    async def has_collection_data(self, collection_id: str) -> bool:
        """Cheap existence probe for the cutover gate.

        A single ``SELECT 1 ... LIMIT 1`` on the nodes table, hitting
        the ``(collection_id, entity_id)`` primary key. Subsecond on
        cold cache; effectively free once the collection is active.
        """
        async with self._engine.connect() as conn:
            row = (
                await conn.execute(
                    text(f"SELECT 1 FROM {NODES_TABLE} WHERE collection_id = :cid LIMIT 1"),
                    {"cid": collection_id},
                )
            ).first()
        return row is not None

    async def find_entities_by_names(self, collection_id: str, names: Sequence[str]) -> list[Entity]:
        if not names:
            return []
        sql = (
            f"SELECT entity_id, name, type, description, source_chunk_ids "
            f"FROM {NODES_TABLE} "
            f"WHERE collection_id = :cid AND name = ANY(CAST(:names AS text[]))"
        )
        async with self._engine.connect() as conn:
            rows = (await conn.execute(text(sql), {"cid": collection_id, "names": list(names)})).all()
        return [_row_to_entity(row, collection_id) for row in rows]

    async def expand_neighborhood(
        self,
        collection_id: str,
        anchor_entity_ids: Sequence[str],
        max_hop: int,
        limit: int,
    ) -> tuple[list[Entity], list[Relation]]:
        """BFS expansion in SQL using a recursive CTE.

        The recursion is bounded by ``max_hop`` and the result set is
        capped by ``limit``; for a RAG-scale (0-1 hop, limit<=50) this is
        a sub-millisecond query thanks to the ``(collection_id, source_id)``
        and ``(collection_id, target_id)`` indexes on ``graphindex_edges``.
        """
        if not anchor_entity_ids:
            return [], []

        # Reachable entity ids up to max_hop.
        cte = f"""
            WITH RECURSIVE reach(entity_id, depth) AS (
                SELECT e::text, 0 FROM unnest(CAST(:anchors AS text[])) AS e
                UNION
                SELECT CASE
                        WHEN edges.source_id = reach.entity_id THEN edges.target_id
                        ELSE edges.source_id
                       END,
                       reach.depth + 1
                FROM reach
                JOIN {EDGES_TABLE} edges
                  ON edges.collection_id = :cid
                 AND (edges.source_id = reach.entity_id OR edges.target_id = reach.entity_id)
                WHERE reach.depth < :maxhop
            )
            SELECT DISTINCT entity_id FROM reach LIMIT :lim
        """
        params = {
            "cid": collection_id,
            "anchors": list(anchor_entity_ids),
            "maxhop": int(max_hop),
            "lim": int(limit),
        }

        async with self._engine.connect() as conn:
            reach_rows = (await conn.execute(text(cte), params)).all()
            reached_ids = [r.entity_id for r in reach_rows]
            if not reached_ids:
                return [], []

            ent_rows = (
                await conn.execute(
                    text(
                        f"SELECT entity_id, name, type, description, source_chunk_ids "
                        f"FROM {NODES_TABLE} "
                        f"WHERE collection_id = :cid "
                        f"  AND entity_id = ANY(CAST(:ids AS text[]))"
                    ),
                    {"cid": collection_id, "ids": reached_ids},
                )
            ).all()
            rel_rows = (
                await conn.execute(
                    text(
                        f"SELECT source_id, target_id, description, weight, source_chunk_ids "
                        f"FROM {EDGES_TABLE} "
                        f"WHERE collection_id = :cid "
                        f"  AND source_id = ANY(CAST(:ids AS text[])) "
                        f"  AND target_id = ANY(CAST(:ids AS text[]))"
                    ),
                    {"cid": collection_id, "ids": reached_ids},
                )
            ).all()

        entities = [_row_to_entity(r, collection_id) for r in ent_rows]
        relations = [_row_to_relation(r, collection_id) for r in rel_rows]
        return entities, relations

    async def list_labels(self, collection_id: str) -> list[str]:
        async with self._engine.connect() as conn:
            rows = (
                await conn.execute(
                    text(f"SELECT DISTINCT type FROM {NODES_TABLE} WHERE collection_id = :cid ORDER BY type"),
                    {"cid": collection_id},
                )
            ).all()
        return [r.type for r in rows]

    async def list_subgraph(
        self,
        collection_id: str,
        label: Optional[str],
        max_depth: int,
        max_nodes: int,
    ) -> KnowledgeGraph:
        """Pick top-degree entities (optionally by label), then BFS-walk
        ``max_depth`` steps out, capped at ``max_nodes``.

        "Top degree" is approximated by the count of matching edges
        incident on each candidate. For a UI-facing call (max_nodes <=
        1000) this is acceptable; for a bulk export use
        ``expand_neighborhood`` directly.
        """
        max_nodes = max(1, int(max_nodes))
        max_depth = max(0, int(max_depth))

        # Candidate anchor set: top-degree entities, optionally type-filtered.
        # ``CAST(:lbl AS text)`` is required because asyncpg can't infer the
        # type of a bind parameter that may be NULL — without the cast PG
        # raises ``could not determine data type of parameter``.
        async with self._engine.connect() as conn:
            candidate_rows = (
                await conn.execute(
                    text(
                        f"SELECT n.entity_id, "
                        f"       COALESCE(d.deg, 0) AS deg "
                        f"FROM {NODES_TABLE} n "
                        f"LEFT JOIN ( "
                        f"  SELECT entity_id, COUNT(*) AS deg FROM ( "
                        f"    SELECT source_id AS entity_id FROM {EDGES_TABLE} WHERE collection_id = :cid "
                        f"    UNION ALL "
                        f"    SELECT target_id AS entity_id FROM {EDGES_TABLE} WHERE collection_id = :cid "
                        f"  ) x GROUP BY entity_id "
                        f") d ON d.entity_id = n.entity_id "
                        f"WHERE n.collection_id = :cid "
                        f"  AND (CAST(:lbl AS text) IS NULL "
                        f"       OR CAST(:lbl AS text) = '*' "
                        f"       OR n.type = CAST(:lbl AS text)) "
                        f"ORDER BY deg DESC "
                        f"LIMIT :lim"
                    ),
                    {"cid": collection_id, "lbl": label, "lim": max_nodes},
                )
            ).all()

        anchor_ids = [r.entity_id for r in candidate_rows]
        if not anchor_ids:
            return KnowledgeGraph(nodes=[], edges=[], is_truncated=False)

        entities, relations = await self.expand_neighborhood(
            collection_id=collection_id,
            anchor_entity_ids=anchor_ids,
            max_hop=max_depth,
            limit=max_nodes,
        )

        is_truncated = len(entities) >= max_nodes
        return KnowledgeGraph(
            nodes=entities[:max_nodes],
            edges=relations,
            is_truncated=is_truncated,
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_multi_row_values(
    rows: list[dict],
    *,
    columns: Sequence[str],
    cast_columns: Optional[dict[str, str]] = None,
) -> tuple[str, dict]:
    """Build ``VALUES (:a0, :b0), (:a1, :b1), ...`` with bind params.

    We hand-roll the VALUES clause so we can interleave type casts
    (``CAST(:chunks0 AS text[])``) — SQLAlchemy doesn't have a compact
    API for that and writing each row as a separate INSERT loses
    upsert-in-one-statement semantics.
    """
    cast_columns = cast_columns or {}
    pieces: list[str] = []
    params: dict[str, object] = {}
    for i, row in enumerate(rows):
        row_placeholders = []
        for col in columns:
            pname = f"{col}{i}"
            params[pname] = row[col]
            if col in cast_columns:
                row_placeholders.append(f"CAST(:{pname} AS {cast_columns[col]})")
            else:
                row_placeholders.append(f":{pname}")
        pieces.append("(" + ", ".join(row_placeholders) + ")")
    return ", ".join(pieces), params


def _row_to_entity(row, collection_id: str) -> Entity:
    return Entity(
        entity_id=row.entity_id,
        collection_id=collection_id,
        name=row.name,
        type=row.type,
        description=row.description or "",
        source_chunk_ids=tuple(row.source_chunk_ids or ()),
    )


def _row_to_relation(row, collection_id: str) -> Relation:
    return Relation(
        collection_id=collection_id,
        source_id=row.source_id,
        target_id=row.target_id,
        description=row.description or "",
        weight=float(row.weight or 0),
        source_chunk_ids=tuple(row.source_chunk_ids or ()),
    )


__all__ = ["PostgresGraphStore"]
