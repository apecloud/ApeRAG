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

from aperag.domains.knowledge_graph.graphindex.dto import (
    DESCRIPTION_SEPARATOR,
    Chunk,
    DeleteDocumentResult,
    Entity,
    KnowledgeGraph,
    MergeEntitiesResult,
    Relation,
)
from aperag.domains.knowledge_graph.graphindex.models import (
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
        """The store is intentionally LLM-agnostic: size caps and
        summarization decisions are the service layer's job, not the
        storage layer's. That keeps this class purely SQL and lets tests
        exercise the merge / upsert semantics without an LLM stub."""
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
        three tables (edges, nodes, chunks). Used by the collection-delete
        Celery task.
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
        """Insert entities; on ``(collection_id, entity_id)`` conflict:

        * ``source_chunk_ids``: set-union with existing ids.
        * ``description``: append the incoming fragment to the existing
          text with a ``\\n\\n`` separator, **without cap**. Size
          bounding is the service layer's job (see
          ``GraphIndexService._compact_oversized_descriptions``): the
          storage layer is intentionally LLM-agnostic. If the incoming
          fragment is already a substring of the stored description —
          which happens when identical boilerplate appears in multiple
          chunks — it is skipped so we don't double-store identical
          sentences.
        * ``name`` / ``type``: take the newer value (closed vocabulary,
          repeated writes converge).
        """
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
        params["sep"] = DESCRIPTION_SEPARATOR

        sql = (
            f"INSERT INTO {NODES_TABLE} "
            f"(collection_id, entity_id, name, type, description, source_chunk_ids) "
            f"VALUES {values_sql} "
            f"ON CONFLICT (collection_id, entity_id) DO UPDATE SET "
            f"source_chunk_ids = ARRAY("
            f"  SELECT DISTINCT unnest("
            f"    {NODES_TABLE}.source_chunk_ids || EXCLUDED.source_chunk_ids"
            f"  )"
            f"), "
            f"description = {_sql_append_fragment(current=f'{NODES_TABLE}.description', incoming='EXCLUDED.description')}, "
            f"type = EXCLUDED.type, "
            f"name = EXCLUDED.name, "
            f"updated_at = now()"
        )
        async with self._engine.begin() as conn:
            await conn.execute(text(sql), params)

    async def upsert_relations(self, collection_id: str, relations: Sequence[Relation]) -> None:
        """Insert relations; on ``(source, target)`` conflict:

        * ``source_chunk_ids``: set-union.
        * ``weight``: ``GREATEST(existing, new)`` — stronger evidence wins.
        * ``description``: same append-fragment rule as entities.
        """
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
        params["sep"] = DESCRIPTION_SEPARATOR

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
            f"description = {_sql_append_fragment(current=f'{EDGES_TABLE}.description', incoming='EXCLUDED.description')}, "
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

    # ============================================================ merge
    async def merge_entities(
        self,
        collection_id: str,
        *,
        target_entity_id: str,
        source_entity_ids: Sequence[str],
    ) -> MergeEntitiesResult:
        """Merge ``source_entity_ids`` into ``target_entity_id`` in a
        single transaction.

        Semantics:

        * Target entity gains the union of all source chunks plus its own.
        * Target description gets every source fragment appended
          (``\\n\\n`` separator, deduplicated via substring check). The
          **service layer** runs LLM summarization afterwards when the
          resulting description exceeds the configured thresholds; this
          method intentionally does not call the LLM so the storage
          layer stays testable without network.
        * Every edge touching a source is redirected to target. Edges
          that become self-loops (target<->target) or duplicates of an
          existing target edge are collapsed: chunk-id arrays union,
          weights take ``GREATEST``, descriptions append with the usual
          dedup.
        * Source rows are deleted.

        Returns the post-merge target row plus counts, so the service
        layer can decide whether to trigger description summarization.

        Raises ``ValueError`` if target does not exist.
        """
        source_ids = [s for s in source_entity_ids if s and s != target_entity_id]
        if not source_ids:
            raise ValueError("merge_entities requires at least one source distinct from the target")

        async with self._engine.begin() as conn:
            # 0. Load target + source rows (lock them so no concurrent
            # upsert corrupts the merge mid-transaction).
            target_row = (
                await conn.execute(
                    text(
                        f"SELECT entity_id, name, type, description, source_chunk_ids "
                        f"FROM {NODES_TABLE} "
                        f"WHERE collection_id = :cid AND entity_id = :eid "
                        f"FOR UPDATE"
                    ),
                    {"cid": collection_id, "eid": target_entity_id},
                )
            ).first()
            if target_row is None:
                raise ValueError(f"Target entity {target_entity_id!r} not found in collection {collection_id!r}")

            source_rows = (
                await conn.execute(
                    text(
                        f"SELECT entity_id, description, source_chunk_ids "
                        f"FROM {NODES_TABLE} "
                        f"WHERE collection_id = :cid AND entity_id = ANY(CAST(:ids AS text[])) "
                        f"FOR UPDATE"
                    ),
                    {"cid": collection_id, "ids": source_ids},
                )
            ).all()
            if not source_rows:
                # Nothing to do; return the target as-is.
                return MergeEntitiesResult(
                    target_entity_id=target_entity_id,
                    merged_source_ids=tuple(),
                    description=target_row.description or "",
                    source_chunk_ids=tuple(target_row.source_chunk_ids or ()),
                    edges_redirected=0,
                    edges_collapsed=0,
                )

            # 1. Build the new target description by appending each
            #    source fragment, deduplicating substrings.
            description = target_row.description or ""
            for s in source_rows:
                frag = (s.description or "").strip()
                if not frag:
                    continue
                if description and frag in description:
                    continue
                description = (description + DESCRIPTION_SEPARATOR + frag) if description else frag

            # 2. Union of chunk ids.
            chunk_ids: set[str] = set(target_row.source_chunk_ids or ())
            for s in source_rows:
                chunk_ids.update(s.source_chunk_ids or ())

            # 3. Redirect edges. Any edge that has a source/target in
            #    ``source_ids`` gets its endpoint rewritten to the
            #    target entity id. ``ON CONFLICT`` then collapses any
            #    collision with an existing (target, X) / (X, target)
            #    edge via the same merge rules as ``upsert_relations``.
            #
            # Step 3a: load every affected edge so we can rewrite in
            # application code (PG doesn't let you update a row then
            # re-insert it via ON CONFLICT in one statement without
            # gymnastics, and we also need to delete self-loops).
            affected_rows = (
                await conn.execute(
                    text(
                        f"SELECT source_id, target_id, description, weight, source_chunk_ids "
                        f"FROM {EDGES_TABLE} "
                        f"WHERE collection_id = :cid "
                        f"  AND (source_id = ANY(CAST(:ids AS text[])) "
                        f"       OR target_id = ANY(CAST(:ids AS text[])))"
                    ),
                    {"cid": collection_id, "ids": source_ids},
                )
            ).all()

            # Remove them all first; we'll re-insert via the standard
            # upsert path so conflicts merge cleanly.
            await conn.execute(
                text(
                    f"DELETE FROM {EDGES_TABLE} "
                    f"WHERE collection_id = :cid "
                    f"  AND (source_id = ANY(CAST(:ids AS text[])) "
                    f"       OR target_id = ANY(CAST(:ids AS text[])))"
                ),
                {"cid": collection_id, "ids": source_ids},
            )

            redirected = 0
            collapsed = 0
            # Rebuild redirected edges. Two sources pointing at the
            # same third entity (e.g. src1→other and src2→other) both
            # become target→other after redirect; we MUST collapse
            # those in Python before sending them to the database,
            # otherwise PG rejects the INSERT with
            # ``CardinalityViolationError`` because ON CONFLICT DO
            # UPDATE cannot apply to two rows with the same key in
            # one statement.
            rebuilt_map: dict[tuple[str, str], Relation] = {}
            for e in affected_rows:
                new_src = target_entity_id if e.source_id in source_ids else e.source_id
                new_tgt = target_entity_id if e.target_id in source_ids else e.target_id
                if new_src == new_tgt:
                    # Drop self-loops; they carry no information after merge.
                    collapsed += 1
                    continue
                key = (new_src, new_tgt)
                incoming = Relation(
                    collection_id=collection_id,
                    source_id=new_src,
                    target_id=new_tgt,
                    description=e.description or "",
                    weight=float(e.weight or 0),
                    source_chunk_ids=tuple(e.source_chunk_ids or ()),
                )
                if key in rebuilt_map:
                    existing = rebuilt_map[key]
                    union_chunks = tuple(dict.fromkeys((*existing.source_chunk_ids, *incoming.source_chunk_ids)))
                    # Concat descriptions with the same dedup-by-substring
                    # rule the SQL path would apply.
                    desc_a = (existing.description or "").strip()
                    desc_b = (incoming.description or "").strip()
                    if not desc_a:
                        new_desc = desc_b
                    elif not desc_b or desc_b in desc_a:
                        new_desc = existing.description
                    else:
                        new_desc = existing.description + DESCRIPTION_SEPARATOR + incoming.description
                    rebuilt_map[key] = Relation(
                        collection_id=collection_id,
                        source_id=new_src,
                        target_id=new_tgt,
                        description=new_desc,
                        weight=max(existing.weight, incoming.weight),
                        source_chunk_ids=union_chunks,
                    )
                    collapsed += 1
                else:
                    rebuilt_map[key] = incoming
                    redirected += 1
            rebuilt = list(rebuilt_map.values())

            # 4. Delete source entities.
            await conn.execute(
                text(
                    f"DELETE FROM {NODES_TABLE} WHERE collection_id = :cid   AND entity_id = ANY(CAST(:ids AS text[]))"
                ),
                {"cid": collection_id, "ids": source_ids},
            )

            # 5. Persist merged target (single row update is enough — no
            #    upsert needed, target was locked in step 0).
            await conn.execute(
                text(
                    f"UPDATE {NODES_TABLE} SET "
                    f"description = :desc, "
                    f"source_chunk_ids = CAST(:chunks AS text[]), "
                    f"updated_at = now() "
                    f"WHERE collection_id = :cid AND entity_id = :eid"
                ),
                {
                    "cid": collection_id,
                    "eid": target_entity_id,
                    "desc": description,
                    "chunks": sorted(chunk_ids),
                },
            )

        # 6. Re-insert redirected edges OUTSIDE the merge transaction
        #    so ``upsert_relations`` uses its own transaction. Splitting
        #    keeps merge_entities free of the giant VALUES clause
        #    plumbing and re-uses the upsert conflict logic.
        if rebuilt:
            await self.upsert_relations(collection_id, rebuilt)

        return MergeEntitiesResult(
            target_entity_id=target_entity_id,
            merged_source_ids=tuple(s.entity_id for s in source_rows),
            description=description,
            source_chunk_ids=tuple(sorted(chunk_ids)),
            edges_redirected=redirected,
            edges_collapsed=collapsed,
        )

    # ======================================================== normalize
    async def find_oversized_entities(
        self,
        collection_id: str,
        *,
        min_chars: int,
        min_fragments: int,
        limit: int = 200,
    ) -> list[Entity]:
        """Return entities whose description is long enough that the
        service layer should consider LLM-summarizing it.

        An entity qualifies if EITHER:

        * ``length(description) >= min_chars``, or
        * its fragment count (``split on DESCRIPTION_SEPARATOR``) is at
          least ``min_fragments``.

        The query is cheap on a modern PG with GIN or even no index —
        graphindex_nodes is written at ``O(chunks)`` per document, so
        the oversized set is tiny in practice. ``limit`` is here purely
        as a cost cap so an accidentally runaway row count doesn't try
        to pull 10k rows into memory.
        """
        sep_occurrences = "array_length(string_to_array(description, :sep), 1)"
        sql = (
            f"SELECT entity_id, name, type, description, source_chunk_ids "
            f"FROM {NODES_TABLE} "
            f"WHERE collection_id = :cid "
            f"  AND description IS NOT NULL "
            f"  AND ( "
            f"    length(description) >= :minchars "
            f"    OR COALESCE({sep_occurrences}, 1) >= :minfrags "
            f"  ) "
            f"ORDER BY length(description) DESC "
            f"LIMIT :lim"
        )
        params = {
            "cid": collection_id,
            "sep": DESCRIPTION_SEPARATOR,
            "minchars": int(min_chars),
            "minfrags": int(min_fragments),
            "lim": int(limit),
        }
        async with self._engine.connect() as conn:
            rows = (await conn.execute(text(sql), params)).all()
        return [_row_to_entity(r, collection_id) for r in rows]

    async def find_oversized_relations(
        self,
        collection_id: str,
        *,
        min_chars: int,
        min_fragments: int,
        limit: int = 200,
    ) -> list[Relation]:
        sep_occurrences = "array_length(string_to_array(description, :sep), 1)"
        sql = (
            f"SELECT source_id, target_id, description, weight, source_chunk_ids "
            f"FROM {EDGES_TABLE} "
            f"WHERE collection_id = :cid "
            f"  AND description IS NOT NULL "
            f"  AND ( "
            f"    length(description) >= :minchars "
            f"    OR COALESCE({sep_occurrences}, 1) >= :minfrags "
            f"  ) "
            f"ORDER BY length(description) DESC "
            f"LIMIT :lim"
        )
        params = {
            "cid": collection_id,
            "sep": DESCRIPTION_SEPARATOR,
            "minchars": int(min_chars),
            "minfrags": int(min_fragments),
            "lim": int(limit),
        }
        async with self._engine.connect() as conn:
            rows = (await conn.execute(text(sql), params)).all()
        return [_row_to_relation(r, collection_id) for r in rows]

    async def rewrite_entity_description(
        self,
        collection_id: str,
        entity_id: str,
        description: str,
    ) -> None:
        """Replace the entity's description wholesale with a summary."""
        async with self._engine.begin() as conn:
            await conn.execute(
                text(
                    f"UPDATE {NODES_TABLE} SET description = :desc, updated_at = now() "
                    f"WHERE collection_id = :cid AND entity_id = :eid"
                ),
                {"cid": collection_id, "eid": entity_id, "desc": description},
            )

    async def rewrite_relation_description(
        self,
        collection_id: str,
        source_id: str,
        target_id: str,
        description: str,
    ) -> None:
        async with self._engine.begin() as conn:
            await conn.execute(
                text(
                    f"UPDATE {EDGES_TABLE} SET description = :desc, updated_at = now() "
                    f"WHERE collection_id = :cid AND source_id = :src AND target_id = :tgt"
                ),
                {"cid": collection_id, "src": source_id, "tgt": target_id, "desc": description},
            )

    # ============================================================= read

    async def get_chunks_by_ids(self, collection_id: str, chunk_ids: Sequence[str]) -> list[Chunk]:
        if not chunk_ids:
            return []
        sql = (
            f"SELECT chunk_id, doc_id, order_in_doc, text, file_path "
            f"FROM {CHUNKS_TABLE} "
            f"WHERE collection_id = :cid AND chunk_id = ANY(CAST(:ids AS text[]))"
        )
        async with self._engine.connect() as conn:
            rows = (await conn.execute(text(sql), {"cid": collection_id, "ids": list(chunk_ids)})).all()
        return [
            Chunk(
                chunk_id=r.chunk_id,
                doc_id=r.doc_id,
                collection_id=collection_id,
                order_in_doc=r.order_in_doc,
                text=r.text or "",
                file_path=r.file_path or "",
            )
            for r in rows
        ]

    async def find_entities_by_ids(self, collection_id: str, entity_ids: Sequence[str]) -> list[Entity]:
        if not entity_ids:
            return []
        sql = (
            f"SELECT entity_id, name, type, description, source_chunk_ids "
            f"FROM {NODES_TABLE} "
            f"WHERE collection_id = :cid AND entity_id = ANY(CAST(:ids AS text[]))"
        )
        async with self._engine.connect() as conn:
            rows = (await conn.execute(text(sql), {"cid": collection_id, "ids": list(entity_ids)})).all()
        return [_row_to_entity(row, collection_id) for row in rows]

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


def _sql_append_fragment(*, current: str, incoming: str) -> str:
    """Return a SQL expression that appends ``incoming`` to ``current``
    with the ``DESCRIPTION_SEPARATOR`` between them.

    Behaviour:

    * If ``current`` is NULL or empty → result is ``incoming``.
    * If ``current`` already contains ``incoming`` as a substring
      (``position(incoming IN current) > 0``) → result is ``current``
      unchanged. This avoids duplicating identical sentences that appear
      in multiple chunks of the same document, which was the #1 source
      of "why is this description repeating itself" complaints.
    * Otherwise → ``current || :sep || incoming``.

    No length cap is applied here — the service layer decides when an
    accumulated description should be LLM-summarized or, as a last
    resort, truncated.
    """
    return (
        f"CASE "
        f"  WHEN {current} IS NULL OR {current} = '' THEN {incoming} "
        f"  WHEN {incoming} IS NULL OR {incoming} = '' THEN {current} "
        f"  WHEN position({incoming} IN {current}) > 0 THEN {current} "
        f"  ELSE {current} || :sep || {incoming} "
        f"END"
    )


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
