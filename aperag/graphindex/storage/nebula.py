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

"""Nebula Graph implementation of the ``GraphStore`` Protocol.

Uses ``nebula3-python`` (sync ConnectionPool) with ``asyncio.to_thread``
wrappers for every public method so the async Protocol is honoured
without forcing every Nebula SDK call to go through a hand-rolled async
wrapper.

Multi-tenancy: one Nebula SPACE per ApeRAG collection, named
``{space_prefix}_{collection_id}``. Spaces are cheap in Nebula (similar
to Postgres schemas) and give natural isolation — ``DROP SPACE`` is the
fastest way to nuke a collection's graph.

Tag / Edge Type schema:
  - TAG ``entity(entity_id string, name string, type string,
          description string, source_chunk_ids string)``
  - TAG ``chunk(doc_id string, order_in_doc int, text string,
          file_path string)``
  - EDGE TYPE ``relates_to(description string, weight double,
               source_chunk_ids string)``

``source_chunk_ids`` is stored as a comma-separated string because
Nebula does not have a list<string> property type in all versions.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import threading
from typing import Any, Optional, Sequence

from aperag.graphindex.dto import (
    DESCRIPTION_SEPARATOR,
    Chunk,
    DeleteDocumentResult,
    Entity,
    KnowledgeGraph,
    MergeEntitiesResult,
    Relation,
)

logger = logging.getLogger(__name__)

_ENTITY_TAG = "entity"
_CHUNK_TAG = "chunk"
_EDGE_TYPE = "relates_to"


def _escape(s: str) -> str:
    """Escape a string for nGQL string literals."""
    return s.replace("\\", "\\\\").replace('"', '\\"').replace("'", "\\'")


def _encode_chunk_ids(ids: Sequence[str]) -> str:
    return json.dumps(list(ids))


def _decode_chunk_ids(raw: str) -> tuple[str, ...]:
    if not raw:
        return ()
    try:
        return tuple(json.loads(raw))
    except (json.JSONDecodeError, TypeError):
        return tuple(raw.split(",")) if raw else ()


def _space_name(prefix: str, collection_id: str) -> str:
    safe_id = re.sub(r"[^a-zA-Z0-9_]", "_", collection_id)
    return f"{prefix}_{safe_id}"


class NebulaGraphStore:
    """``GraphStore`` backed by Nebula Graph.

    Thread safety: the ``ConnectionPool`` is thread-safe; each public
    method acquires a session from the pool inside ``asyncio.to_thread``.
    """

    _pool: Any = None
    _lock = threading.Lock()

    def __init__(
        self,
        *,
        hosts: str,
        username: str = "root",
        password: str = "",
        space_prefix: str = "aperag",
    ) -> None:
        self._hosts = hosts
        self._username = username
        self._password = password
        self._space_prefix = space_prefix

    def _get_pool(self):
        if self._pool is not None:
            return self._pool
        with self._lock:
            if self._pool is not None:
                return self._pool
            from nebula3.Config import Config as NebulaConfig
            from nebula3.gclient.net import ConnectionPool

            config = NebulaConfig()
            config.max_connection_pool_size = 20
            config.timeout = 60000

            host_pairs = []
            for part in self._hosts.split(","):
                part = part.strip()
                if ":" in part:
                    h, p = part.rsplit(":", 1)
                    host_pairs.append((h, int(p)))
                else:
                    host_pairs.append((part, 9669))

            pool = ConnectionPool()
            pool.init(host_pairs, config)
            self._pool = pool
            return pool

    def _execute(self, space: str, stmt: str) -> Any:
        """Run a single nGQL statement in a sync session."""
        pool = self._get_pool()
        session = pool.get_session(self._username, self._password)
        try:
            if space:
                r = session.execute(f"USE `{space}`")
                if not r.is_succeeded():
                    raise RuntimeError(f"Nebula USE failed: {r.error_msg()}")
            result = session.execute(stmt)
            if not result.is_succeeded():
                raise RuntimeError(f"Nebula query failed: {result.error_msg()}\nStatement: {stmt}")
            return result
        finally:
            session.release()

    def _execute_multi(self, space: str, stmts: list[str]) -> None:
        pool = self._get_pool()
        session = pool.get_session(self._username, self._password)
        try:
            if space:
                r = session.execute(f"USE `{space}`")
                if not r.is_succeeded():
                    raise RuntimeError(f"Nebula USE failed: {r.error_msg()}")
            for stmt in stmts:
                result = session.execute(stmt)
                if not result.is_succeeded():
                    raise RuntimeError(f"Nebula query failed: {result.error_msg()}\nStatement: {stmt}")
        finally:
            session.release()

    def _space(self, collection_id: str) -> str:
        return _space_name(self._space_prefix, collection_id)

    # =========================================================== schema
    async def ensure_schema(self) -> None:
        pass

    def _ensure_space(self, collection_id: str) -> str:
        space = self._space(collection_id)
        self._execute(
            "",
            f"CREATE SPACE IF NOT EXISTS `{space}` (vid_type=FIXED_STRING(128), partition_num=1, replica_factor=1)",
        )
        import time

        time.sleep(1)

        stmts = [
            f"CREATE TAG IF NOT EXISTS `{_ENTITY_TAG}`("
            f"entity_id string, name string, type string, "
            f"description string, source_chunk_ids string)",
            f"CREATE TAG IF NOT EXISTS `{_CHUNK_TAG}`("
            f"chunk_id string, doc_id string, order_in_doc int, "
            f"text string, file_path string)",
            f"CREATE EDGE TYPE IF NOT EXISTS `{_EDGE_TYPE}`("
            f"description string, weight double, source_chunk_ids string)",
        ]
        self._execute_multi(space, stmts)
        import time

        time.sleep(1)
        return space

    async def drop_collection(self, collection_id: str) -> None:
        space = self._space(collection_id)
        await asyncio.to_thread(self._execute, "", f"DROP SPACE IF EXISTS `{space}`")
        logger.info("nebula graphstore: dropped space %s", space)

    # ============================================================ write
    async def upsert_chunks(self, collection_id: str, chunks: Sequence[Chunk]) -> None:
        if not chunks:
            return

        def _do():
            space = self._ensure_space(collection_id)
            for c in chunks:
                vid = _escape(c.chunk_id)
                stmt = (
                    f"INSERT VERTEX IF NOT EXISTS `{_CHUNK_TAG}`"
                    f"(chunk_id, doc_id, order_in_doc, text, file_path) "
                    f'VALUES "{vid}":('
                    f'"{_escape(c.chunk_id)}", "{_escape(c.doc_id)}", '
                    f'{c.order_in_doc}, "{_escape(c.text)}", "{_escape(c.file_path or "")}")'
                )
                self._execute(space, stmt)

        await asyncio.to_thread(_do)

    async def upsert_entities(self, collection_id: str, entities: Sequence[Entity]) -> None:
        if not entities:
            return

        def _do():
            space = self._ensure_space(collection_id)
            for e in entities:
                vid = _escape(e.entity_id)
                # Try to read existing description for append logic
                try:
                    result = self._execute(
                        space,
                        f'FETCH PROP ON `{_ENTITY_TAG}` "{vid}" '
                        f"YIELD properties(vertex).description AS d, "
                        f"properties(vertex).source_chunk_ids AS c",
                    )
                    existing_desc = ""
                    existing_chunks: list[str] = []
                    if result.row_size() > 0:
                        existing_desc = (
                            str(result.row_values(0)[0].as_string()) if result.row_values(0)[0].is_string() else ""
                        )
                        raw_c = str(result.row_values(0)[1].as_string()) if result.row_values(0)[1].is_string() else ""
                        existing_chunks = list(_decode_chunk_ids(raw_c))
                except Exception:
                    existing_desc = ""
                    existing_chunks = []

                new_desc = e.description or ""
                if existing_desc and new_desc and new_desc not in existing_desc:
                    final_desc = existing_desc + DESCRIPTION_SEPARATOR + new_desc
                elif existing_desc:
                    final_desc = existing_desc
                else:
                    final_desc = new_desc

                all_chunks = list(dict.fromkeys(existing_chunks + list(e.source_chunk_ids)))
                all_chunks_str = _escape(_encode_chunk_ids(all_chunks))

                stmt = (
                    f"INSERT VERTEX `{_ENTITY_TAG}`"
                    f"(entity_id, name, type, description, source_chunk_ids) "
                    f'VALUES "{vid}":('
                    f'"{_escape(e.entity_id)}", "{_escape(e.name)}", '
                    f'"{_escape(e.type)}", "{_escape(final_desc)}", '
                    f'"{all_chunks_str}")'
                )
                self._execute(space, stmt)

        await asyncio.to_thread(_do)

    async def upsert_relations(self, collection_id: str, relations: Sequence[Relation]) -> None:
        if not relations:
            return

        def _do():
            space = self._ensure_space(collection_id)
            for r in relations:
                src = _escape(r.source_id)
                tgt = _escape(r.target_id)
                chunks_str = _escape(_encode_chunk_ids(r.source_chunk_ids))
                stmt = (
                    f"INSERT EDGE `{_EDGE_TYPE}`"
                    f"(description, weight, source_chunk_ids) "
                    f'VALUES "{src}"->"{tgt}":('
                    f'"{_escape(r.description)}", {float(r.weight)}, '
                    f'"{chunks_str}")'
                )
                self._execute(space, stmt)

        await asyncio.to_thread(_do)

    # ============================================================ merge
    async def merge_entities(
        self,
        collection_id: str,
        *,
        target_entity_id: str,
        source_entity_ids: Sequence[str],
    ) -> MergeEntitiesResult:
        source_ids = [s for s in source_entity_ids if s and s != target_entity_id]
        if not source_ids:
            raise ValueError("merge_entities requires at least one source distinct from the target")

        def _do() -> MergeEntitiesResult:
            space = self._ensure_space(collection_id)
            # Load target
            result = self._execute(
                space,
                f'FETCH PROP ON `{_ENTITY_TAG}` "{_escape(target_entity_id)}" '
                f"YIELD properties(vertex).description AS d, "
                f"properties(vertex).source_chunk_ids AS c",
            )
            if result.row_size() == 0:
                raise ValueError(f"Target entity {target_entity_id!r} not found")
            desc = result.row_values(0)[0].as_string() if result.row_values(0)[0].is_string() else ""
            chunks = set(
                _decode_chunk_ids(result.row_values(0)[1].as_string() if result.row_values(0)[1].is_string() else "")
            )

            merged = []
            for sid in source_ids:
                try:
                    r = self._execute(
                        space,
                        f'FETCH PROP ON `{_ENTITY_TAG}` "{_escape(sid)}" '
                        f"YIELD properties(vertex).description AS d, "
                        f"properties(vertex).source_chunk_ids AS c",
                    )
                    if r.row_size() > 0:
                        merged.append(sid)
                        frag = r.row_values(0)[0].as_string() if r.row_values(0)[0].is_string() else ""
                        if frag and frag not in desc:
                            desc = (desc + DESCRIPTION_SEPARATOR + frag) if desc else frag
                        chunks.update(
                            _decode_chunk_ids(r.row_values(0)[1].as_string() if r.row_values(0)[1].is_string() else "")
                        )
                except Exception:
                    continue

            if not merged:
                return MergeEntitiesResult(
                    target_entity_id=target_entity_id,
                    merged_source_ids=(),
                    description=desc,
                    source_chunk_ids=tuple(sorted(chunks)),
                    edges_redirected=0,
                    edges_collapsed=0,
                )

            # Redirect edges: collect all affected edges into an
            # in-memory map keyed by (new_src, new_tgt), then collapse
            # duplicates matching upsert_relations semantics (GREATEST
            # weight, description append with dedup, union chunk ids)
            # before writing back. This mirrors the PG backend's
            # Python-side dedup that avoids duplicate-key issues.
            edges_redirected = 0
            edges_collapsed = 0
            # Map (new_src, new_tgt) -> (desc, weight, chunk_ids_set)
            redirected_map: dict[tuple[str, str], tuple[str, float, set[str]]] = {}

            for sid in merged:
                for direction in ("outgoing", "incoming"):
                    try:
                        if direction == "outgoing":
                            r = self._execute(
                                space,
                                f'GO FROM "{_escape(sid)}" OVER `{_EDGE_TYPE}` '
                                f"YIELD `{_EDGE_TYPE}`._dst AS dst, "
                                f"`{_EDGE_TYPE}`.description AS d, "
                                f"`{_EDGE_TYPE}`.weight AS w, "
                                f"`{_EDGE_TYPE}`.source_chunk_ids AS c",
                            )
                        else:
                            r = self._execute(
                                space,
                                f'GO FROM "{_escape(sid)}" OVER `{_EDGE_TYPE}` REVERSELY '
                                f"YIELD `{_EDGE_TYPE}`._src AS src, "
                                f"`{_EDGE_TYPE}`.description AS d, "
                                f"`{_EDGE_TYPE}`.weight AS w, "
                                f"`{_EDGE_TYPE}`.source_chunk_ids AS c",
                            )
                        for i in range(r.row_size()):
                            row = r.row_values(i)
                            other = row[0].as_string() if row[0].is_string() else ""
                            if direction == "outgoing":
                                new_src, new_tgt = target_entity_id, other
                            else:
                                new_src, new_tgt = other, target_entity_id
                            if new_src == new_tgt or other in source_ids:
                                edges_collapsed += 1
                                continue
                            key = (new_src, new_tgt)
                            edge_desc = row[1].as_string() if row[1].is_string() else ""
                            edge_w = float(row[2].as_double()) if row[2].is_double() else 0.0
                            edge_c = set(_decode_chunk_ids(row[3].as_string() if row[3].is_string() else ""))

                            if key in redirected_map:
                                ex_desc, ex_w, ex_c = redirected_map[key]
                                # Collapse: append description (with dedup),
                                # GREATEST weight, union chunk ids
                                if edge_desc and edge_desc not in ex_desc:
                                    ex_desc = (ex_desc + DESCRIPTION_SEPARATOR + edge_desc) if ex_desc else edge_desc
                                redirected_map[key] = (ex_desc, max(ex_w, edge_w), ex_c | edge_c)
                                edges_collapsed += 1
                            else:
                                redirected_map[key] = (edge_desc, edge_w, edge_c)
                                edges_redirected += 1
                    except Exception:
                        logger.exception("nebula merge: edge redirect failed for source %s", sid)

            # Write collapsed edges
            for (new_src, new_tgt), (e_desc, e_w, e_c) in redirected_map.items():
                try:
                    self._execute(
                        space,
                        f"INSERT EDGE `{_EDGE_TYPE}`(description, weight, source_chunk_ids) "
                        f'VALUES "{_escape(new_src)}"->"{_escape(new_tgt)}":('
                        f'"{_escape(e_desc)}", {float(e_w)}, '
                        f'"{_escape(_encode_chunk_ids(sorted(e_c)))}")',
                    )
                except Exception:
                    logger.exception("nebula merge: failed to write redirected edge %s->%s", new_src, new_tgt)

            # Delete source vertices (and their original edges)
            for sid in merged:
                self._execute(space, f'DELETE VERTEX "{_escape(sid)}" WITH EDGE')

            # Update target
            self._execute(
                space,
                f'UPDATE VERTEX ON `{_ENTITY_TAG}` "{_escape(target_entity_id)}" '
                f'SET description = "{_escape(desc)}", '
                f'source_chunk_ids = "{_escape(_encode_chunk_ids(sorted(chunks)))}"',
            )

            return MergeEntitiesResult(
                target_entity_id=target_entity_id,
                merged_source_ids=tuple(merged),
                description=desc,
                source_chunk_ids=tuple(sorted(chunks)),
                edges_redirected=edges_redirected,
                edges_collapsed=edges_collapsed,
            )

        return await asyncio.to_thread(_do)

    # ======================================================== normalize
    async def find_oversized_entities(
        self, collection_id: str, *, min_chars: int, min_fragments: int, limit: int = 200
    ) -> list[Entity]:
        def _do() -> list[Entity]:
            space = self._space(collection_id)
            try:
                result = self._execute(
                    space,
                    f"LOOKUP ON `{_ENTITY_TAG}` "
                    f"YIELD properties(vertex).entity_id AS eid, "
                    f"properties(vertex).name AS name, "
                    f"properties(vertex).type AS type, "
                    f"properties(vertex).description AS desc, "
                    f"properties(vertex).source_chunk_ids AS chunks "
                    f"| ORDER BY $-.desc DESC | LIMIT {limit}",
                )
            except Exception:
                return []
            out = []
            for i in range(result.row_size()):
                row = result.row_values(i)
                d = row[3].as_string() if row[3].is_string() else ""
                frags = len(d.split(DESCRIPTION_SEPARATOR)) if d else 0
                if len(d) >= min_chars or frags >= min_fragments:
                    out.append(
                        Entity(
                            entity_id=row[0].as_string(),
                            collection_id=collection_id,
                            name=row[1].as_string() if row[1].is_string() else "",
                            type=row[2].as_string() if row[2].is_string() else "",
                            description=d,
                            source_chunk_ids=_decode_chunk_ids(row[4].as_string() if row[4].is_string() else ""),
                        )
                    )
            return out

        return await asyncio.to_thread(_do)

    async def find_oversized_relations(
        self, collection_id: str, *, min_chars: int, min_fragments: int, limit: int = 200
    ) -> list[Relation]:
        def _do() -> list[Relation]:
            space = self._space(collection_id)
            try:
                result = self._execute(
                    space,
                    f"LOOKUP ON `{_EDGE_TYPE}` "
                    f"YIELD src(edge) AS src, dst(edge) AS dst, "
                    f"properties(edge).description AS desc, "
                    f"properties(edge).weight AS w, "
                    f"properties(edge).source_chunk_ids AS chunks "
                    f"| LIMIT {limit}",
                )
            except Exception:
                return []
            out = []
            for i in range(result.row_size()):
                row = result.row_values(i)
                d = row[2].as_string() if row[2].is_string() else ""
                frags = len(d.split(DESCRIPTION_SEPARATOR)) if d else 0
                if len(d) >= min_chars or frags >= min_fragments:
                    src_id = row[0].as_string() if row[0].is_string() else ""
                    tgt_id = row[1].as_string() if row[1].is_string() else ""
                    if src_id and tgt_id:
                        try:
                            out.append(
                                Relation(
                                    collection_id=collection_id,
                                    source_id=src_id,
                                    target_id=tgt_id,
                                    description=d,
                                    weight=float(row[3].as_double()) if row[3].is_double() else 0.0,
                                    source_chunk_ids=_decode_chunk_ids(
                                        row[4].as_string() if row[4].is_string() else ""
                                    ),
                                )
                            )
                        except (ValueError, KeyError):
                            continue
            return out

        return await asyncio.to_thread(_do)

    async def rewrite_entity_description(self, collection_id: str, entity_id: str, description: str) -> None:
        def _do():
            space = self._space(collection_id)
            self._execute(
                space,
                f'UPDATE VERTEX ON `{_ENTITY_TAG}` "{_escape(entity_id)}" SET description = "{_escape(description)}"',
            )

        await asyncio.to_thread(_do)

    async def rewrite_relation_description(
        self, collection_id: str, source_id: str, target_id: str, description: str
    ) -> None:
        def _do():
            space = self._space(collection_id)
            self._execute(
                space,
                f'UPDATE EDGE ON `{_EDGE_TYPE}` "{_escape(source_id)}"->"{_escape(target_id)}" '
                f'SET description = "{_escape(description)}"',
            )

        await asyncio.to_thread(_do)

    # =========================================================== delete
    async def delete_document_rows(self, collection_id: str, doc_id: str) -> DeleteDocumentResult:
        """Prune + orphan cleanup, matching the PG semantics:

        1. Find chunk ids for the document.
        2. Delete chunk vertices.
        3. For every entity/relation, remove those chunk ids from
           ``source_chunk_ids``.
        4. Delete entities/relations whose ``source_chunk_ids`` became
           empty (orphans).
        """

        def _do() -> DeleteDocumentResult:
            space = self._space(collection_id)
            # 1. Find chunk ids
            try:
                result = self._execute(
                    space,
                    f'LOOKUP ON `{_CHUNK_TAG}` WHERE `{_CHUNK_TAG}`.doc_id == "{_escape(doc_id)}" '
                    f"YIELD id(vertex) AS vid, properties(vertex).chunk_id AS cid",
                )
            except Exception:
                return DeleteDocumentResult(doc_id=doc_id, chunks_removed=0, entities_removed=0, relations_removed=0)

            chunk_ids_set: set[str] = set()
            chunk_vids: list[str] = []
            for i in range(result.row_size()):
                row = result.row_values(i)
                chunk_vids.append(row[0].as_string())
                cid = row[1].as_string() if row[1].is_string() else ""
                if cid:
                    chunk_ids_set.add(cid)

            if not chunk_ids_set:
                return DeleteDocumentResult(doc_id=doc_id, chunks_removed=0, entities_removed=0, relations_removed=0)

            # 2. Delete chunk vertices
            for vid in chunk_vids:
                self._execute(space, f'DELETE VERTEX "{_escape(vid)}"')

            # 3. Prune chunk ids from entities; collect orphans
            entities_removed = 0
            try:
                ent_result = self._execute(
                    space,
                    f"LOOKUP ON `{_ENTITY_TAG}` YIELD id(vertex) AS vid, properties(vertex).source_chunk_ids AS chunks",
                )
                for i in range(ent_result.row_size()):
                    row = ent_result.row_values(i)
                    vid = row[0].as_string()
                    raw_chunks = row[1].as_string() if row[1].is_string() else ""
                    current = set(_decode_chunk_ids(raw_chunks))
                    pruned = current - chunk_ids_set
                    if current != pruned:
                        if not pruned:
                            self._execute(space, f'DELETE VERTEX "{_escape(vid)}" WITH EDGE')
                            entities_removed += 1
                        else:
                            self._execute(
                                space,
                                f'UPDATE VERTEX ON `{_ENTITY_TAG}` "{_escape(vid)}" '
                                f'SET source_chunk_ids = "{_escape(_encode_chunk_ids(sorted(pruned)))}"',
                            )
            except Exception:
                logger.exception("nebula delete_document_rows: entity prune failed")

            # 4. Prune relations
            relations_removed = 0
            try:
                rel_result = self._execute(
                    space,
                    f"LOOKUP ON `{_EDGE_TYPE}` "
                    f"YIELD src(edge) AS src, dst(edge) AS dst, "
                    f"properties(edge).source_chunk_ids AS chunks",
                )
                for i in range(rel_result.row_size()):
                    row = rel_result.row_values(i)
                    src = row[0].as_string() if row[0].is_string() else ""
                    dst = row[1].as_string() if row[1].is_string() else ""
                    raw_chunks = row[2].as_string() if row[2].is_string() else ""
                    current = set(_decode_chunk_ids(raw_chunks))
                    pruned = current - chunk_ids_set
                    if current != pruned:
                        if not pruned:
                            self._execute(
                                space,
                                f'DELETE EDGE `{_EDGE_TYPE}` "{_escape(src)}"->"{_escape(dst)}"',
                            )
                            relations_removed += 1
                        else:
                            self._execute(
                                space,
                                f'UPDATE EDGE ON `{_EDGE_TYPE}` "{_escape(src)}"->"{_escape(dst)}" '
                                f'SET source_chunk_ids = "{_escape(_encode_chunk_ids(sorted(pruned)))}"',
                            )
            except Exception:
                logger.exception("nebula delete_document_rows: relation prune failed")

            return DeleteDocumentResult(
                doc_id=doc_id,
                chunks_removed=len(chunk_vids),
                entities_removed=entities_removed,
                relations_removed=relations_removed,
            )

        return await asyncio.to_thread(_do)

    # ============================================================= read
    async def get_chunks_by_ids(self, collection_id: str, chunk_ids: Sequence[str]) -> list[Chunk]:
        if not chunk_ids:
            return []

        def _do() -> list[Chunk]:
            space = self._space(collection_id)
            vids = ", ".join(f'"{_escape(cid)}"' for cid in chunk_ids)
            try:
                result = self._execute(
                    space,
                    f"FETCH PROP ON `{_CHUNK_TAG}` {vids} "
                    f"YIELD properties(vertex).chunk_id AS cid, "
                    f"properties(vertex).doc_id AS did, "
                    f"properties(vertex).order_in_doc AS ord, "
                    f"properties(vertex).text AS txt, "
                    f"properties(vertex).file_path AS fp",
                )
            except Exception:
                return []
            out = []
            for i in range(result.row_size()):
                row = result.row_values(i)
                out.append(
                    Chunk(
                        chunk_id=row[0].as_string() if row[0].is_string() else "",
                        doc_id=row[1].as_string() if row[1].is_string() else "",
                        collection_id=collection_id,
                        order_in_doc=row[2].as_int() if row[2].is_int() else 0,
                        text=row[3].as_string() if row[3].is_string() else "",
                        file_path=row[4].as_string() if row[4].is_string() else "",
                    )
                )
            return out

        return await asyncio.to_thread(_do)

    async def find_entities_by_ids(self, collection_id: str, entity_ids: Sequence[str]) -> list[Entity]:
        if not entity_ids:
            return []

        def _do() -> list[Entity]:
            space = self._space(collection_id)
            vids = ", ".join(f'"{_escape(eid)}"' for eid in entity_ids)
            try:
                result = self._execute(
                    space,
                    f"FETCH PROP ON `{_ENTITY_TAG}` {vids} "
                    f"YIELD properties(vertex).entity_id AS eid, "
                    f"properties(vertex).name AS name, "
                    f"properties(vertex).type AS type, "
                    f"properties(vertex).description AS desc, "
                    f"properties(vertex).source_chunk_ids AS chunks",
                )
                out = []
                for i in range(result.row_size()):
                    row = result.row_values(i)
                    out.append(
                        Entity(
                            entity_id=row[0].as_string() if row[0].is_string() else "",
                            collection_id=collection_id,
                            name=row[1].as_string() if row[1].is_string() else "",
                            type=row[2].as_string() if row[2].is_string() else "",
                            description=row[3].as_string() if row[3].is_string() else "",
                            source_chunk_ids=_decode_chunk_ids(row[4].as_string() if row[4].is_string() else ""),
                        )
                    )
                return out
            except Exception:
                return []

        return await asyncio.to_thread(_do)

    async def find_entities_by_names(self, collection_id: str, names: Sequence[str]) -> list[Entity]:
        if not names:
            return []

        def _do() -> list[Entity]:
            space = self._space(collection_id)
            out = []
            for name in names:
                try:
                    result = self._execute(
                        space,
                        f'LOOKUP ON `{_ENTITY_TAG}` WHERE `{_ENTITY_TAG}`.name == "{_escape(name)}" '
                        f"YIELD properties(vertex).entity_id AS eid, "
                        f"properties(vertex).name AS name, "
                        f"properties(vertex).type AS type, "
                        f"properties(vertex).description AS desc, "
                        f"properties(vertex).source_chunk_ids AS chunks",
                    )
                    for i in range(result.row_size()):
                        row = result.row_values(i)
                        out.append(
                            Entity(
                                entity_id=row[0].as_string() if row[0].is_string() else "",
                                collection_id=collection_id,
                                name=row[1].as_string() if row[1].is_string() else "",
                                type=row[2].as_string() if row[2].is_string() else "",
                                description=row[3].as_string() if row[3].is_string() else "",
                                source_chunk_ids=_decode_chunk_ids(row[4].as_string() if row[4].is_string() else ""),
                            )
                        )
                except Exception:
                    continue
            return out

        return await asyncio.to_thread(_do)

    async def expand_neighborhood(
        self,
        collection_id: str,
        anchor_entity_ids: Sequence[str],
        max_hop: int,
        limit: int,
    ) -> tuple[list[Entity], list[Relation]]:
        if not anchor_entity_ids:
            return [], []

        def _do() -> tuple[list[Entity], list[Relation]]:
            space = self._space(collection_id)
            vids = ", ".join(f'"{_escape(eid)}"' for eid in anchor_entity_ids)
            try:
                result = self._execute(
                    space,
                    f"GO 0 TO {max(0, int(max_hop))} STEPS FROM {vids} "
                    f"OVER `{_EDGE_TYPE}` BIDIRECT "
                    f"YIELD $$.`{_ENTITY_TAG}`.entity_id AS eid, "
                    f"$$.`{_ENTITY_TAG}`.name AS name, "
                    f"$$.`{_ENTITY_TAG}`.type AS type, "
                    f"$$.`{_ENTITY_TAG}`.description AS desc, "
                    f"$$.`{_ENTITY_TAG}`.source_chunk_ids AS chunks, "
                    f"`{_EDGE_TYPE}`.description AS r_desc, "
                    f"`{_EDGE_TYPE}`.weight AS r_w, "
                    f"`{_EDGE_TYPE}`.source_chunk_ids AS r_chunks, "
                    f"`{_EDGE_TYPE}`._src AS r_src, "
                    f"`{_EDGE_TYPE}`._dst AS r_dst "
                    f"| LIMIT {int(limit) * 2}",
                )
            except Exception:
                logger.exception("Nebula expand_neighborhood failed")
                return [], []

            entity_map: dict[str, Entity] = {}
            relation_set: set[tuple[str, str]] = set()
            relations: list[Relation] = []

            # Add anchors themselves
            for eid in anchor_entity_ids:
                try:
                    r = self._execute(
                        space,
                        f'FETCH PROP ON `{_ENTITY_TAG}` "{_escape(eid)}" '
                        f"YIELD properties(vertex).entity_id AS eid, "
                        f"properties(vertex).name AS name, "
                        f"properties(vertex).type AS type, "
                        f"properties(vertex).description AS desc, "
                        f"properties(vertex).source_chunk_ids AS chunks",
                    )
                    if r.row_size() > 0:
                        row = r.row_values(0)
                        entity_map[eid] = Entity(
                            entity_id=eid,
                            collection_id=collection_id,
                            name=row[1].as_string() if row[1].is_string() else "",
                            type=row[2].as_string() if row[2].is_string() else "",
                            description=row[3].as_string() if row[3].is_string() else "",
                            source_chunk_ids=_decode_chunk_ids(row[4].as_string() if row[4].is_string() else ""),
                        )
                except Exception:
                    continue

            for i in range(result.row_size()):
                row = result.row_values(i)
                eid = row[0].as_string() if row[0].is_string() else ""
                if eid and eid not in entity_map:
                    entity_map[eid] = Entity(
                        entity_id=eid,
                        collection_id=collection_id,
                        name=row[1].as_string() if row[1].is_string() else "",
                        type=row[2].as_string() if row[2].is_string() else "",
                        description=row[3].as_string() if row[3].is_string() else "",
                        source_chunk_ids=_decode_chunk_ids(row[4].as_string() if row[4].is_string() else ""),
                    )
                src = row[8].as_string() if row[8].is_string() else ""
                dst = row[9].as_string() if row[9].is_string() else ""
                if src and dst and (src, dst) not in relation_set and src != dst:
                    relation_set.add((src, dst))
                    try:
                        relations.append(
                            Relation(
                                collection_id=collection_id,
                                source_id=src,
                                target_id=dst,
                                description=row[5].as_string() if row[5].is_string() else "",
                                weight=float(row[6].as_double()) if row[6].is_double() else 0.0,
                                source_chunk_ids=_decode_chunk_ids(row[7].as_string() if row[7].is_string() else ""),
                            )
                        )
                    except (ValueError, KeyError):
                        continue

            entities = list(entity_map.values())[:limit]
            return entities, relations

        return await asyncio.to_thread(_do)

    async def list_labels(self, collection_id: str) -> list[str]:
        def _do() -> list[str]:
            space = self._space(collection_id)
            try:
                result = self._execute(
                    space,
                    f"LOOKUP ON `{_ENTITY_TAG}` YIELD DISTINCT properties(vertex).type AS t",
                )
                types = set()
                for i in range(result.row_size()):
                    t = result.row_values(i)[0].as_string() if result.row_values(i)[0].is_string() else ""
                    if t:
                        types.add(t)
                return sorted(types)
            except Exception:
                return []

        return await asyncio.to_thread(_do)

    async def list_subgraph(
        self,
        collection_id: str,
        label: Optional[str],
        max_depth: int,
        max_nodes: int,
    ) -> KnowledgeGraph:
        max_nodes = max(1, int(max_nodes))
        max_depth = max(0, int(max_depth))

        def _do() -> list[str]:
            space = self._space(collection_id)
            where = ""
            if label and label != "*":
                where = f'WHERE `{_ENTITY_TAG}`.type == "{_escape(label)}" '
            try:
                result = self._execute(
                    space,
                    f"LOOKUP ON `{_ENTITY_TAG}` {where}YIELD properties(vertex).entity_id AS eid | LIMIT {max_nodes}",
                )
                return [result.row_values(i)[0].as_string() for i in range(result.row_size())]
            except Exception:
                return []

        anchor_ids = await asyncio.to_thread(_do)
        if not anchor_ids:
            return KnowledgeGraph(nodes=[], edges=[], is_truncated=False)

        entities, relations = await self.expand_neighborhood(
            collection_id=collection_id,
            anchor_entity_ids=anchor_ids,
            max_hop=max_depth,
            limit=max_nodes,
        )
        is_truncated = len(entities) >= max_nodes
        return KnowledgeGraph(nodes=entities[:max_nodes], edges=relations, is_truncated=is_truncated)


__all__ = ["NebulaGraphStore"]
