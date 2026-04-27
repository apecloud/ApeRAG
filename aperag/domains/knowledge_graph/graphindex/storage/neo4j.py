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

"""Neo4j implementation of the ``GraphStore`` Protocol.

Uses the native async driver from ``neo4j`` 5.x+. Each graph entity is a
``(:Entity {collection_id, entity_id, ...})`` node; relations are
``[:RELATES_TO {...}]`` edges. Multi-tenancy is property-based via
``collection_id`` on every node and relationship, with a composite
uniqueness constraint ``(collection_id, entity_id)`` on Entity nodes.

Connection pooling: the ``AsyncDriver`` is process-level (created once
per ``Neo4jGraphStore`` instance). Callers should share a single
``Neo4jGraphStore`` across the process, which is what
``integration.py`` already does for the PG backend.
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence

from aperag.domains.knowledge_graph.graphindex.dto import (
    DESCRIPTION_SEPARATOR,
    Chunk,
    DeleteDocumentResult,
    Entity,
    KnowledgeGraph,
    MergeEntitiesResult,
    Relation,
)

logger = logging.getLogger(__name__)

_ENTITY_LABEL = "Entity"
_CHUNK_LABEL = "Chunk"
_EDGE_TYPE = "RELATES_TO"


class Neo4jGraphStore:
    """``GraphStore`` backed by Neo4j (async driver).

    Thread safety: ``AsyncDriver`` is connection-pool safe across
    coroutines. One instance per process is fine.
    """

    def __init__(self, *, uri: str, username: str = "neo4j", password: str = "") -> None:
        from neo4j import AsyncGraphDatabase

        self._driver = AsyncGraphDatabase.driver(uri, auth=(username, password))

    async def close(self) -> None:
        await self._driver.close()

    # =========================================================== schema
    async def ensure_schema(self) -> None:
        async with self._driver.session() as session:
            await session.run(
                f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:{_ENTITY_LABEL}) "
                f"REQUIRE (n.collection_id, n.entity_id) IS UNIQUE"
            )
            await session.run(
                f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:{_CHUNK_LABEL}) "
                f"REQUIRE (n.collection_id, n.chunk_id) IS UNIQUE"
            )
            await session.run(f"CREATE INDEX IF NOT EXISTS FOR (n:{_ENTITY_LABEL}) ON (n.collection_id, n.name)")

    async def drop_collection(self, collection_id: str) -> None:
        async with self._driver.session() as session:
            await session.run(
                "MATCH (n {collection_id: $cid}) DETACH DELETE n",
                cid=collection_id,
            )
        logger.info("neo4j graphstore: dropped all nodes for collection %s", collection_id)

    # ============================================================ write
    async def upsert_chunks(self, collection_id: str, chunks: Sequence[Chunk]) -> None:
        if not chunks:
            return
        query = (
            f"UNWIND $rows AS row "
            f"MERGE (c:{_CHUNK_LABEL} {{collection_id: $cid, chunk_id: row.chunk_id}}) "
            f"SET c.doc_id = row.doc_id, c.order_in_doc = row.order_in_doc, "
            f"c.text = row.text, c.file_path = row.file_path"
        )
        rows = [
            {
                "chunk_id": c.chunk_id,
                "doc_id": c.doc_id,
                "order_in_doc": c.order_in_doc,
                "text": c.text,
                "file_path": c.file_path or "",
            }
            for c in chunks
        ]
        async with self._driver.session() as session:
            await session.run(query, cid=collection_id, rows=rows)

    async def upsert_entities(self, collection_id: str, entities: Sequence[Entity]) -> None:
        if not entities:
            return
        query = (
            f"UNWIND $rows AS row "
            f"MERGE (n:{_ENTITY_LABEL} {{collection_id: $cid, entity_id: row.eid}}) "
            f"ON CREATE SET "
            f"  n.name = row.name, n.type = row.type, "
            f"  n.description = row.desc, "
            f"  n.source_chunk_ids = row.chunks "
            f"ON MATCH SET "
            f"  n.name = row.name, n.type = row.type, "
            f"  n.description = CASE "
            f"    WHEN n.description IS NULL OR n.description = '' THEN row.desc "
            f"    WHEN row.desc IS NULL OR row.desc = '' THEN n.description "
            f"    WHEN n.description CONTAINS row.desc THEN n.description "
            f"    ELSE n.description + $sep + row.desc "
            f"  END, "
            f"  n.source_chunk_ids = apoc.coll.toSet(n.source_chunk_ids + row.chunks)"
        )
        rows = [
            {
                "eid": e.entity_id,
                "name": e.name,
                "type": e.type,
                "desc": e.description,
                "chunks": list(e.source_chunk_ids),
            }
            for e in entities
        ]
        async with self._driver.session() as session:
            try:
                await session.run(query, cid=collection_id, rows=rows, sep=DESCRIPTION_SEPARATOR)
            except Exception:
                # Fallback for Neo4j without APOC: use plain list concatenation
                query_no_apoc = query.replace(
                    "apoc.coll.toSet(n.source_chunk_ids + row.chunks)",
                    "[x IN (n.source_chunk_ids + row.chunks) WHERE x IS NOT NULL | x]",
                )
                await session.run(query_no_apoc, cid=collection_id, rows=rows, sep=DESCRIPTION_SEPARATOR)

    async def upsert_relations(self, collection_id: str, relations: Sequence[Relation]) -> None:
        if not relations:
            return
        query = (
            f"UNWIND $rows AS row "
            f"MATCH (a:{_ENTITY_LABEL} {{collection_id: $cid, entity_id: row.src}}) "
            f"MATCH (b:{_ENTITY_LABEL} {{collection_id: $cid, entity_id: row.tgt}}) "
            f"MERGE (a)-[r:{_EDGE_TYPE} {{collection_id: $cid}}]->(b) "
            f"ON CREATE SET "
            f"  r.description = row.desc, r.weight = row.w, "
            f"  r.source_chunk_ids = row.chunks "
            f"ON MATCH SET "
            f"  r.weight = CASE WHEN r.weight > row.w THEN r.weight ELSE row.w END, "
            f"  r.description = CASE "
            f"    WHEN r.description IS NULL OR r.description = '' THEN row.desc "
            f"    WHEN row.desc IS NULL OR row.desc = '' THEN r.description "
            f"    WHEN r.description CONTAINS row.desc THEN r.description "
            f"    ELSE r.description + $sep + row.desc "
            f"  END, "
            f"  r.source_chunk_ids = [x IN (r.source_chunk_ids + row.chunks) WHERE x IS NOT NULL | x]"
        )
        rows = [
            {
                "src": r.source_id,
                "tgt": r.target_id,
                "desc": r.description,
                "w": float(r.weight),
                "chunks": list(r.source_chunk_ids),
            }
            for r in relations
        ]
        async with self._driver.session() as session:
            await session.run(query, cid=collection_id, rows=rows, sep=DESCRIPTION_SEPARATOR)

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

        async with self._driver.session() as session:
            # Load target
            result = await session.run(
                f"MATCH (n:{_ENTITY_LABEL} {{collection_id: $cid, entity_id: $eid}}) "
                f"RETURN n.description AS desc, n.source_chunk_ids AS chunks",
                cid=collection_id,
                eid=target_entity_id,
            )
            target_rec = await result.single()
            if target_rec is None:
                raise ValueError(f"Target entity {target_entity_id!r} not found")

            description = target_rec["desc"] or ""
            chunk_ids: set[str] = set(target_rec["chunks"] or [])

            # Load sources
            result = await session.run(
                f"MATCH (n:{_ENTITY_LABEL} {{collection_id: $cid}}) "
                f"WHERE n.entity_id IN $ids "
                f"RETURN n.entity_id AS eid, n.description AS desc, n.source_chunk_ids AS chunks",
                cid=collection_id,
                ids=source_ids,
            )
            source_rows = [rec async for rec in result]
            merged_source_ids = tuple(r["eid"] for r in source_rows)

            for r in source_rows:
                frag = (r["desc"] or "").strip()
                if frag and frag not in description:
                    description = (description + DESCRIPTION_SEPARATOR + frag) if description else frag
                chunk_ids.update(r["chunks"] or [])

            # Redirect edges from sources to target. Multiple sources can
            # collide onto the same (target, other) key, so collapse those
            # in Python first, then hand the rebuilt edges to the normal
            # relation upsert path.
            edges_redirected = 0
            edges_collapsed = 0
            rebuilt_map: dict[tuple[str, str], Relation] = {}
            for direction in ["outgoing", "incoming"]:
                if direction == "outgoing":
                    match_q = (
                        f"MATCH (s:{_ENTITY_LABEL} {{collection_id: $cid}})-[r:{_EDGE_TYPE}]->(other) "
                        f"WHERE s.entity_id IN $src_ids "
                        f"RETURN s.entity_id AS src, other.entity_id AS tgt, "
                        f"r.description AS desc, r.weight AS w, r.source_chunk_ids AS chunks"
                    )
                else:
                    match_q = (
                        f"MATCH (other)-[r:{_EDGE_TYPE}]->(s:{_ENTITY_LABEL} {{collection_id: $cid}}) "
                        f"WHERE s.entity_id IN $src_ids "
                        f"RETURN other.entity_id AS src, s.entity_id AS tgt, "
                        f"r.description AS desc, r.weight AS w, r.source_chunk_ids AS chunks"
                    )
                result = await session.run(match_q, cid=collection_id, src_ids=source_ids)
                edge_rows = [rec async for rec in result]
                for e in edge_rows:
                    new_src = target_entity_id if e["src"] in source_ids else e["src"]
                    new_tgt = target_entity_id if e["tgt"] in source_ids else e["tgt"]
                    if new_src == new_tgt:
                        edges_collapsed += 1
                        continue
                    key = (new_src, new_tgt)
                    incoming = Relation(
                        collection_id=collection_id,
                        source_id=new_src,
                        target_id=new_tgt,
                        description=e["desc"] or "",
                        weight=float(e["w"] or 0),
                        source_chunk_ids=tuple(e["chunks"] or ()),
                    )
                    if key in rebuilt_map:
                        existing = rebuilt_map[key]
                        desc_a = (existing.description or "").strip()
                        desc_b = (incoming.description or "").strip()
                        if not desc_a:
                            merged_desc = desc_b
                        elif not desc_b or desc_b in desc_a:
                            merged_desc = existing.description
                        else:
                            merged_desc = existing.description + DESCRIPTION_SEPARATOR + incoming.description
                        rebuilt_map[key] = Relation(
                            collection_id=collection_id,
                            source_id=new_src,
                            target_id=new_tgt,
                            description=merged_desc,
                            weight=max(existing.weight, incoming.weight),
                            source_chunk_ids=tuple(
                                dict.fromkeys((*existing.source_chunk_ids, *incoming.source_chunk_ids))
                            ),
                        )
                        edges_collapsed += 1
                    else:
                        rebuilt_map[key] = incoming
                        edges_redirected += 1

            # Delete source edges
            await session.run(
                f"MATCH (s:{_ENTITY_LABEL} {{collection_id: $cid}})-[r:{_EDGE_TYPE}]-() "
                f"WHERE s.entity_id IN $ids DELETE r",
                cid=collection_id,
                ids=source_ids,
            )
            await session.run(
                f"MATCH ()-[r:{_EDGE_TYPE}]->(s:{_ENTITY_LABEL} {{collection_id: $cid}}) "
                f"WHERE s.entity_id IN $ids DELETE r",
                cid=collection_id,
                ids=source_ids,
            )

            # Delete source nodes
            await session.run(
                f"MATCH (n:{_ENTITY_LABEL} {{collection_id: $cid}}) WHERE n.entity_id IN $ids DELETE n",
                cid=collection_id,
                ids=source_ids,
            )

            # Update target
            await session.run(
                f"MATCH (n:{_ENTITY_LABEL} {{collection_id: $cid, entity_id: $eid}}) "
                f"SET n.description = $desc, n.source_chunk_ids = $chunks",
                cid=collection_id,
                eid=target_entity_id,
                desc=description,
                chunks=sorted(chunk_ids),
            )

        rebuilt = list(rebuilt_map.values())
        if rebuilt:
            await self.upsert_relations(collection_id, rebuilt)

        return MergeEntitiesResult(
            target_entity_id=target_entity_id,
            merged_source_ids=merged_source_ids,
            description=description,
            source_chunk_ids=tuple(sorted(chunk_ids)),
            edges_redirected=edges_redirected,
            edges_collapsed=edges_collapsed,
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
        query = (
            f"MATCH (n:{_ENTITY_LABEL} {{collection_id: $cid}}) "
            f"WHERE n.description IS NOT NULL AND ("
            f"  size(n.description) >= $minchars OR "
            f"  size(split(n.description, $sep)) >= $minfrags"
            f") "
            f"RETURN n ORDER BY size(n.description) DESC LIMIT $lim"
        )
        async with self._driver.session() as session:
            result = await session.run(
                query,
                cid=collection_id,
                minchars=min_chars,
                minfrags=min_fragments,
                sep=DESCRIPTION_SEPARATOR,
                lim=limit,
            )
            return [_node_to_entity(rec["n"], collection_id) async for rec in result]

    async def find_oversized_relations(
        self,
        collection_id: str,
        *,
        min_chars: int,
        min_fragments: int,
        limit: int = 200,
    ) -> list[Relation]:
        query = (
            f"MATCH (a:{_ENTITY_LABEL} {{collection_id: $cid}})"
            f"-[r:{_EDGE_TYPE} {{collection_id: $cid}}]->"
            f"(b:{_ENTITY_LABEL} {{collection_id: $cid}}) "
            f"WHERE r.description IS NOT NULL AND ("
            f"  size(r.description) >= $minchars OR "
            f"  size(split(r.description, $sep)) >= $minfrags"
            f") "
            f"RETURN a.entity_id AS src, b.entity_id AS tgt, "
            f"r.description AS desc, r.weight AS w, r.source_chunk_ids AS chunks "
            f"ORDER BY size(r.description) DESC LIMIT $lim"
        )
        async with self._driver.session() as session:
            result = await session.run(
                query,
                cid=collection_id,
                minchars=min_chars,
                minfrags=min_fragments,
                sep=DESCRIPTION_SEPARATOR,
                lim=limit,
            )
            return [
                Relation(
                    collection_id=collection_id,
                    source_id=rec["src"],
                    target_id=rec["tgt"],
                    description=rec["desc"] or "",
                    weight=float(rec["w"] or 0),
                    source_chunk_ids=tuple(rec["chunks"] or ()),
                )
                async for rec in result
            ]

    async def rewrite_entity_description(self, collection_id: str, entity_id: str, description: str) -> None:
        async with self._driver.session() as session:
            await session.run(
                f"MATCH (n:{_ENTITY_LABEL} {{collection_id: $cid, entity_id: $eid}}) SET n.description = $desc",
                cid=collection_id,
                eid=entity_id,
                desc=description,
            )

    async def rewrite_relation_description(
        self, collection_id: str, source_id: str, target_id: str, description: str
    ) -> None:
        async with self._driver.session() as session:
            await session.run(
                f"MATCH (a:{_ENTITY_LABEL} {{collection_id: $cid, entity_id: $src}})"
                f"-[r:{_EDGE_TYPE}]->"
                f"(b:{_ENTITY_LABEL} {{collection_id: $cid, entity_id: $tgt}}) "
                f"SET r.description = $desc",
                cid=collection_id,
                src=source_id,
                tgt=target_id,
                desc=description,
            )

    # =========================================================== delete
    async def delete_document_rows(self, collection_id: str, doc_id: str) -> DeleteDocumentResult:
        async with self._driver.session() as session:
            # 1. Find chunk ids for this document
            result = await session.run(
                f"MATCH (c:{_CHUNK_LABEL} {{collection_id: $cid, doc_id: $did}}) RETURN c.chunk_id AS cid",
                cid=collection_id,
                did=doc_id,
            )
            chunk_ids = [rec["cid"] async for rec in result]
            if not chunk_ids:
                return DeleteDocumentResult(doc_id=doc_id, chunks_removed=0, entities_removed=0, relations_removed=0)

            # 2. Delete chunks
            result = await session.run(
                f"MATCH (c:{_CHUNK_LABEL} {{collection_id: $cid, doc_id: $did}}) DELETE c RETURN count(c) AS cnt",
                cid=collection_id,
                did=doc_id,
            )
            rec = await result.single()
            chunks_removed = rec["cnt"] if rec else 0

            # 3. Prune chunk ids from entities and remove orphans
            await session.run(
                f"MATCH (n:{_ENTITY_LABEL} {{collection_id: $cid}}) "
                f"WHERE any(c IN $chunk_ids WHERE c IN n.source_chunk_ids) "
                f"SET n.source_chunk_ids = [x IN n.source_chunk_ids WHERE NOT x IN $chunk_ids]",
                cid=collection_id,
                chunk_ids=chunk_ids,
            )

            # 4. Prune chunk ids from relations and remove orphans
            await session.run(
                f"MATCH (:{_ENTITY_LABEL} {{collection_id: $cid}})"
                f"-[r:{_EDGE_TYPE}]->"
                f"(:{_ENTITY_LABEL} {{collection_id: $cid}}) "
                f"WHERE any(c IN $chunk_ids WHERE c IN r.source_chunk_ids) "
                f"SET r.source_chunk_ids = [x IN r.source_chunk_ids WHERE NOT x IN $chunk_ids]",
                cid=collection_id,
                chunk_ids=chunk_ids,
            )

            # 5. Delete orphan relations (empty source_chunk_ids)
            result = await session.run(
                f"MATCH (:{_ENTITY_LABEL} {{collection_id: $cid}})"
                f"-[r:{_EDGE_TYPE}]->"
                f"(:{_ENTITY_LABEL} {{collection_id: $cid}}) "
                f"WHERE size(r.source_chunk_ids) = 0 "
                f"DELETE r RETURN count(r) AS cnt",
                cid=collection_id,
            )
            rec = await result.single()
            relations_removed = rec["cnt"] if rec else 0

            # 6. Delete orphan entities
            result = await session.run(
                f"MATCH (n:{_ENTITY_LABEL} {{collection_id: $cid}}) "
                f"WHERE size(n.source_chunk_ids) = 0 "
                f"DETACH DELETE n RETURN count(n) AS cnt",
                cid=collection_id,
            )
            rec = await result.single()
            entities_removed = rec["cnt"] if rec else 0

        return DeleteDocumentResult(
            doc_id=doc_id,
            chunks_removed=int(chunks_removed),
            entities_removed=int(entities_removed),
            relations_removed=int(relations_removed),
        )

    # ============================================================= read
    async def get_chunks_by_ids(self, collection_id: str, chunk_ids: Sequence[str]) -> list[Chunk]:
        if not chunk_ids:
            return []
        async with self._driver.session() as session:
            result = await session.run(
                f"MATCH (c:{_CHUNK_LABEL} {{collection_id: $cid}}) WHERE c.chunk_id IN $ids RETURN c",
                cid=collection_id,
                ids=list(chunk_ids),
            )
            return [
                Chunk(
                    chunk_id=rec["c"]["chunk_id"],
                    doc_id=rec["c"]["doc_id"],
                    collection_id=collection_id,
                    order_in_doc=rec["c"].get("order_in_doc", 0),
                    text=rec["c"].get("text", ""),
                    file_path=rec["c"].get("file_path", ""),
                )
                async for rec in result
            ]

    async def find_entities_by_ids(self, collection_id: str, entity_ids: Sequence[str]) -> list[Entity]:
        if not entity_ids:
            return []
        async with self._driver.session() as session:
            result = await session.run(
                f"MATCH (n:{_ENTITY_LABEL} {{collection_id: $cid}}) WHERE n.entity_id IN $ids RETURN n",
                cid=collection_id,
                ids=list(entity_ids),
            )
            return [_node_to_entity(rec["n"], collection_id) async for rec in result]

    async def find_entities_by_names(self, collection_id: str, names: Sequence[str]) -> list[Entity]:
        if not names:
            return []
        async with self._driver.session() as session:
            result = await session.run(
                f"MATCH (n:{_ENTITY_LABEL} {{collection_id: $cid}}) WHERE n.name IN $names RETURN n",
                cid=collection_id,
                names=list(names),
            )
            return [_node_to_entity(rec["n"], collection_id) async for rec in result]

    async def expand_neighborhood(
        self,
        collection_id: str,
        anchor_entity_ids: Sequence[str],
        max_hop: int,
        limit: int,
    ) -> tuple[list[Entity], list[Relation]]:
        if not anchor_entity_ids:
            return [], []

        query = (
            f"MATCH (start:{_ENTITY_LABEL} {{collection_id: $cid}}) "
            f"WHERE start.entity_id IN $anchors "
            f"CALL apoc.path.subgraphAll(start, {{maxLevel: $maxhop, "
            f"labelFilter: '+{_ENTITY_LABEL}', "
            f"relationshipFilter: '{_EDGE_TYPE}'}}) "
            f"YIELD nodes, relationships "
            f"UNWIND nodes AS n "
            f"WITH COLLECT(DISTINCT n) AS allNodes, "
            f"COLLECT(DISTINCT relationships) AS allRelsNested "
            f"UNWIND allRelsNested AS rels "
            f"UNWIND rels AS r "
            f"WITH allNodes, COLLECT(DISTINCT r) AS allRels "
            f"RETURN allNodes, allRels"
        )
        # Fallback for Neo4j without APOC: variable-length path
        fallback_query = (
            f"MATCH path = (start:{_ENTITY_LABEL} {{collection_id: $cid}})"
            f"-[:{_EDGE_TYPE}*0..{max(0, int(max_hop))}]-"
            f"(end:{_ENTITY_LABEL} {{collection_id: $cid}}) "
            f"WHERE start.entity_id IN $anchors "
            f"WITH COLLECT(DISTINCT end) + COLLECT(DISTINCT start) AS allNodesList "
            f"UNWIND allNodesList AS n "
            f"WITH COLLECT(DISTINCT n)[..{int(limit)}] AS nodes "
            f"UNWIND nodes AS n "
            f"WITH COLLECT(n) AS nodes, [x IN nodes | x.entity_id] AS nodeIds "
            f"OPTIONAL MATCH (a:{_ENTITY_LABEL} {{collection_id: $cid}})"
            f"-[r:{_EDGE_TYPE}]->"
            f"(b:{_ENTITY_LABEL} {{collection_id: $cid}}) "
            f"WHERE a.entity_id IN nodeIds AND b.entity_id IN nodeIds "
            f"RETURN nodes AS allNodes, COLLECT(DISTINCT {{r: r, src: a.entity_id, tgt: b.entity_id}}) AS rels"
        )

        async with self._driver.session() as session:
            try:
                result = await session.run(query, cid=collection_id, anchors=list(anchor_entity_ids), maxhop=max_hop)
                rec = await result.single()
                if rec is None:
                    return [], []
                entity_nodes = rec["allNodes"]
                rel_records = rec["allRels"]
                entities = [_node_to_entity(n, collection_id) for n in entity_nodes[:limit]]
                relations = []
                for r in rel_records:
                    try:
                        relations.append(
                            Relation(
                                collection_id=collection_id,
                                source_id=r.start_node["entity_id"],
                                target_id=r.end_node["entity_id"],
                                description=r.get("description", "") or "",
                                weight=float(r.get("weight", 0) or 0),
                                source_chunk_ids=tuple(r.get("source_chunk_ids") or ()),
                            )
                        )
                    except (ValueError, KeyError):
                        continue
                return entities, relations
            except Exception:
                # Fallback: no APOC available
                result = await session.run(fallback_query, cid=collection_id, anchors=list(anchor_entity_ids))
                rec = await result.single()
                if rec is None:
                    return [], []
                entities = [_node_to_entity(n, collection_id) for n in (rec["allNodes"] or [])[:limit]]
                relations = []
                for item in rec.get("rels") or []:
                    r = item.get("r")
                    if r is None:
                        continue
                    try:
                        relations.append(
                            Relation(
                                collection_id=collection_id,
                                source_id=item["src"],
                                target_id=item["tgt"],
                                description=r.get("description", "") or "",
                                weight=float(r.get("weight", 0) or 0),
                                source_chunk_ids=tuple(r.get("source_chunk_ids") or ()),
                            )
                        )
                    except (ValueError, KeyError):
                        continue
                return entities, relations

    async def list_subgraph(
        self,
        collection_id: str,
        label: Optional[str],
        max_depth: int,
        max_nodes: int,
    ) -> KnowledgeGraph:
        max_nodes = max(1, int(max_nodes))
        max_depth = max(0, int(max_depth))

        # Get top-degree entities
        type_filter = ""
        params: dict = {"cid": collection_id, "lim": max_nodes}
        if label and label != "*":
            type_filter = "AND n.type = $lbl "
            params["lbl"] = label

        query = (
            f"MATCH (n:{_ENTITY_LABEL} {{collection_id: $cid}}) "
            f"WHERE true {type_filter}"
            f"OPTIONAL MATCH (n)-[r:{_EDGE_TYPE}]-() "
            f"WITH n, count(r) AS deg "
            f"ORDER BY deg DESC LIMIT $lim "
            f"RETURN n.entity_id AS eid"
        )
        async with self._driver.session() as session:
            result = await session.run(query, **params)
            anchor_ids = [rec["eid"] async for rec in result]

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


def _node_to_entity(node, collection_id: str) -> Entity:
    return Entity(
        entity_id=node["entity_id"],
        collection_id=collection_id,
        name=node.get("name", ""),
        type=node.get("type", ""),
        description=node.get("description", "") or "",
        source_chunk_ids=tuple(node.get("source_chunk_ids") or ()),
    )


__all__ = ["Neo4jGraphStore"]
