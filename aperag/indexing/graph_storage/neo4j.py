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

"""Neo4j implementation of :class:`LineageGraphStore` — Wave 4 T8 chunk 2.

Mirrors :class:`PostgresLineageGraphStore` (chunk 1 reference impl) over
Neo4j 5.x. Both backends realise the §D.3.5 lineage SET semantics; the
Postgres adapter expresses each SET as a JSONB array, here we use the
**parallel-list** encoding because Neo4j node properties cannot hold
``LIST<MAP>`` directly (only homogeneous lists of primitives are
allowed). Per architect ratify msg=95179f2a, the dedup key
``(document_id, parse_version)`` and the strip-by-document_id semantics
must port cross-backend; this module realises both purely in Cypher
without requiring APOC.

Storage layout — one row per entity / relation, modelled as a label
``aperag_LineageEntity`` / ``aperag_LineageRelation`` node (relations
are NOT modelled as edges here because the §D.3 Protocol does not
require traversal — that is a separate retrieval concern owned by the
legacy ``GraphStore``; keeping relations as their own labelled nodes
mirrors the Postgres two-table schema exactly). The ``aperag_`` label
prefix (Wave 5 P5A item 4) prevents collisions with user-owned graphs
in shared Neo4j deployments.

    (:aperag_LineageEntity {
        collection_id, name, entity_type,
        source_lineage,                  -- list<string>, JSON-encoded LineageMember
        source_lineage_doc_ids,          -- list<string>, parallel index, dedup + strip-by-doc filter
        source_lineage_parse_versions,   -- list<string>, parallel index, dedup composite key
        description_parts,               -- list<string>, JSON-encoded DescriptionPart
        description_parts_doc_ids,
        description_parts_parse_versions,
        gmt_created, gmt_updated
    })

    (:aperag_LineageRelation {
        collection_id, source, target, relation_type,
        evidence_lineage,
        evidence_lineage_doc_ids,
        evidence_lineage_parse_versions,
        description_parts,
        description_parts_doc_ids,
        description_parts_parse_versions,
        gmt_created, gmt_updated
    })

Wave 4 chunk 4 dropped the legacy ``description`` property — the
per-document fragments in ``description_parts`` are the canonical
source. The standalone overwritten-on-every-upsert ``description``
property had no consumer on the §D.3 / §G.5 read path
(``RelationWithLineage`` does not expose it), so dropping it
cross-backend keeps "ORM 100% mirror" with the Postgres adapter.

Rationale for parallel lists: Cypher list comprehensions can filter by
index (``[i IN range(0, size(L)-1) WHERE L[i] <> $key]``) but a list of
maps is not a valid property type. Storing the dedup key fields in
parallel ``list<string>`` properties keeps the strip-then-append a
single Cypher statement (and therefore atomic under Neo4j MERGE's
write-lock on the row) while preserving the full JSON serialisation of
each member alongside.

§D.3.5 Protocol semantics → Cypher:

* ``find_entity_ids_with_lineage(document_id)`` →
  ``MATCH (n:LineageEntity {collection_id:$cid}) WHERE $document_id IN
  n.source_lineage_doc_ids RETURN n.name``. Pure list membership; no
  containment-style index but Neo4j scans the entity label, which the
  per-collection ``collection_id`` index narrows.
* ``remove_entity_lineage_member(entity_name, document_id)`` →
  one ``MATCH … SET …`` statement that filters BOTH lineage and
  description-part lists by ``doc_id != $d``. Single statement so two
  concurrent syncs cannot race on the read-modify-write.
* ``upsert_entity_with_lineage`` → ``MERGE`` plus a strip-then-append
  ``SET`` that drops the existing ``(doc_id, parse_v)`` element (if
  any) and re-appends with the new payload. Replaces semantics, single
  statement.
* ``gc_*_if_orphan`` → ``MATCH … WHERE size(...) = 0 DELETE`` returns
  the count for the boolean.

Concurrency: Neo4j MERGE issues an exclusive lock on the merged node /
constraint key for the duration of the statement, so the
strip-then-append SET that follows cannot interleave with another
upsert against the same row. ``EntityLock`` is therefore unnecessary
for this backend (per architect msg=f2921ae0 — Neo4j with native list
ops doesn't need the per-entity Redis lock that Nebula's
read-modify-write loop requires).

Hard-cut second round: this module is the new Neo4j adapter. The
legacy ``aperag/domains/knowledge_graph/graphindex/storage/neo4j.py``
is deleted in chunk 4 of the same Wave 4 PR.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from aperag.indexing.graph import (
    DescriptionPart,
    EntityRecord,
    EntityWithLineage,
    LineageMember,
    RelationRecord,
    RelationWithLineage,
)

logger = logging.getLogger(__name__)


# Wave 5 P5A item 4 (per `feedback_production_readiness_invariant.md`
# multi-tenant Neo4j hygiene): namespace the lineage labels with an
# ``aperag_`` prefix so deployments that share a Neo4j instance with
# user-owned graphs don't collide on a generic ``LineageEntity`` /
# ``LineageRelation`` label. Constraint names get the same prefix so
# ``ensure_schema`` is idempotent against fresh and prefixed-but-already-
# created clusters alike. Wave 4 graph indexing was gated behind
# ``enable_knowledge_graph=False`` by default so there is no production
# data on the legacy unprefixed labels — no data migration needed for
# the hard-cut second-round bump.
_ENTITY_LABEL = "aperag_LineageEntity"
_RELATION_LABEL = "aperag_LineageRelation"

_ENTITY_CONSTRAINT_NAME = "aperag_lineage_entity_collection_name_unique"
_RELATION_CONSTRAINT_NAME = "aperag_lineage_relation_collection_triple_unique"


# ---------------------------------------------------------------------
# JSON serialisation helpers — kept module-private so the dedup-key
# and parallel-list encoding stays in one place.
# ---------------------------------------------------------------------


def _lineage_member_json(member: LineageMember) -> str:
    return json.dumps(member.to_dict())


def _description_part_json(part: DescriptionPart) -> str:
    return json.dumps(part.to_dict())


# ---------------------------------------------------------------------
# Public adapter.
# ---------------------------------------------------------------------


class Neo4jLineageGraphStore:
    """:class:`LineageGraphStore` backed by Neo4j (5.x async driver).

    One instance per ``collection_id``. The ``worker_factory`` (chunk 4)
    caches one instance per collection, mirroring the
    :class:`PostgresLineageGraphStore` pattern. The adapter owns no
    schema-creation responsibility in production; :meth:`ensure_schema`
    is the test / dev convenience for spinning up the constraints when
    a fresh Neo4j instance is involved.
    """

    def __init__(
        self,
        *,
        driver: Any,
        collection_id: str,
        database: str | None = None,
    ) -> None:
        """Construct the adapter.

        ``driver`` is a ``neo4j.AsyncDriver`` (typed as ``Any`` so this
        module does not import ``neo4j`` at top level — keeps optional-
        dep handling consistent with the legacy ``Neo4jGraphStore``).
        ``database`` is forwarded to ``driver.session(database=...)``;
        ``None`` defers to the driver default.
        """
        self._driver = driver
        self._collection_id = collection_id
        self._database = database

    def _session(self):
        if self._database is None:
            return self._driver.session()
        return self._driver.session(database=self._database)

    # -- schema -------------------------------------------------------

    async def ensure_schema(self) -> None:
        """Create the composite unique constraints if missing.

        Neo4j 5.x uses node property uniqueness constraints; the
        ``IF NOT EXISTS`` clause makes this idempotent so production
        rollouts that already created the constraints (via terraform /
        a cluster admin script) just no-op.
        """
        async with self._session() as session:
            await session.run(
                f"CREATE CONSTRAINT {_ENTITY_CONSTRAINT_NAME} IF NOT EXISTS "
                f"FOR (n:{_ENTITY_LABEL}) REQUIRE (n.collection_id, n.name) IS UNIQUE"
            )
            await session.run(
                f"CREATE CONSTRAINT {_RELATION_CONSTRAINT_NAME} IF NOT EXISTS "
                f"FOR (r:{_RELATION_LABEL}) "
                f"REQUIRE (r.collection_id, r.source, r.target, r.relation_type) IS UNIQUE"
            )

    # -- find-by-document scans (pre-rebuild phase) -------------------

    async def find_entity_ids_with_lineage(self, *, document_id: str) -> list[str]:
        query = (
            f"MATCH (n:{_ENTITY_LABEL} {{collection_id: $collection_id}}) "
            f"WHERE $document_id IN n.source_lineage_doc_ids "
            f"RETURN n.name AS name"
        )
        async with self._session() as session:
            result = await session.run(
                query,
                collection_id=self._collection_id,
                document_id=document_id,
            )
            return [rec["name"] async for rec in result]

    async def find_relation_keys_with_lineage(self, *, document_id: str) -> list[tuple[str, str, str]]:
        query = (
            f"MATCH (r:{_RELATION_LABEL} {{collection_id: $collection_id}}) "
            f"WHERE $document_id IN r.evidence_lineage_doc_ids "
            f"RETURN r.source AS source, r.target AS target, r.relation_type AS relation_type"
        )
        async with self._session() as session:
            result = await session.run(
                query,
                collection_id=self._collection_id,
                document_id=document_id,
            )
            return [(rec["source"], rec["target"], rec["relation_type"]) async for rec in result]

    # -- strip-by-document (pre-rebuild phase) ------------------------

    async def remove_entity_lineage_member(self, *, entity_name: str, document_id: str) -> None:
        # Single MATCH … SET statement: list-comprehension filters keep
        # the indices that don't match $document_id and projects every
        # parallel list onto those indices. Two concurrent syncs cannot
        # race because Neo4j locks the row for the SET.
        query = (
            f"MATCH (n:{_ENTITY_LABEL} {{collection_id: $collection_id, name: $name}}) "
            f"WITH n, "
            f"  [i IN range(0, size(n.source_lineage_doc_ids) - 1) "
            f"   WHERE n.source_lineage_doc_ids[i] <> $document_id] AS sl_keep, "
            f"  [i IN range(0, size(n.description_parts_doc_ids) - 1) "
            f"   WHERE n.description_parts_doc_ids[i] <> $document_id] AS dp_keep "
            f"SET n.source_lineage = [i IN sl_keep | n.source_lineage[i]], "
            f"    n.source_lineage_doc_ids = [i IN sl_keep | n.source_lineage_doc_ids[i]], "
            f"    n.source_lineage_parse_versions = [i IN sl_keep | n.source_lineage_parse_versions[i]], "
            f"    n.description_parts = [i IN dp_keep | n.description_parts[i]], "
            f"    n.description_parts_doc_ids = [i IN dp_keep | n.description_parts_doc_ids[i]], "
            f"    n.description_parts_parse_versions = [i IN dp_keep | n.description_parts_parse_versions[i]], "
            f"    n.gmt_updated = datetime()"
        )
        async with self._session() as session:
            await session.run(
                query,
                collection_id=self._collection_id,
                name=entity_name,
                document_id=document_id,
            )

    async def remove_relation_lineage_member(self, *, source: str, target: str, type: str, document_id: str) -> None:
        query = (
            f"MATCH (r:{_RELATION_LABEL} "
            f"  {{collection_id: $collection_id, source: $source, target: $target, relation_type: $relation_type}}) "
            f"WITH r, "
            f"  [i IN range(0, size(r.evidence_lineage_doc_ids) - 1) "
            f"   WHERE r.evidence_lineage_doc_ids[i] <> $document_id] AS el_keep, "
            f"  [i IN range(0, size(r.description_parts_doc_ids) - 1) "
            f"   WHERE r.description_parts_doc_ids[i] <> $document_id] AS dp_keep "
            f"SET r.evidence_lineage = [i IN el_keep | r.evidence_lineage[i]], "
            f"    r.evidence_lineage_doc_ids = [i IN el_keep | r.evidence_lineage_doc_ids[i]], "
            f"    r.evidence_lineage_parse_versions = [i IN el_keep | r.evidence_lineage_parse_versions[i]], "
            f"    r.description_parts = [i IN dp_keep | r.description_parts[i]], "
            f"    r.description_parts_doc_ids = [i IN dp_keep | r.description_parts_doc_ids[i]], "
            f"    r.description_parts_parse_versions = [i IN dp_keep | r.description_parts_parse_versions[i]], "
            f"    r.gmt_updated = datetime()"
        )
        async with self._session() as session:
            await session.run(
                query,
                collection_id=self._collection_id,
                source=source,
                target=target,
                relation_type=type,
                document_id=document_id,
            )

    # -- GC (post-rebuild) --------------------------------------------

    async def gc_entity_if_orphan(self, entity_name: str) -> bool:
        query = (
            f"MATCH (n:{_ENTITY_LABEL} {{collection_id: $collection_id, name: $name}}) "
            f"WHERE size(n.source_lineage) = 0 "
            f"DELETE n "
            f"RETURN count(n) AS deleted"
        )
        async with self._session() as session:
            result = await session.run(query, collection_id=self._collection_id, name=entity_name)
            rec = await result.single()
            return bool(rec and rec["deleted"] > 0)

    async def gc_relation_if_orphan(self, source: str, target: str, type: str) -> bool:
        query = (
            f"MATCH (r:{_RELATION_LABEL} "
            f"  {{collection_id: $collection_id, source: $source, target: $target, relation_type: $relation_type}}) "
            f"WHERE size(r.evidence_lineage) = 0 "
            f"DELETE r "
            f"RETURN count(r) AS deleted"
        )
        async with self._session() as session:
            result = await session.run(
                query,
                collection_id=self._collection_id,
                source=source,
                target=target,
                relation_type=type,
            )
            rec = await result.single()
            return bool(rec and rec["deleted"] > 0)

    # -- unconditional delete (Wave 7 W7-1, used by curation merge) ---

    async def delete_entity(self, entity_name: str) -> bool:
        query = (
            f"MATCH (n:{_ENTITY_LABEL} {{collection_id: $collection_id, name: $name}}) "
            f"DELETE n "
            f"RETURN count(n) AS deleted"
        )
        async with self._session() as session:
            result = await session.run(query, collection_id=self._collection_id, name=entity_name)
            rec = await result.single()
            return bool(rec and rec["deleted"] > 0)

    async def delete_relation(self, source: str, target: str, type: str) -> bool:
        query = (
            f"MATCH (r:{_RELATION_LABEL} "
            f"  {{collection_id: $collection_id, source: $source, target: $target, relation_type: $relation_type}}) "
            f"DELETE r "
            f"RETURN count(r) AS deleted"
        )
        async with self._session() as session:
            result = await session.run(
                query,
                collection_id=self._collection_id,
                source=source,
                target=target,
                relation_type=type,
            )
            rec = await result.single()
            return bool(rec and rec["deleted"] > 0)

    # -- upserts (rebuild phase) --------------------------------------

    async def upsert_entity_with_lineage(
        self,
        *,
        record: EntityRecord,
        lineage: LineageMember,
        compacted_description: str | None = None,
    ) -> None:
        """Add (or replace by ``(document_id, parse_version)`` key) the
        lineage member + its description part. Single Cypher statement
        so two concurrent rebuilds against the same entity cannot race
        on read-modify-write — Neo4j MERGE locks the merged row for the
        duration of the trailing SET.

        Wave 7 W7-1: ``compacted_description=None`` (default) preserves
        any existing column value via ``COALESCE``; non-None overwrites.
        """
        member_json = _lineage_member_json(lineage)
        part_json = _description_part_json(
            DescriptionPart(
                document_id=lineage.document_id,
                parse_version=lineage.parse_version,
                text=record.description,
            )
        )
        query = (
            f"MERGE (n:{_ENTITY_LABEL} {{collection_id: $collection_id, name: $name}}) "
            # initialise on create — start with empty parallel lists so
            # the strip-then-append below has a defined input list.
            # ``compacted_description`` initialises to NULL on create
            # (W7-1: derived field; the trailing ``COALESCE`` SET below
            # writes a non-None value if ``$compacted_description`` is
            # supplied).
            f"ON CREATE SET "
            f"  n.source_lineage = [], "
            f"  n.source_lineage_doc_ids = [], "
            f"  n.source_lineage_parse_versions = [], "
            f"  n.description_parts = [], "
            f"  n.description_parts_doc_ids = [], "
            f"  n.description_parts_parse_versions = [], "
            f"  n.compacted_description = NULL, "
            f"  n.gmt_created = datetime() "
            f"WITH n, "
            f"  [i IN range(0, size(n.source_lineage_doc_ids) - 1) "
            f"   WHERE NOT (n.source_lineage_doc_ids[i] = $document_id "
            f"              AND n.source_lineage_parse_versions[i] = $parse_version)] AS sl_keep, "
            f"  [i IN range(0, size(n.description_parts_doc_ids) - 1) "
            f"   WHERE NOT (n.description_parts_doc_ids[i] = $document_id "
            f"              AND n.description_parts_parse_versions[i] = $parse_version)] AS dp_keep "
            f"SET n.entity_type = $entity_type, "
            f"    n.source_lineage = [i IN sl_keep | n.source_lineage[i]] + [$member_json], "
            f"    n.source_lineage_doc_ids = [i IN sl_keep | n.source_lineage_doc_ids[i]] + [$document_id], "
            f"    n.source_lineage_parse_versions = "
            f"      [i IN sl_keep | n.source_lineage_parse_versions[i]] + [$parse_version], "
            f"    n.description_parts = [i IN dp_keep | n.description_parts[i]] + [$part_json], "
            f"    n.description_parts_doc_ids = "
            f"      [i IN dp_keep | n.description_parts_doc_ids[i]] + [$document_id], "
            f"    n.description_parts_parse_versions = "
            f"      [i IN dp_keep | n.description_parts_parse_versions[i]] + [$parse_version], "
            # W7-1: COALESCE preserves existing if param is NULL, else
            # overwrites. On CREATE both sides are NULL → still NULL.
            f"    n.compacted_description = COALESCE($compacted_description, n.compacted_description), "
            f"    n.gmt_updated = datetime()"
        )
        async with self._session() as session:
            await session.run(
                query,
                collection_id=self._collection_id,
                name=record.name,
                entity_type=record.entity_type,
                document_id=lineage.document_id,
                parse_version=lineage.parse_version,
                member_json=member_json,
                part_json=part_json,
                compacted_description=compacted_description,
            )

    async def upsert_relation_with_lineage(
        self,
        *,
        record: RelationRecord,
        lineage: LineageMember,
        compacted_description: str | None = None,
    ) -> None:
        member_json = _lineage_member_json(lineage)
        part_json = _description_part_json(
            DescriptionPart(
                document_id=lineage.document_id,
                parse_version=lineage.parse_version,
                text=record.description,
            )
        )
        query = (
            f"MERGE (r:{_RELATION_LABEL} "
            f"  {{collection_id: $collection_id, source: $source, target: $target, relation_type: $relation_type}}) "
            f"ON CREATE SET "
            f"  r.evidence_lineage = [], "
            f"  r.evidence_lineage_doc_ids = [], "
            f"  r.evidence_lineage_parse_versions = [], "
            f"  r.description_parts = [], "
            f"  r.description_parts_doc_ids = [], "
            f"  r.description_parts_parse_versions = [], "
            f"  r.compacted_description = NULL, "
            f"  r.gmt_created = datetime() "
            f"WITH r, "
            f"  [i IN range(0, size(r.evidence_lineage_doc_ids) - 1) "
            f"   WHERE NOT (r.evidence_lineage_doc_ids[i] = $document_id "
            f"              AND r.evidence_lineage_parse_versions[i] = $parse_version)] AS el_keep, "
            f"  [i IN range(0, size(r.description_parts_doc_ids) - 1) "
            f"   WHERE NOT (r.description_parts_doc_ids[i] = $document_id "
            f"              AND r.description_parts_parse_versions[i] = $parse_version)] AS dp_keep "
            f"SET r.evidence_lineage = [i IN el_keep | r.evidence_lineage[i]] + [$member_json], "
            f"    r.evidence_lineage_doc_ids = "
            f"      [i IN el_keep | r.evidence_lineage_doc_ids[i]] + [$document_id], "
            f"    r.evidence_lineage_parse_versions = "
            f"      [i IN el_keep | r.evidence_lineage_parse_versions[i]] + [$parse_version], "
            f"    r.description_parts = [i IN dp_keep | r.description_parts[i]] + [$part_json], "
            f"    r.description_parts_doc_ids = "
            f"      [i IN dp_keep | r.description_parts_doc_ids[i]] + [$document_id], "
            f"    r.description_parts_parse_versions = "
            f"      [i IN dp_keep | r.description_parts_parse_versions[i]] + [$parse_version], "
            f"    r.compacted_description = COALESCE($compacted_description, r.compacted_description), "
            f"    r.gmt_updated = datetime()"
        )
        async with self._session() as session:
            await session.run(
                query,
                collection_id=self._collection_id,
                source=record.source,
                target=record.target,
                relation_type=record.relation_type,
                document_id=lineage.document_id,
                parse_version=lineage.parse_version,
                member_json=member_json,
                part_json=part_json,
                compacted_description=compacted_description,
            )

    # -- read-path ----------------------------------------------------

    async def get_entity(self, entity_name: str) -> EntityWithLineage | None:
        query = (
            f"MATCH (n:{_ENTITY_LABEL} {{collection_id: $collection_id, name: $name}}) "
            f"RETURN n.name AS name, n.entity_type AS entity_type, "
            f"       n.source_lineage AS source_lineage, "
            f"       n.description_parts AS description_parts, "
            f"       n.compacted_description AS compacted_description"
        )
        async with self._session() as session:
            result = await session.run(query, collection_id=self._collection_id, name=entity_name)
            rec = await result.single()
            if rec is None:
                return None
            return EntityWithLineage(
                name=rec["name"],
                entity_type=rec["entity_type"] or "",
                source_lineage=tuple(LineageMember.from_dict(json.loads(s)) for s in (rec["source_lineage"] or [])),
                description_parts=tuple(
                    DescriptionPart.from_dict(json.loads(s)) for s in (rec["description_parts"] or [])
                ),
                compacted_description=rec["compacted_description"],
            )

    async def get_relation(self, source: str, target: str, type: str) -> RelationWithLineage | None:
        query = (
            f"MATCH (r:{_RELATION_LABEL} "
            f"  {{collection_id: $collection_id, source: $source, target: $target, relation_type: $relation_type}}) "
            f"RETURN r.source AS source, r.target AS target, r.relation_type AS relation_type, "
            f"       r.evidence_lineage AS evidence_lineage, "
            f"       r.description_parts AS description_parts, "
            f"       r.compacted_description AS compacted_description"
        )
        async with self._session() as session:
            result = await session.run(
                query,
                collection_id=self._collection_id,
                source=source,
                target=target,
                relation_type=type,
            )
            rec = await result.single()
            if rec is None:
                return None
            return RelationWithLineage(
                source=rec["source"],
                target=rec["target"],
                relation_type=rec["relation_type"],
                evidence_lineage=tuple(LineageMember.from_dict(json.loads(s)) for s in (rec["evidence_lineage"] or [])),
                description_parts=tuple(
                    DescriptionPart.from_dict(json.loads(s)) for s in (rec["description_parts"] or [])
                ),
                compacted_description=rec["compacted_description"],
            )

    # -- LightRAG-style query layer (Wave 6 #33 chunk 2) --------------

    async def query_entities_by_keyword(
        self,
        *,
        query: str,
        top_k: int,
    ) -> list[EntityWithLineage]:
        if not query or not query.strip() or top_k <= 0:
            return []
        # Cypher's CONTAINS is case-sensitive; we lowercase both sides
        # to honour the Protocol contract of "case-insensitive lexical
        # recall". Cypher 5.x has no built-in case-insensitive CONTAINS,
        # so this is the canonical idiom.
        # The Cypher bind variable is named ``$keyword`` (NOT ``$query``)
        # because ``AsyncSession.run`` reserves the kwarg ``query`` for
        # the Cypher string itself; passing ``query=...`` as a parameter
        # raises ``TypeError: AsyncSession.run() got multiple values for
        # argument 'query'``. Renaming the bind avoids the clash without
        # changing the Protocol method signature.
        cypher = (
            f"MATCH (n:{_ENTITY_LABEL} {{collection_id: $collection_id}}) "
            f"WHERE toLower(n.name) CONTAINS toLower($keyword) "
            f"RETURN n.name AS name, n.entity_type AS entity_type, "
            f"       n.source_lineage AS source_lineage, "
            f"       n.description_parts AS description_parts, "
            f"       n.compacted_description AS compacted_description "
            f"ORDER BY n.name "
            f"LIMIT $top_k"
        )
        async with self._session() as session:
            result = await session.run(
                cypher,
                collection_id=self._collection_id,
                keyword=query.strip(),
                top_k=int(top_k),
            )
            return [
                EntityWithLineage(
                    name=rec["name"],
                    entity_type=rec["entity_type"] or "",
                    source_lineage=tuple(LineageMember.from_dict(json.loads(s)) for s in (rec["source_lineage"] or [])),
                    description_parts=tuple(
                        DescriptionPart.from_dict(json.loads(s)) for s in (rec["description_parts"] or [])
                    ),
                    compacted_description=rec["compacted_description"],
                )
                async for rec in result
            ]

    async def expand_neighbors_n_hops(
        self,
        *,
        entity_names: list[str],
        hops: int = 1,
    ) -> tuple[list[EntityWithLineage], list[RelationWithLineage]]:
        if not entity_names:
            return ([], [])

        seen_entities: dict[str, EntityWithLineage] = {}
        seen_relations: dict[tuple[str, str, str], RelationWithLineage] = {}

        async def _fetch_entities(session, names: list[str]) -> None:
            if not names:
                return
            cypher = (
                f"MATCH (n:{_ENTITY_LABEL} {{collection_id: $collection_id}}) "
                f"WHERE n.name IN $names "
                f"RETURN n.name AS name, n.entity_type AS entity_type, "
                f"       n.source_lineage AS source_lineage, "
                f"       n.description_parts AS description_parts, "
                f"       n.compacted_description AS compacted_description"
            )
            result = await session.run(cypher, collection_id=self._collection_id, names=names)
            async for rec in result:
                if rec["name"] in seen_entities:
                    continue
                seen_entities[rec["name"]] = EntityWithLineage(
                    name=rec["name"],
                    entity_type=rec["entity_type"] or "",
                    source_lineage=tuple(LineageMember.from_dict(json.loads(s)) for s in (rec["source_lineage"] or [])),
                    description_parts=tuple(
                        DescriptionPart.from_dict(json.loads(s)) for s in (rec["description_parts"] or [])
                    ),
                    compacted_description=rec["compacted_description"],
                )

        async def _fetch_relations_touching(session, names: list[str]) -> set[str]:
            if not names:
                return set()
            cypher = (
                f"MATCH (r:{_RELATION_LABEL} {{collection_id: $collection_id}}) "
                f"WHERE r.source IN $names OR r.target IN $names "
                f"RETURN r.source AS source, r.target AS target, r.relation_type AS relation_type, "
                f"       r.evidence_lineage AS evidence_lineage, "
                f"       r.description_parts AS description_parts, "
                f"       r.compacted_description AS compacted_description"
            )
            result = await session.run(cypher, collection_id=self._collection_id, names=names)
            next_frontier: set[str] = set()
            async for rec in result:
                key = (rec["source"], rec["target"], rec["relation_type"])
                if key not in seen_relations:
                    seen_relations[key] = RelationWithLineage(
                        source=rec["source"],
                        target=rec["target"],
                        relation_type=rec["relation_type"],
                        evidence_lineage=tuple(
                            LineageMember.from_dict(json.loads(s)) for s in (rec["evidence_lineage"] or [])
                        ),
                        description_parts=tuple(
                            DescriptionPart.from_dict(json.loads(s)) for s in (rec["description_parts"] or [])
                        ),
                        compacted_description=rec["compacted_description"],
                    )
                for endpoint in (rec["source"], rec["target"]):
                    if endpoint not in seen_entities and endpoint not in next_frontier:
                        next_frontier.add(endpoint)
            return next_frontier

        async with self._session() as session:
            current = [n for n in entity_names if n]
            await _fetch_entities(session, current)

            for _ in range(max(hops, 0)):
                next_frontier = await _fetch_relations_touching(session, current)
                if not next_frontier:
                    break
                next_list = list(next_frontier)
                await _fetch_entities(session, next_list)
                current = next_list

        entities = sorted(seen_entities.values(), key=lambda e: e.name)
        relations = sorted(seen_relations.values(), key=lambda r: (r.source, r.target, r.relation_type))
        return (entities, relations)

    # -- UI label list (Wave 6 #40 narrow replacement) -----------------

    async def list_entity_labels(self) -> list[str]:
        cypher = (
            f"MATCH (n:{_ENTITY_LABEL} {{collection_id: $collection_id}}) "
            f"WHERE n.entity_type IS NOT NULL "
            f"RETURN DISTINCT n.entity_type AS label "
            f"ORDER BY label"
        )
        async with self._session() as session:
            result = await session.run(cypher, collection_id=self._collection_id)
            return [rec["label"] async for rec in result if rec["label"]]


__all__ = [
    "Neo4jLineageGraphStore",
]
