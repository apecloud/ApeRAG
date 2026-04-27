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

"""PostgreSQL implementation of :class:`LineageGraphStore` — Wave 4 T8 chunk 1.

Stores entities and relations as two tables (``aperag_lineage_entity``
and ``aperag_lineage_relation``) with JSONB columns for the
``source_lineage`` / ``evidence_lineage`` SETs and the
``description_parts`` lists. Both tables include a
``collection_id`` column so a process-wide store instance handles
all collections; the SQLAlchemy queries always filter on
``collection_id`` to keep tenants isolated. Each store instance is
constructed with the ``collection_id`` it serves; the worker_factory
caches one instance per collection.

§D.3.5 Protocol semantics mapped to SQL:

* ``find_entity_ids_with_lineage(document_id)`` → SELECT name FROM
  entity WHERE collection_id=$c AND source_lineage @>
  '[{"document_id": "$d"}]' (JSONB containment filter).
* ``remove_entity_lineage_member(entity_name, document_id)`` →
  UPDATE entity SET source_lineage = (SELECT jsonb_agg(elem) FROM
  jsonb_array_elements(source_lineage) elem WHERE elem->>'document_id'
  != $d), description_parts = (same shape) WHERE collection_id=$c AND
  name=$n. Single statement so concurrent syncs cannot race on the
  read-modify-write.
* ``upsert_entity_with_lineage`` → INSERT … ON CONFLICT DO UPDATE
  with a ``COALESCE`` + ``jsonb_build_object`` expression that
  inserts the new ``LineageMember`` if the
  ``(document_id, parse_version)`` key is absent, replaces the
  matching element otherwise. Same shape for ``DescriptionPart``.
* ``gc_*_if_orphan`` → DELETE … WHERE jsonb_array_length(...) = 0.

Every method opens its own ``engine.begin()`` transaction so a
caller that wants to batch must do that at a higher layer; the
adapter keeps the boundary crisp (per legacy ``Postgres.GraphStore``
convention).

Hard-cut second round: this module replaces
``aperag/domains/knowledge_graph/graphindex/storage/postgres.py``;
the legacy file is deleted in the same PR (chunk 4).
"""

from __future__ import annotations

import json
import logging

from sqlalchemy import (
    Column,
    DateTime,
    Index,
    String,
    select,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlalchemy.orm import declarative_base

from aperag.indexing.graph import (
    DescriptionPart,
    EntityRecord,
    EntityWithLineage,
    LineageMember,
    RelationRecord,
    RelationWithLineage,
)

# utc_now is no longer needed: Wave 5 P5B switched ORM gmt_* columns
# to ``server_default=CURRENT_TIMESTAMP`` mirroring the alembic
# migration declaration.

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# ORM models — declared on a private ``Base`` so the adapter can own
# its own metadata without colliding with the application's main
# ``aperag.db.base.Base``. Alembic migration owns the actual DDL; the
# ``ensure_schema`` method below is a fallback for tests / dev.
# ---------------------------------------------------------------------


_LineageGraphBase = declarative_base()

ENTITY_TABLE = "aperag_lineage_entity"
RELATION_TABLE = "aperag_lineage_relation"


class _LineageEntityRow(_LineageGraphBase):
    __tablename__ = ENTITY_TABLE
    # The ``(collection_id, name)`` composite primary key already
    # enforces uniqueness; the upsert SQL uses the column-list
    # ``ON CONFLICT (collection_id, name)`` form so no separate
    # named UniqueConstraint is required. The GIN index accelerates
    # the JSONB ``@>`` containment scan in
    # ``find_entity_ids_with_lineage`` against multi-tenant tables.
    __table_args__ = (
        Index(
            "idx_lineage_entity_source_lineage_gin",
            "source_lineage",
            postgresql_using="gin",
        ),
    )

    collection_id = Column(String(64), primary_key=True)
    name = Column(String(512), primary_key=True)
    # Wave 6 #36: column renamed ``type`` → ``entity_type`` to free the
    # bare ``type`` name across the §D.3 Protocol surface (frees Python
    # ``type`` builtin from ORM attribute shadow). Alembic migration
    # ``f8a4c5b9d3e1`` performs the column rename hard-cut (no
    # production data per earayu2 msg=30c81478).
    entity_type = Column(String(64), nullable=False)
    source_lineage = Column(JSONB, nullable=False, server_default=text("'[]'::jsonb"))
    description_parts = Column(JSONB, nullable=False, server_default=text("'[]'::jsonb"))
    # Wave 5 P5B: ORM uses the same ``server_default=CURRENT_TIMESTAMP``
    # the alembic migration declares — strict ORM↔migration mirror so
    # ``alembic check`` cannot drift on schema-touching follow-ups.
    # Pre-Wave-5 ORM used ``default=utc_now`` Python-side; alembic
    # check passed because Postgres treats both as semantic-equivalent
    # but the per-mirror discipline is stronger when both layers
    # speak the same dialect.
    gmt_created = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP"),
    )
    gmt_updated = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP"),
    )


class _LineageRelationRow(_LineageGraphBase):
    __tablename__ = RELATION_TABLE
    __table_args__ = (
        Index(
            "idx_lineage_relation_evidence_lineage_gin",
            "evidence_lineage",
            postgresql_using="gin",
        ),
    )

    collection_id = Column(String(64), primary_key=True)
    source = Column(String(512), primary_key=True)
    target = Column(String(512), primary_key=True)
    # Wave 6 #36: column renamed ``type`` → ``relation_type``.
    relation_type = Column(String(64), primary_key=True)
    # Wave 4 chunk 4 dropped the legacy ``description`` Text column —
    # the per-document fragments in ``description_parts`` are the
    # canonical source. ``RelationWithLineage`` does not expose a
    # standalone ``description`` field so the column had no consumer.
    evidence_lineage = Column(JSONB, nullable=False, server_default=text("'[]'::jsonb"))
    description_parts = Column(JSONB, nullable=False, server_default=text("'[]'::jsonb"))
    # Wave 5 P5B: same ORM↔migration mirror discipline as
    # ``_LineageEntityRow`` above.
    gmt_created = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP"),
    )
    gmt_updated = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP"),
    )


# ---------------------------------------------------------------------
# Helpers — JSONB SET-element manipulation. Centralised so the SQL
# expressions are inspectable and tests can call them directly.
# ---------------------------------------------------------------------


def _lineage_to_json_array(lineage: LineageMember) -> str:
    """Serialise one :class:`LineageMember` to the canonical JSON
    fragment stored in the ``source_lineage`` / ``evidence_lineage``
    JSONB array. The key for SET-element dedup is
    ``(document_id, parse_version)``."""
    return json.dumps(lineage.to_dict())


def _description_part_to_json(part: DescriptionPart) -> str:
    return json.dumps(part.to_dict())


# ---------------------------------------------------------------------
# Public adapter.
# ---------------------------------------------------------------------


class PostgresLineageGraphStore:
    """:class:`LineageGraphStore` implementation backed by PostgreSQL.

    One instance per ``collection_id``; the
    ``worker_factory`` caches instances. The adapter owns no
    schema-creation responsibility in production — alembic migration
    ``aperag_lineage_graph_postgres`` (Wave 4 T8) creates the tables.
    The :meth:`ensure_schema` helper is for tests + dev.
    """

    def __init__(self, *, engine: AsyncEngine, collection_id: str) -> None:
        self._engine = engine
        self._collection_id = collection_id

    # -- schema -------------------------------------------------------

    async def ensure_schema(self) -> None:
        """Create tables if missing. Idempotent. Production uses
        alembic; this is the test / dev convenience.
        """
        async with self._engine.begin() as conn:
            await conn.run_sync(_LineageGraphBase.metadata.create_all)

    # -- find-by-document scans (pre-rebuild phase) -------------------

    async def find_entity_ids_with_lineage(self, *, document_id: str) -> list[str]:
        """Return entity names with a lineage member matching
        ``document_id`` (any parse_version)."""
        # JSONB containment ``@>`` matches when the right side is a
        # subset of the left; we wrap the document_id object in a
        # one-element array so the operator scans the lineage SET.
        # ``CAST(:document_id AS TEXT)`` makes the asyncpg adapter
        # infer the parameter type — without the cast asyncpg sees
        # an indeterminate datatype because the parameter sits inside
        # ``jsonb_build_object`` (which accepts ``ANY``).
        sql = text(
            f"""
            SELECT name
            FROM {ENTITY_TABLE}
            WHERE collection_id = :collection_id
              AND source_lineage @> jsonb_build_array(
                    jsonb_build_object('document_id', CAST(:document_id AS TEXT))
                  )
            """
        )
        async with self._engine.connect() as conn:
            result = await conn.execute(sql, {"collection_id": self._collection_id, "document_id": document_id})
            return [row.name for row in result]

    async def find_relation_keys_with_lineage(self, *, document_id: str) -> list[tuple[str, str, str]]:
        sql = text(
            f"""
            SELECT source, target, relation_type
            FROM {RELATION_TABLE}
            WHERE collection_id = :collection_id
              AND evidence_lineage @> jsonb_build_array(
                    jsonb_build_object('document_id', CAST(:document_id AS TEXT))
                  )
            """
        )
        async with self._engine.connect() as conn:
            result = await conn.execute(sql, {"collection_id": self._collection_id, "document_id": document_id})
            return [(row.source, row.target, row.relation_type) for row in result]

    # -- strip-by-document (pre-rebuild phase) ------------------------

    async def remove_entity_lineage_member(self, *, entity_name: str, document_id: str) -> None:
        """Strip every ``LineageMember`` and ``DescriptionPart`` whose
        document_id matches. Single SQL statement so concurrent syncs
        do not race on read-modify-write."""
        sql = text(
            f"""
            UPDATE {ENTITY_TABLE}
            SET source_lineage = COALESCE(
                    (SELECT jsonb_agg(elem) FROM jsonb_array_elements(source_lineage) elem
                     WHERE elem->>'document_id' != :document_id),
                    '[]'::jsonb
                ),
                description_parts = COALESCE(
                    (SELECT jsonb_agg(part) FROM jsonb_array_elements(description_parts) part
                     WHERE part->>'document_id' != :document_id),
                    '[]'::jsonb
                ),
                gmt_updated = NOW()
            WHERE collection_id = :collection_id AND name = :name
            """
        )
        async with self._engine.begin() as conn:
            await conn.execute(
                sql,
                {
                    "collection_id": self._collection_id,
                    "name": entity_name,
                    "document_id": document_id,
                },
            )

    async def remove_relation_lineage_member(self, *, source: str, target: str, type: str, document_id: str) -> None:
        sql = text(
            f"""
            UPDATE {RELATION_TABLE}
            SET evidence_lineage = COALESCE(
                    (SELECT jsonb_agg(elem) FROM jsonb_array_elements(evidence_lineage) elem
                     WHERE elem->>'document_id' != :document_id),
                    '[]'::jsonb
                ),
                description_parts = COALESCE(
                    (SELECT jsonb_agg(part) FROM jsonb_array_elements(description_parts) part
                     WHERE part->>'document_id' != :document_id),
                    '[]'::jsonb
                ),
                gmt_updated = NOW()
            WHERE collection_id = :collection_id
              AND source = :source AND target = :target AND relation_type = :relation_type
            """
        )
        async with self._engine.begin() as conn:
            await conn.execute(
                sql,
                {
                    "collection_id": self._collection_id,
                    "source": source,
                    "target": target,
                    "relation_type": type,
                    "document_id": document_id,
                },
            )

    # -- GC (post-rebuild) --------------------------------------------

    async def gc_entity_if_orphan(self, entity_name: str) -> bool:
        sql = text(
            f"""
            DELETE FROM {ENTITY_TABLE}
            WHERE collection_id = :collection_id
              AND name = :name
              AND jsonb_array_length(source_lineage) = 0
            """
        )
        async with self._engine.begin() as conn:
            result = await conn.execute(sql, {"collection_id": self._collection_id, "name": entity_name})
            return (result.rowcount or 0) > 0

    async def gc_relation_if_orphan(self, source: str, target: str, type: str) -> bool:
        sql = text(
            f"""
            DELETE FROM {RELATION_TABLE}
            WHERE collection_id = :collection_id
              AND source = :source AND target = :target AND relation_type = :relation_type
              AND jsonb_array_length(evidence_lineage) = 0
            """
        )
        async with self._engine.begin() as conn:
            result = await conn.execute(
                sql,
                {
                    "collection_id": self._collection_id,
                    "source": source,
                    "target": target,
                    "relation_type": type,
                },
            )
            return (result.rowcount or 0) > 0

    # -- upserts (rebuild phase) --------------------------------------

    async def upsert_entity_with_lineage(self, *, record: EntityRecord, lineage: LineageMember) -> None:
        """Add (or replace by `(document_id, parse_version)` key)
        the lineage member + a corresponding description part. Single
        statement so concurrent rebuilds do not race."""
        member_json = _lineage_to_json_array(lineage)
        part_json = _description_part_to_json(
            DescriptionPart(
                document_id=lineage.document_id,
                parse_version=lineage.parse_version,
                text=record.description,
            )
        )
        sql = text(
            f"""
            INSERT INTO {ENTITY_TABLE} (
                collection_id, name, entity_type, source_lineage, description_parts,
                gmt_created, gmt_updated
            )
            VALUES (
                :collection_id, :name, :entity_type,
                jsonb_build_array(CAST(:member_json AS jsonb)),
                jsonb_build_array(CAST(:part_json AS jsonb)),
                NOW(), NOW()
            )
            ON CONFLICT (collection_id, name) DO UPDATE
            SET entity_type = EXCLUDED.entity_type,
                source_lineage = (
                    -- strip any existing element with the same
                    -- (document_id, parse_version) key, then append
                    -- the new one.
                    COALESCE(
                        (SELECT jsonb_agg(elem) FROM jsonb_array_elements({ENTITY_TABLE}.source_lineage) elem
                         WHERE NOT (elem->>'document_id' = :document_id AND elem->>'parse_version' = :parse_version)),
                        '[]'::jsonb
                    ) || jsonb_build_array(CAST(:member_json AS jsonb))
                ),
                description_parts = (
                    COALESCE(
                        (SELECT jsonb_agg(part) FROM jsonb_array_elements({ENTITY_TABLE}.description_parts) part
                         WHERE NOT (part->>'document_id' = :document_id AND part->>'parse_version' = :parse_version)),
                        '[]'::jsonb
                    ) || jsonb_build_array(CAST(:part_json AS jsonb))
                ),
                gmt_updated = NOW()
            """
        )
        async with self._engine.begin() as conn:
            await conn.execute(
                sql,
                {
                    "collection_id": self._collection_id,
                    "name": record.name,
                    "entity_type": record.entity_type,
                    "document_id": lineage.document_id,
                    "parse_version": lineage.parse_version,
                    "member_json": member_json,
                    "part_json": part_json,
                },
            )

    async def upsert_relation_with_lineage(self, *, record: RelationRecord, lineage: LineageMember) -> None:
        member_json = _lineage_to_json_array(lineage)
        part_json = _description_part_to_json(
            DescriptionPart(
                document_id=lineage.document_id,
                parse_version=lineage.parse_version,
                text=record.description,
            )
        )
        sql = text(
            f"""
            INSERT INTO {RELATION_TABLE} (
                collection_id, source, target, relation_type,
                evidence_lineage, description_parts,
                gmt_created, gmt_updated
            )
            VALUES (
                :collection_id, :source, :target, :relation_type,
                jsonb_build_array(CAST(:member_json AS jsonb)),
                jsonb_build_array(CAST(:part_json AS jsonb)),
                NOW(), NOW()
            )
            ON CONFLICT (collection_id, source, target, relation_type) DO UPDATE
            SET evidence_lineage = (
                    COALESCE(
                        (SELECT jsonb_agg(elem) FROM jsonb_array_elements({RELATION_TABLE}.evidence_lineage) elem
                         WHERE NOT (elem->>'document_id' = :document_id AND elem->>'parse_version' = :parse_version)),
                        '[]'::jsonb
                    ) || jsonb_build_array(CAST(:member_json AS jsonb))
                ),
                description_parts = (
                    COALESCE(
                        (SELECT jsonb_agg(part) FROM jsonb_array_elements({RELATION_TABLE}.description_parts) part
                         WHERE NOT (part->>'document_id' = :document_id AND part->>'parse_version' = :parse_version)),
                        '[]'::jsonb
                    ) || jsonb_build_array(CAST(:part_json AS jsonb))
                ),
                gmt_updated = NOW()
            """
        )
        async with self._engine.begin() as conn:
            await conn.execute(
                sql,
                {
                    "collection_id": self._collection_id,
                    "source": record.source,
                    "target": record.target,
                    "relation_type": record.relation_type,
                    "document_id": lineage.document_id,
                    "parse_version": lineage.parse_version,
                    "member_json": member_json,
                    "part_json": part_json,
                },
            )

    # -- read-path ----------------------------------------------------

    async def get_entity(self, entity_name: str) -> EntityWithLineage | None:
        async with self._engine.connect() as conn:
            result = await conn.execute(
                select(
                    _LineageEntityRow.name,
                    _LineageEntityRow.entity_type,
                    _LineageEntityRow.source_lineage,
                    _LineageEntityRow.description_parts,
                ).where(
                    _LineageEntityRow.collection_id == self._collection_id,
                    _LineageEntityRow.name == entity_name,
                )
            )
            row = result.first()
            if row is None:
                return None
            return EntityWithLineage(
                name=row.name,
                entity_type=row.entity_type,
                source_lineage=tuple(LineageMember.from_dict(elem) for elem in (row.source_lineage or [])),
                description_parts=tuple(DescriptionPart.from_dict(part) for part in (row.description_parts or [])),
            )

    async def get_relation(self, source: str, target: str, type: str) -> RelationWithLineage | None:
        async with self._engine.connect() as conn:
            result = await conn.execute(
                select(
                    _LineageRelationRow.source,
                    _LineageRelationRow.target,
                    _LineageRelationRow.relation_type,
                    _LineageRelationRow.evidence_lineage,
                    _LineageRelationRow.description_parts,
                ).where(
                    _LineageRelationRow.collection_id == self._collection_id,
                    _LineageRelationRow.source == source,
                    _LineageRelationRow.target == target,
                    _LineageRelationRow.relation_type == type,
                )
            )
            row = result.first()
            if row is None:
                return None
            return RelationWithLineage(
                source=row.source,
                target=row.target,
                relation_type=row.relation_type,
                evidence_lineage=tuple(LineageMember.from_dict(elem) for elem in (row.evidence_lineage or [])),
                description_parts=tuple(DescriptionPart.from_dict(part) for part in (row.description_parts or [])),
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
        # ILIKE substring match — case-insensitive lexical recall.
        # Backend-defined match semantics per Protocol contract.
        like = f"%{query.strip()}%"
        async with self._engine.connect() as conn:
            result = await conn.execute(
                select(
                    _LineageEntityRow.name,
                    _LineageEntityRow.entity_type,
                    _LineageEntityRow.source_lineage,
                    _LineageEntityRow.description_parts,
                )
                .where(
                    _LineageEntityRow.collection_id == self._collection_id,
                    _LineageEntityRow.name.ilike(like),
                )
                .order_by(_LineageEntityRow.name)
                .limit(top_k)
            )
            return [
                EntityWithLineage(
                    name=row.name,
                    entity_type=row.entity_type,
                    source_lineage=tuple(LineageMember.from_dict(elem) for elem in (row.source_lineage or [])),
                    description_parts=tuple(DescriptionPart.from_dict(part) for part in (row.description_parts or [])),
                )
                for row in result
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

        async def _fetch_entities(names: set[str]) -> None:
            if not names:
                return
            result = await conn.execute(
                select(
                    _LineageEntityRow.name,
                    _LineageEntityRow.entity_type,
                    _LineageEntityRow.source_lineage,
                    _LineageEntityRow.description_parts,
                ).where(
                    _LineageEntityRow.collection_id == self._collection_id,
                    _LineageEntityRow.name.in_(list(names)),
                )
            )
            for row in result:
                if row.name in seen_entities:
                    continue
                seen_entities[row.name] = EntityWithLineage(
                    name=row.name,
                    entity_type=row.entity_type,
                    source_lineage=tuple(LineageMember.from_dict(elem) for elem in (row.source_lineage or [])),
                    description_parts=tuple(DescriptionPart.from_dict(part) for part in (row.description_parts or [])),
                )

        async def _fetch_relations_touching(names: set[str]) -> set[str]:
            """Fetch relations where source or target is in ``names``.
            Returns the set of neighbour names (the OTHER endpoint not
            already in ``names``) for the next-hop frontier."""
            if not names:
                return set()
            result = await conn.execute(
                select(
                    _LineageRelationRow.source,
                    _LineageRelationRow.target,
                    _LineageRelationRow.relation_type,
                    _LineageRelationRow.evidence_lineage,
                    _LineageRelationRow.description_parts,
                ).where(
                    _LineageRelationRow.collection_id == self._collection_id,
                    (_LineageRelationRow.source.in_(list(names))) | (_LineageRelationRow.target.in_(list(names))),
                )
            )
            next_frontier: set[str] = set()
            for row in result:
                key = (row.source, row.target, row.relation_type)
                if key not in seen_relations:
                    seen_relations[key] = RelationWithLineage(
                        source=row.source,
                        target=row.target,
                        relation_type=row.relation_type,
                        evidence_lineage=tuple(LineageMember.from_dict(elem) for elem in (row.evidence_lineage or [])),
                        description_parts=tuple(
                            DescriptionPart.from_dict(part) for part in (row.description_parts or [])
                        ),
                    )
                for endpoint in (row.source, row.target):
                    if endpoint not in seen_entities and endpoint not in next_frontier:
                        next_frontier.add(endpoint)
            return next_frontier

        async with self._engine.connect() as conn:
            current = {n for n in entity_names if n}
            await _fetch_entities(current)

            for _ in range(max(hops, 0)):
                next_frontier = await _fetch_relations_touching(current)
                if not next_frontier:
                    break
                await _fetch_entities(next_frontier)
                # next iteration's seeds are the new neighbours so we
                # don't re-walk relations from the previous frontier.
                current = next_frontier

        entities = sorted(seen_entities.values(), key=lambda e: e.name)
        relations = sorted(seen_relations.values(), key=lambda r: (r.source, r.target, r.relation_type))
        return (entities, relations)


__all__ = [
    "ENTITY_TABLE",
    "PostgresLineageGraphStore",
    "RELATION_TABLE",
]
