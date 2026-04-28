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

"""Nebula Graph implementation of :class:`LineageGraphStore` — Wave 4 T8 chunk 3.

Mirrors :class:`PostgresLineageGraphStore` (chunk 1 reference) and
:class:`Neo4jLineageGraphStore` (chunk 2) over Nebula 3.x. The §D.3.5
lineage SET dedup-by-``(document_id, parse_version)`` semantic ports
to Nebula via a **JSON STRING property** encoding because Nebula's
property model does not support list-of-MAP and has only limited
list ops (per architect msg=95179f2a). Each lineage SET is serialised
into a single ``string`` column holding a JSON array; reads parse the
JSON, writes mutate in Python and write the new JSON back.

That round-trip is **inherently a read-modify-write loop** — two
concurrent ``upsert_*_with_lineage`` against the same row will race
unless serialised. Per architect msg=f2921ae0, the Nebula backend
**must** acquire an :class:`EntityLock` keyed by entity name (or
relation triple) for every read-modify-write; the Postgres / Neo4j
backends don't need this because they have native single-statement
strip-then-append. The :class:`EntityLock` is injected at construction
time so callers control the lock scope (``InMemoryEntityLock`` for
single-process tests, ``RedisEntityLock`` for multi-process production
worker pools).

Storage layout — one Nebula SPACE per ``collection_id`` (the natural
Nebula tenancy boundary, mirroring how the legacy ``NebulaGraphStore``
isolates collections; cheap and cleaner than a per-row collection_id
filter on a single space):

    SPACE ``{space_prefix}_{collection_id}``
        TAG ``lineage_entity(
            name string,
            type string,
            source_lineage_json string,
            description_parts_json string,
            gmt_created datetime,
            gmt_updated datetime
        )``
        TAG ``lineage_relation(
            source string,
            target string,
            type string,
            description string,
            evidence_lineage_json string,
            description_parts_json string,
            gmt_created datetime,
            gmt_updated datetime
        )``

Vertex VID convention (Nebula needs every vertex to have a string VID
that fits ``FIXED_STRING(N)``):

* entity vertex VID = ``"e|" + name``
* relation vertex VID = ``"r|" + source + "|" + target + "|" + type``

The ``|`` separator is escaped in inputs (URL-encode style) so the VID
parse is unambiguous. ``r|`` and ``e|`` prefixes guarantee no VID
collision between entity and relation rows even when an entity
happens to share its name with a relation triple component.

Every public method is async to honour the Protocol; the underlying
``nebula3-python`` SDK is sync and CPU/IO blocking, so each method
funnels its work through ``asyncio.to_thread`` (matching the legacy
adapter pattern). The :class:`EntityLock` acquisition is async so it
composes correctly across the thread boundary.

§D.3.5 Protocol semantics → nGQL:

* ``find_entity_ids_with_lineage(document_id)`` → ``LOOKUP ON
  lineage_entity`` to enumerate all entities, then in-Python filter
  on JSON for the ``document_id`` match. Without a JSON containment
  index, this is O(N) over the entity tag — acceptable for the
  lineage SET cardinality the §D.3 design pack describes (typically
  < 100 docs/entity); high-cardinality entities are a Wave 5 perf
  optimisation candidate.
* ``remove_entity_lineage_member(name, document_id)`` → FETCH PROP
  + JSON parse + filter members where ``document_id != X`` + UPDATE
  VERTEX with the new JSON. The whole block is wrapped in
  ``EntityLock.acquire(name)`` so two concurrent strips on the same
  row serialise.
* ``upsert_entity_with_lineage`` → MERGE-style: FETCH (or initialise
  empty) + remove any existing member with same
  ``(document_id, parse_version)`` key + append new member +
  INSERT/UPDATE VERTEX. Lock-protected.
* ``gc_*_if_orphan`` → FETCH lineage; if empty list, ``DELETE
  VERTEX``. Returns True if the delete actually happened. Also
  lock-protected so a concurrent re-upsert on the same key does not
  resurrect-then-delete.

Hard-cut second round: this module is the new Nebula adapter. The
legacy ``aperag/domains/knowledge_graph/graphindex/storage/nebula.py``
is deleted in chunk 4 of the same Wave 4 PR.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import threading
import time
from typing import Any

from aperag.indexing.graph import (
    DescriptionPart,
    EntityLock,
    EntityRecord,
    EntityWithLineage,
    LineageMember,
    RelationRecord,
    RelationWithLineage,
)

logger = logging.getLogger(__name__)


_ENTITY_TAG = "lineage_entity"
_RELATION_TAG = "lineage_relation"
_ENTITY_VID_PREFIX = "e|"
_RELATION_VID_PREFIX = "r|"

# Schema-visibility retry settings: Nebula metad propagates new tags /
# spaces to storaged on a heartbeat; freshly-created tag/space writes
# can briefly raise ``SpaceNotFound`` / ``No schema found for`` /
# ``TagNotFound`` before the heartbeat catches up.
_SCHEMA_VISIBILITY_RETRIES = 30
_SCHEMA_VISIBILITY_DELAY_SECONDS = 1.0
_SCHEMA_VISIBILITY_ERROR_FRAGMENTS = (
    "No schema found for",
    "TagNotFound",
    "EdgeNotFound",
    "SpaceNotFound",
    # Storage-side variant (text spelling differs from metad's): the
    # storaged process emits ``Storage Error: Tag not found`` /
    # ``Edge not found`` while metad emits the camelCased forms above.
    # Wave 4 chunk 4c surfaced this on rapid space-drop/recreate
    # cycles (cross-backend contract fixture's per-test SPACE setup).
    "Tag not found",
    "Edge not found",
)


def _is_schema_visibility_error(exc: BaseException) -> bool:
    msg = str(exc)
    return any(fragment in msg for fragment in _SCHEMA_VISIBILITY_ERROR_FRAGMENTS)


# Wave 7 W7-1: Nebula has no ``IF NOT EXISTS`` for ``ALTER TAG ADD``;
# running ADD against a column that already exists raises an error
# whose text differs by version ("Column already exists" / "Duplicated
# property"). The fragments below cover both 3.x phrasings so the
# idempotent ALTER survives both fresh deploys (column added by CREATE
# TAG, ALTER raises duplicate) and upgrades (column missing, ALTER
# adds).
_DUPLICATE_PROPERTY_ERROR_FRAGMENTS = (
    "Existed",
    "existed",
    "already exist",
    "Duplicated",
    "duplicated",
    "duplicate",
)


def _is_duplicate_property_error(exc: BaseException) -> bool:
    msg = str(exc)
    return any(fragment in msg for fragment in _DUPLICATE_PROPERTY_ERROR_FRAGMENTS)


# ---------------------------------------------------------------------
# Helpers — VID encoding, JSON SET manipulation, escaping.
# ---------------------------------------------------------------------


def _escape_str(s: str) -> str:
    """Escape a Python string for embedding inside an nGQL string
    literal. Uses ``json.dumps`` and strips the surrounding quotes so
    the embedded backslash / quote / unicode handling matches what the
    Nebula parser expects."""
    return json.dumps(s, ensure_ascii=False)[1:-1]


def _vid_escape_segment(s: str) -> str:
    """Escape ``|`` and backslash inside a VID segment so the
    ``r|<source>|<target>|<type>`` decomposition is unambiguous."""
    return s.replace("\\", "\\\\").replace("|", "\\p")


def _entity_vid(name: str) -> str:
    return _ENTITY_VID_PREFIX + _vid_escape_segment(name)


def _relation_vid(source: str, target: str, type: str) -> str:
    return (
        _RELATION_VID_PREFIX
        + _vid_escape_segment(source)
        + "|"
        + _vid_escape_segment(target)
        + "|"
        + _vid_escape_segment(type)
    )


def _space_name(prefix: str, collection_id: str) -> str:
    """Sanitise the collection_id so it can be embedded in a Nebula
    SPACE name (Nebula identifiers must match ``[a-zA-Z0-9_]``)."""
    safe_id = re.sub(r"[^a-zA-Z0-9_]", "_", collection_id)
    return f"{prefix}_{safe_id}"


def _members_to_json(members: list[LineageMember]) -> str:
    return json.dumps([m.to_dict() for m in members])


def _parts_to_json(parts: list[DescriptionPart]) -> str:
    return json.dumps([p.to_dict() for p in parts])


def _members_from_json(raw: str) -> list[LineageMember]:
    if not raw:
        return []
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return []
    if not isinstance(data, list):
        return []
    return [LineageMember.from_dict(item) for item in data if isinstance(item, dict)]


def _parts_from_json(raw: str) -> list[DescriptionPart]:
    if not raw:
        return []
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return []
    if not isinstance(data, list):
        return []
    return [DescriptionPart.from_dict(item) for item in data if isinstance(item, dict)]


# ---------------------------------------------------------------------
# Public adapter.
# ---------------------------------------------------------------------


class NebulaLineageGraphStore:
    """:class:`LineageGraphStore` implementation backed by Nebula 3.x.

    Key design decisions (per architect msg=95179f2a + msg=f2921ae0):

    1. **JSON STRING property** for the lineage SET — Nebula property
       types are scalars + ``string``; no list-of-map. Each SET is one
       ``string`` column holding a JSON array.
    2. **Per-entity / per-relation lock for read-modify-write** — JSON
       round-trip is inherently racy across concurrent syncs;
       :class:`EntityLock` makes the serialisation explicit.
    3. **One SPACE per collection_id** — natural Nebula tenancy
       (cheap; ``DROP SPACE`` is the fastest way to wipe a collection).
       The store-instance binds to a ``collection_id`` at construction
       time, mirroring the per-store-instance binding pattern used by
       :class:`PostgresLineageGraphStore` and
       :class:`Neo4jLineageGraphStore` (per architect msg=95179f2a
       Design point 2).

    The constructor accepts a Nebula 3.x ``ConnectionPool`` (the
    caller — typically the worker_factory — controls pool lifecycle)
    plus username / password / collection_id / entity_lock. Production
    deployments inject a :class:`RedisEntityLock` so the lock
    serialises across worker processes; tests inject the
    :class:`InMemoryEntityLock` reference impl.
    """

    def __init__(
        self,
        *,
        pool: Any,
        username: str,
        password: str,
        collection_id: str,
        entity_lock: EntityLock,
        space_prefix: str = "aperag_lineage",
    ) -> None:
        self._pool = pool
        self._username = username
        self._password = password
        self._collection_id = collection_id
        self._entity_lock = entity_lock
        self._space_prefix = space_prefix
        self._space = _space_name(space_prefix, collection_id)
        self._schema_init_lock = threading.Lock()
        self._schema_initialised = False

    # -- low-level session execution ----------------------------------

    def _execute(self, space: str, stmt: str) -> Any:
        """Run a single nGQL statement in a sync session. Raises
        ``RuntimeError`` on Nebula-side failure; the caller is
        responsible for translating known transient errors (e.g.
        schema-visibility) into retries."""
        session = self._pool.get_session(self._username, self._password)
        try:
            if space:
                use_result = session.execute(f"USE `{space}`")
                if not use_result.is_succeeded():
                    raise RuntimeError(f"Nebula USE failed: {use_result.error_msg()}")
            result = session.execute(stmt)
            if not result.is_succeeded():
                raise RuntimeError(f"Nebula query failed: {result.error_msg()}\nStatement: {stmt}")
            return result
        finally:
            session.release()

    def _execute_with_schema_retry(self, space: str, stmt: str) -> Any:
        last_error: RuntimeError | None = None
        for _ in range(_SCHEMA_VISIBILITY_RETRIES):
            try:
                return self._execute(space, stmt)
            except RuntimeError as exc:
                if not _is_schema_visibility_error(exc):
                    raise
                last_error = exc
                time.sleep(_SCHEMA_VISIBILITY_DELAY_SECONDS)
        if last_error is not None:
            raise last_error
        raise RuntimeError("Nebula schema retry loop exited without a result")

    def _space_exists(self, space: str) -> bool:
        try:
            result = self._execute("", "SHOW SPACES")
        except Exception:
            return False
        for i in range(result.row_size()):
            row = result.row_values(i)
            if row and row[0].is_string() and row[0].as_string() == space:
                return True
        return False

    # -- schema -------------------------------------------------------

    def _ensure_schema_sync(self) -> None:
        """Create SPACE + TAGs if absent. Idempotent; uses a process
        lock so concurrent `__init__`-then-`upsert` from multiple
        coroutines don't double-create.
        """
        if self._schema_initialised:
            return
        with self._schema_init_lock:
            if self._schema_initialised:
                return

            self._execute(
                "",
                f"CREATE SPACE IF NOT EXISTS `{self._space}` "
                f"(vid_type=FIXED_STRING(256), partition_num=1, replica_factor=1)",
            )

            tag_stmts = [
                # Wave 6 #36: tag-prop renamed ``type`` → ``entity_type`` /
                # ``relation_type`` per architect Pattern 3 ruling.
                # Hard-cut per earayu2 msg=30c81478 (no production data).
                #
                # Wave 7 W7-1: ``compacted_description`` is the
                # GraphIndexCompactor-derived unified description (W7-2).
                # Nullable; new deployments get it via CREATE TAG, older
                # deployments via ALTER TAG ADD below (idempotent).
                f"CREATE TAG IF NOT EXISTS `{_ENTITY_TAG}`("
                f"name string, entity_type string, "
                f"source_lineage_json string, description_parts_json string, "
                f"compacted_description string NULL, "
                f"gmt_created datetime, gmt_updated datetime)",
                f"CREATE TAG IF NOT EXISTS `{_RELATION_TAG}`("
                f"source string, target string, relation_type string, "
                f"evidence_lineage_json string, description_parts_json string, "
                f"compacted_description string NULL, "
                f"gmt_created datetime, gmt_updated datetime)",
                f"CREATE TAG INDEX IF NOT EXISTS `idx_{_ENTITY_TAG}_name` ON `{_ENTITY_TAG}`(name(256))",
                f"CREATE TAG INDEX IF NOT EXISTS `idx_{_ENTITY_TAG}_entity_type` ON `{_ENTITY_TAG}`(entity_type(128))",
                f"CREATE TAG INDEX IF NOT EXISTS `idx_{_RELATION_TAG}_source` ON `{_RELATION_TAG}`(source(256))",
                f"CREATE TAG INDEX IF NOT EXISTS `idx_{_RELATION_TAG}_target` ON `{_RELATION_TAG}`(target(256))",
            ]
            # Wave 7 W7-1 backfill: ALTER TAG ADD for tags created on a
            # pre-Wave-7 schema. Nebula has no ``IF NOT EXISTS`` for ALTER
            # TAG ADD; running it on a tag that already has the column
            # raises a "field already exists" / "duplicated property"
            # error which we swallow (idempotent contract).
            alter_stmts = [
                f"ALTER TAG `{_ENTITY_TAG}` ADD (compacted_description string NULL)",
                f"ALTER TAG `{_RELATION_TAG}` ADD (compacted_description string NULL)",
            ]
            last_error: RuntimeError | None = None
            for _ in range(_SCHEMA_VISIBILITY_RETRIES):
                try:
                    if not self._space_exists(self._space):
                        time.sleep(_SCHEMA_VISIBILITY_DELAY_SECONDS)
                        continue
                    for stmt in tag_stmts:
                        self._execute(self._space, stmt)
                    # Wave 7 W7-1: idempotent ALTER TAG for pre-Wave-7
                    # schemas. Swallow "duplicate property" errors so
                    # fresh tags (which already have the column from
                    # CREATE TAG above) don't trip the migration.
                    for stmt in alter_stmts:
                        try:
                            self._execute(self._space, stmt)
                        except RuntimeError as alter_exc:
                            if not _is_duplicate_property_error(alter_exc):
                                raise
                    # Allow heartbeat to propagate the new tags before
                    # any caller writes; otherwise the first INSERT
                    # races schema visibility.
                    time.sleep(_SCHEMA_VISIBILITY_DELAY_SECONDS)
                    self._schema_initialised = True
                    return
                except RuntimeError as exc:
                    if not _is_schema_visibility_error(exc):
                        raise
                    last_error = exc
                    time.sleep(_SCHEMA_VISIBILITY_DELAY_SECONDS)

            if last_error is not None:
                raise last_error
            raise RuntimeError("Nebula schema visibility never converged for space %s" % self._space)

    async def ensure_schema(self) -> None:
        await asyncio.to_thread(self._ensure_schema_sync)

    # -- read helpers (sync-blocking, called via to_thread) -----------

    def _read_entity_lineage(
        self, entity_name: str
    ) -> tuple[str, list[LineageMember], list[DescriptionPart], str | None] | None:
        """Return ``(type, source_lineage, description_parts, compacted_description)``
        for the entity, or ``None`` if the vertex doesn't exist yet.

        Wave 7 W7-1: ``compacted_description`` is ``None`` if not yet
        computed (NULL column) or the empty string was never written.
        """
        vid = _entity_vid(entity_name)
        stmt = (
            f'FETCH PROP ON `{_ENTITY_TAG}` "{_escape_str(vid)}" '
            f"YIELD `{_ENTITY_TAG}`.entity_type AS entity_type, "
            f"`{_ENTITY_TAG}`.source_lineage_json AS sl, "
            f"`{_ENTITY_TAG}`.description_parts_json AS dp, "
            f"`{_ENTITY_TAG}`.compacted_description AS cd"
        )
        result = self._execute_with_schema_retry(self._space, stmt)
        if result.row_size() == 0:
            return None
        row = result.row_values(0)
        type_value = row[0].as_string() if row[0].is_string() else ""
        sl_raw = row[1].as_string() if row[1].is_string() else ""
        dp_raw = row[2].as_string() if row[2].is_string() else ""
        compacted = row[3].as_string() if row[3].is_string() else None
        return type_value, _members_from_json(sl_raw), _parts_from_json(dp_raw), compacted

    def _read_relation_lineage(
        self, source: str, target: str, type: str
    ) -> tuple[list[LineageMember], list[DescriptionPart], str | None] | None:
        vid = _relation_vid(source, target, type)
        stmt = (
            f'FETCH PROP ON `{_RELATION_TAG}` "{_escape_str(vid)}" '
            f"YIELD `{_RELATION_TAG}`.evidence_lineage_json AS el, "
            f"`{_RELATION_TAG}`.description_parts_json AS dp, "
            f"`{_RELATION_TAG}`.compacted_description AS cd"
        )
        result = self._execute_with_schema_retry(self._space, stmt)
        if result.row_size() == 0:
            return None
        row = result.row_values(0)
        el_raw = row[0].as_string() if row[0].is_string() else ""
        dp_raw = row[1].as_string() if row[1].is_string() else ""
        compacted = row[2].as_string() if row[2].is_string() else None
        return _members_from_json(el_raw), _parts_from_json(dp_raw), compacted

    def _list_all_entity_vids(self) -> list[str]:
        """Return all VIDs tagged with ``lineage_entity`` in the
        bound space. Used by the doc-id scan in
        ``find_entity_ids_with_lineage`` — Nebula has no JSON
        containment index, so we enumerate then filter in Python.
        """
        # ``LOOKUP ON tag YIELD id(vertex)`` returns the VID of every
        # vertex that carries the tag — independent of the property
        # values. The ``WHERE`` clause filtering by JSON content
        # would need a column index Nebula can't build for a string,
        # so we paginate-via-LIMIT-only to keep the query simple.
        stmt = f"LOOKUP ON `{_ENTITY_TAG}` YIELD id(vertex) AS vid | LIMIT 100000"
        try:
            result = self._execute_with_schema_retry(self._space, stmt)
        except RuntimeError as exc:
            # Empty space / no vertices yet → some Nebula versions
            # raise instead of returning 0 rows. Treat as empty.
            if "not exist" in str(exc).lower() or "no vertex" in str(exc).lower():
                return []
            raise
        out = []
        for i in range(result.row_size()):
            row = result.row_values(i)
            if row and row[0].is_string():
                out.append(row[0].as_string())
        return out

    def _list_all_relation_vids(self) -> list[str]:
        stmt = f"LOOKUP ON `{_RELATION_TAG}` YIELD id(vertex) AS vid | LIMIT 100000"
        try:
            result = self._execute_with_schema_retry(self._space, stmt)
        except RuntimeError as exc:
            if "not exist" in str(exc).lower() or "no vertex" in str(exc).lower():
                return []
            raise
        out = []
        for i in range(result.row_size()):
            row = result.row_values(i)
            if row and row[0].is_string():
                out.append(row[0].as_string())
        return out

    def _lookup_relation_vids_by_endpoint(self, *, property_name: str, names: set[str]) -> list[str]:
        if property_name not in {"source", "target"}:
            raise ValueError(f"unsupported relation endpoint property {property_name!r}")
        if not names:
            return []
        names_literal = ", ".join(f'"{_escape_str(name)}"' for name in sorted(names))
        stmt = (
            f"LOOKUP ON `{_RELATION_TAG}` "
            f"WHERE `{_RELATION_TAG}`.`{property_name}` IN [{names_literal}] "
            f"YIELD id(vertex) AS vid | LIMIT 100000"
        )
        try:
            result = self._execute_with_schema_retry(self._space, stmt)
        except RuntimeError:
            logger.exception(
                "Nebula indexed relation lookup failed; falling back to relation VID scan property=%s space=%s",
                property_name,
                self._space,
            )
            return self._list_all_relation_vids()
        out = []
        for i in range(result.row_size()):
            row = result.row_values(i)
            if row and row[0].is_string():
                out.append(row[0].as_string())
        return out

    def _lookup_entity_vids_by_type(self, *, entity_type: str) -> list[str]:
        stmt = (
            f"LOOKUP ON `{_ENTITY_TAG}` "
            f'WHERE `{_ENTITY_TAG}`.`entity_type` == "{_escape_str(entity_type)}" '
            f"YIELD id(vertex) AS vid | LIMIT 100000"
        )
        try:
            result = self._execute_with_schema_retry(self._space, stmt)
        except RuntimeError:
            logger.exception(
                "Nebula indexed entity type lookup failed; falling back to entity VID scan space=%s",
                self._space,
            )
            return self._list_all_entity_vids()
        out = []
        for i in range(result.row_size()):
            row = result.row_values(i)
            if row and row[0].is_string():
                out.append(row[0].as_string())
        return out

    # -- write helpers ------------------------------------------------

    def _write_entity_vertex(
        self,
        *,
        name: str,
        type_value: str,
        source_lineage: list[LineageMember],
        description_parts: list[DescriptionPart],
        compacted_description: str | None,
    ) -> None:
        vid = _entity_vid(name)
        compacted_literal = "NULL" if compacted_description is None else f'"{_escape_str(compacted_description)}"'
        stmt = (
            f"INSERT VERTEX `{_ENTITY_TAG}`"
            f"(name, entity_type, source_lineage_json, description_parts_json, "
            f"compacted_description, gmt_created, gmt_updated) "
            f'VALUES "{_escape_str(vid)}":('
            f'"{_escape_str(name)}", "{_escape_str(type_value)}", '
            f'"{_escape_str(_members_to_json(source_lineage))}", '
            f'"{_escape_str(_parts_to_json(description_parts))}", '
            f"{compacted_literal}, "
            f"datetime(), datetime())"
        )
        self._execute_with_schema_retry(self._space, stmt)

    def _write_relation_vertex(
        self,
        *,
        source: str,
        target: str,
        type_value: str,
        evidence_lineage: list[LineageMember],
        description_parts: list[DescriptionPart],
        compacted_description: str | None,
    ) -> None:
        vid = _relation_vid(source, target, type_value)
        compacted_literal = "NULL" if compacted_description is None else f'"{_escape_str(compacted_description)}"'
        stmt = (
            f"INSERT VERTEX `{_RELATION_TAG}`"
            f"(source, target, relation_type, "
            f"evidence_lineage_json, description_parts_json, "
            f"compacted_description, gmt_created, gmt_updated) "
            f'VALUES "{_escape_str(vid)}":('
            f'"{_escape_str(source)}", "{_escape_str(target)}", "{_escape_str(type_value)}", '
            f'"{_escape_str(_members_to_json(evidence_lineage))}", '
            f'"{_escape_str(_parts_to_json(description_parts))}", '
            f"{compacted_literal}, "
            f"datetime(), datetime())"
        )
        self._execute_with_schema_retry(self._space, stmt)

    def _delete_entity_vertex(self, name: str) -> None:
        vid = _entity_vid(name)
        stmt = f'DELETE VERTEX "{_escape_str(vid)}" WITH EDGE'
        self._execute_with_schema_retry(self._space, stmt)

    def _delete_relation_vertex(self, source: str, target: str, type: str) -> None:
        vid = _relation_vid(source, target, type)
        stmt = f'DELETE VERTEX "{_escape_str(vid)}" WITH EDGE'
        self._execute_with_schema_retry(self._space, stmt)

    # -- find-by-document scans (pre-rebuild phase) -------------------

    async def find_entity_ids_with_lineage(self, *, document_id: str) -> list[str]:
        await self.ensure_schema()

        def _scan() -> list[str]:
            names: list[str] = []
            for vid in self._list_all_entity_vids():
                row = self._read_entity_lineage_by_vid(vid)
                if row is None:
                    continue
                name, _type, members, _parts, _compacted = row
                if any(m.document_id == document_id for m in members):
                    names.append(name)
            return names

        return await asyncio.to_thread(_scan)

    async def find_relation_keys_with_lineage(self, *, document_id: str) -> list[tuple[str, str, str]]:
        await self.ensure_schema()

        def _scan() -> list[tuple[str, str, str]]:
            keys: list[tuple[str, str, str]] = []
            for vid in self._list_all_relation_vids():
                row = self._read_relation_lineage_by_vid(vid)
                if row is None:
                    continue
                source, target, type_value, members, _parts, _compacted = row
                if any(m.document_id == document_id for m in members):
                    keys.append((source, target, type_value))
            return keys

        return await asyncio.to_thread(_scan)

    # Variants that return the natural-key columns alongside the
    # lineage so the doc-scan helpers don't have to re-parse the VID.
    def _read_entity_lineage_by_vid(
        self, vid: str
    ) -> tuple[str, str, list[LineageMember], list[DescriptionPart], str | None] | None:
        stmt = (
            f'FETCH PROP ON `{_ENTITY_TAG}` "{_escape_str(vid)}" '
            f"YIELD `{_ENTITY_TAG}`.name AS name, "
            f"`{_ENTITY_TAG}`.entity_type AS entity_type, "
            f"`{_ENTITY_TAG}`.source_lineage_json AS sl, "
            f"`{_ENTITY_TAG}`.description_parts_json AS dp, "
            f"`{_ENTITY_TAG}`.compacted_description AS cd"
        )
        result = self._execute_with_schema_retry(self._space, stmt)
        if result.row_size() == 0:
            return None
        row = result.row_values(0)
        name = row[0].as_string() if row[0].is_string() else ""
        type_value = row[1].as_string() if row[1].is_string() else ""
        sl_raw = row[2].as_string() if row[2].is_string() else ""
        dp_raw = row[3].as_string() if row[3].is_string() else ""
        compacted = row[4].as_string() if row[4].is_string() else None
        return name, type_value, _members_from_json(sl_raw), _parts_from_json(dp_raw), compacted

    def _read_relation_lineage_by_vid(
        self, vid: str
    ) -> tuple[str, str, str, list[LineageMember], list[DescriptionPart], str | None] | None:
        stmt = (
            f'FETCH PROP ON `{_RELATION_TAG}` "{_escape_str(vid)}" '
            f"YIELD `{_RELATION_TAG}`.source AS source, "
            f"`{_RELATION_TAG}`.target AS target, "
            f"`{_RELATION_TAG}`.relation_type AS relation_type, "
            f"`{_RELATION_TAG}`.evidence_lineage_json AS el, "
            f"`{_RELATION_TAG}`.description_parts_json AS dp, "
            f"`{_RELATION_TAG}`.compacted_description AS cd"
        )
        result = self._execute_with_schema_retry(self._space, stmt)
        if result.row_size() == 0:
            return None
        row = result.row_values(0)
        source = row[0].as_string() if row[0].is_string() else ""
        target = row[1].as_string() if row[1].is_string() else ""
        type_value = row[2].as_string() if row[2].is_string() else ""
        el_raw = row[3].as_string() if row[3].is_string() else ""
        dp_raw = row[4].as_string() if row[4].is_string() else ""
        compacted = row[5].as_string() if row[5].is_string() else None
        return (
            source,
            target,
            type_value,
            _members_from_json(el_raw),
            _parts_from_json(dp_raw),
            compacted,
        )

    # -- strip-by-document (pre-rebuild phase) ------------------------

    async def remove_entity_lineage_member(self, *, entity_name: str, document_id: str) -> None:
        await self.ensure_schema()
        async with self._entity_lock.acquire(entity_name):

            def _strip() -> None:
                row = self._read_entity_lineage(entity_name)
                if row is None:
                    return
                type_value, members, parts, compacted = row
                new_members = [m for m in members if m.document_id != document_id]
                new_parts = [p for p in parts if p.document_id != document_id]
                if len(new_members) == len(members) and len(new_parts) == len(parts):
                    return
                self._write_entity_vertex(
                    name=entity_name,
                    type_value=type_value,
                    source_lineage=new_members,
                    description_parts=new_parts,
                    # W7-1: strip preserves compacted_description so a
                    # subsequent re-sync of the SAME doc keeps the
                    # cache; compactor decides when to recompute.
                    compacted_description=compacted,
                )

            await asyncio.to_thread(_strip)

    async def remove_relation_lineage_member(self, *, source: str, target: str, type: str, document_id: str) -> None:
        await self.ensure_schema()
        async with self._entity_lock.acquire(_relation_vid(source, target, type)):

            def _strip() -> None:
                row = self._read_relation_lineage(source, target, type)
                if row is None:
                    return
                members, parts, compacted = row
                new_members = [m for m in members if m.document_id != document_id]
                new_parts = [p for p in parts if p.document_id != document_id]
                if len(new_members) == len(members) and len(new_parts) == len(parts):
                    return
                self._write_relation_vertex(
                    source=source,
                    target=target,
                    type_value=type,
                    evidence_lineage=new_members,
                    description_parts=new_parts,
                    compacted_description=compacted,
                )

            await asyncio.to_thread(_strip)

    # -- GC (post-rebuild) --------------------------------------------

    async def gc_entity_if_orphan(self, entity_name: str) -> bool:
        await self.ensure_schema()
        async with self._entity_lock.acquire(entity_name):

            def _gc() -> bool:
                row = self._read_entity_lineage(entity_name)
                if row is None:
                    return False
                _type_value, members, _parts, _compacted = row
                if members:
                    return False
                self._delete_entity_vertex(entity_name)
                return True

            return await asyncio.to_thread(_gc)

    async def gc_relation_if_orphan(self, source: str, target: str, type: str) -> bool:
        await self.ensure_schema()
        async with self._entity_lock.acquire(_relation_vid(source, target, type)):

            def _gc() -> bool:
                row = self._read_relation_lineage(source, target, type)
                if row is None:
                    return False
                members, _parts, _compacted = row
                if members:
                    return False
                self._delete_relation_vertex(source, target, type)
                return True

            return await asyncio.to_thread(_gc)

    # -- unconditional delete (Wave 7 W7-1, used by curation merge) ---

    async def delete_entity(self, entity_name: str) -> bool:
        await self.ensure_schema()
        async with self._entity_lock.acquire(entity_name):

            def _delete() -> bool:
                row = self._read_entity_lineage(entity_name)
                if row is None:
                    return False
                self._delete_entity_vertex(entity_name)
                return True

            return await asyncio.to_thread(_delete)

    async def delete_relation(self, source: str, target: str, type: str) -> bool:
        await self.ensure_schema()
        async with self._entity_lock.acquire(_relation_vid(source, target, type)):

            def _delete() -> bool:
                row = self._read_relation_lineage(source, target, type)
                if row is None:
                    return False
                self._delete_relation_vertex(source, target, type)
                return True

            return await asyncio.to_thread(_delete)

    # -- upserts (rebuild phase) --------------------------------------

    async def upsert_entity_with_lineage(
        self,
        *,
        record: EntityRecord,
        lineage: LineageMember,
        compacted_description: str | None = None,
    ) -> None:
        """Add (or replace by ``(document_id, parse_version)`` key) the
        lineage member + corresponding description part.

        The whole read-modify-write is wrapped in
        ``EntityLock.acquire(name)`` so two concurrent rebuilds for the
        same entity serialise. Without this serialisation, Nebula's
        property-update semantics (no native list ops) would race —
        per architect msg=f2921ae0.

        Wave 7 W7-1: ``compacted_description=None`` (default) preserves
        the existing column; non-None overwrites. The Postgres COALESCE
        equivalent is implemented in Python here because Nebula has no
        native COALESCE on INSERT VERTEX (full-row overwrite semantics).
        """
        await self.ensure_schema()
        new_part = DescriptionPart(
            document_id=lineage.document_id,
            parse_version=lineage.parse_version,
            text=record.description,
        )
        async with self._entity_lock.acquire(record.name):

            def _upsert() -> None:
                row = self._read_entity_lineage(record.name)
                if row is None:
                    new_members = [lineage]
                    new_parts = [new_part]
                    existing_compacted: str | None = None
                else:
                    _existing_type, members, parts, existing_compacted = row
                    new_members = [m for m in members if m.key() != lineage.key()] + [lineage]
                    new_parts = [p for p in parts if p.key() != new_part.key()] + [new_part]
                # W7-1: preserve existing if param is None.
                final_compacted = compacted_description if compacted_description is not None else existing_compacted
                self._write_entity_vertex(
                    name=record.name,
                    type_value=record.entity_type,
                    source_lineage=new_members,
                    description_parts=new_parts,
                    compacted_description=final_compacted,
                )

            await asyncio.to_thread(_upsert)

    async def bulk_upsert_entity_with_lineage_parts(
        self,
        *,
        parts,
    ) -> None:
        """Wave 8 W8-2: Nebula bulk variant — single ``EntityLock``
        acquire + single read-modify-write applies the whole ``parts``
        list. Reuses the read/Python-merge/write pattern of
        :meth:`upsert_entity_with_lineage` but folds the strip-then-
        append over the **set** of incoming keys, so N×M parts collapse
        to one write.
        """
        if not parts:
            return
        target_name = parts[0][0].name
        if any(record.name != target_name for record, _ in parts):
            raise ValueError("bulk_upsert_entity_with_lineage_parts: all records must share the same name")

        # Dedup last-wins by (document_id, parse_version).
        deduped: dict[tuple[str, str], tuple[EntityRecord, LineageMember]] = {}
        for record, lineage in parts:
            deduped[(lineage.document_id, lineage.parse_version)] = (record, lineage)

        new_members_in: list[LineageMember] = []
        new_parts_in: list[DescriptionPart] = []
        last_entity_type: str = parts[0][0].entity_type
        for record, lineage in deduped.values():
            new_members_in.append(lineage)
            new_parts_in.append(
                DescriptionPart(
                    document_id=lineage.document_id,
                    parse_version=lineage.parse_version,
                    text=record.description,
                )
            )
            last_entity_type = record.entity_type

        keys_to_strip = set(deduped.keys())

        await self.ensure_schema()
        async with self._entity_lock.acquire(target_name):

            def _upsert() -> None:
                row = self._read_entity_lineage(target_name)
                if row is None:
                    merged_members = list(new_members_in)
                    merged_parts = list(new_parts_in)
                    existing_compacted: str | None = None
                else:
                    _existing_type, members, parts_existing, existing_compacted = row
                    kept_members = [m for m in members if m.key() not in keys_to_strip]
                    kept_parts = [p for p in parts_existing if p.key() not in keys_to_strip]
                    merged_members = kept_members + new_members_in
                    merged_parts = kept_parts + new_parts_in
                self._write_entity_vertex(
                    name=target_name,
                    type_value=last_entity_type,
                    source_lineage=merged_members,
                    description_parts=merged_parts,
                    # Bulk path never touches compacted_description
                    # (preserves existing, mirror Postgres / Neo4j).
                    compacted_description=existing_compacted,
                )

            await asyncio.to_thread(_upsert)

    async def upsert_relation_with_lineage(
        self,
        *,
        record: RelationRecord,
        lineage: LineageMember,
        compacted_description: str | None = None,
    ) -> None:
        await self.ensure_schema()
        new_part = DescriptionPart(
            document_id=lineage.document_id,
            parse_version=lineage.parse_version,
            text=record.description,
        )
        async with self._entity_lock.acquire(_relation_vid(record.source, record.target, record.relation_type)):

            def _upsert() -> None:
                row = self._read_relation_lineage(record.source, record.target, record.relation_type)
                if row is None:
                    new_members = [lineage]
                    new_parts = [new_part]
                    existing_compacted: str | None = None
                else:
                    members, parts, existing_compacted = row
                    new_members = [m for m in members if m.key() != lineage.key()] + [lineage]
                    new_parts = [p for p in parts if p.key() != new_part.key()] + [new_part]
                final_compacted = compacted_description if compacted_description is not None else existing_compacted
                self._write_relation_vertex(
                    source=record.source,
                    target=record.target,
                    type_value=record.relation_type,
                    evidence_lineage=new_members,
                    description_parts=new_parts,
                    compacted_description=final_compacted,
                )

            await asyncio.to_thread(_upsert)

    # -- read-path ----------------------------------------------------

    async def get_entity(self, entity_name: str) -> EntityWithLineage | None:
        await self.ensure_schema()

        def _read() -> EntityWithLineage | None:
            row = self._read_entity_lineage(entity_name)
            if row is None:
                return None
            type_value, members, parts, compacted = row
            return EntityWithLineage(
                name=entity_name,
                entity_type=type_value,
                source_lineage=tuple(members),
                description_parts=tuple(parts),
                compacted_description=compacted,
            )

        return await asyncio.to_thread(_read)

    async def get_relation(self, source: str, target: str, type: str) -> RelationWithLineage | None:
        await self.ensure_schema()

        def _read() -> RelationWithLineage | None:
            row = self._read_relation_lineage(source, target, type)
            if row is None:
                return None
            members, parts, compacted = row
            return RelationWithLineage(
                source=source,
                target=target,
                relation_type=type,
                evidence_lineage=tuple(members),
                description_parts=tuple(parts),
                compacted_description=compacted,
            )

        return await asyncio.to_thread(_read)

    # -- Graph-RAG query layer (Wave 6 #33 chunk 2) -------------------

    async def query_entities_by_keyword(
        self,
        *,
        query: str,
        top_k: int,
    ) -> list[EntityWithLineage]:
        if not query or not query.strip() or top_k <= 0:
            return []
        await self.ensure_schema()

        needle = query.strip().lower()

        def _scan() -> list[EntityWithLineage]:
            # Nebula has no JSON / text-search index on tag string
            # properties; we enumerate all entity VIDs via LOOKUP and
            # filter substring-match in Python. For collections with
            # very large entity counts this is slower than the
            # SQL/Cypher equivalents — acceptable per simple-stable
            # directive (no per-collection text-index infra to manage).
            matches: list[EntityWithLineage] = []
            for vid in self._list_all_entity_vids():
                row = self._read_entity_lineage_by_vid(vid)
                if row is None:
                    continue
                name, type_value, members, parts, compacted = row
                if needle not in name.lower():
                    continue
                matches.append(
                    EntityWithLineage(
                        name=name,
                        entity_type=type_value,
                        source_lineage=tuple(members),
                        description_parts=tuple(parts),
                        compacted_description=compacted,
                    )
                )
            matches.sort(key=lambda e: e.name)
            return matches[:top_k]

        return await asyncio.to_thread(_scan)

    async def expand_neighbors_n_hops(
        self,
        *,
        entity_names: list[str],
        hops: int = 1,
    ) -> tuple[list[EntityWithLineage], list[RelationWithLineage]]:
        if not entity_names:
            return ([], [])
        await self.ensure_schema()

        def _walk() -> tuple[list[EntityWithLineage], list[RelationWithLineage]]:
            seen_entities: dict[str, EntityWithLineage] = {}
            seen_relations: dict[tuple[str, str, str], RelationWithLineage] = {}

            def _add_entity(name: str) -> None:
                if name in seen_entities:
                    return
                row = self._read_entity_lineage(name)
                if row is None:
                    return
                type_value, members, parts, compacted = row
                seen_entities[name] = EntityWithLineage(
                    name=name,
                    entity_type=type_value,
                    source_lineage=tuple(members),
                    description_parts=tuple(parts),
                    compacted_description=compacted,
                )

            current = {n for n in entity_names if n}
            for name in current:
                _add_entity(name)

            # Nebula has no edge type for relations (relations are
            # tag-vertices). Walk by enumerating all relation VIDs and
            # filtering by source/target ∈ current — same approach as
            # ``find_relation_keys_with_lineage``.
            for _ in range(max(hops, 0)):
                next_frontier: set[str] = set()
                if not current:
                    break
                candidate_vids = set(self._lookup_relation_vids_by_endpoint(property_name="source", names=current))
                candidate_vids.update(self._lookup_relation_vids_by_endpoint(property_name="target", names=current))
                for vid in candidate_vids:
                    row = self._read_relation_lineage_by_vid(vid)
                    if row is None:
                        continue
                    src, tgt, rtype, members, parts, compacted = row
                    if src not in current and tgt not in current:
                        continue
                    key = (src, tgt, rtype)
                    if key not in seen_relations:
                        seen_relations[key] = RelationWithLineage(
                            source=src,
                            target=tgt,
                            relation_type=rtype,
                            evidence_lineage=tuple(members),
                            description_parts=tuple(parts),
                            compacted_description=compacted,
                        )
                    for endpoint in (src, tgt):
                        if endpoint not in seen_entities and endpoint not in next_frontier:
                            next_frontier.add(endpoint)
                if not next_frontier:
                    break
                for name in next_frontier:
                    _add_entity(name)
                current = next_frontier

            entities = sorted(seen_entities.values(), key=lambda e: e.name)
            relations = sorted(seen_relations.values(), key=lambda r: (r.source, r.target, r.relation_type))
            return (entities, relations)

        return await asyncio.to_thread(_walk)

    # -- UI label list (Wave 6 #40 narrow replacement) -----------------

    async def list_entity_labels(self) -> list[str]:
        await self.ensure_schema()

        def _scan() -> list[str]:
            # Same scan-and-collect approach as ``query_entities_by_keyword``:
            # Nebula has no native distinct-by-property aggregation that
            # we can rely on across versions, and the entity row count
            # for a single collection is bounded by application use.
            labels: set[str] = set()
            for vid in self._list_all_entity_vids():
                row = self._read_entity_lineage_by_vid(vid)
                if row is None:
                    continue
                _, type_value, _, _, _ = row
                if type_value:
                    labels.add(type_value)
            return sorted(labels)

        return await asyncio.to_thread(_scan)

    # -- Wave 7 W7-10: paginated entity list ---------------------------

    async def list_entities(
        self,
        *,
        label: str | None = None,
        limit: int = 1000,
        offset: int = 0,
    ) -> list[EntityWithLineage]:
        if limit <= 0:
            return []
        offset = max(0, offset)
        await self.ensure_schema()

        def _scan() -> list[EntityWithLineage]:
            # If ``label`` is provided, try the entity_type tag index
            # first; otherwise enumerate all entity VIDs. Fetch + sort
            # in Python to preserve the cross-backend
            # ``ORDER BY name SKIP offset LIMIT limit`` contract.
            rows: list[EntityWithLineage] = []
            candidate_vids = (
                self._lookup_entity_vids_by_type(entity_type=label)
                if label is not None
                else self._list_all_entity_vids()
            )
            for vid in candidate_vids:
                row = self._read_entity_lineage_by_vid(vid)
                if row is None:
                    continue
                name, type_value, members, parts, compacted = row
                if label is not None and type_value != label:
                    continue
                rows.append(
                    EntityWithLineage(
                        name=name,
                        entity_type=type_value,
                        source_lineage=tuple(members),
                        description_parts=tuple(parts),
                        compacted_description=compacted,
                    )
                )
            rows.sort(key=lambda e: e.name)
            return rows[offset : offset + limit]

        return await asyncio.to_thread(_scan)


__all__ = [
    "NebulaLineageGraphStore",
]
