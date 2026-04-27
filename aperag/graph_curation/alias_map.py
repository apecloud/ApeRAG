# Copyright 2026 ApeCloud, Inc.
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

"""Alias-map repository — Wave 7 §K.12.7 + §K.12.10b task #6.

Persists user-driven entity merge intent. The :class:`AliasMapRepository`
is the canonical write/read surface; downstream consumers use the two
methods :meth:`AliasMapRepository.resolve_canonical` (read) and
:meth:`AliasMapRepository.upsert_alias` (write, with cycle reject and
transitive flatten).

Design notes
------------

* Per-collection scoping is a row-level concern (column
  ``collection_id``). The repository takes ``collection_id`` on each
  call rather than at construction to keep the repo a stateless
  singleton (mirrors :class:`AsyncBaseRepository` convention).
* Cycle handling is the service-layer invariant from §K.12.10b:
  ``upsert_alias`` ALWAYS resolves the requested ``target`` through
  the existing chain first. If the resolved canonical equals the
  ``alias_name`` itself, we raise :class:`AliasCycleError` instead of
  writing a self-loop.
* Transitive flatten: when ``B → C`` is recorded, any prior
  ``A → B`` row is rewritten to ``A → C`` in the same transaction.
  Readers therefore always see at most one indirection (no chain
  walks at read time, even after multi-step merges).
"""

from __future__ import annotations

import logging

from sqlalchemy import delete, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from aperag.db.repositories.base import AsyncBaseRepository
from aperag.domains.knowledge_graph.db.models import LineageEntityAlias
from aperag.utils.utils import utc_now

logger = logging.getLogger(__name__)


class AliasCycleError(ValueError):
    """Raised when an alias upsert would create a self-loop —
    ``alias_name == resolve_canonical(target)``. The service layer
    aborts the merge and surfaces this to the caller so the user can
    pick a different target.
    """


class AliasMapRepository(AsyncBaseRepository):
    """Read/write surface for ``aperag_lineage_entity_alias``.

    Two write methods (:meth:`upsert_alias`, :meth:`purge_collection`)
    and two read methods (:meth:`resolve_canonical`,
    :meth:`list_aliases_pointing_at`). Everything else can be expressed
    as composition.
    """

    # ------------------------------------------------------------------
    # read path
    # ------------------------------------------------------------------

    async def resolve_canonical(self, *, collection_id: str, name: str) -> str:
        """Return the canonical name for ``name`` in ``collection_id``.

        ``name`` itself is returned when no alias row points at it
        (i.e. ``name`` is already canonical or has never been merged).
        Returns at most one indirection because :meth:`upsert_alias`
        flattens transitively at write time.
        """
        if not name:
            return name

        async def _op(session: AsyncSession) -> str:
            row = await session.get(LineageEntityAlias, (collection_id, name))
            if row is None:
                return name
            return str(row.canonical_name)

        return await self._execute_query(_op)

    async def list_aliases_pointing_at(self, *, collection_id: str, canonical_name: str) -> list[str]:
        """Return every ``alias_name`` whose row points at
        ``canonical_name`` in ``collection_id``.

        Used by :meth:`upsert_alias` to perform transitive flatten and
        by tests / admin tooling. Order is alphabetical for
        determinism."""

        async def _op(session: AsyncSession) -> list[str]:
            stmt = (
                select(LineageEntityAlias.alias_name)
                .where(
                    LineageEntityAlias.collection_id == collection_id,
                    LineageEntityAlias.canonical_name == canonical_name,
                )
                .order_by(LineageEntityAlias.alias_name)
            )
            result = await session.execute(stmt)
            return [r[0] for r in result.all()]

        return await self._execute_query(_op)

    # ------------------------------------------------------------------
    # write path
    # ------------------------------------------------------------------

    async def upsert_alias(
        self,
        *,
        collection_id: str,
        alias_name: str,
        target: str,
        merged_by: str | None = None,
    ) -> str:
        """Record that ``alias_name`` should resolve to ``target`` in
        ``collection_id``.

        Returns the resolved canonical name actually written (which may
        differ from ``target`` if ``target`` itself was already an
        alias — we flatten through to the terminal canonical so readers
        never traverse a chain).

        Cycle reject: if the resolved canonical equals ``alias_name``,
        raise :class:`AliasCycleError` instead of writing a self-loop.

        Transitive flatten: any prior alias row whose ``canonical_name``
        equals the *old* row at ``(collection_id, alias_name)`` gets
        rewritten to point at the new canonical. The flatten + the
        upsert run inside one transaction so a partial flatten can
        never be observed.
        """

        async def _op(session: AsyncSession) -> str:
            # Resolve target through any existing alias row (single
            # indirection, since flatten keeps the table 1-deep).
            target_row = await session.get(LineageEntityAlias, (collection_id, target))
            canonical = str(target_row.canonical_name) if target_row is not None else target

            if canonical == alias_name:
                raise AliasCycleError(
                    f"alias upsert would create a cycle: "
                    f"alias={alias_name!r} → target={target!r} resolves to {canonical!r}"
                )

            now = utc_now()
            existing = await session.get(LineageEntityAlias, (collection_id, alias_name))
            if existing is not None:
                existing.canonical_name = canonical
                existing.merged_by = merged_by
                existing.gmt_updated = now
            else:
                session.add(
                    LineageEntityAlias(
                        collection_id=collection_id,
                        alias_name=alias_name,
                        canonical_name=canonical,
                        merged_by=merged_by,
                        gmt_created=now,
                        gmt_updated=now,
                    )
                )

            # Transitive flatten: any row pointing at ``alias_name`` now
            # needs to point at the new canonical instead. (The alias
            # name was canonical from those rows' perspective; now it is
            # itself an alias of ``canonical``.)
            await session.execute(
                update(LineageEntityAlias)
                .where(
                    LineageEntityAlias.collection_id == collection_id,
                    LineageEntityAlias.canonical_name == alias_name,
                )
                .values(
                    canonical_name=canonical,
                    gmt_updated=now,
                )
            )
            return canonical

        return await self.execute_with_transaction(_op)

    async def purge_collection(self, collection_id: str) -> int:
        """Delete all alias rows for ``collection_id``. Returns the
        number of rows deleted. Used by collection-purge / test
        teardown."""

        async def _op(session: AsyncSession) -> int:
            result = await session.execute(
                delete(LineageEntityAlias).where(LineageEntityAlias.collection_id == collection_id)
            )
            return int(result.rowcount or 0)

        return await self.execute_with_transaction(_op)


__all__ = [
    "AliasCycleError",
    "AliasMapRepository",
]
