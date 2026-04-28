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

"""Lineage graph store decorator that applies user-driven alias redirect.

Wave 7 §K.12.4 invariant #3 (write-side) + Wave 8 W8-3 (read-side):
when the user has merged ``A → C``, every reference to ``A`` should
silently resolve to ``C`` — both writes (indexer hot path) and reads
(MCP / REST / curation surfaces).

Implementation strategy (architect ratify Wave 7 msg=cf860ae4 +
huangheng endorse msg=22816e0d / msg=93d9add1, **Option (b)**): a
thin decorator class wraps any concrete
:class:`aperag.indexing.graph.LineageGraphStore` plus an
:class:`aperag.graph_curation.alias_map.AliasMapRepository`,
intercepts the methods that take entity names as inputs (writes +
direct reads + traversals seeded by entity names) and rewrites them
through the alias map. Methods whose inputs aren't entity names
(``query_entities_by_keyword``, ``list_entity_labels``,
``find_*_with_lineage``) forward unchanged — they either operate on
opaque tokens or already return canonical names.

Read-side methods that DO redirect (Wave 8 task #14):

* ``get_entity(name)`` — direct lookup by name.
* ``get_relation(source, target, type)`` — both endpoints redirected
  symmetrically (mirror :meth:`upsert_relation_with_lineage` write
  redirect).
* ``expand_neighbors_n_hops(entity_names, hops)`` — anchor names
  redirected before traversal so an alias seed walks the canonical
  neighbourhood.

Read-side methods that do NOT redirect:

* ``query_entities_by_keyword`` — keyword is not an entity name; the
  matched rows are themselves canonical (alias rows live in
  ``aperag_lineage_entity_alias``, not in the entity table).
* ``list_entity_labels`` — returns a label set, not entity names.
* ``list_entities`` (W7-10) — filter is by ``label`` + pagination,
  not by name; rows are already canonical for the same reason as
  ``query_entities_by_keyword``.
* ``find_entity_ids_with_lineage`` / ``find_relation_keys_with_lineage`` —
  filter by ``document_id`` (lineage source), not by name.
* ``gc_entity_if_orphan`` / ``gc_relation_if_orphan`` — gc operates on
  the canonical row already; an alias name as input would no-op
  silently (canonical row doesn't carry the alias as its name).
* ``delete_entity`` / ``delete_relation`` — explicit user/admin
  intent; redirecting would silently drop the canonical when the
  caller asked for the alias. We surface ``False`` (no row deleted)
  by passing through unchanged.

Decorator passthrough invariant (huangheng CR lock,
``test_decorator_passthrough_for_non_upsert_methods``): every method
declared on :class:`LineageGraphStore` that does NOT redirect must
forward to ``_inner`` byte-for-byte — no silent behaviour change.
Tests pin this so a future Protocol method addition can't slip past
without an explicit decorator update.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import replace
from typing import TYPE_CHECKING

from aperag.indexing.graph import (
    EntityRecord,
    EntityWithLineage,
    LineageMember,
    RelationRecord,
    RelationWithLineage,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from aperag.graph_curation.alias_map import AliasMapRepository
    from aperag.indexing.graph import LineageGraphStore

logger = logging.getLogger(__name__)


class LineageGraphStoreWithAliasRedirect:
    """Wrap a :class:`LineageGraphStore` so writes go through the alias
    map.

    Constructor takes the inner store, the alias-map repository, and
    the ``collection_id`` the inner store is bound to. The decorator is
    per-collection just like the inner store — both share the same
    binding.
    """

    def __init__(
        self,
        *,
        inner: "LineageGraphStore",
        alias_repo: "AliasMapRepository",
        collection_id: str,
    ) -> None:
        self._inner = inner
        self._alias_repo = alias_repo
        self._collection_id = collection_id

    # ------------------------------------------------------------------
    # Intercepted write paths — apply alias redirect
    # ------------------------------------------------------------------

    async def upsert_entity_with_lineage(
        self,
        *,
        record: EntityRecord,
        lineage: LineageMember,
        compacted_description: str | None = None,
    ) -> None:
        canonical = await self._alias_repo.resolve_canonical(collection_id=self._collection_id, name=record.name)
        redirected = record if canonical == record.name else replace(record, name=canonical)
        if canonical != record.name:
            logger.debug(
                "alias_redirect: entity write %r → %r (collection=%s)",
                record.name,
                canonical,
                self._collection_id,
            )
        await self._inner.upsert_entity_with_lineage(
            record=redirected,
            lineage=lineage,
            compacted_description=compacted_description,
        )

    async def upsert_relation_with_lineage(
        self,
        *,
        record: RelationRecord,
        lineage: LineageMember,
        compacted_description: str | None = None,
    ) -> None:
        # Both endpoints of a relation may have been merged; resolve
        # both. ``relation_type`` is unaffected by entity merges.
        new_source = await self._alias_repo.resolve_canonical(collection_id=self._collection_id, name=record.source)
        new_target = await self._alias_repo.resolve_canonical(collection_id=self._collection_id, name=record.target)
        if new_source != record.source or new_target != record.target:
            logger.debug(
                "alias_redirect: relation write (%r→%r) → (%r→%r) (collection=%s)",
                record.source,
                record.target,
                new_source,
                new_target,
                self._collection_id,
            )
            redirected = replace(record, source=new_source, target=new_target)
        else:
            redirected = record
        await self._inner.upsert_relation_with_lineage(
            record=redirected,
            lineage=lineage,
            compacted_description=compacted_description,
        )

    async def bulk_upsert_entity_with_lineage_parts(self, *, parts) -> None:
        # Wave 8 W8-2: alias-redirect on the bulk write surface,
        # mirror of the single-upsert path. Resolve each ``record.name``
        # through the alias map; if any tuple's name redirects, the
        # whole bulk write retargets to the canonical name. The
        # ``LineageEntityMerger`` is the only known caller and always
        # passes records already pinned to the canonical
        # ``final_target`` so the redirect is a no-op in that flow —
        # but we apply it here for symmetry with the single upsert
        # contract (a future caller writing to an aliased name still
        # gets the right behaviour).
        if not parts:
            return
        redirected: list[tuple[EntityRecord, LineageMember]] = []
        for record, lineage in parts:
            canonical = await self._alias_repo.resolve_canonical(collection_id=self._collection_id, name=record.name)
            if canonical != record.name:
                logger.debug(
                    "alias_redirect: bulk entity part %r → %r (collection=%s)",
                    record.name,
                    canonical,
                    self._collection_id,
                )
                redirected.append((replace(record, name=canonical), lineage))
            else:
                redirected.append((record, lineage))
        await self._inner.bulk_upsert_entity_with_lineage_parts(parts=redirected)

    # ------------------------------------------------------------------
    # Passthrough — forward every non-redirected Protocol method
    # unchanged. Pinned by
    # ``test_decorator_passthrough_for_non_redirected_methods``.
    # ------------------------------------------------------------------

    async def find_entity_ids_with_lineage(self, *, document_id: str) -> list[str]:
        return await self._inner.find_entity_ids_with_lineage(document_id=document_id)

    async def find_relation_keys_with_lineage(self, *, document_id: str) -> list[tuple[str, str, str]]:
        return await self._inner.find_relation_keys_with_lineage(document_id=document_id)

    async def remove_entity_lineage_member(self, *, entity_name: str, document_id: str) -> None:
        await self._inner.remove_entity_lineage_member(entity_name=entity_name, document_id=document_id)

    async def remove_relation_lineage_member(self, *, source: str, target: str, type: str, document_id: str) -> None:
        await self._inner.remove_relation_lineage_member(
            source=source, target=target, type=type, document_id=document_id
        )

    async def gc_entity_if_orphan(self, entity_name: str) -> bool:
        return await self._inner.gc_entity_if_orphan(entity_name)

    async def gc_relation_if_orphan(self, source: str, target: str, type: str) -> bool:
        return await self._inner.gc_relation_if_orphan(source, target, type)

    async def delete_entity(self, entity_name: str) -> bool:
        return await self._inner.delete_entity(entity_name)

    async def delete_relation(self, source: str, target: str, type: str) -> bool:
        return await self._inner.delete_relation(source, target, type)

    # ------------------------------------------------------------------
    # Intercepted read paths — Wave 8 W8-3 task #14
    # ------------------------------------------------------------------

    async def get_entity(self, entity_name: str) -> EntityWithLineage | None:
        canonical = await self._alias_repo.resolve_canonical(collection_id=self._collection_id, name=entity_name)
        if canonical != entity_name:
            logger.debug(
                "alias_redirect: get_entity %r → %r (collection=%s)",
                entity_name,
                canonical,
                self._collection_id,
            )
        return await self._inner.get_entity(canonical)

    async def get_relation(self, source: str, target: str, type: str) -> RelationWithLineage | None:
        # Both endpoints may have been merged independently; resolve
        # symmetrically (mirror write-side redirect in
        # ``upsert_relation_with_lineage``).
        new_source = await self._alias_repo.resolve_canonical(collection_id=self._collection_id, name=source)
        new_target = await self._alias_repo.resolve_canonical(collection_id=self._collection_id, name=target)
        if new_source != source or new_target != target:
            logger.debug(
                "alias_redirect: get_relation (%r→%r) → (%r→%r) (collection=%s)",
                source,
                target,
                new_source,
                new_target,
                self._collection_id,
            )
        return await self._inner.get_relation(new_source, new_target, type)

    async def expand_neighbors_n_hops(
        self, *, entity_names: list[str], hops: int = 1
    ) -> tuple[list[EntityWithLineage], list[RelationWithLineage]]:
        if not entity_names:
            return await self._inner.expand_neighbors_n_hops(entity_names=entity_names, hops=hops)
        # Resolve every anchor through the alias map so a caller seeding
        # ``["Alicia"]`` walks the canonical ``Alice`` neighbourhood.
        # ``asyncio.gather`` keeps the per-anchor lookup cost flat —
        # N is bounded by the caller (typically small).
        resolved = await asyncio.gather(
            *(self._alias_repo.resolve_canonical(collection_id=self._collection_id, name=n) for n in entity_names),
            return_exceptions=False,
        )
        # De-dup so ``["Alicia", "Alice"]`` (alias + canonical mixed)
        # doesn't double-traverse the same anchor; preserve input order
        # for deterministic output.
        seen: set[str] = set()
        canonical_anchors: list[str] = []
        for original, canonical in zip(entity_names, resolved):
            if canonical not in seen:
                seen.add(canonical)
                canonical_anchors.append(canonical)
            if canonical != original:
                logger.debug(
                    "alias_redirect: expand anchor %r → %r (collection=%s)",
                    original,
                    canonical,
                    self._collection_id,
                )
        return await self._inner.expand_neighbors_n_hops(entity_names=canonical_anchors, hops=hops)

    # ------------------------------------------------------------------
    # Read paths that do NOT redirect — passthrough (see module
    # docstring for the rationale per method).
    # ------------------------------------------------------------------

    async def query_entities_by_keyword(self, *, query: str, top_k: int) -> list[EntityWithLineage]:
        return await self._inner.query_entities_by_keyword(query=query, top_k=top_k)

    async def list_entity_labels(self) -> list[str]:
        return await self._inner.list_entity_labels()

    async def list_entities(
        self,
        *,
        label: str | None = None,
        limit: int = 1000,
        offset: int = 0,
    ) -> list[EntityWithLineage]:
        # Filter is by ``label`` (entity type) + pagination, not by name —
        # rows returned are already canonical (alias rows live in
        # ``aperag_lineage_entity_alias``, not in the entity table).
        # Same passthrough rationale as ``list_entity_labels`` /
        # ``query_entities_by_keyword``.
        return await self._inner.list_entities(label=label, limit=limit, offset=offset)


__all__ = ["LineageGraphStoreWithAliasRedirect"]
