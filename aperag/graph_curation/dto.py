# Copyright 2026 ApeCloud, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Curation-side data transfer objects.

Wave 7 W7-10 (close-out) introduces ``CurationEntity`` as the
curation-flow consumer DTO replacing the legacy
``aperag.domains.knowledge_graph.graphindex.dto.Entity``. The shape is
intentionally a 1:1 mirror of the legacy class so the
production-validated ``build_candidate_pairs`` /
``_pair_score`` / ``_jaccard`` algorithms keep their existing
signatures (per architect Q2 ratify msg=838d57c3 — adapter pattern is
``simple-stable directive #3``: changing a production-validated
algorithm is a higher-risk change than introducing a thin adapter).

Construction path: callers materialise an ``EntityWithLineage`` via
``LineageGraphStore.list_entities`` / ``get_entity`` /
``query_entities_by_keyword`` and adapt it via
``CurationEntity.from_lineage(...)`` before feeding to the candidate
generation algorithm.

The DTO is frozen + tuple-coerced so it remains hashable (the candidate
generation algorithm uses entity ids in dict keys).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:  # pragma: no cover — TYPE_CHECKING import avoids circular dep
    from aperag.indexing.graph import EntityWithLineage


@dataclass(frozen=True)
class CurationEntity:
    """Curation-flow entity DTO (mirrors legacy ``Entity`` shape).

    Field semantics:

    * ``entity_id`` — stable natural key. For Wave 7 lineage entities
      this is the entity ``name`` (per Wave 4 design, name IS the
      natural key). Legacy callers used a hash of
      ``(collection_id, normalized_name)`` but the lineage backend now
      enforces uniqueness on ``(collection_id, name)`` directly so the
      hash layer is redundant.
    * ``collection_id`` — kept for backward-compat with the legacy DTO
      shape. Curation services bind a per-collection store instance,
      so this field is informational; the algorithm itself does not
      branch on it.
    * ``name`` — same as legacy.
    * ``type`` — entity type / label. Pulled from
      ``EntityWithLineage.entity_type``.
    * ``description`` — single-string view used by ``_pair_score`` for
      Jaccard token overlap. Prefers
      ``EntityWithLineage.compacted_description`` when populated by the
      Wave 7 ``GraphIndexCompactor``, falling back to the joined
      per-doc ``description_parts`` text so newly-extracted entities
      that have not yet hit the compactor still produce a usable
      similarity signal.
    * ``source_chunk_ids`` — flattened across all
      ``EntityWithLineage.source_lineage`` members so the candidate
      generator's ``shared_chunks`` heuristic still works.
    """

    entity_id: str
    collection_id: str
    name: str
    type: str
    description: str
    source_chunk_ids: Sequence[str] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.entity_id:
            raise ValueError("CurationEntity.entity_id must be non-empty")
        if not self.name:
            raise ValueError("CurationEntity.name must be non-empty")
        # ``frozen=True`` blocks normal assignment; ``object.__setattr__``
        # is the canonical way to coerce to a tuple inside a frozen
        # dataclass post-init (matches legacy ``Entity.__post_init__``).
        object.__setattr__(self, "source_chunk_ids", tuple(self.source_chunk_ids))

    @classmethod
    def from_lineage(
        cls,
        entity: "EntityWithLineage",
        *,
        collection_id: str,
    ) -> "CurationEntity":
        """Build a ``CurationEntity`` from an ``EntityWithLineage``.

        ``collection_id`` is supplied by the caller because the
        ``EntityWithLineage`` view is per-collection-bound at the
        store level and does not carry the id field on the row.
        """
        if entity.compacted_description:
            description = entity.compacted_description
        else:
            description = "\n\n".join(p.text for p in entity.description_parts if p.text)

        chunk_ids: list[str] = []
        for member in entity.source_lineage:
            chunk_ids.extend(member.chunk_ids)

        return cls(
            entity_id=entity.name,
            collection_id=collection_id,
            name=entity.name,
            type=entity.entity_type,
            description=description,
            source_chunk_ids=tuple(chunk_ids),
        )


__all__ = ["CurationEntity"]
