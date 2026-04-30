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
    * ``description`` — preserved for backward-compat with the legacy
      DTO shape; Wave 5 description-NULL invariant (task #31 A3, spec
      § 3.1.5) means new dedup detection paths must NOT read this
      field. ``from_lineage`` always sets it to ``""`` — the Wave 5
      graph extractor no longer emits ``description_parts`` /
      ``compacted_description``, so any non-empty value here would be
      stale residue. Field kept (instead of removed) only so existing
      callers that pass ``description=""`` keyword don't break;
      boundary test ``test_graph_curation_description_free`` grep-
      zeroes any ``entity.description`` *read* in
      ``aperag/graph_curation/**`` + ``aperag/indexing/merge_candidate_detector.py``.
    * ``source_chunk_ids`` — flattened across all
      ``EntityWithLineage.source_lineage`` members so the candidate
      generator's ``shared_chunks`` heuristic still works.
    """

    entity_id: str
    collection_id: str
    name: str
    type: str
    # Wave 5 description-NULL invariant (task #31 A3): always ``""``
    # for entities materialised via :meth:`from_lineage`. Field kept
    # default-empty for backward-compat with callers that explicitly
    # pass ``description=""`` (or pre-Wave-5 fixtures); new code paths
    # must not read it (boundary test enforced).
    description: str = ""
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

        Wave 5 description-NULL invariant (task #31 A3, spec § 3.1.5
        item 4): description input no longer depends on
        ``entity.compacted_description`` / ``entity.description_parts``
        — Wave 5 graph extractor stopped emitting them, so the legacy
        derivation would always collapse to ``""`` and any non-empty
        value (e.g. on a stale row from before the cut-over) would
        leak into dedup scoring. Set explicitly to ``""`` here so the
        invariant is enforced at the boundary, not silently relied on
        downstream.
        """
        chunk_ids: list[str] = []
        for member in entity.source_lineage:
            chunk_ids.extend(member.chunk_ids)

        return cls(
            entity_id=entity.name,
            collection_id=collection_id,
            name=entity.name,
            type=entity.entity_type,
            description="",
            source_chunk_ids=tuple(chunk_ids),
        )


__all__ = ["CurationEntity"]
