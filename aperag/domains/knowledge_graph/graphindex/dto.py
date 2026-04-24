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

"""Data Transfer Objects for the graphindex module.

All types callers see belong here. Internal types (LLM response shapes,
SQL row shapes) live next to their implementations and never cross the
module boundary.

Rules when adding / changing types here:

* Frozen dataclasses only — structural equality, hashability, immutability.
* No imports from ``aperag.domains.knowledge_graph.graphindex.engine`` / ``storage`` / ``models``:
  DTOs are the contract between callers and implementation; the arrow is
  one-way.
* No imports from third-party backend SDKs (``neo4j``, ``pgvector`` etc.).
  If a DTO needs a backend-shaped field, extract the shape into a scalar
  value first.
* Every new field MUST have a docstring explaining what it represents and
  whether it's optional. A DTO is a specification; "look at the code" is
  not an acceptable answer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

# ---------------------------------------------------------------------------
# Core graph primitives
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Chunk:
    """A piece of text extracted from a source document.

    ``chunk_id`` is a stable UUID assigned at ingest time; extraction
    results reference chunks by this id.
    """

    chunk_id: str
    doc_id: str
    collection_id: str
    order_in_doc: int
    text: str
    file_path: str = ""

    def __post_init__(self) -> None:
        if not self.chunk_id:
            raise ValueError("Chunk.chunk_id must be non-empty")
        if self.order_in_doc < 0:
            raise ValueError("Chunk.order_in_doc must be >= 0")


@dataclass(frozen=True)
class Entity:
    """An entity extracted by the LLM, potentially merged from several chunks.

    ``entity_id`` is the stable key used to deduplicate across chunks and
    across extraction rounds: it is computed as a hash of
    ``(collection_id, normalized_name)`` so the same entity name in one
    collection always resolves to the same id.
    """

    entity_id: str
    collection_id: str
    name: str
    type: str
    description: str
    source_chunk_ids: Sequence[str] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.entity_id:
            raise ValueError("Entity.entity_id must be non-empty")
        if not self.name:
            raise ValueError("Entity.name must be non-empty")
        # Store ``source_chunk_ids`` as a tuple so the whole DTO is hashable
        # without the caller having to remember to pass one. Frozen dataclass
        # forbids normal assignment; object.__setattr__ bypasses it.
        object.__setattr__(self, "source_chunk_ids", tuple(self.source_chunk_ids))


@dataclass(frozen=True)
class Relation:
    """A directed edge between two entities.

    ``weight`` is an LLM-assigned confidence score (1..10 range in the
    prompt); callers treat higher as more important but should not assume
    a specific scale.
    """

    collection_id: str
    source_id: str
    target_id: str
    description: str
    weight: float = 1.0
    source_chunk_ids: Sequence[str] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.source_id or not self.target_id:
            raise ValueError("Relation endpoints must be non-empty")
        if self.source_id == self.target_id:
            raise ValueError("Relation must not be a self-loop")
        object.__setattr__(self, "source_chunk_ids", tuple(self.source_chunk_ids))


# ---------------------------------------------------------------------------
# Compound results (returned from ``GraphIndexService`` methods)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IndexDocumentResult:
    """Summary of what a single ``index_document`` call produced."""

    doc_id: str
    chunks_created: int
    entities_extracted: int
    relations_extracted: int


@dataclass(frozen=True)
class DeleteDocumentResult:
    """Rows removed from the three physical tables during ``delete_document``.

    Figures are *rows touched* in the storage layer; an ``entity`` that
    originated from multiple documents has its ``source_chunk_ids`` list
    pruned (1 row updated) rather than deleted (0 rows deleted) — so
    ``entities_removed`` only counts entities that became orphan.
    """

    doc_id: str
    chunks_removed: int
    entities_removed: int
    relations_removed: int


@dataclass(frozen=True)
class GraphContext:
    """Compact context returned by ``query_context``, suitable for stuffing
    into an LLM prompt.

    ``text`` is the pre-rendered human-readable block (entities + relations
    + supporting chunks). Callers that want the structured form can read
    ``entities`` / ``relations`` / ``chunks`` directly.
    """

    text: str
    entities: Sequence[Entity]
    relations: Sequence[Relation]
    chunks: Sequence[Chunk]

    def __post_init__(self) -> None:
        object.__setattr__(self, "entities", tuple(self.entities))
        object.__setattr__(self, "relations", tuple(self.relations))
        object.__setattr__(self, "chunks", tuple(self.chunks))


# Separator placed between accumulated description fragments. Storage
# backends use it in their ``UPDATE ... SET description = ...`` concat
# expression; the service layer splits on it to recover fragments before
# sending them to the summarization LLM. Kept here (in the data-contract
# module) rather than in any specific backend so every layer agrees on
# the format.
DESCRIPTION_SEPARATOR: str = "\n\n"


@dataclass(frozen=True)
class MergeEntitiesResult:
    """Summary of a ``merge_entities`` call returned from the storage
    layer. The service layer consumes this to decide whether to run
    post-merge LLM summarization on the target description.

    Fields:

    * ``target_entity_id`` — the surviving entity id (echoed so callers
      don't have to track it separately from the ``Result``).
    * ``merged_source_ids`` — source rows that were actually deleted.
      May be shorter than the caller's ``source_entity_ids`` list when
      some ids didn't exist; the merge is idempotent in that direction.
    * ``description`` — the target's post-merge description, BEFORE
      any LLM summarization. The service layer inspects this to decide
      whether to trigger a summary pass.
    * ``source_chunk_ids`` — union of every merged source's chunks plus
      the target's own.
    * ``edges_redirected`` — edges whose endpoint was rewritten from a
      source entity to the target.
    * ``edges_collapsed`` — edges dropped because redirecting produced a
      self-loop (``target == target``) or duplicate of a pre-existing
      ``(target, X)`` edge. Chunk ids of collapsed duplicates are
      unioned into the surviving edge so provenance is preserved.
    """

    target_entity_id: str
    merged_source_ids: tuple[str, ...]
    description: str
    source_chunk_ids: tuple[str, ...]
    edges_redirected: int
    edges_collapsed: int


@dataclass(frozen=True)
class KnowledgeGraph:
    """A subgraph returned for UI display.

    ``is_truncated`` tells the UI to show a warning banner ("showing the
    first N nodes out of M"). Storage layers are responsible for setting
    it when they hit ``max_nodes``.
    """

    nodes: Sequence[Entity]
    edges: Sequence[Relation]
    is_truncated: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "nodes", tuple(self.nodes))
        object.__setattr__(self, "edges", tuple(self.edges))


__all__ = [
    "Chunk",
    "Entity",
    "Relation",
    "IndexDocumentResult",
    "DeleteDocumentResult",
    "MergeEntitiesResult",
    "GraphContext",
    "KnowledgeGraph",
    "DESCRIPTION_SEPARATOR",
]
