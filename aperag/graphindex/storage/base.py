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

"""Backend-agnostic storage contract for graphindex v2.

This Protocol is intentionally smaller than LightRAG's ``BaseGraphStorage``
(10 methods vs 24). Rule of thumb: every method here has a real caller in
``engine/`` or ``service.py``. No "nice-to-have" defaults; no batch
variants that duplicate the single-item version; no hooks for features
we don't use.

Backends (``postgres.py`` today; ``neo4j.py``, ``nebula.py`` later if
customer demand appears) implement this Protocol against whatever native
query language they speak. All inputs / outputs are DTOs from
``aperag.graphindex.dto`` — no native SDK types leak across.
"""

from __future__ import annotations

from typing import Optional, Protocol, Sequence

from aperag.graphindex.dto import (
    Chunk,
    DeleteDocumentResult,
    Entity,
    KnowledgeGraph,
    Relation,
)


class GraphStore(Protocol):
    """The full contract of a graphindex backend.

    Lifecycle:
    * ``ensure_schema()`` is idempotent; called once per process at
      service startup, after that any call is a no-op.
    * There's no ``initialize_*`` / ``finalize_*`` pair — schemas live
      across the whole process and connection pooling is the backend's
      private concern.
    """

    # ---- schema / collection lifecycle --------------------------------
    async def ensure_schema(self) -> None:
        """Create the physical storage (tables / keyspace / graph space).

        Must be safe to call concurrently from many workers.
        """

    async def drop_collection(self, collection_id: str) -> None:
        """Delete every row that belongs to the given collection.

        Used by the collection-delete Celery task. This is **not** a
        DDL drop — it only touches rows tagged with ``collection_id``.
        Physical tables remain for other tenants.
        """

    # ---- writes -------------------------------------------------------
    async def upsert_chunks(self, collection_id: str, chunks: Sequence[Chunk]) -> None:
        """Insert or update chunks. Chunks are idempotent by
        ``(collection_id, chunk_id)``."""

    async def upsert_entities(self, collection_id: str, entities: Sequence[Entity]) -> None:
        """Insert entities; on id conflict merge ``source_chunk_ids`` and
        extend ``description`` (keeping the longer of the two for
        simplicity — see ``postgres.py`` for the exact policy)."""

    async def upsert_relations(self, collection_id: str, relations: Sequence[Relation]) -> None:
        """Insert relations; on ``(source_id, target_id)`` conflict merge
        ``source_chunk_ids`` and keep the max weight observed."""

    # ---- deletes ------------------------------------------------------
    async def delete_document_rows(self, collection_id: str, doc_id: str) -> DeleteDocumentResult:
        """Delete every chunk of the document, prune the chunk id from
        entities / relations, and remove entities / relations that have
        become orphan (empty ``source_chunk_ids``).

        Must run in a single DB transaction so an interrupted call leaves
        the graph consistent.
        """

    # ---- reads --------------------------------------------------------
    async def find_entities_by_names(self, collection_id: str, names: Sequence[str]) -> list[Entity]:
        """Exact-match entity lookup by name. The query path uses it
        after the embedding-based recall narrows the candidate set."""

    async def expand_neighborhood(
        self,
        collection_id: str,
        anchor_entity_ids: Sequence[str],
        max_hop: int,
        limit: int,
    ) -> tuple[list[Entity], list[Relation]]:
        """Walk ``max_hop`` steps out from the anchor set, returning up
        to ``limit`` entities plus every relation connecting them.

        ``max_hop == 0`` returns just the anchors and any relation
        **already between anchor ids** (for RAG that's usually what we
        want — "here are the entities you asked for and how they
        connect"). ``max_hop == 1`` is the standard RAG setting.
        """

    async def list_labels(self, collection_id: str) -> list[str]:
        """Distinct entity types seen in the collection, stable ordered.

        Backing the UI label dropdown; cheap query, cache-friendly.
        """

    async def list_subgraph(
        self,
        collection_id: str,
        label: Optional[str],
        max_depth: int,
        max_nodes: int,
    ) -> KnowledgeGraph:
        """Return a subgraph for visualization.

        * ``label=None`` or ``label="*"``: pick top-degree entities
          across all types.
        * Specific ``label``: start from entities of that type, BFS
          outward up to ``max_depth``.

        ``max_nodes`` is a hard cap; when exceeded the backend must set
        ``KnowledgeGraph.is_truncated = True`` so the UI can show a
        "showing first N" banner.
        """


__all__ = ["GraphStore"]
