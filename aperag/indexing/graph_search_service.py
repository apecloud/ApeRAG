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

"""Graph search service — Wave 7 §K.12.5 task #5.

Activates vector recall against the lineage graph. The Wave 6 #33 chunk
3 retrieval cutover only wired keyword recall via
``LineageGraphStore.query_entities_by_keyword`` (chunk 2 vector recall
was deferred); this service finally adds the vector path and composes
both into the LightRAG-style context block the retrieval pipeline
already consumes (so task #8's wiring is a one-line swap).

Per-collection bound — caller wires the four dependencies (lineage
store, vector connector, embedder, optional LLM) to the collection
being searched before constructing the service. The constructor does
no I/O. ``llm`` is reserved for follow-up high-level / low-level
keyword extraction (LightRAG hybrid recall); v1 PR does not invoke it.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Awaitable, Callable, Iterable, Sequence

from aperag.indexing.graph import (
    EntityWithLineage,
    LineageGraphStore,
    RelationWithLineage,
)
from aperag.vectorstore.base import VectorStoreConnector
from aperag.vectorstore.dto import QueryRequest, SearchHit
from aperag.vectorstore.filters import Eq

logger = logging.getLogger(__name__)


# Default vector recall knobs. Mirror the legacy
# ``aperag/graph_curation/service.py`` defaults so sync-driven detector
# (task #4) and search-time recall use the same scoring frontier.
DEFAULT_TOP_K: int = 10
DEFAULT_SCORE_THRESHOLD: float = 0.0
"""``0.0`` here means *no threshold* — the search service surfaces every
hit the vector store returned, leaving threshold tuning to callers.
The detector (task #4) tightens this to ``0.72`` because false-positive
candidates are more expensive than false-positive recall."""


# Vector point indexer tag — pinned by §K.12 invariant #5. Task #3
# writes graph-entity vectors with ``payload["indexer"] ==
# "graph_entity"``; this service filters on the same tag so chunk /
# summary vectors sharing the physical collection never bleed into
# graph recall.
GRAPH_ENTITY_INDEXER: str = "graph_entity"


class GraphSearchService:
    """Vector + graph recall for a single collection.

    Public surface (per §K.12.5 spec, locked by architect ratify
    msg=acbd0003):

    * :meth:`search_entities` — embed the query, ANN-search the entity
      vector index, fetch the matching ``EntityWithLineage`` rows.
    * :meth:`search_relations` — derive relations attached to the
      vector-recalled entities (1-hop expansion).
    * :meth:`get_subgraph` — wrap ``expand_neighbors_n_hops`` as a
      stable read primitive for MCP tools (task #7).
    * :meth:`compose_context` — render the ``EntityWithLineage`` /
      ``RelationWithLineage`` lists into the legacy LightRAG-style
      ``-----Entities (KG)----- / -----Relationships (KG)-----`` text
      block so the retrieval pipeline (task #8) gets a drop-in
      replacement for the keyword-only path.
    """

    def __init__(
        self,
        *,
        store: LineageGraphStore,
        vector_connector: VectorStoreConnector,
        embedder: Any,
        # Reserved for follow-up: LightRAG-style high/low-level
        # keyword extraction from the user query. Not invoked in v1.
        llm: Callable[[str], Awaitable[str]] | None = None,
        top_k: int = DEFAULT_TOP_K,
        score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    ) -> None:
        self._store = store
        self._vector_connector = vector_connector
        self._embedder = embedder
        self._llm = llm
        self._top_k = top_k
        self._score_threshold = score_threshold

    # ------------------------------------------------------------------
    # search_entities — vector recall path
    # ------------------------------------------------------------------

    async def search_entities(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[EntityWithLineage]:
        """Vector-recall the top-K entities matching ``query``.

        Empty / whitespace-only query returns ``[]`` (no spurious
        recall — same convention as
        :meth:`LineageGraphStore.query_entities_by_keyword`).
        """
        text = (query or "").strip()
        if not text:
            return []
        k = top_k if top_k is not None else self._top_k
        if k <= 0:
            return []

        try:
            embedding = await asyncio.to_thread(self._embedder.embed_query, text)
        except Exception:
            logger.warning("graph_search_service: embed failed for query=%r", text, exc_info=True)
            return []

        request = QueryRequest(
            embedding=embedding,
            top_k=k,
            flt=Eq("indexer", GRAPH_ENTITY_INDEXER),
            score_threshold=self._score_threshold or None,
        )
        try:
            hits = await asyncio.to_thread(self._vector_connector.search, request)
        except Exception:
            logger.warning(
                "graph_search_service: vector search failed for query=%r",
                text,
                exc_info=True,
            )
            return []

        names = self._entity_names_from_hits(hits)
        if not names:
            return []
        return await self._batch_get_entities(names)

    @staticmethod
    def _entity_names_from_hits(hits: Iterable[SearchHit]) -> list[str]:
        # Preserve hit order (vector store already ranked by score) and
        # de-dup so the same entity returned twice (alias point + canonical
        # point) only fetches once.
        seen: set[str] = set()
        ordered: list[str] = []
        for hit in hits:
            name = hit.payload.get("entity_name") or hit.payload.get("name")
            if not name:
                continue
            name_str = str(name)
            if name_str in seen:
                continue
            seen.add(name_str)
            ordered.append(name_str)
        return ordered

    async def _batch_get_entities(self, names: Sequence[str]) -> list[EntityWithLineage]:
        # Architect ratify Q2 (msg=acbd0003): batch fetch via
        # ``asyncio.gather`` rather than a Protocol-level batch method —
        # N is bounded by ``top_k`` (≤ 10 in practice) so per-name
        # round-trip cost is acceptable, and it keeps the
        # :class:`LineageGraphStore` Protocol surface narrow (per
        # earayu2 simple-stable directive #1, 不无限扩范围).
        results = await asyncio.gather(
            *(self._store.get_entity(name) for name in names),
            return_exceptions=False,
        )
        return [e for e in results if e is not None]

    # ------------------------------------------------------------------
    # search_relations — derived from search_entities
    # ------------------------------------------------------------------

    async def search_relations(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[RelationWithLineage]:
        """Recall relations attached to the entities that vector-search
        retrieved for ``query``.

        Vector store does not carry separate relation vectors (Wave 7
        scope: only entity vectors are written by task #3; per-relation
        vectors are out-of-scope per architect msg=acbd0003 §K.12.5).
        We therefore derive relations as the 1-hop expansion of the
        entity-search result.
        """
        entities = await self.search_entities(query, top_k=top_k)
        if not entities:
            return []
        _neighbour_entities, relations = await self._store.expand_neighbors_n_hops(
            entity_names=[e.name for e in entities],
            hops=1,
        )
        return relations

    # ------------------------------------------------------------------
    # get_subgraph — MCP-facing read primitive (task #7)
    # ------------------------------------------------------------------

    async def get_subgraph(
        self,
        entity_names: Sequence[str],
        hops: int = 1,
    ) -> tuple[list[EntityWithLineage], list[RelationWithLineage]]:
        """Return entities + relations reachable from ``entity_names``
        within ``hops`` graph traversal steps.

        Thin pass-through to
        :meth:`LineageGraphStore.expand_neighbors_n_hops` so MCP tool
        callers (task #7) and retrieval pipeline (task #8) share one
        traversal surface — backend-specific traversal cost is the
        store's concern."""
        if not entity_names:
            return ([], [])
        return await self._store.expand_neighbors_n_hops(
            entity_names=list(entity_names),
            hops=hops,
        )

    # ------------------------------------------------------------------
    # compose_context — legacy-shape rendering for retrieval pipeline
    # ------------------------------------------------------------------

    @staticmethod
    def compose_context(
        entities: Sequence[EntityWithLineage],
        relations: Sequence[RelationWithLineage],
    ) -> str:
        """Render entities + relations into the LightRAG-style block
        the retrieval pipeline already expects.

        Mirrors :func:`aperag.domains.retrieval.pipeline._render_graph_context_text`
        byte-for-byte so task #8's swap (from in-pipeline rendering to
        ``GraphSearchService.compose_context``) is zero-functional-change
        — same context text, same downstream RAG prompt behaviour. The
        helper here also collapses ``compacted_description`` (task #1
        derived field) into the rendering when present, so callers that
        compact pre-render see the condensed text instead of raw parts.

        Returns ``""`` when both lists are empty so callers can branch
        on truthiness without an explicit length check.
        """
        if not entities and not relations:
            return ""
        lines: list[str] = []
        if entities:
            lines.append("-----Entities (KG)-----")
            for e in entities:
                desc = GraphSearchService._render_description(
                    getattr(e, "compacted_description", None),
                    e.description_parts or (),
                )
                type_label = e.entity_type or "entity"
                lines.append(f"- [{type_label}] {e.name} — {desc}")
        if relations:
            if lines:
                lines.append("")
            lines.append("-----Relationships (KG)-----")
            for r in relations:
                desc = GraphSearchService._render_description(
                    getattr(r, "compacted_description", None),
                    r.description_parts or (),
                )
                lines.append(f"- {r.source} → {r.target}: {desc}")
        return "\n".join(lines)

    @staticmethod
    def _render_description(compacted: str | None, parts: Iterable[Any]) -> str:
        if compacted and str(compacted).strip():
            return str(compacted).strip()
        fragments = [p.text.strip() for p in parts if getattr(p, "text", None) and p.text.strip()]
        return " | ".join(fragments) or "(no description)"


# ----------------------------------------------------------------------
# Per-collection factory — Wave 7 §K.12.6/§K.12.7/§K.12.8 task #7+#8
# ----------------------------------------------------------------------


def build_graph_search_service_for(collection: Any) -> GraphSearchService:
    """Construct a :class:`GraphSearchService` bound to ``collection``.

    Mirrors :func:`aperag.domains.retrieval.pipeline._build_lineage_graph_store_for`
    factory pattern so MCP tool view handlers (task #7) and the
    retrieval pipeline (task #8) share one wiring path. Lazy imports
    keep the module free of indexing / config side-effects at import
    time.

    The four dependencies are resolved per-collection:

    * ``store`` — :class:`LineageGraphStore` via existing
      ``_build_lineage_graph_store`` (Wave 6 #33 / #40 pattern).
    * ``vector_connector`` — Qdrant / pgvector via the shared
      ``_build_collection_qdrant_connector`` helper (Wave 5 P5B).
    * ``embedder`` — ``embedding_service`` returned by
      ``get_collection_embedding_service_sync`` (the same model the
      vector worker uses, so query-time embedding lives on the same
      vector space as write-time).
    * ``llm`` — ``None`` in v1 (kwarg reserved for follow-up
      high-level / low-level keyword extraction; not invoked).
    """
    from aperag.indexing.worker_factory import (
        _build_collection_qdrant_connector,
        _build_lineage_graph_store,
        _resolve_graph_backend_type,
    )

    backend_type = _resolve_graph_backend_type(collection)
    store = _build_lineage_graph_store(backend_type=backend_type, collection=collection)
    adaptor, embedder, _vector_size = _build_collection_qdrant_connector(collection)
    return GraphSearchService(
        store=store,
        vector_connector=adaptor.connector,
        embedder=embedder,
        llm=None,
    )


__all__ = [
    "GraphSearchService",
    "DEFAULT_TOP_K",
    "DEFAULT_SCORE_THRESHOLD",
    "GRAPH_ENTITY_INDEXER",
    "build_graph_search_service_for",
]
