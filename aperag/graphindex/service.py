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

"""``GraphIndexService`` — the single public entry point of the graphindex
module.

Everything outside ``aperag/graphindex/`` that touches knowledge-graph
functionality goes through this class. No exceptions — not ``engine``,
not ``storage``, not ``prompts``. If a caller needs to reach below the
facade, that's a signal to grow the facade, not to bypass it.

The service is **stateless** beyond its injected dependencies (store,
llm, embedding, config). One instance per process is typical; one
instance per test is also fine. Connection pools live in the injected
store / embedding service, not here.
"""

from __future__ import annotations

import logging
from typing import Awaitable, Callable, List, Optional

from aperag.graphindex.config import GraphIndexConfig
from aperag.graphindex.dto import (
    Chunk,
    DeleteDocumentResult,
    Entity,
    GraphContext,
    IndexDocumentResult,
    KnowledgeGraph,
    Relation,
)
from aperag.graphindex.engine import index_document
from aperag.graphindex.storage.base import GraphStore

logger = logging.getLogger(__name__)

# Shape of the injected LLM function. Returns the raw text body of the
# model response. The service expects it to honour JSON output when the
# prompt asks — wiring that into the real CompletionService is the
# deployment layer's job (``response_format={"type":"json_object"}``).
LLMCall = Callable[[str], Awaitable[str]]

# Embedding function for query-side entity anchoring. Given a query
# string, returns its vector. Typed loosely (``list[float]``) to avoid
# pulling numpy into the abstraction.
EmbedQuery = Callable[[str], Awaitable[List[float]]]


class GraphIndexService:
    """Business-facing knowledge-graph service.

    Five public methods, mapping 1:1 to the five ApeRAG business actions
    that remain in-scope for v2:

    * ``index_document`` — persist a document's graph.
    * ``delete_document`` — remove a document's graph contribution.
    * ``query_context`` — answer a question against the graph (used by
      the search pipeline).
    * ``get_labels`` — list entity types (UI dropdown).
    * ``get_knowledge_graph`` — fetch a subgraph for display.

    The three curation actions — merge suggestions, merge execution, KG
    eval export — are intentionally NOT here in v2; they still route
    through the legacy LightRAG path. A follow-up PR will port them.
    """

    def __init__(
        self,
        *,
        store: GraphStore,
        llm: LLMCall,
        embed_query: Optional[EmbedQuery] = None,
        config: Optional[GraphIndexConfig] = None,
    ) -> None:
        """Inject all dependencies. The service does not read env vars —
        that's the deployment-layer factory's job."""
        self._store = store
        self._llm = llm
        self._embed_query = embed_query
        self._config = config or GraphIndexConfig()

    # ============================================================ write
    async def index_document(
        self,
        *,
        collection_id: str,
        doc_id: str,
        content: str,
        file_path: str = "",
    ) -> IndexDocumentResult:
        """Chunk, extract, and persist a document's contribution to the
        knowledge graph.

        Rebuild-safe on ``doc_id``: the service wipes any prior rows for
        this ``(collection_id, doc_id)`` first, then runs the fresh
        extraction. Callers do NOT need to pair this with
        ``delete_document`` — update / retry / reindex are all safe.
        The wipe is idempotent on an empty slate (zero-row delete)."""
        # Rebuild semantics: drop every row whose only source is this
        # doc_id, and prune this doc's chunks from entity/relation
        # source lists on rows that also cite other docs. Without this,
        # the UUID4-based chunk_ids emitted by chunk_document() plus the
        # ARRAY union in upsert_entities / upsert_relations would cause
        # source_chunk_ids to monotonically grow on every re-index.
        await self._store.delete_document_rows(collection_id=collection_id, doc_id=doc_id)

        return await index_document(
            store=self._store,
            llm=self._llm,
            config=self._config,
            collection_id=collection_id,
            doc_id=doc_id,
            content=content,
            file_path=file_path,
        )

    async def delete_document(self, *, collection_id: str, doc_id: str) -> DeleteDocumentResult:
        """Remove every row that exists *only* because of ``doc_id``.

        Entities and relations supported by multiple documents are kept
        but have ``doc_id``'s chunks pruned from their
        ``source_chunk_ids`` list. See
        ``PostgresGraphStore.delete_document_rows`` for the exact atomic
        semantics."""
        return await self._store.delete_document_rows(collection_id=collection_id, doc_id=doc_id)

    async def drop_collection(self, *, collection_id: str) -> None:
        """Wipe the whole collection. Called when the user deletes the
        ApeRAG collection itself (not an individual document)."""
        await self._store.drop_collection(collection_id)

    # ============================================================= read
    async def query_context(
        self,
        *,
        collection_id: str,
        query: str,
        top_k: Optional[int] = None,
    ) -> GraphContext:
        """Build a compact graph-based context block for a user query.

        Current implementation (v2 simple mode):
        1. Use embedding similarity to find anchor entities by name.
           When the service was built without an ``embed_query``
           function, fall back to a straight name-match on the query
           string's tokens.
        2. BFS-expand ``config.default_query_max_hop`` hops from those
           anchors.
        3. Render a fixed-format context block.

        This is deliberately simpler than LightRAG's hybrid keyword+
        vector ranking. Measure before adding complexity — if the
        simple form covers the recall targets, the complex form is
        just more maintenance surface.
        """
        k = int(top_k or self._config.default_query_top_k)
        anchors = await self._resolve_anchor_entities(collection_id=collection_id, query=query, top_k=k)
        if not anchors:
            return GraphContext(text="", entities=[], relations=[], chunks=[])

        entities, relations = await self._store.expand_neighborhood(
            collection_id=collection_id,
            anchor_entity_ids=[e.entity_id for e in anchors],
            max_hop=self._config.default_query_max_hop,
            limit=max(k, 50),
        )

        # Chunks referenced by returned entities/relations. We rehydrate
        # a small set for citation display; RAG ranking happens upstream
        # in the search pipeline with the vector-index context.
        # For now we keep this empty to stay cheap; callers who need
        # citations can look up chunk ids from ``entities[i].source_chunk_ids``.
        chunks: List[Chunk] = []

        text = _render_context_block(entities=entities, relations=relations)
        return GraphContext(text=text, entities=entities, relations=relations, chunks=chunks)

    async def has_data(self, *, collection_id: str) -> bool:
        """Cutover gate: does this collection have any v2 rows yet?

        The business layer uses this to decide whether to read v2 or
        fall back to legacy LightRAG during the transition window. Kept
        on the service (not the store) so callers never need to import
        the store directly — the facade stays the single public surface.
        """
        return await self._store.has_collection_data(collection_id)

    async def get_labels(self, *, collection_id: str) -> list[str]:
        return await self._store.list_labels(collection_id)

    async def get_knowledge_graph(
        self,
        *,
        collection_id: str,
        label: Optional[str] = None,
        max_depth: int = 2,
        max_nodes: int = 500,
    ) -> KnowledgeGraph:
        return await self._store.list_subgraph(
            collection_id=collection_id,
            label=label,
            max_depth=max_depth,
            max_nodes=max_nodes,
        )

    # ---- anchor resolution (private) ---------------------------------
    async def _resolve_anchor_entities(
        self,
        *,
        collection_id: str,
        query: str,
        top_k: int,
    ) -> List[Entity]:
        """Return up to ``top_k`` anchor entities for the query.

        v2 simple algorithm: exact name match on whitespace-split query
        tokens. This has lower recall than embedding-based retrieval but
        doesn't require cross-service coordination with the vector
        store. When an ``embed_query`` is injected AND the deployment
        has populated entity vectors in ``aperag/vectorstore``, a richer
        implementation can be swapped in without touching callers.
        """
        names = [t for t in (query or "").split() if t.strip()]
        if not names:
            return []
        found = await self._store.find_entities_by_names(collection_id=collection_id, names=names[: max(top_k, 16)])
        return found[:top_k]


# ---------------------------------------------------------------------------
# Text rendering
# ---------------------------------------------------------------------------


def _render_context_block(*, entities: List[Entity], relations: List[Relation]) -> str:
    """Pack entities + relations into a deterministic text block for the
    RAG prompt."""
    if not entities and not relations:
        return ""

    id_to_name = {e.entity_id: e.name for e in entities}

    lines: list[str] = []
    if entities:
        lines.append("Entities:")
        for e in entities:
            desc = e.description.strip() or "(no description)"
            lines.append(f"- [{e.type}] {e.name} — {desc}")
    if relations:
        lines.append("")
        lines.append("Relations:")
        for r in relations:
            src = id_to_name.get(r.source_id, r.source_id)
            tgt = id_to_name.get(r.target_id, r.target_id)
            desc = r.description.strip() or "(no description)"
            lines.append(f"- {src} → {tgt}: {desc} (weight={r.weight})")

    return "\n".join(lines)


__all__ = ["GraphIndexService", "LLMCall", "EmbedQuery"]
