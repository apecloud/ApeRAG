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

from aperag.domains.knowledge_graph.graphindex.config import GraphIndexConfig
from aperag.domains.knowledge_graph.graphindex.dto import (
    DESCRIPTION_SEPARATOR,
    Chunk,
    DeleteDocumentResult,
    Entity,
    GraphContext,
    IndexDocumentResult,
    KnowledgeGraph,
    MergeEntitiesResult,
    Relation,
)
from aperag.domains.knowledge_graph.graphindex.engine import index_document
from aperag.domains.knowledge_graph.graphindex.prompts import render_summarization_prompt
from aperag.domains.knowledge_graph.graphindex.storage.base import GraphStore

logger = logging.getLogger(__name__)

# Shape of the injected LLM function. Returns the raw text body of the
# model response. The service expects it to honour JSON output when the
# prompt asks — wiring that into the real CompletionService is the
# deployment layer's job (``response_format={"type":"json_object"}``).
LLMCall = Callable[[str], Awaitable[str]]

# Embedding function: given a list of texts, return a list of vectors.
EmbedTexts = Callable[[List[str]], Awaitable[List[List[float]]]]

# Convenience alias: embed a single query text, return one vector.
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
    * ``merge_entities`` — consolidate N entities into one (the
      replacement for LightRAG's "merge nodes" curation action).

    Merge-suggestion discovery (offering candidate clusters to a human)
    and KG-eval export are not in v2. Both were thin wrappers over
    LightRAG internals; their v2 replacements belong in a separate
    curation module once we have a concrete product design for them —
    building them now would be speculative and the current code base
    can re-add them without touching this service.
    """

    def __init__(
        self,
        *,
        store: GraphStore,
        llm: LLMCall,
        embed_query: Optional[EmbedQuery] = None,
        embed_texts: Optional[EmbedTexts] = None,
        vector_connector: Optional[object] = None,
        config: Optional[GraphIndexConfig] = None,
    ) -> None:
        """Inject all dependencies. The service does not read env vars —
        that's the deployment-layer factory's job.

        ``embed_texts`` and ``vector_connector`` are optional for
        backwards compatibility; when provided, entity/relation
        embeddings are stored in the vectorstore at index time and used
        for semantic recall at query time. When absent, the service
        falls back to the name-match anchor resolution.
        """
        self._store = store
        self._llm = llm
        self._embed_query = embed_query
        self._embed_texts = embed_texts
        self._vector_connector = vector_connector
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
        pre_entity_ids, pre_relation_keys = await self._snapshot_shadow_identity(collection_id)
        await self._store.delete_document_rows(collection_id=collection_id, doc_id=doc_id)

        result = await index_document(
            store=self._store,
            llm=self._llm,
            config=self._config,
            collection_id=collection_id,
            doc_id=doc_id,
            content=content,
            file_path=file_path,
        )

        # Normalization pass: any entity / relation whose description
        # has accumulated enough fragments to be hard for humans (and
        # expensive for retrieval prompts) gets LLM-summarized into a
        # single coherent paragraph. This is the v2 replacement for
        # LightRAG's ``force_llm_summary_on_merge`` behaviour — without
        # destroying information, unlike the plain truncation approach
        # that was tried and rejected in an earlier revision of this
        # module.
        await self._compact_oversized_descriptions(collection_id=collection_id)

        # Embed entity/relation descriptions into the vectorstore so
        # query_context can find them via semantic similarity. Runs
        # after summarization so the embedded text is the compact form.
        post_entity_ids, post_relation_keys = await self._snapshot_shadow_identity(collection_id)
        await self._delete_removed_shadow_vectors(
            removed_entity_ids=pre_entity_ids - post_entity_ids,
            removed_relation_keys=pre_relation_keys - post_relation_keys,
            reason="document rebuild",
        )
        await self._sync_entity_relation_vectors(collection_id=collection_id)

        return result

    async def delete_document(self, *, collection_id: str, doc_id: str) -> DeleteDocumentResult:
        """Remove every row that exists *only* because of ``doc_id``.

        Shadow vector lifecycle is closed via a snapshot-diff approach:

        1. Snapshot the set of surviving entity/relation ids BEFORE the
           topology delete (we need the pre-delete set to know what
           might vanish).
        2. Run the topology delete.
        3. Snapshot the surviving set AFTER.
        4. The diff (pre - post) gives us the ids of entities/relations
           that were removed — delete their shadow vectors by
           deterministic id (``ge_{eid}`` / ``gr_{src}_{tgt}``).
        5. Re-embed surviving entities/relations whose descriptions may
           have changed (chunk-pruning can shorten them).

        This avoids the flawed "ANN search as list-all" approach that
        the first attempt used. Shadow vector ids are deterministic
        functions of entity/relation ids, so we never need to enumerate
        the vectorstore to find stale shadows.
        """
        # 1. Pre-delete snapshot
        pre_entity_ids, pre_relation_keys = await self._snapshot_shadow_identity(collection_id)

        # 2. Topology delete
        result = await self._store.delete_document_rows(collection_id=collection_id, doc_id=doc_id)

        # 3. Post-delete snapshot
        post_entity_ids, post_relation_keys = await self._snapshot_shadow_identity(collection_id)

        # 4. Delete shadow vectors for removed entities/relations
        await self._delete_removed_shadow_vectors(
            removed_entity_ids=pre_entity_ids - post_entity_ids,
            removed_relation_keys=pre_relation_keys - post_relation_keys,
            reason="document delete",
        )

        # 5. Re-embed surviving entities/relations (descriptions may have changed)
        await self._sync_entity_relation_vectors(collection_id=collection_id)

        return result

    async def drop_collection(self, *, collection_id: str) -> None:
        """Wipe the whole collection. Called when the user deletes the
        ApeRAG collection itself (not an individual document).

        Cleans both the graph topology and any shadow vectors in the
        vectorstore. Chunk vectors managed by the vector index pipeline
        are untouched.
        """
        # Snapshot all entity/relation ids before dropping topology, so
        # we can delete their shadow vectors by deterministic id.
        entity_ids, relation_keys = await self._snapshot_shadow_identity(collection_id)

        await self._store.drop_collection(collection_id)

        await self._delete_removed_shadow_vectors(
            removed_entity_ids=entity_ids,
            removed_relation_keys=relation_keys,
            reason="collection drop",
        )

    async def merge_entities(
        self,
        *,
        collection_id: str,
        target_entity_id: str,
        source_entity_ids: List[str],
    ) -> MergeEntitiesResult:
        """Merge several entities into one.

        This is the v2 replacement for the "merge nodes" curation action
        the UI used to call. Compared with LightRAG's multi-step
        pipeline, the implementation here is two simple passes:

        1. ``GraphStore.merge_entities`` performs the structural merge
           in a single SQL transaction — union source chunks, redirect
           edges, collapse duplicates, delete source rows.
        2. The merged description (every source fragment appended to
           the target's) is then handed to the LLM to produce ONE
           coherent paragraph that keeps every fact. Without this step
           the merged description would just be N patchwork fragments;
           with it, the target reads as a single entity just like
           LightRAG's original merge output — only with less code and
           without the ``merge_suggestion`` schema baggage.

        If the post-merge description is already under the
        summarization threshold (small merge, short fragments), step 2
        is skipped to save a round-trip.
        """
        result = await self._store.merge_entities(
            collection_id=collection_id,
            target_entity_id=target_entity_id,
            source_entity_ids=source_entity_ids,
        )

        if self._should_summarize(result.description):
            summary = await self._summarize_description(
                subject_kind="entity",
                subject_label=target_entity_id,
                description=result.description,
            )
            if summary:
                await self._store.rewrite_entity_description(
                    collection_id=collection_id,
                    entity_id=target_entity_id,
                    description=summary,
                )
                result = MergeEntitiesResult(
                    target_entity_id=result.target_entity_id,
                    merged_source_ids=result.merged_source_ids,
                    description=summary,
                    source_chunk_ids=result.source_chunk_ids,
                    edges_redirected=result.edges_redirected,
                    edges_collapsed=result.edges_collapsed,
                )

        return result

    async def list_entities_for_curation(
        self,
        *,
        collection_id: str,
        limit: int,
    ) -> List[Entity]:
        """Return up to ``limit`` entities for offline curation analysis.

        This stays a bounded read on graph truth rather than introducing a
        second enumeration path just for curation. Passing ``0`` thresholds
        intentionally disables the "oversized" predicate and lets the store
        act as a simple collection-scoped entity listing primitive.
        """
        return await self._store.find_oversized_entities(
            collection_id=collection_id,
            min_chars=0,
            min_fragments=0,
            limit=limit,
        )

    async def find_entity_shadow_neighbors(
        self,
        *,
        collection_id: str,
        entity_ids: List[str],
        top_k_per_entity: int,
        score_threshold: float,
    ) -> dict[str, list[tuple[str, float]]]:
        """Return nearest ``graph_entity`` shadow-vector neighbors.

        Used by the graph-curation workflow to enrich deterministic name
        blocking with semantic recall while staying inside the existing
        vectorstore abstraction. The search is restricted to the
        caller-provided ``entity_ids`` set so offline curation stays
        bounded to one analysis run.
        """
        del collection_id  # Tenant scoping is already enforced by the connector.
        if self._vector_connector is None or not entity_ids or top_k_per_entity <= 0:
            return {}

        from aperag.vectorstore.dto import QueryRequest
        from aperag.vectorstore.filters import Eq, _in, all_of

        connector = self._vector_connector
        points = connector.retrieve([f"ge_{entity_id}" for entity_id in entity_ids], with_vectors=True)
        if not points:
            return {}

        allowed_ids = list(dict.fromkeys(entity_ids))
        flt = all_of(Eq("indexer", "graph_entity"), _in("entity_id", allowed_ids))
        neighbors: dict[str, list[tuple[str, float]]] = {}
        for point in points:
            source_id = (point.payload or {}).get("entity_id")
            if not source_id or not point.vector:
                continue
            hits = connector.search(
                QueryRequest(
                    embedding=point.vector,
                    top_k=top_k_per_entity + 1,
                    flt=flt,
                    score_threshold=score_threshold,
                )
            )
            neighbor_list: list[tuple[str, float]] = []
            for hit in hits:
                target_id = (hit.payload or {}).get("entity_id")
                if not target_id or target_id == source_id:
                    continue
                neighbor_list.append((target_id, float(hit.score)))
            if neighbor_list:
                neighbors[source_id] = neighbor_list
        return neighbors

    # ============================================================= read
    async def query_context(
        self,
        *,
        collection_id: str,
        query: str,
        top_k: Optional[int] = None,
    ) -> GraphContext:
        """Build a compact graph-based context block for a user query.

        Pipeline:
        1. **Anchor resolution**: embed the query -> search entity and
           relation vectors in the vectorstore (``index_type`` filter)
           -> collect anchor entity ids. Falls back to name-match when
           no vectorstore is configured.
        2. **BFS expansion**: walk ``config.default_query_max_hop`` hops
           from the anchors to gather related entities and relations.
        3. **Chunk rehydration**: fetch the actual document chunk text
           that the entities/relations point at via ``source_chunk_ids``,
           so the graph context includes supporting evidence alongside
           structured knowledge. This restores the "Document Chunks"
           section that LightRAG's output included.
        4. **Render**: format entities, relations, and chunk excerpts
           into a deterministic text block for the RAG prompt.
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

        # Rehydrate referenced chunks so the graph context includes
        # supporting evidence. Cap at 2x top_k to keep the context
        # manageable (the full chunk set can be very large on
        # high-frequency entities).
        chunk_ids: set[str] = set()
        for e in entities:
            chunk_ids.update(e.source_chunk_ids or ())
        max_chunks = max(k * 2, 20)
        chunks: List[Chunk] = []
        if chunk_ids:
            chunks = await self._store.get_chunks_by_ids(
                collection_id=collection_id,
                chunk_ids=sorted(chunk_ids)[:max_chunks],
            )

        text = _render_context_block(entities=entities, relations=relations, chunks=chunks)
        return GraphContext(text=text, entities=entities, relations=relations, chunks=chunks)

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

    # ---- normalization (private) -------------------------------------
    def _should_summarize(self, description: str) -> bool:
        """Decide whether a description deserves an LLM summary pass.

        Triggers when EITHER:

        * the text has grown to ``summarize_at_fragments`` or more
          fragments (cheap way to detect "high-frequency entity with
          many partial mentions"), OR
        * it has passed ``max_description_chars`` (safety net in case
          the per-fragment cap is set loose).
        """
        if not description:
            return False
        fragments = description.split(DESCRIPTION_SEPARATOR)
        if len(fragments) >= self._config.summarize_at_fragments:
            return True
        if len(description) >= self._config.max_description_chars:
            return True
        return False

    async def _summarize_description(
        self,
        *,
        subject_kind: str,
        subject_label: str,
        description: str,
    ) -> Optional[str]:
        """LLM-summarize an accumulated description.

        Returns the summary, or ``None`` if the LLM call failed or the
        service has no LLM wired in. On failure we fall back to the
        character cap with a word-boundary truncation, which is lossy
        but keeps the database bounded. The caller should treat
        ``None`` as "leave the description as-is".
        """
        fragments = [f for f in description.split(DESCRIPTION_SEPARATOR) if f.strip()]
        if not fragments:
            return None

        if self._llm is None:
            return self._fallback_truncate(description)

        prompt = render_summarization_prompt(
            subject_kind=subject_kind,
            subject_label=subject_label,
            fragments=fragments,
            language=self._config.extraction_language,
            target_chars=self._config.summary_target_chars,
        )
        try:
            raw = await self._llm(prompt)
        except Exception:
            logger.exception(
                "graphindex: summarization LLM call failed for %s %s; falling back to truncation",
                subject_kind,
                subject_label,
            )
            return self._fallback_truncate(description)

        summary = (raw or "").strip()
        if not summary:
            return self._fallback_truncate(description)
        return summary

    def _fallback_truncate(self, description: str) -> str:
        """Last-resort hard cap for descriptions. Only used when the
        LLM summarizer is unavailable (``llm=None``) or failed. Keeps
        the database bounded; annotates the end with a marker so
        operators can find capped rows if needed.
        """
        cap = self._config.max_description_chars
        marker = " … [truncated]"
        if len(description) <= cap:
            return description
        # Prefer a word boundary within the last 64 chars of the cap.
        window = description[: cap - len(marker)]
        cut = window.rfind(" ", max(0, len(window) - 64))
        if cut <= 0:
            cut = len(window)
        return window[:cut].rstrip() + marker

    async def _compact_oversized_descriptions(self, *, collection_id: str) -> None:
        """Post-write sweep: find entities / relations whose descriptions
        have grown past the summarization thresholds, run the LLM, and
        write the summary back.

        Runs once per ``index_document`` call. This is not a generic
        background job — it's synchronous on the critical write path so
        that a user viewing the collection right after an index run
        already sees compact descriptions. On a typical document this
        sweep touches zero to a handful of rows; in the worst case
        (every entity overshoots), the cost is bounded by the LLM
        concurrency cap in the indexer.
        """
        entities = await self._store.find_oversized_entities(
            collection_id=collection_id,
            min_chars=self._config.max_description_chars,
            min_fragments=self._config.summarize_at_fragments,
        )
        for e in entities:
            summary = await self._summarize_description(
                subject_kind="entity",
                subject_label=e.name,
                description=e.description,
            )
            if summary and summary != e.description:
                await self._store.rewrite_entity_description(
                    collection_id=collection_id,
                    entity_id=e.entity_id,
                    description=summary,
                )

        relations = await self._store.find_oversized_relations(
            collection_id=collection_id,
            min_chars=self._config.max_description_chars,
            min_fragments=self._config.summarize_at_fragments,
        )
        for r in relations:
            summary = await self._summarize_description(
                subject_kind="relation",
                subject_label=f"{r.source_id} → {r.target_id}",
                description=r.description,
            )
            if summary and summary != r.description:
                await self._store.rewrite_relation_description(
                    collection_id=collection_id,
                    source_id=r.source_id,
                    target_id=r.target_id,
                    description=summary,
                )

    # ---- entity/relation vector sync (private) -----------------------
    async def _sync_entity_relation_vectors(self, *, collection_id: str) -> None:
        """Embed all entities and relations for this collection into the
        vectorstore so ``query_context`` can do semantic recall.

        Runs after every ``index_document`` call. Only operates when
        ``embed_texts`` and ``vector_connector`` are both wired in;
        silently no-ops otherwise (e.g. tests or deployments that don't
        need graph-based semantic recall).

        Uses the existing vectorstore with ``index_type`` payload to
        distinguish graph vectors from chunk vectors — no separate
        tables or connections needed.
        """
        if self._embed_texts is None or self._vector_connector is None:
            return

        from aperag.vectorstore.dto import VectorPoint

        # Fetch all entities for this collection via oversized finder
        # with very generous thresholds (i.e. find ALL entities).
        entities = await self._store.find_oversized_entities(
            collection_id=collection_id,
            min_chars=0,
            min_fragments=0,
            limit=10000,
        )

        if entities:
            texts = [f"{e.name}\n{e.description}" for e in entities]
            try:
                vectors = await self._embed_texts(texts)
            except Exception:
                logger.exception("graphindex: failed to embed entities for vector sync")
                return

            points = []
            for e, vec in zip(entities, vectors):
                points.append(
                    VectorPoint(
                        id=f"ge_{e.entity_id}",
                        vector=vec,
                        payload={
                            "indexer": "graph_entity",
                            "entity_id": e.entity_id,
                            "entity_name": e.name,
                            "entity_type": e.type,
                            "collection_id": collection_id,
                        },
                    )
                )
            try:
                self._vector_connector.upsert(points)
            except Exception:
                logger.exception("graphindex: failed to upsert entity vectors")

        # Fetch relations (reuse oversized finder with generous thresholds)
        relations = await self._store.find_oversized_relations(
            collection_id=collection_id,
            min_chars=0,
            min_fragments=0,
            limit=10000,
        )

        if relations:
            texts = [f"{r.source_id}\t{r.target_id}\n{r.description}" for r in relations]
            try:
                vectors = await self._embed_texts(texts)
            except Exception:
                logger.exception("graphindex: failed to embed relations for vector sync")
                return

            points = []
            for r, vec in zip(relations, vectors):
                points.append(
                    VectorPoint(
                        id=f"gr_{r.source_id}_{r.target_id}",
                        vector=vec,
                        payload={
                            "indexer": "graph_relation",
                            "source_id": r.source_id,
                            "target_id": r.target_id,
                            "collection_id": collection_id,
                        },
                    )
                )
            try:
                self._vector_connector.upsert(points)
            except Exception:
                logger.exception("graphindex: failed to upsert relation vectors")

    # ---- shadow vector identity helpers (private) ---------------------
    async def _snapshot_shadow_identity(self, collection_id: str) -> tuple[set[str], set[tuple[str, str]]]:
        """Return the current entity / relation identity sets that back
        graph shadow vectors.

        When no vector connector is wired, there are no graph shadow
        vectors to clean up, so callers can short-circuit to empty sets.
        """
        if self._vector_connector is None:
            return set(), set()
        entity_ids = await self._collect_entity_ids(collection_id)
        relation_keys = await self._collect_relation_keys(collection_id)
        return entity_ids, relation_keys

    async def _delete_removed_shadow_vectors(
        self,
        *,
        removed_entity_ids: set[str],
        removed_relation_keys: set[tuple[str, str]],
        reason: str,
    ) -> None:
        """Delete graph shadow vectors whose topology rows disappeared."""
        if self._vector_connector is None:
            return

        shadow_ids = [f"ge_{eid}" for eid in removed_entity_ids]
        shadow_ids.extend(f"gr_{src}_{tgt}" for src, tgt in removed_relation_keys)
        if not shadow_ids:
            return

        try:
            self._vector_connector.delete(shadow_ids)
        except Exception:
            logger.exception("graphindex: failed to delete shadow vectors after %s", reason)

    async def _collect_entity_ids(self, collection_id: str) -> set[str]:
        """Return the set of entity_ids currently in the graph store."""
        entities = await self._store.find_oversized_entities(
            collection_id=collection_id, min_chars=0, min_fragments=0, limit=100000
        )
        return {e.entity_id for e in entities}

    async def _collect_relation_keys(self, collection_id: str) -> set[tuple[str, str]]:
        """Return the set of (source_id, target_id) currently in the graph store."""
        relations = await self._store.find_oversized_relations(
            collection_id=collection_id, min_chars=0, min_fragments=0, limit=100000
        )
        return {(r.source_id, r.target_id) for r in relations}

    # ---- anchor resolution (private) ---------------------------------
    async def _resolve_anchor_entities(
        self,
        *,
        collection_id: str,
        query: str,
        top_k: int,
    ) -> List[Entity]:
        """Return up to ``top_k`` anchor entities for the query.

        Three tiers of recall, tried in order of capability:

        1. **Vector recall** (when ``embed_query`` + ``vector_connector``
           are wired): embed the query, search for similar entity and
           relation descriptions in the vectorstore, gather the
           referenced entity ids, then fetch from the graph store. This
           is the LightRAG-equivalent hybrid recall path that searches
           both entities (local) and relations (global) in one shot.
        2. **Name-match fallback**: split the query on whitespace and do
           exact entity-name matching in the graph store. Lower recall,
           but works without any vector infrastructure.
        3. **Empty**: if neither produces results, ``query_context``
           returns an empty ``GraphContext``.
        """
        # Tier 1: vector-based entity + relation recall
        if self._embed_query is not None and self._vector_connector is not None:
            try:
                return await self._vector_recall(collection_id=collection_id, query=query, top_k=top_k)
            except Exception:
                logger.exception("graphindex: vector recall failed; falling back to name match")

        # Tier 2: exact name match
        names = [t for t in (query or "").split() if t.strip()]
        if not names:
            return []
        found = await self._store.find_entities_by_names(collection_id=collection_id, names=names[: max(top_k, 16)])
        return found[:top_k]

    async def _vector_recall(
        self,
        *,
        collection_id: str,
        query: str,
        top_k: int,
    ) -> List[Entity]:
        """Semantic entity recall via the existing vectorstore.

        1. Embed the user query once.
        2. Search entity description vectors (``index_type=graph_entity``).
        3. Search relation description vectors (``index_type=graph_relation``),
           extract the source/target entity ids from matched relations.
        4. Union, deduplicate, fetch full Entity objects from GraphStore.

        This replaces LightRAG's ``entities_vdb.query`` + ``relationships_vdb.query``
        without maintaining a separate vector infrastructure. The same
        vectorstore (Qdrant or pgvector) that holds chunk embeddings also
        holds entity/relation embeddings, differentiated by ``index_type``.
        """
        from aperag.vectorstore.dto import QueryRequest
        from aperag.vectorstore.filters import Eq

        query_vector = await self._embed_query(query)
        connector = self._vector_connector

        entity_ids: list[str] = []

        # Entity recall
        entity_hits = connector.search(
            QueryRequest(
                embedding=query_vector,
                top_k=top_k,
                flt=Eq("indexer", "graph_entity"),
                score_threshold=0.0,
            )
        )
        for hit in entity_hits:
            eid = (hit.payload or {}).get("entity_id")
            if eid:
                entity_ids.append(eid)

        # Relation recall — gather entity ids from both endpoints
        relation_hits = connector.search(
            QueryRequest(
                embedding=query_vector,
                top_k=top_k,
                flt=Eq("indexer", "graph_relation"),
                score_threshold=0.0,
            )
        )
        for hit in relation_hits:
            p = hit.payload or {}
            if p.get("source_id"):
                entity_ids.append(p["source_id"])
            if p.get("target_id"):
                entity_ids.append(p["target_id"])

        # Deduplicate, preserving order
        unique_ids = list(dict.fromkeys(entity_ids))
        if not unique_ids:
            return []

        # Fetch full Entity objects from the graph store by id.
        # This covers BOTH entity-direct hits and relation-sourced hits
        # uniformly — the reviewer correctly flagged that the previous
        # name-only fetch path silently dropped relation-only anchors.
        found = await self._store.find_entities_by_ids(
            collection_id=collection_id,
            entity_ids=unique_ids[: top_k * 2],
        )
        return found[:top_k]


# ---------------------------------------------------------------------------
# Text rendering
# ---------------------------------------------------------------------------


def _render_context_block(
    *,
    entities: List[Entity],
    relations: List[Relation],
    chunks: Optional[List[Chunk]] = None,
) -> str:
    """Pack entities + relations + chunks into a deterministic text block
    for the RAG prompt.

    Three-section format (matching the LightRAG convention so downstream
    prompts that expect structured graph context continue to work):

    - Entities (KG)
    - Relationships (KG)
    - Document Chunks (DC)  — new in v2.2, rehydrated from the graph store
    """
    if not entities and not relations:
        return ""

    id_to_name = {e.entity_id: e.name for e in entities}

    lines: list[str] = []
    if entities:
        lines.append("-----Entities (KG)-----")
        for e in entities:
            desc = e.description.strip() or "(no description)"
            lines.append(f"- [{e.type}] {e.name} — {desc}")
    if relations:
        lines.append("")
        lines.append("-----Relationships (KG)-----")
        for r in relations:
            src = id_to_name.get(r.source_id, r.source_id)
            tgt = id_to_name.get(r.target_id, r.target_id)
            desc = r.description.strip() or "(no description)"
            lines.append(f"- {src} → {tgt}: {desc} (weight={r.weight})")
    if chunks:
        lines.append("")
        lines.append("-----Document Chunks (DC)-----")
        for c in chunks[:20]:
            text = (c.text or "").strip()
            if text:
                lines.append(f"[{c.chunk_id[:12]}] {text[:500]}")

    return "\n".join(lines)


__all__ = ["GraphIndexService", "LLMCall", "EmbedQuery", "EmbedTexts"]
