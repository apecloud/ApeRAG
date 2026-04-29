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
both into the graph-RAG context block the retrieval pipeline
already consumes (so task #8's wiring is a one-line swap).

Per-collection bound — caller wires the four dependencies (lineage
store, vector connector, embedder, optional LLM) to the collection
being searched before constructing the service. The constructor does
no I/O. ``llm`` is reserved for follow-up high-level / low-level
keyword extraction (hybrid keyword recall); v1 PR does not invoke it.
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


# Vector point indexer tags — pinned by §K.12 invariant #5. Task #3
# writes graph-entity vectors with ``payload["indexer"] ==
# "graph_entity"`` and graph-relation vectors with
# ``payload["indexer"] == "graph_relation"``; this service filters on
# the matching tag so chunk / summary vectors sharing the physical
# collection never bleed into graph recall.
GRAPH_ENTITY_INDEXER: str = "graph_entity"
GRAPH_RELATION_INDEXER: str = "graph_relation"


# ``entity_name`` payload field for ``graph_relation`` points carries the
# composite ``f"{source}->{target}"`` per the task #3 writer
# (``aperag/indexing/graph.py:1631``). The arrow has no surrounding
# whitespace; we split exactly once to recover the endpoints.
RELATION_PAYLOAD_ARROW: str = "->"


class GraphSearchService:
    """Vector + graph recall for a single collection.

    Public surface (per §K.12.5 spec, locked by architect ratify
    msg=acbd0003):

    * :meth:`search_entities` — embed the query, ANN-search the entity
      vector index, fetch the matching ``EntityWithLineage`` rows.
    * :meth:`search_relations` — embed the query, ANN-search the
      relation vector index, fetch the matching ``RelationWithLineage``
      rows. Wave 8 W8-1 (task #12) replaced the Wave 7 conservative
      1-hop expansion with direct vector recall now that task #3 is
      writing relation vectors.
    * :meth:`get_subgraph` — wrap ``expand_neighbors_n_hops`` as a
      stable read primitive for MCP tools (task #7).
    * :meth:`compose_context` — render the ``EntityWithLineage`` /
      ``RelationWithLineage`` lists into the canonical graph-RAG
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
        # Reserved for follow-up: hybrid high/low-level keyword
        # extraction from the user query. Not invoked in v1.
        llm: Callable[[str], Awaitable[str]] | None = None,
        top_k: int = DEFAULT_TOP_K,
        score_threshold: float = DEFAULT_SCORE_THRESHOLD,
        # 任务 #5 step 8: alias 解析层. 用户搜的字串可能是某个 entity 的
        # alias (例如 "Foo" → canonical "Foo Inc"). 写路径已经把 alias
        # 重定向到 canonical, 所以 store 里只有 canonical entity 名字;
        # 读路径需要 alias_repo.resolve_canonical 来兜底精确匹配 miss
        # 的情况. 设计文档 §5 三层检索的"别名+模糊"层.
        alias_repo: Any = None,
        collection_id: str = "",
    ) -> None:
        self._store = store
        self._vector_connector = vector_connector
        self._embedder = embedder
        self._llm = llm
        self._top_k = top_k
        self._score_threshold = score_threshold
        self._alias_repo = alias_repo
        self._collection_id = collection_id

    # ------------------------------------------------------------------
    # search_entities — vector-first path with name-driven fallback
    # ------------------------------------------------------------------

    async def search_entities(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[EntityWithLineage]:
        """Recall the top-K entities matching ``query``.

        Prefer entity-vector recall when the graph vector layer is
        available. Exact / alias / lexical lookup remains in the path
        as ranking guardrail and fallback:

        * alias is resolved before vector recall so alias queries can
          use the canonical entity name for embedding and exact boost;
        * exact entity hits are hard-boosted to the front so a literal
          entity-name query remains predictable;
        * keyword/fuzzy lookup only supplements or handles vector
          failures.

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

        canonical = await self._safe_resolve_alias(text)
        query_for_embedding = canonical if canonical and canonical != text else text

        exact_candidates: list[EntityWithLineage] = []
        exact_seen: set[str] = set()
        for candidate in (text, query_for_embedding):
            if not candidate or candidate in exact_seen:
                continue
            exact_seen.add(candidate)
            exact = await self._safe_get_exact_entity(candidate)
            if exact is not None and exact.name not in {entity.name for entity in exact_candidates}:
                exact_candidates.append(exact)

        vector_entities: list[EntityWithLineage] = []
        try:
            embedding = await asyncio.to_thread(self._embedder.embed_query, query_for_embedding)
        except Exception:
            logger.warning("graph_search_service: embed failed for query=%r", text, exc_info=True)
            embedding = None

        if embedding is not None:
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
                hits = []

            names = self._entity_names_from_hits(hits)
            if names:
                vector_entities = await self._batch_get_entities(names)

        results: list[EntityWithLineage] = []
        seen_names: set[str] = set()

        def _append(entity: EntityWithLineage) -> bool:
            if entity.name in seen_names:
                return len(results) >= k
            results.append(entity)
            seen_names.add(entity.name)
            return len(results) >= k

        for entity in exact_candidates:
            if _append(entity):
                return results

        for entity in vector_entities:
            if _append(entity):
                return results

        lexical_matches = await self._safe_query_entities_by_keyword(text, max(k - len(results), 0))
        for entity in lexical_matches:
            if _append(entity):
                return results
        return results

    async def _safe_get_exact_entity(self, text: str) -> EntityWithLineage | None:
        try:
            return await self._store.get_entity(text)
        except Exception:
            logger.warning("graph_search_service: exact entity lookup failed for query=%r", text, exc_info=True)
            return None

    async def _safe_query_entities_by_keyword(self, text: str, top_k: int) -> list[EntityWithLineage]:
        if top_k <= 0:
            return []
        query = getattr(self._store, "query_entities_by_keyword", None)
        if query is None:
            return []
        try:
            return await query(query=text, top_k=top_k)
        except Exception:
            logger.warning("graph_search_service: lexical entity lookup failed for query=%r", text, exc_info=True)
            return []

    async def _safe_resolve_alias(self, text: str) -> str:
        """走 alias map 把 alias 翻译成 canonical entity 名字.

        ``alias_repo`` / ``collection_id`` 缺省时直接返回原字符串
        (向后兼容: GraphSearchService 不强制要求 alias 注入). 任何 alias
        查询异常按 best-effort 降级 — 兜底层失败应该静默回退到 lexical /
        vector, 不让别名缺失阻断整个搜索.
        """
        if self._alias_repo is None or not self._collection_id:
            return text
        try:
            canonical = await self._alias_repo.resolve_canonical(
                collection_id=self._collection_id,
                name=text,
            )
        except Exception:
            logger.warning(
                "graph_search_service: alias lookup failed for query=%r",
                text,
                exc_info=True,
            )
            return text
        return canonical or text

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
    # search_relations — vector recall path (Wave 8 W8-1 task #12)
    # ------------------------------------------------------------------

    async def search_relations(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[RelationWithLineage]:
        """Vector-recall the top-K relations matching ``query``.

        Wave 8 W8-1 (architect ratify task #12): replaces the Wave 7
        conservative 1-hop expansion with a direct ANN search filtered
        on ``Eq("indexer", "graph_relation")`` — task #3 (PR #1757) has
        been writing relation vectors all along; this is finally the
        consumer side. Mirrors :meth:`search_entities` shape for
        symmetry.

        Hits resolve to full :class:`RelationWithLineage` rows via
        per-hit ``store.get_relation`` reverse lookup so
        :meth:`compose_context` keeps its byte-parity with the legacy
        rendering (per task #5 invariant). Hits whose payload doesn't
        parse cleanly (missing arrow / empty endpoints / no
        ``entity_type``) are dropped silently — better to skip a hit
        than to surface a relation we can't reconstruct.
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
            logger.warning(
                "graph_search_service: embed failed for relation query=%r",
                text,
                exc_info=True,
            )
            return []

        request = QueryRequest(
            embedding=embedding,
            top_k=k,
            flt=Eq("indexer", GRAPH_RELATION_INDEXER),
            score_threshold=self._score_threshold or None,
        )
        try:
            hits = await asyncio.to_thread(self._vector_connector.search, request)
        except Exception:
            logger.warning(
                "graph_search_service: vector search failed for relation query=%r",
                text,
                exc_info=True,
            )
            return []

        keys = self._relation_keys_from_hits(hits)
        if not keys:
            return []
        return await self._batch_get_relations(keys)

    @staticmethod
    def _relation_keys_from_hits(
        hits: Iterable[SearchHit],
    ) -> list[tuple[str, str, str]]:
        # Preserve hit order (vector store ranked by score) and de-dup
        # so the same edge returned twice (e.g. alias-redirect side-effect)
        # only fetches once.
        seen: set[tuple[str, str, str]] = set()
        ordered: list[tuple[str, str, str]] = []
        for hit in hits:
            payload = hit.payload or {}
            composite = payload.get("entity_name") or payload.get("name")
            relation_type = payload.get("entity_type") or payload.get("relation_type")
            if not composite or not relation_type:
                continue
            composite_str = str(composite)
            if RELATION_PAYLOAD_ARROW not in composite_str:
                continue
            source, _, target = composite_str.partition(RELATION_PAYLOAD_ARROW)
            if not source or not target:
                continue
            key = (source, target, str(relation_type))
            if key in seen:
                continue
            seen.add(key)
            ordered.append(key)
        return ordered

    async def _batch_get_relations(self, keys: Sequence[tuple[str, str, str]]) -> list[RelationWithLineage]:
        # Same ``asyncio.gather`` shape as :meth:`_batch_get_entities`:
        # N is bounded by ``top_k`` (≤ 10 in practice) so per-key
        # round-trip cost is acceptable, and the Protocol surface stays
        # narrow (no batch-get method).
        results = await asyncio.gather(
            *(self._store.get_relation(s, t, ty) for s, t, ty in keys),
            return_exceptions=False,
        )
        return [r for r in results if r is not None]

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
        """Render entities + relations into the graph-RAG context block
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
    from aperag.graph_curation.alias_map import AliasMapRepository
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
        alias_repo=AliasMapRepository(),
        collection_id=str(collection.id),
    )


__all__ = [
    "GraphSearchService",
    "DEFAULT_TOP_K",
    "DEFAULT_SCORE_THRESHOLD",
    "GRAPH_ENTITY_INDEXER",
    "GRAPH_RELATION_INDEXER",
    "RELATION_PAYLOAD_ARROW",
    "build_graph_search_service_for",
]
