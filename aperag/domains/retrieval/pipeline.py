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

"""Direct Python orchestration for collection / chat search.

Relocated from ``aperag/service/search_pipeline_service.py`` by the
Phase 2 hard-cut. Preserves the existing pipeline shape and async
semantics byte-for-byte — no algorithmic change (Non-goal 2).

Wave 6 #33 chunk 3 (per architect Option C ruling msg=6fccd9ab):
the knowledge-graph recall path now routes through the new
:class:`aperag.indexing.graph.LineageGraphStore` Protocol
(``query_entities_by_keyword`` + ``expand_neighbors_n_hops``)
instead of the legacy ``GraphIndexService.query_context``. The
retrieval pipeline composes context text from
``EntityWithLineage`` / ``RelationWithLineage`` directly — see
:func:`_render_graph_context_text`. The legacy
``aperag.domains.knowledge_graph.graphindex`` package is retained
in the codebase because its UI / curation methods (``get_labels`` /
``get_knowledge_graph`` / ``merge_entities``) still serve other
flows (knowledge_graph/service.py + graph_curation/*); the retrieval
side has no remaining import of it.
"""

from __future__ import annotations

import asyncio
import logging
from functools import partial
from typing import Any, List, Optional, Protocol, Tuple

from aperag.config import build_vector_db_context, settings
from aperag.db.ops import async_db_ops
from aperag.domains.retrieval.context.context import ContextManager
from aperag.domains.retrieval.schemas import SearchRequest, SearchResultItem, SearchResultMetadata
from aperag.exceptions import ValidationException
from aperag.llm.embed.base_embedding import get_collection_embedding_service_sync
from aperag.llm.llm_error_types import (
    EmbeddingError,
    InvalidConfigurationError,
    ProviderNotFoundError,
    RerankError,
)
from aperag.observability import start_span
from aperag.platform.query.query import DocumentWithScore
from aperag.schema.utils import parseCollectionConfig
from aperag.utils.utils import generate_vector_db_collection_name

logger = logging.getLogger(__name__)


class CollectionRow(Protocol):
    """Structural view of the legacy ``aperag.db.models.Collection``
    that the retrieval pipeline is allowed to read.

    Same shape / rationale as
    ``aperag.domains.knowledge_graph.ports.CollectionRow`` — each
    domain declares its own narrow port so neither domain binds
    transitively to the other through a shared Protocol module. The
    pipeline reads only ``id``, ``user``, and ``config`` (the raw JSON
    blob parsed via ``parseCollectionConfig``).
    """

    id: str
    user: str
    config: Any


def _render_graph_context_text(entities: list[Any], relations: list[Any]) -> str:
    """Compose a LightRAG-style text block from entities + relations.

    Mirrors the legacy ``GraphIndexService.query_context`` rendering
    convention so downstream RAG prompts that expect the
    ``-----Entities (KG)----- / -----Relationships (KG)-----``
    sectioned text continue to work after the Wave 6 #33 chunk 3
    migration.

    ``EntityWithLineage`` / ``RelationWithLineage`` carry their
    description across multiple ``DescriptionPart`` rows (one per
    contributing parse). We join the non-empty fragments with `` | ``
    so the rendered context surfaces every contribution; an entity
    with no descriptions falls back to ``"(no description)"``.
    """
    if not entities and not relations:
        return ""
    lines: list[str] = []
    if entities:
        lines.append("-----Entities (KG)-----")
        for e in entities:
            desc_parts = [p.text.strip() for p in (e.description_parts or ()) if p.text and p.text.strip()]
            desc = " | ".join(desc_parts) or "(no description)"
            type_label = e.entity_type or "entity"
            lines.append(f"- [{type_label}] {e.name} — {desc}")
    if relations:
        if lines:
            lines.append("")
        lines.append("-----Relationships (KG)-----")
        for r in relations:
            desc_parts = [p.text.strip() for p in (r.description_parts or ()) if p.text and p.text.strip()]
            desc = " | ".join(desc_parts) or "(no description)"
            lines.append(f"- {r.source} → {r.target}: {desc}")
    return "\n".join(lines)


def _build_lineage_graph_store_for(collection: CollectionRow) -> Any:
    """Build a :class:`LineageGraphStore` for ``collection`` using the
    same backend dispatch the indexing worker_factory uses. The import
    is kept local so the pipeline module does not pay the indexing
    import tax unless a graph recall is actually requested.

    Wave 6 #33 chunk 3 (per architect Option C ruling msg=6fccd9ab):
    retrieval-side LightRAG-style query routes through the new
    LineageGraphStore Protocol (`query_entities_by_keyword` +
    `expand_neighbors_n_hops`) instead of the legacy
    ``GraphIndexService.query_context``.
    """
    from aperag.indexing.worker_factory import (
        _build_lineage_graph_store,
        _resolve_graph_backend_type,
    )

    backend_type = _resolve_graph_backend_type(collection)
    return _build_lineage_graph_store(backend_type=backend_type, collection=collection)


def _deduplicate_vision_results(results: List[DocumentWithScore]) -> List[DocumentWithScore]:
    """Prefer vision_to_text hits when both image index variants return the same asset."""
    vision_to_text_keys = set()
    for doc in results:
        metadata = doc.metadata or {}
        if (
            metadata.get("indexer") == "vision"
            and metadata.get("index_method") == "vision_to_text"
            and metadata.get("collection_id") is not None
            and metadata.get("document_id") is not None
            and metadata.get("asset_id") is not None
        ):
            key = (
                metadata["collection_id"],
                metadata["document_id"],
                metadata["asset_id"],
            )
            vision_to_text_keys.add(key)

    if not vision_to_text_keys:
        return results

    deduplicated_results = []
    for doc in results:
        metadata = doc.metadata or {}
        if (
            metadata.get("indexer") == "vision"
            and metadata.get("index_method") != "vision_to_text"
            and metadata.get("collection_id") is not None
            and metadata.get("document_id") is not None
            and metadata.get("asset_id") is not None
        ):
            key = (
                metadata["collection_id"],
                metadata["document_id"],
                metadata["asset_id"],
            )
            if key in vision_to_text_keys:
                logger.info(f"Removing duplicate vision document for asset {key[2]} from document {key[1]}")
                continue
        deduplicated_results.append(doc)

    return deduplicated_results


class SearchPipelineService:
    """Direct Python orchestration for collection and chat search."""

    async def execute_search(
        self,
        data: SearchRequest,
        collection_id: str,
        search_user_id: str,
        chat_id: Optional[str] = None,
    ) -> Tuple[List[SearchResultItem], str]:
        with start_span(
            "retrieval.search",
            tracer_name=__name__,
            **{
                "aperag.domain": "retrieval",
                "aperag.operation": "retrieval.search",
                "aperag.collection.id": collection_id,
                "aperag.user.id": search_user_id,
                "aperag.chat.id": chat_id,
            },
        ):
            query = (data.query or "").strip()
            if not query:
                raise ValidationException("query is required")

            recall_tasks = []
            collection = await async_db_ops.query_collection(search_user_id, collection_id)
            if not collection:
                raise ValidationException(f"collection not found: {collection_id}")

            if data.vector_search:
                recall_tasks.append(
                    self._vector_search(
                        collection=collection,
                        query=query,
                        top_k=data.vector_search.topk,
                        similarity_threshold=data.vector_search.similarity,
                        chat_id=chat_id,
                    )
                )
            if data.fulltext_search:
                recall_tasks.append(
                    self._fulltext_search(
                        collection=collection,
                        query=query,
                        top_k=data.fulltext_search.topk,
                        keywords=data.fulltext_search.keywords,
                        user_id=search_user_id,
                        chat_id=chat_id,
                    )
                )
            if data.graph_search:
                recall_tasks.append(
                    self._graph_search(
                        collection=collection,
                        query=query,
                        top_k=data.graph_search.topk,
                    )
                )
            if data.summary_search:
                recall_tasks.append(
                    self._summary_search(
                        collection=collection,
                        query=query,
                        top_k=data.summary_search.topk,
                        similarity_threshold=data.summary_search.similarity,
                    )
                )
            if data.vision_search:
                recall_tasks.append(
                    self._vision_search(
                        collection=collection,
                        query=query,
                        top_k=data.vision_search.topk,
                        similarity_threshold=data.vision_search.similarity,
                    )
                )

            if not recall_tasks:
                raise ValidationException("At least one search strategy must be enabled")

            recall_results = await asyncio.gather(*recall_tasks)
            merged_docs = self._merge_results(recall_results)
            reranked_docs = await self._rerank(
                query=query,
                docs=merged_docs,
                user_id=search_user_id,
                use_rerank=bool(data.rerank),
            )

            items = []
            for idx, doc in enumerate(reranked_docs):
                metadata = doc.metadata or {}
                public_metadata = SearchResultMetadata.from_raw(metadata)
                source = public_metadata.source if public_metadata and public_metadata.source else ""
                items.append(
                    SearchResultItem(
                        rank=idx + 1,
                        score=doc.score,
                        content=doc.text,
                        source=source,
                        recall_type=metadata.get("recall_type", ""),
                        metadata=public_metadata,
                    )
                )

            return items, "rerank"

    async def _vector_search(
        self,
        collection: CollectionRow,
        query: str,
        top_k: int,
        similarity_threshold: float,
        chat_id: Optional[str] = None,
    ) -> List[DocumentWithScore]:
        try:
            collection_name = generate_vector_db_collection_name(collection.id)
            embedding_model, vector_size = get_collection_embedding_service_sync(collection)
            vectordb_ctx = build_vector_db_context(collection_name, vector_size=vector_size)
            context_manager = ContextManager(collection_name, embedding_model, settings.vector_db_type, vectordb_ctx)

            vector = await asyncio.to_thread(embedding_model.embed_query, query)
            query_fn = partial(
                context_manager.query,
                query,
                score_threshold=similarity_threshold,
                topk=top_k,
                vector=vector,
                index_types=["vector"],
                chat_id=chat_id,
            )
            results = await asyncio.to_thread(query_fn)
            for item in results:
                if item.metadata is None:
                    item.metadata = {}
                item.metadata["recall_type"] = "vector_search"
            return results
        except ProviderNotFoundError as e:
            logger.warning(f"Vector search skipped for collection {collection.id} due to provider not found: {str(e)}")
            return []
        except EmbeddingError as e:
            logger.warning(f"Vector search skipped for collection {collection.id} due to embedding error: {str(e)}")
            return []
        except Exception as e:
            logger.error(f"Vector search failed for collection {collection.id}: {str(e)}")
            return []

    async def _fulltext_search(
        self,
        collection: CollectionRow,
        query: str,
        top_k: int,
        keywords: Optional[List[str]],
        user_id: str,
        chat_id: Optional[str] = None,
    ) -> List[DocumentWithScore]:
        """Fulltext recall (Wave 3 T3.1 chunk 3 — inline ES query).

        Wave 3 hard-cut deleted the legacy
        ``aperag/domains/indexing/fulltext_index.py:FulltextIndexer.
        search_document``; this method now talks to Elasticsearch
        directly through the same query shape (the retrieval-side
        query is stateless against whatever
        ``aperag.indexing.fulltext.FulltextModality.sync()`` wrote).
        T3.2 search-lane work (Bryce) is purely additive on
        ``SearchResultMetadata`` and does not introduce a new search
        backend abstraction; the inline query is the canonical path.
        """
        from elasticsearch import AsyncElasticsearch

        from aperag.indexing.keyword_extract import extract_keywords
        from aperag.utils.utils import generate_fulltext_index_name

        config = parseCollectionConfig(collection.config)
        if config.enable_fulltext is False:
            logger.info("Skipping fulltext search for collection %s because enable_fulltext=false", collection.id)
            return []

        index_name = generate_fulltext_index_name(collection.id)
        final_keywords = list(keywords or [])
        if not final_keywords:
            extractor_ctx = {
                "index_name": index_name,
                "es_host": settings.es_host,
                "es_timeout": settings.es_timeout,
                "es_max_retries": settings.es_max_retries,
                "user_id": user_id,
            }
            final_keywords = await extract_keywords(query, extractor_ctx)

        final_keywords = list(set(final_keywords))
        if not final_keywords:
            logger.warning(
                "Fulltext keyword extraction degraded for collection %s; falling back to raw query token",
                collection.id,
            )
            final_keywords = [query]

        es_config = {
            "request_timeout": settings.es_timeout,
            "max_retries": settings.es_max_retries,
            "retry_on_timeout": True,
        }
        async_es = AsyncElasticsearch(settings.es_host, **es_config)

        try:
            exists = await async_es.indices.exists(index=index_name)
            if not exists.body:
                return []

            es_query = {
                "bool": {
                    "should": [{"match": {"content": kw}} for kw in final_keywords]
                    + [{"match": {"title": kw}} for kw in final_keywords],
                    "minimum_should_match": "80%",
                    "filter": [{"term": {"collection_id": str(collection.id)}}],
                }
            }
            if chat_id:
                es_query["bool"]["filter"].append(
                    {
                        "bool": {
                            "should": [
                                {"term": {"chat_id": str(chat_id)}},
                                {"term": {"metadata.chat_id": str(chat_id)}},
                            ],
                            "minimum_should_match": 1,
                        }
                    }
                )

            resp = await async_es.search(
                index=index_name,
                query=es_query,
                sort=[{"_score": {"order": "desc"}}],
                size=top_k * 3,
                routing=str(collection.id),
            )
            hits = resp.body["hits"]["hits"]
        except Exception as exc:
            logger.warning("Fulltext search degraded for collection %s: %s", collection.id, exc)
            try:
                await async_es.close()
            except Exception:  # noqa: BLE001
                pass
            return []

        try:
            await async_es.close()
        except Exception:  # noqa: BLE001
            pass

        results: List[DocumentWithScore] = []
        for hit in hits:
            source = hit.get("_source", {})
            metadata = {
                "source": source.get("name", ""),
                "document_id": source.get("document_id"),
                "chunk_id": source.get("chunk_id"),
                "recall_type": "fulltext_search",
            }
            if source.get("title"):
                metadata["title"] = source["title"]
            if source.get("metadata"):
                metadata.update(source["metadata"])
                metadata["recall_type"] = "fulltext_search"
            results.append(
                DocumentWithScore(
                    text=source.get("content", ""),
                    score=hit.get("_score", 0.0),
                    metadata=metadata,
                )
            )
        return results

    async def _graph_search(
        self,
        collection: CollectionRow,
        query: str,
        top_k: int,
    ) -> List[DocumentWithScore]:
        """Knowledge-graph retrieval path via :class:`GraphSearchService`.

        Wave 7 §K.12.5 / §K.12.8 (task #8) cutover: replaces the
        keyword-only Wave 6 #33 chunk 3 path with the full
        vector + 1-hop traversal composition the legacy
        ``GraphIndexService.query_context`` always intended. Steps:

        1. :meth:`GraphSearchService.search_entities` — embed the query
           against the per-collection vector index and ANN-recall the
           top-K entities (semantic match, not exact name match).
        2. :meth:`GraphSearchService.get_subgraph` — pull the direct
           neighbours + connecting relations of the anchor entities.
        3. :meth:`GraphSearchService.compose_context` — render the
           ``-----Entities (KG)----- / -----Relationships (KG)-----``
           text block.

        The render is byte-for-byte identical to Wave 6's
        ``_render_graph_context_text`` (locked by
        ``test_compose_context_matches_retrieval_pipeline_render_byte_for_byte``
        in PR #1756) so the swap is zero-functional-change for
        downstream RAG prompts: same context shape, same dedup rule,
        same fallback marker. Vector recall now actually happens
        — the Wave 4 → Wave 6 vacuum noted in §K.12.1 is closed.

        A collection that hasn't been indexed yet (no vector points)
        returns ``[]`` — ``search_entities`` swallows backend
        embed/search faults and returns an empty list, mirroring the
        Wave 6 graceful-degrade convention. ``enable_knowledge_graph``
        gating preserved for backward compat.
        """
        config = parseCollectionConfig(collection.config)
        if not config.enable_knowledge_graph:
            logger.warning(f"Collection {collection.id} does not have knowledge graph enabled")
            return []

        from aperag.indexing.graph_search_service import (
            GraphSearchService,
            build_graph_search_service_for,
        )

        try:
            service = build_graph_search_service_for(collection)
        except Exception:
            logger.warning(
                "graph_search: factory failed for collection %s; degrading to empty result",
                collection.id,
                exc_info=True,
            )
            return []

        anchors = await service.search_entities(query=query, top_k=top_k)
        if not anchors:
            return []
        anchor_names = [e.name for e in anchors]
        entities, relations = await service.get_subgraph(entity_names=anchor_names, hops=1)
        text = GraphSearchService.compose_context(entities, relations)
        if not text:
            return []
        return [DocumentWithScore(text=text, metadata={"recall_type": "graph_search"})]

    async def _summary_search(
        self,
        collection: CollectionRow,
        query: str,
        top_k: int,
        similarity_threshold: float,
    ) -> List[DocumentWithScore]:
        try:
            collection_name = generate_vector_db_collection_name(collection.id)
            embedding_model, vector_size = get_collection_embedding_service_sync(collection)
            vectordb_ctx = build_vector_db_context(collection_name, vector_size=vector_size)
            context_manager = ContextManager(collection_name, embedding_model, settings.vector_db_type, vectordb_ctx)

            vector = await asyncio.to_thread(embedding_model.embed_query, query)
            query_fn = partial(
                context_manager.query,
                query,
                score_threshold=similarity_threshold,
                topk=top_k,
                vector=vector,
                index_types=["summary"],
            )
            results = await asyncio.to_thread(query_fn)
            for item in results:
                if item.metadata is None:
                    item.metadata = {}
                item.metadata["recall_type"] = "summary_search"
            return results
        except ProviderNotFoundError as e:
            logger.warning(f"Summary search skipped for collection {collection.id} due to provider not found: {str(e)}")
            return []
        except EmbeddingError as e:
            logger.warning(f"Summary search skipped for collection {collection.id} due to embedding error: {str(e)}")
            return []
        except Exception as e:
            logger.error(f"Summary search failed for collection {collection.id}: {str(e)}")
            return []

    async def _vision_search(
        self,
        collection: CollectionRow,
        query: str,
        top_k: int,
        similarity_threshold: float,
    ) -> List[DocumentWithScore]:
        try:
            collection_name = generate_vector_db_collection_name(collection.id)
            embedding_model, vector_size = get_collection_embedding_service_sync(collection)
            vectordb_ctx = build_vector_db_context(collection_name, vector_size=vector_size)
            context_manager = ContextManager(collection_name, embedding_model, settings.vector_db_type, vectordb_ctx)

            vector = await asyncio.to_thread(embedding_model.embed_query, query)
            expanded_top_k = top_k * 2
            query_fn = partial(
                context_manager.query,
                query,
                score_threshold=similarity_threshold,
                topk=expanded_top_k,
                vector=vector,
                index_types=["vision"],
            )
            results = await asyncio.to_thread(query_fn)
            for item in results:
                if item.metadata is None:
                    item.metadata = {}
                item.metadata["recall_type"] = "vision_search"
            results = _deduplicate_vision_results(results)
            return results[:expanded_top_k]
        except ProviderNotFoundError as e:
            logger.warning(f"Vision search skipped for collection {collection.id} due to provider not found: {str(e)}")
            return []
        except EmbeddingError as e:
            logger.warning(f"Vision search skipped for collection {collection.id} due to embedding error: {str(e)}")
            return []
        except Exception as e:
            logger.error(f"Vision search failed for collection {collection.id}: {str(e)}")
            return []

    def _merge_results(self, result_sets: List[List[DocumentWithScore]]) -> List[DocumentWithScore]:
        all_docs = []
        for docs in result_sets:
            all_docs.extend(docs)

        seen = set()
        unique_docs = []
        for doc in all_docs:
            key = doc.text or ""
            if key in seen:
                continue
            seen.add(key)
            unique_docs.append(doc)
        return unique_docs

    async def _rerank(
        self,
        query: str,
        docs: List[DocumentWithScore],
        user_id: str,
        use_rerank: bool,
    ) -> List[DocumentWithScore]:
        if not docs:
            return []

        if not use_rerank:
            return self._apply_fallback_strategy(docs)

        model_id = await self._resolve_default_rerank_model_id(user_id)
        if not model_id:
            return self._apply_fallback_strategy(docs)

        try:
            from aperag.llm.runtime.invocation_service import model_invocation_service

            ranked = await model_invocation_service.rerank(model_id, user_id, query, [doc.text for doc in docs])
            reranked_docs = []
            for item in ranked:
                idx = item["index"]
                if 0 <= idx < len(docs):
                    reranked_docs.append(
                        DocumentWithScore(
                            text=docs[idx].text,
                            score=float(item.get("relevance_score", item.get("score", 0.0))),
                            metadata=docs[idx].metadata,
                        )
                    )
            return reranked_docs
        except (InvalidConfigurationError, ProviderNotFoundError, RerankError) as e:
            logger.warning(f"Rerank configuration/runtime issue, using fallback strategy: {str(e)}")
            return self._apply_fallback_strategy(docs)
        except Exception as e:
            logger.error(f"Unexpected rerank failure, using fallback strategy: {str(e)}")
            return self._apply_fallback_strategy(docs)

    async def _resolve_default_rerank_model_id(self, user_id: str) -> Optional[str]:
        """Resolve the user's configured retrieval rerank model id."""
        try:
            model_uses = await async_db_ops.query_model_uses(user_id)
        except Exception:
            logger.warning("retrieval: default rerank lookup failed; using fallback strategy", exc_info=True)
            return None

        for item in model_uses or []:
            if item.scenario == "retrieval_rerank" and item.primary_model_id:
                return item.primary_model_id
        return None

    def _apply_fallback_strategy(self, docs: List[DocumentWithScore]) -> List[DocumentWithScore]:
        graph_results = []
        other_results = []

        for doc in docs:
            metadata = doc.metadata or {}
            if metadata.get("recall_type", "") == "graph_search":
                graph_results.append(doc)
            else:
                other_results.append(doc)

        other_results.sort(key=lambda x: x.score if x.score is not None else 0.0, reverse=True)
        return graph_results + other_results


search_pipeline_service = SearchPipelineService()


__all__ = ["SearchPipelineService", "search_pipeline_service"]
