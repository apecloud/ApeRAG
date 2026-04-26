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

The knowledge-graph recall path consumes its provider through the
``GraphSearchContract`` protocol declared in
``aperag.domains.retrieval.ports``. The retrieval domain must not
import ``aperag.domains.knowledge_graph`` directly — doing so would
re-establish a cross-domain static dependency. Instead the
``_graph_search`` method type-binds to the Protocol; the concrete
graphindex service instance structurally satisfies it at runtime.
``aperag.domains.knowledge_graph.graphindex.*`` stays legal here because it is infrastructure,
not a forbidden aggregate.
"""

from __future__ import annotations

import asyncio
import logging
from functools import partial
from typing import Any, List, Optional, Protocol, Tuple

from aperag.config import build_vector_db_context, settings
from aperag.db.ops import async_db_ops
from aperag.domains.retrieval.context.context import ContextManager
from aperag.domains.retrieval.ports import GraphSearchContract
from aperag.domains.retrieval.schemas import SearchRequest, SearchResultItem, SearchResultMetadata
from aperag.exceptions import ValidationException
from aperag.indexing.keyword_extract import extract_keywords
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
from aperag.utils.utils import generate_fulltext_index_name, generate_vector_db_collection_name

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


def _graph_search_service_for(collection: CollectionRow) -> GraphSearchContract:
    """Build the graph-search provider for a collection.

    The import is kept local so the pipeline module does not pay the
    graphindex import tax unless a graph recall is actually requested.
    The return type is annotated as the Protocol so the boundary is
    explicit: the caller only sees ``query_context`` and nothing else.
    """
    from aperag.domains.knowledge_graph.graphindex.integration import make_service_for_collection

    return make_service_for_collection(collection)  # type: ignore[return-value]


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
        from aperag.domains.indexing.fulltext_index import FulltextSearchDegradedError, fulltext_indexer

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

        try:
            docs = await fulltext_indexer.search_document(
                index_name,
                str(collection.id),
                final_keywords,
                top_k * 3,
                chat_id=chat_id,
            )
        except FulltextSearchDegradedError as e:
            logger.warning("Fulltext search degraded for collection %s: %s", collection.id, e)
            return []

        for doc in docs:
            if doc.metadata is None:
                doc.metadata = {}
            doc.metadata["recall_type"] = "fulltext_search"
        return docs

    async def _graph_search(
        self,
        collection: CollectionRow,
        query: str,
        top_k: int,
    ) -> List[DocumentWithScore]:
        """Knowledge-graph retrieval path. Always routes to graphindex v2
        through the ``GraphSearchContract`` protocol.

        A collection that hasn't been indexed yet returns no context;
        this is the correct behaviour — search pipelines compose
        (vector + graph + fulltext), and a blank graph just means
        "graph contributes nothing this time", not "fall back to
        something stale".
        """
        config = parseCollectionConfig(collection.config)
        if not config.enable_knowledge_graph:
            logger.warning(f"Collection {collection.id} does not have knowledge graph enabled")
            return []

        svc: GraphSearchContract = _graph_search_service_for(collection)
        ctx = await svc.query_context(collection_id=str(collection.id), query=query, top_k=top_k)
        if not ctx.text:
            return []
        return [DocumentWithScore(text=ctx.text, metadata={"recall_type": "graph_search"})]

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
