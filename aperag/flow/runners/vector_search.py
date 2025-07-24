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

import json
import logging
from typing import List, Optional, Tuple

from pydantic import BaseModel, Field

from aperag.config import settings
from aperag.context.context import ContextManager
from aperag.db.models import Collection
from aperag.db.ops import async_db_ops
from aperag.flow.base.models import BaseNodeRunner, SystemInput, register_node_runner
from aperag.llm.embed.base_embedding import get_collection_embedding_service_sync
from aperag.llm.llm_error_types import (
    EmbeddingError,
    ProviderNotFoundError,
)
from aperag.query.query import DocumentWithScore
from aperag.utils.utils import generate_vector_db_collection_name

logger = logging.getLogger(__name__)


def _deduplicate_vision_results(results: List[DocumentWithScore]) -> List[DocumentWithScore]:
    """
    Deduplicates vision search results while preserving the original order.

    If both a `vision_to_text` and a `multimodal_embedding` result exist for the same asset,
    this function removes the `multimodal_embedding` result, keeping the `vision_to_text` one.
    The relative order of the kept documents is maintained because the original list is
    iterated over sequentially to build the new, deduplicated list.
    """
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

    # If no vision-to-text results, no deduplication is needed
    if not vision_to_text_keys:
        return results

    deduplicated_results = []
    for doc in results:
        metadata = doc.metadata or {}
        # Check if the current doc is a candidate for removal
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
            # If its key matches a vision-to-text result, skip it
            if key in vision_to_text_keys:
                logger.info(f"Removing duplicate vision document for asset {key[2]} from document {key[1]}")
                continue
        deduplicated_results.append(doc)

    return deduplicated_results


# User input model for vector search node
class VectorSearchInput(BaseModel):
    top_k: int = Field(5, description="Number of top results to return")
    similarity_threshold: float = Field(0.2, description="Similarity threshold for vector search")
    collection_ids: Optional[List[str]] = Field(default_factory=list, description="Collection IDs")


# User output model for vector search node
class VectorSearchOutput(BaseModel):
    docs: List[DocumentWithScore]


# Database operations interface
class VectorSearchRepository:
    """Repository interface for vector search database operations"""

    async def get_collection(self, user, collection_id: str) -> Optional[Collection]:
        """Get collection by ID for the user"""
        return await async_db_ops.query_collection(user, collection_id)


# Business logic service
class VectorSearchService:
    """Service class containing vector search business logic"""

    def __init__(self, repository: VectorSearchRepository):
        self.repository = repository

    async def execute_vector_search(
        self, user, query: str, top_k: int, similarity_threshold: float, collection_ids: List[str]
    ) -> List[DocumentWithScore]:
        """Execute vector search with given parameters"""
        collection = None
        if collection_ids:
            collection = await self.repository.get_collection(user, collection_ids[0])

        if not collection:
            return []

        try:
            collection_name = generate_vector_db_collection_name(collection.id)
            embedding_model, vector_size = get_collection_embedding_service_sync(collection)
            vectordb_ctx = json.loads(settings.vector_db_context)
            vectordb_ctx["collection"] = collection_name
            context_manager = ContextManager(collection_name, embedding_model, settings.vector_db_type, vectordb_ctx)

            vector = embedding_model.embed_query(query)

            # Vision indexing might produce two types of vectors for the same image: multimodal embedding and text embedding,
            # which could lead to the same document chunk being retrieved twice. To ensure the number of unique results
            # is sufficient after deduplication, we double the top_k value before querying and then deduplicate the results.
            top_k = top_k * 2
            results = context_manager.query(query, score_threshold=similarity_threshold, topk=top_k, vector=vector)

            # Add recall type metadata
            for item in results:
                if item.metadata is None:
                    item.metadata = {}
                item.metadata["recall_type"] = "vector_search"

            # Deduplicate vision results
            results = _deduplicate_vision_results(results)

            return results[:top_k]
        except ProviderNotFoundError as e:
            # Configuration error - gracefully degrade by returning empty results
            logger.warning(f"Vector search skipped for collection {collection.id} due to provider not found: {str(e)}")
            return []
        except EmbeddingError as e:
            # Embedding error - gracefully degrade by returning empty results
            logger.warning(f"Vector search skipped for collection {collection.id} due to embedding error: {str(e)}")
            return []
        except Exception as e:
            logger.error(f"Vector search failed for collection {collection.id}: {str(e)}")
            return []


@register_node_runner(
    "vector_search",
    input_model=VectorSearchInput,
    output_model=VectorSearchOutput,
)
class VectorSearchNodeRunner(BaseNodeRunner):
    def __init__(self):
        self.repository = VectorSearchRepository()
        self.service = VectorSearchService(self.repository)

    async def run(self, ui: VectorSearchInput, si: SystemInput) -> Tuple[VectorSearchOutput, dict]:
        """
        Run vector search node. ui: user configurable params; si: system injected params (SystemInput).
        Returns (uo, so)
        """
        results = await self.service.execute_vector_search(
            user=si.user,
            query=si.query,
            top_k=ui.top_k,
            similarity_threshold=ui.similarity_threshold,
            collection_ids=ui.collection_ids or [],
        )

        return VectorSearchOutput(docs=results), {}
