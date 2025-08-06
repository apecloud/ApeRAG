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


# User input model for summary search node
class SummarySearchInput(BaseModel):
    top_k: int = Field(5, description="Number of top results to return")
    similarity_threshold: float = Field(0.2, description="Similarity threshold for summary search")
    collection_ids: Optional[List[str]] = Field(default_factory=list, description="Collection IDs")


# User output model for summary search node
class SummarySearchOutput(BaseModel):
    docs: List[DocumentWithScore]


# Database operations interface
class SummarySearchRepository:
    """Repository interface for summary search database operations"""

    async def get_collection(self, user, collection_id: str) -> Optional[Collection]:
        """Get collection by ID for the user"""
        return await async_db_ops.query_collection(user, collection_id)


# Business logic service
class SummarySearchService:
    """Service class containing summary search business logic"""

    def __init__(self, repository: SummarySearchRepository):
        self.repository = repository

    async def execute_summary_search(
        self, user, query: str, top_k: int, similarity_threshold: float, collection_ids: List[str]
    ) -> List[DocumentWithScore]:
        """Execute summary search with given parameters"""
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

            # Query vector database for summary vectors only
            results = context_manager.query(
                query, score_threshold=similarity_threshold, topk=top_k, vector=vector, index_types=["summary"]
            )

            # Add recall type metadata for summary search
            for item in results:
                if item.metadata is None:
                    item.metadata = {}
                item.metadata["recall_type"] = "summary_search"

            return results

        except ProviderNotFoundError as e:
            # Configuration error - gracefully degrade by returning empty results
            logger.warning(f"Summary search skipped for collection {collection.id} due to provider not found: {str(e)}")
            return []
        except EmbeddingError as e:
            # Embedding error - gracefully degrade by returning empty results
            logger.warning(f"Summary search skipped for collection {collection.id} due to embedding error: {str(e)}")
            return []
        except Exception as e:
            logger.error(f"Summary search failed for collection {collection.id}: {str(e)}")
            return []


@register_node_runner(
    "summary_search",
    input_model=SummarySearchInput,
    output_model=SummarySearchOutput,
)
class SummarySearchNodeRunner(BaseNodeRunner):
    def __init__(self):
        self.repository = SummarySearchRepository()
        self.service = SummarySearchService(self.repository)

    async def run(self, ui: SummarySearchInput, si: SystemInput) -> Tuple[SummarySearchOutput, dict]:
        """
        Run summary search node. ui: user configurable params; si: system injected params (SystemInput).
        Returns (uo, so)
        """
        results = await self.service.execute_summary_search(
            user=si.user,
            query=si.query,
            top_k=ui.top_k,
            similarity_threshold=ui.similarity_threshold,
            collection_ids=ui.collection_ids or [],
        )

        return SummarySearchOutput(docs=results), {}
