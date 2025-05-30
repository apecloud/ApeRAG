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
from typing import List, Optional, Tuple

from pydantic import BaseModel, Field

from aperag.context.context import ContextManager
from aperag.db.ops import query_collection
from aperag.embed.base_embedding import get_collection_embedding_service
from aperag.flow.base.models import BaseNodeRunner, SystemInput, register_node_runner
from aperag.query.query import DocumentWithScore
from aperag.utils.utils import generate_vector_db_collection_name
from config import settings


# User input model for vector search node
class VectorSearchInput(BaseModel):
    top_k: int = Field(5, description="Number of top results to return")
    similarity_threshold: float = Field(0.7, description="Similarity threshold for vector search")
    collection_ids: Optional[List[str]] = Field(default_factory=list, description="Collection IDs")


# User output model for vector search node
class VectorSearchOutput(BaseModel):
    docs: List[DocumentWithScore]


@register_node_runner(
    "vector_search",
    input_model=VectorSearchInput,
    output_model=VectorSearchOutput,
)
class VectorSearchNodeRunner(BaseNodeRunner):
    async def run(self, ui: VectorSearchInput, si: SystemInput) -> Tuple[VectorSearchOutput, dict]:
        """
        Run vector search node. up: user configurable params; sp: system injected params (SystemInput).
        Returns (uo, so)
        """
        query: str = si.query
        topk: int = ui.top_k
        score_threshold: float = ui.similarity_threshold
        collection_ids: List[str] = ui.collection_ids or []
        collection = None
        if collection_ids:
            collection = await query_collection(si.user, collection_ids[0])
        if not collection:
            return VectorSearchOutput(docs=[]), {}

        collection_name = generate_vector_db_collection_name(collection.id)
        embedding_model, vector_size = await get_collection_embedding_service(collection)
        vectordb_ctx = json.loads(settings.VECTOR_DB_CONTEXT)
        vectordb_ctx["collection"] = collection_name
        context_manager = ContextManager(collection_name, embedding_model, settings.VECTOR_DB_TYPE, vectordb_ctx)

        vector = embedding_model.embed_query(query)
        results = context_manager.query(query, score_threshold=score_threshold, topk=topk, vector=vector)
        for item in results:
            if item.metadata is None:
                item.metadata = {}
            item.metadata["recall_type"] = "vector_search"
        return VectorSearchOutput(docs=results), {}
