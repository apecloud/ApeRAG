import json
from typing import Any, Dict, List, Optional, Tuple

from asgiref.sync import sync_to_async
from pydantic import BaseModel, Field

from aperag.context.context import ContextManager
from aperag.db.models import Collection
from aperag.embed.base_embedding import get_collection_embedding_service
from aperag.flow.base.models import BaseNodeRunner, register_node_runner
from aperag.query.query import DocumentWithScore
from aperag.utils.utils import generate_vector_db_collection_name
from config import settings


# User input model for vector search node
class VectorSearchInput(BaseModel):
    top_k: int = Field(5, description="Number of top results to return")
    similarity_threshold: float = Field(0.7, description="Similarity threshold for vector search")
    collection_ids: Optional[List[str]] = Field(default_factory=list, description="Collection IDs")


# User output model for vector search node
class VectorSearchUserOutput(BaseModel):
    docs: List[DocumentWithScore]


@register_node_runner(
    "vector_search",
    input_model=VectorSearchInput,
    output_model=VectorSearchUserOutput,
)
class VectorSearchNodeRunner(BaseNodeRunner):
    async def run(self, ui: VectorSearchInput, si: Dict[str, Any]) -> Tuple[VectorSearchUserOutput, dict]:
        """
        Run vector search node. up: user configurable params; sp: system injected params (dict).
        Returns (uo, so)
        """
        query: str = si["query"]
        topk: int = ui.top_k
        score_threshold: float = ui.similarity_threshold
        collection_ids: List[str] = ui.collection_ids or []
        collection = None
        if collection_ids:
            collections = await sync_to_async(Collection.objects.filter(id__in=collection_ids).all)()
            async for item in collections:
                collection = item
                break
        if not collection:
            return VectorSearchUserOutput(docs=[]), {}

        collection_name = generate_vector_db_collection_name(collection.id)
        embedding_model, vector_size = await get_collection_embedding_service(collection)
        vectordb_ctx = json.loads(settings.VECTOR_DB_CONTEXT)
        vectordb_ctx["collection"] = collection_name
        context_manager = ContextManager(collection_name, embedding_model, settings.VECTOR_DB_TYPE, vectordb_ctx)

        vector = embedding_model.embed_query(query)
        results = context_manager.query(query, score_threshold=score_threshold, topk=topk, vector=vector)
        return VectorSearchUserOutput(docs=results), {}
