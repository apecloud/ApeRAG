import json
from typing import Any, Dict
from aperag.db.models import Collection
from aperag.embed.base_embedding import get_collection_embedding_service
from aperag.flow.base.models import BaseNodeRunner, register_node_runner, NodeInstance
from aperag.context.context import ContextManager
from aperag.utils.utils import generate_vector_db_collection_name
from config import settings

@register_node_runner("vector_search")
class VectorSearchNodeRunner(BaseNodeRunner):
    async def run(self, node: NodeInstance, inputs: Dict[str, Any]):
        collection: Collection = inputs.get("collection")

        collection_name = generate_vector_db_collection_name(collection.id)
        embedding_model, vector_size = await get_collection_embedding_service(collection)
        vectordb_ctx = json.loads(settings.VECTOR_DB_CONTEXT)
        vectordb_ctx["collection"] = collection_name
        context_manager = ContextManager(collection_name, embedding_model, settings.VECTOR_DB_TYPE,
                                              vectordb_ctx)
        topk = inputs.get("top_k", 5)
        score_threshold = inputs.get("similarity_threshold", 0.1)
        context_manager = ContextManager(collection_name, embedding_model, settings.VECTOR_DB_TYPE, vectordb_ctx)

        query = inputs["query"]
        vector = embedding_model.embed_query(query)
        results = context_manager.query(query, score_threshold=score_threshold, topk=topk, vector=vector)
        docs = [doc.dict() if hasattr(doc, "dict") else doc for doc in results]
        return {"vector_search_docs": docs} 
