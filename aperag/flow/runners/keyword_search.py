
import logging
from aperag.flow.base.models import NodeInstance, register_node_runner, BaseNodeRunner 
from aperag.query.query import DocumentWithScore
from aperag.utils.utils import generate_vector_db_collection_name
from config import settings
from typing import Any, Dict
from aperag.db.models import Collection

logger = logging.getLogger(__name__)

@register_node_runner("keyword_search")
class KeywordSearchNodeRunner(BaseNodeRunner):
    async def run(self, node: NodeInstance, inputs: Dict[str, Any]):
        query: str = inputs["query"]
        topk: int = inputs.get("top_k", 5)
        collection: Collection = inputs.get("collection")

        from aperag.pipeline.keyword_extractor import IKExtractor
        from aperag.context.full_text import search_document
        index = generate_vector_db_collection_name(collection.id)
        async with IKExtractor({"index_name": index, "es_host": settings.ES_HOST}) as extractor:
            keywords = await extractor.extract(query)

        # find the related documents using keywords
        docs = await search_document(index, keywords, topk * 3)
        result = []
        if docs:
            result = [DocumentWithScore(text=doc["content"], score=doc.get("score", 0.5), metadata=doc) for doc in docs]
        return {"keyword_search_docs": result}