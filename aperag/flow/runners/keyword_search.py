import logging
from typing import Any, Dict, List, Optional, Tuple

from asgiref.sync import sync_to_async
from pydantic import BaseModel, Field

from aperag.db.models import Collection
from aperag.flow.base.models import BaseNodeRunner, NodeInstance, register_node_runner
from aperag.query.query import DocumentWithScore
from aperag.utils.utils import generate_vector_db_collection_name
from config import settings

logger = logging.getLogger(__name__)


class KeywordSearchInput(BaseModel):
    query: str = Field(..., description="User's question or query")
    top_k: int = Field(5, description="Number of top results to return")
    collection_ids: Optional[List[str]] = Field(default_factory=list, description="Collection IDs")

class KeywordSearchOutput(BaseModel):
    docs: List[DocumentWithScore]


@register_node_runner(
    "keyword_search",
    input_model=KeywordSearchInput,
    output_model=KeywordSearchOutput,
)
class KeywordSearchNodeRunner(BaseNodeRunner):
    async def run(self, ui: KeywordSearchInput, si: Dict[str, any]) -> Tuple[KeywordSearchOutput, dict]:
        """
        Run keyword search node. ui: user input; si: system input (dict).
        Returns (output, system_output)
        """
        query = ui.query
        topk = ui.top_k
        collection_ids = ui.collection_ids or []
        collection = None
        if collection_ids:
            collections = await sync_to_async(Collection.objects.filter(id__in=collection_ids).all)()
            async for item in collections:
                collection = item
                break
        if not collection:
            return KeywordSearchOutput(docs=[]), {}

        from aperag.context.full_text import search_document
        from aperag.pipeline.keyword_extractor import IKExtractor

        index = generate_vector_db_collection_name(collection.id)
        async with IKExtractor({"index_name": index, "es_host": settings.ES_HOST}) as extractor:
            keywords = await extractor.extract(query)

        # find the related documents using keywords
        docs = await search_document(index, keywords, topk * 3)
        result = []
        if docs:
            result = [DocumentWithScore(text=doc["content"], score=doc.get("score", 0.5), metadata=doc) for doc in docs]
        return KeywordSearchOutput(docs=result), {}
