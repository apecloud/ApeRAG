from aperag.flow.base.models import BaseNodeRunner, register_node_runner, NodeInstance
from aperag.query.query import DocumentWithScore
from aperag.rank.reranker import rerank
from typing import Any, Dict


@register_node_runner("rerank")
class RerankNodeRunner(BaseNodeRunner):
    async def run(self, node: NodeInstance, inputs: Dict[str, Any]):
        query = inputs.get("query")
        docs = []
        for doc in inputs["docs"]:
            docs.append(DocumentWithScore(
                text=doc["text"],
                score=doc["score"],
                metadata=doc["metadata"],
                source=doc["source"],
            ))
        result = []
        if docs:
            result = await rerank(query, docs)
        return {"docs": result} 
