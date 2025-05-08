from aperag.flow.base.models import BaseNodeRunner, register_node_runner, NodeInstance
from aperag.query.query import DocumentWithScore
from aperag.rank.reranker import rerank
from typing import Any, Dict, List


@register_node_runner("rerank")
class RerankNodeRunner(BaseNodeRunner):
    async def run(self, node: NodeInstance, inputs: Dict[str, Any]):
        query: str = inputs.get("query")
        docs: List[DocumentWithScore] = inputs.get("docs", [])

        result = []
        if docs:
            result = await rerank(query, docs)
        return {"docs": result} 
