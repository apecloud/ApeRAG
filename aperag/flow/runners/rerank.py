from typing import Any, Dict, List

from aperag.flow.base.models import BaseNodeRunner, NodeInstance, register_node_runner
from aperag.query.query import DocumentWithScore
from aperag.rank.reranker import rerank


@register_node_runner("rerank")
class RerankNodeRunner(BaseNodeRunner):
    async def run(self, node: NodeInstance, inputs: Dict[str, Any]):
        query: str = inputs.get("query")
        docs: List[DocumentWithScore] = inputs.get("docs", [])

        result = []
        if docs:
            result = await rerank(query, docs)
        return {"docs": result}
