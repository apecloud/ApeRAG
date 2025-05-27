from typing import Any, Dict, List

from aperag.flow.base.exceptions import ValidationError
from aperag.flow.base.models import BaseNodeRunner, NodeInstance, register_node_runner
from aperag.query.query import DocumentWithScore


@register_node_runner("merge")
class MergeNodeRunner(BaseNodeRunner):
    async def run(self, node: NodeInstance, inputs: Dict[str, Any]):
        docs_a: List[DocumentWithScore] = inputs.get("vector_search_docs", [])
        docs_b: List[DocumentWithScore] = inputs.get("keyword_search_docs", [])
        merge_strategy: str = inputs.get("merge_strategy", "union")
        deduplicate: bool = inputs.get("deduplicate", True)

        if merge_strategy == "union":
            all_docs = docs_a + docs_b
            if deduplicate:
                seen = set()
                unique_docs = []
                for doc in all_docs:
                    if doc.text not in seen:
                        seen.add(doc.text)
                        unique_docs.append(doc)
                return {"docs": unique_docs}
            return {"docs": all_docs}
        else:
            raise ValidationError(f"Unknown merge strategy: {merge_strategy}")
