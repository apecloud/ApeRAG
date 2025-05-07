from aperag.flow.base.models import BaseNodeRunner, register_node_runner, NodeInstance
from aperag.flow.base.exceptions import ValidationError
from typing import Any, Dict

@register_node_runner("merge")
class MergeNodeRunner(BaseNodeRunner):
    async def run(self, node: NodeInstance, inputs: Dict[str, Any]):
        docs_a = inputs["vector_search_docs"]
        docs_b = inputs["keyword_search_docs"]
        merge_strategy = inputs.get("merge_strategy", "union")
        deduplicate = inputs.get("deduplicate", True)
        if merge_strategy == "union":
            all_docs = docs_a + docs_b
            if deduplicate:
                seen = set()
                unique_docs = []
                for doc in all_docs:
                    content = doc.get("content") or doc.get("text")
                    if content not in seen:
                        seen.add(content)
                        unique_docs.append(doc)
                return {"docs": unique_docs}
            return {"docs": all_docs}
        else:
            raise ValidationError(f"Unknown merge strategy: {merge_strategy}") 
        
