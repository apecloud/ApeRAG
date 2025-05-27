from typing import Any, Dict

from aperag.flow.base.models import BaseNodeRunner, NodeInstance, register_node_runner


@register_node_runner("start")
class StartNodeRunner(BaseNodeRunner):
    async def run(self, node: NodeInstance, inputs: Dict[str, Any]):
        return {"query": inputs.get("query")}
