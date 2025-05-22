from aperag.flow.base.models import BaseNodeRunner, register_node_runner, NodeInstance
from typing import Any, Dict


@register_node_runner("start")
class StartNodeRunner(BaseNodeRunner):
    async def run(self, node: NodeInstance, inputs: Dict[str, Any]):
        return {"query": inputs.get("query")}
