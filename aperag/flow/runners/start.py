from typing import Tuple

from pydantic import BaseModel, Field

from aperag.flow.base.models import BaseNodeRunner, SystemInput, register_node_runner


class StartInput(BaseModel):
    query: str = Field(..., description="User's question or query")


class StartOutput(BaseModel):
    query: str


@register_node_runner(
    "start",
    input_model=StartInput,
    output_model=StartOutput,
)
class StartNodeRunner(BaseNodeRunner):
    async def run(self, ui: StartInput, si: SystemInput) -> Tuple[StartOutput, dict]:
        """
        Run start node. ui: user input; si: system input (SystemInput).
        Returns (output, system_output)
        """
        return StartOutput(query=si.query), {}
