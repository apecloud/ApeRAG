from typing import List, Tuple

from pydantic import BaseModel, Field

from aperag.flow.base.models import BaseNodeRunner, SystemInput, register_node_runner
from aperag.query.query import DocumentWithScore
from aperag.rank.reranker import rerank


class RerankInput(BaseModel):
    model: str = Field(..., description="Rerank model name")
    model_service_provider: str = Field(..., description="Model service provider")
    docs: List[DocumentWithScore]


class RerankOutput(BaseModel):
    docs: List[DocumentWithScore]


@register_node_runner(
    "rerank",
    input_model=RerankInput,
    output_model=RerankOutput,
)
class RerankNodeRunner(BaseNodeRunner):
    async def run(self, ui: RerankInput, si: SystemInput) -> Tuple[RerankOutput, dict]:
        """
        Run rerank node. ui: user input; si: system input (SystemInput).
        Returns (output, system_output)
        """
        query = si.query
        docs = ui.docs
        result = []
        if docs:
            result = await rerank(query, docs)
        return RerankOutput(docs=result), {}
