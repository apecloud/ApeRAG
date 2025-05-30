# Copyright 2025 ApeCloud, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
