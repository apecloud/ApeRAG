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

import logging
from typing import List, Tuple

from pydantic import BaseModel, Field

from aperag.db.ops import async_db_ops
from aperag.flow.base.models import BaseNodeRunner, SystemInput, register_node_runner
from aperag.query.query import DocumentWithScore
from aperag.rerank.rerank_service import RerankService

logger = logging.getLogger(__name__)


class RerankInput(BaseModel):
    model: str = Field(..., description="Rerank model name")
    model_service_provider: str = Field(..., description="Model service provider")
    custom_llm_provider: str = Field(..., description="Custom LLM provider (e.g., 'jina_ai', 'openai')")
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
            # Get API key and base URL from user's model service provider settings
            msp_dict = await async_db_ops.query_msp_dict(si.user)
            if ui.model_service_provider not in msp_dict:
                raise ValueError(
                    f"Model service provider '{ui.model_service_provider}' not configured. "
                    f"Please configure it in the settings."
                )

            msp = msp_dict[ui.model_service_provider]

            # Get base_url from LLMProvider
            try:
                llm_provider = await async_db_ops.query_llm_provider_by_name(ui.model_service_provider)
                base_url = llm_provider.base_url
            except Exception:
                raise ValueError(f"LLMProvider '{ui.model_service_provider}' not found")

            # Create rerank service with configuration from model service provider
            rerank_service = RerankService(
                rerank_provider=ui.custom_llm_provider,
                rerank_model=ui.model,
                rerank_service_url=base_url,
                rerank_service_api_key=msp.api_key,
            )

            logger.info(
                f"Using rerank service with provider: {ui.model_service_provider}, model: {ui.model}, url: {base_url}"
            )

            result = await rerank_service.rank(query, docs)

        return RerankOutput(docs=result), {}
