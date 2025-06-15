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
from aperag.llm.llm_error_types import (
    InvalidConfigurationError,
    ProviderNotFoundError,
    RerankError,
)
from aperag.llm.rerank.rerank_service import RerankService
from aperag.query.query import DocumentWithScore

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

        if not docs:
            logger.info("No documents to rerank, returning empty result")
            return RerankOutput(docs=result), {}

        try:
            # Validate input configuration
            if not ui.model_service_provider:
                raise InvalidConfigurationError(
                    "model_service_provider", ui.model_service_provider, "Model service provider cannot be empty"
                )

            if not ui.model:
                raise InvalidConfigurationError("model", ui.model, "Model name cannot be empty")

            if not ui.custom_llm_provider:
                raise InvalidConfigurationError(
                    "custom_llm_provider", ui.custom_llm_provider, "Custom LLM provider cannot be empty"
                )

            # Get API key and base URL from user's model service provider settings
            try:
                msp_dict = await async_db_ops.query_msp_dict(si.user)
            except Exception as e:
                logger.error(f"Failed to query model service providers: {str(e)}")
                raise RerankError(f"Failed to query model service providers: {str(e)}") from e

            if ui.model_service_provider not in msp_dict:
                available_providers = list(msp_dict.keys())
                logger.error(
                    f"Model service provider '{ui.model_service_provider}' not configured. Available: {available_providers}"
                )
                raise ProviderNotFoundError(ui.model_service_provider, "Rerank")

            msp = msp_dict[ui.model_service_provider]

            if not msp.api_key:
                raise InvalidConfigurationError(
                    "api_key", None, f"API key not configured for provider '{ui.model_service_provider}'"
                )

            # Get base_url from LLMProvider
            try:
                llm_provider = await async_db_ops.query_llm_provider_by_name(ui.model_service_provider)
                if not llm_provider:
                    raise ProviderNotFoundError(ui.model_service_provider, "Rerank")
                base_url = llm_provider.base_url
            except Exception as e:
                logger.error(f"Failed to query LLM provider '{ui.model_service_provider}': {str(e)}")
                raise ProviderNotFoundError(ui.model_service_provider, "Rerank") from e

            if not base_url:
                raise InvalidConfigurationError(
                    "base_url", base_url, f"Base URL not configured for provider '{ui.model_service_provider}'"
                )

            # Create rerank service with configuration from model service provider
            rerank_service = RerankService(
                rerank_provider=ui.custom_llm_provider,
                rerank_model=ui.model,
                rerank_service_url=base_url,
                rerank_service_api_key=msp.api_key,
            )

            # Validate the service configuration
            rerank_service.validate_configuration()

            logger.info(
                f"Using rerank service with provider: {ui.model_service_provider}, "
                f"model: {ui.model}, url: {base_url}, max_docs: {rerank_service.max_documents}"
            )

            # Perform reranking
            result = await rerank_service.rank(query, docs)
            logger.info(f"Successfully reranked {len(result)} documents")

        except (InvalidConfigurationError, ProviderNotFoundError) as e:
            # Configuration errors - log and return empty result to gracefully degrade
            logger.error(f"Rerank configuration error: {str(e)}")
            # For flow execution, we gracefully degrade instead of failing the entire flow
            result = docs  # Return original documents without reranking
        except RerankError as e:
            # Rerank-specific errors - log and return original documents
            logger.error(f"Rerank operation failed: {str(e)}")
            # For flow execution, we gracefully degrade instead of failing the entire flow
            result = docs  # Return original documents without reranking
        except Exception as e:
            # Unexpected errors - log and return original documents
            logger.error(f"Unexpected error during rerank: {str(e)}")
            # For flow execution, we gracefully degrade instead of failing the entire flow
            result = docs  # Return original documents without reranking

        return RerankOutput(docs=result), {}
