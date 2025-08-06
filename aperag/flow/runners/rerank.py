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
from typing import List, Optional, Tuple

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
    use_rerank_service: bool = Field(default=True, description="Whether to use rerank service or fallback strategy")
    model: Optional[str] = Field(default=None, description="Rerank model name")
    model_service_provider: Optional[str] = Field(default=None, description="Model service provider")
    custom_llm_provider: Optional[str] = Field(
        default=None, description="Custom LLM provider (e.g., 'jina_ai', 'openai')"
    )
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
        Smart rerank node:
        - use_rerank_service=False: directly use fallback strategy
        - use_rerank_service=True: try rerank service, fallback on failure
        """
        docs = ui.docs

        if not docs:
            logger.info("No documents to rerank, returning empty result")
            return RerankOutput(docs=[]), {}

        # Strategy 1: If not using rerank service, directly use fallback strategy
        if not ui.use_rerank_service:
            logger.info("Rerank service disabled, using fallback strategy")
            result = self._apply_fallback_strategy(docs)
            return RerankOutput(docs=result), {}

        # Strategy 2: Try to use rerank service
        try:
            # Check configuration completeness
            if not self._is_rerank_config_valid(ui):
                logger.info("Rerank service configuration incomplete, using fallback strategy")
                result = self._apply_fallback_strategy(docs)
                return RerankOutput(docs=result), {}

            # Execute actual rerank
            result = await self._perform_actual_rerank(ui, si)
            logger.info(f"Successfully reranked {len(result)} documents using rerank service")
            return RerankOutput(docs=result), {}

        except (InvalidConfigurationError, ProviderNotFoundError) as e:
            logger.warning(f"Rerank service configuration error, using fallback strategy: {str(e)}")
            result = self._apply_fallback_strategy(docs)
            return RerankOutput(docs=result), {}

        except RerankError as e:
            logger.warning(f"Rerank service operation failed, using fallback strategy: {str(e)}")
            result = self._apply_fallback_strategy(docs)
            return RerankOutput(docs=result), {}

        except Exception as e:
            logger.error(f"Unexpected error during rerank service, using fallback strategy: {str(e)}")
            result = self._apply_fallback_strategy(docs)
            return RerankOutput(docs=result), {}

    def _is_rerank_config_valid(self, ui: RerankInput) -> bool:
        """Check if rerank service configuration is valid"""
        return (
            ui.model
            and ui.model.strip()
            and ui.model_service_provider
            and ui.model_service_provider.strip()
            and ui.custom_llm_provider
            and ui.custom_llm_provider.strip()
        )

    async def _perform_actual_rerank(self, ui: RerankInput, si: SystemInput) -> List[DocumentWithScore]:
        """Execute actual rerank operation"""
        query = si.query
        docs = ui.docs

        # Validate configuration
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

        # Get API key and base_url
        api_key = await async_db_ops.query_provider_api_key(ui.model_service_provider, si.user)
        if not api_key:
            raise InvalidConfigurationError(
                "api_key", api_key, f"API KEY not found for LLM Provider:{ui.model_service_provider}"
            )

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

        # Create and execute rerank service
        rerank_service = RerankService(
            rerank_provider=ui.custom_llm_provider,
            rerank_model=ui.model,
            rerank_service_url=base_url,
            rerank_service_api_key=api_key,
        )

        rerank_service.validate_configuration()

        logger.info(
            f"Using rerank service with provider: {ui.model_service_provider}, "
            f"model: {ui.model}, url: {base_url}, max_docs: {rerank_service.max_documents}"
        )

        return await rerank_service.async_rerank(query, docs)

    def _apply_fallback_strategy(self, docs: List[DocumentWithScore]) -> List[DocumentWithScore]:
        """
        Apply fallback rerank strategy:
        1. Graph search results first (better quality, typically 1 result)
        2. Sort remaining vector and fulltext results by score in descending order
        """
        if not docs:
            return docs

        graph_results = []
        other_results = []

        for doc in docs:
            recall_type = doc.metadata.get("recall_type", "")
            if recall_type == "graph_search":
                graph_results.append(doc)
            else:
                other_results.append(doc)

        # Sort other results by score in descending order
        other_results.sort(key=lambda x: x.score if x.score is not None else 0.0, reverse=True)

        result = graph_results + other_results

        logger.info(
            f"Applied fallback rerank strategy: {len(graph_results)} graph results, "
            f"{len(other_results)} other results sorted by score"
        )

        return result
