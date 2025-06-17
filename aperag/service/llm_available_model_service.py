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

from typing import List, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from aperag.db.ops import AsyncDatabaseOps, async_db_ops
from aperag.schema import view_models


class LlmAvailableModelService:
    """LLM Available Model service that handles business logic for available models"""

    def __init__(self, session: AsyncSession = None):
        # Use global db_ops instance by default, or create custom one with provided session
        if session is None:
            self.db_ops = async_db_ops  # Use global instance
        else:
            self.db_ops = AsyncDatabaseOps(session)  # Create custom instance for transaction control

    async def get_available_models(
        self, user: str, tag_filter_request: view_models.TagFilterRequest
    ) -> view_models.ModelConfigList:
        """Get available models with optional tag filtering"""
        # Get all supported providers from model configs
        supported_providers = await self._build_model_config_objects()

        # Check which providers have API keys configured
        available_providers = []
        for provider in supported_providers:
            api_key = await self.db_ops.query_provider_api_key(provider.name, user)
            if api_key:  # Only include providers that have API keys configured
                available_providers.append(provider)

        # Apply tag filtering based on request
        if tag_filter_request.tag_filters is None or len(tag_filter_request.tag_filters) == 0:
            # Default to showing only models with 'recommend' tag when no filters are provided
            default_filter = [view_models.TagFilterCondition(tags=["recommend"], operation="AND")]
            filtered_providers = self._filter_providers_by_tags(available_providers, default_filter)
        else:
            filtered_providers = self._filter_providers_by_tags(available_providers, tag_filter_request.tag_filters)

        return view_models.ModelConfigList(items=filtered_providers, pageResult=None)

    def _filter_models_by_tags(
        self, models: List[dict], tag_filters: Optional[List[view_models.TagFilterCondition]]
    ) -> List[dict]:
        """Filter models by tag conditions

        Args:
            models: List of model dictionaries with 'tags' field
            tag_filters: List of TagFilterCondition objects

        Returns:
            Filtered list of models
        """
        if not tag_filters:
            return models

        filtered_models = []

        for model in models:
            model_tags = set(model.get("tags", []) or [])

            # Check if model matches any of the filter conditions (OR between conditions)
            matches_any_condition = False

            for condition in tag_filters:
                operation = condition.operation.upper() if condition.operation else "AND"
                required_tags = set(condition.tags or [])

                if not required_tags:
                    continue

                if operation == "AND":
                    # All tags must be present
                    if required_tags.issubset(model_tags):
                        matches_any_condition = True
                        break
                elif operation == "OR":
                    # At least one tag must be present
                    if required_tags.intersection(model_tags):
                        matches_any_condition = True
                        break

            if matches_any_condition:
                filtered_models.append(model)

        return filtered_models

    async def _build_model_config_objects(self) -> List[view_models.ModelConfig]:
        """Build ModelConfig objects from database data

        This function replaces the functionality from llm_config_service.py
        """
        # Get providers and provider models from database
        providers = await self.db_ops.query_llm_providers()
        provider_models = await self.db_ops.query_llm_provider_models()

        # Group models by provider and API type
        provider_model_map = {}
        for model in provider_models:
            if model.provider_name not in provider_model_map:
                provider_model_map[model.provider_name] = {"completion": [], "embedding": [], "rerank": []}

            model_dict = {
                "model": model.model,
                "custom_llm_provider": model.custom_llm_provider,
            }
            if model.max_tokens:
                model_dict["max_tokens"] = model.max_tokens
            if model.tags:
                model_dict["tags"] = model.tags

            provider_model_map[model.provider_name][model.api].append(model_dict)

        # Build the final configuration list
        config_list = []
        for provider in providers:
            provider_config = {
                "name": provider.name,
                "label": provider.label,
                "completion_dialect": provider.completion_dialect,
                "embedding_dialect": provider.embedding_dialect,
                "rerank_dialect": provider.rerank_dialect,
                "allow_custom_base_url": provider.allow_custom_base_url,
                "base_url": provider.base_url,
            }

            # Add model configurations
            if provider.name in provider_model_map:
                provider_config.update(provider_model_map[provider.name])
            else:
                # Ensure these keys exist even if no models
                provider_config.update({"completion": [], "embedding": [], "rerank": []})

            config_list.append(provider_config)

        return [view_models.ModelConfig(**config) for config in config_list]

    def _filter_providers_by_tags(
        self, providers: List[view_models.ModelConfig], tag_filters: Optional[List[view_models.TagFilterCondition]]
    ) -> List[view_models.ModelConfig]:
        """Helper function to filter providers by tags - filters at model level"""
        filtered_providers = []

        for provider in providers:
            provider_dict = provider.model_dump()

            # Filter each model type separately
            has_any_models = False
            for model_type in ["completion", "embedding", "rerank"]:
                models = provider_dict.get(model_type, [])
                if models:
                    # Filter out None values and ensure we only process valid models
                    valid_models = [model for model in models if model is not None]

                    # Apply tag filtering to each model
                    filtered_models = self._filter_models_by_tags(valid_models, tag_filters)

                    # Update the provider with filtered models
                    provider_dict[model_type] = filtered_models

                    # Track if we have any models left
                    if filtered_models:
                        has_any_models = True

            # Only include provider if it has at least one matching model
            if has_any_models:
                filtered_providers.append(view_models.ModelConfig(**provider_dict))

        return filtered_providers


# Create a global service instance for easy access
# This uses the global db_ops instance and doesn't require session management in views
llm_available_model_service = LlmAvailableModelService()
