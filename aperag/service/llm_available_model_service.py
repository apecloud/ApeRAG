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
        self, user_id: str, tag_filter_request: view_models.TagFilterRequest
    ) -> view_models.ModelConfigList:
        """Get available models with optional tag filtering"""
        # Get providers and models data with user_id information
        providers_and_models_data = await self.db_ops.query_available_providers_with_models(user_id, True)
        available_providers = providers_and_models_data["providers"]
        provider_models = providers_and_models_data["models"]

        # Build provider configs
        provider_configs = self._build_provider_configs_from_data(available_providers, provider_models)

        # Apply filtering logic uniformly to all providers
        if tag_filter_request.tag_filters is None or len(tag_filter_request.tag_filters) == 0:
            # Show all providers if no filter is specified
            filtered_providers = provider_configs
        else:
            # Apply user-specified filters to all providers
            filtered_providers = filter_providers_by_tags(provider_configs, tag_filter_request.tag_filters)

        return view_models.ModelConfigList(items=filtered_providers, pageResult=None)

    def _build_provider_configs_from_data(self, available_providers, provider_models) -> List[view_models.ModelConfig]:
        """Build ModelConfig objects from already fetched data"""
        from collections import defaultdict

        provider_model_map = defaultdict(lambda: {"completion": [], "embedding": [], "rerank": []})

        for model in provider_models:
            model_dict = _build_model_dict(model)
            provider_model_map[model.provider_name][model.api].append(model_dict)

        # Build the final configuration list
        return [
            view_models.ModelConfig(
                name=provider.name,
                label=provider.label,
                completion_dialect=provider.completion_dialect,
                embedding_dialect=provider.embedding_dialect,
                rerank_dialect=provider.rerank_dialect,
                allow_custom_base_url=provider.allow_custom_base_url,
                base_url=provider.base_url,
                **provider_model_map[provider.name],  # This will use default empty lists if no models
            )
            for provider in available_providers
        ]


def _build_model_dict(model) -> dict:
    """Build model dictionary from LLMProviderModel object"""
    model_dict = {
        "model": model.model,
        "custom_llm_provider": model.custom_llm_provider,
    }
    if model.context_window:
        model_dict["context_window"] = model.context_window
    if model.max_input_tokens:
        model_dict["max_input_tokens"] = model.max_input_tokens
    if model.max_output_tokens:
        model_dict["max_output_tokens"] = model.max_output_tokens
    if model.tags:
        model_dict["tags"] = model.tags
    return model_dict


def filter_models_by_tags(
    models: List[dict], tag_filters: Optional[List[view_models.TagFilterCondition]]
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


def filter_providers_by_tags(
    providers: List[view_models.ModelConfig], tag_filters: Optional[List[view_models.TagFilterCondition]]
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
                filtered_models = filter_models_by_tags(valid_models, tag_filters)

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
