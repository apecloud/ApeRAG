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


from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm.attributes import flag_modified

from aperag.db.ops import AsyncDatabaseOps, async_db_ops
from aperag.exceptions import BusinessException, ErrorCode
from aperag.schema import view_models


class DefaultModelService:
    """Default Model service that handles business logic for default model configurations"""

    def __init__(self, session: AsyncSession = None):
        # Use global db_ops instance by default, or create custom one with provided session
        if session is None:
            self.db_ops = async_db_ops  # Use global instance
        else:
            self.db_ops = AsyncDatabaseOps(session)  # Create custom instance for transaction control

    async def get_default_models(self, user_id: str) -> view_models.DefaultModelsResponse:
        """Get current default model configurations for different scenarios"""
        scenarios = [
            "default_for_collection_completion",
            "default_for_agent_completion",
            "default_for_embedding",
            "default_for_rerank",
        ]

        default_configs = []
        for scenario in scenarios:
            # Find model with specific tag
            models = await self.db_ops.find_models_by_tag(user_id, scenario)

            # Check if we found a model and if its provider has valid API key
            selected_model = None
            if models:
                for model in models:
                    # Check if provider has API key configured
                    api_key = await self.db_ops.query_provider_api_key(model.provider_name, user_id, True)
                    if api_key:
                        selected_model = model
                        break

            # Create config entry
            if selected_model:
                default_configs.append(
                    view_models.DefaultModelConfig(
                        scenario=scenario, provider_name=selected_model.provider_name, model=selected_model.model
                    )
                )
            else:
                default_configs.append(
                    view_models.DefaultModelConfig(scenario=scenario, provider_name=None, model=None)
                )

        return view_models.DefaultModelsResponse(items=default_configs)

    async def get_default_rerank_config(self, user_id: str) -> tuple:
        """Get default rerank model configuration"""
        # Get all default models using existing method
        default_models = await self.get_default_models(user_id)

        # Find the rerank model configuration
        for config in default_models.items:
            if config.scenario == "default_for_rerank":
                if config.provider_name and config.model:
                    # Return configuration tuple: (model, model_service_provider, custom_llm_provider)
                    return (config.model, config.provider_name, config.provider_name)
                else:
                    # No valid rerank model found
                    return (None, None, None)

        # Should not reach here since get_default_models includes all scenarios
        return (None, None, None)

    async def update_default_models(
        self, user_id: str, update_request: view_models.DefaultModelsUpdateRequest
    ) -> view_models.DefaultModelsResponse:
        """Update default model configurations"""

        # Pre-validate all configurations before starting database operations
        validated_configs = []
        for config in update_request.defaults:
            if config.provider_name and config.model:
                # Check if provider is public (only public providers can be set as default)
                provider = await self.db_ops.query_llm_provider_by_name(config.provider_name)
                if not provider:
                    raise BusinessException(
                        ErrorCode.LLM_MODEL_NOT_FOUND, f"Provider '{config.provider_name}' not found"
                    )
                if provider.user_id != "public":
                    raise BusinessException(
                        ErrorCode.PROVIDER_NOT_PUBLIC,
                        f"Provider '{config.provider_name}' is not a public provider and cannot be set as default model",
                    )

                # Determine the API type based on scenario
                api_type = self._get_api_type_from_scenario(config.scenario)
                validated_configs.append((config, api_type))

        # Execute all operations in a single transaction
        async def _update_operation(session):
            # Use a set of unique scenarios from the request to avoid redundant DB queries
            all_scenarios_in_request = {c.scenario for c in update_request.defaults}
            for scenario in all_scenarios_in_request:
                models_to_update = await self.db_ops.find_models_by_tag_in_session(session, user_id, scenario)
                for model in models_to_update:
                    if model.tags and scenario in model.tags:
                        model.tags.remove(scenario)
                        flag_modified(model, "tags")

            # Then, add new default tags
            for config, api_type in validated_configs:
                # Find the specific model
                from sqlalchemy import select

                from aperag.db.models import LLMProvider, LLMProviderModel

                stmt = (
                    select(LLMProviderModel)
                    .join(LLMProvider, LLMProviderModel.provider_name == LLMProvider.name)
                    .where(
                        LLMProvider.gmt_deleted.is_(None),
                        LLMProviderModel.gmt_deleted.is_(None),
                        LLMProviderModel.provider_name == config.provider_name,
                        LLMProviderModel.api == api_type,
                        LLMProviderModel.model == config.model,
                        LLMProvider.user_id == "public",
                    )
                )

                result = await session.execute(stmt)
                model_obj = result.scalars().first()

                if model_obj:
                    if model_obj.tags is None:
                        model_obj.tags = []
                    if config.scenario not in model_obj.tags:
                        model_obj.tags.append(config.scenario)
                        flag_modified(model_obj, "tags")

        # Execute the entire operation in a single transaction
        await self.db_ops.execute_with_transaction(_update_operation)

        # Return updated configuration
        return await self.get_default_models(user_id)

    def _get_api_type_from_scenario(self, scenario: str) -> str:
        """Map scenario to API type"""
        if scenario in ["default_for_collection_completion", "default_for_agent_completion"]:
            return "completion"
        elif scenario == "default_for_embedding":
            return "embedding"
        elif scenario == "default_for_rerank":
            return "rerank"
        else:
            raise ValueError(f"Unknown scenario: {scenario}")


# Create a global service instance for easy access
default_model_service = DefaultModelService()
