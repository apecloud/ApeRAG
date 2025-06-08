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

"""LLM Configuration Service

Service for managing LLM provider and model configurations from database.
Replaces the previous JSON file-based configuration system.
"""

from typing import Dict, List

from aperag.db.ops import async_db_ops
from aperag.schema.view_models import ModelConfig


class LLMConfigService:
    """Service for managing LLM configurations from database"""

    @classmethod
    async def get_providers(cls):
        """Get all active LLM providers from database"""
        return await async_db_ops.query_llm_providers()

    @classmethod
    async def get_provider_models(cls):
        """Get all active LLM provider models from database"""
        return await async_db_ops.query_llm_provider_models()

    @classmethod
    async def get_provider_by_name(cls, name: str):
        """Get provider by name"""
        return await async_db_ops.query_llm_provider_by_name(name)

    @classmethod
    async def get_models_by_provider(cls, provider_name: str):
        """Get all models for a specific provider"""
        return await async_db_ops.query_llm_provider_models(provider_name)

    @classmethod
    async def get_models_by_provider_and_api(cls, provider_name: str, api_type: str):
        """Get models for a specific provider and API type"""
        models = await cls.get_models_by_provider(provider_name)
        return [model for model in models if model.api == api_type]

    @classmethod
    async def build_model_configs(cls) -> List[Dict]:
        """Build model configuration list in the same format as the old JSON file

        This maintains compatibility with existing code that expects the JSON format.
        """
        providers = await cls.get_providers()
        provider_models = await cls.get_provider_models()

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

        return config_list

    @classmethod
    async def build_model_config_objects(cls) -> List[ModelConfig]:
        """Build ModelConfig objects for API responses"""
        config_list = await cls.build_model_configs()
        return [ModelConfig(**config) for config in config_list]


# Global function to get model configurations for backward compatibility
async def get_model_configs() -> List[Dict]:
    """Get model configurations from database

    This function provides backward compatibility for code that expects
    the same format as settings.model_configs.
    """
    return await LLMConfigService.build_model_configs()


async def get_model_config_objects() -> List[ModelConfig]:
    """Get ModelConfig objects from database"""
    return await LLMConfigService.build_model_config_objects()


async def get_supported_provider_names() -> set:
    """Get set of supported provider names"""
    configs = await get_model_configs()
    return {config["name"] for config in configs}
