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

from http import HTTPStatus
from typing import Optional

from aperag.db.ops import async_db_ops
from aperag.views.utils import fail, success


async def get_llm_configuration():
    """Get complete LLM configuration including providers and models"""
    try:
        # Get all providers
        providers = await async_db_ops.query_llm_providers()
        providers_data = []

        for provider in providers:
            providers_data.append(
                {
                    "name": provider.name,
                    "label": provider.label,
                    "completion_dialect": provider.completion_dialect,
                    "embedding_dialect": provider.embedding_dialect,
                    "rerank_dialect": provider.rerank_dialect,
                    "allow_custom_base_url": provider.allow_custom_base_url,
                    "base_url": provider.base_url,
                    "extra": provider.extra,
                    "created": provider.gmt_created,
                    "updated": provider.gmt_updated,
                }
            )

        # Get all models
        models = await async_db_ops.query_llm_provider_models()
        models_data = []

        for model in models:
            models_data.append(
                {
                    "provider_name": model.provider_name,
                    "api": model.api,
                    "model": model.model,
                    "custom_llm_provider": model.custom_llm_provider,
                    "max_tokens": model.max_tokens,
                    "tags": model.tags or [],
                    "created": model.gmt_created,
                    "updated": model.gmt_updated,
                }
            )

        return success(
            {
                "providers": providers_data,
                "models": models_data,
            }
        )
    except Exception as e:
        return fail(HTTPStatus.INTERNAL_SERVER_ERROR, f"Failed to get LLM configuration: {str(e)}")


async def list_llm_providers():
    """List all LLM providers"""
    try:
        providers = await async_db_ops.query_llm_providers()
        providers_data = []

        for provider in providers:
            providers_data.append(
                {
                    "name": provider.name,
                    "label": provider.label,
                    "completion_dialect": provider.completion_dialect,
                    "embedding_dialect": provider.embedding_dialect,
                    "rerank_dialect": provider.rerank_dialect,
                    "allow_custom_base_url": provider.allow_custom_base_url,
                    "base_url": provider.base_url,
                    "extra": provider.extra,
                    "created": provider.gmt_created,
                    "updated": provider.gmt_updated,
                }
            )

        return success(
            {
                "items": providers_data,
                "pageResult": {
                    "page_number": 1,
                    "page_size": len(providers_data),
                    "count": len(providers_data),
                },
            }
        )
    except Exception as e:
        return fail(HTTPStatus.INTERNAL_SERVER_ERROR, f"Failed to list LLM providers: {str(e)}")


async def create_llm_provider(provider_data: dict):
    """Create a new LLM provider or restore a soft-deleted one with the same name"""
    try:
        # First check if there's an active provider with the same name
        active_existing = await async_db_ops.query_llm_provider_by_name(provider_data["name"])

        if active_existing:
            return fail(HTTPStatus.BAD_REQUEST, f"Provider with name '{provider_data['name']}' already exists")

        # Try to restore a soft-deleted provider if it exists
        provider = await async_db_ops.restore_llm_provider(provider_data["name"])

        if provider:
            # Update the restored provider with new data
            provider = await async_db_ops.update_llm_provider(
                name=provider_data["name"],
                label=provider_data["label"],
                completion_dialect=provider_data.get("completion_dialect", "openai"),
                embedding_dialect=provider_data.get("embedding_dialect", "openai"),
                rerank_dialect=provider_data.get("rerank_dialect", "jina_ai"),
                allow_custom_base_url=provider_data.get("allow_custom_base_url", False),
                base_url=provider_data["base_url"],
                extra=provider_data.get("extra"),
            )
        else:
            # Create new provider
            provider = await async_db_ops.create_llm_provider(
                name=provider_data["name"],
                label=provider_data["label"],
                completion_dialect=provider_data.get("completion_dialect", "openai"),
                embedding_dialect=provider_data.get("embedding_dialect", "openai"),
                rerank_dialect=provider_data.get("rerank_dialect", "jina_ai"),
                allow_custom_base_url=provider_data.get("allow_custom_base_url", False),
                base_url=provider_data["base_url"],
                extra=provider_data.get("extra"),
            )

        return success(
            {
                "name": provider.name,
                "label": provider.label,
                "completion_dialect": provider.completion_dialect,
                "embedding_dialect": provider.embedding_dialect,
                "rerank_dialect": provider.rerank_dialect,
                "allow_custom_base_url": provider.allow_custom_base_url,
                "base_url": provider.base_url,
                "extra": provider.extra,
                "created": provider.gmt_created,
                "updated": provider.gmt_updated,
            }
        )
    except Exception as e:
        return fail(HTTPStatus.INTERNAL_SERVER_ERROR, f"Failed to create LLM provider: {str(e)}")


async def get_llm_provider(provider_name: str):
    """Get a specific LLM provider by name"""
    try:
        provider = await async_db_ops.query_llm_provider_by_name(provider_name)

        if not provider:
            return fail(HTTPStatus.NOT_FOUND, f"Provider '{provider_name}' not found")

        return success(
            {
                "name": provider.name,
                "label": provider.label,
                "completion_dialect": provider.completion_dialect,
                "embedding_dialect": provider.embedding_dialect,
                "rerank_dialect": provider.rerank_dialect,
                "allow_custom_base_url": provider.allow_custom_base_url,
                "base_url": provider.base_url,
                "extra": provider.extra,
                "created": provider.gmt_created,
                "updated": provider.gmt_updated,
            }
        )
    except Exception as e:
        return fail(HTTPStatus.INTERNAL_SERVER_ERROR, f"Failed to get LLM provider: {str(e)}")


async def update_llm_provider(provider_name: str, update_data: dict):
    """Update an existing LLM provider"""
    try:
        existing_provider = await async_db_ops.query_llm_provider_by_name(provider_name)

        if not existing_provider:
            return fail(HTTPStatus.NOT_FOUND, f"Provider '{provider_name}' not found")

        # Update provider using the DatabaseOps method
        provider = await async_db_ops.update_llm_provider(
            name=provider_name,
            label=update_data.get("label"),
            completion_dialect=update_data.get("completion_dialect"),
            embedding_dialect=update_data.get("embedding_dialect"),
            rerank_dialect=update_data.get("rerank_dialect"),
            allow_custom_base_url=update_data.get("allow_custom_base_url"),
            base_url=update_data.get("base_url"),
            extra=update_data.get("extra"),
        )

        return success(
            {
                "name": provider.name,
                "label": provider.label,
                "completion_dialect": provider.completion_dialect,
                "embedding_dialect": provider.embedding_dialect,
                "rerank_dialect": provider.rerank_dialect,
                "allow_custom_base_url": provider.allow_custom_base_url,
                "base_url": provider.base_url,
                "extra": provider.extra,
                "created": provider.gmt_created,
                "updated": provider.gmt_updated,
            }
        )
    except Exception as e:
        return fail(HTTPStatus.INTERNAL_SERVER_ERROR, f"Failed to update LLM provider: {str(e)}")


async def delete_llm_provider(provider_name: str):
    """Delete an LLM provider (soft delete)"""
    try:
        provider = await async_db_ops.query_llm_provider_by_name(provider_name)

        if not provider:
            return fail(HTTPStatus.NOT_FOUND, f"Provider '{provider_name}' not found")

        # Soft delete the provider and its models
        await async_db_ops.delete_llm_provider(provider_name)

        return success(None)
    except Exception as e:
        return fail(HTTPStatus.INTERNAL_SERVER_ERROR, f"Failed to delete LLM provider: {str(e)}")


async def list_llm_provider_models(provider_name: Optional[str] = None):
    """List LLM provider models, optionally filtered by provider"""
    try:
        models = await async_db_ops.query_llm_provider_models(provider_name)

        models_data = []
        for model in models:
            models_data.append(
                {
                    "provider_name": model.provider_name,
                    "api": model.api,
                    "model": model.model,
                    "custom_llm_provider": model.custom_llm_provider,
                    "max_tokens": model.max_tokens,
                    "tags": model.tags or [],
                    "created": model.gmt_created,
                    "updated": model.gmt_updated,
                }
            )

        return success(
            {
                "items": models_data,
                "pageResult": {
                    "page_number": 1,
                    "page_size": len(models_data),
                    "count": len(models_data),
                },
            }
        )
    except Exception as e:
        return fail(HTTPStatus.INTERNAL_SERVER_ERROR, f"Failed to list LLM provider models: {str(e)}")


async def create_llm_provider_model(provider_name: str, model_data: dict):
    """Create a new model for a specific provider or restore a soft-deleted one with the same combination"""
    try:
        # Check if provider exists
        provider = await async_db_ops.query_llm_provider_by_name(provider_name)

        if not provider:
            return fail(HTTPStatus.NOT_FOUND, f"Provider '{provider_name}' not found")

        # First check if there's an active model with the same combination
        active_existing = await async_db_ops.query_llm_provider_model(
            provider_name, model_data["api"], model_data["model"]
        )

        if active_existing:
            return fail(
                HTTPStatus.BAD_REQUEST,
                f"Model '{model_data['model']}' for API '{model_data['api']}' already exists for provider '{provider_name}'",
            )

        # Try to restore a soft-deleted model if it exists
        model = await async_db_ops.restore_llm_provider_model(provider_name, model_data["api"], model_data["model"])

        if model:
            # Update the restored model with new data
            model = await async_db_ops.update_llm_provider_model(
                provider_name=provider_name,
                api=model_data["api"],
                model=model_data["model"],
                custom_llm_provider=model_data["custom_llm_provider"],
                max_tokens=model_data.get("max_tokens"),
                tags=model_data.get("tags", []),
            )
        else:
            # Create new model
            model = await async_db_ops.create_llm_provider_model(
                provider_name=provider_name,
                api=model_data["api"],
                model=model_data["model"],
                custom_llm_provider=model_data["custom_llm_provider"],
                max_tokens=model_data.get("max_tokens"),
                tags=model_data.get("tags", []),
            )

        return success(
            {
                "provider_name": model.provider_name,
                "api": model.api,
                "model": model.model,
                "custom_llm_provider": model.custom_llm_provider,
                "max_tokens": model.max_tokens,
                "tags": model.tags or [],
                "created": model.gmt_created,
                "updated": model.gmt_updated,
            }
        )
    except Exception as e:
        return fail(HTTPStatus.INTERNAL_SERVER_ERROR, f"Failed to create LLM provider model: {str(e)}")


async def update_llm_provider_model(provider_name: str, api: str, model: str, update_data: dict):
    """Update a specific model of a provider"""
    try:
        existing_model = await async_db_ops.query_llm_provider_model(provider_name, api, model)

        if not existing_model:
            return fail(
                HTTPStatus.NOT_FOUND, f"Model '{model}' for API '{api}' not found for provider '{provider_name}'"
            )

        # Update model using the DatabaseOps method
        model_obj = await async_db_ops.update_llm_provider_model(
            provider_name=provider_name,
            api=api,
            model=model,
            custom_llm_provider=update_data.get("custom_llm_provider"),
            max_tokens=update_data.get("max_tokens"),
            tags=update_data.get("tags"),
        )

        return success(
            {
                "provider_name": model_obj.provider_name,
                "api": model_obj.api,
                "model": model_obj.model,
                "custom_llm_provider": model_obj.custom_llm_provider,
                "max_tokens": model_obj.max_tokens,
                "tags": model_obj.tags or [],
                "created": model_obj.gmt_created,
                "updated": model_obj.gmt_updated,
            }
        )
    except Exception as e:
        return fail(HTTPStatus.INTERNAL_SERVER_ERROR, f"Failed to update LLM provider model: {str(e)}")


async def delete_llm_provider_model(provider_name: str, api: str, model: str):
    """Delete a specific model of a provider"""
    try:
        model_obj = await async_db_ops.query_llm_provider_model(provider_name, api, model)

        if not model_obj:
            return fail(
                HTTPStatus.NOT_FOUND, f"Model '{model}' for API '{api}' not found for provider '{provider_name}'"
            )

        # Soft delete the model
        await async_db_ops.delete_llm_provider_model(provider_name, api, model)

        return success(None)
    except Exception as e:
        return fail(HTTPStatus.INTERNAL_SERVER_ERROR, f"Failed to delete LLM provider model: {str(e)}")
