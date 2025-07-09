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

from typing import Optional

from aperag.db.ops import async_db_ops
from aperag.exceptions import PermissionDeniedError, ResourceNotFoundException, invalid_param
from aperag.views.utils import generate_random_provider_name, mask_api_key

# Constants
PUBLIC_USER_ID = "public"


def _can_access_provider(user_id: str, is_admin: bool, target_user_id: str) -> bool:
    """Check if user can access a provider based on ownership

    Args:
        user_id: Current user ID
        is_admin: Whether current user is admin
        target_user_id: The user_id field of the provider

    Returns:
        True if user can access the provider
    """
    if is_admin:
        return target_user_id in (PUBLIC_USER_ID, user_id)
    else:
        return target_user_id == user_id


def _check_edit_permission(user_id: str, is_admin: bool, target_user_id: str = None, provider_name: str = None):
    """Check if user has permission to edit the resource

    Args:
        user_id: Current user ID
        is_admin: Whether current user is admin
        target_user_id: The user_id field of the resource being edited
        provider_name: Provider name for error message

    Raises:
        PermissionDeniedError: If user doesn't have permission
    """
    if not _can_access_provider(user_id, is_admin, target_user_id):
        if target_user_id == PUBLIC_USER_ID:
            error_msg = "You don't have permission to edit public provider"
        else:
            error_msg = "You don't have permission to edit another user's provider"

        if provider_name:
            error_msg += f" '{provider_name}'"

        raise PermissionDeniedError(error_msg)


async def get_llm_configuration(user_id: str, is_admin: bool = False):
    """Get complete LLM configuration including providers and models

    Args:
        user_id: User ID requesting the configuration
        is_admin: Whether the user is an admin
    """
    # Admin can see public + their own providers, non-admin can only see their own
    need_public = is_admin
    providers = await async_db_ops.query_llm_providers(user_id=user_id, need_public=need_public)

    providers_data = []
    provider_names = []

    for provider in providers:
        provider_data = {
            "name": provider.name,
            "user_id": provider.user_id,
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

        # Add masked API key (providers already filtered by access permissions)
        api_key = await async_db_ops.query_provider_api_key(provider.name, user_id)
        if api_key:
            provider_data["api_key"] = mask_api_key(api_key)

        providers_data.append(provider_data)
        provider_names.append(provider.name)

    # Get models only for the providers user has access to
    models = await async_db_ops.query_llm_provider_models_by_provider_list(provider_names)
    models_data = []

    for model in models:
        models_data.append(
            {
                "provider_name": model.provider_name,
                "api": model.api,
                "model": model.model,
                "custom_llm_provider": model.custom_llm_provider,
                "context_window": model.context_window,
                "max_input_tokens": model.max_input_tokens,
                "max_output_tokens": model.max_output_tokens,
                "tags": model.tags or [],
                "created": model.gmt_created,
                "updated": model.gmt_updated,
            }
        )

    return {
        "providers": providers_data,
        "models": models_data,
    }


async def create_llm_provider(provider_data: dict, user_id: str, is_admin: bool = False):
    """Create a new LLM provider or restore a soft-deleted one with the same name

    Args:
        provider_data: Provider configuration data
        user_id: User ID creating the provider
        is_admin: Whether the user is an admin
    """
    # Generate a random provider name if not provided
    if "name" not in provider_data or not provider_data["name"]:
        # Generate random name and ensure it doesn't conflict
        max_attempts = 10
        for _ in range(max_attempts):
            generated_name = generate_random_provider_name()
            existing = await async_db_ops.query_llm_provider_by_name(generated_name)
            if not existing:
                provider_data["name"] = generated_name
                break
        else:
            raise Exception("Failed to generate unique provider name")

    # First check if there's an active provider with the same name
    active_existing = await async_db_ops.query_llm_provider_by_name(provider_data["name"])

    if active_existing:
        raise invalid_param("name", f"Provider with name '{provider_data['name']}' already exists")

    # Try to restore a soft-deleted provider if it exists
    provider = await async_db_ops.restore_llm_provider(provider_data["name"])

    if provider:
        # Update the restored provider with new data
        provider = await async_db_ops.update_llm_provider(
            name=provider_data["name"],
            user_id=user_id,
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
            user_id=user_id,
            label=provider_data["label"],
            completion_dialect=provider_data.get("completion_dialect", "openai"),
            embedding_dialect=provider_data.get("embedding_dialect", "openai"),
            rerank_dialect=provider_data.get("rerank_dialect", "jina_ai"),
            allow_custom_base_url=provider_data.get("allow_custom_base_url", False),
            base_url=provider_data["base_url"],
            extra=provider_data.get("extra"),
        )

    # Handle status parameter for enable/disable functionality
    status = provider_data.get("status")
    if status == "enable":
        # Enable: Create or update API key if provided
        api_key = provider_data.get("api_key")
        if api_key and api_key.strip():
            await async_db_ops.upsert_msp(name=provider_data["name"], api_key=api_key)
    elif status == "disable":
        # Disable: Delete API key (MSP) for this provider
        await async_db_ops.delete_msp_by_name(provider_data["name"])
    else:
        # Legacy behavior: Handle API key if provided (for backward compatibility)
        api_key = provider_data.get("api_key")
        if api_key and api_key.strip():  # Only create/update if non-empty API key is provided
            # Create or update API key for this provider
            await async_db_ops.upsert_msp(name=provider_data["name"], api_key=api_key)

    return {
        "name": provider.name,
        "user_id": provider.user_id,
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


async def get_llm_provider(provider_name: str, user_id: str, is_admin: bool = False):
    """Get a specific LLM provider by name

    Args:
        provider_name: Name of the provider to get
        user_id: User ID requesting the provider
        is_admin: Whether the user is an admin
    """
    provider = await async_db_ops.query_llm_provider_by_name(provider_name)
    if provider and not _can_access_provider(user_id, is_admin, provider.user_id):
        raise PermissionDeniedError(f"You don't have permission to access provider '{provider_name}'")

    if not provider:
        raise ResourceNotFoundException("Provider", provider_name)

    provider_data = {
        "name": provider.name,
        "user_id": provider.user_id,
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

    # Get masked API key (access already verified above)
    api_key = await async_db_ops.query_provider_api_key(provider_name, user_id)
    if api_key:
        provider_data["api_key"] = mask_api_key(api_key)

    return provider_data


async def update_llm_provider(provider_name: str, update_data: dict, user_id: str, is_admin: bool = False):
    """Update an existing LLM provider

    Args:
        provider_name: Name of the provider to update
        update_data: Data to update
        user_id: User ID making the update
        is_admin: Whether the user is an admin
    """
    existing_provider = await async_db_ops.query_llm_provider_by_name(provider_name)

    if not existing_provider:
        raise ResourceNotFoundException("Provider", provider_name)

    # Check edit permission
    _check_edit_permission(user_id, is_admin, existing_provider.user_id, provider_name)

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

    # Handle status parameter for enable/disable functionality
    status = update_data.get("status")
    if status == "enable":
        # Enable: Create or update API key if provided
        api_key = update_data.get("api_key")
        if api_key and api_key.strip():
            await async_db_ops.upsert_msp(name=provider_name, api_key=api_key)
    elif status == "disable":
        # Disable: Delete API key (MSP) for this provider
        await async_db_ops.delete_msp_by_name(provider_name)
    else:
        # Legacy behavior: Handle API key if provided (for backward compatibility)
        api_key = update_data.get("api_key")
        if api_key and api_key.strip():
            await async_db_ops.upsert_msp(name=provider_name, api_key=api_key)

    return {
        "name": provider.name,
        "user_id": provider.user_id,
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


async def delete_llm_provider(provider_name: str, user_id: str, is_admin: bool = False) -> Optional[bool]:
    """Delete an LLM provider (soft delete, idempotent operation)

    Args:
        provider_name: Name of the provider to delete
        user_id: User ID making the deletion
        is_admin: Whether the user is an admin

    Returns:
        True if deleted, None if already deleted/not found
    """
    provider = await async_db_ops.query_llm_provider_by_name(provider_name)

    if not provider:
        return None  # Idempotent operation, not found is success

    # Check edit permission
    _check_edit_permission(user_id, is_admin, provider.user_id, provider_name)

    # Soft delete the provider and its models
    await async_db_ops.delete_llm_provider(provider_name)

    # Physical delete the API key for this provider
    await async_db_ops.delete_msp_by_name(provider_name)

    return True


async def list_llm_provider_models(provider_name: Optional[str] = None, user_id: str = None, is_admin: bool = False):
    """List LLM provider models, optionally filtered by provider

    Args:
        provider_name: Optional provider name to filter by
        user_id: User ID requesting the models
        is_admin: Whether the user is an admin

    Returns:
        Dict with models list and page result
    """
    if provider_name:
        # Check if user can access this specific provider
        provider = await async_db_ops.query_llm_provider_by_name(provider_name)
        if not provider:
            raise ResourceNotFoundException("Provider", provider_name)

        # Check access permission
        if not _can_access_provider(user_id, is_admin, provider.user_id):
            raise PermissionDeniedError(f"You don't have permission to access provider '{provider_name}'")

        # Get models for specific provider
        models = await async_db_ops.query_llm_provider_models(provider_name)
    else:
        # Get all accessible providers first
        need_public = is_admin
        providers = await async_db_ops.query_llm_providers(user_id=user_id, need_public=need_public)
        provider_names = [p.name for p in providers]

        # Get models only for accessible providers
        models = await async_db_ops.query_llm_provider_models_by_provider_list(provider_names)

    # Format models data
    models_data = []
    for model in models:
        models_data.append(
            {
                "provider_name": model.provider_name,
                "api": model.api,
                "model": model.model,
                "custom_llm_provider": model.custom_llm_provider,
                "context_window": model.context_window,
                "max_input_tokens": model.max_input_tokens,
                "max_output_tokens": model.max_output_tokens,
                "tags": model.tags or [],
                "created": model.gmt_created,
                "updated": model.gmt_updated,
            }
        )

    return {"items": models_data, "pageResult": None}


async def create_llm_provider_model(provider_name: str, model_data: dict, user_id: str, is_admin: bool = False):
    """Create a new model for a specific provider or restore a soft-deleted one with the same combination

    Args:
        provider_name: Name of the provider
        model_data: Model configuration data
        user_id: User ID creating the model
        is_admin: Whether the user is an admin
    """
    # Check if provider exists
    provider = await async_db_ops.query_llm_provider_by_name(provider_name)

    if not provider:
        raise ResourceNotFoundException("Provider", provider_name)

    # Check edit permission for the provider
    _check_edit_permission(user_id, is_admin, provider.user_id, provider_name)

    # First check if there's an active model with the same combination
    active_existing = await async_db_ops.query_llm_provider_model(provider_name, model_data["api"], model_data["model"])

    if active_existing:
        raise invalid_param(
            "model",
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
            context_window=model_data.get("context_window"),
            max_input_tokens=model_data.get("max_input_tokens"),
            max_output_tokens=model_data.get("max_output_tokens"),
            tags=model_data.get("tags", []),
        )
    else:
        # Create new model
        model = await async_db_ops.create_llm_provider_model(
            provider_name=provider_name,
            api=model_data["api"],
            model=model_data["model"],
            custom_llm_provider=model_data["custom_llm_provider"],
            context_window=model_data.get("context_window"),
            max_input_tokens=model_data.get("max_input_tokens"),
            max_output_tokens=model_data.get("max_output_tokens"),
            tags=model_data.get("tags", []),
        )

    return {
        "provider_name": model.provider_name,
        "api": model.api,
        "model": model.model,
        "custom_llm_provider": model.custom_llm_provider,
        "context_window": model.context_window,
        "max_input_tokens": model.max_input_tokens,
        "max_output_tokens": model.max_output_tokens,
        "tags": model.tags or [],
        "created": model.gmt_created,
        "updated": model.gmt_updated,
    }


async def update_llm_provider_model(
    provider_name: str, api: str, model: str, update_data: dict, user_id: str, is_admin: bool = False
):
    """Update a specific model of a provider

    Args:
        provider_name: Name of the provider
        api: API type of the model
        model: Model name
        update_data: Data to update
        user_id: User ID making the update
        is_admin: Whether the user is an admin
    """
    existing_model = await async_db_ops.query_llm_provider_model(provider_name, api, model)

    if not existing_model:
        raise ResourceNotFoundException(f"Model '{model}' for API '{api}'", f"provider '{provider_name}'")

    # Check if provider exists and permission to edit
    provider = await async_db_ops.query_llm_provider_by_name(provider_name)
    if not provider:
        raise ResourceNotFoundException("Provider", provider_name)

    # Check edit permission for the provider
    _check_edit_permission(user_id, is_admin, provider.user_id, provider_name)

    # Update model using the DatabaseOps method
    model_obj = await async_db_ops.update_llm_provider_model(
        provider_name=provider_name,
        api=api,
        model=model,
        custom_llm_provider=update_data.get("custom_llm_provider"),
        context_window=update_data.get("context_window"),
        max_input_tokens=update_data.get("max_input_tokens"),
        max_output_tokens=update_data.get("max_output_tokens"),
        tags=update_data.get("tags"),
    )

    return {
        "provider_name": model_obj.provider_name,
        "api": model_obj.api,
        "model": model_obj.model,
        "custom_llm_provider": model_obj.custom_llm_provider,
        "context_window": model_obj.context_window,
        "max_input_tokens": model_obj.max_input_tokens,
        "max_output_tokens": model_obj.max_output_tokens,
        "tags": model_obj.tags or [],
        "created": model_obj.gmt_created,
        "updated": model_obj.gmt_updated,
    }


async def delete_llm_provider_model(
    provider_name: str, api: str, model: str, user_id: str, is_admin: bool = False
) -> Optional[bool]:
    """Delete a specific model of a provider (idempotent operation)

    Args:
        provider_name: Name of the provider
        api: API type of the model
        model: Model name
        user_id: User ID making the deletion
        is_admin: Whether the user is an admin

    Returns:
        True if deleted, None if already deleted/not found
    """
    model_obj = await async_db_ops.query_llm_provider_model(provider_name, api, model)

    if not model_obj:
        return None  # Idempotent operation, not found is success

    # Check if provider exists and permission to edit
    provider = await async_db_ops.query_llm_provider_by_name(provider_name)
    if not provider:
        raise ResourceNotFoundException("Provider", provider_name)

    # Check edit permission for the provider
    _check_edit_permission(user_id, is_admin, provider.user_id, provider_name)

    # Soft delete the model
    await async_db_ops.delete_llm_provider_model(provider_name, api, model)

    return True
