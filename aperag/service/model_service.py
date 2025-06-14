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

from datetime import datetime
from http import HTTPStatus
from typing import List, Optional

import aperag.db.models as db_models
from aperag.db.ops import async_db_ops
from aperag.schema import view_models
from aperag.service.llm_config_service import get_model_config_objects, get_model_configs, get_supported_provider_names
from aperag.views.utils import fail, success


async def build_model_service_provider_response(
    msp: db_models.ModelServiceProvider, supported_msp: view_models.ModelServiceProvider
) -> view_models.ModelServiceProvider:
    """Build ModelServiceProvider response object for API return."""
    return view_models.ModelServiceProvider(
        name=msp.name,
        label=supported_msp.label,
        api_key=msp.api_key,
    )


async def list_model_service_providers(user: str) -> view_models.ModelServiceProviderList:
    model_configs = await get_model_configs()
    supported_msp_dict = {msp["name"]: view_models.ModelServiceProvider(**msp) for msp in model_configs}
    msp_list = await async_db_ops.query_msp_list(user)
    response = []
    for msp in msp_list:
        if msp.name in supported_msp_dict:
            supported_msp = supported_msp_dict[msp.name]
            response.append(await build_model_service_provider_response(msp, supported_msp))
    return success(view_models.ModelServiceProviderList(items=response))


async def update_model_service_provider(
    user: str,
    provider: str,
    mspIn: view_models.ModelServiceProviderUpdate,
    supported_providers: List[view_models.ModelServiceProvider],
):
    supported_msp_names = {provider.name for provider in supported_providers if provider.name}
    if provider not in supported_msp_names:
        return fail(HTTPStatus.BAD_REQUEST, f"unsupported model service provider {provider}")

    # Only handle api_key for ModelServiceProvider
    msp = await async_db_ops.query_msp(user, provider, filterDeletion=False)
    if msp is None:
        msp = db_models.ModelServiceProvider(
            user=user,
            name=provider,
            api_key=mspIn.api_key,
            status=db_models.ModelServiceProviderStatus.ACTIVE,
        )
    else:
        if msp.status == db_models.ModelServiceProviderStatus.DELETED:
            msp.status = db_models.ModelServiceProviderStatus.ACTIVE
            msp.gmt_deleted = None
        msp.api_key = mspIn.api_key
    await async_db_ops.update_msp(msp)
    return success({})


async def delete_model_service_provider(user: str, provider: str):
    supported_msp_names = await get_supported_provider_names()
    if provider not in supported_msp_names:
        return fail(HTTPStatus.BAD_REQUEST, f"unsupported model service provider {provider}")
    msp = await async_db_ops.query_msp(user, provider)
    if msp is None:
        return fail(HTTPStatus.NOT_FOUND, f"model service provider {provider} not found")
    msp.status = db_models.ModelServiceProviderStatus.DELETED
    msp.gmt_deleted = datetime.utcnow()
    await async_db_ops.delete_msp(msp)
    return success({})


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


async def get_available_models(
    user: str, tag_filter_request: view_models.TagFilterRequest
) -> view_models.ModelConfigList:
    """Get available models with optional tag filtering"""
    supported_providers = await get_model_config_objects()
    supported_msp_dict = {provider.name: provider for provider in supported_providers}
    msp_list = await async_db_ops.query_msp_list(user)
    available_providers = []
    for msp in msp_list:
        if msp.name in supported_msp_dict:
            available_providers.append(supported_msp_dict[msp.name])

    # Apply tag filtering based on request
    if tag_filter_request.tag_filters is None or len(tag_filter_request.tag_filters) == 0:
        default_filter = None
        filtered_providers = _filter_providers_by_tags(available_providers, default_filter)
    else:
        filtered_providers = _filter_providers_by_tags(available_providers, tag_filter_request.tag_filters)

    return success(view_models.ModelConfigList(items=filtered_providers, pageResult=None).model_dump(exclude_none=True))


def _filter_providers_by_tags(
    providers: List[view_models.ModelConfig], tag_filters: List[view_models.TagFilterCondition]
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


async def list_supported_model_service_providers() -> view_models.ModelServiceProviderList:
    model_configs = await get_model_configs()
    response = []
    for supported_msp in model_configs:
        provider = view_models.ModelServiceProvider(
            name=supported_msp["name"],
            label=supported_msp["label"],
        )
        response.append(provider)
    return success(view_models.ModelServiceProviderList(items=response))
