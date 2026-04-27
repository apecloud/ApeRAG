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

from fastapi import APIRouter, Body, Depends, Path, Query, Response

from aperag.domains.model_platform.ports import AuthenticatedUser
from aperag.domains.model_platform.schemas import (
    DefaultModelsResponse,
    DefaultModelsUpdateRequest,
    LlmConfigurationResponse,
    LlmProvider,
    LlmProviderCreateWithApiKey,
    LlmProviderModel,
    LlmProviderModelCreateRequest,
    LlmProviderModelList,
    LlmProviderModelUpdate,
    LlmProviderUpdateWithApiKey,
    ModelConfigList,
    TagFilterRequest,
)
from aperag.domains.model_platform.service.default_model_service import default_model_service
from aperag.domains.model_platform.service.llm_available_model_service import llm_available_model_service
from aperag.domains.model_platform.service.llm_provider_service import (
    create_llm_provider,
    create_llm_provider_model,
    delete_llm_provider,
    delete_llm_provider_model,
    get_llm_configuration,
    get_llm_provider,
    list_llm_provider_models,
    publish_llm_provider,
    update_llm_provider,
    update_llm_provider_model,
)
from aperag.exceptions import PermissionDeniedError
from aperag.utils.audit_decorator import audit
from aperag.views.auth import required_user

router = APIRouter(tags=["providers-v2"])


@router.get("/providers/configuration", response_model=LlmConfigurationResponse)
async def get_provider_configuration_view(user: AuthenticatedUser = Depends(required_user)) -> LlmConfigurationResponse:
    """Return providers and configured models visible to the current user."""

    return await get_llm_configuration(str(user.id), user.role == "admin")


@router.post("/providers/available-models", response_model=ModelConfigList)
async def get_available_provider_models_view(
    body: TagFilterRequest | None = Body(default=None),
    user: AuthenticatedUser = Depends(required_user),
) -> ModelConfigList:
    """Return provider catalog models, optionally filtered by tags."""

    return await llm_available_model_service.get_available_models(str(user.id), body or TagFilterRequest())


@router.get("/default-models", response_model=DefaultModelsResponse)
async def get_default_models_view(user: AuthenticatedUser = Depends(required_user)) -> DefaultModelsResponse:
    """Return default model selections for each product scenario."""

    return await default_model_service.get_default_models(str(user.id))


@router.put("/default-models", response_model=DefaultModelsResponse)
async def update_default_models_view(
    body: DefaultModelsUpdateRequest,
    user: AuthenticatedUser = Depends(required_user),
) -> DefaultModelsResponse:
    """Update default model selections for each product scenario."""

    return await default_model_service.update_default_models(str(user.id), body)


@router.post("/providers", response_model=LlmProvider)
@audit(resource_type="llm_provider", api_name="CreateLLMProviderV2")
async def create_provider_view(
    body: LlmProviderCreateWithApiKey,
    user: AuthenticatedUser = Depends(required_user),
) -> LlmProvider:
    """Create a provider owned by the current user."""

    return await create_llm_provider(body.model_dump(), str(user.id), user.role == "admin")


@router.get("/providers/{provider_name}", response_model=LlmProvider)
async def get_provider_view(
    provider_name: str,
    user: AuthenticatedUser = Depends(required_user),
) -> LlmProvider:
    """Return one provider visible to the current user."""

    return await get_llm_provider(provider_name, str(user.id), user.role == "admin")


@router.put("/providers/{provider_name}", response_model=LlmProvider)
@audit(resource_type="llm_provider", api_name="UpdateLLMProviderV2")
async def update_provider_view(
    provider_name: str,
    body: LlmProviderUpdateWithApiKey,
    user: AuthenticatedUser = Depends(required_user),
) -> LlmProvider:
    """Update a provider owned by the current user."""

    return await update_llm_provider(provider_name, body.model_dump(), str(user.id), user.role == "admin")


@router.delete("/providers/{provider_name}", status_code=204)
@audit(resource_type="llm_provider", api_name="DeleteLLMProviderV2")
async def delete_provider_view(
    provider_name: str,
    user: AuthenticatedUser = Depends(required_user),
) -> Response:
    """Delete a provider owned by the current user."""

    await delete_llm_provider(provider_name, str(user.id), user.role == "admin")
    return Response(status_code=204)


@router.post("/providers/{provider_name}/publish", response_model=LlmProvider)
@audit(resource_type="llm_provider", api_name="PublishLLMProviderV2")
async def publish_provider_view(
    provider_name: str,
    user: AuthenticatedUser = Depends(required_user),
) -> LlmProvider:
    """Publish a private provider to the public catalog. Admin only."""

    if user.role != "admin":
        raise PermissionDeniedError("Only admin can publish provider to public")
    return await publish_llm_provider(provider_name, str(user.id))


@router.get("/provider-models", response_model=LlmProviderModelList)
async def list_provider_models_view(
    provider_name: str | None = Query(default=None),
    user: AuthenticatedUser = Depends(required_user),
) -> LlmProviderModelList:
    """Return models for all visible providers, optionally filtered by provider."""

    return await list_llm_provider_models(
        provider_name=provider_name, user_id=str(user.id), is_admin=user.role == "admin"
    )


@router.get("/providers/{provider_name}/models", response_model=LlmProviderModelList)
async def list_provider_models_by_provider_view(
    provider_name: str,
    user: AuthenticatedUser = Depends(required_user),
) -> LlmProviderModelList:
    """Return models for one provider visible to the current user."""

    return await list_llm_provider_models(
        provider_name=provider_name, user_id=str(user.id), is_admin=user.role == "admin"
    )


@router.post("/providers/{provider_name}/models", response_model=LlmProviderModel)
@audit(resource_type="llm_provider_model", api_name="CreateProviderModelV2")
async def create_provider_model_view(
    provider_name: str,
    body: LlmProviderModelCreateRequest,
    user: AuthenticatedUser = Depends(required_user),
) -> LlmProviderModel:
    """Create a model under a provider owned by the current user."""

    return await create_llm_provider_model(provider_name, body.model_dump(), str(user.id), user.role == "admin")


@router.put("/providers/{provider_name}/models/{api}/{model:path}", response_model=LlmProviderModel)
@audit(resource_type="llm_provider_model", api_name="UpdateProviderModelV2")
async def update_provider_model_view(
    provider_name: str,
    api: str,
    body: LlmProviderModelUpdate,
    model: str = Path(..., description="Model name. Supports names with slashes."),
    user: AuthenticatedUser = Depends(required_user),
) -> LlmProviderModel:
    """Update a model under a provider owned by the current user."""

    return await update_llm_provider_model(
        provider_name,
        api,
        model,
        body.model_dump(exclude_unset=True),
        str(user.id),
        user.role == "admin",
    )


@router.delete("/providers/{provider_name}/models/{api}/{model:path}", status_code=204)
@audit(resource_type="llm_provider_model", api_name="DeleteProviderModelV2")
async def delete_provider_model_view(
    provider_name: str,
    api: str,
    model: str = Path(..., description="Model name. Supports names with slashes."),
    user: AuthenticatedUser = Depends(required_user),
) -> Response:
    """Delete a model under a provider owned by the current user."""

    await delete_llm_provider_model(provider_name, api, model, str(user.id), user.role == "admin")
    return Response(status_code=204)
