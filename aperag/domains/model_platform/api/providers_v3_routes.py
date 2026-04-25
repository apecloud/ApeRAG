from fastapi import APIRouter, Depends, Path, Response

from aperag.domains.identity.service.auth_dependencies import required_user
from aperag.domains.model_platform.ports import AuthenticatedUser
from aperag.domains.model_platform.schemas import (
    Model,
    ModelAccount,
    ModelAccountCreate,
    ModelAccountList,
    ModelAccountUpdate,
    ModelList,
    ModelProviderList,
    ModelUpdate,
    ModelUse,
    ModelUseList,
    ModelUseScenario,
    ModelUseUpdate,
    ModelValidationResponse,
)
from aperag.domains.model_platform.schemas import ModelCreate as ModelCreateSchema
from aperag.domains.model_platform.service.model_service import model_platform_service
from aperag.utils.audit_decorator import audit

router = APIRouter(tags=["model-platform"])


@router.get("/model-providers", response_model=ModelProviderList)
async def list_model_providers_view() -> ModelProviderList:
    return await model_platform_service.list_providers()


@router.get("/model-accounts", response_model=ModelAccountList)
async def list_model_accounts_view(user: AuthenticatedUser = Depends(required_user)) -> ModelAccountList:
    return await model_platform_service.list_accounts(str(user.id))


@router.post("/model-accounts", response_model=ModelAccount)
@audit(resource_type="model_account", api_name="CreateModelAccount")
async def create_model_account_view(
    body: ModelAccountCreate,
    user: AuthenticatedUser = Depends(required_user),
) -> ModelAccount:
    return await model_platform_service.create_account(str(user.id), body)


@router.put("/model-accounts/{account_id}", response_model=ModelAccount)
@audit(resource_type="model_account", api_name="UpdateModelAccount")
async def update_model_account_view(
    account_id: str,
    body: ModelAccountUpdate,
    user: AuthenticatedUser = Depends(required_user),
) -> ModelAccount:
    return await model_platform_service.update_account(account_id, str(user.id), body)


@router.delete("/model-accounts/{account_id}", status_code=204)
@audit(resource_type="model_account", api_name="DeleteModelAccount")
async def delete_model_account_view(account_id: str, user: AuthenticatedUser = Depends(required_user)) -> Response:
    await model_platform_service.update_account(account_id, str(user.id), ModelAccountUpdate(status="INACTIVE"))
    return Response(status_code=204)


@router.post("/model-accounts/{account_id}/validate", response_model=ModelValidationResponse)
async def validate_model_account_view(
    account_id: str, user: AuthenticatedUser = Depends(required_user)
) -> ModelValidationResponse:
    return await model_platform_service.validate_account(account_id, str(user.id))


@router.get("/model-accounts/{account_id}/models", response_model=ModelList)
async def list_model_account_models_view(
    account_id: str, user: AuthenticatedUser = Depends(required_user)
) -> ModelList:
    return await model_platform_service.list_models(str(user.id), account_id=account_id)


@router.get("/models", response_model=ModelList)
async def list_models_view(user: AuthenticatedUser = Depends(required_user)) -> ModelList:
    return await model_platform_service.list_models(str(user.id))


@router.post("/models", response_model=Model)
@audit(resource_type="model", api_name="CreateModel")
async def create_model_view(
    body: ModelCreateSchema,
    user: AuthenticatedUser = Depends(required_user),
) -> Model:
    return await model_platform_service.create_model(str(user.id), body)


@router.put("/models/{model_id}", response_model=Model)
@audit(resource_type="model", api_name="UpdateModel")
async def update_model_view(
    model_id: str,
    body: ModelUpdate,
    user: AuthenticatedUser = Depends(required_user),
) -> Model:
    return await model_platform_service.update_model(model_id, str(user.id), body)


@router.delete("/models/{model_id}", status_code=204)
@audit(resource_type="model", api_name="DeleteModel")
async def delete_model_view(model_id: str, user: AuthenticatedUser = Depends(required_user)) -> Response:
    await model_platform_service.update_model(model_id, str(user.id), ModelUpdate(status="INACTIVE"))
    return Response(status_code=204)


@router.post("/models/{model_id}/validate", response_model=ModelValidationResponse)
async def validate_model_view(
    model_id: str, user: AuthenticatedUser = Depends(required_user)
) -> ModelValidationResponse:
    return await model_platform_service.validate_model(model_id, str(user.id))


@router.get("/model-uses", response_model=ModelUseList)
async def list_model_uses_view(user: AuthenticatedUser = Depends(required_user)) -> ModelUseList:
    return await model_platform_service.list_model_uses(str(user.id))


@router.put("/model-uses/{scenario}", response_model=ModelUse)
@audit(resource_type="model_use", api_name="UpdateModelUse")
async def update_model_use_view(
    body: ModelUseUpdate,
    scenario: ModelUseScenario = Path(...),
    user: AuthenticatedUser = Depends(required_user),
) -> ModelUse:
    return await model_platform_service.update_model_use(str(user.id), scenario, body)
