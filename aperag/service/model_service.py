from http import HTTPStatus
from typing import List
from aperag.schema.view_models import ModelServiceProviderList, ModelServiceProvider
from aperag.views.utils import fail, success
from config import settings
from aperag.db.ops import query_msp, query_msp_list
from aperag.db import models as db_models
from django.utils import timezone
from aperag.schema import view_models

def build_model_service_provider_response(msp: db_models.ModelServiceProvider, supported_msp: view_models.ModelServiceProvider) -> view_models.ModelServiceProvider:
    """Build ModelServiceProvider response object for API return."""
    return view_models.ModelServiceProvider(
        name=msp.name,
        dialect=msp.dialect,
        label=supported_msp.label,
        allow_custom_base_url=supported_msp.allow_custom_base_url,
        base_url=msp.base_url,
        api_key=msp.api_key,
    )

async def list_model_service_providers(user: str) -> view_models.ModelServiceProviderList:
    supported_msp_dict = {msp["name"]: ModelServiceProvider(**msp)
                          for msp in settings.MODEL_CONFIGS}
    msp_list = await query_msp_list(user)
    response = []
    for msp in msp_list:
        if msp.name in supported_msp_dict:
            supported_msp = supported_msp_dict[msp.name]
            response.append(build_model_service_provider_response(msp, supported_msp))
    return success(ModelServiceProviderList(items=response))

async def update_model_service_provider(user: str, provider: str, mspIn: view_models.ModelServiceProviderUpdate, supported_providers: List[view_models.ModelServiceProvider]):
    supported_msp_names = {provider.name for provider in supported_providers if provider.name}
    if provider not in supported_msp_names:
        return fail(HTTPStatus.BAD_REQUEST, f"unsupported model service provider {provider}")
    msp_config = next(item for item in supported_providers if item.name == provider)
    if not msp_config.allow_custom_base_url and mspIn.base_url is not None:
        return fail(HTTPStatus.BAD_REQUEST, f"model service provider {provider} does not support setting base_url")
    msp = await query_msp(user, provider, filterDeletion=False)
    if msp is None:
        msp = db_models.ModelServiceProvider(
            user=user,
            name=provider,
            dialect=msp_config.dialect,
            api_key=mspIn.api_key,
            base_url=mspIn.base_url if msp_config.allow_custom_base_url else msp_config.base_url,
            extra=mspIn.extra,
            status=db_models.ModelServiceProvider.Status.ACTIVE,
        )
    else:
        if msp.status == db_models.ModelServiceProvider.Status.DELETED:
            msp.status = db_models.ModelServiceProvider.Status.ACTIVE
            msp.gmt_deleted = None
        msp.dialect = msp.dialect
        msp.api_key = mspIn.api_key
        if (msp_config.allow_custom_base_url and mspIn.base_url is not None):
            msp.base_url = mspIn.base_url
        msp.extra = mspIn.extra
    await msp.asave()
    return success({})

async def delete_model_service_provider(user: str, provider: str):
    supported_msp_names = {item["name"] for item in settings.MODEL_CONFIGS}
    if provider not in supported_msp_names:
        return fail(HTTPStatus.BAD_REQUEST, f"unsupported model service provider {provider}")
    msp = await query_msp(user, provider)
    if msp is None:
        return fail(HTTPStatus.NOT_FOUND, f"model service provider {provider} not found")
    msp.status = db_models.ModelServiceProvider.Status.DELETED
    msp.gmt_deleted = timezone.now()
    await msp.asave()
    return success({})

async def list_available_models(user: str) -> view_models.ModelConfigList:
    from aperag.schema.view_models import ModelConfig, ModelConfigList
    supported_providers = [ModelConfig(**msp) for msp in settings.MODEL_CONFIGS]
    supported_msp_dict = {provider.name: provider for provider in supported_providers}
    msp_list = await query_msp_list(user)
    available_providers = []
    for msp in msp_list:
        if msp.name in supported_msp_dict:
            available_providers.append(supported_msp_dict[msp.name])
    return success(ModelConfigList(items=available_providers, pageResult=None).model_dump(exclude_none=True))

async def list_supported_model_service_providers() -> view_models.ModelServiceProviderList:
    response = []
    for supported_msp in settings.MODEL_CONFIGS:
        provider = ModelServiceProvider(
            name=supported_msp["name"],
            dialect=supported_msp["dialect"],
            label=supported_msp["label"],
            allow_custom_base_url=supported_msp["allow_custom_base_url"],
            base_url=supported_msp["base_url"],
        )
        response.append(provider)
    return success(ModelServiceProviderList(items=response))
