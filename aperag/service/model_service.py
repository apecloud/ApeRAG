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
from typing import List

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


async def list_available_models(user: str) -> view_models.ModelConfigList:
    supported_providers = await get_model_config_objects()
    supported_msp_dict = {provider.name: provider for provider in supported_providers}
    msp_list = await async_db_ops.query_msp_list(user)
    available_providers = []
    for msp in msp_list:
        if msp.name in supported_msp_dict:
            available_providers.append(supported_msp_dict[msp.name])
    return success(
        view_models.ModelConfigList(items=available_providers, pageResult=None).model_dump(exclude_none=True)
    )


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
