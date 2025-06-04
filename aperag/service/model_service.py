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

from sqlalchemy.ext.asyncio import AsyncSession

from aperag.config import get_session, settings
from aperag.db import models as db_models
from aperag.db.ops import DatabaseOps, db_ops
from aperag.schema import view_models
from aperag.schema.view_models import ModelServiceProvider, ModelServiceProviderList
from aperag.views.utils import fail, success


class ModelServiceProviderService:
    """Model Service Provider service that handles business logic for MSPs"""

    def __init__(self, session: AsyncSession = None):
        # Use global db_ops instance by default, or create custom one with provided session
        if session is None:
            self.db_ops = db_ops  # Use global instance
            self._custom_session = None
        else:
            self.db_ops = DatabaseOps(session)  # Create custom instance for transaction control
            self._custom_session = session

    async def _execute_with_session(self, operation):
        """Execute operation with proper session management for write operations"""
        if self._custom_session:
            # Use provided session
            return await operation(self._custom_session)
        else:
            # Create new session for this operation
            async for session in get_session():
                try:
                    result = await operation(session)
                    await session.commit()
                    return result
                except Exception:
                    await session.rollback()
                    raise

    def build_model_service_provider_response(
        self, msp: db_models.ModelServiceProvider, supported_msp: view_models.ModelServiceProvider
    ) -> view_models.ModelServiceProvider:
        """Build ModelServiceProvider response object for API return."""
        return view_models.ModelServiceProvider(
            name=msp.name,
            dialect=msp.dialect,
            label=supported_msp.label,
            allow_custom_base_url=supported_msp.allow_custom_base_url,
            base_url=msp.base_url,
            api_key=msp.api_key,
        )

    async def list_model_service_providers(self, user: str) -> view_models.ModelServiceProviderList:
        supported_msp_dict = {msp["name"]: ModelServiceProvider(**msp) for msp in settings.model_configs}
        msp_list = await self.db_ops.query_msp_list(user)
        response = []
        for msp in msp_list:
            if msp.name in supported_msp_dict:
                supported_msp = supported_msp_dict[msp.name]
                response.append(self.build_model_service_provider_response(msp, supported_msp))
        return success(ModelServiceProviderList(items=response))

    async def update_model_service_provider(
        self,
        user: str,
        provider: str,
        mspIn: view_models.ModelServiceProviderUpdate,
        supported_providers: List[view_models.ModelServiceProvider],
    ):
        supported_msp_names = {provider.name for provider in supported_providers if provider.name}
        if provider not in supported_msp_names:
            return fail(HTTPStatus.BAD_REQUEST, f"unsupported model service provider {provider}")

        msp_config = next(item for item in supported_providers if item.name == provider)
        if not msp_config.allow_custom_base_url and mspIn.base_url is not None:
            return fail(HTTPStatus.BAD_REQUEST, f"model service provider {provider} does not support setting base_url")

        async def _update_operation(session):
            db_ops_session = DatabaseOps(session)
            msp = await db_ops_session.query_msp(user, provider, filterDeletion=False)

            if msp is None:
                msp = db_models.ModelServiceProvider(
                    user=user,
                    name=provider,
                    dialect=msp_config.dialect,
                    api_key=mspIn.api_key,
                    base_url=mspIn.base_url if msp_config.allow_custom_base_url else msp_config.base_url,
                    extra=mspIn.extra,
                    status=db_models.ModelServiceProviderStatus.ACTIVE,
                )
            else:
                if msp.status == db_models.ModelServiceProviderStatus.DELETED:
                    msp.status = db_models.ModelServiceProviderStatus.ACTIVE
                    msp.gmt_deleted = None
                msp.dialect = msp.dialect
                msp.api_key = mspIn.api_key
                if msp_config.allow_custom_base_url and mspIn.base_url is not None:
                    msp.base_url = mspIn.base_url
                msp.extra = mspIn.extra

            session.add(msp)
            await session.flush()
            await session.refresh(msp)
            return {}

        try:
            result = await self._execute_with_session(_update_operation)
            return success(result)
        except Exception as e:
            return fail(HTTPStatus.INTERNAL_SERVER_ERROR, f"Failed to update model service provider: {str(e)}")

    async def delete_model_service_provider(self, user: str, provider: str):
        supported_msp_names = {item["name"] for item in settings.model_configs}
        if provider not in supported_msp_names:
            return fail(HTTPStatus.BAD_REQUEST, f"unsupported model service provider {provider}")

        # First check if MSP exists
        msp = await self.db_ops.query_msp(user, provider)
        if msp is None:
            return fail(HTTPStatus.NOT_FOUND, f"model service provider {provider} not found")

        async def _delete_operation(session):
            db_ops_session = DatabaseOps(session)
            msp = await db_ops_session.query_msp(user, provider)
            if msp is None:
                raise ValueError(f"model service provider {provider} not found")

            msp.status = db_models.ModelServiceProviderStatus.DELETED
            msp.gmt_deleted = datetime.utcnow()
            session.add(msp)
            await session.flush()
            await session.refresh(msp)
            return {}

        try:
            result = await self._execute_with_session(_delete_operation)
            return success(result)
        except ValueError as e:
            return fail(HTTPStatus.NOT_FOUND, str(e))
        except Exception as e:
            return fail(HTTPStatus.INTERNAL_SERVER_ERROR, f"Failed to delete model service provider: {str(e)}")

    async def list_available_models(self, user: str) -> view_models.ModelConfigList:
        from aperag.schema.view_models import ModelConfig, ModelConfigList

        supported_providers = [ModelConfig(**msp) for msp in settings.model_configs]
        supported_msp_dict = {provider.name: provider for provider in supported_providers}
        msp_list = await self.db_ops.query_msp_list(user)
        available_providers = []
        for msp in msp_list:
            if msp.name in supported_msp_dict:
                available_providers.append(supported_msp_dict[msp.name])
        return success(ModelConfigList(items=available_providers).model_dump(exclude_none=True))

    async def list_supported_model_service_providers(self) -> view_models.ModelServiceProviderList:
        response = []
        for supported_msp in settings.model_configs:
            provider = ModelServiceProvider(
                name=supported_msp["name"],
                dialect=supported_msp["dialect"],
                label=supported_msp["label"],
                allow_custom_base_url=supported_msp["allow_custom_base_url"],
                base_url=supported_msp["base_url"],
            )
            response.append(provider)
        return success(ModelServiceProviderList(items=response))


# Create a global service instance for easy access
# This uses the global db_ops instance and doesn't require session management in views
model_service_provider_service = ModelServiceProviderService()
