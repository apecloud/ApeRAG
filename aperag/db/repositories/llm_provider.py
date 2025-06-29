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

from sqlalchemy import select

from aperag.db.models import (
    LLMProvider,
    LLMProviderModel,
    ModelServiceProvider,
    ModelServiceProviderStatus,
)
from aperag.db.repositories.base import (
    AsyncRepositoryProtocol,
    SyncRepositoryProtocol,
)
from aperag.utils.utils import utc_now


class LlmProviderRepositoryMixin(SyncRepositoryProtocol):
    def query_llm_provider_by_name(self, name: str, user_id: str = None):
        def _query(session):
            stmt = select(LLMProvider).where(LLMProvider.name == name, LLMProvider.gmt_deleted.is_(None))
            if user_id:
                # Get both public providers and user's private providers
                stmt = stmt.where((LLMProvider.user_id == "public") | (LLMProvider.user_id == user_id))
            result = session.execute(stmt)
            return result.scalars().first()

        return self._execute_query(_query)

    def query_provider_api_key(self, provider_name: str, user_id: str = None, need_public: bool = True) -> str:
        """Query provider API key with user access control using single SQL JOIN (sync version)

        Args:
            provider_name: Provider name to query
            user_id: User ID for private provider access
            need_public: Whether to include public providers

        Returns:
            API key string if found, None otherwise
        """

        def _query(session):
            # Join LLMProvider and ModelServiceProvider tables
            from sqlalchemy import join

            stmt = (
                select(ModelServiceProvider.api_key)
                .select_from(join(LLMProvider, ModelServiceProvider, LLMProvider.name == ModelServiceProvider.name))
                .where(
                    LLMProvider.name == provider_name,
                    LLMProvider.gmt_deleted.is_(None),
                    ModelServiceProvider.status != ModelServiceProviderStatus.DELETED,
                    ModelServiceProvider.gmt_deleted.is_(None),
                )
            )

            # Add user access control conditions
            conditions = []
            if need_public:
                conditions.append(LLMProvider.user_id == "public")
            if user_id:
                conditions.append(LLMProvider.user_id == user_id)

            if conditions:
                if len(conditions) == 1:
                    stmt = stmt.where(conditions[0])
                else:
                    from sqlalchemy import or_

                    stmt = stmt.where(or_(*conditions))

            result = session.execute(stmt)
            return result.scalar()

        return self._execute_query(_query)

    def query_msp_dict(self, user: str = None):
        def _query(session):
            stmt = select(ModelServiceProvider).where(
                ModelServiceProvider.status != ModelServiceProviderStatus.DELETED,
                ModelServiceProvider.gmt_deleted.is_(None),
            )
            # For now, return all MSPs since we removed user column
            result = session.execute(stmt)
            return {msp.name: msp for msp in result.scalars().all()}

        return self._execute_query(_query)


class AsyncLlmProviderRepositoryMixin(AsyncRepositoryProtocol):
    async def query_provider_api_key(self, provider_name: str, user_id: str = None, need_public: bool = True) -> str:
        """Query provider API key with user access control using single SQL JOIN

        Args:
            provider_name: Provider name to query
            user_id: User ID for private provider access
            need_public: Whether to include public providers

        Returns:
            API key string if found, None otherwise

            SELECT model_service_provider.api_key
                FROM llm_provider
                JOIN model_service_provider ON llm_provider.name = model_service_provider.name
                WHERE llm_provider.name = :provider_name
                AND llm_provider.gmt_deleted IS NULL
                AND model_service_provider.status != 'DELETED'
                AND model_service_provider.gmt_deleted IS NULL
                AND (llm_provider.user_id = 'public' OR llm_provider.user_id = :user_id)
        """

        async def _query(session):
            # Join LLMProvider and ModelServiceProvider tables
            from sqlalchemy import join

            stmt = (
                select(ModelServiceProvider.api_key)
                .select_from(join(LLMProvider, ModelServiceProvider, LLMProvider.name == ModelServiceProvider.name))
                .where(
                    LLMProvider.name == provider_name,
                    LLMProvider.gmt_deleted.is_(None),
                    ModelServiceProvider.status != ModelServiceProviderStatus.DELETED,
                    ModelServiceProvider.gmt_deleted.is_(None),
                )
            )

            # Add user access control conditions
            conditions = []
            if need_public:
                conditions.append(LLMProvider.user_id == "public")
            if user_id:
                conditions.append(LLMProvider.user_id == user_id)

            if conditions:
                if len(conditions) == 1:
                    stmt = stmt.where(conditions[0])
                else:
                    from sqlalchemy import or_

                    stmt = stmt.where(or_(*conditions))

            result = await session.execute(stmt)
            return result.scalar()

        return await self._execute_query(_query)

    async def upsert_msp(self, name: str, api_key: str):
        """Create or update model service provider API key"""

        async def _operation(session):
            # Try to find existing MSP
            stmt = select(ModelServiceProvider).where(
                ModelServiceProvider.name == name, ModelServiceProvider.gmt_deleted.is_(None)
            )
            result = await session.execute(stmt)
            msp = result.scalars().first()

            if msp:
                # Update existing
                msp.api_key = api_key
                msp.gmt_updated = utc_now()
                session.add(msp)
            else:
                # Create new
                from aperag.db.models import ModelServiceProviderStatus

                msp = ModelServiceProvider(name=name, status=ModelServiceProviderStatus.ACTIVE, api_key=api_key)
                session.add(msp)

            await session.flush()
            await session.refresh(msp)
            return msp

        return await self.execute_with_transaction(_operation)

    async def delete_msp_by_name(self, name: str):
        """Physical delete model service provider by name"""

        async def _operation(session):
            stmt = select(ModelServiceProvider).where(
                ModelServiceProvider.name == name, ModelServiceProvider.gmt_deleted.is_(None)
            )
            result = await session.execute(stmt)
            msp = result.scalars().first()

            if msp:
                await session.delete(msp)
                await session.flush()
                return True
            return False

        return await self.execute_with_transaction(_operation)

    async def delete_msp(self, msp: ModelServiceProvider):
        """Physical delete model service provider"""

        async def _operation(session):
            await session.delete(msp)
            await session.flush()

        return await self.execute_with_transaction(_operation)

    async def query_msp(self, user: str = None, provider: str = None, filterDeletion: bool = True):
        async def _query(session):
            stmt = select(ModelServiceProvider).where(ModelServiceProvider.name == provider)
            if filterDeletion:
                stmt = stmt.where(
                    ModelServiceProvider.status != ModelServiceProviderStatus.DELETED,
                    ModelServiceProvider.gmt_deleted.is_(None),
                )
            result = await session.execute(stmt)
            return result.scalars().first()

        return await self._execute_query(_query)

    async def query_msp_dict(self, user: str = None):
        async def _query(session):
            stmt = select(ModelServiceProvider).where(
                ModelServiceProvider.status != ModelServiceProviderStatus.DELETED,
                ModelServiceProvider.gmt_deleted.is_(None),
            )
            # For now, return all MSPs since we removed user column
            result = await session.execute(stmt)
            return {msp.name: msp for msp in result.scalars().all()}

        return await self._execute_query(_query)

    async def query_msp_list(self, user: str = None):
        async def _query(session):
            stmt = select(ModelServiceProvider).where(
                ModelServiceProvider.status != ModelServiceProviderStatus.DELETED,
                ModelServiceProvider.gmt_deleted.is_(None),
            )
            # For now, return all MSPs since we removed user column
            result = await session.execute(stmt)
            return result.scalars().all()

        return await self._execute_query(_query)

    async def update_msp(self, msp: ModelServiceProvider):
        async def _operation(session):
            session.add(msp)
            await session.flush()
            await session.refresh(msp)
            return msp

        return await self.execute_with_transaction(_operation)

    async def upsert_model_service_provider(self, user: str, name: str, api_key: str):
        """Create or update model service provider API key (legacy method)"""
        # This is a wrapper for the new upsert_msp method to maintain compatibility
        return await self.upsert_msp(name, api_key)

    # LLM Provider Operations
    async def query_llm_providers(self, user_id: str = None, need_public: bool = True):
        """Get all active LLM providers, optionally filtered by user and public providers

        Args:
            user_id: User ID to filter by user's private providers
            need_public: Whether to include public providers
        """

        async def _query(session):
            stmt = select(LLMProvider).where(LLMProvider.gmt_deleted.is_(None))

            conditions = []

            # Add public providers condition if needed
            if need_public:
                conditions.append(LLMProvider.user_id == "public")

            # Add user's private providers condition if user_id is provided
            if user_id:
                conditions.append(LLMProvider.user_id == user_id)

            # Apply conditions
            if conditions:
                if len(conditions) == 1:
                    stmt = stmt.where(conditions[0])
                else:
                    from sqlalchemy import or_

                    stmt = stmt.where(or_(*conditions))
            # If no conditions (user_id=None, need_public=False), return all providers

            result = await session.execute(stmt)
            return result.scalars().all()

        return await self._execute_query(_query)

    async def query_llm_provider_by_name(self, name: str):
        """Get LLM provider by name"""

        async def _query(session):
            stmt = select(LLMProvider).where(LLMProvider.name == name, LLMProvider.gmt_deleted.is_(None))
            result = await session.execute(stmt)
            return result.scalars().first()

        return await self._execute_query(_query)

    async def query_llm_provider_by_name_user(self, name: str, user_id: str) -> LLMProvider:
        """Get LLM provider by name and user_id"""

        async def _query(session):
            stmt = select(LLMProvider).where(
                LLMProvider.name == name, LLMProvider.user_id == user_id, LLMProvider.gmt_deleted.is_(None)
            )
            result = await session.execute(stmt)
            return result.scalars().first()

        return await self._execute_query(_query)

    async def create_llm_provider(
        self,
        name: str,
        user_id: str,
        label: str,
        completion_dialect: str = "openai",
        embedding_dialect: str = "openai",
        rerank_dialect: str = "jina_ai",
        allow_custom_base_url: bool = False,
        base_url: str = "",
        extra: str = None,
    ) -> LLMProvider:
        """Create a new LLM provider"""

        async def _operation(session):
            provider = LLMProvider(
                name=name,
                user_id=user_id,
                label=label,
                completion_dialect=completion_dialect,
                embedding_dialect=embedding_dialect,
                rerank_dialect=rerank_dialect,
                allow_custom_base_url=allow_custom_base_url,
                base_url=base_url,
                extra=extra,
            )
            session.add(provider)
            await session.flush()
            await session.refresh(provider)
            return provider

        return await self.execute_with_transaction(_operation)

    async def update_llm_provider(
        self,
        name: str,
        user_id: str = None,
        label: str = None,
        completion_dialect: str = None,
        embedding_dialect: str = None,
        rerank_dialect: str = None,
        allow_custom_base_url: bool = None,
        base_url: str = None,
        extra: str = None,
    ) -> Optional[LLMProvider]:
        """Update an existing LLM provider"""

        async def _operation(session):
            stmt = select(LLMProvider).where(LLMProvider.name == name, LLMProvider.gmt_deleted.is_(None))
            result = await session.execute(stmt)
            provider = result.scalars().first()

            if provider:
                if user_id is not None:
                    provider.user_id = user_id
                if label is not None:
                    provider.label = label
                if completion_dialect is not None:
                    provider.completion_dialect = completion_dialect
                if embedding_dialect is not None:
                    provider.embedding_dialect = embedding_dialect
                if rerank_dialect is not None:
                    provider.rerank_dialect = rerank_dialect
                if allow_custom_base_url is not None:
                    provider.allow_custom_base_url = allow_custom_base_url
                if base_url is not None:
                    provider.base_url = base_url
                if extra is not None:
                    provider.extra = extra

                provider.gmt_updated = utc_now()
                session.add(provider)
                await session.flush()
                await session.refresh(provider)

            return provider

        return await self.execute_with_transaction(_operation)

    async def delete_llm_provider(self, name: str) -> Optional[LLMProvider]:
        """Soft delete LLM provider and its models"""

        async def _operation(session):
            stmt = select(LLMProvider).where(LLMProvider.name == name, LLMProvider.gmt_deleted.is_(None))
            result = await session.execute(stmt)
            provider = result.scalars().first()

            if provider:
                # Soft delete the provider
                provider.gmt_deleted = utc_now()
                provider.gmt_updated = utc_now()
                session.add(provider)

                # Also soft delete all models for this provider
                models_stmt = select(LLMProviderModel).where(
                    LLMProviderModel.provider_name == name, LLMProviderModel.gmt_deleted.is_(None)
                )
                models_result = await session.execute(models_stmt)
                models = models_result.scalars().all()
                for model in models:
                    model.gmt_deleted = utc_now()
                    model.gmt_updated = utc_now()
                    session.add(model)

                await session.flush()
                await session.refresh(provider)

            return provider

        return await self.execute_with_transaction(_operation)

    async def restore_llm_provider(self, name: str) -> Optional[LLMProvider]:
        """Restore a soft-deleted LLM provider"""

        async def _operation(session):
            stmt = select(LLMProvider).where(LLMProvider.name == name, LLMProvider.gmt_deleted.is_not(None))
            result = await session.execute(stmt)
            provider = result.scalars().first()

            if provider:
                provider.gmt_deleted = None
                provider.gmt_updated = utc_now()
                session.add(provider)

                # Also restore all models for this provider
                models_stmt = select(LLMProviderModel).where(
                    LLMProviderModel.provider_name == name, LLMProviderModel.gmt_deleted.is_not(None)
                )
                models_result = await session.execute(models_stmt)
                models = models_result.scalars().all()
                for model in models:
                    model.gmt_deleted = None
                    model.gmt_updated = utc_now()
                    session.add(model)

                await session.flush()
                await session.refresh(provider)

            return provider

        return await self.execute_with_transaction(_operation)

    # LLM Provider Model Operations
    async def query_llm_provider_models(self, provider_name: str = None):
        """Get all active LLM provider models, optionally filtered by provider"""

        async def _query(session):
            stmt = select(LLMProviderModel).where(LLMProviderModel.gmt_deleted.is_(None))
            if provider_name:
                stmt = stmt.where(LLMProviderModel.provider_name == provider_name)
            result = await session.execute(stmt)
            return result.scalars().all()

        return await self._execute_query(_query)

    async def query_llm_provider_models_by_provider_list(self, provider_names: list):
        """Get all active LLM provider models for a list of providers

        Args:
            provider_names: List of provider names to filter by

        Returns:
            List of LLMProviderModel objects
        """

        async def _query(session):
            if not provider_names:
                return []

            stmt = select(LLMProviderModel).where(
                LLMProviderModel.provider_name.in_(provider_names), LLMProviderModel.gmt_deleted.is_(None)
            )
            result = await session.execute(stmt)
            return result.scalars().all()

        return await self._execute_query(_query)

    async def query_llm_provider_model(self, provider_name: str, api: str, model: str):
        """Get a specific LLM provider model"""

        async def _query(session):
            stmt = select(LLMProviderModel).where(
                LLMProviderModel.provider_name == provider_name,
                LLMProviderModel.api == api,
                LLMProviderModel.model == model,
                LLMProviderModel.gmt_deleted.is_(None),
            )
            result = await session.execute(stmt)
            return result.scalars().first()

        return await self._execute_query(_query)

    async def create_llm_provider_model(
        self,
        provider_name: str,
        api: str,
        model: str,
        custom_llm_provider: str,
        context_window: int = None,
        max_input_tokens: int = None,
        max_output_tokens: int = None,
        tags: list = None,
    ) -> LLMProviderModel:
        """Create a new LLM provider model"""

        async def _operation(session):
            from aperag.db.models import APIType

            # Convert enum to string if needed
            api_value = api.value if isinstance(api, APIType) else api

            model_obj = LLMProviderModel(
                provider_name=provider_name,
                api=api_value,
                model=model,
                custom_llm_provider=custom_llm_provider,
                context_window=context_window,
                max_input_tokens=max_input_tokens,
                max_output_tokens=max_output_tokens,
                tags=tags or [],
            )
            session.add(model_obj)
            await session.flush()
            await session.refresh(model_obj)
            return model_obj

        return await self.execute_with_transaction(_operation)

    async def update_llm_provider_model(
        self,
        provider_name: str,
        api: str,
        model: str,
        custom_llm_provider: str = None,
        context_window: int = None,
        max_input_tokens: int = None,
        max_output_tokens: int = None,
        tags: list = None,
    ) -> Optional[LLMProviderModel]:
        """Update an existing LLM provider model"""

        async def _operation(session):
            stmt = select(LLMProviderModel).where(
                LLMProviderModel.provider_name == provider_name,
                LLMProviderModel.api == api,
                LLMProviderModel.model == model,
                LLMProviderModel.gmt_deleted.is_(None),
            )
            result = await session.execute(stmt)
            model_obj = result.scalars().first()

            if model_obj:
                if custom_llm_provider is not None:
                    model_obj.custom_llm_provider = custom_llm_provider
                if context_window is not None:
                    model_obj.context_window = context_window
                if max_input_tokens is not None:
                    model_obj.max_input_tokens = max_input_tokens
                if max_output_tokens is not None:
                    model_obj.max_output_tokens = max_output_tokens
                if tags is not None:
                    model_obj.tags = tags

                model_obj.gmt_updated = utc_now()
                session.add(model_obj)
                await session.flush()
                await session.refresh(model_obj)

            return model_obj

        return await self.execute_with_transaction(_operation)

    async def delete_llm_provider_model(self, provider_name: str, api: str, model: str) -> Optional[LLMProviderModel]:
        """Soft delete a specific LLM provider model"""

        async def _operation(session):
            stmt = select(LLMProviderModel).where(
                LLMProviderModel.provider_name == provider_name,
                LLMProviderModel.api == api,
                LLMProviderModel.model == model,
                LLMProviderModel.gmt_deleted.is_(None),
            )
            result = await session.execute(stmt)
            model_obj = result.scalars().first()

            if model_obj:
                model_obj.gmt_deleted = utc_now()
                model_obj.gmt_updated = utc_now()
                session.add(model_obj)
                await session.flush()
                await session.refresh(model_obj)

            return model_obj

        return await self.execute_with_transaction(_operation)

    async def restore_llm_provider_model(self, provider_name: str, api: str, model: str) -> Optional[LLMProviderModel]:
        """Restore a soft-deleted LLM provider model"""

        async def _operation(session):
            stmt = select(LLMProviderModel).where(
                LLMProviderModel.provider_name == provider_name,
                LLMProviderModel.api == api,
                LLMProviderModel.model == model,
                LLMProviderModel.gmt_deleted.is_not(None),
            )
            result = await session.execute(stmt)
            model_obj = result.scalars().first()

            if model_obj:
                model_obj.gmt_deleted = None
                model_obj.gmt_updated = utc_now()
                session.add(model_obj)
                await session.flush()
                await session.refresh(model_obj)

            return model_obj

        return await self.execute_with_transaction(_operation)

    async def query_available_providers_with_models(self, user_id: str = None, need_public: bool = True):
        """Query available providers with their models in a single optimized query

        This method uses JOIN queries to efficiently get:
        1. Providers that user can access (public + user's private)
        2. Only providers that have configured API keys
        3. All models for those providers

        Args:
            user_id: User ID for private provider access
            need_public: Whether to include public providers

        Returns:
            Dict with 'providers' and 'models' keys containing the data
        """

        async def _query(session):
            from sqlalchemy import join, or_

            # First query: Get providers with API keys using JOIN
            provider_stmt = (
                select(LLMProvider, ModelServiceProvider.api_key)
                .select_from(join(LLMProvider, ModelServiceProvider, LLMProvider.name == ModelServiceProvider.name))
                .where(
                    LLMProvider.gmt_deleted.is_(None),
                    ModelServiceProvider.status != ModelServiceProviderStatus.DELETED,
                    ModelServiceProvider.gmt_deleted.is_(None),
                )
            )

            # Add user access control conditions
            conditions = []
            if need_public:
                conditions.append(LLMProvider.user_id == "public")
            if user_id:
                conditions.append(LLMProvider.user_id == user_id)

            if conditions:
                if len(conditions) == 1:
                    provider_stmt = provider_stmt.where(conditions[0])
                else:
                    provider_stmt = provider_stmt.where(or_(*conditions))

            # Execute provider query
            provider_result = await session.execute(provider_stmt)
            provider_rows = provider_result.all()

            # Extract provider names for model query
            available_provider_names = [row[0].name for row in provider_rows]

            if not available_provider_names:
                return {"providers": [], "models": []}

            # Second query: Get models for available providers only
            model_stmt = select(LLMProviderModel).where(
                LLMProviderModel.provider_name.in_(available_provider_names), LLMProviderModel.gmt_deleted.is_(None)
            )

            model_result = await session.execute(model_stmt)
            models = model_result.scalars().all()

            # Return structured data
            providers = [row[0] for row in provider_rows]  # LLMProvider objects

            return {"providers": providers, "models": models}

        return await self._execute_query(_query)
