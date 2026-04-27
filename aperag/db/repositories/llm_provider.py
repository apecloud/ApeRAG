from __future__ import annotations

from typing import Optional

import sqlalchemy as sa
from sqlalchemy import select

from aperag.db.repositories.base import AsyncRepositoryProtocol, SyncRepositoryProtocol
from aperag.domains.model_platform.db.models import Model, ModelAccount, ModelProvider, ModelUse
from aperag.utils.utils import utc_now


class LlmProviderRepositoryMixin(SyncRepositoryProtocol):
    def query_model_runtime(self, model_id: str, user_id: str):
        def _query(session):
            model = session.get(Model, model_id)
            if not model or model.gmt_deleted is not None or model.status != "ACTIVE":
                return None
            account = session.get(ModelAccount, model.account_id)
            if (
                not account
                or account.gmt_deleted is not None
                or account.status != "ACTIVE"
                or account.user_id not in (user_id, "public")
            ):
                return None
            return model, account

        return self._execute_query(_query)

    def resolve_legacy_model_id(
        self,
        user_id: str,
        *,
        provider_type: str,
        provider_model_id: str,
        capability: str | None = None,
    ) -> Optional[str]:
        """Sync companion to the async resolver in
        :class:`AsyncLlmProviderRepositoryMixin`. Used by sync collection
        config callers (``base_embedding``, ``base_completion``) that
        need to map the legacy ``{model, provider}`` triple to a
        new-schema ``model.id`` without an event loop.
        """

        def _query(session):
            stmt = (
                select(Model, ModelAccount)
                .join(ModelAccount, Model.account_id == ModelAccount.id)
                .where(
                    Model.gmt_deleted.is_(None),
                    Model.status == "ACTIVE",
                    Model.provider_model_id == provider_model_id,
                    ModelAccount.gmt_deleted.is_(None),
                    ModelAccount.status == "ACTIVE",
                    ModelAccount.provider_type == provider_type,
                    (ModelAccount.user_id == user_id) | (ModelAccount.user_id == "public"),
                )
            )
            if capability:
                stmt = stmt.where(Model.capability == capability)
            result = session.execute(stmt)
            rows = result.all()
            if not rows:
                return None
            # Prefer user-owned over public.
            user_rows = [(model, account) for model, account in rows if account.user_id == user_id]
            chosen = user_rows[0] if user_rows else rows[0]
            return chosen[0].id

        return self._execute_query(_query)


class AsyncLlmProviderRepositoryMixin(AsyncRepositoryProtocol):
    async def query_model_providers(self):
        async def _query(session):
            stmt = (
                select(ModelProvider)
                .where(ModelProvider.gmt_deleted.is_(None), ModelProvider.enabled.is_(True))
                .order_by(ModelProvider.sort_order, ModelProvider.display_name)
            )
            result = await session.execute(stmt)
            return result.scalars().all()

        return await self._execute_query(_query)

    async def query_model_account_api_key(
        self, provider_type: str, user_id: str, fallback_to_public: bool = False
    ) -> Optional[str]:
        """Resolve the encrypted API key for the most recently-updated
        ACTIVE ``ModelAccount`` for ``provider_type`` owned by ``user_id``.

        When ``fallback_to_public`` is true, fall back to any ACTIVE
        ``public`` account when the user has no personal account configured.
        Returns ``None`` when no account is configured.

        Primitive for callers that need the raw provider api key without
        having to know the ``ModelAccount`` id (e.g. Jina web reading /
        web search). This is consumed by
        ``ModelPlatformService.get_user_provider_api_key``; cross-domain
        callers go through the service, not the repository.
        """

        async def _query(session):
            user_filter = ModelAccount.user_id == user_id
            if fallback_to_public:
                user_filter = user_filter | (ModelAccount.user_id == "public")
            stmt = (
                select(ModelAccount)
                .where(
                    ModelAccount.provider_type == provider_type,
                    ModelAccount.gmt_deleted.is_(None),
                    ModelAccount.status == "ACTIVE",
                    user_filter,
                )
                # The user's personal account ALWAYS wins over a ``public`` row,
                # even when ``public`` was updated more recently — otherwise a
                # later admin edit on the shared key would silently shadow the
                # caller's own credential.
                .order_by(
                    sa.case((ModelAccount.user_id == user_id, 0), else_=1),
                    ModelAccount.gmt_updated.desc(),
                )
            )
            result = await session.execute(stmt)
            account = result.scalars().first()
            return account.encrypted_api_key if account else None

        return await self._execute_query(_query)

    async def query_model_accounts(self, user_id: str):
        async def _query(session):
            stmt = (
                select(ModelAccount)
                .where(
                    ModelAccount.gmt_deleted.is_(None),
                    ModelAccount.status == "ACTIVE",
                    (ModelAccount.user_id == user_id) | (ModelAccount.user_id == "public"),
                )
                .order_by(ModelAccount.display_name)
            )
            result = await session.execute(stmt)
            return result.scalars().all()

        return await self._execute_query(_query)

    async def query_model_account(self, account_id: str, user_id: str) -> Optional[ModelAccount]:
        async def _query(session):
            stmt = select(ModelAccount).where(
                ModelAccount.id == account_id,
                ModelAccount.gmt_deleted.is_(None),
                ModelAccount.status == "ACTIVE",
                (ModelAccount.user_id == user_id) | (ModelAccount.user_id == "public"),
            )
            result = await session.execute(stmt)
            return result.scalars().first()

        return await self._execute_query(_query)

    async def create_model_account(
        self,
        user_id: str,
        provider_type: str,
        name: str,
        display_name: str,
        base_url: str,
        api_key: str,
        auth_config: dict | None = None,
        extra: dict | None = None,
    ) -> ModelAccount:
        async def _operation(session):
            account = ModelAccount(
                user_id=user_id,
                provider_type=provider_type,
                name=name,
                display_name=display_name,
                base_url=base_url,
                encrypted_api_key=api_key,
                auth_config=auth_config or {},
                extra=extra or {},
            )
            session.add(account)
            await session.flush()
            await session.refresh(account)
            return account

        return await self.execute_with_transaction(_operation)

    async def update_model_account(self, account_id: str, user_id: str, values: dict) -> Optional[ModelAccount]:
        async def _operation(session):
            result = await session.execute(
                select(ModelAccount).where(
                    ModelAccount.id == account_id,
                    ModelAccount.gmt_deleted.is_(None),
                    ModelAccount.status == "ACTIVE",
                    ModelAccount.user_id == user_id,
                )
            )
            account = result.scalars().first()
            if account is None:
                return None
            for key in ["name", "display_name", "base_url", "auth_config", "status", "extra"]:
                if key in values and values[key] is not None:
                    setattr(account, key, values[key])
            if values.get("api_key"):
                account.encrypted_api_key = values["api_key"]
            account.gmt_updated = utc_now()
            session.add(account)
            await session.flush()
            await session.refresh(account)
            return account

        return await self.execute_with_transaction(_operation)

    async def query_models(self, user_id: str, account_id: str | None = None):
        async def _query(session):
            stmt = (
                select(Model)
                .join(ModelAccount, Model.account_id == ModelAccount.id)
                .where(
                    Model.gmt_deleted.is_(None),
                    Model.status == "ACTIVE",
                    ModelAccount.gmt_deleted.is_(None),
                    ModelAccount.status == "ACTIVE",
                    (ModelAccount.user_id == user_id) | (ModelAccount.user_id == "public"),
                )
                .order_by(Model.display_name)
            )
            if account_id:
                stmt = stmt.where(Model.account_id == account_id)
            result = await session.execute(stmt)
            return result.scalars().all()

        return await self._execute_query(_query)

    async def query_model(self, model_id: str, user_id: str) -> Optional[Model]:
        async def _query(session):
            stmt = (
                select(Model)
                .join(ModelAccount, Model.account_id == ModelAccount.id)
                .where(
                    Model.id == model_id,
                    Model.gmt_deleted.is_(None),
                    Model.status == "ACTIVE",
                    ModelAccount.gmt_deleted.is_(None),
                    ModelAccount.status == "ACTIVE",
                    (ModelAccount.user_id == user_id) | (ModelAccount.user_id == "public"),
                )
            )
            result = await session.execute(stmt)
            return result.scalars().first()

        return await self._execute_query(_query)

    async def query_model_runtime(self, model_id: str, user_id: str):
        async def _query(session):
            stmt = (
                select(Model, ModelAccount)
                .join(ModelAccount, Model.account_id == ModelAccount.id)
                .where(
                    Model.id == model_id,
                    Model.gmt_deleted.is_(None),
                    Model.status == "ACTIVE",
                    ModelAccount.gmt_deleted.is_(None),
                    ModelAccount.status == "ACTIVE",
                    (ModelAccount.user_id == user_id) | (ModelAccount.user_id == "public"),
                )
            )
            result = await session.execute(stmt)
            return result.first()

        return await self._execute_query(_query)

    async def create_model(
        self,
        user_id: str,
        account_id: str,
        provider_model_id: str,
        display_name: str,
        capability: str,
        runner_type: str,
        runner_config: dict | None = None,
        context_window: int | None = None,
        max_input_tokens: int | None = None,
        max_output_tokens: int | None = None,
        embedding_dimensions: int | None = None,
        supports_vision: bool = False,
        supports_tool_calling: bool = False,
        supports_multimodal_embedding: bool = False,
        extra: dict | None = None,
    ) -> Model:
        async def _operation(session):
            model = Model(
                account_id=account_id,
                provider_model_id=provider_model_id,
                display_name=display_name,
                capability=capability.value if hasattr(capability, "value") else capability,
                runner_type=runner_type,
                runner_config=runner_config or {},
                context_window=context_window,
                max_input_tokens=max_input_tokens,
                max_output_tokens=max_output_tokens,
                embedding_dimensions=embedding_dimensions,
                supports_vision=supports_vision,
                supports_tool_calling=supports_tool_calling,
                supports_multimodal_embedding=supports_multimodal_embedding,
                extra=extra or {},
            )
            session.add(model)
            await session.flush()
            await session.refresh(model)
            return model

        return await self.execute_with_transaction(_operation)

    async def update_model(self, model_id: str, user_id: str, values: dict) -> Optional[Model]:
        async def _operation(session):
            result = await session.execute(
                select(Model)
                .join(ModelAccount, Model.account_id == ModelAccount.id)
                .where(
                    Model.id == model_id,
                    Model.gmt_deleted.is_(None),
                    Model.status == "ACTIVE",
                    ModelAccount.gmt_deleted.is_(None),
                    ModelAccount.status == "ACTIVE",
                    ModelAccount.user_id == user_id,
                )
            )
            model = result.scalars().first()
            if model is None:
                return None
            for key, value in values.items():
                if value is None:
                    continue
                if hasattr(value, "value"):
                    value = value.value
                setattr(model, key, value)
            model.gmt_updated = utc_now()
            session.add(model)
            await session.flush()
            await session.refresh(model)
            return model

        return await self.execute_with_transaction(_operation)

    async def query_model_uses(self, user_id: str):
        async def _query(session):
            stmt = (
                select(ModelUse)
                .where(
                    ModelUse.gmt_deleted.is_(None),
                    ModelUse.enabled.is_(True),
                    (ModelUse.user_id == user_id) | (ModelUse.user_id == "public"),
                )
                .order_by(ModelUse.scenario)
            )
            result = await session.execute(stmt)
            return result.scalars().all()

        return await self._execute_query(_query)

    async def upsert_model_use(self, user_id: str, scenario: str, values: dict) -> ModelUse:
        async def _operation(session):
            stmt = select(ModelUse).where(
                ModelUse.user_id == user_id,
                ModelUse.scenario == scenario,
                ModelUse.gmt_deleted.is_(None),
            )
            result = await session.execute(stmt)
            item = result.scalars().first()
            if item is None:
                item = ModelUse(
                    user_id=user_id,
                    scenario=scenario,
                    capability=values.get("capability") or "chat",
                )
            for key in ["strategy", "primary_model_id", "fallback_model_ids", "enabled", "extra"]:
                if key in values:
                    value = values[key]
                    if hasattr(value, "value"):
                        value = value.value
                    setattr(item, key, value)
            item.gmt_updated = utc_now()
            session.add(item)
            await session.flush()
            await session.refresh(item)
            return item

        return await self.execute_with_transaction(_operation)
