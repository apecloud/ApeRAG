"""Cross-process bootstrap helpers for the ApeRAG runtime.

Both ``aperag/app.py`` (API process) and ``aperag/cli/indexing_worker.py``
(indexing-worker process) must wire the same set of cross-domain DI
seams before any request / queue payload is served. PR #1884 (task #17)
split the two processes apart but only ported the worker entrypoints —
the DI setters stayed inline in ``app.py`` and were never invoked in
the worker process. The worker would crash the moment it executed any
code path that called ``_get_prompt_template_ops()`` /
``_get_quota_ops()`` / ``_get_marketplace_ops()`` (collection summary
regen, agent runtime, quota check). e2e-http-provider surfaced it; this
module hot-fixes the gap.

``wire_cross_domain_di_seams`` is the single source of truth for the
ten setters. ``tests/boundaries/test_worker_di_parity.py`` AST-greps
both entry points to enforce that they call this function — no future
process split can drift again.
"""

from __future__ import annotations

from aperag.domains.agent_runtime.runtime import set_prompt_template_ops as _ar_set_prompt_template_ops
from aperag.domains.conversation.service.bot_service import set_quota_ops as _conv_set_quota_ops
from aperag.domains.identity.service.user_manager import (
    set_bot_init_ops as _id_set_bot_init_ops,
)
from aperag.domains.identity.service.user_manager import (
    set_chat_init_ops as _id_set_chat_init_ops,
)
from aperag.domains.identity.service.user_manager import (
    set_quota_init_ops as _id_set_quota_init_ops,
)
from aperag.domains.knowledge_base.service.collection_service import (
    set_marketplace_collection_ops as _kb_set_marketplace_collection_ops,
)
from aperag.domains.knowledge_base.service.collection_service import (
    set_marketplace_ops as _kb_set_marketplace_ops,
)
from aperag.domains.knowledge_base.service.collection_service import (
    set_quota_ops as _kb_set_quota_ops,
)
from aperag.domains.knowledge_base.service.collection_service import (
    set_search_pipeline_ops as _kb_set_search_pipeline_ops,
)
from aperag.domains.marketplace.service.marketplace_collection_service import (
    marketplace_collection_service as _legacy_marketplace_collection_service,
)
from aperag.domains.marketplace.service.marketplace_service import (
    marketplace_service as _legacy_marketplace_service,
)
from aperag.domains.model_platform.api.prompts_routes import set_prompt_crud_ops as _set_prompt_crud_ops
from aperag.service.prompt_template_service import (
    build_agent_query_prompt as _legacy_build_agent_query_prompt,
)
from aperag.service.prompt_template_service import (
    prompt_template_service as _legacy_prompt_template_service,
)
from aperag.service.quota_service import quota_service as _legacy_quota_service
from aperag.service.search_pipeline_service import search_pipeline_service as _legacy_search_pipeline_service


class _PromptTemplateOpsAdapter:
    async def resolve_agent_system_prompt(self, *, bot, user_id):
        return await _legacy_prompt_template_service.resolve_agent_system_prompt(bot=bot, user_id=user_id)

    async def resolve_agent_query_prompt(self, *, bot, user_id):
        return await _legacy_prompt_template_service.resolve_agent_query_prompt(bot=bot, user_id=user_id)

    def build_agent_query_prompt(self, chat_id, *, agent_message, user, template=None, has_chat_files=False):
        return _legacy_build_agent_query_prompt(
            chat_id,
            agent_message=agent_message,
            user=user,
            template=template,
            has_chat_files=has_chat_files,
        )


class _BotInitOpsAdapter:
    async def create_default_bot_for_user(self, user_id: str) -> None:
        from aperag.domains.conversation.db.models import BotType
        from aperag.domains.conversation.schemas import BotCreate
        from aperag.domains.conversation.service.bot_service import bot_service

        bot_create = BotCreate(
            title="Default Agent Bot",
            type=BotType.AGENT,
            description="Default agent bot created on registration.",
            collection_ids=[],
        )
        await bot_service.create_bot(user=user_id, bot_in=bot_create, skip_quota_check=True)

        await self._create_summary_bot_for_user(user_id)

    @staticmethod
    async def _create_summary_bot_for_user(user_id: str) -> None:
        """Create the hidden ``BotType.SUMMARY, is_system=True`` row for ``user_id``.

        Bypasses the user-facing ``bot_service.create_bot`` because (a)
        ``BotCreate.type`` is ``Literal["agent"]`` so the public schema
        can't express ``"summary"``, and (b) system bots intentionally
        skip user-quota / user-visibility logic.
        """
        from aperag.config import get_async_session
        from aperag.domains.conversation.db.models import Bot, BotStatus, BotType

        bot = Bot(
            user=user_id,
            title="Summary Generation Bot",
            type=BotType.SUMMARY,
            description="System-managed bot for collection summary regen (Wave 10 §K.13).",
            status=BotStatus.ACTIVE,
            config='{"agent": {"system_prompt_template": null}}',
            is_system=True,
        )
        async for session in get_async_session():
            session.add(bot)
            try:
                await session.commit()
            except Exception:  # noqa: BLE001
                # Concurrent register-time race / backfill migration
                # already created the row; let the partial unique index
                # reject silently. The lazy fallback in the regen
                # service will fetch the existing row on first use.
                await session.rollback()
            return


class _ChatInitOpsAdapter:
    async def create_default_chat_for_user(self, user_id: str) -> None:
        from aperag.domains.conversation.service.chat_collection_service import chat_collection_service

        await chat_collection_service.initialize_user_chat_collection(user_id)


class _QuotaInitOpsAdapter:
    async def initialize_user_quota(self, user_id: str) -> None:
        await _legacy_quota_service.initialize_user_quotas(user_id)


def wire_cross_domain_di_seams() -> None:
    """Wire every cross-domain DI seam that ``app.py`` used to set inline.

    Idempotent — module-level setters overwrite the previous binding
    each time. Safe to call from both the API entrypoint and the
    indexing-worker entrypoint.
    """
    _kb_set_marketplace_ops(_legacy_marketplace_service)
    _kb_set_marketplace_collection_ops(_legacy_marketplace_collection_service)
    _kb_set_search_pipeline_ops(_legacy_search_pipeline_service)
    _kb_set_quota_ops(_legacy_quota_service)

    _conv_set_quota_ops(_legacy_quota_service)

    _ar_set_prompt_template_ops(_PromptTemplateOpsAdapter())
    _set_prompt_crud_ops(_legacy_prompt_template_service)

    _id_set_bot_init_ops(_BotInitOpsAdapter())
    _id_set_chat_init_ops(_ChatInitOpsAdapter())
    _id_set_quota_init_ops(_QuotaInitOpsAdapter())


__all__ = ["wire_cross_domain_di_seams"]
