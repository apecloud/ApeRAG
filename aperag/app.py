# ruff: noqa: E402
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

import asyncio  # noqa: E402
import contextlib  # noqa: E402

from aperag.config import settings
from aperag.observability import (
    bind_observability_context,
    build_observability_config,
    configure_fastapi,
    configure_logging,
    configure_process_observability,
    reset_observability_context,
)
from aperag.observability.tracing import inject_carrier

observability_config = build_observability_config(settings)
configure_logging(observability_config)
configure_process_observability(observability_config)

from fastapi import FastAPI  # noqa: E402
from starlette.middleware.base import BaseHTTPMiddleware  # noqa: E402

from aperag.domains.agent_runtime.api.routes import router as agent_runtime_router
from aperag.domains.agent_runtime.runtime import set_prompt_template_ops as _ar_set_prompt_template_ops
from aperag.domains.conversation.api.openai_routes import router as openai_router
from aperag.domains.conversation.api.routes import (
    bots_router as bots_v2_router,
)
from aperag.domains.conversation.api.routes import (
    chat_router as chat_router,
)
from aperag.domains.conversation.service.bot_service import set_quota_ops as _conv_set_quota_ops
from aperag.domains.evaluation.api.routes import router as evaluation_v2_router
from aperag.domains.governance.api.apikeys_routes import router as apikeys_router
from aperag.domains.governance.api.audit_routes import router as audit_router
from aperag.domains.governance.api.quota_routes import router as quota_router
from aperag.domains.identity.api.auth_routes import router as auth_router
from aperag.domains.identity.api.config_routes import router as config_router
from aperag.domains.identity.service.user_manager import (
    set_bot_init_ops as _id_set_bot_init_ops,
)
from aperag.domains.identity.service.user_manager import (
    set_chat_init_ops as _id_set_chat_init_ops,
)
from aperag.domains.identity.service.user_manager import (
    set_quota_init_ops as _id_set_quota_init_ops,
)
from aperag.domains.knowledge_base.api.export_routes import router as export_router
from aperag.domains.knowledge_base.api.routes import router as knowledge_base_router
from aperag.domains.knowledge_base.api.settings_routes import router as settings_router
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
from aperag.domains.knowledge_graph.api.routes import router as knowledge_graph_router
from aperag.domains.marketplace.api.routes import router as marketplace_router
from aperag.domains.marketplace.service.marketplace_collection_service import (
    marketplace_collection_service as _legacy_marketplace_collection_service,
)
from aperag.domains.marketplace.service.marketplace_service import (
    marketplace_service as _legacy_marketplace_service,
)
from aperag.domains.model_platform.api.llm_routes import router as llm_router
from aperag.domains.model_platform.api.prompts_routes import router as prompts_router
from aperag.domains.model_platform.api.prompts_routes import set_prompt_crud_ops as _set_prompt_crud_ops
from aperag.domains.model_platform.api.providers_v2_routes import router as providers_v2_router
from aperag.domains.retrieval.api.routes import router as retrieval_router
from aperag.domains.web_access.api.routes import router as web_access_router
from aperag.exception_handlers import register_exception_handlers
from aperag.llm.litellm_track import register_custom_llm_track
from aperag.mcp import mcp_server
from aperag.openapi_spec import custom_generate_unique_id
from aperag.service.quota_service import quota_service as _legacy_quota_service
from aperag.service.search_pipeline_service import search_pipeline_service as _legacy_search_pipeline_service

# Wire the knowledge_base domain's consumer-owned Protocol DI slots
# (Phase 3 Step 5b2c). All four legacy service singletons now
# structurally satisfy the KB Protocols directly — Phase 4 Step 4-S4
# dropped the transitional ``_MarketplaceCollectionOpsAdapter`` once
# the ``marketplace_collection_service`` move renamed
# ``_check_marketplace_access`` to the public ``check_marketplace_access``
# per msg=6ab7d211 Q2.
_kb_set_marketplace_ops(_legacy_marketplace_service)
_kb_set_marketplace_collection_ops(_legacy_marketplace_collection_service)
_kb_set_search_pipeline_ops(_legacy_search_pipeline_service)
_kb_set_quota_ops(_legacy_quota_service)

# Wire the conversation domain's consumer-owned QuotaOps DI slot for
# ``bot_service``. ``quota_service`` is a standalone-infrastructure
# module with no natural domain home, so the Protocol + DI seam is
# the permanent integration point. The singleton structurally
# satisfies the narrower conversation ``QuotaOps`` Protocol
# (``check_and_consume_quota`` + ``release_quota``) directly.
_conv_set_quota_ops(_legacy_quota_service)

# Wire the agent_runtime domain's consumer-owned PromptTemplateOps DI
# slot. ``prompt_template_service`` is a standalone-infrastructure
# module (no natural domain home), so the Protocol + DI seam is the
# permanent integration point — an adapter exposes the three Protocol
# methods onto the singleton + module-level ``build_agent_query_prompt``
# helper.
from aperag.service.prompt_template_service import (  # noqa: E402
    build_agent_query_prompt as _legacy_build_agent_query_prompt,
)
from aperag.service.prompt_template_service import (  # noqa: E402
    prompt_template_service as _legacy_prompt_template_service,
)


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


_ar_set_prompt_template_ops(_PromptTemplateOpsAdapter())

# Wire the model_platform domain's prompt user-CRUD Protocol seam
# (Phase 8 #49 G3, D7 v2 hard-cut). The same ``prompt_template_service``
# singleton already wired into agent_runtime structurally satisfies
# the model_platform ``PromptCRUDOps`` Protocol — the user-CRUD method
# names map 1:1, so no adapter is needed.
_set_prompt_crud_ops(_legacy_prompt_template_service)


# Wire the identity domain's consumer-owned Protocol DI slots (Phase
# 4 Step 4-S7d). ``UserManager.on_after_register`` invokes three
# side effects — default bot + default chat collection + per-user
# quota seed — through ``BotInitOps`` / ``ChatInitOps`` /
# ``QuotaInitOps``. The three concrete providers live in legacy
# ``aperag/service/`` today; Phase 5 moves bot_service and
# chat_collection_service into the conversation domain while quota
# stays legacy per msg=896584ee. Thin adapters below expose the
# public Protocol surface (e.g. ``create_default_bot_for_user``)
# onto the concrete services' existing method names; the adapters
# collapse when Phase 5 conversation services land at the canonical
# location.
class _BotInitOpsAdapter:
    async def create_default_bot_for_user(self, user_id: str) -> None:
        # Lazy imports keep ``aperag/app.py`` start-up cost low and
        # avoid pulling the KB domain services into the identity DI
        # wiring path before the app is constructed.
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

        # Wave 10 §K.13: register-time creation of the per-user
        # hidden summary bot used by ``collection_regen_service``
        # Stage 1. Same transaction as the user's default agent bot
        # so a successful registration always provides both. The
        # ``collection_regen_service`` keeps a defense-in-depth
        # ``get_or_create_summary_bot_for_user`` lazy fallback for
        # the edge case where this hook fails (registration does
        # not roll back on init-hook errors per
        # ``user_manager.py:137``).
        await self._create_summary_bot_for_user(user_id)

    @staticmethod
    async def _create_summary_bot_for_user(user_id: str) -> None:
        """Create the hidden ``BotType.SUMMARY, is_system=True`` row
        for ``user_id``. Bypasses the user-facing ``bot_service.create_bot``
        because (a) ``BotCreate.type`` is ``Literal["agent"]`` so
        the public schema can't express ``"summary"``, and (b) system
        bots intentionally skip user-quota / user-visibility logic.
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
                # Concurrent register-time race or backfill migration
                # already created the row — let the partial unique
                # index reject silently. The lazy fallback in the
                # regen service will fetch the existing row on first
                # use.
                await session.rollback()
            return


class _ChatInitOpsAdapter:
    async def create_default_chat_for_user(self, user_id: str) -> None:
        from aperag.domains.conversation.service.chat_collection_service import chat_collection_service

        await chat_collection_service.initialize_user_chat_collection(user_id)


class _QuotaInitOpsAdapter:
    async def initialize_user_quota(self, user_id: str) -> None:
        await _legacy_quota_service.initialize_user_quotas(user_id)


_id_set_bot_init_ops(_BotInitOpsAdapter())
_id_set_chat_init_ops(_ChatInitOpsAdapter())
_id_set_quota_init_ops(_QuotaInitOpsAdapter())


# Initialize MCP server integration with stateless HTTP to fix OpenAI tool call sequence issues
mcp_app = mcp_server.http_app(path="/", stateless_http=True)


async def combined_lifespan(app: FastAPI):
    """Combined lifespan manager for the API + MCP server + indexing runtime.

    The indexing runtime (Wave 3 T3.1 wire-in) launches the per-modality
    worker pool + reconciler + cleanup loop only when
    ``settings.indexing_mode == "async"``. In ``inline`` mode the
    upload-side ``dispatch_indexing(mode=INLINE)`` runs derive + sync +
    cutover within the request coroutine, so no background workers are
    needed (per design pack §L Tier-1 deployment).

    The runtime is started as background asyncio tasks (not subprocesses)
    so a single FastAPI process owns its workers — matches the §E.2
    "one Python process per modality" architecture for the in-process
    deployment topology. Tier-3 horizontal scale-out runs separate
    worker processes; that wiring lives in a future ops launcher.
    """
    # task #17 (PR #1884) hard cut: API 进程不再启动 worker / reconciler /
    # cleanup task — 这些迁到独立 ``indexing-worker`` deployment 跑
    # ``python -m aperag.cli.indexing_worker``. API 只保留 queue / quota /
    # metrics 给 enqueue 用, 不需要 ``indexing_runtime_tasks`` /
    # ``indexing_shutdown`` event 了.

    if settings.indexing_mode == "async":
        # Lazy imports — pulling the indexing runtime symbols at app
        # start-up time keeps ``aperag/app.py`` cold-start fast and
        # confines the import surface to the wired branch.
        # task #17 (PR #1884): API 进程不再启动任何 indexing worker /
        # reconciler / cleanup loop — 改由独立 ``indexing-worker``
        # deployment 跑 ``python -m aperag.cli.indexing_worker``. API
        # 只保留轻量 enqueue runtime (queue + quota_backend + metrics +
        # IndexingRuntime), 不构造 ProductionWorkerFactory, 不创建
        # worker / reconciler / cleanup task. 见
        # ``docs/zh-CN/architecture/task-system-hard-cut-v8.md``.
        from aperag.config import sync_engine
        from aperag.indexing import (
            InMemoryWorkQueue,
            NoopMetricsEmitter,
            OTLPMetricsEmitter,
            RedisWorkQueue,
        )

        # Wave 4 T4: dispatch on ``INDEXING_QUEUE_BACKEND`` setting
        # (default ``inmemory`` for backward-compat single-pod
        # deployments; production multi-pod sets ``redis`` to enable
        # BLPOP transport per design pack §E.2). InMemoryWorkQueue is
        # process-local — multi-pod deployments lose tasks pushed to
        # one process and BLPOP'd by another, so production must run
        # ``INDEXING_QUEUE_BACKEND=redis`` for correctness.
        if settings.indexing_queue_backend.lower() == "redis":
            queue = RedisWorkQueue(redis_url=settings.indexing_queue_redis_url)
        else:
            queue = InMemoryWorkQueue()
        engine = sync_engine

        # Wave 4 T6: dispatch on ``INDEXING_METRICS_EMITTER`` setting
        # (default ``noop`` for backward-compat; production multi-pod
        # sets ``otlp`` to wire the four §J.1 SLIs onto the
        # ``MeterProvider`` configured by ``aperag.observability``).
        # NoopMetricsEmitter silently drops every sample, so operators
        # running Tier 2/3 production must explicitly opt into ``otlp``
        # — otherwise queue-backlog / failure-rate alerts on the
        # collector side never receive data.
        if settings.indexing_metrics_emitter.lower() == "otlp":
            # Wave 5 P5B: cross-check that the broader observability
            # mode is also OTLP-shaped — operators that flip
            # ``INDEXING_METRICS_EMITTER=otlp`` without configuring
            # the parent ``APERAG_OBSERVABILITY_MODE`` end up with
            # an :class:`OTLPMetricsEmitter` whose underlying
            # ``MeterProvider`` was never installed by
            # ``aperag.observability.metrics.init_metrics_provider``.
            # The samples then no-op silently — the same operator-
            # visible failure mode we explicitly avoided when
            # making ``noop`` the default.
            obs_mode = (settings.aperag_observability_mode or "").lower()
            if obs_mode not in ("otlp", "collector"):
                import logging as _logging

                _logging.getLogger(__name__).warning(
                    "INDEXING_METRICS_EMITTER=otlp but APERAG_OBSERVABILITY_MODE=%r "
                    "(expected 'otlp' or 'collector') — the OTLP MeterProvider "
                    "may not be installed, indexing SLI samples will silently "
                    "no-op. Set APERAG_OBSERVABILITY_MODE=otlp to enable real "
                    "OTLP export, OR revert INDEXING_METRICS_EMITTER=noop to "
                    "make the gap explicit.",
                    settings.aperag_observability_mode,
                )
            metrics_emitter = OTLPMetricsEmitter()
        else:
            metrics_emitter = NoopMetricsEmitter()

        # Wave 4 T5: dispatch on ``INDEXING_QUOTA_BACKEND`` setting
        # (default ``inmemory`` for backward-compat single-pod
        # deployments; production multi-pod sets ``redis`` so worker
        # processes share §H.5 token-bucket state via Redis logical
        # db=3 per §H.5.1 amendment). InMemoryQuotaBackend is process-
        # local — multi-pod deployments running ``inmemory`` would
        # have each pod's worker independently exhaust its tenant
        # quota, which silently breaks the per-tenant rate limit
        # invariant (§H.5).
        from aperag.indexing.quota import (
            InMemoryQuotaBackend,
            QuotaPolicyRegistry,
            RedisQuotaBackend,
        )

        quota_registry = QuotaPolicyRegistry()
        if settings.indexing_quota_backend.lower() == "redis":
            try:
                from redis import asyncio as redis_asyncio
            except ImportError as exc:  # pragma: no cover — redis is a base dep
                raise RuntimeError("INDEXING_QUOTA_BACKEND=redis but redis package not installed") from exc
            quota_redis = redis_asyncio.from_url(
                settings.indexing_quota_redis_url,
                encoding="utf-8",
                decode_responses=False,
            )
            quota_backend = RedisQuotaBackend(quota_redis, quota_registry)
        else:
            quota_redis = None
            quota_backend = InMemoryQuotaBackend(quota_registry)

        # task #17 (PR #1884) hard cut: API 进程不再构造 ProductionWorkerFactory
        # 也不启动 worker / reconciler / cleanup task. 这些都迁到独立
        # ``indexing-worker`` deployment (``python -m aperag.cli.indexing_worker``).
        # API 只保留轻量 enqueue runtime: queue (push to broker) +
        # quota_backend (检查租户配额) + metrics_emitter (上报 SLI).
        #
        # cleanup_worker_factory 之前由 ``ProductionWorkerFactory.build_for_cleanup_row``
        # 提供给 IndexingRuntime, 让 service 层能在 API 请求路径直接执行
        # backend cleanup. task #17 hard gate (ziang msg=cecb0d88 + huangheng
        # msg=f97b7c5f #6) 显式禁止 API 请求路径执行重型 cleanup — 改成只
        # 标记 DB intent (``Document.status=DELETED + gmt_deleted``), worker
        # cleanup loop 异步扫描完成. 因此 API 这里 ``cleanup_worker_factory=None``,
        # service 层调 cleanup 必须返回 no-op (由 task #19 ziang 的 cleanup
        # SoT 改造保证).

        # Stash on app state so request handlers can dispatch via the
        # same queue / engine the workers consume.
        app.state.indexing_queue = queue
        app.state.indexing_engine = engine
        app.state.indexing_metrics_emitter = metrics_emitter
        app.state.indexing_quota_backend = quota_backend
        # Wave 4 T5: stash the underlying Redis client (only when
        # ``INDEXING_QUOTA_BACKEND=redis``) so the lifespan finally
        # block can close it on shutdown — mirrors the T4 RedisWorkQueue
        # close lifecycle.
        app.state.indexing_quota_redis = quota_redis

        # Service-layer callers (aperag/domains/**) consume the same
        # triple through the process-wide IndexingRuntime singleton —
        # they don't have a Request handle for app.state.
        # task #17: workers={} 已经是空; cleanup_worker_factory=None 强制
        # service 层不再在 API 请求路径执行重型 backend cleanup.
        from aperag.indexing.runtime import IndexingRuntime, set_runtime

        set_runtime(
            IndexingRuntime(
                engine=engine,
                queue=queue,
                workers={},
                metrics_emitter=metrics_emitter,
                cleanup_worker_factory=None,
                quota_backend=quota_backend,
            )
        )
    else:
        app.state.indexing_queue = None
        app.state.indexing_engine = None
        app.state.indexing_metrics_emitter = None
        from aperag.indexing.runtime import set_runtime

        set_runtime(None)

    try:
        async with mcp_app.lifespan(app):
            yield
    finally:
        # task #17 hard cut: API 进程没有 worker / reconciler / cleanup task
        # 需要 drain — 这些都在独立 ``indexing-worker`` pod, 由它的
        # ``aperag/cli/indexing_worker.py`` 自己 SIGTERM handler 处理.
        # API 这里只关 queue + quota redis client + metrics provider.

        # Wave 4 T4: release the indexing queue's underlying connection
        # pool (Redis client owns one); InMemoryWorkQueue has no
        # ``close`` so guard with hasattr.
        queue_obj = getattr(app.state, "indexing_queue", None)
        if queue_obj is not None and hasattr(queue_obj, "close"):
            with contextlib.suppress(Exception):
                await queue_obj.close()
        # Wave 4 T5: release the quota Redis client (only present when
        # ``INDEXING_QUOTA_BACKEND=redis`` was selected at startup).
        # ``InMemoryQuotaBackend`` has no underlying client.
        quota_redis_obj = getattr(app.state, "indexing_quota_redis", None)
        if quota_redis_obj is not None:
            with contextlib.suppress(Exception):
                await quota_redis_obj.aclose()
        # Wave 4 T6: flush + shut down the OTLP MeterProvider so the
        # PeriodicExportingMetricReader drains any pending metric
        # samples before the process exits. Mirrors the T4 graceful
        # shutdown pattern and addresses huangheng pass-1 observation A
        # (msg=5d450300). ``shutdown_metrics_provider`` is a no-op when
        # the SDK MeterProvider was never installed (default
        # ``noop`` emitter / OTLP endpoint missing).
        from aperag.observability.metrics import shutdown_metrics_provider

        with contextlib.suppress(Exception):
            await asyncio.to_thread(shutdown_metrics_provider)


# Create the main FastAPI app with combined lifespan
app = FastAPI(
    title="ApeRAG API",
    description="Knowledge management and retrieval system",
    version="1.0.0",
    lifespan=combined_lifespan,  # Combined lifecycle management
    generate_unique_id_function=custom_generate_unique_id,
)


class ObservabilityContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        request_id = request.headers.get("x-request-id") or request.headers.get("x-correlation-id")
        tokens = bind_observability_context(request_id=request_id)
        try:
            response = await call_next(request)
            if request_id:
                response.headers["x-request-id"] = request_id
            trace_headers = inject_carrier({})
            if "traceparent" in trace_headers:
                response.headers["traceparent"] = trace_headers["traceparent"]
            return response
        finally:
            reset_observability_context(tokens)


app.add_middleware(ObservabilityContextMiddleware)
configure_fastapi(app, observability_config)

# Register global exception handlers
register_exception_handlers(app)

register_custom_llm_track()


# Health check endpoint
@app.get("/health")
async def health_check():
    """Simple health check endpoint for container health monitoring"""
    return {"status": "healthy", "service": "aperag-api"}


app.include_router(auth_router, prefix="/api/v2/auth")
app.include_router(export_router, prefix="/api/v2")  # KB-domain export router (Phase 8 #47 G1, D7 v2 hard-cut)
app.include_router(audit_router, prefix="/api/v2")  # Governance: audit-logs (hard-cut to v2 in #50)
app.include_router(apikeys_router, prefix="/api/v2")  # Governance: api_keys (hard-cut to v2 in #51)
app.include_router(quota_router, prefix="/api/v2")  # Governance: quota/system defaults (hard-cut to v2 in #66)
app.include_router(llm_router, prefix="/api/v1")  # Model platform: embed/rerank (OpenAI-compat)
app.include_router(marketplace_router, prefix="/api/v2")  # Marketplace domain router (Phase 8 #52 G4c, D7 v2 hard-cut)
app.include_router(settings_router, prefix="/api/v2")  # KB domain settings (carved from views/ in #48)
app.include_router(prompts_router, prefix="/api/v2")  # Phase 8 #49 G3, D7 v2 hard-cut
app.include_router(web_access_router, prefix="/api/v2", tags=["web_access"])  # Add web_access domain router
app.include_router(retrieval_router, prefix="/api/v2", tags=["retrieval"])  # Add retrieval domain router
app.include_router(
    knowledge_graph_router, prefix="/api/v2", tags=["knowledge_graph"]
)  # Add knowledge_graph domain router
app.include_router(chat_router, prefix="/api/v2")
app.include_router(openai_router, prefix="/v1")
app.include_router(config_router, prefix="/api/v2/config")
app.include_router(agent_runtime_router, prefix="/api/v2")
app.include_router(bots_v2_router, prefix="/api/v2")
app.include_router(evaluation_v2_router, prefix="/api/v2")
app.include_router(providers_v2_router, prefix="/api/v2")  # Model platform: model accounts / models / model uses
app.include_router(knowledge_base_router, prefix="/api/v2")  # KB domain router (collections_v2 + documents_v2)

# Mount the MCP server at /mcp path
app.mount("/mcp", mcp_app)
