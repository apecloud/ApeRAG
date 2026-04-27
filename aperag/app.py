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
from aperag.domains.model_platform.api.providers_v3_routes import router as providers_v3_router
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
    indexing_runtime_tasks: list[asyncio.Task[None]] = []
    indexing_shutdown: asyncio.Event | None = None

    if settings.indexing_mode == "async":
        # Lazy imports — pulling the indexing runtime symbols at app
        # start-up time keeps ``aperag/app.py`` cold-start fast and
        # confines the import surface to the wired branch.
        from aperag.config import sync_engine
        from aperag.indexing import (
            InMemoryWorkQueue,
            NoopMetricsEmitter,
            OTLPMetricsEmitter,
            RedisWorkQueue,
            run_cleanup_loop,
            run_fulltext_worker,
            run_graph_worker,
            run_reconcile_loop,
            run_summary_worker,
            run_vector_worker,
            run_vision_worker,
        )

        indexing_shutdown = asyncio.Event()
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
            metrics_emitter = OTLPMetricsEmitter()
        else:
            metrics_emitter = NoopMetricsEmitter()

        # Worker factory — per-task lazy construction. The async
        # worker entrypoints (``run_*_worker``) call this closure on
        # every BLPOP'd payload to materialise the concrete
        # :class:`ModalityWorker` for that ``(collection, modality)``
        # pair. ``ProductionWorkerFactory`` resolves the collection
        # row, picks the right backend (Qdrant / Elasticsearch +
        # configured embedder / completion model), and constructs the
        # worker — all backed by the existing build helpers
        # (``get_collection_embedding_service_sync`` /
        # ``get_vector_db_connector`` / ``get_object_store``) so this
        # is composition, not re-implementation. Construction failures
        # raise :class:`WorkerFactoryError`; the orchestrator runner
        # catches that and finalises the row FAILED so §I.2 retry
        # picks it up. Per architect msg=7782ebe0.
        from aperag.indexing.worker_factory import ProductionWorkerFactory

        worker_factory = ProductionWorkerFactory(engine=engine)

        worker_kwargs = dict(
            engine=engine,
            queue=queue,
            worker_factory=worker_factory,
            shutdown=indexing_shutdown,
        )
        indexing_runtime_tasks.append(asyncio.create_task(run_vector_worker(**worker_kwargs)))
        indexing_runtime_tasks.append(asyncio.create_task(run_fulltext_worker(**worker_kwargs)))
        indexing_runtime_tasks.append(asyncio.create_task(run_graph_worker(**worker_kwargs)))
        indexing_runtime_tasks.append(asyncio.create_task(run_summary_worker(**worker_kwargs)))
        indexing_runtime_tasks.append(asyncio.create_task(run_vision_worker(**worker_kwargs)))
        indexing_runtime_tasks.append(
            asyncio.create_task(
                run_reconcile_loop(
                    engine=engine,
                    queue=queue,
                    shutdown=indexing_shutdown,
                )
            )
        )
        indexing_runtime_tasks.append(
            asyncio.create_task(
                run_cleanup_loop(
                    engine=engine,
                    workers={},  # T3.3 follow-up: pass concrete worker registry
                    shutdown=indexing_shutdown,
                )
            )
        )

        # Stash on app state so request handlers can dispatch via the
        # same queue / engine the workers consume.
        app.state.indexing_queue = queue
        app.state.indexing_engine = engine
        app.state.indexing_metrics_emitter = metrics_emitter

        # Service-layer callers (aperag/domains/**) consume the same
        # triple through the process-wide IndexingRuntime singleton —
        # they don't have a Request handle for app.state. Workers map
        # is empty in the async-default deployment; T3.3 follow-up
        # populates concrete factories per modality.
        from aperag.indexing.runtime import IndexingRuntime, set_runtime

        set_runtime(
            IndexingRuntime(
                engine=engine,
                queue=queue,
                workers={},
                metrics_emitter=metrics_emitter,
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
        if indexing_shutdown is not None:
            indexing_shutdown.set()
        if indexing_runtime_tasks:
            # Drain in-flight worker / reconciler / cleanup loops with
            # a short grace window so a SIGTERM does not abort mid-task.
            await asyncio.gather(*indexing_runtime_tasks, return_exceptions=True)
        # Wave 4 T4: release the indexing queue's underlying connection
        # pool (Redis client owns one); InMemoryWorkQueue has no
        # ``close`` so guard with hasattr.
        queue_obj = getattr(app.state, "indexing_queue", None)
        if queue_obj is not None and hasattr(queue_obj, "close"):
            with contextlib.suppress(Exception):
                await queue_obj.close()


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
app.include_router(providers_v3_router, prefix="/api/v3")  # Model platform: model accounts / models / model uses
app.include_router(knowledge_base_router, prefix="/api/v2")  # KB domain router (collections_v2 + documents_v2)

# Mount the MCP server at /mcp path
app.mount("/mcp", mcp_app)
