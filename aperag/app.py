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

from aperag.bootstrap import wire_cross_domain_di_seams
from aperag.domains.agent_runtime.api.routes import router as agent_runtime_router
from aperag.domains.conversation.api.openai_routes import router as openai_router
from aperag.domains.conversation.api.routes import (
    bots_router as bots_v2_router,
)
from aperag.domains.conversation.api.routes import (
    chat_router as chat_router,
)
from aperag.domains.evaluation.api.routes import router as evaluation_v2_router
from aperag.domains.governance.api.apikeys_routes import router as apikeys_router
from aperag.domains.governance.api.audit_routes import router as audit_router
from aperag.domains.governance.api.quota_routes import router as quota_router
from aperag.domains.identity.api.auth_routes import router as auth_router
from aperag.domains.identity.api.config_routes import router as config_router
from aperag.domains.knowledge_base.api.export_routes import router as export_router
from aperag.domains.knowledge_base.api.routes import router as knowledge_base_router
from aperag.domains.knowledge_base.api.settings_routes import router as settings_router
from aperag.domains.knowledge_graph.api.routes import router as knowledge_graph_router
from aperag.domains.marketplace.api.routes import router as marketplace_router
from aperag.domains.model_platform.api.llm_routes import router as llm_router
from aperag.domains.model_platform.api.prompts_routes import router as prompts_router
from aperag.domains.model_platform.api.providers_v2_routes import router as providers_v2_router
from aperag.domains.retrieval.api.routes import router as retrieval_router
from aperag.domains.web_access.api.routes import router as web_access_router
from aperag.exception_handlers import register_exception_handlers
from aperag.llm.litellm_track import register_custom_llm_track
from aperag.mcp import mcp_server
from aperag.openapi_spec import custom_generate_unique_id
from aperag.server.health import router as health_router

# Wire every cross-domain DI seam. ``aperag/cli/indexing_worker.py``
# calls the same helper at startup so the API and indexing-worker
# processes share an identical seam set; the boundary test
# ``tests/boundaries/test_worker_di_parity.py`` enforces parity.
wire_cross_domain_di_seams()


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


app.include_router(health_router, prefix="/health")
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
