"""task #17 (PR #1884): 独立 indexing worker 进程入口.

ApeRAG 部署架构 hard cut. ``aperag/app.py`` lifespan 不再启动任何
indexing worker / reconciler / cleanup loop. 取而代之, ``indexing-worker``
deployment 跑这个 CLI, 进程内启动跟原 lifespan 等价的 10 个 asyncio
后台任务 (7 modality worker + parse + reconciler + cleanup).

为什么拆 deployment:
* 新加坡 503 根因 (huangzhangshu task #13 + msg=b3bf4733): API + 重型
  indexing worker 共进程, graph 索引压力把 ``/health`` / 事件循环 / 线程池
  / DB 连接池一起拖死, kubelet 杀 pod, ALB 503.
* hard cut 拆开后: API pod 只做 HTTP 路由 + 轻量入队, 不再受 worker 资源
  压力影响; ``/health/live`` / ``/health/ready`` 永远稳定.
* DocumentIndex 仍是业务状态真源, RedisWorkQueue 跨进程 transport, 跟
  既有 reconciler 5-stage / RedisQuotaBackend 完全兼容 (per ziang
  msg=5eedb951 代码审计).

跟 ``aperag/app.py`` 老 lifespan 行为完全等价 — 选 queue / quota / metrics
emitter 的 dispatch 逻辑跟 app.py 同款, 只是搬到独立进程, 不引入新概念.

启动: ``python -m aperag.cli.indexing_worker`` (Helm
``indexing-worker-deployment.yaml`` 的 args).

退出: SIGTERM / SIGINT 触发 graceful shutdown — 等所有 in-flight 任务
drain 完, 关闭 RedisWorkQueue / quota redis, 退出.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import signal

# ziang msg=4ea65100 #1: 用现有 module-level ``settings``, 不引 ``get_settings()`` helper.
from aperag.bootstrap import wire_cross_domain_di_seams
from aperag.config import settings, sync_engine
from aperag.indexing import (
    InMemoryWorkQueue,
    NoopMetricsEmitter,
    OTLPMetricsEmitter,
    RedisWorkQueue,
    run_cleanup_loop,
    run_fulltext_worker,
    run_graph_facts_worker,
    run_graph_vectors_worker,
    run_graph_worker,
    run_parse_worker,
    run_reconcile_loop,
    run_summary_worker,
    run_vector_worker,
    run_vision_worker,
)

# ziang #1: ``ProductionWorkerFactory`` 从 ``aperag.indexing.worker_factory`` 直接 import,
# 不通过 ``aperag.indexing`` 间接 (跟 app.py 现有写法一致).
from aperag.indexing.quota import (
    InMemoryQuotaBackend,
    QuotaPolicyRegistry,
    RedisQuotaBackend,
)
from aperag.indexing.runtime import IndexingRuntime, set_runtime
from aperag.indexing.worker_factory import ProductionWorkerFactory
from aperag.llm.litellm_track import register_custom_llm_track
from aperag.objectstore.base import get_object_store
from aperag.observability import (
    build_observability_config,
    configure_logging,
    configure_process_observability,
)
from aperag.observability.metrics import shutdown_metrics_provider

logger = logging.getLogger(__name__)


def _install_worker_runtime(
    *,
    engine,
    queue,
    metrics_emitter,
    quota_backend,
    worker_factory: ProductionWorkerFactory,
) -> None:
    """Install the process-local runtime used by worker-side graph helpers.

    The API installs its own enqueue-only runtime in ``app.py``. The
    standalone worker process needs the same process-local singleton so
    graph workers can resolve ``DocumentIndex.tenant_scope_key`` through
    ``aperag.indexing.worker_factory._resolve_tenant_scope_key``.
    """

    set_runtime(
        IndexingRuntime(
            engine=engine,
            queue=queue,
            workers={},
            metrics_emitter=metrics_emitter,
            cleanup_worker_factory=worker_factory.build_for_cleanup_row,
            quota_backend=quota_backend,
        )
    )


async def _amain() -> None:
    """worker 主循环.

    跟 ``aperag/app.py`` lifespan 老路径行为完全等价: 选 queue / quota /
    metrics emitter 的 dispatch 逻辑相同; 启动 7 modality worker + parse +
    reconciler + cleanup 共 10 个 asyncio 后台任务; SIGTERM 时优雅退出.

    跨进程 DI parity: ``wire_cross_domain_di_seams()`` 跟 ``app.py`` 启动
    时调的同一个 helper, 把 10 个 cross-domain Protocol seam 全 wire 上
    (PromptTemplateOps / KB marketplace + quota + search_pipeline / conv
    quota / agent_runtime / model_platform prompt CRUD / identity init).
    PR #1884 task #17 hard cut 把 worker 拆出 API 进程时漏 port 这步,
    worker 一走 collection summary regen / agent runtime / quota 路径就
    挂在 ``*Ops not wired`` AttributeError. boundary test
    ``tests/boundaries/test_worker_di_parity.py`` 钉死两进程同集合.
    """
    shutdown = asyncio.Event()

    # SIGTERM / SIGINT 优雅退出. kubelet 默认 30s grace period.
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, shutdown.set)

    # 跨进程 DI parity. 必须先于任何 worker entrypoint 启动 — 否则
    # parse/summary/cleanup 路径 BLPOP 出 payload 后调
    # ``_get_*_ops()`` 时 raise.
    wire_cross_domain_di_seams()

    # ziang msg=4ea65100 #2 + 跟 app.py 现有写法一致: ``QuotaPolicyRegistry``
    # 直接构造, ``RedisQuotaBackend(quota_redis, quota_registry)`` /
    # ``InMemoryQuotaBackend(quota_registry)``.
    quota_registry = QuotaPolicyRegistry()
    quota_redis = None

    # 选 queue (生产 redis, dev inmemory). 跟 app.py 同款 dispatch.
    if settings.indexing_queue_backend.lower() == "redis":
        queue = RedisWorkQueue(redis_url=settings.indexing_queue_redis_url)
    else:
        queue = InMemoryWorkQueue()

    # 选 quota backend (生产 redis 跨副本共享 token bucket, dev inmemory).
    if settings.indexing_quota_backend.lower() == "redis":
        from redis import asyncio as redis_asyncio

        quota_redis = redis_asyncio.from_url(
            settings.indexing_quota_redis_url,
            encoding="utf-8",
            decode_responses=False,
        )
        quota_backend = RedisQuotaBackend(quota_redis, quota_registry)
    else:
        quota_backend = InMemoryQuotaBackend(quota_registry)
    # quota_backend / metrics_emitter 当前 worker entrypoint 不直接 acquire (跟 app.py
    # 现状一致, 真正 acquire 接线点是 task #24). 保留构造让 quota_redis 引用不被
    # 提前回收, 也保证 OTLP MeterProvider 在 worker 启动时就 install.
    _ = quota_backend

    # 选 metrics emitter (生产 OTLP, dev noop).
    if settings.indexing_metrics_emitter.lower() == "otlp":
        metrics_emitter = OTLPMetricsEmitter()
    else:
        metrics_emitter = NoopMetricsEmitter()
    _ = metrics_emitter

    # ProductionWorkerFactory: per-task 懒构造. worker entrypoint 在 BLPOP 出
    # payload 后调用, 按 (collection_id, modality) 构造 ModalityWorker.
    worker_factory = ProductionWorkerFactory(engine=sync_engine)
    _install_worker_runtime(
        engine=sync_engine,
        queue=queue,
        metrics_emitter=metrics_emitter,
        quota_backend=quota_backend,
        worker_factory=worker_factory,
    )
    worker_kwargs = dict(
        engine=sync_engine,
        queue=queue,
        worker_factory=worker_factory,
        shutdown=shutdown,
    )

    async def _resolve_object_store():
        """sync ``get_object_store`` 的 async wrapper, 跟 app.py 行为一致."""
        return await asyncio.to_thread(get_object_store)

    # 启动 10 个后台任务. 顺序跟 app.py lifespan 一致 (per ziang msg=7ff9efd7
    # #4 含 legacy graph lane).
    tasks: list[asyncio.Task[None]] = [
        asyncio.create_task(run_vector_worker(**worker_kwargs)),
        asyncio.create_task(run_fulltext_worker(**worker_kwargs)),
        # legacy ``graph`` lane 兼容期保留 (PR #1871 §4.5 老 GRAPH 模态行
        # 不会自动迁移到 graph_facts/graph_vectors). 删除留给单独 spec.
        asyncio.create_task(run_graph_worker(**worker_kwargs)),
        asyncio.create_task(run_graph_facts_worker(**worker_kwargs)),
        asyncio.create_task(run_graph_vectors_worker(**worker_kwargs)),
        asyncio.create_task(run_summary_worker(**worker_kwargs)),
        asyncio.create_task(run_vision_worker(**worker_kwargs)),
        asyncio.create_task(
            run_parse_worker(
                engine=sync_engine,
                queue=queue,
                object_store_factory=_resolve_object_store,
                shutdown=shutdown,
            ),
        ),
        asyncio.create_task(
            run_reconcile_loop(engine=sync_engine, queue=queue, shutdown=shutdown),
        ),
        asyncio.create_task(
            run_cleanup_loop(
                engine=sync_engine,
                worker_factory=worker_factory.build_for_cleanup_row,
                shutdown=shutdown,
            ),
        ),
    ]

    logger.info(
        "indexing-worker started: 10 tasks (vector/fulltext/graph/graph_facts/"
        "graph_vectors/summary/vision/parse/reconciler/cleanup)"
    )

    try:
        await shutdown.wait()
        logger.info("indexing-worker shutdown signal received, draining %d tasks...", len(tasks))

        # cuiwenbo msg=f7868d2c #3 + Weston msg=ce324047 #2: 给 graceful drain 设 25s 上限
        # (kubelet 默认 30s grace), 超时强制 cancel, 避免某个 worker 不响应 shutdown
        # event 时挂到 SIGKILL 丢 in-flight 状态.
        try:
            await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=25.0,
            )
        except asyncio.TimeoutError:
            logger.warning("indexing-worker shutdown timeout after 25s, cancelling %d tasks", len(tasks))
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

        # cuiwenbo msg=f7868d2c #2 + Weston msg=ce324047 #1: flush OTLP MeterProvider,
        # 否则 PeriodicExportingMetricReader 残留样本会随 worker 退出丢失 (跟 app.py
        # finally 段调 shutdown_metrics_provider 等价).
        with contextlib.suppress(Exception):
            await asyncio.to_thread(shutdown_metrics_provider)

        # 关闭 RedisWorkQueue / quota redis. 失败 swallow (跟 app.py 同样模式).
        if hasattr(queue, "close"):
            with contextlib.suppress(Exception):
                await queue.close()
        if quota_redis is not None:
            with contextlib.suppress(Exception):
                await quota_redis.aclose()
    finally:
        set_runtime(None)
    logger.info("indexing-worker shutdown complete")


def main() -> None:
    """``python -m aperag.cli.indexing_worker`` sync entrypoint.

    Process-level setup (Weston 三分类 类 2): same observability +
    LiteLLM tracking that ``aperag/app.py`` configures at module load.
    Worker process must mirror these so structured logs / OTLP traces /
    LiteLLM custom-track callbacks match between API and worker. The
    only ``app.py`` calls intentionally **not** mirrored are
    ``configure_fastapi`` and ``register_exception_handlers`` (类 3
    FastAPI-only).
    """
    observability_config = build_observability_config(settings)
    configure_logging(observability_config)
    configure_process_observability(observability_config)
    register_custom_llm_track()

    asyncio.run(_amain())


if __name__ == "__main__":
    main()
