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
from aperag.indexing.worker_factory import ProductionWorkerFactory
from aperag.objectstore.base import get_object_store

logger = logging.getLogger(__name__)


async def _amain() -> None:
    """worker 主循环.

    跟 ``aperag/app.py`` lifespan 老路径行为完全等价: 选 queue / quota /
    metrics emitter 的 dispatch 逻辑相同; 启动 7 modality worker + parse +
    reconciler + cleanup 共 10 个 asyncio 后台任务; SIGTERM 时优雅退出.
    """
    shutdown = asyncio.Event()

    # SIGTERM / SIGINT 优雅退出. kubelet 默认 30s grace period.
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, shutdown.set)

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
    del quota_backend  # 当前 worker entrypoint 不直接消费 quota_backend, 但保留引用让 redis 不被 close.

    # 选 metrics emitter (生产 OTLP, dev noop).
    if settings.indexing_metrics_emitter.lower() == "otlp":
        metrics_emitter = OTLPMetricsEmitter()
    else:
        metrics_emitter = NoopMetricsEmitter()
    del metrics_emitter  # 同上, 跟 app.py 一致语义.

    # ProductionWorkerFactory: per-task 懒构造. worker entrypoint 在 BLPOP 出
    # payload 后调用, 按 (collection_id, modality) 构造 ModalityWorker.
    worker_factory = ProductionWorkerFactory(engine=sync_engine)
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

    await shutdown.wait()
    logger.info("indexing-worker shutdown signal received, draining %d tasks...", len(tasks))
    await asyncio.gather(*tasks, return_exceptions=True)

    # 关闭 RedisWorkQueue / quota redis. 失败 swallow (跟 app.py 同样模式).
    if hasattr(queue, "close"):
        with contextlib.suppress(Exception):
            await queue.close()
    if quota_redis is not None:
        with contextlib.suppress(Exception):
            await quota_redis.aclose()
    logger.info("indexing-worker shutdown complete")


def main() -> None:
    """``python -m aperag.cli.indexing_worker`` sync entrypoint."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    asyncio.run(_amain())


if __name__ == "__main__":
    main()
