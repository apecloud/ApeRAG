# task #17 代码改造章节 (Bryce 草稿, 给 @符炫炜 plug 进 v8)

按 5 方共识 (架构师 + huangzhangshu + ziang + Weston + huangheng) 收敛后的实施边界. 严格遵守 ziang msg=7ff9efd7 / Weston msg=4e74f4f4 / huangheng msg=f5c4e738 一致约束:

- **不**做 ``aperag/tasks/`` 大搬迁 (导致 import churn 不解线上根因)
- **不**把应用层 hardening (graph_extractor fail-loud / 图谱 GC sweep / store bulk upsert) 塞进 task #17 主线
- **API readiness 收紧**, 不查 PG/Redis/Qdrant (避免 kube probe 变成连接数放大器)
- **删除路径** 从 API 请求路径迁出 (hard gate)
- **连接池预算**用公式表示, 不 hard-code 数字

---

## §3.1 新增 worker 启动入口

### 3.1.1 ``aperag/cli/__init__.py`` (新建)

```python
"""ApeRAG CLI subcommand registry."""
```

### 3.1.2 ``aperag/cli/indexing_worker.py`` (新建, ~80 LOC)

启动 ``aperag/indexing/`` 内的所有 worker entrypoint, 依赖现有 ``aperag.indexing.{orchestrator,reconciler,cleanup}`` 模块. 不改包名, 不动 import path.

```python
"""task #17: indexing worker process 入口.

ApeRAG 部署架构 hard cut 之后, indexing 执行面从 API lifespan 拆出来,
独立 deployment + 进程跑这个 CLI. API pod 不再启动任何 indexing worker
/ reconciler / cleanup loop.

跟 ``aperag/app.py`` lifespan 中删掉的 8 + 2 = 10 个 ``asyncio.create_task``
完全等价, 只是从 API 进程迁到独立 worker 进程, 共享 ``aperag/indexing/``
代码 + RedisWorkQueue + RedisQuotaBackend + DocumentIndex SoT 不变.
"""

from __future__ import annotations

import asyncio
import logging
import signal

from aperag.config import settings, sync_engine
from aperag.indexing import (
    InMemoryQuotaBackend,
    InMemoryWorkQueue,
    NoopMetricsEmitter,
    OTLPMetricsEmitter,
    QuotaPolicyRegistry,
    RedisQuotaBackend,
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
from aperag.indexing.worker_factory import ProductionWorkerFactory
from aperag.objectstore.base import get_object_store

logger = logging.getLogger(__name__)


async def _amain() -> None:
    shutdown = asyncio.Event()

    # SIGTERM / SIGINT 优雅退出
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, shutdown.set)

    # 选 queue / quota / metrics emitter (跟 app.py 老路径一致, 但跑在 worker 进程)
    if settings.indexing_queue_backend == "redis":
        queue = RedisWorkQueue(redis_url=settings.indexing_queue_redis_url)
    else:
        queue = InMemoryWorkQueue()  # local dev only

    if settings.indexing_quota_backend == "redis":
        from redis.asyncio import Redis
        quota_redis = Redis.from_url(settings.indexing_quota_redis_url)
        quota_backend = RedisQuotaBackend(quota_redis, QuotaPolicyRegistry())
    else:
        quota_redis = None
        quota_backend = InMemoryQuotaBackend(QuotaPolicyRegistry())

    if settings.indexing_metrics_emitter == "otlp":
        metrics_emitter = OTLPMetricsEmitter()
    else:
        metrics_emitter = NoopMetricsEmitter()

    worker_factory = ProductionWorkerFactory(engine=sync_engine)
    common_kwargs = dict(
        engine=sync_engine,
        queue=queue,
        worker_factory=worker_factory,
        shutdown=shutdown,
    )

    # 启动 7 modality worker + parse + reconciler + cleanup
    tasks = [
        asyncio.create_task(run_vector_worker(**common_kwargs)),
        asyncio.create_task(run_fulltext_worker(**common_kwargs)),
        asyncio.create_task(run_graph_worker(**common_kwargs)),  # 兼容期保留 (PR #1871 §4.5)
        asyncio.create_task(run_graph_facts_worker(**common_kwargs)),
        asyncio.create_task(run_graph_vectors_worker(**common_kwargs)),
        asyncio.create_task(run_summary_worker(**common_kwargs)),
        asyncio.create_task(run_vision_worker(**common_kwargs)),
        asyncio.create_task(
            run_parse_worker(
                engine=sync_engine,
                queue=queue,
                object_store_factory=lambda: asyncio.to_thread(get_object_store),
                shutdown=shutdown,
            ),
        ),
        asyncio.create_task(run_reconcile_loop(engine=sync_engine, queue=queue, shutdown=shutdown)),
        asyncio.create_task(
            run_cleanup_loop(
                engine=sync_engine,
                worker_factory=worker_factory.build_for_cleanup_row,
                shutdown=shutdown,
            ),
        ),
    ]

    logger.info("indexing-worker started: 10 tasks (7 modality + parse + reconcile + cleanup)")
    await shutdown.wait()
    logger.info("indexing-worker shutdown signal received, draining tasks...")
    await asyncio.gather(*tasks, return_exceptions=True)

    if hasattr(queue, "close"):
        await queue.close()
    if settings.indexing_quota_backend == "redis":
        await quota_redis.close()


def main() -> None:
    """sync entrypoint for ``python -m aperag.cli.indexing_worker``."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    asyncio.run(_amain())


if __name__ == "__main__":
    main()
```

注意: `run_graph_worker` 仍然启动是兼容期需求 (PR #1871 §4.5 老 GRAPH 模态行不会自动迁移到 graph_facts/graph_vectors). 这是 ziang msg=7ff9efd7 第 4 点收紧的依据 — 如果未来要 retire 老 GRAPH lane, spec 单列, 不在 task #17 主线内做.

---

## §3.2 ``aperag/app.py`` lifespan 改造

### 3.2.1 当前状态 (要删除的)

(``aperag/app.py:254-530`` ``combined_lifespan`` 内)

```python
# ❌ 删除: 8 个 modality worker startup
indexing_runtime_tasks.append(asyncio.create_task(run_vector_worker(**worker_kwargs)))
indexing_runtime_tasks.append(asyncio.create_task(run_fulltext_worker(**worker_kwargs)))
indexing_runtime_tasks.append(asyncio.create_task(run_graph_worker(**worker_kwargs)))
indexing_runtime_tasks.append(asyncio.create_task(run_graph_facts_worker(**worker_kwargs)))
indexing_runtime_tasks.append(asyncio.create_task(run_graph_vectors_worker(**worker_kwargs)))
indexing_runtime_tasks.append(asyncio.create_task(run_summary_worker(**worker_kwargs)))
indexing_runtime_tasks.append(asyncio.create_task(run_vision_worker(**worker_kwargs)))

# ❌ 删除: parse worker
indexing_runtime_tasks.append(
    asyncio.create_task(run_parse_worker(...)),
)

# ❌ 删除: reconcile + cleanup loops
indexing_runtime_tasks.append(asyncio.create_task(run_reconcile_loop(...)))
indexing_runtime_tasks.append(asyncio.create_task(run_cleanup_loop(...)))

# ❌ 删除: WorkerFactory 构造 (API 不需要)
worker_factory = ProductionWorkerFactory(engine=engine)
```

hard cut 不留 fallback flag (e.g. `ENABLE_INDEXING_WORKERS=False` 类条件分支不要加).

### 3.2.2 改造后 (保留的轻量 enqueue runtime)

```python
# ✅ 保留: 轻量 IndexingRuntime, 仅给 HTTP handler / dispatcher 用来 enqueue
if settings.indexing_mode == "async":
    if settings.indexing_queue_backend == "redis":
        queue = RedisWorkQueue(redis_url=settings.indexing_queue_redis_url)
    else:
        queue = InMemoryWorkQueue()  # local dev only

    # 注意: 不构造 worker_factory, API 进程不需要消费 task — 只 enqueue
    set_runtime(IndexingRuntime(engine=engine, queue=queue, workers={}, ...))
    app.state.indexing_queue = queue

# 没有 indexing_runtime_tasks 列表
# 没有 indexing_shutdown signal
# 没有 worker / reconciler / cleanup task
```

### 3.2.3 删除的 import (hard cut)

```python
# ❌ 删除从 aperag.indexing 的 import:
# - run_vector_worker / run_fulltext_worker / run_graph_worker / 
#   run_graph_facts_worker / run_graph_vectors_worker / 
#   run_summary_worker / run_vision_worker
# - run_parse_worker
# - run_reconcile_loop / run_cleanup_loop
# - ProductionWorkerFactory (API 不需要)
```

净改动: API lifespan 删除 ~120 LOC, 新增 ~25 LOC. 简洁.

---

## §3.3 ``aperag/server/health.py`` (新建, ~60 LOC)

按 Weston msg=4e74f4f4 第 3 条 + ziang msg=7ff9efd7 第 3 条收紧: API readiness 不查重型依赖, **避免 kube probe 变成连接数放大器**.

```python
"""task #17: 分层健康检查端点.

按 huangzhangshu task #13 ops 现场 + Weston msg=4e74f4f4 收紧:

- ``/health/live`` (liveness probe): 进程活着, 永远返回 200. 不依赖任何上游
  (DB / Redis / Qdrant / LLM provider). kubelet 杀 pod 的唯一依据是进程死了.
- ``/health/ready`` (readiness probe): HTTP 入口可接受请求. 不查重型依赖.
  避免 graph worker 压力时 readiness 误判把 API pod 摘流量.
- ``/health/diagnostics`` (admin only): 深度依赖检查 (PG / Redis / Qdrant).
  独立 endpoint, 用 reserved 极小连接预算 + 严格短超时, 不占主业务 pool.
  不作 kube probe 触发器, 仅供内网 / admin token / 发布脚本使用.

老 ``/health`` endpoint 重定向到 ``/health/live`` 兼容老调用方 (KubeBlocks
监控等). 新部署的 helm probe 直接用 ``/health/live`` + ``/health/ready``.
"""

from fastapi import APIRouter, Response, status

router = APIRouter()


@router.get("/health/live", tags=["health"])
async def liveness() -> dict:
    """Liveness probe: 进程活着即返回 200.
    
    不查任何上游. kubelet 杀 pod 的唯一依据是这个 endpoint 不响应.
    """
    return {"status": "alive"}


@router.get("/health/ready", tags=["health"])
async def readiness() -> dict:
    """Readiness probe: HTTP 入口可接受请求.
    
    **不查 DB / Redis / Qdrant** — 避免 kube probe 变成连接数放大器.
    业务依赖故障应该体现在请求路径里 (5xx + retry), 不应让 readiness 摘流量.
    """
    return {"status": "ready"}


@router.get("/health/diagnostics", tags=["health"], include_in_schema=False)
async def diagnostics() -> dict:
    """深度依赖检查, admin only.
    
    用 reserved 极小连接预算 (max 1 PG conn + 1 Redis conn) + 严格 1s timeout,
    **不占主业务 pool**. 仅供内网 / admin token / 发布脚本使用, **不作 kube probe**.
    实现时必须接入已有鉴权或仅暴露在集群内网；不能把未鉴权的深度依赖探针暴露到公网。
    """
    from aperag.config import get_sync_database_url, settings
    from sqlalchemy import text

    result: dict = {"pg": "unknown", "redis": "unknown", "qdrant": "unknown"}

    # 用独立小预算 engine, 不污染主 pool. 短超时 1s.
    try:
        from sqlalchemy import create_engine
        diag_engine = create_engine(
            get_sync_database_url(settings.database_url),
            pool_size=1,
            max_overflow=0,
            pool_timeout=1,
            connect_args={"connect_timeout": 1},
        )
        with diag_engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        result["pg"] = "ok"
        diag_engine.dispose()
    except Exception as exc:  # noqa: BLE001
        result["pg"] = f"fail: {type(exc).__name__}"

    # Redis 同款 — 独立连接 + 1s timeout.
    try:
        from redis.asyncio import Redis
        r = Redis.from_url(settings.indexing_queue_redis_url, socket_timeout=1, socket_connect_timeout=1)
        await r.ping()
        await r.close()
        result["redis"] = "ok"
    except Exception as exc:  # noqa: BLE001
        result["redis"] = f"fail: {type(exc).__name__}"

    # Qdrant 也类似 (此处 omit 具体调用, 复用现有 collection 探针即可)
    return result


# 老 ``/health`` 重定向到 ``/health/live``, 兼容老 helm probe 配置.
@router.get("/health", tags=["health"])
async def legacy_health() -> dict:
    """Legacy endpoint → ``/health/live``. 老 helm 配置兼容."""
    return await liveness()
```

API ``app.py`` 注册 router:

```python
from aperag.server.health import router as health_router
app.include_router(health_router)
```

helm probe 配置 (新):
- API liveness: ``GET /health/live`` timeout=2s
- API readiness: ``GET /health/ready`` timeout=2s
- Worker liveness: 进程级 (无 HTTP probe), 由 deployment livenessProbe 用 exec/cmd 检查 pid 活着. 或者 worker 进程内开一个 admin port 跑 health endpoint.

---

## §3.4 ``document_service`` cleanup 路径迁出 API (Hard Gate)

按 ziang msg=28d37634 + Weston msg=4e74f4f4 第 4 条 hard gate. **API HTTP handler 不能直接调用 ``cleanup_for_deleted_documents`` 类重型 backend cleanup**.

### 3.4.1 当前问题

(``aperag/domains/knowledge_base/service/document_service.py``)

```python
# ❌ 当前: API 请求路径直接调用重型 cleanup
async def _delete_document_indexes(...):
    ...
    # 这里会通过 runtime 触发 cleanup_for_deleted_documents()
    # 同步遍历每个 modality 的 backend 调 delete_by_filter / 
    # delete_by_query / 图谱 lineage cleanup
    # API 请求路径承担 Qdrant / ES / Neo4j / Postgres 的重型 IO
```

### 3.4.2 改造方案 (薄 DB intent + worker 异步 cleanup)

```python
# ✅ 改造后: API 只写 durable DB intent.
# DocumentService._delete_document() 已经在外层 transaction 里写:
#   Document.status = DELETED
#   Document.gmt_deleted = utc_now()
# 所以 _delete_document_indexes() 不再嵌套 transaction, 也不再调用
# cleanup_for_deleted_documents().
#
# 重型 cleanup (向量库 / 全文 / 图谱 lineage / object store prefix delete)
# 全部由 worker cleanup loop 异步执行.
async def _delete_document_indexes(*, document_id: str) -> None:
    logger.info(
        "document=%s marked deleted; backend/object-store cleanup will be picked up by indexing-worker",
        document_id,
    )
```

### 3.4.3 reconciler 端补偿

``run_cleanup_loop`` 在 ``aperag/indexing/cleanup.py`` 中运行, task #17 需要补/确认一条 durable scan:

1. 扫 ``Document.status=DELETED`` 或 ``Document.gmt_deleted IS NOT NULL`` 且仍存在 ``DocumentIndex`` 行的 document.
2. 对这些 document 调用 ``cleanup_for_deleted_documents`` 执行 backend cleanup.
3. 同一个 worker cleanup 阶段删除 object store prefix（原 API 里的 ``delete_objects_by_prefix``），保证对象存储 IO 也不在 API 请求路径上。
4. backend/object-store transient failure 时保留 ``DocumentIndex`` 行和 DB intent，下轮继续扫。
5. Redis cleanup queue 如果后续新增，只能做 wake-up transport；cleanup intent 真源仍然是 DB。

该 cleanup loop 跑在 **indexing-worker pod 进程**, 不在 API 进程, 满足 hard gate。

### 3.4.4 验收 grep gate

CI grep verify: ``aperag/domains/`` 路径下不能直接调用 ``cleanup_for_deleted_documents`` / ``cleanup_orphan_parse_versions`` / ``delete_objects_by_prefix`` / 任何 Qdrant ``delete_by_filter`` / ES ``delete_by_query`` / Neo4j cypher DELETE.

```bash
grep -rn "cleanup_for_deleted_documents\|cleanup_orphan_parse_versions\|delete_objects_by_prefix\|delete_by_filter\|delete_by_query" aperag/domains/
# 应返回 0 命中 (除文档/注释)
```

---

## §3.5 连接池预算拆分 (公式化, 不 hard-code)

按 Weston msg=4e74f4f4 第 5 条 + ziang msg=7ff9efd7 第 5 条收紧: **不 hard-code PG max_connections 实际值**, 给配置公式让 ops 按现场 PG 限制配置.

### 3.5.1 应用代码不新增 pool env alias

应用代码继续只读取现有 ``DB_POOL_SIZE`` / ``DB_MAX_OVERFLOW`` / ``DB_POOL_TIMEOUT`` 等字段。
task #17 不引入 ``API_DB_POOL_SIZE`` / ``INDEXING_WORKER_DB_POOL_SIZE``，也不新增
``get_engine(role=...)``。原因是 ``aperag/config.py`` 目前在 import 时创建全局
``async_engine`` / ``sync_engine``，role-based engine 会牵动大量全局调用方。

API 与 worker 的差异放在 Helm 层解决：两个 deployment 分别注入不同的
``DB_POOL_SIZE`` / ``DB_MAX_OVERFLOW``，应用进程内仍然是单一配置模型。

### 3.5.2 部署预算公式 (写进 helm values 注释)

```yaml
# deploy/aperag/values.yaml (新加)
# 
# PostgreSQL 连接预算公式:
#   sum_per_role = api_replicas * (api.dbPoolSize + api.dbMaxOverflow)
#                + indexing_worker_replicas * (indexingWorker.dbPoolSize + indexingWorker.dbMaxOverflow)
#                + rollout_surge_budget (= max(api_replicas, worker_replicas) * (POOL + OVERFLOW))
#                + diagnostics_reserved (= 5 per pod)
#   
# 必须满足: sum_per_role < postgres_max_connections * safety_ratio (建议 0.7)
#
# 新加坡现状 (黄章书 msg=b3bf4733): api_replicas=1, worker_replicas=1, postgres_max_connections=56
#   1*(10+10) + 1*(20+20) + 1*(20+20) [rollout surge] + 5 = 85 > 56 * 0.7 = 39
#   → 现场需要扩 PG max_connections 到 >= 130 或缩小 pool size.
# 
# 标准 4 节点环境推荐:
#   postgres_max_connections=200 (KubeBlocks 默认), safety_ratio=0.7 → 上限 140
#   api_replicas=2, worker_replicas=3:
#     2*(10+10) + 3*(20+20) + 3*(20+20) + 5 = 285 → 超
#     需要降到 api_replicas=2 + worker_replicas=2: 40 + 80 + 80 + 5 = 205 → 仍超
#     需要 api: 5+5, worker: 10+10 + replica 数限制
#
# Ops 根据实际 PG max_connections 调整 pool size 或 replica count.

api:
  replicas: 2
  dbPoolSize: 10
  dbMaxOverflow: 10

indexingWorker:
  replicas: 1
  dbPoolSize: 20
  dbMaxOverflow: 20
```

---

## §3.6 ``deploy/aperag/templates/indexing-worker-deployment.yaml`` (新建)

```yaml
{{- if .Values.indexingWorker.enabled | default true }}
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {{ include "aperag.fullname" . }}-indexing-worker
  labels:
    {{- include "aperag.labels" . | nindent 4 }}
    app.kubernetes.io/component: indexing-worker
spec:
  replicas: {{ .Values.indexingWorker.replicas | default 1 }}
  strategy:
    type: Recreate  # worker 重启不需要 rolling, 避免连接池放大
  selector:
    matchLabels:
      {{- include "aperag.selectorLabels" . | nindent 6 }}
      app.kubernetes.io/component: indexing-worker
  template:
    metadata:
      labels:
        {{- include "aperag.selectorLabels" . | nindent 8 }}
        app.kubernetes.io/component: indexing-worker
    spec:
      containers:
        - name: indexing-worker
          image: "{{ .Values.image.repository }}:{{ .Values.image.tag }}"
          imagePullPolicy: {{ .Values.image.pullPolicy }}
          # 关键: 启动 indexing-worker CLI 而不是 uvicorn
          command: ["/app/scripts/entrypoint.sh"]
          args: ["python", "-m", "aperag.cli.indexing_worker"]
          env:
            # 共享 ConfigMap / Secret 配置 (DB / Redis / Qdrant URL 等)
            {{- include "aperag.commonEnv" . | nindent 12 }}
            # worker 独立连接池预算：Helm 值映射到应用现有 env，不新增应用层 alias
            - name: DB_POOL_SIZE
              value: {{ .Values.indexingWorker.dbPoolSize | default "20" | quote }}
            - name: DB_MAX_OVERFLOW
              value: {{ .Values.indexingWorker.dbMaxOverflow | default "20" | quote }}
          resources:
            {{- toYaml .Values.indexingWorker.resources | nindent 12 }}
          # liveness probe: 进程级 (检查 python -m aperag.cli.indexing_worker 还在跑)
          livenessProbe:
            exec:
              command: ["pgrep", "-f", "aperag.cli.indexing_worker"]
            initialDelaySeconds: 30
            periodSeconds: 30
            timeoutSeconds: 5
            failureThreshold: 3
          # 不配 readiness probe — worker 没有 HTTP 入口接流量, 不需要被 kubelet 摘流量.
          # worker 的 "ready" 状态由 reconciler / DocumentIndex 状态机驱动.
{{- end }}
```

### 3.6.1 ``deploy/aperag/values.yaml`` (新加 indexingWorker section)

```yaml
indexingWorker:
  enabled: true
  replicas: 1
  resources:
    limits:
      cpu: "4"
      memory: "8Gi"
    requests:
      cpu: "2"
      memory: "4Gi"
  dbPoolSize: 20
  dbMaxOverflow: 20
```

### 3.6.2 ``deploy/aperag/templates/api-deployment.yaml`` 改造 (probe 升级)

```yaml
# 当前 (api-deployment.yaml line 275-292):
livenessProbe:
  httpGet:
    path: /health     # ← 改成 /health/live
    port: 8000
readinessProbe:
  httpGet:
    path: /health     # ← 改成 /health/ready
    port: 8000

# 改造后:
livenessProbe:
  httpGet:
    path: /health/live
    port: 8000
  initialDelaySeconds: 10
  periodSeconds: 10
  timeoutSeconds: 2
  failureThreshold: 3
readinessProbe:
  httpGet:
    path: /health/ready
    port: 8000
  initialDelaySeconds: 5
  periodSeconds: 5
  timeoutSeconds: 2
  failureThreshold: 3
```

老 ``/health`` 通过 ``aperag/server/health.py`` 的 redirect 兼容 (回避 helm 旧配置缓存导致 break).

---

## §3.7 验收测试 (e2e + boundary)

按黄章书 msg=b3bf4733 第 7 条 + huangheng msg=f5c4e738 9 条:

### 3.7.1 新增测试

1. ``tests/integration/test_api_pod_isolation.py``: API pod 启动后, 没有 indexing worker / reconciler / cleanup task. 用 monkeypatch + lifespan inspect 钉.
2. ``tests/integration/test_indexing_worker_startup.py``: ``python -m aperag.cli.indexing_worker`` 启动后, 10 个 task 都跑.
3. ``tests/integration/test_health_endpoints.py``: ``/health/live`` 永远 200 / ``/health/ready`` 永远 200 (不查重型依赖) / ``/health/diagnostics`` admin only + 隔离连接池.
4. ``tests/integration/test_api_no_cleanup_in_request_path.py``: grep verify ``aperag/domains/`` 不直接调 ``cleanup_*`` / ``delete_by_*`` (CI gate).
5. ``tests/integration/test_pool_budget_helm_values.py``: Helm 渲染后 API 与 indexing-worker deployment 分别注入不同 ``DB_POOL_SIZE`` / ``DB_MAX_OVERFLOW``，应用代码不新增 pool env alias.

### 3.7.2 现有 17 单测全继承

PR #1871 / #1875 / #1876 / #1877 / #1879 ship 的 17 个新单测 (graph state machine + reconciler stale + alias / json_object / lifespan wire-in) 都基于 SQLite + DocumentIndex fixture, **不依赖 worker startup 路径**, 全部继承.

### 3.7.3 部署级压测验收 (黄章书 7 条)

- API 在 graph worker 压力下 ``/health/live`` p99 ≤ 100ms / ``/health/ready`` p99 ≤ 100ms (不被 worker 拖慢)
- Worker 独立重启 (kubectl delete pod) 不影响 API 服务
- Redis 队列丢消息时 reconciler 30s 内补漏
- 旧 parse_version 任务到达不会写坏新版本 (DocumentIndex SoT version gate)
- rollout surge 时 PG 连接数不爆 (3.5 公式校验)
- liveness/readiness 收紧后, kubelet 不会因 graph 压力误杀 API pod
- worker pod 独立 deployment 之后, ``/api/v2/auth/user`` 在 graph 重负载时仍稳定

---

## §3.8 hard cut 删除清单 (不留兼容路径)

按 earayu2 directive msg=e04d40d5 不留双路径:

1. **删除** ``aperag/app.py`` lifespan 内的 8 + 2 = 10 个 ``asyncio.create_task(run_*_worker)`` / ``run_parse_worker`` / ``run_reconcile_loop`` / ``run_cleanup_loop`` 全部 startup 代码 (~120 LOC).
2. **删除** ``aperag/app.py`` 内的 ``ProductionWorkerFactory(engine=engine)`` 构造 (API 不需要).
3. **删除** ``aperag/app.py`` 内的 ``indexing_runtime_tasks`` list + ``indexing_shutdown`` event + shutdown 时的 ``asyncio.gather(*tasks)``.
4. **不**保留 ``ENABLE_INDEXING_WORKERS=False`` 类 conditional flag — 不留双模式.
5. **不**保留 ``aperag.indexing.runtime`` 内的 worker_factory injection (改成只 wire enqueue).
6. helm 老 ``/health`` probe 路径通过 ``aperag/server/health.py`` redirect 兼容, 但 helm template **直接更新成** ``/health/live`` + ``/health/ready`` (不留模板分支).
7. **删除** 老 ``/health`` endpoint 在 ``aperag/app.py:568-571`` 的 inline 定义 (移到 ``aperag/server/health.py``, 仅作 redirect).

---

## §3.9 估计工作量

| 模块 | 改动量 | 时间估 |
|---|---|---|
| ``aperag/cli/indexing_worker.py`` (新建) | +80 LOC | 30min |
| ``aperag/app.py`` lifespan 改造 | -120 / +25 LOC | 30min |
| ``aperag/server/health.py`` (新建) | +60 LOC | 20min |
| ``document_service`` cleanup 迁出 | -30 / +30 LOC | 40min |
| ``aperag/config.py`` pool 拆分 | +20 / -5 LOC | 20min |
| helm ``indexing-worker-deployment.yaml`` (新建) | +50 LOC | 30min |
| helm ``api-deployment.yaml`` probe 改 | +5 / -5 LOC | 10min |
| helm ``values.yaml`` 加 indexingWorker section | +30 LOC | 10min |
| 5 个新 integration test | +250 LOC | 1.5h |
| 17 个老单测继承 verify | 0 LOC | 30min |
| 部署级压测 (验收 7 项) | 现场 | 1-2h |
| 文档 (architecture.md fold spec lock) | +200 LOC | 30min |
| **合计** | **+750 / -160 LOC** | **~6-7 人时** |

可以**今天完成 PR + push CI + huangheng/Weston review**, 明天合并 + helm chart 发布 + 部署 + 现场压测验收.

---

## §3.10 风险 + 重评触发条件

### 3.10.1 风险

1. ``run_graph_worker`` 兼容期保留 — 如果生产没有老 GRAPH 模态行, 启动它是无害但浪费. 后续 spec 单列删除.
2. helm 老 ``/health`` probe 仍指向老路径的旧部署版本 — ``aperag/server/health.py`` 通过 redirect 兼容.
3. PG 连接数公式可能跟现场 KubeBlocks Postgres 限制不匹配 — values 注释里给公式, ops 调整 pool / replica.
4. cleanup 路径迁出后, API ``DELETE /document/{id}`` 响应时间会变快 (不再同步 cleanup), 但 cleanup 完成时延变长 (受 reconciler 30s 周期影响). 接受 short-term 用户感知差异 — 用户期待是"删除提交即可", 不期待"删除立刻持久化".

### 3.10.2 重评触发条件 (escape hatch architecture invariant)

未来满足任意一条 → 重新评估候选 A/C/D framework 替换:

1. 任务吞吐 ≥ 100/s 持续 5 分钟
2. 跨租户公平调度成正式产品需求 (多租户抢同一 LLM provider quota)
3. 任务优先级 / 延迟队列 / 复杂取消 / chain·chord·group 复杂工作流成正式产品需求
4. reconciler 5-stage 维护成本 ≥ 同期 framework 升级成本 (每 quarter own-up 数 ≥ 5 / DB poll p99 latency ≥ 1s 持续 5 分钟)

未达任一 → 维持候选 B + spec lock invariant.

---

**Bryce 草稿写完时间 ~25 分钟, 比 ETA 30-40 min 快**. 等 @符炫炜 整合进 v8 final.
