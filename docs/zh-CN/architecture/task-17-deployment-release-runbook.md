# task #17 部署、发布、回滚执行方案

本文是 task #17 的部署/SRE 执行 runbook。代码实现开始前，先把部署边界、连接池预算、probe 语义、发布步骤和回滚策略锁进 PR，避免实现阶段把执行面隔离做成不完整的半切换。

## 1. 目标稳态

### 1.1 Deployment 拓扑

task #17 完成后的稳态只有两个应用执行面：

| Deployment | 职责 | 启动命令 | 是否消费 indexing queue |
|---|---|---|---|
| `api` | HTTP API、鉴权、普通请求处理、轻量 enqueue | `/app/scripts/entrypoint.sh /app/scripts/start-api.sh` | 否，只能 enqueue |
| `indexing-worker` | parse、vector、fulltext、graph、graph_facts、graph_vectors、summary、vision、reconciler、cleanup | `python -m aperag.cli.indexing_worker` | 是 |

API 进程不得启动任何 heavy indexing execution plane：

- 不启动 `run_vector_worker`
- 不启动 `run_fulltext_worker`
- 不启动 `run_graph_worker`
- 不启动 `run_graph_facts_worker`
- 不启动 `run_graph_vectors_worker`
- 不启动 `run_summary_worker`
- 不启动 `run_vision_worker`
- 不启动 `run_parse_worker`
- 不启动 `run_reconcile_loop`
- 不启动 `run_cleanup_loop`

`indexing-worker` 必须覆盖当前 `aperag/app.py` lifespan 里已有的全部 worker lane。legacy `graph` lane 继续保留；如果未来要删除，必须单独论证并补迁移测试，不在 task #17 中顺手删除。

### 1.2 状态真源

PostgreSQL 中的业务状态仍是唯一真源：

- `DocumentIndex` 决定 indexing task 的业务状态。
- `Document.status` / `Document.gmt_deleted` 加 remaining `DocumentIndex` rows 决定 cleanup intent。
- Redis queue 只作为可丢失 transport，不是业务状态真源。

Redis 消息丢失时，worker/reconciler/cleanup loop 必须能从 DB 状态补回。

### 1.3 全局并发 / quota 口径

task #17 只保证 RedisQuotaBackend 的跨副本基础设施配置在 API/worker 拆分后仍可初始化运行，不承诺 worker 已经实际调用 `quota_backend.acquire()` 做跨副本限流。

已知 gap：

- `RedisQuotaBackend` 和 Lua atomic token bucket 已存在；
- 当前 worker 消费路径尚未接入 `quota_backend.acquire()`；
- collection/provider 等更细维度也不在 task #17 范围内。

因此部署验收中若发现 worker 并发超过预期，应记录为 task #24 backlog 风险，不作为 task #17 hard-cut blocker。task #24 负责「图谱/索引 worker quota 接入 + 多维度扩展（collection_id/provider）跨副本限流生效」。

## 2. Helm 改造要求

### 2.1 新增 `indexing-worker-deployment.yaml`

新增 Helm template：

```text
deploy/aperag/templates/indexing-worker-deployment.yaml
```

建议字段：

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: indexing-worker
spec:
  replicas: {{ .Values.indexingWorker.replicaCount }}
  strategy:
    type: Recreate
  template:
    metadata:
      labels:
        app.aperag.io/component: indexing-worker
    spec:
      containers:
        - name: aperag-indexing-worker
          image: "{{ .Values.image.repository }}:{{ .Values.image.tag | default .Chart.AppVersion }}"
          imagePullPolicy: {{ .Values.image.pullPolicy }}
          command:
            - /bin/sh
            - -c
            - |
              mkdir -p /data/.cache
              mkdir -p /root/.cache
              ln -sf /data/.cache/huggingface /root/.cache/
              ln -sf /data/.cache/torch /root/.cache/
              /app/scripts/entrypoint.sh python -m aperag.cli.indexing_worker
```

`indexing-worker` 使用同一后端镜像，不新增 Celery/Dramatiq/RQ 依赖。

### 2.2 API Deployment 改造

`api` deployment 保留原 API 启动命令，只修改 probe 与连接池配置。API image 仍然挂载 `/data`，因为 API 仍需接收上传文件和 enqueue parse；但 API 请求路径不得运行 indexing worker 或 backend cleanup。

`api-deployment.yaml` 中的注释要更新：不能再写 “indexing runtime spawned inside this api pod”。

### 2.3 Values 新增结构

`values.yaml` 增加：

```yaml
indexingWorker:
  enabled: true
  replicaCount: 1
  resources: {}
  dbPoolSize: "10"
  dbMaxOverflow: "10"
  livenessProbe:
    exec:
      command:
        - /bin/sh
        - -c
        - pgrep -f "aperag.cli.indexing_worker" >/dev/null
    initialDelaySeconds: 20
    periodSeconds: 10
    timeoutSeconds: 2
    failureThreshold: 6
```

`api` 增加 Helm 层字段：

```yaml
api:
  dbPoolSize: "5"
  dbMaxOverflow: "5"
```

重要：应用代码仍只认现有 `DB_POOL_SIZE` / `DB_MAX_OVERFLOW`。Helm 层字段 `api.dbPoolSize`、`api.dbMaxOverflow`、`indexingWorker.dbPoolSize`、`indexingWorker.dbMaxOverflow` 分别映射到容器内现有 env：

```yaml
- name: DB_POOL_SIZE
  value: {{ .Values.api.dbPoolSize | default .Values.api.env.DB_POOL_SIZE | quote }}
- name: DB_MAX_OVERFLOW
  value: {{ .Values.api.dbMaxOverflow | default .Values.api.env.DB_MAX_OVERFLOW | quote }}
```

worker deployment 同理映射：

```yaml
- name: DB_POOL_SIZE
  value: {{ .Values.indexingWorker.dbPoolSize | quote }}
- name: DB_MAX_OVERFLOW
  value: {{ .Values.indexingWorker.dbMaxOverflow | quote }}
```

不要新增应用层 `API_DB_POOL_SIZE` / `INDEXING_WORKER_DB_POOL_SIZE` env alias。

## 3. Probe 语义

### 3.1 API Probe

API probe 使用统一路径：

| Endpoint | 用途 | 依赖检查 |
|---|---|---|
| `/health/live` | liveness | 不查 DB/Redis/Qdrant/ObjectStore |
| `/health/ready` | readiness | 只证明 HTTP 入口可接，短超时，不查重依赖 |
| `/health/diagnostics` | 人工/发布诊断 | 可查 PG/Redis/Qdrant/ObjectStore，但必须鉴权或仅内网暴露 |
| `/health` | legacy compatibility | 指向 liveness 语义 |

API readiness 不能成为连接池放大器。不得默认在 kubelet readiness 中检查 PG/Redis/Qdrant；如果未来必须检查依赖，只能用隔离的小预算连接和严格 timeout，且不能占用主业务 pool。

### 3.2 Worker Probe

`indexing-worker` 没有对外 HTTP 流量入口，不需要 readiness 来接入 Service 流量。建议：

- liveness 使用 process-level exec probe。
- worker 健康状态通过日志、metrics、Redis queue depth、DocumentIndex status counts 和 worker restart count 观测。
- 不配置会因为 Redis/PG transient 抖动而让 kubelet 反复杀 worker 的重依赖 readiness。

## 4. 连接池预算

本轮不接 PgBouncer。先把 API/worker 连接池预算拆开并公式化，这是后续 PgBouncer 接入的前置条件。

### 4.1 预算公式

发布前必须满足：

```text
sum(replicas * (DB_POOL_SIZE + DB_MAX_OVERFLOW))
  + rollout_surge_budget
  + diagnostics_reserved
  + postgres_reserved
  < postgres_max_connections * safety_ratio
```

推荐 `safety_ratio <= 0.7`。`postgres_reserved` 给 PostgreSQL 内部连接、管理员连接、migration job 和临时诊断保留。

### 4.2 新加坡默认建议

新加坡环境已经出现过 PostgreSQL `too many clients`，默认值必须偏保守：

| Role | replicas | DB_POOL_SIZE | DB_MAX_OVERFLOW | 单 pod 理论上限 |
|---|---:|---:|---:|---:|
| API | 2 | 5 | 5 | 10 |
| indexing-worker | 1 | 10 | 10 | 20 |

示例预算：

```text
api: 2 * (5 + 5) = 20
worker: 1 * (10 + 10) = 20
surge: 1 * 10 = 10
diagnostics_reserved: 5
postgres_reserved: 10
total = 65
```

如果实际 `postgres_max_connections` 低于该预算所需值，则必须进一步降低 `replicaCount`、`dbPoolSize`、`dbMaxOverflow` 或调整 rollout strategy，不能靠 API/worker 共进程规避。

### 4.3 PgBouncer 后续选项

PgBouncer 是后续连接池基础设施方向，不阻塞 task #17。

未来接入 PgBouncer 前必须验证：

- transaction pooling 与当前 SQLAlchemy / asyncpg 使用方式兼容；
- prepared statement 行为不会被 pooling mode 破坏；
- Alembic / migration job 不经过不兼容的 transaction pooling 路径；
- 长事务、session-level 设置、LISTEN/NOTIFY 如存在必须单独评估；
- 接入后仍保留 API/worker 独立预算，不允许因为有 PgBouncer 就无限扩 worker replica。

## 5. Cleanup 迁出 API 的部署验收

API 删除路径不得直接做 backend cleanup 或 object store prefix delete。

必须迁出的调用类型：

- `cleanup_for_deleted_documents()`
- vector/fulltext/graph/summary/vision backend cleanup
- object store prefix delete，例如 `delete_objects_by_prefix()`

API 只写 durable DB intent。Worker cleanup loop 或 cleanup queue 负责执行重型清理。

验收要求：

```text
grep -R "cleanup_for_deleted_documents(" aperag/domains aperag/server aperag/views
grep -R "delete_objects_by_prefix(" aperag/domains aperag/server aperag/views
```

HTTP request path 中不得出现上述重型 cleanup 调用。Redis cleanup queue 只能作为 wake-up transport；删除 Redis 消息后，worker cleanup scan 仍必须能通过 DB intent 恢复。

## 6. 发布计划

### 6.1 合并前检查

合并 task #17 前必须完成：

1. Helm template 渲染通过。
2. `api` deployment probe 指向 `/health/live` 和 `/health/ready`。
3. `indexing-worker` deployment 存在，启动命令为 `python -m aperag.cli.indexing_worker`。
4. `api` 与 `indexing-worker` 分别映射到现有 `DB_POOL_SIZE` / `DB_MAX_OVERFLOW`。
5. 文档包含连接池预算公式和环境示例。
6. 回滚步骤包含执行面唯一性检查。

### 6.2 发布步骤

1. 发布前记录：
   - current image tag；
   - API replica count；
   - PostgreSQL `max_connections`；
   - 当前 active connections；
   - Redis queue depth；
   - `DocumentIndex` status counts。
2. 先部署新 Helm release，包含 `api` 与 `indexing-worker` 两个 deployment。
3. 确认新 API pod 不再启动 worker/reconciler/cleanup。
4. 确认 `indexing-worker` pod Ready/Running，日志出现所有 lane 启动信息。
5. 观察 10-15 分钟：
   - API p95/p99；
   - `/health/live` / `/health/ready` 成功率；
   - `/api/v2/auth/user` 成功率；
   - PG active connections；
   - Redis queue depth；
   - `DocumentIndex` status counts；
   - worker restart count。
6. 执行 graph/indexing 压力场景，确认 API 不被 worker 负载拖垮。
7. 执行 worker pod restart，确认 API 不受影响，任务可恢复。

### 6.3 发布中止条件

出现任一条件即中止发布并进入回滚：

- API `/health/live` 或 `/health/ready` 持续失败；
- API `/api/v2/auth/user` p95/p99 明显劣化且和 worker rollout 同步；
- PG active connections 接近或超过预算；
- `indexing-worker` crashloop；
- `DocumentIndex.RUNNING` 持续增长且 heartbeat 不更新；
- Redis queue depth 持续增长且 worker 无消费；
- 出现旧 API worker 与新 worker deployment 双执行。

## 7. 回滚策略

### 7.1 执行面唯一性 hard gate

禁止只把 API image 回滚到旧版本，同时保留新 `indexing-worker` deployment 运行。

原因：旧 API image 可能仍在 FastAPI lifespan 中启动 worker/reconciler/cleanup；如果新 `indexing-worker` 继续运行，会形成双执行面。

允许的回滚方式只有两种：

1. Helm release 整体回滚到上一版本，同时移除 `indexing-worker` deployment；
2. 先把 `indexing-worker` scale 到 0，再回滚 API image。

### 7.2 回滚步骤

1. 暂停流量变更，记录当前 queue depth 与 `DocumentIndex` status counts。
2. 选择整体 Helm rollback 或先 scale worker 到 0。
3. 执行回滚。
4. 验证执行面唯一：
   - 如果回到旧 API，`indexing-worker` deployment 必须不存在或 replicas=0；
   - 如果保留新架构，API 必须不启动 worker。
5. 验证 API `/health` 和 `/api/v2/auth/user`。
6. 验证 Redis queue 与 `DocumentIndex` 状态不需要人工 SQL 修复；reconciler/cleanup 应能从 DB 恢复。

## 8. 发布验收信号

| 信号 | 目标 |
|---|---|
| API `/health/live` | 稳定成功 |
| API `/health/ready` | 稳定成功，不做重依赖检查 |
| API `/api/v2/auth/user` | graph/indexing 压力下仍稳定 |
| PG active connections | 小于预算公式上限 |
| Redis queue depth | 有消费趋势，不永久增长 |
| `DocumentIndex` status counts | PENDING/RUNNING 不永久堆积 |
| Worker restart count | 不持续增长 |
| Worker 独立重启 | API 不受影响，任务可恢复 |
