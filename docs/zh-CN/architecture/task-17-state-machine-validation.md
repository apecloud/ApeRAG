# task #17 状态机与失败场景验收

本文档补充 task #17 API/Worker hard cut 的状态机、cleanup SoT、失败恢复和 grep gate 验收要求。它只约束本轮运行时隔离主线，不引入 Dramatiq / Celery / RQ，不改变 DocumentIndex 业务状态模型。

## 1. 不变量

1. **DocumentIndex 是唯一业务状态真源**：用户可见索引状态由 `DocumentIndex.status`、`parse_version`、`is_serving`、`updated_at`、`last_heartbeat`、`retry_after` 等 DB 字段决定。Redis queue 只是可丢 transport。
2. **队列丢失必须可恢复**：`q:parse`、`q:indexing:{modality}`、未来可能出现的 cleanup queue 都不能作为唯一真源。Redis 消息丢失后，必须由 DB scan / reconciler / cleanup loop 补回。
3. **API 不拥有重型执行面**：API 只能处理 HTTP、鉴权、轻量 enqueue 和 durable intent 写入。API 请求路径不得直接构建 `ProductionWorkerFactory`，不得启动 modality worker / reconciler / cleanup loop，不得调用 backend cleanup。
4. **indexing-worker 拥有重型执行面**：独立 worker 进程负责 parse、vector、fulltext、graph、graph_facts、graph_vectors、summary、vision、reconciler、cleanup。
5. **旧任务不能写回新状态**：所有 worker 成功/失败写回必须受当前 DB 行、status、parse_version、serving 语义保护。旧 parse_version、已删除 document、已 supersede row 到达时只能 no-op 或走可恢复失败路径。
6. **cleanup intent 真源在 DB**：document delete 后的 cleanup intent 由 `Document.status=DELETED` 或 `Document.gmt_deleted IS NOT NULL` 加 remaining `DocumentIndex` rows 表示。Redis cleanup queue 如果未来新增，只能作为 wake-up transport。
7. **object store cleanup 也不在 API 请求路径**：原 API 内的 `delete_objects_by_prefix()` 属于重 IO cleanup，必须随 backend cleanup 迁到 worker cleanup loop 或 durable cleanup worker。

## 2. 五类失败场景闭环

| 场景 | 检测真源 | 恢复路径 | 验收目标 |
|---|---|---|---|
| worker crash，Redis 消息已 BLPOP | `DocumentIndex.status=PENDING/RUNNING` + `last_heartbeat` | 未 claim 的 PENDING 由 `reconcile_pending_dispatch` 重入队；已 claim 的 RUNNING 由 `reconcile_running_reclaim` 超时转回 PENDING，再重入队 | worker pod kill 后任务最终 ACTIVE/FAILED，不永久 RUNNING；API 不受影响 |
| worker pop 后 DB 写失败 | DB 仍是 RUNNING/PENDING，Redis 已无原消息 | heartbeat 过期或 pending scan 重新入队 | 模拟 finalize 抛错后，下一轮 reconciler 能恢复，不依赖 Redis 原消息 |
| DB 已写 PENDING 但消息丢失 | `DocumentIndex.status=PENDING` | `reconcile_pending_dispatch` 扫 DB 入 `q:indexing:{modality}`；parse 丢失由 stuck-parse scan 重推 `q:parse` | 清空 Redis queue 后，reconciler 能把 PENDING 补回队列 |
| Redis 重启 / 队列全丢 | DB 全量状态 | reconciler 从 PENDING / RUNNING / stale graph_vectors / failed retry 状态重建 queue | Redis flush 后不需要人工 SQL 修复，任务继续推进 |
| 删除/重建旧任务到达 | `DocumentIndex` 当前行 + parse_version / serving 状态 + Document deleted 状态 | 旧行 claim/finalize 被拒绝或 no-op；新版本 row 不被旧任务覆盖 | 同一 document rebuild/delete 并发时，旧任务不能把新 parse_version 写成 ACTIVE |

时间目标按现有默认配置写成验收上限：pending dispatch 约 30s tick；RUNNING stale reclaim 约 60s heartbeat；cleanup scan 以配置周期为准。报告和 PR 不应把这些时间写成 framework 承诺。

## 3. cleanup 迁出 API 的专门验收

1. API 删除 document 后快速返回，只在 DB 中持久化 `Document.status=DELETED`、`gmt_deleted` 和必要 durable intent。
2. API 请求路径不直接调用 `cleanup_for_deleted_documents()`，不执行 `delete_objects_by_prefix()`，不构建 graph/vector/fulltext backend cleanup worker。
3. worker cleanup loop 扫描 deleted document + remaining `DocumentIndex` rows 后执行 backend cleanup 和 object store cleanup。
4. backend cleanup 或 object store cleanup transient failure 时，不删除 `DocumentIndex` row；下一轮继续重试。
5. Redis cleanup queue 如果存在，只是 wake-up；删除 queue message 后，DB scan 仍能补漏。
6. collection delete 相关 cleanup 也必须符合同一原则：API 只写 durable intent，重型级联 cleanup 由 worker 执行或由可恢复 DB scan 兜底。

## 4. 必补测试

1. **API boundary test**：monkeypatch `cleanup_for_deleted_documents` 和 object store `delete_objects_by_prefix` 为 fail-fast，调用 document delete，确认 API 路径不触发 backend/object-store cleanup。
2. **worker cleanup recovery test**：构造 `Document.status=DELETED` + `DocumentIndex` rows，跑 cleanup loop 单轮，确认 backend cleanup 被 worker 执行且 rows 被清理；transient failure 时 rows 保留。
3. **queue-loss test**：创建 PENDING row 后清空 Redis queue，跑 reconciler，确认队列被重建。
4. **worker-crash test**：构造 RUNNING + stale heartbeat，跑 `reconcile_running_reclaim` + `reconcile_pending_dispatch`，确认恢复为可消费任务。
5. **stale-write test**：旧 parse_version / 旧 index row 到达时，不能覆盖新 serving row。
6. **API isolation e2e**：worker 压 graph/LLM-heavy 任务时，API `/health/live`、`/health/ready`、`/api/v2/auth/user` 响应稳定，PG 连接数不超过预算公式。
7. **startup grep test**：`aperag/app.py` 不再 import/启动任何 `run_*_worker`、`run_reconcile_loop`、`run_cleanup_loop`；`aperag/cli/indexing_worker.py` 覆盖全部 lane。

## 5. grep gate

PR 合并前必须通过下面的人工或 CI 检查。允许命中测试、文档和 worker 侧实现；不允许命中 API HTTP request handler 的重型执行路径。

```bash
# API lifespan 不得启动 worker/reconciler/cleanup
rg "asyncio\\.create_task\\(run_.*worker|run_reconcile_loop|run_cleanup_loop|ProductionWorkerFactory" aperag/app.py

# API domains 不得直接执行 backend/object-store cleanup
rg "cleanup_for_deleted_documents|cleanup_orphan_parse_versions|delete_objects_by_prefix|delete_by_filter|delete_by_query" aperag/domains/

# worker CLI 必须覆盖所有当前 lane
rg "run_(parse|vector|fulltext|graph|graph_facts|graph_vectors|summary|vision)_worker|run_reconcile_loop|run_cleanup_loop" aperag/cli/indexing_worker.py
```

## 6. 发布观测指标

发布时至少观察 6 个信号：

- API p95 / p99 latency
- `/health/live` 和 `/health/ready` 成功率
- PostgreSQL active connections
- Redis queue depth：`q:parse` + 每个 `q:indexing:{modality}`
- `DocumentIndex` status counts：PENDING / RUNNING / FAILED / ACTIVE
- worker restart count

任一信号异常，优先 scale worker、调低 worker pool 或暂停 worker。不得把 worker 重新塞回 API lifespan 作为回退方式。

