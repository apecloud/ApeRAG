# ApeRAG 异步任务系统 — Architecture Invariants

**作用**：本文档锁定 ApeRAG 异步任务系统的不可变 architecture invariants，未来 PR review 必引用，防止 incremental drift。

**起源**：task #17 v8 spec lock fold（详见 `task-system-hard-cut-v8.md`）。

---

## 1. 部署架构 invariant

### 1.1 API / worker 进程隔离
- `api` deployment 只跑 FastAPI HTTP 入口 + 轻量 enqueue，**永远不启动重型 indexing worker / reconciler / cleanup loop**
- `indexing-worker` deployment 跑全部 indexing 执行面（parse / vector / fulltext / graph / graph_facts / graph_vectors / summary / vision / reconciler / cleanup）
- 任何让 API 进程启动重型后台任务的代码 = BLOCKER

### 1.2 健康检查不能成为连接池放大器
- `/health/live`：仅证明进程活，不访问 PG / Redis / Qdrant / LLM provider
- `/health/ready`：仅证明 HTTP 入口可接 + 短超时，**不访问深度依赖**
- `/health/diagnostics`：人工/发布脚本用，独立小预算 + 严格 timeout，**不能作为 kubelet probe 默认路径**
- 任何让 readiness 默认检查 PG / Redis / Qdrant 的代码 = BLOCKER

### 1.3 连接池预算公式（不允许 hard-code）
```
sum(replicas × (pool_size + max_overflow)) + rollout_surge_budget + reserved_connections
  < postgres_max_connections × safety_ratio
```
- safety_ratio: 0.7-0.8
- API/worker pool 分别配置；扩容 API 不应同步放大 worker 连接，反之亦然

### 1.4 回滚执行面唯一性（双执行面 hard gate）
- **禁止**单回滚 API image 同时保留新 `indexing-worker` deployment
- 回滚 binary：(1) Helm release 整体回滚 OR (2) 先 `kubectl scale deployment/indexing-worker --replicas=0` → 再回滚 API image
- 发布 checklist 必须含「执行面唯一性确认」: `kubectl get deploy,pod` confirm 不存在双执行面共存

---

## 2. 业务状态 invariant

### 2.1 DocumentIndex 是唯一业务状态真源
- `DocumentIndex.status / parse_version / is_serving / updated_at / last_heartbeat / retry_after` 决定用户可见索引状态
- Redis 队列只做可丢 transport
- 任何让队列/broker state 跟 DB state 双真源的设计 = BLOCKER

### 2.2 cleanup intent 真源在 DB
- `Document.status=DELETED / gmt_deleted + DocumentIndex rows` 是 cleanup intent SoT
- Redis cleanup queue（如有）只能是可丢 transport，丢消息时 worker cleanup scan 必须能从 DB 补回

### 2.3 API 不拥有重型执行面
- API request handler 不得直接调用 `cleanup_for_deleted_documents` / `delete_objects_by_prefix` / `ProductionWorkerFactory(...)` / `run_cleanup_loop`
- API 删除文档：只标记 `Document.status=DELETED + gmt_deleted` 进 DB；重型 cleanup（向量库 / 对象存储 prefix delete）由 worker cleanup loop 异步执行
- grep CI gate：`grep -rn 'cleanup_for_deleted_documents\|delete_objects_by_prefix\|ProductionWorkerFactory\|run_cleanup_loop' aperag/api/ aperag/views/` 必须为空

### 2.4 旧任务防写回（version/token gate）
- worker 成功/失败写回必须受 DocumentIndex 当前行 + status + parse_version + is_serving 语义保护
- 旧 parse_version / 已删除 document / 已 supersede 行到达只能 no-op 或失败进入可恢复路径

---

## 3. 任务系统选型 invariant

### 3.1 不引入新的 task queue framework
- 不引入 Celery / RQ / Dramatiq / 等成熟 framework
- 保留现有 ApeRAG 内嵌 RedisWorkQueue + RedisQuotaBackend Lua + reconciler 5-stage + DocumentIndex SoT

### 3.2 6 条 YAGNI 边界（永远不做）
1. 任务优先级（FIFO 够用）
2. 高吞吐 broker（10k+ msg/s — ApeRAG 文档级几个/秒）
3. 复杂队列路由 / 多 routing key
4. 死信交换机（DLX）
5. 任务链 / chord / group
6. cron 调度（reconciler 30s poll 已 cover）

### 3.3 4 条 escape hatch 触发条件（满足任一才允许重评 framework）
1. 任务吞吐 ≥ 100/s 持续 5 分钟（reconciler poll 跟不上）
2. 跨租户公平调度成正式产品需求（多 tenant 抢同一 LLM provider quota）
3. 任务优先级 / 延迟队列 / 复杂取消 / chain·chord·group 复杂工作流成正式产品需求
4. 自研代码维护成本 ≥ 同期 framework 升级总成本（每 quarter own-up ≥ 5 + DB 瓶颈 p99 latency ≥ 1s 持续 5 分钟，且已尝试索引优化 / 批量扫描 / poll 分片 / 连接池预算后仍无法解决）

未达任一 → 永远不引入 framework 替换。

### 3.4 PgBouncer 后续选项
- 本轮 task #17 不做（按连接池公式先解决）
- 后续：API/worker 副本数继续增长 / PG 连接数成扩容瓶颈时引入
- 优先 transaction pooling + 评估 SQLAlchemy / asyncpg / prepared statement / Alembic / 长事务 / session-level settings 兼容性
- 接入后保留 API/worker 独立预算，**不允许「因有 PgBouncer 就无限加 worker replica」**

---

## 4. CR mandatory checklist

任意修改 task system 相关代码 / 部署 / 文档的 PR 必经过：

### 4.1 5 cross-check
1. 候选/方案粒度等量（不允许「类似 X」隐性引用）
2. 同文档不同 section 一致性（fact label cross-check）
3. fact 数字合理性挑战（不只 label spectrum 还有数字本身）
4. framework claim 分级标注 enforcement（已证实 / 待验证 / 需 PoC，到正文级别）
5. 推荐 evidence-grounded（具体代码 file path / line / git blame / framework 文档 URL / 失败场景闭环时间）

### 4.2 mandatory pattern checklist
- Lesson #11（5-step runtime wire-in）
- Lesson #12（grep-all-callers）+ extension v3（a-e 5 条）
- Mini-pattern 17（跨真源状态漂移检测）
- one-shot-no-phased（接受 hard cut + schema break）

### 4.3 6 hard gate（每 PR 必卡）
1. API 不启动任何重型执行面（grep `aperag/app.py` lifespan）
2. cleanup 真源必须 DB（grep cleanup loop 主路径）
3. cleanup 不在 API 请求路径（grep API request handler）
4. Readiness 不成连接池放大器
5. 连接池公式化（不 hard-code）
6. 回滚执行面唯一性

---

## 5. 关联文档

- task #17 spec: `task-system-hard-cut-v8.md`
- task #17 代码改造: `task-17-code-changes.md`
- 状态机/失败场景验收: `task-17-state-machine-validation.md`（待 ziang 补）
- 部署/发布/回滚 runbook: 待 huangzhangshu 补

---

**起草**：@符炫炜 总架构师
**日期**：2026-04-29
**版本**：v1（task #17 PR #1884 同 commit fold-in）
