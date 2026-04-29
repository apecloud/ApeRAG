# ApeRAG 异步任务系统重构 — v8.2 执行方案（5 方共识 + PR 文档版）

**起草**：@符炫炜 总架构师定稿
**日期**：2026-04-29
**版本**：v8.2 — earayu2 directive msg=074b8019「今天做完明天发布」节奏 + msg=c1c4ba2f「先补文档，协调一致后宣布开工」+ 5 方独立 evidence-based 共识 + 全部 BLOCKER 收紧

---

## Executive Summary

### 推荐：候选 B — 保留 ApeRAG 内嵌 task system + 一次性部署 hard cut

**1 句话**：保留现有 `aperag/indexing/*` 模块位置不动 + 新增薄 `aperag/cli/indexing_worker.py` 启动入口 + 拆 API/worker deployment + probe 分层 + 连接池公式预算 + 全局并发跨副本 + 删除 cleanup 路径迁出 API + 部署级 e2e 压测验收。

**同 PR 文档项**：本文件归档 spec lock，并在 [`task-system-invariants.md`](./task-system-invariants.md) 补 architecture invariant；这是文档 fold-in，不计入 8 项 runtime hard cut 主线。

**不做**：
- ❌ 不引入 Dramatiq / Celery / RQ
- ❌ 不做 `aperag/tasks/` 包名大搬迁
- ❌ 不把应用层 hardening 4 项混进 task #17 主 PR
- ❌ 不留 lifespan worker fallback 双模式

### 5 方独立 evidence-based 推荐 100% 收敛

- 架构师 v8（本文档定稿）
- @huangzhangshu SRE 部署侧（msg=b3bf4733 + msg=5be8396e + msg=03112f15）
- @ziang 代码审计 origin/main=643c178c（msg=5eedb951 + msg=7ff9efd7 + msg=ba559be9 + msg=cecb0d88）
- @huangheng CR 视角（msg=f5c4e738 + msg=5f72446f + msg=910bcca7）
- @Weston 架构评审（msg=0d401ca7 + msg=4e74f4f4 + msg=2dae8ae9 + msg=0413ed68）

5 方共识：候选 B + 4 escape hatch architecture invariant + Dramatiq/Celery/RQ 留 PoC 路径

---

## 一、上下文（详见 v7 §1，简版）

- **新加坡 503 根因**：API + worker 共 process（黄章书 msg=0fa3489a 现场诊断）
- **Wave 3 历史**：Celery hard-cut 删除部署 + worker 退化 in-process（`deploy/aperag/values.yaml` 显式标注）
- **Redis 已有**：标准依赖 + KubeBlocks 维护（earayu2 msg=bf27b395）
- **已实现 task system**：`RedisWorkQueue` (orchestrator.py 826 LOC) + `RedisQuotaBackend Lua` (quota.py 411 LOC) + `reconciler 5-stage` (reconciler.py 889 LOC) + DocumentIndex SoT + 17 单测 + 5 ship PR

---

## 二、不可变 hard gate（必须满足）

按 @黄章书 msg=b30d07ca 7 条 + 2 层 invariant：

### 2.1 黄章书 7 条现场约束
1. API 和 worker 不能共命运
2. probe 分层
3. 连接池上限按进程预算
4. 全局并发上限跨副本
5. DocumentIndex 仍是状态 SoT
6. 删除/重建旧任务防写回
7. e2e 压测验收

### 2.2 两层架构 invariant
- 第 1 层 任务执行框架：消息投递 / worker 进程 / ack & retry / crash 再投递
- 第 2 层 ApeRAG 业务状态：DocumentIndex SoT / version-token-status gate / facts↔vectors stale / serving / 用户可见 ACTIVE

**第 2 层永不依赖 framework state**。

---

## 三、代码改造（@Bryce msg=1c46bf74 + @ziang msg=32ac48e3 详细 file-by-file）

详细 file-by-file 改造见：
- 本 PR 后续代码 diff（按 §3 总结 8 项主线改动逐一 implement）
- @ziang msg=32ac48e3 thread 章节 + 7 项 CR 修正（msg=981960cd Bryce accept）
- 实施时严格按 §10 mandatory checklist + §7 状态机失败场景验收

### §3 总结 8 项 task #17 运行时主线改动（per Weston msg=d2c46eb7 BLOCKER 1 修订统一）

1. **`aperag/cli/indexing_worker.py`** (新建 ~80 LOC) — 启动入口
2. **`aperag/app.py`** lifespan 改造（-120 / +25 LOC）— 删 worker / reconciler / cleanup 启动
3. **`aperag/server/health.py`** (新建 ~60 LOC) — `/health/live` + `/health/ready` + `/health/diagnostics`（鉴权 + sync URL，per ziang msg=7ae2e308 #2 统一 path 前缀；保留旧 `/health` 指 liveness 兼容老探针）
4. **`document_service._delete_document_indexes()`** + **object store prefix delete (`delete_objects_by_prefix()`)** 也迁出 API（per huangheng msg=f97b7c5f #6 + ziang msg=1e51c082）+ `run_cleanup_loop` 补 deleted Document scan + grep CI gate（API request handler 不出现 `cleanup_for_deleted_documents` / `delete_objects_by_prefix` / `ProductionWorkerFactory` / `run_cleanup_loop` 调用）
5. **连接池预算公式化（Helm 层 only，per ziang msg=7ae2e308 #1）**：Helm values 字段叫 `api.dbPoolSize / api.dbMaxOverflow / indexingWorker.dbPoolSize / indexingWorker.dbMaxOverflow`，**分别映射到应用现有 `DB_POOL_SIZE / DB_MAX_OVERFLOW` env**，应用代码 0 改动 0 双 env alias
6. **Helm `indexing-worker-deployment.yaml`** 新建 + `api-deployment.yaml` probe 改造 + `values.yaml` 加 indexingWorker section
7. **验收测试** 5 个新 integration test + 17 老单测继承 + 黄章书 7 hard gate 部署级压测
8. **Hard cut 删除清单**（不留兼容路径，7 项）

**Spec lock document fold** 进 `docs/zh-CN/architecture/task-system-hard-cut-v8.md` + `docs/zh-CN/architecture/task-system-invariants.md` — 是 task #17 同 PR 文档归档项，不计入运行时主线（per Weston BLOCKER 1 修订）。

工作量估算：+750 / -160 LOC，~6-7 人时

---

## 四、部署改造（@huangzhangshu msg=5be8396e 完整章节）

### 4.1 目标稳态
- `api` deployment：HTTP 入口 + 轻量 enqueue，不启动 worker
- `indexing-worker` deployment：parse / vector / fulltext / graph / graph_facts / graph_vectors / summary / vision / reconciler / cleanup 全部 lane

### 4.2 Helm 文件改造
- 新增：`indexing-worker-deployment.yaml`
- 修改：`api-deployment.yaml` / `aperag-secret.yaml` / `values.yaml`
- 不恢复：celery / flower 已删 deployment

### 4.3 API deployment env
- `INDEXING_MODE=async` / `INDEXING_QUEUE_BACKEND=redis` / `INDEXING_QUOTA_BACKEND=redis`
- `DB_POOL_SIZE=<api_pool_size>` / `DB_MAX_OVERFLOW=<api_max_overflow>`

### 4.4 Worker deployment env
- 同 API + 不同 pool size / overflow
- 启动命令：`python -m aperag.cli.indexing_worker`
- 必须挂载与 API 相同 PVC / 配置 / Secret

### 4.5 Probe 语义（关键 hard gate）
- API liveness：`/health/live` 仅证明进程活，不访问 PG/Redis/Qdrant/LLM；旧 `/health` 保留为 liveness 兼容入口
- API readiness：`/health/ready` 仅证明 HTTP 入口可接，短超时，**不能成为连接池放大器**
- API diagnostics：`/health/diagnostics` 给发布脚本 / 人工，独立小预算 + 严格 timeout，不作为 kube readiness 默认探针
- Worker liveness/readiness：进程 + 事件循环活；Redis queue 可连接；不做昂贵 provider 检查

### 4.6 连接池预算公式
```
sum(replicas × (pool_size + max_overflow)) + rollout_surge_budget + reserved_connections
  < postgres_max_connections × safety_ratio
```
- safety_ratio: 0.7-0.8
- reserved_connections: 给 psql 运维 / migration 预留
- API/worker pool 分别配置；扩容互不放大

### 4.7 PgBouncer 后续选项（earayu2 msg=befa2ae5 + Weston msg=0413ed68 + huangzhangshu msg=03112f15）
- 本轮不做（按连接池公式先解决）
- 后续：API/worker 副本数继续增长 / PG 连接数成扩容瓶颈时引入
- 优先 transaction pooling + 评估 SQLAlchemy / asyncpg / prepared statement / transaction pooling 兼容性
- 接 PgBouncer 后保留 API/worker 独立预算，不允许「因有 PgBouncer 就无限加 worker replica」

---

## 五、发布计划（@huangzhangshu §5 完整）

### 5.1 合并前检查
1. CI 全绿
2. Helm template 本地渲染：API 不含 worker 启动；indexing-worker deployment 存在
3. **Grep gate**：
   - `aperag/app.py` 不再 `asyncio.create_task(run_*_worker...)`
   - API request handler 不直接调用 `cleanup_for_deleted_documents()`
   - `indexing_worker.py` 覆盖全部 lane
4. 单测覆盖 7 条 hard gate

### 5.2 新加坡发布步骤
1. 打 enterprise image
2. 部署前读取 PG `max_connections` + 当前连接数 + 保留预算
3. 生成 Helm values（API replicas / pool / worker replicas / pool / rollout surge）
4. 部署到新加坡 demo
5. API rollout → 验证公网入口
6. Worker rollout → 验证 queue / reconciler / cleanup
7. 触发小文档 reindex smoke
8. 观察 10-15 分钟（API latency / `/health/live` / PG conn count / Redis backlog / worker logs）

### 5.3 发布验收（必须全过）
- API 在 graph 压力下 `/health/live` 稳定 + `/api/v2/auth/user` 401 正常
- Worker pod 可单独重启，API 不重启不摘流
- Redis queue 丢消息后 reconciler 能从 DB 补漏
- 旧 parse_version/token 不能写坏新 serving
- 删除文档 API 快速返回，重型 cleanup 由 worker 完成
- PG 连接数满足公式预算，rollout 不爆
- graph-heavy collection 重建时 API readiness/liveness 不超时

---

## 六、回滚策略（@huangzhangshu §6 + @ziang msg=3a0713d3 + msg=e015912b 双执行面 hard gate）

### 6.1 回滚 binary（不允许中间态）

**禁止单回滚 API image 同时保留新 `indexing-worker` deployment**（防双执行面）：旧 API image 仍会在 lifespan 启动 worker/reconciler/cleanup，跟新 worker deployment 同时跑会出现同 `DocumentIndex` 被双消费。

回滚二选一：
1. **Helm release 整体回滚**：API + worker deployment + values + probe + env 一起回到上一版
2. **手动应急回滚**：先 `kubectl scale deployment/indexing-worker --replicas=0` 确认无 worker → 再回滚 API image

### 6.2 数据状态回滚原则
- DocumentIndex 是真源不以 Redis queue 为准
- 旧任务到达由 parse_version/token/status gate 拦截
- 如新 worker 写入异常，先停 worker 再判断恢复路径

### 6.3 发布中止条件（任一即中止）
- API liveness/readiness 持续失败
- PG 连接数接近安全水位且持续上涨
- Worker 重启循环 + 队列 backlog 快速增长
- DocumentIndex 大量 RUNNING 卡死且 heartbeat 不更新
- 删除/重建路径仍在 API 请求内执行重 cleanup

### 6.4 发布 checklist 加入「执行面唯一性确认」
回滚前 `kubectl get deploy,pod` confirm 不存在「旧 API lifespan worker + 新 indexing-worker deployment」共存状态。

---

## 七、状态机 / 失败场景验收（@ziang msg=ba559be9 完整）

### 7.1 6 不变量
1. DocumentIndex 是唯一业务状态真源
2. 队列丢失必须可恢复
3. API 不拥有执行面
4. Worker 拥有重型执行面
5. 旧任务不能写回新状态
6. **cleanup intent 真源在 DB**：`Document.status=DELETED/gmt_deleted + DocumentIndex rows`，Redis cleanup queue 是可丢 transport（ziang msg=cecb0d88 钉死）

### 7.2 5 失败场景闭环

| 场景 | 检测真源 | 恢复路径 |
|---|---|---|
| Worker crash | DocumentIndex.status + last_heartbeat | reconciler running_reclaim 60s + pending_dispatch 30s |
| Ack 后 DB 写失败 | DB RUNNING / Redis 已无消息 | heartbeat 过期 reclaim |
| DB 已写 PENDING / 消息丢失 | DocumentIndex.status=PENDING | reconcile_pending_dispatch 入队 |
| Redis 重启 / 队列全丢 | DB 全量状态 | reconciler 重建 queue |
| 删除/重建旧任务到达 | DocumentIndex 当前行 + parse_version/serving | 旧行 claim 拒绝 / no-op |

### 7.3 cleanup 迁出 API 验收
1. API 删除文档立即返回，只持久化 `Document.status=DELETED`
2. Worker cleanup loop 扫到 deleted document → 执行 backend cleanup
3. Backend cleanup transient failure → 不删 DocumentIndex row → 下轮重试
4. Redis cleanup queue 仅 wake-up
5. **Grep hard gate**：API request path 不出现 `cleanup_for_deleted_documents(` / `ProductionWorkerFactory(` / `run_cleanup_loop(`

### 7.4 必补测试 7 项（详见 ziang §7.4）

### 7.5 发布观测 6 信号
API p95/p99 / `/health/live` `/health/ready` 成功率 / PG active connections / Redis queue depth / DocumentIndex status counts / worker restart count

---

## 八、架构评审 verify（@Weston msg=0212ae33 8 块 review checklist）

每块缺一不可：
1. 目标稳态（API / worker / Redis / PG / Qdrant 拓扑 + 旧路径删除）
2. 代码改造清单（具体文件 + LOC change）
3. 不做清单（hardening / 框架替换 / module 搬迁不进主线）
4. 部署改造清单（Helm + values + probe + rollout）
5. 连接池预算公式
6. 健康检查语义
7. 测试和压测
8. 发布和回滚计划

---

## 九、Spec lock fold 进 architecture 文档（document only，不动代码）

### 9.1 6 条 YAGNI 边界（永远不做）
1. 任务优先级（FIFO 够）
2. 高吞吐 broker（10k+ msg/s — ApeRAG 文档级几个/秒）
3. 复杂队列路由 / 多 routing key
4. 死信交换机（DLX）
5. 任务链 / chord / group
6. cron 调度（reconciler 30s poll 已 cover）

### 9.2 4 条 escape hatch 触发条件（满足任一才升级）
1. 任务吞吐 ≥ 100/s 持续 5 分钟
2. 跨租户公平调度成正式产品需求
3. 任务优先级 / 延迟队列 / 复杂取消 / chain·chord·group 复杂工作流成正式产品需求
4. 自研代码维护成本 ≥ 同期 framework 升级总成本（每 quarter own-up ≥ 5 + DB 瓶颈 p99 latency ≥ 1s 持续 5 分钟，且已尝试索引优化 / 批量扫描 / poll 分片 / 连接池预算后仍无法解决）

未达任一 → 永远不引入 framework 替换。

### 9.3 后续独立切片候选（不在 v8 ratify 默认项，需单独批准启动，per Weston msg=d2c46eb7 BLOCKER 2 修订）

以下是已识别的有价值 follow-up 切片，**不属于 task #17 主 PR 必做**，**不默认进同 batch ship**：

- **候选 task #18**: collection 配置校验（防 task #13 类 ops issue）
- **候选 task #19**: graph_extractor.py:184 fail-loud（own-up #4 follow-up）
- **候选 task #20**: 图谱向量点 GC sweep（task #11 落地）
- **候选 task #21**: 图谱 store bulk upsert API（task #6 spec amend）

每条都是独立完整切片，需 earayu2 单独批准启动。**不混进 task #17 发布风险**。如 earayu2 同意启动，每条作独立 task + 独立 PR + 独立 review + 独立 ship 节奏。

### 9.4 PgBouncer 后续选项（不在 task #17）

按 §4.7 评估 + 兼容性测试，作为后续基础设施切片。

---

## 十、CR mandatory checklist（@huangheng msg=910bcca7 + msg=5f72446f）

task #17 主 PR + 子 PR 来时按以下检查走 CR：

### 10.1 5 个 cross-check
- (a) 4 候选粒度等量
- (b) 节间一致
- (c) 数字合理
- (d) 框架声称分级
- (e) 推荐 evidence-based

### 10.2 mandatory checklist
- Lesson #11（5-step runtime wire-in）
- Lesson #12（grep-all-callers）
- Lesson #12 extension v3（a-e 5 条）
- Mini-pattern 17（跨真源状态漂移检测 — cleanup 路径同款 SoT 原则）
- one-shot-no-phased（接受 hard cut + schema break）

### 10.3 5 hard gate（CR 必卡）
1. API 不启动任何重型执行面（grep `aperag/app.py` lifespan）
2. cleanup 真源必须 DB（grep cleanup loop 主路径）
3. cleanup 不在 API 请求路径（grep API request handler）
4. Readiness 不成连接池放大器
5. 连接池公式化（不 hard-code）

---

## 十一、节奏 + 团队分工

按 earayu2 directive「今天做完明天发布」：

| Milestone | 内容 | 责任 | ETA |
|---|---|---|---|
| M1 | v8.2 文档补齐进 PR #1884 | 全员按文件边界补文档 | **现在** |
| M2 | 文档协调一致 + earayu2 宣布开工 | 全员给 ready / blocker | 文档 ready 后 |
| M3 | 代码实施 | @Bryce 主线 + @明书 / @cuiwenbo 协助 | 开工后 |
| M4 | 部署改造（Helm） | @huangzhangshu | 开工后 |
| ~~M5~~ | ~~task #18-21~~ | （后续独立切片候选，不在 v8 ratify 默认项，需 earayu2 单独批准启动 — per Weston msg=d2c46eb7 BLOCKER 2 + ziang msg=cd4761aa） | 单独决策 |
| M6 | CR | @huangheng 5 cross-check + @ziang 状态机 + @Weston 架构 | 实施后 |
| M7 | 验收压测 + spec lock fold | @huangzhangshu + 架构师 | 实施后 |
| M8 | Ship | 全员 | 明天 |

---

## 十二、文档 ready 后待 earayu2 宣布开工

按 earayu2 directive 团队 OWN 推荐 — 技术方案由团队收敛；本 PR 当前先补齐文档并协调一致：

1. 候选 B + 8 项 task #17 主线已作为团队推荐进入 PR 文档（薄 CLI + lifespan 删 + Helm + probe + 连接池公式 + 全局并发 + cleanup 迁出 + e2e 压测）
2. §9.3 后续独立切片候选（task #18-21）只保留为后续路径；**是否启动**需单独 product 决策
3. §9.1 YAGNI 6 条 + §9.2 escape hatch 4 条作为 architecture invariant
4. §6.1 回滚 binary（不允许中间态）+ §6.4 「执行面唯一性确认」作为发布硬门槛
5. §4.7 PgBouncer 是后续基础设施选项，不在 task #17 主线

task #17 已创建并由 @Bryce claim。PR #1884 先补齐本文档、部署 runbook、状态机验收文档并协调一致；待 @earayu2 宣布开工后，再按 task #17 主线实现。task #18-21 仅作为后续候选，需单独批准。

---

**附录**：
- v1-v7 报告均 obsolete，本 v8.2 是 PR 文档版定稿
- 5 方独立 evidence-based 推荐 100% 收敛
- 全部 BLOCKER + cross-check + Lesson refinement 收紧
- 详细 file-by-file 改造见本 PR 内 `docs/zh-CN/architecture/task-17-code-changes.md` 及后续代码 diff
- 详细部署 / 发布 / 回滚见 huangzhangshu §4-§6（msg=5be8396e）
- 详细状态机 / 失败场景见 ziang §7（msg=ba559be9）
- 详细架构评审见 Weston §8（msg=0212ae33）
