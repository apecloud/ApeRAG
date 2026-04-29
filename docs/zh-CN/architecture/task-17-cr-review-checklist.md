# task #17 CR review checklist（huangheng CR 角色）

> 本文档定义 task #17 PR 合并前 CR 必须走完的全套检查清单。
> 适用于 `bryce/task-17-deployment-hard-cut` 主 PR + 未来同模式架构 hard cut PR。
> 对应方案文档：`docs/zh-CN/architecture/task-system-hard-cut-v8.md`（架构师 v8.2 final）。

---

## 一、5 条交叉核对（cross-check）

### (a) 4 候选粒度等量

候选评估文档每个候选必须填同口径表格（4 列：框架提供 / ApeRAG 必须保留 / 衔接风险 / 最终稳态产物），每列内容粒度等量。**不允许「类似候选 X，区别如下」类隐性引用**。

### (b) 节间一致

同一份文档内 `§x` 与 `§y` 的事实标注、数字、scope 边界必须互相一致。最常见漂移点：

- Executive Summary 主线项数 vs §3 主线项数 vs §12 待 ratify 主线项数
- §4 框架对比表的 deployment / 依赖 row vs §3 各候选 deployment 描述
- §6 回滚策略 vs §5 发布步骤的回滚约定

### (c) 数字合理

代码量、测试数量、工作量人天、PG `max_connections` 等数字必须可验证（grep / wc / git blame / 现场实测），不能写「大约几千行」类模糊描述。

### (d) framework claim 分级到正文

任何「framework 提供 / 接管 / 天然支持」类技术声称必须分三级：

- **已证实**：framework 官方文档 / source code / 本仓库 PoC 可直接 cite
- **待验证**：仅根据 framework high-level 描述，需进一步 PoC / 集成测试
- **需 PoC**：ApeRAG 集成层未实测，必须独立 PoC 验证

不允许「framework X 可以 Y」的 unverified high-level claim 直接进正文。

### (e) 推荐 evidence-grounded

任何架构推荐 / 选型决策必须 ground 在具体 evidence：

- 具体代码 file path / line / git blame（grep 实证）
- 具体框架文档 URL（不只 reference 名）
- 具体失败场景闭环时间数据（不只「快/慢」label）
- 具体「选错会怎样」+「未来什么证据触发重评」的可量化条件
- 不允许「我倾向 X」「X 看起来更好」类 high-level 论证

---

## 二、6 条架构 hard gate（CR 必卡，ziang msg=4ea65100 + msg=ad6a610d 收敛版）

### gate 1：API 不启动任何重型执行面

- `aperag/app.py` lifespan 不允许出现 `asyncio.create_task(run_*_worker(...))` 调用任意 modality worker / reconciler / cleanup loop
- `grep` CI 自动化检查 `app.py` 内 `run_vector_worker` / `run_fulltext_worker` / `run_graph_worker` / `run_graph_facts_worker` / `run_graph_vectors_worker` / `run_summary_worker` / `run_vision_worker` / `run_parse_worker` / `run_reconcile_loop` / `run_cleanup_loop` 全部消失

### gate 2：cleanup intent 真源是 DB

- `Document.status=DELETED` / `gmt_deleted` + remaining `DocumentIndex` rows 是 cleanup intent 唯一真源
- Redis cleanup queue 仅作 wake-up transport，可丢消息
- `worker cleanup loop` 必须能从 DB scan 补回任何 Redis 丢失的 cleanup intent
- `grep` CI 自动化检查 worker cleanup loop 含 DB scan 路径（`Document.status == DELETED` 或 `gmt_deleted IS NOT NULL`）

### gate 3：object store cleanup 也迁出 API 请求路径

- `delete_objects_by_prefix()` 重 IO 调用必须从 API HTTP request handler 移到 worker cleanup loop
- `cleanup_for_deleted_documents()` backend cleanup 同样移出
- `grep` CI 自动化检查 API request handler 不出现以上两类调用

### gate 4：API readiness 必须轻量

- API liveness：仅证明进程活，不访问 PG / Redis / Qdrant / LLM provider
- API readiness：仅证明 HTTP 入口可接 + 短超时（建议 ≤500ms）
- 深度依赖检查必须放 `/health/diagnostics` 独立 endpoint，使用隔离小预算连接池
- **不允许 readiness 成为 PG / Redis / Qdrant 连接池放大器**
- 建议：readiness 不读 PG，必要时读 PG 用独立 ≤2 连接池 + ≤200ms 超时

### gate 5：连接池 Helm 层映射现有 env

- 应用代码不引入 `API_DB_POOL_SIZE` / `INDEXING_WORKER_DB_POOL_SIZE` 等新 env alias
- Helm values 字段 `api.dbPoolSize` / `indexingWorker.dbPoolSize` 直接映射应用现有 `DB_POOL_SIZE` / `DB_MAX_OVERFLOW`
- 部署文档必须含连接池预算公式：

  ```text
  sum(replicas × (pool_size + max_overflow)) + rollout_surge_budget + reserved_connections
    < postgres_max_connections × safety_ratio
  ```

  `safety_ratio` 建议 0.7-0.8。

### gate 6：回滚执行面唯一性

- 禁止单回滚 API image + 保留新 `indexing-worker` deployment（旧 API lifespan worker + 新 worker deployment 双跑）
- 回滚二选一：(a) Helm release 整体回滚 / (b) 先 `kubectl scale indexing-worker --replicas=0` 确认无 worker 后再回滚 API
- 发布 checklist 加入「执行面唯一性确认」：`kubectl get deploy,pod` 验证

---

## 三、7 条实现修正（ziang msg=4ea65100 + msg=76f6f465 + Bryce msg=981960cd accept 版）

### 修正 1：使用现有 `settings` module 实例

- `aperag.config.get_settings()` helper 不存在
- 必须用现有 module 级 `settings` 实例
- `ProductionWorkerFactory` 从 `aperag.indexing.worker_factory` import（不是 `aperag.indexing` `__init__`）

### 修正 2：`QuotaPolicyRegistry` 直接创建

- `settings.indexing_quota_registry` 字段不存在
- 按 `app.py` 现有写法：`RedisQuotaBackend(quota_redis, registry=QuotaPolicyRegistry())`
- `InMemoryQuotaBackend(QuotaPolicyRegistry())` 同理

### 修正 3：连接池仅 Helm 层映射

跟 gate 5 重叠 — 应用代码不引双 env alias，仅 Helm 层 values 字段映射应用现有 env。

### 修正 4：`_delete_document_indexes` 不嵌套 transaction

- 外层 `_delete_document` 已在 transaction 里标记 `Document.status=DELETED` / `gmt_deleted`
- 不允许 helper 内再开 `execute_with_transaction`
- 删除 helper 对 `cleanup_for_deleted_documents()` 的调用，改成只写 DB intent

### 修正 5：object store delete 也迁出 API

跟 gate 3 重叠。

### 修正 6：`run_cleanup_loop` 补 deleted Document scan

- 现 `run_cleanup_loop` path A 仅做 orphan parse_version GC
- task #17 必须 verify path B/C 是否完整覆盖 `Document.status=DELETED + remaining DocumentIndex rows` scan
- 缺则同 PR 内补，不得留作后续 task

### 修正 7：`/health/diagnostics` 鉴权 + sync URL

- diagnostics endpoint 必须真鉴权或仅内网暴露（`include_in_schema=False` 或 admin token）
- PG diagnostics 的 sync engine 必须使用 `get_sync_database_url(settings.database_url)` 转换 async URL（postgresql+asyncpg → postgresql+psycopg2），不能直接传 async URL 给 `create_engine`

---

## 四、CR 必应用的现有 lessons

### Lesson #11：runtime wire-in 5-step checklist

新增 modality / worker / async consumer 必走 5 步：

1. **factory 注册**：worker_factory.py 等 builder 注册
2. **dispatcher 路由**：dispatcher.py 等 enum → queue 路由
3. **reconciler enqueue**：状态转换什么时候补漏
4. **lifespan startup launch**：进程启动入口（API / worker deployment lifecycle）
5. **e2e narrative 测试**：PENDING → ACTIVE 端到端真链路 verify

### Lesson #12：grep-all-callers checklist

shared utility / 默认行为 / 函数签名改动必须 grep 全 caller，不允许信 PR description / function 名 framing。

### Lesson #12 extension v3：架构候选评估文档 4 cross-check

(a) 候选粒度等量 / (b) §x vs §y 一致 / (c) 数字合理 / (d) framework claim 分级到正文。本文档 §一 已展开。

### Mini-pattern 17：跨真源状态漂移检测

跨 truth source（DB / 文件 / cache / queue / 外部服务）状态依赖必须 enumerate 自动 detection 机制（cache key 含上游 version / 周期巡检 stale check / startup sanity check 三选一）。

### feedback：一次性不分阶段

主要架构改动接受 hard cut + schema change + break，不留兼容路径 fallback。CR 时不允许接受「先做 Phase 1 紧急止血 + Phase 2 hardening + Phase 3 产品化」类拆分。Wave-style multi-PR 每条是 complete slice 不算分阶段；hot-fix 临时 patch 例外。

---

## 五、CR 工作流

### 5.1 PR 上线后 CR 顺序

1. 拉 PR 分支 + 验证 CI 全绿
2. 走 §一 5 条 cross-check（粒度 / 一致 / 数字 / claim / evidence）
3. 走 §二 6 条 hard gate（API 隔离 / cleanup SoT / object store 迁出 / readiness 轻量 / 连接池 Helm / 回滚唯一）
4. 走 §三 7 条实现修正（settings / QuotaPolicyRegistry / 连接池 / 删除 helper / object store / cleanup loop / diagnostics）
5. 走 §四 lessons 全应用
6. 输出 verdict：`🟢 同意通过 / 🛑 阻塞 + 具体修法 / 🟡 部分通过 + 待修小项`

### 5.2 verdict 表述规范

不混用英文流程词。表述用中文：

- ✅ **同意通过**（同 LGTM）
- ⏳ **阻塞**（同 BLOCKER）— 必修才能合并
- 💡 **小修建议**（同 NIT）— 不阻塞合并

技术专有名词（PR / CI / Redis / Helm / probe / DocumentIndex / framework）保留英文。

### 5.3 false positive 自我修正

CR cross-check 表里任何 ✅ 标注被 surface 为 false positive 时，立即在 thread 公开 own-up 并撤回 ✅ → 改为对应等级（部分通过 / 阻塞）。**不允许 silent 修正不公开**。

### 5.4 cross-check 与团队协作

- CR 与 implementer / 架构师 / SRE / 测试专家 / 部署侧 多人协作时，按文件边界分工不重叠
- 发现跨边界问题先在 PR thread 点对方，不直接跨边界大改
- 合并 gate：5 lane（implementer + 架构 + SRE + 状态机/CR + 部署）独立给 verdict，全 ack 后 squash merge

---

## 六、CR 历史 sediment 引用

- `memory/feedback_e2e_dataflow_trace.md` Lesson #11 + Lesson #12 + extension v3
- `memory/feedback_collab_before_solo_pr.md` 同 tag/scope 多 agent 默认 co-design 单 PR
- `memory/feedback_object_store_path_drift.md` `OBJECT_STORE_LOCAL_ROOT_DIR` 漂移诊断 playbook
- `memory/feedback_one_shot_no_phased_rollout.md` 一次性不分阶段 directive
- `memory/feedback_simple_stable_deploy_and_forget.md` simple-stable + 私有化部署免维护
- `memory/feedback_cr_must_cross_reference_arch_doc.md` CR 必对照架构文档逐条 invariant 走

---

## 七、task #17 PR final review verdict（待 PR 合并前填）

| 检查项 | 状态 | 备注 |
|------|------|------|
| §一 5 cross-check 全过 | _待填_ | |
| §二 6 hard gate 全 verify | _待填_ | |
| §三 7 实现修正全 fold-in | _待填_ | |
| §四 lessons 全应用 | _待填_ | |
| 团队 5 lane LGTM ack 收齐 | _待填_ | |
| 架构师 v8.2 docs 入仓 | _待填_ | |
| 黄章书 部署/发布/回滚 docs 入仓 | _待填_ | |
| ziang 状态机/cleanup SoT 验收 docs 入仓 | _待填_ | |
| Weston scope 一致性 verify | _待填_ | |
| Bryce 代码主线 file-by-file align ziang 7 修正 | _待填_ | |

最终 verdict 由 huangheng 在 PR 合并前填，附 thread message ID 引用。
