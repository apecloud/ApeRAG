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

### 6 hard gate 与具体测试文件 mapping（冬柏 msg=d56bb0f7 补充）

verify 必须落到具体 test 文件锚点，不允许抽象「verify」描述：

| Hard gate | 验证脚本 / test 文件 | 默认 owner |
|-----------|---------------------|----------|
| #1 API 不启重型执行面 | `tests/boundaries/test_app_lifespan_no_workers.py` grep CI gate + 单测 | Bryce (#20) |
| #2 cleanup intent DB SoT | `tests/integration/test_cleanup_recovery_redis_outage.py` Redis kill + DB scan 补漏 | ziang (#19) |
| #3 object store 迁出 API | `tests/boundaries/test_api_no_objectstore_calls.py` grep gate + 行为测试 | ziang (#19) |
| #4 readiness 轻量 | `tests/load/test_health_endpoints_under_load.py` p95 < 500ms + DB pool 满时仍稳定 | Planetegg (#22) |
| #5 连接池 Helm 映射 | `tests/integration/test_helm_pool_budget.py` 验证 env 透传 + 预算公式不超 | huangzhangshu (#18) |
| #6 回滚执行面唯一 | runbook + `tests/load/test_rollback_dryrun.py`（k8s scale + 切回，验证 no double-execute）| huangzhangshu + Planetegg |

owner 可调，关键是每个 gate 有可执行 test 文件锚点。CR 时 verdict 表 §七要求填具体 test commit / 行号。

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

### Lesson #11 v5：entry-point migration sub-check（架构师 + huangheng + Weston 联名升级）

**触发条件**：进程 split / lifecycle hard cut（worker / scheduler / async consumer 从 source 进程搬到 target 进程）。

**核心 insight**：原 Lesson #11 步 4「lifespan startup launch」只 cover「launch worker」单一概念，没显式覆盖「source 进程的 init 代码必须 port 到 target」。task #17 hard cut 把 worker 从 `app.py` lifespan 搬到 `cli/indexing_worker.py` 时，10 个 cross-domain DI setter 是 init 代码不是 worker spawn，所以漏 port → e2e-http-provider 失败 + 5 PR 卡 phase 2 close。

**实施 3 步**：

1. **Step A: grep**：扫 source 进程顶层所有 `*_set_*_ops()` / `configure_*()` / `register_*()` / DI setter call site（不只 worker entrypoint）
2. **Step B: 三分类决策**（Weston msg=d05e56c0 三分类框架）：
   - **类 1 必须对称 port**：domain ops + runtime ops setter → boundary test 钉等价
   - **类 2 process-level 显式决策**：logging / observability / metrics / shutdown hook / register_custom_llm_track → PR description 必须显式写明 worker 是否需要 + 理由
   - **类 3 明确排除**：FastAPI-only / 网络绑定 / 路由 → boundary test 反向钉不漏进 worker
3. **Step C: AST boundary test**：用 import alias canonical-name + bi-directional check + allowlist 外置 frozenset 自动 detect 未来漂移（架构师 3 refinement: walk lifespan / 函数体 / 实参 detect / allowlist 强制更新）

**task #17 hot-fix 实证清单（PR #1893 commit `d4b65e27`）**：

- **类 1（10 setter）**：KB 4（`set_marketplace_ops` / `set_marketplace_collection_ops` / `set_search_pipeline_ops` / `set_quota_ops`）+ conv 1（`set_quota_ops`）+ agent runtime 1（`set_prompt_template_ops`）+ model_platform 1（`set_prompt_crud_ops`）+ identity 3（`set_bot_init_ops` / `set_chat_init_ops` / `set_quota_init_ops`）
- **类 2（3 process-level）**：`configure_logging` / `configure_process_observability` / `register_custom_llm_track`
- **类 3（2 FastAPI-only）**：`configure_fastapi` / `register_exception_handlers`（不进 worker）
- **实施模式**：`aperag/bootstrap/__init__.py` 单一 source of truth + `wire_cross_domain_di_seams()` 函数 + `tests/boundaries/test_worker_di_parity.py` AST 级 3 重防回归（call site / setter 集合 / FastAPI-only 反向）

**CR cross-check 应用**：未来 PR 如果涉及进程 split / lifecycle hard cut，CR 必走完 3 步 + 三分类决策矩阵；缺任一类的 PR 不允许 LGTM 通过。

### Lesson #12：grep-all-callers checklist

shared utility / 默认行为 / 函数签名改动必须 grep 全 caller，不允许信 PR description / function 名 framing。

### Lesson #12 extension v3：架构候选评估文档 4 cross-check

(a) 候选粒度等量 / (b) §x vs §y 一致 / (c) 数字合理 / (d) framework claim 分级到正文。本文档 §一 已展开。

### Lesson #12 extension v4：PR `lint-and-unit` CI 全绿是 mandatory ratify gate

「本地实证通过」≠「CI 全绿」。PR ratify / squash merge 前必须 verify GitHub Actions `lint-and-unit` 全绿，不能仅凭本地 pytest / 部署级压测数据放行。

来源：task #17 PR #1884 ratify 过程中 huangheng 在 verdict 表只 check 了部署级压测（Planetegg 153.66s / 40 ACTIVE）+ § 二 6 hard gate 实证文件，漏 verify 了 `lint-and-unit` CI status；CI 在 PR 合并前刚好暴露 `tests/boundaries/test_app_lifespan_launches_all_graph_indexing_worker_lanes` 与 task #17 hard cut 冲突（PR #1876 时代加的，task #17 让 lifespan 不再 launch worker，老 test 不可能再过），Weston 用 `335fe586` 删掉 obsolete test 让 CI 转绿。CR verdict 表 § 七 必须新增「PR `lint-and-unit` CI 全绿」一行 mandatory，verify 后才能 ratify。

### Lesson #12 v5：CI status 解读 trust framing 反模式（架构师升格独立条目）

**触发条件**：CI 失败 / variant matrix 部分红时，第一反应贴「flaky / matrix shape」标签前必须 grep 实证。

**核心 insight**：Lesson #12 grep-all-callers 不只是「shared utility default 改动前 grep 全 caller」，**CI status 解读阶段 trust framing 也是同模式**。task #17 PR #1893 hot-fix CI 1/3 e2e-http-provider fail 时，huangheng 第一反应贴 matrix flake / provider key 标签（msg=3fa1854c），没 grep 跨 PR 同 variant fail pattern → PM forensics 5 PR 数据 surface 真根因（pre-existing Neo4j flake，跟 hot-fix 无因果）。huangheng own-up：trust framing 反模式延伸到 CI 解读阶段。

**实施步骤**：

1. **Step A**：CI fail surface 时不直接贴 flaky / matrix flake / 环境问题标签
2. **Step B**：grep 跨 PR 数据 — `gh run list --branch ...` / `gh api repos/.../check-runs` 看 main + 同期其他 PR 同 variant 是否同失败
3. **Step C**：grep 失败 stack vs PR diff 文件路径交集，零交集 → 高置信度 pre-existing
4. **Step D**：多 reviewer 独立 forensics（不 trust 单一 framing — 自我 + 跨 reviewer cross-check）

**示例**：架构师 msg=df1ed687 + Planetegg msg=7b1ec4eb + ziang msg=96a455bc + Bryce msg=ac2b8fe4 四方独立 grep verify Neo4j variant fail，互相 cross-check 同根因，避免 trust 单一 framing — 这是 Lesson #12 v5 正确实施的 first-application demo。

### Lesson #12 v6：grep line number ≠ 执行顺序，必 walk function scope

**触发条件**：CR 判断代码 init 顺序 / 调用顺序 / 时序时。

**核心 insight**：仅靠 grep 行号不足以判断执行顺序，必须 walk function scope 看 callee 在哪个 function（`main()` outer sync vs `_amain()` async）+ function 间调用关系。task #17 PR #1893 CR 时 huangheng 把 `cli/indexing_worker.py:235-237 configure_logging` 误判为 `_amain` 内 wire 之后（NIT 2 顺序非对称），实际是 sync `main()` 在 `asyncio.run(_amain())` **之前**执行（架构师 msg=c605cc1f + Weston msg=3b02abee 纠正）。huangheng own-up：scope walk verify 缺失。

**实施步骤**：

1. **Step A**：grep 找到 call site 行号
2. **Step B**：确认 call site 在哪个 function scope（不能仅看行号大小判断顺序）
3. **Step C**：追溯 function call graph（`main()` → `asyncio.run(_amain())`）确认实际执行顺序
4. **Step D**：跨 reviewer cross-check 顺序判断（避免单 reviewer 漏看 outer function）

**对应 Lesson #12 同根**：行号 / status / 错误类型 / 函数名 这类 surface signal 都是 trust-framing 反模式入口，必须 grep + scope walk + cross-reviewer verify。

### Lesson #12 v6 sub-form：scope walk 三层覆盖（架构师 msg=9c5c32d1 升级）

`scope walk` 不只「function/method 边界」单一形态，task #36 PR #1899 fix-forward² 实战 surface 三个 sub-form，架构师 msg=9c5c32d1 同意 fold sub-form：

- **v6.1 function scope**：原 v6 — callee 在哪个 function（sync `main()` vs async `_amain()`）+ function call graph 追溯
- **v6.2 endpoint scope**：同 jsonpath 字符串在不同 endpoint 语义不同（task #36 fix-forward² L148 case：`count >= 3` 在 `/api/v2/models` endpoint 是「全部 model 计数」, 在 `/api/v2/model-uses` endpoint 是「scenario 计数」— 同 jsonpath 字符串语义完全不同。huangheng over-fix 误把 model-uses 的 L148 跟 models 的 L103 同改 → Bryce `git show 623c7c72` verify ziang commit context 挡住）
- **v6.3 data type scope**：count assertion 关联的不同 data source / list vs dict / 不同字段类型 — 跟 v6.2 同根「surface signal 字面相同但底层语义不同，必须 walk 上下文 context」

**实施步骤补充**：每条 surface signal（line number / jsonpath string / status code / 错误类型 / 函数名）必须 walk 至少一层上下文（function / endpoint / data type）才能 ground 判断。

### Lesson #12 v7：grep 必跨 caller signature → backend schema → runtime fallback 三层

**触发条件**：判断系统某 contract（默认值 / 字段是否暴露 / 行为是否启用）的实际 runtime 行为时。

**核心 insight**：grep caller chain 找到入口 default 不等于 grep 完整 — 必须跨：

1. **Caller signature 层**（MCP tool / API endpoint default 参数）
2. **Backend schema validation 层**（Pydantic Field default / serializer 默认）
3. **Runtime fallback 行为层**（when missing config 是 raise / silent skip / silent fallback）

三层任一层 default 不一致 → contract drift（用户从 sig 看到 True，runtime 实际 False，运维 + 文档 + 用户预期都错位）。这跟 Lesson #12 v6（行号 ≠ 执行顺序，必 walk function scope）同源「grep 表层 ≠ 真实运行行为」。

**实证来源**：rerank task #34 调研中（earayu2 msg=70cb0f6b），huangheng msg=e539848f 初版 grep 只覆盖 caller chain 入口（MCP tool `rerank: bool = True`）+ retrieval pipeline `_rerank` 实施层；架构师 msg=b12fec5d thoroughness=very thorough trace 补 `retrieval/schemas.py:287 SearchRequest.rerank: Optional[bool] = Field(False, ...)` schema-layer 默认覆盖（MCP tool sig True 被 backend schema False 覆盖，实际 runtime default 已经是 False）+ `pipeline.py:630-632` graceful fallback 路径。huangheng own-up msg=e5c6b105 修法建议 1（MCP True → False）方向对但理由错（不是 runtime default 故障，是 sig/schema contract drift）。

### Lesson #12 v7.1：composite key invariant（修改返回字段时必须 verify 下游 caller 所有 required parameter 能从新返回 reconstruct）

**触发条件**：新增 / 修改 endpoint 返回字段时，下游 caller signature 含 composite key（多字段 required parameter）。

**核心 insight**：单字段（chunk_id）不能取代 composite key（document_id + chunk_id）。修改返回字段时必须 grep verify 下游 caller 所有 required parameter 都能从新返回 reconstruct，否则 agent / consumer 拿到字段也无法 chained 调用。

**实证来源**：task #32 spec v1 PR #1905 § 3.1.1 早期版本写 `evidence_chunk_ids: list[str]`（只 chunk_id），但 `read_document_chunk(collection_id, document_id, chunk_id)` 是 document-scoped composite key — chunk_id **不全局唯一**，必须 composite。huangheng CR (msg=cba16b73) + 架构师初版 spec 双侧漏 caller signature verify，Weston msg=7500e57d 第三 reviewer cross-check catch BLOCKER → spec fix-forward `74d0951` 改 `evidence_refs: [{document_id, chunk_id, parse_version?}]`。huangheng + 架构师 double own-up msg=64c0c838 / msg=fbe0ee8a。

**实施步骤**：修改返回字段 PR 必走 4 步：

1. **Step A**：grep 全部 downstream caller（per Lesson #12 grep-all-callers）
2. **Step B**：列每个 caller 的 required parameter 全集
3. **Step C**：verify 新返回字段能 reconstruct 全部 required parameter（不只是「相关字段」）
4. **Step D**：缺任一 required parameter → 立刻 surface BLOCKER fix spec / impl，不允许「下个 PR 补」

### Lesson #12 v7.1 sub-form：backend 投影层 + acceptance 跨 endpoint chained chain 双层 verify（架构师 msg=f04b36a8 升级）

composite key invariant 应在「**backend 投影层**（PR #1909 commit `8d5ffa97` textbook）+ **acceptance 跨 endpoint chained chain**（PR #1912 commit `eb2a805b` textbook）双层 verify」。单层 verify 容易漏 endpoint 间一致性 drift。

**backend 层 verify 模式**（PR #1909 demo）：

- 投影层 schema 定义 composite key（`GraphEvidenceRef(document_id, chunk_id, parse_version?)`）
- service projection 函数 (`_lineage_to_evidence_refs`) 实现 composite key dedup（`seen: set[tuple[str, str, str]]`）+ 确定性排序 + per-member 迭代 + bounded limit
- unit test 钉投影层 schema 字段（`test_entity_view_exposes_bounded_evidence_refs_for_read_document_chunk`）

**acceptance 层 verify 模式**（PR #1912 demo）：

- patch-based isolation：复用 D9 base helpers（`_patch_doc_lookup` + `document_service.get_document_chunks`）
- minimal fixture 跟 backend 投影层 unit test 字段对齐（`document_id="doc1" / chunk_id="chunk-a" / parse_version="v1"`）
- 跨 endpoint chained call assertion：`entity.evidence_refs[0].document_id + chunk_id` → `read_document_chunk(collection_id, document_id, chunk_id)` → 返回 chunk content
- patch-level sanity round-trip check（防 stub 自身 silent 漂移）— huangheng msg=ad42c07b + Weston msg=a25a3820 独立 surface 同 idiom 验证 v6 cross-reviewer cross-check

**双层 verify 必要性**：

- 单 backend 层 verify：会漏「字段返回 OK 但 caller 实际无法 chained 调用」的 endpoint 间 drift
- 单 acceptance 层 verify：会漏「字段定义不一致」的 backend schema drift
- 双层联合：覆盖 schema correctness + chained chain executability，缺一不可

### Lesson #13：invariant evolution 必须双侧 rewrite obsolete regression test

invariant 演化时（旧 invariant → 新 invariant 反向），单纯加新 test 不够，必须主动 search + delete/rewrite 与新 invariant 冲突的旧 regression test，并在 PR description 显式声明这次 invariant 反转。

具体演化轨迹（task #17 真实案例）：

| 阶段 | invariant | 落地 test |
|-----|----------|---------|
| PR #1876 (task #12 时代) | `app.py` lifespan 必须 launch 三条 graph worker lane（防 silent miss） | `tests/boundaries/test_app_lifespan_launches_all_graph_indexing_worker_lanes` 正向 assert lifespan 启动 worker |
| PR #1884 (task #17 hard cut) | `app.py` lifespan 不能 launch 任何 worker（解新加坡 503 部署根因） | `tests/boundaries/test_app_lifespan_no_workers.py` 负向 + `tests/boundaries/test_cli_worker_starts_every_runtime_loop` 正向（双侧钉死新 invariant） |

PR #1884 完成时新 test 已加，但旧 PR #1876 test 漏删 → CI fail → fix-forward 删 obsolete test。Lesson：invariant 反转 PR 必须主动 grep 全 codebase 找旧 invariant test 同时删，不能等 CI 暴露后补救。

CR cross-check 应用：CR 时如果 PR 涉及核心 invariant 反转（lifespan worker / cleanup intent SoT / readiness 行为 / API 是否触 heavy 路径等），必须 grep 旧 invariant 关键词找 obsolete test 并要求作者 in-PR 删除，不允许「下个 PR 补」。

### Lesson #13 v2.1：import-level dual-side rewrite（删 source 必删对应 obsolete test 文件 / 函数）

**触发条件**：PR 删除 source 模块 / 类 / 函数。

**核心 insight**：删除 source 后 obsolete regression test 文件如果 `from <deleted module> import ...`，pytest collection ERROR → 卡死整套 unit test runner（不只是单个 test fail）。invariant 反转 PR 必须主动 grep + delete obsolete test 文件 / 函数，不能等 CI 暴露。

**first-application demo**：task #17 PR #1884 deleted `app.py lifespan worker launch`，PR #1876 时代旧 test `test_app_lifespan_launches_all_graph_indexing_worker_lanes` 漏删 → CI fail → Weston commit `335fe586` fix-forward 删除。

**second-application demo**：task #36 PR #1899 deleted `aperag/llm/rerank/rerank_service.py` (RerankService class)，但 `tests/unit_test/llm/test_rerank_service.py` 漏删 → pytest ImportError → fix-forward `fdb5f161` 整文件删 + 4 BONUS 顺手 cleanup（test_v1_ghost_guard / test_model_platform_v1_compat / test_model_runtime_resolver / hurl files）。

**实施步骤**：source 删除 PR 必走 grep + 删除 4 类 obsolete test artifact:

1. 直接 import 删除 module 的 test 文件
2. 测试已删除 endpoint / API path 的 test 函数
3. 测试已删除 schema 字段的 test 函数
4. hurl / e2e fixture 创建已删除资源（model / scenario）的 hurl block

### Lesson #13 v2.2：value-level dual-side rewrite（删 source 字段 / 数据必删对应 stale assertion / count）

**触发条件**：PR 删除 source 字段 / 模型 record / config option。

**核心 insight**：删除 source 数据后 obsolete test data assertion（count / value / count >= N）如果未同步 update，CI fail。这是 Lesson #13 v2.1 import-level 的 value-level 延伸 — 不只删 import，还要 update 依赖该数据的 test 数值。

**first-application demo**：task #36 PR #1899 fix-forward²（commit `629bb4ef`）— 删除 `dashscope_rerank` model registration in `10_provider_llm.hurl`，但 L103 + L108 `count >= 3` / `count >= 2` assertion 未同步 update。BLOCKER² catch (msg=3e4898bc) → 改 L103 `>= 2` + L108 `>= 1`（保留 L148 `>= 3` for model-uses endpoint per Lesson #12 v6.2 endpoint scope walk）。

**second-application demo**：task #47 PR #1910 删除 `ModelCapability.RERANK` enum value，同 PR `tests/unit_test/test_model_platform_v2_contract.py` 删 `default_allowed_scenarios(RERANK) == []` 断言（value-level dual-side rewrite）— 一次性 PR 内完成，不需 fix-forward²。

**实施步骤**：删除字段 / 数据 PR 必同步 update：

1. test 数据 fixture 中创建 / 引用该字段的所有 block
2. test count assertion（jsonpath count >= N，N 跟实际创建数量挂钩）
3. test value assertion（exact value match / `== []` 等死值断言）
4. allowlist / forbidden_set 配置常量（如该字段在 forbidden 列表里）

**Lesson #12 v6.2 联动**：value-level update 时必须 walk endpoint scope — 同 jsonpath count 在不同 endpoint 语义不同（PR #1899 fix-forward² L103/L108 是 models endpoint vs L148 是 model-uses endpoint）。

### Lesson #13 v3：boundary test 不重复事实保证 invariant，只覆盖可能 drift 的 contract（架构师 msg=036dd8b2 升格）

**触发条件**：删除/移除某 invariant 时，决定 boundary test 覆盖范围。

**核心 insight**：boundary test 是为「可能 drift 的 contract」设计 — 如果某 invariant 已经被「事实保证」（删除整 dir、DB migration 删 row、enum 整删等），不需要 boundary test 重复测试事实保证 invariant。

**first-application demo**：task #46 PR #1906 `tests/boundaries/test_no_rerank_in_mcp.py`：

- ✅ 覆盖 invariant 1（MCP tool signatures no rerank param）+ invariant 3（schema fields no Rerank）— 这两类是「contract 层」，可能 drift 通过未来 PR 误加回
- ❌ 不覆盖 invariant 2（`aperag/llm/rerank/` 模块 import）— 整 dir 已删除，任何 import 触发 `ModuleNotFoundError` 自动 catch → **事实保证**
- ❌ 不覆盖 invariant 4（`model_use` scenario `retrieval_rerank` string literal）— DB migration 已 DELETE + enum 覆盖 → **事实保证**

**判断准则**：如果某 invariant 的违反方式只能通过「重新创建已删除的模块 / 数据」实现（高门槛，明显工程行为），事实保证够用 — boundary test 重复覆盖增加维护成本无收益。如果违反方式是「在新 PR 加回字段 / 参数」（低门槛，容易 silent drift），boundary test 必须钉死。

**实施步骤**：设计 boundary test 范围时三步：

1. **Step A**：列出全部 invariant
2. **Step B**：每个 invariant 判断违反路径：高门槛（重创已删资源）vs 低门槛（新 PR 加字段）
3. **Step C**：高门槛 invariant 走「事实保证」路径（不写 boundary test）+ 文档显式标注为何不写 + 跨 PR 引用为何 missing test 是 by-design / 低门槛 invariant 必走 boundary test 钉

### Migration chain 时序 invariant：enum hard-cut PR 必先 chain DELETE FROM 旧 enum value migration

**触发条件**：PR 修改 SQLAlchemy / Pydantic enum (删除 enum value)，且 DB 表中可能有该 enum value 的 row。

**核心 insight**：enum hard-cut 部署后，应用启动期反序列化 DB 中现有 row 时，遇到已删除 enum value → ValueError / DeserializationError → 启动失败。必须先 chain DB migration `DELETE FROM <table> WHERE <column> = '<old_value>'`，再 deploy enum 删除代码。

**first-application demo**：task #47 PR #1910 删除 `ModelCapability.RERANK = "rerank"` enum value：

- 新 migration `20260430034600-3c7d2f81b5e9.py` chain 在 ziang `a8f4c2d9e1b7`（task #38 PR #1898 删除 `model_use.scenario='retrieval_rerank'`）之后
- migration upgrade: `op.execute("DELETE FROM model WHERE capability = 'rerank'")`
- migration docstring 显式说明 "before the enum hard-cut"
- downgrade: `pass` + docstring 注明 "deleted rows cannot be reconstructed safely"

**实施步骤**：enum hard-cut PR 必走 4 步：

1. **Step A**：grep DB schema 列出所有可能含旧 enum value 的 column / table
2. **Step B**：写 migration `DELETE FROM <table> WHERE <column> = '<old_value>'` chain 在最新已合并 migration 后
3. **Step C**：migration docstring 显式说明 "before the enum hard-cut" + downgrade `pass` + 不可逆理由
4. **Step D**：CR 时 verify migration 在 enum 删除代码之前 chain（migration revision DAG）

**对应 Lesson #12 v7 (caller sig + backend schema + runtime fallback 三层)**：DB 是 backend schema 层的具体实施，enum hard-cut 必须 chain DB migration 是 v7 在 DB 实施层的样态。

### Lesson #14：架构 invariant 删除多轮迭代收尾（task #35 6 轮 fix-forward 实证）

**触发条件**：PR description 含「彻底删除」/「全删」/「整删」类 sweeping cleanup directive。

**核心 insight**：架构 invariant 删除涉及多 layer（runtime / schema / DB / config / docs / generated artifacts / deploy / env / CI），单 PR 通常无法一次性 cover 全集。每轮 grep gate verify surface 下一批残留 → fix-forward task → 多轮迭代收尾是工程常态，**不是 spec 失败**。

**first-application demo**：task #35「彻底删除 rerank」走完 **6 轮迭代**：

1. **task #36 BE core**（PR #1899）：pipeline._rerank + 4 runner + RerankService + invocation_service + endpoint
2. **task #37/#39 UI/docs**（PR #1897/#1908）：前端 SearchTest + quickstart docs + MCP docs
3. **task #38 MCP + DB scenario**（PR #1898）：MCP tool 参数 + model_use retrieval_rerank scenario migration
4. **task #46 boundary test**（PR #1906）：test_no_rerank_in_mcp invariant lock
5. **task #47 model_platform residual**（PR #1910）：ModelCapability.RERANK enum + DashScope/Jina presets + DB row migration
6. **task #49 deploy/env + comment residual**（PR #1911）：CACHE_RERANK_TTL_SECONDS env/values/secrets + view_models stale shim comment
7. **task #40 final 验收**：32 tests pass + helm lint + grep gate 0 active path → task #35 正式 close

**实施模式**（避免「spec 失败」误判）：

1. **Step A**：spec 列「彻底删除」directive 是正确，**不需要预先列尽全集**（list 必有遗漏）
2. **Step B**：每轮 fix-forward 走 grep gate verify → surface 残留 → 立 fix-forward task
3. **Step C**：迭代到 grep gate 0 active path 闭环（仅剩 allowlist：boundary test 自身 / 历史 migration 记录 / 删除说明 docstring）
4. **Step D**：最终 验收 lane（task #40 模式）跑 final grep gate + smoke + helm lint 三层 confirm

**CR 应用**：CR sweeping cleanup PR 时不要 expect 单 PR 一次到位，而是 verify「本轮 grep gate 是否 surface 下一批残留 + fix-forward task 链是否 actively 推进」。spec 完整性 ≠ 单 PR 完整性，spec 完整性 = **多 PR 迭代收尾闭环**。

### Lesson #15：file-move PR 必须三步 verify（task #33 P1 PR #1917 实证）

**触发条件**：PR diff 含 `git mv` / 文件位置改动 / 子目录化重组（无内容改动，仅 path 变化）。

**核心 insight**：`pytest --collect-only` 验证 import-time 不破，**但**不跑 fixture / test body 里的 `Path(__file__).resolve().parents[N]` 运行时路径解析；`additions:0 deletions:0` 也只 verify 内容不变，不 verify path-relative behavior。三层 verify 缺一不可。

**first-application demo**（task #33 P1 PR #1917 unit_test 子目录化）：
- collect-only baseline = 1434 = post-move = 1434（diff = 0）✅ → reviewer (含 chenyexuan / 冬柏 / huangzhangshu) 全 LGTM
- merge 前 CI 实跑 → **15 test failures** 因 `Path(__file__).resolve().parents[2]` 现指 `tests/` 不再是 repo root
- fix-forward `parents[2]` → `parents[3]` 修 3 文件 → CI 10/10 重跑全绿

**Mandatory 三步 verify**（同 PR 内 必须 全跑）：

1. **Step A — collect-only diff = 0**：`pytest <subdir>/ --collect-only -q` baseline 跟 post-move 数量一致（import 层）
2. **Step B — pytest <subdir>/ -q 真跑**：完整跑 mv 涉及子目录的 test body（fixture / `Path(__file__)` 解析层）。**禁止仅依赖 collect-only**
3. **Step C — `grep -n "Path(__file__)" <moved files>`**：扫所有 mv 文件硬编码相对路径，每条按新位置调整 `parents[N]` 或换 `project_root` 锚点（path resolution 层）

**CR 应用**：reviewer 看 file-move PR 时按上述三步 cross-check，**仅 collect-only diff = 0 不构成 LGTM 充分依据**。reviewer cite 必须包含 Step B + Step C 实证。

**关联 own-up**：冬柏 msg=cd428dc1 + chenyexuan msg=18acb5e7 双方 own-up — 仅 collect-only 不可能 catch `__file__`-relative path break，必须扩三步走。

### Lesson #12 v6.4：function-self-verify ≠ aggregation-chain-verify（task #30 B1 PR #1923 BLOCKER 1 实证）

**触发条件**：CR 时 verify 一个 helper 函数自身正确性后，必须 walk 它在 caller 链路中的 **聚合 / 批处理 / 循环 scope** 是否仍保留 invariant。

**核心 insight**：函数自身在 element-level 验证正确，**不**等于 caller 在 aggregation-level 应用正确 — 同一个 helper 在 per-element vs document-level union 调用下行为完全不同。

**first-application demo**：task #36 PR #1899 fix-forward² L148 case（hurl assertion `count >= 3` 在 `/api/v2/models` endpoint 是「全部 model 计数」, 在 `/api/v2/model-uses` endpoint 是「scenario 计数」— 同 jsonpath 在不同 endpoint scope 语义不同）。

**second-application demo**：task #30 B1 PR #1923 commit `163b77c1`（`source_chunk_ids_validity` 函数自身严格 verify allowed_set subset，但 caller `aggregate_sample` 用 document-level union 而非 per-window — window-0 entity 引用 window-1 chunk_id 会被 union allowed_ids 误判为 valid，跨 window 污染逃过 verify）。

**实施步骤**（CR 时必走）：

1. **Step A**：grep helper 函数自身 unit test，verify element-level 行为正确
2. **Step B**：grep helper 的 caller 函数（按 import / call site 列表），列每个 caller 的 scope 维度（per-element / per-batch / per-document / per-collection）
3. **Step C**：判断每 caller 的 scope 是否跟 helper 的 invariant 维度匹配（element 级 invariant 必须 element 级调用，不能用 union / aggregation 替代）
4. **Step D**：缺任一 caller 的 scope-correctness verify → BLOCKER fix-forward 加 caller-chain test 钉 cross-scope 污染 case

**对应 Lesson #12 v6 family**：v6.1 function scope walk / v6.2 endpoint scope walk / v6.3 data type scope walk / **v6.4 aggregation chain scope walk** — 都是「surface signal 字面相同但底层 scope 不同」反 pattern 在不同维度的应用。

**CR cross-check 应用**：reviewer 看 helper + aggregation 类 PR 时，仅 cite「helper 自身 unit test pass」**不构成 LGTM 充分依据**，必须 cite 调用层 caller chain test 实证 aggregation scope 不漂。

### Lesson #12 v7 second-application demos（task #30 A2 + B1 实证累计）

Lesson #12 v7「caller signature → backend schema → runtime fallback 三层 grep」入仓后（PR #1916 commit `b3c3a0e0`，2026-04-30）数小时内连续触发 3 类 second-application demos，每类 cover 一个三层中漏 verify 的具体层次：

**second-application 1 — backend schema 层漏 verify (task #32 PR #1909 GraphEvidenceRef composite key)**：
- 来源 spec PR #1905 huangheng + 架构师 double own-up，Weston msg=7500e57d 第三 reviewer cross-check catch
- 模式：caller signature 验证完整 + runtime 投影完整，但 schema 字段不含 caller 实际需要的 composite key 全集（chunk_id 单字段不能取代 document_id+chunk_id 复合键）
- sub-form: **v7.1 composite key invariant**（已 sediment 进 PR #1916 § 四）

**second-application 2 — backend schema 层漏 verify (task #30 A2 PR #1921 KnowledgeGraphConfig schema 漏 2 字段)**：
- 来源 ziang msg=f7dc20ef + Weston msg=9f356fe9 双 reviewer 独立 surface
- 模式：runtime resolver 读取完整 + caller 字段定义完整，但 Pydantic `BaseModel` 默认 extra-ignore policy 让未在 schema class 内显式声明的字段在 `model_validate` 时被 silently dropped
- 修法：每个 runtime resolver 读取的 collection-level config 字段必须在 KnowledgeGraphConfig 中显式声明，跟 OpenAPI / `web/src/api-v2/schema.d.ts` 同步 regen
- sub-form: **v7.2 Pydantic schema layer mandatory exposure** — 任何 `_resolve_*_kg_config(field)` 必须 grep verify field 在 KnowledgeGraphConfig class 中存在；不允许「runtime resolver 接 magic string field 不在 schema 中暴露」

**second-application 3 — caller signature 层默认值漂移 (task #30 B1 PR #1923 response_format default false)**：
- 来源 ziang msg=c170ad75 BLOCKER 2 catch
- 模式：backend (graph_extractor.py:144) 已 lock `response_format={"type":"json_object"}` 作 production invariant，但 caller (benchmark runner CLI) 默认 false → benchmark baseline 不等价 production runtime
- 修法：跨 PR caller chain 的同一 contract 在所有入口默认值必须一致；caller 默认应反映 production runtime 的 invariant，不允许「production ON / benchmark OFF」类不对称默认
- sub-form: **v7.3 cross-PR default value alignment** — 同一 contract 在 caller chain 的多个入口（CLI / API / config / test fixture）默认值必须一致；如果 production runtime 已 lock 某 default，所有 caller 入口必须同步 lock

### Lesson #12 v8：fake guardrail anti-pattern（task #30 A2 PR #1921 BLOCKER 2 实证）

**触发条件**：写 verification / guardrail / cap-overflow check 函数时。

**核心 insight**：guardrail 函数如果接受 synthetic / static / hardcoded 参数估算（不消费 actual runtime data），就是 **fake guardrail** — 看起来防回归，实际 runtime 真实数据永远不会触发它。

**first-application demo**：task #30 A2 PR #1921 first iteration `_estimate_window_prompt_tokens(window_chunk_count, base_chunk_size=400)` — 硬编码 `base_chunk_size=400` token 估算，10 个 4000-char chunk 真实 window 估 ~5.4k 不会触发 32k cap，但实际 rendered prompt 是 ~40.5k 应该触发 — guardrail 形同虚设。

**修法（fix-forward `6d2db64`）**：

- guardrail 函数 signature 改为接 actual runtime data: `_estimate_window_prompt_tokens(window: _GraphChunkWindow, *, few_shot_locale: str | None = None)`
- 内部 sum `_estimate_graph_chunk_tokens(chunk)` over 真实 chunk text + 实际 few-shot envelope (only when opt-in)
- 保留 synthetic 调用模式 (`window_chunk_count=N, base_chunk_size=K`) 仅用于 boundary test 的公式 verify，**不用于 runtime path**
- runtime path test 钉「10 chunks × 4000-char real text > 32k 触发 skip」实证 guardrail 真生效

**实施步骤**（CR 时必走）：

1. **Step A**：grep guardrail 函数 signature — 看是否 accept 真实 runtime data 或 synthetic placeholder
2. **Step B**：runtime path call site grep — verify guardrail 调用时传的是真 runtime data（actual window / actual chunk text / actual config）
3. **Step C**：boundary / invariant test 必须含 runtime-path test — 用 realistic large-content fixture 触发 guardrail，不能仅 synthetic placeholder
4. **Step D**：synthetic 模式（如 boundary test 公式 verify）必须显式标注 "test-only" / "formula-only"，不允许混用

**对应 Lesson family**：跟 Lesson #12 v6 (scope walk) 同根「surface check 字面 OK 不等于 runtime 行为 OK」；跟 Lesson #12 v7 (3-layer grep) 互补 — v7 cover field/value 层，v8 cover guardrail/check function 层。

**CR cross-check 应用**：CR 看任何「cap / guardrail / overflow check / validation」类函数 PR 时必走 4 步；缺 runtime-path test → BLOCKER fix-forward。

### Lesson #13 v3 application demo：未实证 invariant 不预先锁（task #30 A2 PR #1921 NIT defer）

Lesson #13 v3 (boundary 不重复事实保证 invariant) 反过来应用：**未实证的 invariant 也不应预先锁**。锁过紧的类型约束（如 `Literal["zh", "en"]`）在 spec 时点会 cap 未来扩展空间；除非已经实战验证哪些 value 真的需要 + 哪些不需要，否则保留 `Optional[str]` open string + resolver 层 allowlist warning 给未来扩展留空间。

**first-application demo**：task #30 A2 PR #1921 NIT (Weston msg=c9c561fa + ziang msg=eed0d017)：

- `graph_extraction_few_shot_locale: Optional[str]` (default None)
- 当前 A3 supports `zh` / `en` / `cross_chunk` 但 Phase B benchmark 数据没出来前不锁 Literal — 不预先把 value space cap 死
- 等 Phase B 跑完 benchmark 数据看哪些 locale 真用 + 哪些没用，再决定是否收紧到 Literal

**判断准则**：

- 已实证 invariant（DB 删 row + dir 整删 + enum hard-cut 等）→ Lesson #13 v3 boundary test 不重复 fact
- 未实证 invariant（type narrowing / value space cap / behavior contract pre-locked）→ 也不应预先锁，给 Phase B 数据出来后再 narrow

**对应 Lesson #13 v3 family**：v3 (boundary 不重复事实保证) + 本条 (未实证 invariant 不预先锁) — 都是「不在错误时机 codify」的应用，不同方向（v3 是 over-codify 浪费，本条是 over-codify 限制扩展）。

### Lesson #12 v7.4：external API raw contract verify（task #61 P0-B PR #1930 实证）

Lesson #12 v7 三层 grep（caller signature / backend schema / runtime fallback）应用到**跨 backend adapter** 时必须扩展第 4 层：**外部 API raw contract 实测**。in-tree docstring 假设的「backend 返回值约定」可能跟外部 API 实际行为不一致 — trust docstring assumption 不 grep verify external API runtime behavior 是 v7 反 pattern 的跨边界变体。

**first-application demo**：task #61 P0-B PR #1930 first iteration（commit `052665fc`）的 Qdrant euclid raw score convention：

- `aperag/vectorstore/base.py` docstring 写「Qdrant native ... yields negative L2 distance」假设跟 PGVector `_score_expr = -(<->)` 同 convention
- 实际 Qdrant `query_points()` Euclid distance 返 **positive L2 (smaller-is-better)** per `qdrant_client.local.LocalCollection.search()` `DistanceOrder.SMALLER_IS_BETTER`
- shared helper `normalize_score("euclid", positive_L2)` 走 `max(0.0, -positive_L2) = 0.0` → 全 clamp 成 score=1.0；threshold pushdown 反向无效 → Weston msg=86e05a8e local `:memory:` 复现 `score_threshold=0.9 → []` empty
- huangheng line-level CR (msg=5eb7315c) miss own-up：直接读 base.py docstring 没 grep verify external API raw return convention，trust in-tree assumption 是 v7 三层 grep 反 pattern 跨边界扩展（应该有 v7.4 第 4 层 external API contract verify）

**修法（fix-forward `1e30a00e`）**：

- shared helper `normalize_score` / `denormalize_threshold_to_native` 接口稳定（input contract = canonical higher-is-better raw）
- Qdrant adapter `search()` 在 euclid path 自己 negate raw `p.score` 再进 helper；threshold pushdown 在 euclid path `native_threshold = -inv` 翻回正 L2 upper-bound
- base.py docstring 显式声明 「Canonical raw conventions assumed below」+ 标注 Qdrant Euclid 是 asymmetric metric，adapter 负责 raw → canonical 转换
- responsibility 在 adapter 层 cohesive；不暴露 `backend=` 参数避免 helper 接口扩散

**实施步骤**（v7 sub-form 升级到 4 层 grep）：

1. **Layer 1**：caller signature grep（v7 现有）
2. **Layer 2**：backend schema layer grep（v7 现有，v7.2 sediment）
3. **Layer 3**：runtime fallback grep（v7 现有）
4. **Layer 4 (新)**：external API raw contract grep — 跨 backend adapter / 第三方 SDK / 外部 service 边界时，必须 grep 真实 SDK / API 文档 verify return value 方向 + 范围 + sign convention，不 trust in-tree docstring 假设

**对应 Lesson family**：v7 (3-layer in-tree grep) + 本条 v7.4 (external API contract 4th layer) — 跨边界 contract verify 是 v7 family 的跨进程扩展，跟 v6.4 (function-self-verify ≠ aggregation-chain-verify) 同根「single-source verify 不等于 chain verify」。

**CR cross-check 应用**：CR 看「跨 backend adapter / 第三方 SDK 调用」类 PR 时必走 4 层 grep；Layer 4 缺位 → BLOCKER fix-forward。

### Lesson #12 v8 second-application：test docstring fake guardrail（task #61 P0-G1 PR #1927 实证）

Lesson #12 v8 (fake guardrail anti-pattern) 跨 layer 应用：**test 层 docstring 也可能是 fake guardrail**。case docstring 声明「contract X must hold」但 assertion 不 pin contract X → docstring 看起来防回归，实际 backend 漂移 contract X 时 case 仍 pass 不 catch。

**second-application demo**：task #61 P0-G1 PR #1927 first iteration（commit `381d7a75`）的 `bulk_upsert_entity_with_lineage_parts` 测试 case：

- `_round_trip` docstring 写「3 distinct (document_id, parse_version) parts visible」+ assertion 仅 `keys == {("doc-A", "v1"), ...}` — 钉 lineage key 但**没钉 description 文本**
- backend 正确写 lineage 但 bulk path 把 description text 丢掉（或 replay 后保留旧文本）→ docstring 声称防该回归，assertion 不 catch
- huangzhangshu testing primary CR (msg=5bbc5d1a) catch：「会漏掉一种真实回归：后端正确写 lineage key，但 bulk 路径把 description text 丢掉或 replay 后保留旧文本」

**修法（fix-forward `1953933a`）**：

- `_round_trip`：3 个 (doc_id, parse_version) parts 加 `got.description_parts` key→text 全 verify (`from-doc-A-v1` / `from-doc-A-v2` / `from-doc-B-v1`)
- `_dedup_last_wins_within_bulk`：钉 same-key parts 必须 keep `last-write` not `first-write`
- `_replaces_existing_same_key`：钉 bulk 必须 overwrite single 写的 description text (`bulk` not `single`)
- `_replay_is_idempotent`：钉 replay 必须 overwrite first call's description（last-wins on replay，**顺手 fold 第 4 处** — 原 NIT 提 3 处但 replay 同 family）

**实施步骤**（CR 看 boundary / contract test 类 PR 时必走）：

1. **Step A**：每个 case docstring 列出的 contract claim → 对应 assertion 必须 pin
2. **Step B**：assertion 缺位 → docstring 不能声称防该回归，要么补 assertion 要么改 docstring 不夸大
3. **Step C**：跨 case 找同一 contract 的 dual-side 应用（如 single + bulk 双路径必须双侧 assertion）— Lesson #13 v2.x dual-side rewrite 跟 v8 在 test 层交叉

**对应 Lesson family**：v8 first-application 在 production code (`_estimate_window_prompt_tokens` synthetic placeholder) / 本条 second-application 在 test layer (docstring claim ≠ assertion pin) — 都是「surface check 字面 OK 不等于实际 verify」。

### Lesson #12 v9：first-principles verify catch surface signal mistakes（task #61 双独立 source 实证）

Lesson #12 v5 (CI status 解读 trust framing 反模式) 升级到 PR scope verify 阶段：**reviewer A surface signal X 后，reviewer B / spec author / 架构师不应直接 fold X 进 P0 list 而不做 first-principles re-verify**。double-trust framing 会让 mis-characterized P0 候选混进 fix scope，浪费 fix PR 工作 + spec lock 后 fix-forward。

**first-application demo**：task #61 P0-V1 重新定性（Bryce msg=23a2f514 catch huangheng + 架构师双 first-look 错）：

- huangheng msg=ed2f2973 surface「Qdrant `qdrant_connector.py:668-670` legacy mode 不做 tenant filter → cross-tenant data leak P0 risk」
- 架构师 spec PR #1928 first draft cite 上述 file:line 为「P0-V1 数据正确性 risk CRITICAL」
- Bryce first-principles verify catch：实际 Qdrant legacy mode `_resolve_collection_name()` (line 442-446) 把 `collection_name = tenant_id` per-tenant **物理 collection 隔离**，`retrieve()` 内部不需要 filter — 不构成 cross-tenant leak。huangheng + 架构师都没 walk `_resolve_collection_name()` 的 legacy branch 实证 collection 隔离机制。
- 重新定性：P0-V1 降级 P1-V4 defense-in-depth 不对称（legacy 路径少一层 belt-and-braces filter）+ legacy mode deprecation follow-up 候选

**second-application demo**：task #61 P0-B PR #1930 Qdrant euclid raw direction（Weston msg=86e05a8e catch Bryce + huangheng + 架构师三 reviewer 同时 miss）：

- Bryce P0-B fix PR `052665fc` first iteration `normalize_score("euclid", raw)` 假设 PGVector 风格 negative L2
- huangheng line-level CR (msg=5eb7315c) verify P0-B 数学正确，但**没 grep verify Qdrant 实际 raw score convention**（trust base.py docstring 假设）
- 架构师 ratify standing (msg=06902347) 也跟 huangheng 同样 miss
- Weston local `:memory:` 复现 `score_threshold=0.9 → []` empty + near/mid/far 全返 1.0 → first-principles verify catch real correctness bug

**判断准则**：

- reviewer A surface signal X → reviewer B 必须**独立 first-principles re-verify**（不只是 trust framing fold-in）
- spec author / architect 在 spec lock 前必须跑「first-principles pre-check」每条 P0：「为什么这是 leak/correctness/atomicity 风险？换种实现可能不是？」
- multi-reviewer cross-check 收敛后还有 surface signal mistake 漏过 → 是 cross-reviewer 各自 trust framing 没 first-principles verify 的 systemic 问题，不是 single-reviewer 失误

**对应 Lesson family**：v5 (CI status trust framing) + 本条 v9 (PR scope first-principles verify) — 都是「reviewer 不能用 framing 替代实际 verify」的不同 layer 应用；v5 cover CI status 层，v9 cover PR signal/evidence 层。

**CR cross-check 应用**：CR 看 multi-reviewer chain reference 同一 P0/BLOCKER claim 时必须独立 first-principles re-verify；缺位 → BLOCKER fix-forward 复发。

### Lesson #13 v2.3：deploy manifest dual-side rewrite（task #61 P0-D1 PR #1929 实证）

Lesson #13 v2.x dual-side rewrite 应用到 **deploy / k8s / Helm manifest 层**：当一组配套 manifest（API deployment + worker deployment + sidecar 等）共享 invariant（同 backend 凭据 / 同 env config / 同 secret refs）时，单侧 manifest 改动必须 dual-side 同步；否则跨 service / 跨进程 silent drift。

**first-application demo**：task #61 P0-D1 PR #1929（commit `9720342`）Helm Neo4j worker env injection：

- `deploy/aperag/templates/api-deployment.yaml` 已注入 `NEO4J_URI` / `NEO4J_USERNAME` / `NEO4J_PASSWORD` env / secret refs
- `deploy/aperag/templates/indexing-worker-deployment.yaml` 缺对应 Neo4j env 注入 — 仅挂 `aperag-env` 默认空值
- 后果：Helm `GRAPH_DB_TYPE=neo4j` 时 API 读图谱有凭据，worker 写图谱拿空 Neo4j config → graph 写入路径全静默失败 → DB 0 entity / 0 relation → API 返回空数据 + Singapore graph 可视化「empty + error 混淆」FE surface 现象的 deploy root cause
- cuiwenbo task #70 P1 候选 2「FE 状态分离 / 后端 200+empty 区分」surface 当时归 FE root cause，实际是这条 deploy gap 的 silent fan-out 现象

**修法（PR #1929）**：

- worker deployment 加 `{{- if .Values.neo4j.enabled }}` 条件块 + `NEO4J_URI` / `USERNAME` / `PASSWORD` env / secret refs，跟 API deployment Neo4j 注入完全 mirror
- `helm template --set neo4j.enabled=true --set api.env.GRAPH_DB_TYPE=neo4j` render 实证 worker manifest 包含 `NEO4J_URI=bolt://neo4j-cluster-neo4j:7687` + secret refs

**实施步骤**（CR 看 deploy / Helm / k8s manifest 类 PR 必走）：

1. **Step A**：grep 同套 manifest 集合（api-deployment / worker-deployment / sidecar / cron / init / scale-helper），看 secret refs / env vars / volume mounts 是否完整 mirror
2. **Step B**：`helm template --set <toggle>` render 实证目标 manifest 真有目标 env，不只 source mirror 看上去对（Lesson #11 v5 entry-point migration sub-check 思路 — manifest render 实证类比 startup-time wire-in verify）
3. **Step C**：grep 跨 backend (Neo4j / Nebula / PG / Qdrant / PGVector) — 每 backend 的 env / secret 注入是否对每个消费 service 都 dual-side mirror，不能某 backend 修了 api 漏 worker
4. **Step D**：跨 backend follow-up scope inventory 必须显式标注（如 dongdong msg=4201465a 「Nebula 缺 Helm first-class dependency/secret」/「shape matrix 缺 3 组合」/「typed schema 缺 vector backend capability 暴露」），不能 silent assume PR 修一 backend 等于全 backend cover

**对应 Lesson family**：v2.1 (import-level dual-side) + v2.2 (value-level dual-side) + 本条 v2.3 (deploy manifest dual-side) — 都是「invariant evolution dual-side rewrite 必须同步」的不同 layer 应用；v2.1/v2.2 cover code 层，v2.3 cover yaml/manifest 层。跟 Lesson #11 v5 entry-point migration sub-check 互补 —— v5 是 startup-time wire-in 跨 process parity，v2.3 是 deploy-time manifest 跨 service parity。

**CR cross-check 应用**：CR 看 deploy / Helm / docker-compose / e2e shape env / typed schema 类 PR 必走 4 步；缺 manifest render 实证 OR 跨 service mirror 缺位 → BLOCKER fix-forward。

### Lesson #13 v3 application demo 2：cross-source default value alignment（task #30 B3 PR #1925 实证）

Lesson #13 v3 (boundary 不重复事实保证) 跨 source application：**同 default value 在多 source 暴露时（code const / Pydantic Field description / FE generated TypeScript schema / 架构 spec doc），任一 source 改动必须同步全 source**，否则 reader / API consumer / FE typed schema 三方拿到漂移 default。

**first-application demo**：task #30 B3 PR #1925（commit `dae43f5`）`graph_extraction_window_size` 默认值锁定：

- `aperag/indexing/graph_extractor.py:81` `_DEFAULT_GRAPH_EXTRACTION_WINDOW_SIZE = 1 → 2` ✅ 第 1 source
- `aperag/schema/common.py:170` `KnowledgeGraphConfig.graph_extraction_window_size` description 「default 1 if unset」→「default 2 if unset」+ override 推荐文案 ✅ 第 2 source
- `web/src/api-v2/schema.d.ts:4963` generated TypeScript description 必须 regen — 第一版 PR diff 漏（commit `67f578f` 仅修 spec NIT 没 regen schema），Weston msg=1e6b0838 BLOCKER catch
- `docs/zh-CN/architecture/task-30-graph-chunk-window-spec-v1.md` § 3.1.1 line 85「初始 default `1` = 旧行为兼容回退」内部不一致（§ 4.2 lock default=2 但 § 3.1.1 仍说 1）— huangheng msg=bf785b12 NIT + Planetegg msg=c63acbf5 + Weston msg=1e6b0838 三独立 source 同时 surface

**修法（fix-forward `dae43f5`）**：

- `web/src/api-v2/schema.d.ts:4963` regen 跟 Pydantic Field description align
- spec § 3.1.1 line 85 重写「B3 lock default `2`...保守 override `1` / 强模型 Gemini override `5`」

**实施步骤**（CR 看 default value / public-facing constant / typed schema exposed value 类 PR 必走）：

1. **Step A**：grep 同 default value 在 codebase 所有 source — Python const / Pydantic Field / Field description / FE schema.d.ts / 架构 spec doc / migration default / Helm values default — 必须 enumerate 全 source
2. **Step B**：每 source 是否同步更新；missed source → BLOCKER（如 schema.d.ts 漏 regen → FE / API consumer 拿旧 default 是 typed contract drift）
3. **Step C**：spec doc 内部一致性（不同 § 章节引用同一 default 必须同值，避免 reader 在不同段落看到漂移）— Lesson #14 architect invariant evolution multi-iteration cleanup 同 family
4. **Step D**：generated schema (OpenAPI / TypeScript / GraphQL) 必须 explicit `make`/`yarn generate` regen + commit，不只手改 Pydantic 假定 generation

**对应 Lesson family**：v3 (boundary 不重复事实保证) 反 pattern — default value 是事实但其 exposure source 必须 dual/multi-side rewrite 同步；本条 cross-source alignment 是 v3 跨 source layer 的 dual-side 应用，跟 v2.1/v2.2/v2.3 dual-side rewrite family 同根。跟 Lesson #14 architect invariant evolution multi-iteration cleanup 互补 — 默认值漂浮也是 invariant evolution 漏 source。

**CR cross-check 应用**：CR 看 default value 改动类 PR 必走 4 步；schema.d.ts / spec 章节内部不一致 缺位 → BLOCKER fix-forward。

### Lesson #14 application demo：task #30 B3 spec internal default 漂浮 cleanup（task #30 B3 PR #1925 实证）

Lesson #14 (架构 invariant 删除多轮迭代收尾) 应用到 **spec 内部 default value 漂浮 cleanup**：spec § 早期章节（spec lock 时点的「初始 default」描述）在 default 经 benchmark 数据 lock 后必须同步收尾，不能只改 § 4.x lock 章节留 § 3.x 「初始 default」历史残留。

**second-application demo**：task #30 B3 PR #1925 fix-forward `dae43f5` § 3.1.1 line 85 cleanup：

- spec § 4.2 在 B3 PR amend 时 lock default=2 + 完整 sweet spot rationale + B2 evidence 表
- spec § 3.1.1 line 85（「5 业务边界」初始 default 描述章节）仍写「初始 default `1` = 旧行为兼容回退，benchmark 后再 confirm 默认值」— 内部不一致
- maintainer / reader 阅读时如先看 § 3.1.1 不看 § 4.2 lock 章节会拿旧 default
- huangheng / Planetegg / Weston 三独立 source 同时 surface 该 NIT — sediment 强度 high 升格为 v3 cross-source alignment + Lesson #14 multi-iteration cleanup demo

**修法（fix-forward `dae43f5`）**：

- spec § 3.1.1 line 85 重写「B3 lock default `2` per § 4.2 sweet spot；保守 override `1` / 强模型 Gemini override `5`」

**对应 Lesson family**：first-application 在 task #35 6 轮 fix-forward (PR #1899/1897/1898/1906/1910/1911) — 大型 sweeping cleanup directive 单 PR 无法一次 cover；本条 second-application 在 spec doc 内部 default 漂浮 cleanup — small-scale invariant evolution 单 spec 多 § 章节也需要 multi-iteration cleanup 收尾。Lesson #14 不仅 cover 跨 PR 大型 invariant 删除，也 cover 跨 § 同 spec 内部 invariant evolution。

**CR cross-check 应用**：CR 看 spec amend 类 PR 必 grep 全 spec 跨 § 章节同 invariant 引用，避免 lock 章节修了 描述章节漂移。

### Lesson #16：CI workflow paths filter dead reference 反 pattern（task #61 P0-W1 PR #1926 实证）

CI workflow `paths:` filter（GitHub Actions / GitLab CI / 等）跟实际 source code 路径 drift 是「silent CI gate bypass」反 pattern — 比 Lesson #12 v8 fake guardrail 更隐性：fake guardrail 至少 CI 跑了但语义失语；workflow paths filter dead reference **CI 根本不跑**，所有 cross-adapter / cross-backend / cross-domain regression test 形同虚设。

**first-application demo**：task #61 P0-W1 PR #1926（commit `a0403cf`）compat-test paths filter dead reference：

- `.github/workflows/compat-test.yml` paths filter 指向 `aperag/domains/knowledge_graph/graphindex/**`
- Wave 7 graph 层重写时 graph store impl 已迁到 `aperag/indexing/graph_storage/{neo4j,nebula,postgres}.py`，旧 `aperag/domains/knowledge_graph/graphindex/` 目录已删（dead reference 不是 stale）
- 后果：任何 PR 修 `aperag/indexing/graph_storage/*.py` 都不 trigger Backend-Compat-Test workflow — 30+ cross-backend test 形同虚设
- 冬柏 task #67 testing scan (msg=3e93bb64) 实证「该目录已被 Wave 7 删除 — `ls` 直接 No such file or directory」+ chenyexuan workflow gate audit (msg=f298011e) surface paths filter stale
- task #25 Neo4j labels 500 vs Nebula/PG pass 这种 P0 cross-adapter bug 没在 compat-test 抓到的 root cause — workflow filter dead reference 完全跳过实际 backend code path

**修法（PR #1926）**：

- 加 `aperag/indexing/graph_storage/**` 真实 graph store impl 路径
- 保留 legacy `aperag/domains/knowledge_graph/graphindex/**` defensive fallback（low cost，移除 footgun）
- inline comment 写明 historical drift + invariant defense rationale

**实施步骤**（CR 看 file-move / dir-rename / wave-style 重写 类 PR 必走）：

1. **Step A**：grep 全仓 `.github/workflows/*.yml` paths filter — 列出所有引用 source 目录的 filter
2. **Step B**：file-move 后实证目标目录存在（`ls` 不 No such file or directory）；若 source 目录已删则 paths filter 是 dead reference 必同步修
3. **Step C**：file-move 跟 workflow paths filter sync 是 Lesson #15 file-move 3-step verify 第 4 步扩展（grep `.github/workflows/*.yml` paths 同步）— Lesson #15 升级到 v2 4-step verify
4. **Step D**：CI gate 真触发实证 — paths filter 修后下一个修目标目录的 PR 必看 workflow 真 trigger（不是 source mirror 看上去对）

**对应 Lesson family**：跟 Lesson #15 (file-move 3-step verify) 同 family — Lesson #15 cover code import / hurl reference / boundary test fixture 三步，本条扩展第 4 步 CI workflow paths sync。跟 Lesson #12 v8 fake guardrail 同根「surface check 形同虚设」但 v8 是 guardrail 函数 fake，本条是 CI gate 全 dead。

**CR cross-check 应用**：CR 看 file-move / wave-style 重写 类 PR 必走 4 步；workflow paths filter sync 缺位 → BLOCKER fix-forward。

### Lesson #17：backend 收敛 contract 优于上层 fork（task #69 + task #70 cross-PR 一次性收敛实证）

simple-stable + private-deploy paramount directive (earayu2 msg=1224bec8) 在 cross-adapter / cross-backend contract 设计时具体应用：**当 backend 可以在 adapter 层收敛同一 contract（同 score 范围 / 同 score 方向 / 同 filter exception 类型 / 同 capability flag）时，优先 push backend adapter 收敛而不是让上层（FE / agent / MCP / API consumer）加 backend-aware branch 复制 backend 差异**。

**核心 insight**：上层 fork (FE 加 `metric === 'cosine' ? ...` / `1 - score` / `score < 0.5 ? "low" : "high"` 类条件分支) 把 backend 差异复制到所有消费 surface，每加一个新消费 surface 都得 dup branch logic — 长期维护成本高 + drift risk 高。backend adapter 层收敛 (在 backend code 接口处一刀切转换成统一 contract) 维护点单一 + 上层消费 simple unconditional。

**first-application demo**：task #69 P0-B（PR #1930 vector adapter score normalization）+ task #70 P1 候选 1 cross-PR 一次性收敛：

- task #70 cuiwenbo msg=dfebf706 surface FE「PGVector cosine_distance（0=match）vs Qdrant cosine_similarity（1=match）— 同 query 跨 adapter 显示语义反向」P1 候选 1，初版修法是「FE 加 distance/similarity 标签 + branch」
- task #69 P0-B 实施时 backend 一刀切收敛 0-1 higher=better contract（`normalize_score()` + `SearchHit.__post_init__` validator + `VectorStoreConnector.search()` docstring contract）
- cuiwenbo verify (msg=cedc7703) FE 三处调用点都是 raw 显示无 manual flip：`web/src/...search-result-drawer.tsx:88` + `agent-turn-renderer.tsx:1161` + `message-reference.tsx:63` 全 `(score || 0).toFixed(2)` pattern
- backend 收敛后 task #70 P1 候选 1 自动 fully resolved + FE 0 改动 — 「FE 加 distance/similarity 标签」修法变 obsolete

**判断准则**（CR 看 cross-adapter / cross-backend 差异类 PR 时）：

1. **第一选择**：backend adapter 层是否能用 helper / DTO validator / Protocol contract docstring 一刀切收敛差异？如果能 → 选这条
2. **第二选择**：差异本质不可收敛（如 hint 名称 `ef_search` vs `hnsw_ef`）→ explicit capability declaration in adapter Protocol docstring + `typed schema` 暴露 capability flag，让上层 read flag 而不是 read backend type
3. **最后才选**：上层加 backend-aware branch — 仅当上述两条不可行时（如 FE 必须做 backend-specific UI），且必须显式标注「FE branch 是 last-resort，跟 spec § X cross-link 防 silent drift」

**对应 Lesson family**：跟 simple-stable + 私有化部署免维护 directive (`memory/feedback_simple_stable_deploy_and_forget.md`) 同源。跟 Lesson #11 v5 (entry-point migration cross-process parity) 互补 — v5 是「同 entry-point 跨 process」parity，#17 是「同 contract 跨 backend」parity，同 family 不同维度。跟 Lesson #14 (架构 invariant 删除多轮迭代) 配对 — 上层 backend-aware fork 一旦累积，未来收敛时是 multi-iteration cleanup（task #35 6 轮 fix-forward 实证），#17 在设计时点防止 fork。

**CR cross-check 应用**：CR 看「上层加 backend-aware branch / 处理 backend 差异」类 PR 时必先问「能否 backend adapter 层收敛？」；若能 → push backend 收敛 PR 不接受上层 fork PR。

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

### 5.5 失败注入方法规范（冬柏 msg=d56bb0f7 补充）

任何「Redis 丢消息 reconciler 补漏」/「worker crash」/「DB 写失败」类失败场景测试，**禁止 mock client 绕过真路径**。必须用真实失败注入手法：

- **Redis 丢消息 / Redis 重启**：`kubectl scale redis --replicas=0` 暂停 + 恢复，或 `iptables -A OUTPUT -p tcp --dport 6379 -j DROP` 模拟网络断开
- **worker crash**：`kubectl delete pod indexing-worker-...` 强制 pod kill，verify reconciler 60s 心跳超时回收
- **DB 写失败**：iptables 阻断 PG 端口，或临时撤销 DB 连接权限模拟 transient failure
- **rollout surge 连接池打满**：`kubectl rollout restart api` 触发 rollout，监测 PG `max_connections` 是否被 surge 配置打满

**禁止 mock 绕过原因**：mock 测出的「reconciler 能补漏」可能在真实路径上失效，因为真实路径含 Redis 客户端重连 / asyncpg 连接池行为 / k8s probe 抢占资源等 mock 不能复现的副作用。Lesson 沉淀来源：Wave 3 PR #1729 mock 路径过 + 真路径 fail。

---

## 六、CR 历史 sediment 引用

- `memory/feedback_e2e_dataflow_trace.md` Lesson #11 + Lesson #11 v5 (entry-point migration) + Lesson #12 + extension v3 + extension v4 + Lesson #12 v5 (CI status trust framing) + Lesson #12 v6 (scope walk) + v6 sub-form (v6.1/v6.2/v6.3) + Lesson #12 v7 (3-layer grep) + v7.1 (composite key invariant) + v7.1 sub-form (backend + acceptance 双层 verify) + Lesson #13 + v2.1 (import-level) + v2.2 (value-level) + v3 (boundary 不重复事实保证) + Migration chain 时序 invariant + Lesson #14 (架构 invariant 删除多轮迭代收尾)
- `memory/feedback_collab_before_solo_pr.md` 同 tag/scope 多 agent 默认 co-design 单 PR
- `memory/feedback_object_store_path_drift.md` `OBJECT_STORE_LOCAL_ROOT_DIR` 漂移诊断 playbook
- `memory/feedback_one_shot_no_phased_rollout.md` 一次性不分阶段 directive
- `memory/feedback_simple_stable_deploy_and_forget.md` simple-stable + 私有化部署免维护
- `memory/feedback_cr_must_cross_reference_arch_doc.md` CR 必对照架构文档逐条 invariant 走
- `memory/project_task17_pr_1884_active_focus.md` task #17 ship 完成节点（merge `5a0aa804` / 6 hard gate 全实证 / Planetegg 压测数据 / phase 2 directive）
- PR #1893 hot-fix: `aperag/bootstrap/__init__.py` + `wire_cross_domain_di_seams()` + `tests/boundaries/test_worker_di_parity.py` AST 级 3 重防回归 (commit `d4b65e27`)，Lesson #11 v5 first-application demo
- PR #1909 (commit `8d5ffa97`): `GraphEvidenceRef` composite key + `_lineage_to_evidence_refs` 投影层，Lesson #12 v7.1 backend 投影层 textbook
- PR #1912 (commit `eb2a805b`): `tests/integration/test_graph_evidence_refs_chain.py` 跨 endpoint chained chain 4 case 验证，Lesson #12 v7.1 acceptance 层 textbook（跟 PR #1909 配对完整）
- task #35 6 轮 fix-forward 收尾 (PR #1899/#1897/#1898/#1906/#1910/#1911 + task #40 final 验收) ，Lesson #14 架构 invariant 删除多轮迭代收尾 first-application demo
- PR #1921 (commit `6d2db64`): A2 5 const co-scale + Pydantic schema layer 漏 own-up + fake guardrail anti-pattern 双 BLOCKER fix-forward — Lesson #12 v7.2 + v8 first-application demo
- PR #1923 (commit `163b77c1`): B1 benchmark harness window-scoped validity + response_format default ON 双 BLOCKER fix-forward — Lesson #12 v6.4 + v7.3 first/second-application demo
- PR #1922 (commit `0058507e`): chenyexuan task #33 P1 Lesson #15 file-move 3-step verify sediment（独立条目）— task #33 Layer 2 sediment trail
- PR #1926 (commit `a0403cf`): chenyexuan task #61 P0-W1 `compat-test.yml` paths filter dead reference fix（`aperag/indexing/graph_storage/**` 真实路径加进 + 保留 legacy `aperag/domains/knowledge_graph/graphindex/**` defensive fallback）— Lesson #16 first-application demo + Lesson #15 file-move 3-step verify 升级到 v2 4-step（CI workflow paths sync 第 4 步）
- PR #1929 (commit `9720342`): dongdong task #61 P0-D1 `indexing-worker-deployment.yaml` Helm Neo4j env / secret refs mirror API deployment — Lesson #13 v2.3 deploy manifest dual-side rewrite first-application demo
- PR #1930 (commit `7ab474b9`): Bryce task #61 P0-A + P0-B cross-adapter filter fail-loud + score normalization — Lesson #12 v8 first-cleanup（Qdrant `_normalize_filter_input` `return None` 静默退化 → `UnsupportedFilterError(TypeError)` fail-loud）+ Lesson #12 v7.4 first-application demo（commit `1e30a00e` Weston BLOCKER fix-forward Qdrant Euclid raw direction asymmetry — external API raw convention verify）+ Lesson #12 v9 second-application demo（Weston msg=86e05a8e first-principles verify catch huangheng + Bryce + 架构师三 reviewer chain trust framing miss）
- PR #1928 (commit `ed8def22`): 架构师 task #61 spec v1 入仓（DB adapter contract matrix + capability/degradation 显式 list + sub-task 拆分 + sample 限制免责章节）— task #61 spec source of truth
- PR #1927 (commit `9c94cbc1`): 冬柏 task #61 P0-G1 `bulk_upsert_entity_with_lineage_parts` cross-backend test 38 cases × Neo4j/Nebula/PG — Lesson #12 v8 second-application demo（test docstring fake guardrail commit `1953933a` huangzhangshu testing primary CR catch description_parts text key→value assertion 缺位 fix-forward）+ Lesson #12 v5 cross-reviewer cross-check 实战实证（huangheng + ziang + huangzhangshu 三 reviewer 独立 surface 不同维度 invariant fold-in）
- PR #1925 (commit `43648f94`): 架构师 task #30 B3 `_DEFAULT_GRAPH_EXTRACTION_WINDOW_SIZE = 1 → 2` lock + B2 evidence + sample 限制免责章节 — Lesson #13 v3 application demo 2 first-application（commit `dae43f5` cross-source default value alignment：Pydantic Field description + `web/src/api-v2/schema.d.ts:4963` regen + spec § 3.1.1 line 85 cleanup 三 source 同步）+ Lesson #14 application demo（spec § 3.1.1 「初始 default `1`」历史残留 multi-iteration cleanup）+ Lesson #12 v5 cross-reviewer cross-check（PM CI status 误报 × 2 实证 必 grep verify）

---

## 七、task #17 PR #1884 final review verdict（ship 后回填，merge commit `5a0aa804` / 2026-04-30 01:13 UTC）

| 检查项 | 状态 | 备注（具体 test commit / 行号 / 数据） |
|------|------|--------------------------------------|
| §一 5 cross-check 全过 | ✅ | scope 8 项主线一致，§一 vs §三 vs §七 数字一致；framework claim 分级到 § YAGNI / escape hatch 正文 |
| §二 6 hard gate 全 verify（每条引用具体 test 文件 + 通过截图/数据） | ✅ | gate 1 `tests/boundaries/test_app_lifespan_no_workers.py` + `test_cli_worker_starts_every_runtime_loop`；gate 2 `tests/integration/test_cleanup_recovery_redis_outage.py`；gate 3 `tests/boundaries/test_api_no_objectstore_calls.py`；gate 4 health p95 ≤ 2ms 实测（Planetegg）；gate 5 `tests/integration/test_helm_pool_budget.py` PG 34 ≤ 55 budget ≤ 70；gate 6 runbook + scale dry-run |
| §三 7 实现修正全 fold-in（grep 验证应用代码 0 引入双 env / 0 嵌套 transaction / 等） | ✅ | Bryce file-by-file align ziang msg=4ea65100 7 项修正（settings module / QuotaPolicyRegistry / Helm-only 池 / 不嵌套 transaction / object store 迁出 / cleanup loop path B/C / diagnostics 鉴权 + sync URL 转换） |
| §四 lessons 全应用 | ✅ | Lesson #11 / #12 / #12 ext v3 / Mini-pattern 17 / 一次性不分阶段 全 align |
| 团队 7 lane LGTM ack 收齐 | ✅ | Bryce / huangzhangshu / ziang / Weston / 符炫炜 / Planetegg / chenyexuan + 冬柏 + cuiwenbo + dongdong + huangheng final verdict 全 ack |
| 架构师 v8.2 docs 入仓 | ✅ | `docs/zh-CN/architecture/task-system-hard-cut-v8.md` + `task-system-invariants.md` |
| 黄章书 部署/发布/回滚 docs 入仓 | ✅ | `docs/zh-CN/architecture/task-17-deployment-release-runbook.md` |
| ziang 状态机/cleanup SoT 验收 docs 入仓 | ✅ | `docs/zh-CN/architecture/task-17-state-machine-validation.md` |
| Weston scope 一致性 verify | ✅ | commit `9a6dd243` scope 收紧到 8 项主线 + invariant uniform |
| Bryce 代码主线 file-by-file align ziang 7 修正 | ✅ | Bryce file-by-file 章节 in `task-17-code-changes.md`，PR diff 全 align |
| **多文档并发压测阈值通过**（Planetegg #22 主跑） | ✅ | Planetegg 真实压测：10 docs × 5 concurrent / 153.66s 收敛 / 40/40 ACTIVE / health p95 ≤ 2ms / PG 34 ≤ 55 budget ≤ 70 / Redis 无积压 |
| **smoke regression diff = 0**（PR merge 前后同一 hurl smoke set 全 pass，新 fail = 0） | ✅ | 6 套 hurl smoke baseline 含 web_access #1794 / 模型基准 #1863 全 pass |
| 失败注入用真路径不允许 mock（按 §5.5 规范） | ✅ | Redis kill / k8s pod kill / PG iptables drop 全用真路径，无 mock client 绕过 |
| **PR `lint-and-unit` CI 全绿**（Lesson #12 extension v4 mandatory） | ✅ | commit `335fe586` Weston push 删除 PR #1876 时代 obsolete test `test_app_lifespan_launches_all_graph_indexing_worker_lanes`（与 task #17 hard cut 冲突）后 CI 全绿 |

最终 verdict：🟢 同意通过 / squash merge by 符炫炜 → `5a0aa804`（msg=10954036 final ratify / msg=c96816c4 PM 确认 / msg=eb95612a earayu2 phase 2 directive）。

---

## 八、checklist 修订记录

- 2026-04-29 23:54 commit `d9438f8`：huangheng 初版，5 cross-check + 6 hard gate + 7 实现修正 + lessons + 工作流 + verdict 表
- 2026-04-30 00:05 fold-in 冬柏 msg=d56bb0f7 补充 4 条：6 hard gate test 文件 mapping 子表（§二）/ 失败注入真路径规范（§5.5）/ 压测阈值具体化 + smoke regression baseline（§七 verdict 表新增 3 行）
- 2026-04-30 01:13 task #17 PR #1884 squash merge 至 main（commit `5a0aa804`），架构师 final ratify msg=10954036，PM 确认 msg=c96816c4
- 2026-04-30 phase 2 huangheng follow-up（commit `3b30923`）：§ 七 verdict 表全部 backfill ship 实数据 / 新增「PR `lint-and-unit` CI 全绿」mandatory 行 / § 四 新增 Lesson #12 extension v4（CI gate mandatory）+ Lesson #13（invariant evolution dual-side rewrite obsolete regression test）/ § 六 sediment 引用追加 phase 2 ship 节点
- 2026-04-30 02:03 task #17 hot-fix PR #1893 squash merge 至 main（commit `d4b65e27`），DI wire-up gap 修复（worker CLI 缺 10 个 cross-domain DI setter）；架构师 final ratify msg=df6811e4 / msg=b04ed722，PM ratify msg=5e73ce8b
- 2026-04-30 phase 2 huangheng follow-up post-hot-fix rebase sediment（commit `99f721a`，PR #1891 第 2 commit）：§ 四 新增 Lesson #11 v5（entry-point migration sub-check + Weston 三分类框架 + task #17 hot-fix 10+3+2 setter 实证清单）+ Lesson #12 v5（CI status 解读 trust framing 反模式，架构师升格独立条目）+ Lesson #12 v6（grep line number ≠ 执行顺序，必 walk function scope，huangheng NIT 2 own-up 升格）；§ 六 sediment 引用追加 PR #1893 hot-fix bootstrap module + boundary test 锚点
- 2026-04-30 task #32 Phase A close 后 huangheng follow-up 子 PR（本次 commit）：§ 四 新增 7 lesson sediment（task #32 + task #35 全 close 后多轮迭代实证累计）：
  - **Lesson #12 v6 sub-form**（v6.1 function scope / v6.2 endpoint scope / v6.3 data type scope）— 架构师 msg=9c5c32d1 升级，task #36 PR #1899 fix-forward² L148 case 实战 surface
  - **Lesson #12 v7**（caller signature → backend schema → runtime fallback 三层 grep）— task #34 rerank 调研 huangheng msg=e539848f own-up + 架构师 msg=b12fec5d thoroughness=very thorough trace 实证
  - **Lesson #12 v7.1**（composite key invariant — 修改返回字段必 verify 下游 caller 所有 required parameter 能 reconstruct）— task #32 spec PR #1905 Weston msg=7500e57d BLOCKER catch + huangheng + 架构师 double own-up
  - **Lesson #12 v7.1 sub-form**（backend 投影层 + acceptance 跨 endpoint chained chain 双层 verify）— 架构师 msg=f04b36a8 升级，PR #1909 commit `8d5ffa97` backend textbook + PR #1912 commit `eb2a805b` acceptance textbook 配对完整
  - **Lesson #13 v2.1**（import-level dual-side rewrite — 删 source 必删对应 obsolete test 文件 / 函数）— task #17 PR #1884 first-application + task #36 PR #1899 fix-forward¹ second-application
  - **Lesson #13 v2.2**（value-level dual-side rewrite — 删 source 字段 / 数据必同步 update stale assertion / count）— task #36 PR #1899 fix-forward² L103/L108 first-application + task #47 PR #1910 contract test second-application
  - **Lesson #13 v3**（boundary 不重复事实保证 invariant，只覆盖可能 drift 的 contract）— 架构师 msg=036dd8b2 升格，task #46 PR #1906 first-application demo
  - **Migration chain 时序 invariant**（enum hard-cut PR 必先 chain DELETE FROM 旧 enum value migration）— task #47 PR #1910 first-application（migration `3c7d2f81b5e9` chain 在 ziang `a8f4c2d9e1b7` 后）
  - **Lesson #14**（架构 invariant 删除多轮迭代收尾 — sweeping cleanup directive 单 PR 无法一次性 cover，多轮 grep gate verify + fix-forward task 迭代是工程常态）— task #35 6 轮 fix-forward 实证 first-application demo
- 2026-04-30 task #30 Phase A 全闭环后 huangheng follow-up 子 PR 2（本次 commit）：§ 四 新增 4 lesson sediment（task #30 A2 + B1 实证累计 + 跨 PR cross-reviewer 独立 forensics 多 reviewer catch first/second-application demo）：
  - **Lesson #12 v6.4**（function-self-verify ≠ aggregation-chain-verify — helper 函数自身正确不等于 caller 在 aggregation/批处理/loop scope 应用正确）— task #36 PR #1899 fix-forward² L148 endpoint scope first-application + task #30 B1 PR #1923 commit `163b77c1` source_chunk_ids window-scoped second-application
  - **Lesson #12 v7 second-application demos** 累计 3 类（v7.1 composite key invariant 已 sediment 进 PR #1916 / **v7.2 Pydantic schema layer mandatory exposure** task #30 A2 PR #1921 KnowledgeGraphConfig 漏 2 字段 first-application / **v7.3 cross-PR default value alignment** task #30 B1 PR #1923 response_format default 漂移 first-application）
  - **Lesson #12 v8**（fake guardrail anti-pattern — guardrail 函数 signature 必须接 actual runtime data 不能仅 synthetic placeholder）— task #30 A2 PR #1921 `_estimate_window_prompt_tokens(window_chunk_count, base_chunk_size=400)` first iteration first-application + fix-forward `6d2db64` 修法
  - **Lesson #13 v3 application demo**（未实证 invariant 不预先锁 — type narrowing / value space cap pre-locking 反过来应用 v3 不重复事实保证）— task #30 A2 PR #1921 NIT defer `Optional[str]` vs `Literal["zh", "en"]` first-application
  - § 六 sediment 引用追加 PR #1921 / PR #1923 / PR #1922 (chenyexuan Lesson #15 trail) 三 commit cross-link
- 2026-04-30 task #30 全 phases + task #61 全 P0 闭环后 huangheng follow-up 子 PR 3（本次 commit）：§ 四 新增 8 lesson sediment（task #30 B3 + task #61 全 P0 实证累计 + cross-PR 多独立 source 同源 catch trail）+ § 六 sediment 引用追加 PR #1925 / #1926 / #1927 / #1928 / #1929 / #1930 六 commit cross-link：
  - **Lesson #12 v7.4**（external API raw contract verify — Lesson #12 v7 三层 grep 跨 backend adapter / 第三方 SDK 边界扩展第 4 层 external API contract verify）— task #61 P0-B PR #1930 first-iteration `normalize_score("euclid", positive_L2)` Qdrant raw direction asymmetry first-application + fix-forward `1e30a00e` Qdrant adapter 边界 negate raw + threshold pushdown sign flip
  - **Lesson #12 v8 second-application**（test docstring fake guardrail — case docstring 声称 contract X must hold 但 assertion 不 pin contract X，docstring 看起来防回归实际 backend 漂移 contract X 时 case 仍 pass）— task #61 P0-G1 PR #1927 first-iteration `_round_trip` 钉 lineage key 不钉 description text first-application + fix-forward `1953933a` 4 处 case `description_parts` text key→value 全 verify
  - **Lesson #12 v9**（first-principles verify catch surface signal mistakes — reviewer A surface signal 后 reviewer B / spec author / 架构师不应直接 trust framing fold-in，必须独立 first-principles re-verify 防 mis-characterized P0/BLOCKER 候选混进 fix scope）— task #61 P0-V1 重新定性 Bryce msg=23a2f514 first-application catch huangheng + 架构师双 first-look 错（Qdrant legacy mode physical collection 隔离机制）+ task #61 P0-B Qdrant euclid raw direction Weston msg=86e05a8e second-application catch Bryce + huangheng + 架构师三 reviewer 同时 miss
  - **Lesson #13 v2.3**（deploy manifest dual-side rewrite — invariant evolution dual-side rewrite family 跨 yaml/manifest 层应用：API + worker + sidecar 等同套 manifest 共享 invariant 必须 dual-side 同步）— task #61 P0-D1 PR #1929 commit `9720342` `indexing-worker-deployment.yaml` Helm Neo4j env / secret refs mirror API deployment first-application demo
  - **Lesson #13 v3 application demo 2**（cross-source default value alignment — 同 default value 跨 multi-source 暴露时（code const / Pydantic Field / FE generated TypeScript schema / 架构 spec doc）任一改动必须 multi-source 同步）— task #30 B3 PR #1925 commit `dae43f5` `_DEFAULT_GRAPH_EXTRACTION_WINDOW_SIZE` Pydantic + schema.d.ts + spec § 3.1.1/§ 4.2 三 source 同步 first-application
  - **Lesson #14 application demo**（spec 内部 default value 漂浮 multi-iteration cleanup — 跨 § 章节同 invariant 引用必须同值，不能 lock 章节修了描述章节漂移）— task #30 B3 PR #1925 fix-forward `dae43f5` § 3.1.1 line 85 「初始 default `1`」历史残留 cleanup second-application demo（first-application 在 task #35 6 轮 fix-forward）
  - **Lesson #16**（CI workflow paths filter dead reference 反 pattern — workflow `paths:` filter 跟实际 source code 路径 drift 是「silent CI gate bypass」反 pattern，比 fake guardrail 更隐性：CI 根本不跑 cross-adapter / cross-backend regression test 形同虚设；Lesson #15 file-move 3-step verify 升级到 v2 4-step 加 grep `.github/workflows/*.yml` paths 同步）— task #61 P0-W1 PR #1926 commit `a0403cf` compat-test paths filter dead reference fix first-application demo + 冬柏 msg=3e93bb64 + chenyexuan msg=f298011e 双独立 source 同时 surface
  - **Lesson #17**（backend 收敛 contract 优于上层 fork — simple-stable + private-deploy paramount directive earayu2 msg=1224bec8 在 cross-adapter / cross-backend contract 设计时具体应用：当 backend 可以在 adapter 层收敛同一 contract 时优先 push backend adapter 收敛而不是让上层 FE / agent / MCP / API consumer 加 backend-aware branch）— task #69 P0-B PR #1930 + task #70 P1 候选 1 cross-PR 一次性收敛 first-application demo（cuiwenbo msg=cedc7703 实证三处 FE 调用点 raw 显示无 manual flip → backend 收敛后 FE 0 改动 + task #70 P1 候选 1 fully resolved）
- 2026-04-30 sediment 多源同源 catch trail (task #61 close 累计实证)：
  - **cross-PR 双独立 source 同源** 多次 demo（Lesson #12 v9 Bryce + Weston / Lesson #16 chenyexuan + 冬柏 / Lesson #17 cuiwenbo + Bryce / Lesson #13 v3 application demo 2 huangheng + Planetegg + Weston 三独立 source）— sediment 强度 high 验证 framework 跨 reviewer 独立 surface 同源 invariant 是 systemic 信号
  - **architect msg=03c892e0 + msg=daaeeab5** 总结性 sediment dispatch 显式列「Lesson #16 / v3.1 / v9 双 source / v7 extension / #17 simple-stable family / 3 deploy capability + 本 PR Lesson #14 cleanup demo + v7.3 cross-source default + 描述 NIT 反 fake guardrail demo」全 8+ 项 — 全 sediment 候选 cross-link 完整
