# 中文文档重写 — skeleton 与主笔方案（task #4）

> **Scope**: task #4 of `#文档重写`（总任务 #27 之后的下一轮派活）。
> **Deliverable type**: 分析 + skeleton + 主笔方案，**不落正文**（PM msg=d1f81953 硬边界）。
> **Baseline**: 初稿 baseline `origin/main @ 28a9f531`（Phase 6 / PR #1635 merged）+ `docs/modularization/architecture.md`（PR #1636 ready）。已按 Weston task #3 在 `522610ea` 上确认 zh-CN 仍为 38 篇 Markdown — 两 baseline 下文件清单一致。
> **Path**: 临时 `docs/zh-CN/rewrite-plan.md`；Weston task #3 最终 IA 落地后若需迁移到 canonical 位置再 git mv（PM msg=d1f81953）。

---

## 1. 本文作用 + 如何阅读

本文回答 3 个问题：

1. 当前 `docs/zh-CN/**` 38 份文档各自去哪（Section 3 新 IA mapping + Section 4 三桶 matrix）。
2. 每份 rewrite 文档由谁主笔（Section 5 主笔 lane assignment）。
3. 下一轮 rewrite 的执行顺序与收口方式（Section 6 cadence）。

**本文不做**：不落任何具体 doc 的正文草稿；不开 `git mv`；不在此 PR 中删任何文件（PM 锁 doc-only 边界 + 先产分析、后批量执行）。

---

## 2. 上游输入依赖

| 上游 task | Owner | Deliverable | 对本文影响 |
| --- | --- | --- | --- |
| task #2 | 燧木（PM） | 逐文件 keep/delete/rewrite matrix | 本文 Section 4 现用本地 triage 作初版；PM #2 output 出后合并校对 |
| task #3 | Weston | 新 IA + 保留/删除判据 | PM msg=234f7481 已锁定 IA 8 类；本文 Section 3 直接引用 |
| task #5 | cuiwenbo | Phase 3 SME source pack（KB / dual-hook / schema/common.py + indexing/retrieval/KG structural） | 填 architecture/indexing / architecture/retrieval / architecture/knowledge-graph / architecture/vector-db-abstraction / user-guide/content-import 等 |
| task #6 | Bryce | Phase 4 SME source pack（identity / governance / model_platform / marketplace） | 填 `architecture/identity-governance-model-platform-marketplace.md`（consolidated 4-domain 单篇）+ `user-guide/collection-marketplace.md` + `admin-guide/{quota-system,api-keys,audit-log}.md` + `integration/openai-compat.md` |
| task #7 batch 1 | chenyexuan | Phase 5/6 SME source pack 已出（msg=f4dfe81e） | 填 architecture/agent-runtime / architecture/conversation-chat-history / architecture/evaluation / architecture/prompt-templates |

**本文不等所有上游 task 齐全才起稿**（PM msg=d1f81953）。当前 batch 1 输入已足起 skeleton；task #5 / #6 / batch 2 到达后 append / revise。

---

## 3. 新信息架构（IA）— 8 类别（锁 per PM msg=234f7481 + msg=15ca5230）

| 目录 | 读者 | 内容性质 |
| --- | --- | --- |
| `getting-started/` | 首次接触 ApeRAG 的用户 | 快速上手、最小可运行 demo |
| `deployment/` | 部署 / 运维 | Docker / Helm / LLM provider 配置 |
| `user-guide/` | 终端用户 | 使用流程：collection 创建 → 导入 → 对话 → 评估 → 分享 |
| `admin-guide/` | 系统管理员 | quota / tag 权限 / API Key / 审计 |
| `integration/` | 接入方 | OpenAI-compat / MCP / Dify |
| `architecture/` | 架构维护者 | **仅 current-state**，不写历史方案；与 `docs/modularization/architecture.md` 协同 |
| `development/` | 贡献者 | 开发环境 / 提交规范 / 测试运行 |
| `reference/` | 全体 | API 测试手册 / 调试 cheatsheet / 配置示例 |

**原则**（PM msg=234f7481）：
- **不按旧树一比一重写** — 目标是收敛到**更少**的、按读者任务组织的文档。
- **`architecture/` 只写 current-state** — 不复述 Phase 0→6 迁移历史（那在 `docs/modularization/architecture.md` + `docs/modularization/breaking-changes/phase*.md` 里）。
- **允许多份旧 design doc 合并**为一篇新 current-state 文档（例：多份 graph/lightrag design 合并为单一 `architecture/knowledge-graph.md`）。
- **PR 说明 / TODO / milestone / 纯历史 design** 默认 delete。

**Cross-reference 惯例**（架构师 msg=aee91f5b）：新 REWRITE 文档中 `import path`、`Protocol <X>Ops`、`CRITICAL_WIRINGS entry`、G-gate 等后端架构 invariant 只以 hyperlink 指向 `docs/modularization/architecture.md` 对应 section，不在中文文档里重复定义。这样 architecture.md 是 single source of truth，未来 invariant 更新不需要同步改中文 docs。

**`architecture/` 本轮 skeleton 采用 consolidated approach**（PM msg=c11e57fb 锁定）— 不先拆成 12 个 domain 或十几篇细分 architecture docs；每篇内部按 domain / subsystem 分章节：

- `architecture/overview.md` — short entry + links（**不写**第二份长总述）
- `architecture/domains.md` — 12-domain 静态地图（link 到 architecture.md + breaking-changes）
- `architecture/indexing-retrieval-kg.md` — KB / indexing / retrieval / knowledge_graph / vector / LightRAG / vision 合并
- `architecture/conversation-agent-evaluation.md` — conversation / agent_runtime / evaluation / prompt architecture 合并
- `architecture/identity-governance-model-platform-marketplace.md` — Phase 4 四域合并
- `architecture/web-access.md` — Phase 2a web_access 独立

正文阶段若单篇过大再二次拆分。

---

## 4. 逐文件 Triage matrix（38 份）

### 4.1 design/ (31 份) — 目标位置已 consolidated per PM msg=c11e57fb

| 原文件 | LOC | 处置 | consolidated 目标 | 内部章节定位 |
| --- | --- | --- | --- | --- |
| architecture.md | 850 | **REWRITE → 合并** | `architecture/overview.md` + `architecture/domains.md` | 拆分后：overview.md = short entry + links；domains.md = 12-domain 静态地图 |
| indexing_architecture.md | 555 | **REWRITE → 合并** | `architecture/indexing-retrieval-kg.md` | 双链路 reconciler 章节 |
| authentication.md | 502 | **REWRITE → 合并** | `architecture/identity-governance-model-platform-marketplace.md` | identity 章节（User / Role / OAuth / fastapi-users） |
| agent_runtime_v3.md | 568 | **REWRITE → 合并** | `architecture/conversation-agent-evaluation.md` | agent_runtime 章节（PromptTemplateOps / G18 alt / Turn / SSE） |
| agent-backend.md | 528 | **DELETE** | — | chenyexuan msg=f4dfe81e triaged；pre-refactor, superseded |
| chat_history_design.md | 584 | **REWRITE → 合并** | `architecture/conversation-agent-evaluation.md` | conversation 章节（chat_router v1 / chat_service_global） |
| evaluation-design.md | 156 | **KEEP + MINOR → 合并** | `architecture/conversation-agent-evaluation.md` | evaluation 章节（is_terminal classmethod / dispatch_fn seam） |
| prompt_customization_design.md | 238 | **REWRITE → 合并** | `architecture/conversation-agent-evaluation.md` | prompt architecture 章节（PromptTemplateOps standalone-infra） |
| prompt_customization_api_test.md | 163 | **KEEP + MINOR** | `reference/prompt-api.md` | SK 改占位 |
| prompt_customization_integration_todo.md | 617 | **DELETE** | — | TODO 已完成，过时 |
| collection_marketplace_design.md | 1303 | **REWRITE → split** | `user-guide/collection-marketplace.md`（用户流程）+ `architecture/identity-governance-model-platform-marketplace.md`（架构 sections：marketplace 章节） | user-facing 分享/订阅 → user-guide；marketplace 域技术实现 → architecture consolidated |
| tag_based_permission_design.md | 477 | **DELETE** | — | Feature 未实装（per msg=8f67ce65 + PM msg=4a73b5a4 lock）；不保留文档避免描述与现实不符 |
| quota-system-design.md | 482 | **REWRITE → split** | `admin-guide/quota-system.md`（admin）+ `architecture/identity-governance-model-platform-marketplace.md`（架构 section：quota standalone-infra + QuotaOps Protocol） | — |
| collection_knowledge_export_design.md | 322 | **REWRITE** | `user-guide/knowledge-export.md` | user-facing export |
| url_and_text_import_design.md | 590 | **REWRITE** | `user-guide/content-import.md` | user-facing URL / 文本导入 |
| document_upload_design.md | 1077 | **REWRITE → split** | `user-guide/document-upload.md`（用户流程）+ `architecture/indexing-retrieval-kg.md`（document ingestion 章节） | 巨型文档；拆 user-facing + 架构 |
| search_flow_design.md | 654 | **REWRITE → 合并** | `architecture/indexing-retrieval-kg.md` | retrieval 章节（Flow Engine / DAG / MCP） |
| web-search-design.md | 739 | **REWRITE** | `architecture/web-access.md` | Phase 2a web_access 独立文档 |
| vision_index_creation.md | 261 | **REWRITE → 合并** | `architecture/indexing-retrieval-kg.md` | vision index 章节 |
| vector_db_abstraction.md | 435 | **REWRITE → 合并** | `architecture/indexing-retrieval-kg.md` | vector db abstraction 章节（M2/M3 landed current） |
| vector_db_abstraction_m2_pr.md | 145 | **DELETE** | — | PR-specific 实现说明，合并后冗余 |
| vector_db_abstraction_m3_pr.md | 279 | **DELETE** | — | 同上 |
| qdrant_memory_optimization.md | 633 | **DELETE** | — | 运营层历史优化记录，非产品文档 |
| connected_components_optimization.md | 280 | **DELETE** | — | 历史优化 log |
| graph_curation.md | 373 | **MERGE → 合并** | `architecture/indexing-retrieval-kg.md` | KG curation 章节 |
| graph_index_creation.md | 1084 | **MERGE → 合并** | `architecture/indexing-retrieval-kg.md` | KG index creation 章节 |
| graph_db_abstraction.md | 619 | **DELETE** | — | 设计与计划文档，M1-M3 已落地 |
| graph_normalization_merge_full_analysis.md | 1137 | **DELETE** | — | 审计分析，历史 snapshot |
| graphindex_rewrite.md | 509 | **DELETE** | — | 重构历史 log |
| lightrag_refactor.md | 581 | **DELETE** | — | 重构计划 |
| lightrag_entity_extraction_and_merging.md | 695 | **MERGE → 合并** | `architecture/indexing-retrieval-kg.md` | 实体提取 / merging 章节 |

### 4.2 非 design/ (7 份)

| 原文件 | LOC | 处置 | 新位置 |
| --- | --- | --- | --- |
| development/development-guide.md | 378 | **KEEP + UPDATE** | `development/development-guide.md` | 更新 import path 示例到 `aperag/domains/*`；加 G1-G19 快速索引 |
| integration/mcp-api.md | 333 | **KEEP + UPDATE** | `integration/mcp-api.md` | 校对 MCP endpoint 与 current 一致 |
| integration/dify.md | 168 | **KEEP + UPDATE** | `integration/dify.md` | 校对 Dify 集成步骤 |
| deployment/build-docker-image.md | 49 | **KEEP + UPDATE** | `deployment/build-docker-image.md` | 校对 build 命令；若需要可扩 |
| reference/evaluation-current-guide.md | 155 | **MOVE** | `user-guide/evaluation-guide.md` | 产品使用说明应归 user-guide 而非 reference |
| reference/HOW-TO-DEBUG.md | 50 | **KEEP** | `reference/how-to-debug.md` | |
| reference/how-to-configure-ollama.md | 66 | **MOVE** | `deployment/llm-providers-ollama.md` | Ollama 是 LLM provider 配置，更 fit deployment |

### 4.3 Summary

- **DELETE**：10 份（7 design + 0 non-design）— PR/TODO/milestone/历史 audit/refactor log/重复 PR 说明
- **REWRITE**：15 份（全 design）
- **MERGE**：4 份 design → 合并入 3 篇新 architecture doc
- **KEEP + MINOR/UPDATE**：6 份（3 design + 3 non-design）
- **MOVE (+ minor)**：3 份 non-design
- **合计**：38 份处理完

**新文档数预估**（PM msg=c11e57fb + Weston msg=0d2dd88c consolidated lock 口径）：
- `architecture/` **6 篇**（PM 锁定 consolidated targets）：`overview.md` + `domains.md` + `indexing-retrieval-kg.md` + `conversation-agent-evaluation.md` + `identity-governance-model-platform-marketplace.md` + `web-access.md`
- `user-guide/` ~6 篇（collection-marketplace / knowledge-export / content-import / document-upload / evaluation-guide / chat-interaction）
- `admin-guide/` 4 篇（quota-system / api-keys / audit-log / prompt-customization）— tag-permissions 已移除（feature 未实装，见 Section 4.1）
- `integration/` ~3 篇
- `deployment/` ~2-3 篇
- `development/` ~1-2 篇
- `reference/` ~2-3 篇
- `getting-started/` ~2 篇（**新增**，原无）

**总文档数从 38 → ~24-28**（PM msg=234f7481 "更小的文档集" 方向兑现）。

---

## 5. 主笔 lane mapping（consolidated target doc owner + 内部章节 SME）— per PM msg=c11e57fb

遵循 task #27 模式：目标 doc 主笔 = consolidated doc 的单一 owner；内部章节 SME = thread markdown block 供稿；huangheng（task #4 作者）整合 + 全量 cross-consistency CR。

### 5.1 `architecture/` consolidated targets (6 docs)

| 目标 doc | 主笔 | 内部章节 SME 绑定 | 上游 task |
| --- | --- | --- | --- |
| `architecture/overview.md` | huangheng | 自写（short entry + links to architecture.md / domains.md / 下列 consolidated docs） | task #4 |
| `architecture/domains.md` | huangheng | 各 SME lanes 补充 domain 一行简介 | task #4 + 各 SME |
| `architecture/indexing-retrieval-kg.md` | **Bryce**（owner per PM msg=48ed233e item 3） | cuiwenbo 为 KB / `schema/common.py` / dual-hook 章节 contributor + consistency reviewer（cross-ref 回 `docs/modularization/architecture.md` Section 2.3 / 3.3，不新开独立 zh-CN doc per msg=9b712260）；Bryce 写 indexing / retrieval / KG / vector / vision / graphindex / LightRAG 深度 sections | task #6 primary + task #5 supporting |
| `architecture/conversation-agent-evaluation.md` | chenyexuan 主笔 | conversation 6-services topology / agent_runtime PromptTemplateOps seam / evaluation is_terminal + dispatch_fn seam / prompt architecture standalone-infra permanent — 全在 task #7 batch 1 (msg=f4dfe81e) + batch 2 skeleton 覆盖 | task #7 |
| `architecture/identity-governance-model-platform-marketplace.md` | Bryce 主笔 | identity (User/Role/OAuth/fastapi-users) / governance (ApiKey/Audit/QuotaOps standalone-infra) / marketplace (CollectionMarketplace / Q2 rename) / model_platform (LLM provider + 2-router split) — task #6 Block D 模式 | task #6 |
| `architecture/web-access.md` | **Bryce**（PM msg=4689919d lock） | Phase 2a web_access 独立文档（reader / search / utils sub-packages） | — |

### 5.2 `user-guide/` (6 docs)

| 目标 doc | 主笔 | SME | 上游 task |
| --- | --- | --- | --- |
| `user-guide/collection-marketplace.md` | Bryce | Phase 4 marketplace | task #6 |
| `user-guide/knowledge-export.md` | cuiwenbo | Phase 3 | task #5 |
| `user-guide/content-import.md` | cuiwenbo | Phase 3 ingestion | task #5 |
| `user-guide/document-upload.md` | cuiwenbo | Phase 3 doc_parser | task #5 |
| `user-guide/evaluation-guide.md` | chenyexuan | Phase 5/6 evaluation | task #7 |
| `user-guide/chat-interaction.md` | chenyexuan | Phase 5 conversation | task #7 |

### 5.3 `admin-guide/` (4 docs — tag-permissions 已从 plan 移除因 feature 未实装)

| 目标 doc | 主笔 | SME | 上游 task |
| --- | --- | --- | --- |
| `admin-guide/quota-system.md` | Bryce | Phase 4 governance + quota_service standalone-infra | task #6 |
| `admin-guide/api-keys.md` | Bryce | Phase 4 governance | task #6 |
| `admin-guide/audit-log.md` | Bryce | Phase 4 governance | task #6 |
| `admin-guide/prompt-customization.md` | chenyexuan | prompt_customization admin/管理面（template 管理、三层优先级配置）— per PM msg=6f2fef3d scope split 到 task #7 owner | task #7 |

### 5.4 `integration/` (3 docs)

| 目标 doc | 主笔 | SME | 上游 task |
| --- | --- | --- | --- |
| `integration/mcp.md` | **待 PM 派活**（可 ziang / weihong） | — | KEEP |
| `integration/dify.md` | **待 PM 派活**（可 weihong） | — | KEEP |
| `integration/openai-compat.md` | Bryce | Phase 4 model_platform llm_routes | task #6 |

### 5.5 其他目录

| 目标 doc | 主笔 | SME | 上游 task |
| --- | --- | --- | --- |
| `getting-started/overview.md` | **待 PM 派活** | — | 新增 |
| `getting-started/quickstart.md` | **待 PM 派活** | — | 新增 |
| `deployment/build-docker-image.md` | PM / Weston | — | KEEP |
| `deployment/llm-providers-ollama.md` | PM / Weston | — | MOVE |
| `development/development-guide.md` | huangheng | SME 提 sample code | task #4 |
| `reference/prompt-api.md` | chenyexuan | — | task #7 |
| `reference/how-to-debug.md` | PM / Weston | — | KEEP |

**未决 assignment**（open items，PR 不 block on 它们 per PM msg=1cf82997 tail）：
- `getting-started/overview.md` + `quickstart.md`（2 篇新增）— PM 需派活
- `integration/mcp.md` + `integration/dify.md`（2 篇 KEEP+update）— PM 需派活

---

## 6. 下一轮 rewrite cadence（本轮不做）

task #4 只出 skeleton + 主笔方案，不落正文。**下一轮**（待 PM 派活后）的 rewrite 执行顺序建议：

### Phase A（task #4 merge 之后）：lane 并行起 body draft

1. **SME lanes 并行起稿各自主笔的 doc**，每份 doc 单独 draft PR（doc-only）。
2. 每份 doc 采用 task #27 skeleton-first pattern：先起 section heading + TODO placeholder + 关键 canonical 引用（link 到 architecture.md），再填内容。
3. 每份 doc 单独 CR（architect canonical consistency check + huangheng cross-section consistency）。
4. 新 IA 下原不存在的 `getting-started/`、`integration/openai-compat` 等全新 doc 由对应 lane 新造。

### Phase B：delete batch + MOVE batch

- 一个 batch PR `git rm` 所有 DELETE 类 files + `git mv` 所有 MOVE 类 files。机械改动，任何 lane 可接。CR 简单过。

### Phase C：Final integration + architecture/overview.md 主入口

- 所有 SME lane doc ready 后，huangheng 主笔 `architecture/overview.md` 作整体索引 + cross-reference map，link 到各 SME 主笔的 doc。
- 全量 cross-doc consistency check（符炫炜 audit）。

### Phase D：旧目录清理

- `docs/en-US/**` 19 files + `web/docs/**` 16 files 整批 `git rm`（PM 已定 scope，简单机械操作）。

---

## 7. 开放问题（等 PM / earayu2 派活）

本节只保留尚未派活项；PM 已锁定事项（`architecture/overview.md` short entry + links / 不开 `security/` / `operations/` 顶级 / `docs/modularization/**` 不改）已从 open questions 移除（详见正文对应 section）。

1. **`getting-started/overview.md` + `quickstart.md` 主笔**（2 篇新增 doc，无对应旧 source）— PM 需派活
2. **`integration/mcp.md` + `integration/dify.md` 主笔**（2 篇 KEEP+update）— PM 需派活

---

## 8. 下一步

1. **本 PR doc-only**：仅含本 `docs/zh-CN/rewrite-plan.md` 单文件新增；无其他代码 / 文档改动。
2. **SME lanes 在本 PR thread 内供稿**（markdown block）— 我 append 到 Section 4 matrix 或 Section 5 主笔 mapping 里。
3. **PM / Weston task #2+#3 output 出后** merge 到本文的 Section 3 / 4 精化。
4. **最终 task #4 deliverable** = 本文 ready-for-review 版本 → PM merge gate → task #27 同模式 self-merge。

---

*本文 baseline: `origin/main @ 28a9f531`（初稿）+ Weston task #3 在 `522610ea` 复核。task #4 owner: @huangheng. 上游依赖: task #2 (PM) + task #3 (Weston) + task #5/6/7 SME packs. 本 PR 产出为 skeleton + mapping，不落正文。*
