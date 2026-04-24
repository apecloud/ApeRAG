# 中文文档重写 — skeleton 与主笔方案（task #4）

> **Scope**: task #4 of `#文档重写`（总任务 #27 之后的下一轮派活）。
> **Deliverable type**: 分析 + skeleton + 主笔方案，**不落正文**（PM msg=d1f81953 硬边界）。
> **Baseline**: `origin/main @ 28a9f531`（Phase 6 / PR #1635 merged）+ `docs/modularization/architecture.md`（PR #1636 ready）。
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
| task #6 | Bryce | Phase 4 SME source pack（identity / governance / model_platform / marketplace） | 填 architecture/identity-auth / architecture/model-platform / user-guide/collection-marketplace / admin-guide/quota-system / admin-guide/tag-permissions 等 |
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

---

## 4. 逐文件 Triage matrix（38 份）

### 4.1 design/ (31 份)

| 原文件 | LOC | 处置 | 新位置 | 依据 |
| --- | --- | --- | --- | --- |
| architecture.md | 850 | **REWRITE → 合并** | `architecture/README.md` + 2-3 domain sub-pages | 顶层架构主入口；current-state 收窄 + 指向 `docs/modularization/architecture.md` 作后端 canonical source |
| indexing_architecture.md | 555 | **REWRITE** | `architecture/indexing.md` | 双链路 / reconciler / 状态驱动；current-state 仍对 |
| authentication.md | 502 | **REWRITE** | `architecture/identity-auth.md` | Phase 4 identity domain；更新 canonical home + `AuthenticatedUser(Protocol)` + OAuth 仍在 `aperag/views/auth.py` legacy special-case |
| agent_runtime_v3.md | 568 | **REWRITE** | `architecture/agent-runtime.md` | chenyexuan msg=f4dfe81e triaged；更新 import path + PromptTemplateOps seam + G18 alt registry |
| agent-backend.md | 528 | **DELETE** | — | chenyexuan msg=f4dfe81e triaged；superseded by agent_runtime_v3.md；零 `aperag.domains` reference = pre-refactor |
| chat_history_design.md | 584 | **REWRITE** | `architecture/conversation-chat-history.md` | chenyexuan msg=f4dfe81e triaged；endpoint 仍在；更新 layer path 到 `aperag/domains/conversation/*` |
| evaluation-design.md | 156 | **KEEP + MINOR** | `architecture/evaluation.md` | chenyexuan msg=f4dfe81e triaged；current 主干保留；加 canonical home + `is_terminal()` classmethod update |
| prompt_customization_design.md | 238 | **REWRITE** | `architecture/prompt-templates.md` | chenyexuan msg=f4dfe81e triaged；加 PromptTemplateOps Protocol + standalone-infra permanent seam 段 |
| prompt_customization_api_test.md | 163 | **KEEP + MINOR** | `reference/prompt-api.md` | chenyexuan msg=f4dfe81e triaged；API test 手册仍可用；SK 改占位 |
| prompt_customization_integration_todo.md | 617 | **DELETE** | — | chenyexuan msg=f4dfe81e triaged；TODO 已完成（Phase 5 5-S5b PromptTemplateOps 兑现） |
| collection_marketplace_design.md | 1303 | **REWRITE → split** | `user-guide/collection-marketplace.md`（用户面）+ architecture 引用 | Phase 4 marketplace domain；核心是 user-facing 分享/订阅流程，剥出 user-guide；技术实现压缩到 architecture 超链 |
| tag_based_permission_design.md | 477 | **REWRITE** | `admin-guide/tag-permissions.md` | admin-facing batch-authorization 功能 |
| quota-system-design.md | 482 | **REWRITE** | `admin-guide/quota-system.md` | admin-facing quota；涉及 `QuotaOps` Protocol + DI permanent seam，架构部分 cross-ref architecture.md |
| collection_knowledge_export_design.md | 322 | **REWRITE** | `user-guide/knowledge-export.md` | user-facing 导出功能，MVP 已实现 |
| url_and_text_import_design.md | 590 | **REWRITE** | `user-guide/content-import.md` | user-facing URL / 文本导入 |
| document_upload_design.md | 1077 | **REWRITE → split** | `user-guide/document-upload.md`（用户流程）+ `architecture/document-ingestion.md`（架构） | 巨型文档；拆 user-facing 操作 + 架构视角 |
| search_flow_design.md | 654 | **REWRITE** | `architecture/retrieval-search.md` | Phase 3 retrieval domain + Flow Engine + DAG + MCP |
| web-search-design.md | 739 | **REWRITE** | `architecture/web-access.md` | Phase 2a web_access domain |
| vision_index_creation.md | 261 | **REWRITE** | `architecture/vision-index.md` | 若 feature 仍在 current，保留；cuiwenbo task #5 SME 侧验证 |
| vector_db_abstraction.md | 435 | **REWRITE** | `architecture/vector-db-abstraction.md` | doc 自称 "M2+M3 已落地"；current-state 仍对；cuiwenbo SME 修订 |
| vector_db_abstraction_m2_pr.md | 145 | **DELETE** | — | PR-specific 实现说明；合并到 architecture/vector-db-abstraction.md 主文件后冗余 |
| vector_db_abstraction_m3_pr.md | 279 | **DELETE** | — | 同上 |
| qdrant_memory_optimization.md | 633 | **DELETE** | — | 运营层历史优化记录；不属于产品文档，可保存在 `docs/modularization/` 或 ops notes |
| connected_components_optimization.md | 280 | **DELETE** | — | 历史优化 log；实现细节已在代码 + breaking-changes docs |
| graph_curation.md | 373 | **MERGE → 合并** | `architecture/knowledge-graph.md` 一 section | Phase 3 KG domain 子模块；合并单篇 |
| graph_index_creation.md | 1084 | **MERGE → 合并** | `architecture/knowledge-graph.md` 一 section | 同上；巨型文档压缩 |
| graph_db_abstraction.md | 619 | **DELETE** | — | 设计与计划文档，`lightrag_refactor.md` 已指路；M1-M3 已落地进入实现，非 current-state user / admin / architect 需要 |
| graph_normalization_merge_full_analysis.md | 1137 | **DELETE** | — | 审计分析文档；历史 snapshot；不属于 current-state 面 |
| graphindex_rewrite.md | 509 | **DELETE** | — | 重构历史 log；已落地 |
| lightrag_refactor.md | 581 | **DELETE** | — | 重构计划文档；已落地 |
| lightrag_entity_extraction_and_merging.md | 695 | **MERGE → 合并** | `architecture/knowledge-graph.md` 一 section 或独立 `architecture/entity-extraction.md` | 技术细节值得保留；由 cuiwenbo / Bryce SME 决定合并 vs 独立 |

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

**新文档数预估**：
- `architecture/` ~8-10 篇（README + indexing / retrieval / knowledge-graph / vector-db / agent-runtime / conversation-chat / evaluation / prompt-templates / identity-auth / vision / web-access / document-ingestion / model-platform 等）
- `user-guide/` ~4-6 篇（collection-marketplace / knowledge-export / content-import / document-upload / evaluation-guide / chat-interaction 等）
- `admin-guide/` ~3-4 篇（quota-system / tag-permissions / api-keys / audit-log）
- `integration/` ~3 篇
- `deployment/` ~2-3 篇
- `development/` ~1-2 篇
- `reference/` ~2-3 篇
- `getting-started/` ~2 篇（**新增**，原无）

**总文档数从 38 → ~25-30**（PM msg=234f7481 "更小的文档集" 方向兑现）。

---

## 5. 主笔 lane mapping

遵循 task #27 模式：主笔 = lane owner；SME source pack 供稿 via thread markdown block；huangheng（主笔 task #4 skeleton 作者）整合 + 交叉 CR。

| 新文档 | 主笔 | SME 供稿 | 上游 task |
| --- | --- | --- | --- |
| `getting-started/quickstart.md` | **新增：待 PM 派活** | — | 无对应旧文档 |
| `getting-started/install-guide.md` | **新增：待 PM 派活** | — | 无对应旧文档 |
| `deployment/build-docker-image.md` | PM/Weston | — | KEEP |
| `deployment/llm-providers-ollama.md` | PM/Weston | — | MOVE from reference |
| `user-guide/document-upload.md` | cuiwenbo（Phase 3 doc_parser adjacent） | Bryce if needed | task #5 |
| `user-guide/content-import.md` | cuiwenbo | — | task #5 |
| `user-guide/knowledge-export.md` | cuiwenbo | — | task #5 |
| `user-guide/collection-marketplace.md` | Bryce（Phase 4 marketplace） | — | task #6 |
| `user-guide/evaluation-guide.md` | chenyexuan（Phase 5/6 evaluation） | — | task #7 |
| `user-guide/chat-interaction.md` | chenyexuan | — | task #7 |
| `admin-guide/quota-system.md` | Bryce（Phase 4 governance + quota_service standalone-infra） | — | task #6 |
| `admin-guide/tag-permissions.md` | Bryce | — | task #6 |
| `admin-guide/api-keys.md` | Bryce | — | task #6 |
| `admin-guide/audit-log.md` | Bryce | — | task #6 |
| `integration/mcp-api.md` | ??（可 ziang / weihong）| — | KEEP |
| `integration/dify.md` | ??（可 weihong） | — | KEEP |
| `integration/openai-compat.md` | Bryce（Phase 4 model_platform llm_routes） | — | task #6 |
| `architecture/README.md` | huangheng（主笔 + 整合） | all SMEs + 符炫炜 audit | task #4 + #2-#7 所有 |
| `architecture/indexing.md` | cuiwenbo + Bryce 结构性补深 | — | task #5 |
| `architecture/retrieval-search.md` | cuiwenbo 结构性 + Bryce 深度 | — | task #5 |
| `architecture/knowledge-graph.md` | cuiwenbo 结构性 + Bryce 深度（KG/graphindex） | — | task #5 |
| `architecture/vector-db-abstraction.md` | cuiwenbo | — | task #5 |
| `architecture/identity-auth.md` | Bryce | — | task #6 |
| `architecture/model-platform.md` | Bryce | — | task #6 |
| `architecture/agent-runtime.md` | chenyexuan | — | task #7 |
| `architecture/conversation-chat-history.md` | chenyexuan | — | task #7 |
| `architecture/evaluation.md` | chenyexuan | — | task #7 |
| `architecture/prompt-templates.md` | chenyexuan | — | task #7 |
| `architecture/vision-index.md` | cuiwenbo / Bryce | — | task #5 |
| `architecture/web-access.md` | Bryce（Phase 2a web_access owner per `docs/modularization/breaking-changes/phase2-web_access.md`） | — | Phase 2a 相关 |
| `architecture/document-ingestion.md` | cuiwenbo | Bryce if needed | task #5 |
| `development/development-guide.md` | huangheng（主笔 task #4）| SME 提 sample code | task #4 |
| `reference/prompt-api.md` | chenyexuan | — | task #7 |
| `reference/how-to-debug.md` | PM/Weston | — | KEEP |

**未决 assignment**：
- `getting-started/*.md`（2 篇 新增）— PM 需派活；可能 dongdong（FE view）或 Weston
- `integration/mcp-api.md` / `integration/dify.md`（2 篇 KEEP+update）— PM 需派活；ziang 对 index/search/celery 熟 fit mcp-api；weihong fit general

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

### Phase C：Final integration + architecture/README.md 主入口

- 所有 SME lane doc ready 后，huangheng 主笔 `architecture/README.md` 作整体索引 + cross-reference map，link 到各 SME 主笔的 doc。
- 全量 cross-doc consistency check（符炫炜 audit）。

### Phase D：旧目录清理

- `docs/en-US/**` 19 files + `web/docs/**` 16 files 整批 `git rm`（PM 已定 scope，简单机械操作）。

---

## 7. 开放问题（等 PM / earayu2 裁决）

1. **`getting-started/` 2 篇 新增 doc**：无对应旧文档，内容需要从 `README.md` 或实际用户流程提炼；PM 需决定由谁起草（dongdong FE / Weston / 其他）。
2. **`architecture/README.md` 作为 zh-CN 顶层架构主入口 vs 指向 `docs/modularization/architecture.md`**：前者面向中文读者（user/admin/dev 都可读），后者 canonical 但是 modularization 专题。建议 zh-CN 主入口简短 + link 出去 + 加中文叙事 wrapper。
3. **`integration/mcp-api.md` 主笔未定** — 建议 PM 派给 ziang（其 Agent / MCP 熟悉度）或 weihong。
4. **是否开一个 `security/` 或 `operations/` 顶级目录**？Weston task #3 的 8 类未覆盖 SRE / ops audit / security audit 场景。若这是 zh-CN GitHub doc scope 外（由内部 notes 承担），无需处理。
5. **`docs/modularization/**` 重写后去向**：PM msg=5286eb6a 已锁不改；本 plan 保持这个约束。

---

## 8. 下一步

1. **本 PR doc-only**：仅含本 `docs/zh-CN/rewrite-plan.md` 单文件新增；无其他代码 / 文档改动。
2. **SME lanes 在本 PR thread 内供稿**（markdown block）— 我 append 到 Section 4 matrix 或 Section 5 主笔 mapping 里。
3. **PM / Weston task #2+#3 output 出后** merge 到本文的 Section 3 / 4 精化。
4. **最终 task #4 deliverable** = 本文 ready-for-review 版本 → PM merge gate → task #27 同模式 self-merge。

---

*本文 baseline: `origin/main @ 28a9f531`. task #4 owner: @huangheng. 上游依赖: task #2 (PM) + task #3 (Weston) + task #5/6/7 SME packs. 本 PR 产出为 skeleton + mapping，不落正文。*
