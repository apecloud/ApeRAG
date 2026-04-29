---
title: 架构总览
description: ApeRAG 后端架构入口 — 从哪开始读、去哪找什么
---

# 架构总览

> ApeRAG 后端经过 Phase 0→6 模块化重构后，代码以 **12 个业务 domain** 的形式组织在 `aperag/domains/**`，跨 domain 调用通过 **boundary gates（G1–G19）** 与少量 **permanent DI seams** 约束。本文是整个 `architecture/` 目录的入口，不重复实现细节，只负责导航。

---

## 一句话定位

- **代码按业务职责分 12 个 domain**：`identity` · `governance` · `model_platform` · `marketplace` · `knowledge_base` · `indexing` · `retrieval` · `knowledge_graph` · `conversation` · `agent_runtime` · `evaluation` · `web_access`。
- **每个 domain 自成一格**：自己的 DB 模型 · 自己的 Pydantic schema · 自己的 service · 自己的 FastAPI 路由，都在 `aperag/domains/<domain>/` 下。
- **跨 domain 依赖有明确规则**：provider 已搬进 `aperag/domains/` 的直接 import；没搬进的通过 consumer 自持 `Protocol` + module-level DI slot + 共享 bootstrap helper（`aperag/bootstrap/__init__.py`）注入。
- **有测试守住边界**：`tests/unit_test/test_modularization_boundaries.py` 实现 G1–G19 共 20 条边界测试，跟 `make lint` / `make test-unit` 一起跑在 CI。
- **HTTP API 全程字节稳定**：`scripts/export_openapi.py --check` 在整个重构过程每次 squash merge 都通过。
- **运行时分两个独立进程部署**（task #17 hard cut，2026-04-29 ship）：`aperag-api` 只跑 FastAPI HTTP 入口 + 轻量入队；`aperag-indexing-worker` 跑 parse / vector / fulltext / graph_facts / graph_vectors / summary / vision / reconciler / cleanup 等 10 条 lane。两进程通过 single-source-of-truth `wire_cross_domain_di_seams()` helper 保证 DI parity，受 `tests/boundaries/test_worker_di_parity.py` AST 等价性 gate 守护。

详细的 phase 产出、domain 清单、gate 列表、permanent seam 定义见 [`docs/modularization/architecture.md`](../../modularization/architecture.md)（英文，canonical source-of-truth，本目录所有文档都以它为基准）。

---

## 我该从哪里开始读？

下面按「你关心什么」给一个导航表：

| 你想了解 | 去这里 |
| --- | --- |
| 12 个 domain 各自做什么？domain 之间什么关系？ | [`architecture/domains.md`](./domains.md) |
| identity / governance / model_platform / marketplace 这 4 个 domain 的内部结构 | [`architecture/identity-governance-model-platform-marketplace.md`](./identity-governance-model-platform-marketplace.md) |
| knowledge_base / indexing / retrieval / knowledge_graph 的内部结构与 ingestion / retrieval pipeline | [`architecture/indexing-retrieval-kg.md`](./indexing-retrieval-kg.md) |
| conversation / agent_runtime / evaluation 与 prompt 架构 | [`architecture/conversation-agent-evaluation.md`](./conversation-agent-evaluation.md) |
| 爬虫抓取 / URL 阅读相关的 `web_access` 子包 | [`architecture/web-access.md`](./web-access.md) |
| 异步索引任务系统 hard cut 的部署边界与长期不变式 | [`architecture/task-system-invariants.md`](./task-system-invariants.md) |
| task #17 API/Worker hard cut 执行方案、部署 runbook、状态机验收与 CR checklist | [`task-system-hard-cut-v8.md`](./task-system-hard-cut-v8.md)、[`task-17-deployment-release-runbook.md`](./task-17-deployment-release-runbook.md)、[`task-17-state-machine-validation.md`](./task-17-state-machine-validation.md)、[`task-17-cr-review-checklist.md`](./task-17-cr-review-checklist.md) |
| 英文的 canonical 全景（20 个 boundary 测试、permanent seam、shim lifecycle、future 候选） | [`docs/modularization/architecture.md`](../../modularization/architecture.md) |
| 我想在本地把 ApeRAG 跑起来 / 贡献代码 | [`development/development-guide.md`](../development/development-guide.md) |
| 我想部署 ApeRAG / 配置 LLM provider | `deployment/` 目录 |
| 我是终端用户，想知道怎么创建 collection / 对话 / 评估 | `user-guide/` 目录 |
| 我是系统管理员，想配额 / 审计 / API Key / prompt 定制 | `admin-guide/` 目录 |
| 我是第三方接入方（OpenAI 兼容 / MCP / Dify） | `integration/` 目录 |
| API 测试手册、调试指令、示例配置 | `reference/` 目录 |

---

## 当前架构为什么这样组织

四条约束决定了现在的布局：

- **业务职责清晰可定位**：把代码按业务 domain 聚类（`aperag/domains/<d>/`），而不是按层（`service/` / `models/` / `views/`）。新人看到一个 endpoint 能在一个目录里找齐 DB / schema / 业务 / 路由。
- **跨 domain 依赖显式且可测**：`aperag/domains/**` 内禁止 import 旧聚合层（G1）；跨 domain 访问只有两种形态——provider-in-domain 直接 import、或 consumer-owned Protocol + DI 槽。`tests/unit_test/test_modularization_boundaries.py` 里 20 条 pytest 锁住全部边界规则（G1–G19 + CRITICAL_WIRINGS），跟 `make test-unit` 一起跑，违反 import 会红在 CI。
- **永久性的跨切面基础设施有独立接入点**：跨 domain 的基础设施（配额、prompt template、marketplace、search pipeline、identity init、prompt CRUD 等共 10 条 DI seam）不塞进某个 domain，也不偷偷下沉到共享层，而是用 **permanent DI seam** 显式注入。两进程（API + indexing-worker）共享 `aperag/bootstrap/__init__.py` 里的 `wire_cross_domain_di_seams()` 单一调用契约，受 boundary 测试 AST 等价性 gate（`tests/boundaries/test_worker_di_parity.py`）守护，避免「source 进程加了新 seam，target 进程漏 wire」类 cross-process drift。
- **运行时执行面与请求入口物理分离**：FastAPI HTTP 入口与索引/清理 worker 不能共进程（task #17 hard cut 强制约束）。重型 LLM 调用占用 event loop 会让 `/health` 超时被 kubelet 重启 → 503 风暴；连接池、并发预算、健康探针的语义按进程角色（API vs worker）分别定义。详细 invariant lock 见 [`task-system-invariants.md`](./task-system-invariants.md)（6 hard gate / 6 YAGNI 边界 / 4 escape hatch 触发条件 / cross-process DI parity）。

外部契约零漂移是硬要求：HTTP API（OpenAPI）、前端 typed client、数据库迁移历史在整个重构过程保持字节级稳定。

历史过程（分 phase 的变更、每一步为什么这样拆、lesson-learned）都在 [`docs/modularization/`](../../modularization/) 目录：

- `docs/modularization/architecture.md` — 当前稳态的 canonical 文档（英文）。
- `docs/modularization/breaking-changes/phase*.md` — 每个 phase 的变更说明，改老代码前可读。
- `docs/modularization/roadmap.md` / `target-domain-map.md` / `README.md` — 重构启动前的设计稿，仅历史参考。

---

## 与其他文档的分工

- [`docs/modularization/architecture.md`](../../modularization/architecture.md) — **英文 canonical SSoT**，描述「当前稳态」。所有结构性不变式（12 domain 清单 / G1-G19 定义 / dual-hook 模式 / permanent seam 清单 / legacy shim 清单 / 新代码落位规则）只在这里有一份，本目录的中文文档通过 cross-reference 引用，不复制粘贴，避免随时间漂移。
- 本目录（`docs/zh-CN/architecture/`） — 用中文讲「每个 domain 内部怎么回事」，粒度比 canonical SSoT 细，给需要读具体实现的中文读者。

## 读者原则

- 需要搞清楚「当前主干到底是什么结构」—— 先读 canonical SSoT，再回到本目录找 domain 细节。
- 需要「在某个 domain 上加新代码」—— 读 `domains.md` 定位 domain，再读该 domain 所在 consolidated doc 的章节，最后回 `development/development-guide.md` 找具体命令与门禁要求。
- 需要「对外接入 / 部署 / 操作 / 使用」—— 本目录以外的 `integration/` · `deployment/` · `admin-guide/` · `user-guide/` 才是正确位置，本目录不承载这些内容。

---

*本文档基线：`origin/main @ f13eb80`（task #17 hard cut + Phase 2 sediment 全 merged，含 PR #1884 / #1893 / #1885-#1891）；canonical SSoT 基线 `origin/main @ 28a9f531`（PR #1635 Phase 6 merged）。本目录其他文档随 `docs/modularization/architecture.md` 更新而维护；结构性不变式若发生变更，先改 canonical SSoT，再回头更新本目录。task #17 后新增的 cross-process DI parity / 部署进程拆分等不变式，落在 `task-system-invariants.md` 与 `task-17-cr-review-checklist.md`。*
