---
title: 12 Domain 通览
description: ApeRAG 后端 12 个业务 domain 的职责、目录契约、跨 domain 关系一览
---

# 12 Domain 通览

> 本文回答「后端每个 domain 负责什么、目录里有哪些文件、domain 之间怎么互相调」。结构性不变式（G1–G19、dual-hook、permanent seam、shim 清单）在 [`docs/modularization/architecture.md`](../../modularization/architecture.md)（canonical SSoT）里已经写死，本文不复述，只给中文读者一个落地的导读。

---

## 1. Domain 是什么

`aperag/domains/<domain>/` 是一个业务 domain 的完整目录，默认包含这些槽位（每个 domain 不一定全用）：

| 文件 / 目录 | 作用 |
| --- | --- |
| `db/models.py` | SQLAlchemy ORM 类 + 由本 domain 拥有的 Enum |
| `schemas.py` | 本 domain 拥有的 Pydantic schema；通过 **dual-hook** 绑到 `aperag.schema.view_models`（见 SSoT Section 3.3） |
| `ports.py` | **consumer-owned** `Protocol` — 本 domain 声明「我对别人的依赖长什么样」，provider 结构性满足，provider 永远不 import 本 domain 的 `ports.py` |
| `service/` 或 `service.py` | 业务逻辑。只能直接 import 本 domain 自己的 `db/` / `schemas.py`，以及其他 domain 已经搬到 `aperag/domains/` 的部分 |
| `api/routes.py`（或 `<feature>_routes.py`） | FastAPI 路由 module；router 变量名由 domain 决定；`aperag/views/*.py` 里的同名文件是 re-export shim |

不是每个 domain 都塞满所有槽位 —— 例如 `web_access` 没有实体（DB 层为空），`indexing` 没有 API 路由。

**Domain 之间如何互相访问** 在 SSoT Section 3.1 写死了两种形态：

- **直接 import** — 当 provider 已经搬进 `aperag/domains/`（例：`from aperag.domains.knowledge_base.service.document_service import document_service`）。
- **consumer-owned `Protocol` + DI 槽** — 当 provider 仍然在旧位置（`aperag/service/*.py`）。consumer 在自己的 `ports.py` 声明 `Protocol`，留一个 `_ops: Optional[XOps]` 槽 + `set_x_ops()` setter + `_get_x_ops()` accessor；`aperag/app.py` 在启动时把旧 provider（或一个 adapter）塞进槽里。

> G1 禁止 `aperag/domains/**` 任何文件 import `aperag.service.*` / `aperag.schema.view_models` / `aperag.db.models`。sibling 同域 import 总是直接 import。

---

## 2. 12 Domain 速查表

| Domain | DB 实体 / Enum | Service 模块 | 自持 Port（对外依赖声明） | 主 API 路由 | 细节文档 |
| --- | --- | --- | --- | --- | --- |
| **identity** | `Role` · `User` · `OAuthAccount` | `user_manager`、`identity_user_ops` | `AuthenticatedUser` · `BotInitOps` · `ChatInitOps` · `QuotaInitOps` | 走 `aperag/views/auth.py`（fastapi-users + OAuth） | `architecture/identity-governance-model-platform-marketplace.md`（起稿中） |
| **governance** | `ApiKey` · `AuditLog`（+ `ApiKeyStatus` · `AuditResource`） | `api_key_service`、`audit_service` | `AuthenticatedUser` · `UserView` | `api/routes.py`（合并 api-key + audit-log） | 同上 |
| **model_platform** | `LLMProvider` · `LLMProviderModel`（+ `APIType`） | `default_model_service`、`llm_available_model_service`、`llm_provider_service` | `AuthenticatedUser` | `api/llm_routes.py` + `api/providers_v2_routes.py`（2-router split，见下文） | 同上 |
| **marketplace** | `CollectionMarketplace` · `UserCollectionSubscription`（+ `CollectionMarketplaceStatusEnum`） | `marketplace_service`、`marketplace_collection_service` | `AuthenticatedUser` | `api/routes.py` | 同上 |
| **knowledge_base** | `Collection` · `CollectionSummary` · `Document`（4 个 Enum） | `collection_service`、`collection_summary_service`、`document_service` | `AuthenticatedUser` · `MarketplaceOps` · `MarketplaceCollectionOps` · `SearchPipelineOps` · `QuotaOps` | `api/routes.py` | `architecture/indexing-retrieval-kg.md`（起稿中） |
| **indexing** | `DocumentIndex`（+ `DocumentIndexType` · `DocumentIndexStatus`） | `manager`、`document_parser`、`vector_index`、`fulltext_index`、`graph_index`、`summary_index`、`vision_index` 等功能模块 | `CollectionIndexingView` · `IndexingTrigger` | — | 同上 |
| **retrieval** | `SearchHistory` | `service.py` + `pipeline.py` | `GraphQueryContext` · `GraphSearchContract` | `api/routes.py` | 同上 |
| **knowledge_graph** | `GraphCurationRun` · `GraphCurationSuggestion`（+ 2 个 Enum） | `service.py` + `graphindex/` 11 个子模块 | `CollectionRow` | `api/routes.py`（加 `aperag/views/graph.py` 上的 410-Gone shim） | 同上 |
| **conversation** | `Bot` · `Chat` · `TurnFeedback`（+ 5 个 Enum） | `bot_service`、`chat_service`、`chat_collection_service`、`chat_document_service`、`chat_title_service`、`turn_feedback_service` | `KnowledgeBaseCollectionView` · `AuthenticatedUser` · `QuotaOps` | `api/routes.py`（内部按前缀拆 `chat_router` + `bots_router`） | `architecture/conversation-agent-evaluation.md`（起稿中） |
| **agent_runtime** | `AgentTurn` · `AgentTimelineEvent` · `AgentArtifact`（+ 3 个 Enum） | `runtime`、`services`、`storage` | `AuthenticatedUser` · `PromptTemplateOps` | `api/routes.py` | 同上 |
| **evaluation** | `EvaluationDataset` · `EvaluationDatasetItem` · `EvaluationRun` · `EvaluationRunItem` · `EvaluationRunItemAttempt`（+ 5 个 Enum） | `services`、`worker`（`dispatch_fn` 测试注入 seam）、`tasks`、`judges`、`constants` + `db/repositories/evaluation_v2.py` | `AuthenticatedUser`；`ChatSessionOps` / `AgentTurnDispatchOps` 是 **dead Protocol 字面量**（零运行时调用，见 SSoT Section 2.1 脚注和 Section 8 F14） | `api/routes.py` | 同上 |
| **web_access** | —（无实体） | — （功能子包：`reader/`、`search/`、`utils/`） | — | `api/routes.py` | `architecture/web-access.md`（起稿中） |

详细的实体 schema 数量、service 模块责任、每个 port 的方法签名都在 SSoT Section 2.1 以及每个 domain 对应的 consolidated 文档里；本表只做定位用。

> 标注「起稿中」的 consolidated 架构文档正在其他 lane 并行起草，落 main 后本表与下文 §3 的「去哪读」会把 code path 升级为 Markdown 链接。

---

## 3. 每个 Domain 一段话定位

下面每段只讲「这个 domain 负责什么、对谁重要、去哪读细节」。实现细节、canonical rule 都由 cross-ref 接管。

### 3.1 identity

- **做什么**：用户账号、角色（`Role`）、OAuth 绑定、`fastapi-users` 接入。
- **canonical 点**：它是整个系统里唯一能 `import User` / `import Role` 的地方（G15/G16 禁止其他 domain 这么做）。别的 domain 想读 user 只能通过 `AuthenticatedUser(Protocol)`；想写 user（目前只有 `conversation.chat_collection_service` 更新 `User.chat_collection_id`）必须走 `identity_user_ops.set_chat_collection` 这类 facade（lesson 9a-sexdec 三层优先级）。
- **3 个 `*InitOps` adapter**：`BotInitOps` / `ChatInitOps` / `QuotaInitOps` 在 `aperag/app.py` 启动时注入，用于 `UserManager.on_after_register` —— 新用户注册时自动创建默认 Bot、默认 ChatCollection、默认配额。
- **去哪读**：`architecture/identity-governance-model-platform-marketplace.md`（起稿中）

### 3.2 governance

- **做什么**：API Key 管理 + 审计日志（`AuditLog` / `ApiKey`）。
- **注意**：它**不**负责配额（`quota_service`）— 配额是 standalone-infra permanent seam，通过 `QuotaOps` Protocol 注入给 `knowledge_base` 和 `conversation` 两个消费方（SSoT Section 5.1）。
- **路由**：`aperag/views/api_key.py` 和 `aperag/views/audit.py` 是 shim，真实 router 是 `aperag/domains/governance/api/routes.py`。
- **去哪读**：`architecture/identity-governance-model-platform-marketplace.md`（起稿中）

### 3.3 model_platform

- **做什么**：LLM provider / 模型配置 / 默认模型（`LLMProvider` · `LLMProviderModel`）。
- **2-router split**：`/api/v1/llm_configurations` 在 `llm_routes.py`，`/api/v2/providers` 在 `providers_v2_routes.py`；两个 router 同时挂在 `aperag/app.py`，便于前端在 v1/v2 共存期平滑过渡（SSoT 8.2 F12 记录了未来合并候选）。
- **跨 domain 消费**：`conversation.chat_title_service` 直接 import `default_model_service`；`conversation.chat_collection_service` 直接 import `llm_available_model_service`（都是 provider-in-domain 的直接 import）。
- **去哪读**：`architecture/identity-governance-model-platform-marketplace.md`（起稿中）

### 3.4 marketplace

- **做什么**：公开 collection 发布 / 订阅（`CollectionMarketplace` · `UserCollectionSubscription`）。
- **与 KB 的关系**：`knowledge_base.collection_service` 通过自持的 `MarketplaceOps` / `MarketplaceCollectionOps` Protocol 消费；marketplace 结构性满足，不反向 import KB 的 `ports.py`（consumer-owned 原则）。
- **用户操作面**：发布 / 订阅流程见 `user-guide/collection-marketplace.md`（起稿中）；架构内部细节由 `architecture/identity-governance-model-platform-marketplace.md`（起稿中）覆盖。

### 3.5 knowledge_base

- **做什么**：collection / document / collection-summary —— 知识库领域的主体。
- **消费的 5 个 port**：`AuthenticatedUser` · `MarketplaceOps` · `MarketplaceCollectionOps` · `SearchPipelineOps` · `QuotaOps`。5 个里 1 个是 standalone-infra permanent（`_quota_ops`）、3 个 provider 已搬进 domain（走 `aperag/app.py` adapter）、1 个（`_search_pipeline_ops`）分类未定（SSoT 8.2 F15 记录）。
- **schema**：跨 domain 共享的 `CollectionConfig` / `KnowledgeGraphConfig` / `IndexPrompts` / `Chunk` / `VisionChunk` 等 primitives 落在 `aperag/schema/common.py`（SSoT Section 2.3 严格准入规则）；KB 自己的 `Collection` / `Document` / `CollectionView` 等 Pydantic schema 在 `aperag/domains/knowledge_base/schemas.py`。
- **去哪读**：`architecture/indexing-retrieval-kg.md`（起稿中）（KB 与 indexing / retrieval / knowledge_graph 深度耦合，放在同一篇 consolidated doc 里）

### 3.6 indexing

- **做什么**：索引 reconciler + 每种索引类型的 worker（vector / fulltext / graph / summary / vision）。
- **谁驱动它**：`knowledge_base.collection_service` 通过自持的 `SearchPipelineOps`；indexing 对外暴露 `CollectionIndexingView` / `IndexingTrigger` 给 retrieval / KB 用。
- **注意**：indexing 没有 `api/routes.py` —— 它是内部重建任务的后端，不对外提供 HTTP。
- **去哪读**：`architecture/indexing-retrieval-kg.md`（起稿中）

### 3.7 retrieval

- **做什么**：检索 pipeline 编排 + chunk 聚合 + reranking。
- **与 knowledge_graph 的关系**：retrieval 自持 `GraphQueryContext` · `GraphSearchContract`，knowledge_graph 结构性满足 —— **单向** Protocol（lesson 9a-quad）。G10/G3 禁止 knowledge_graph 反向 import retrieval。
- **去哪读**：`architecture/indexing-retrieval-kg.md`（起稿中）

### 3.8 knowledge_graph

- **做什么**：实体 / 关系 ORM + Nebula Graph 客户端 + `graphindex` reconciler。
- **自持 port**：`CollectionRow` —— 内部抽象，封装 `graphindex` 旧代码对 KB `Collection` 形状的依赖。
- **路由**：`aperag/views/graph.py` 保留一条 410-Gone legacy route（`/collections/{id}/graphs/export/kg-eval`，Phase 2 hard-cut 的 tombstone），其他都在 `aperag/domains/knowledge_graph/api/routes.py`。
- **去哪读**：`architecture/indexing-retrieval-kg.md`（起稿中）

### 3.9 conversation

- **做什么**：Bot / Chat / TurnFeedback + 聊天发起编排。内部 6 个 service（SSoT Section 2.5 详细拓扑）：
  - `bot_service` · `turn_feedback_service` —— 叶子，不消费其他 conversation service。
  - `chat_title_service` —— 读 `chat_service`，调 `model_platform.default_model_service`。
  - `chat_collection_service` —— 调 `knowledge_base.collection_service`、`model_platform.llm_available_model_service`、`identity.identity_user_ops.set_chat_collection`（唯一一个跨 domain 写 `User` 的路径）。
  - `chat_document_service` —— 消费 `chat_collection_service`（Phase 6 entry 2 退役了 `ChatCollectionServiceOps` seam，改走 sibling 直接 import）。
  - `chat_service` —— CRUD 顶层 + 导出 `chat_service_global` 模块级单例 + `ChatRow` ORM 别名（让 ORM 行和 Pydantic `Chat` 响应 schema 在同一模块同名无冲突）。
- **消费的 port**：`KnowledgeBaseCollectionView` · `AuthenticatedUser` · `QuotaOps`（`_quota_ops` 是 standalone-infra permanent 的两条永久 seam 之一）。
- **跨 domain 被谁消费**：`agent_runtime.runtime`（late-import `chat_document_service`）、`evaluation.worker`（late-import `chat_service_global`，在 `dispatch_fn` 内）。late-import 是故意的，为了避免 module-import-time 的 `evaluation → agent_runtime → conversation` 环。
- **路由拆分**：`api/routes.py` 同时导出 `chat_router`（挂 `/chats`）和 `bots_router`（挂 `/bots`）两个 router，app.py 分开挂（前缀不同，便于 OpenAPI 聚合）。
- **去哪读**：`architecture/conversation-agent-evaluation.md`（起稿中）

### 3.10 agent_runtime

- **做什么**：agent turn 编排 / SSE 流式 / artifact 存储（`AgentTurn` · `AgentTimelineEvent` · `AgentArtifact`）。
- **`PromptTemplateOps` seam**：`runtime._prompt_template_ops` 是第 2 条 standalone-infra permanent DI 槽（SSoT Section 5.1 的两条之一），provider 是 `aperag.service.prompt_template_service`（跨切面，跨 `agent_runtime` 执行 / `conversation` bot-config / indexing prompt / 用户面 `/prompts` CRUD 四个地方，无法归入某个 domain）。
- **与 evaluation 的关系**：`evaluation.worker.dispatch_fn` late-import `agent_runtime.runtime` 和 `agent_runtime.services` 做 turn 分派。
- **去哪读**：`architecture/conversation-agent-evaluation.md`（起稿中）

### 3.11 evaluation

- **做什么**：数据集 / 评估 run / judge。大量 ORM 行（10 个实体类 / 5 个 Enum）。
- **重要细节**：`worker.dispatch_fn` 是 **测试注入用的模块级函数引用**，**不是** `Protocol + DI` 槽 —— 测试 monkeypatch 它来替换真实 turn 分派器。G18 alt 注册表不包括 `dispatch_fn`，因为默认实现是一个正常函数而不是 `None` 槽（SSoT Section 5.2 明确列出这个区别）。
- **dead Protocol 字面量**：`ports.py` 里仍写着 `ChatSessionOps` / `AgentTurnDispatchOps` 两个 Protocol class，但零运行时调用（rebase 过程中被搁置，SSoT Section 2.1 脚注 + Section 8 F14 说明）。将来会机械删除，按 Phase 6 entry 4 删 `ChatDocumentOps` 的先例。
- **terminal 状态判断**：`EvaluationRunStatus.is_terminal()` classmethod（Phase 6 从 `_TERMINAL_RUN_STATUSES` 常量升格），任何消费方都走这个 API。
- **去哪读**：`architecture/conversation-agent-evaluation.md`（起稿中）

### 3.12 web_access

- **做什么**：爬虫抓取 / 网络搜索 / URL 阅读 —— 提供给 `knowledge_base` 的 document ingestion 用。
- **当前形态**：**没有实体**、没有 service，只有 schemas、路由、三个功能子包（`reader/` · `search/` · `utils/`）。SSoT 8.2 F11 把「是否给 web_access 补 entity + service」列为未来候选。
- **去哪读**：`architecture/web-access.md`（起稿中）

---

## 4. Domain 之间的依赖关系

下图是稳态关系（边上的标签是「用什么机制」）：

```
identity      ←── AuthenticatedUser Protocol (read)   ← 所有其他 domain
identity      ─── identity_user_ops facade (write)    → conversation.chat_collection_service

governance    （无跨 domain 入边；被 admin UI 消费）
quota_service（standalone-infra）─── QuotaOps Protocol → knowledge_base.collection_service, conversation.bot_service

model_platform ─── default_model_service 直接 import  ← conversation.chat_title_service
model_platform ─── llm_available_model_service 直接    ← conversation.chat_collection_service

marketplace   ─── MarketplaceOps / MarketplaceCollectionOps（结构性满足）→ knowledge_base.collection_service

knowledge_base ─── Collection / Document 直接 import  → conversation.chat_collection_service, retrieval.pipeline, agent_runtime.runtime, web_access routes

indexing      ─── CollectionIndexingView / IndexingTrigger  ↔ knowledge_base / retrieval

retrieval     ─── GraphSearchContract Protocol（单向）→ knowledge_graph
knowledge_graph：禁止反向 import retrieval.ports / retrieval.service / retrieval.schemas（G10/G3）

conversation.chat_service_global      ─── 直接 import ← evaluation.worker（late-import，破环）
conversation.chat_document_service    ─── 直接 import ← agent_runtime.runtime（late-import）

agent_runtime.runtime ─── _prompt_template_ops DI slot  ← standalone-infra: aperag.service.prompt_template_service

evaluation    （无跨 domain provider 角色；被 admin UI 消费；worker 有测试注入 dispatch_fn seam）

web_access    （无实体；消费 KB Collection 做带网页增强的检索）
```

要点：

- **先看 provider 在哪**。如果 provider 已经在 `aperag/domains/<d>/`，consumer 直接 import。
- **否则用 consumer-owned Protocol + DI 槽**。consumer 自持 `Protocol` 和槽，`aperag/app.py` 启动时注入（legacy provider 或 adapter）。
- **standalone-infra permanent（类 B）** 只有两条：`QuotaOps` 和 `PromptTemplateOps` —— 它们不会搬进任何 domain，永久用 DI 注入（SSoT Section 3.2）。
- **late-import** 只用在破 module-import 循环。目前唯一的活用法是 `evaluation.worker.dispatch_fn` 和 `agent_runtime.runtime` 对 `conversation` 服务的调用。

---

## 5. 不属于 domain 的共享基础设施

下面这些 top-level 模块是 domain 共享基础设施，**不**归入任何一个 domain：

| 模块 / 目录 | 作用 |
| --- | --- |
| `aperag/app.py` | FastAPI app factory；模块级 DI wire-up —— 7 个 Phase 3+4 槽 + 2 个 Phase 5/6 permanent 槽（SSoT Section 5） |
| `aperag/config.py` | Settings / 环境变量 / 数据库 URL |
| `aperag/openapi_spec.py` | OpenAPI 生成 + `HIDDEN_FROM_PUBLIC_PATH_PREFIXES` 注册表 |
| `aperag/db/base.py` | SQLAlchemy declarative `Base`；所有 per-domain `db/models.py` 共用这一个 metadata 注册表（让 alembic autogenerate 能跨 domain 发现模型） |
| `aperag/db/ops.py` | 异步 DB 查询 helper |
| `aperag/db/models.py` | **re-export shim** — 把各 domain 的 ORM 重新导出给还没迁移的调用方 |
| `aperag/db/repositories/*.py` | Repository 层（domain-owned + legacy 混杂） |
| `aperag/schema/common.py` | **跨 domain 共享的原语 Pydantic 类型**，严格准入（SSoT Section 2.3） |
| `aperag/schema/view_models.py` | **re-export shim + dual-hook** —— 给老调用方兜底（SSoT Section 3.3） |
| `aperag/llm/` | LiteLLM 集成（completion / embedding / rerank / cache）—— 共享基础设施，不是 domain |
| `aperag/utils/` | `audit_decorator`、常量、日期 / 分页 helper、spider、LLM response 处理等 |
| `aperag/service/*.py` | **re-export shim** + 一小撮 legacy-only service（`chat_completion_service` / `evaluation_service` / `export_service` / `prompt_template_service` / `quota_service` / `search_pipeline_service` 等，SSoT Section 6.1） |
| `aperag/views/*.py` | router re-export shim + 一小撮 legacy 非 domain view（`auth.py`、`config.py`、`main.py`、`openai.py`、`prompts.py` 等） |
| `aperag/agent_runtime/`、`aperag/evaluation_v2/` | **legacy top-level package** —— 仅作 re-export shim，让旧 import 路径仍可用；真实实现在 `aperag/domains/agent_runtime/` / `aperag/domains/evaluation/` |

所有 legacy shim 的退役计划都在 SSoT Section 6 + Section 8（F2 / F3 / F9 future candidates）记录，新代码**不要**走 shim 路径。

---

## 6. 新代码落位速决

如果你要加新代码，先回答三个问题：

1. **功能属于哪个 domain？**
   - 找到对应 `aperag/domains/<d>/`，按目录契约放（`db/models.py` / `schemas.py` / `service/` / `api/routes.py`）。
   - 不确定归哪个 domain，对照上面 §3 的一段话定位。
2. **要用到的 primitive 是跨 domain 共用的吗？**
   - 如果是 ≥2 个 domain 都会用、且是纯值对象（无 ORM 依赖 / 无 domain 特定语义），放 `aperag/schema/common.py`（准入严格，对照 SSoT Section 2.3）。
   - 否则只放本 domain 的 `schemas.py`。
3. **是真正的跨切面基础设施吗？**
   - 共享的 LLM 封装 → `aperag/llm/`；OAuth / fastapi-users 集成仍留在 `aperag/views/auth.py`；DB session helper → `aperag/db/` 下。
   - **不要**新建 `aperag/domains/infrastructure/` 之类的 —— domain 就是业务 domain，基础设施继续留 top-level。

跨 domain 调用：

- Provider 已经在 `aperag/domains/` → 直接 import（**不要**绕一层 `ports.py`）。
- Provider 还没搬 → consumer 自持 `Protocol` + DI 槽，到 `aperag/app.py` 注入。
- Provider 是永久跨切面基础设施（`QuotaOps` / `PromptTemplateOps`）→ 同样走 DI 槽，保持永久性（**不要**试图把它搬进 domain）。

新代码上线前，`make test-unit` 会跑 `tests/unit_test/test_modularization_boundaries.py` 里的 20 条边界测试；G1-G19 定义见 SSoT Section 4 + [`development/development-guide.md`](../development/development-guide.md) 里的「Domain 边界门禁 G1–G19 速查」。

---

*本文档基线：`origin/main @ 10cabcf`（PR #1637 merged），与 canonical SSoT `docs/modularization/architecture.md` 同步。domain 清单 / cross-domain rule / permanent seam 任何变更先改 SSoT，再回头更新本表。*
