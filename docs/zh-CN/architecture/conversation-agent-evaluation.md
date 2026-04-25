---
title: Conversation / Agent Runtime / Evaluation
position: 34
---

# Conversation / Agent Runtime / Evaluation 三域架构

> 本文是 Post-Phase-6 current-state 架构文档，覆盖 `conversation` / `agent_runtime` / `evaluation` 三个后端 domain。后端跨 domain 的**规则与 invariant**（direct import vs Protocol + DI、G1–G19 gates、两条永久 `CRITICAL_WIRINGS`、User write hierarchy）集中记录在 [`docs/modularization/architecture.md`](../../modularization/architecture.md)，本文引用它的 section anchor 而不重复定义。

> **Baseline**: `origin/main @ 10cabcf`（Phase 6 / PR #1635 + Architecture doc / PR #1636 + zh-CN blueprint / PR #1637 merged）。

---

## 1. 三域边界一览

这三个 domain 合并在同一篇是因为它们构成了 ApeRAG 的一条核心 runtime flow：

```
用户发起聊天 (conversation) → 运行一次 agent turn (agent_runtime) → 聚合评估结果 (evaluation)
```

三域分工：

| Domain | 职责 | 关键对象 |
| --- | --- | --- |
| `conversation` | 对话生命周期管理（Bot / Chat / Turn feedback / 文件上传） | `Bot`、`Chat`、`TurnFeedback`、6 个服务（bot/chat/chat_document/chat_collection/chat_title/turn_feedback） |
| `agent_runtime` | 单次 Agent turn 的执行引擎（PydanticAI + MCP + SSE 事件） | `AgentTurn`、`AgentTimelineEvent`、`AgentArtifact`、`agent_runtime_manager` 单例 |
| `evaluation` | 评估数据集 + 运行 + item + attempt 的编排 | `EvaluationDataset` / `Item`、`EvaluationRun` / `Item` / `Attempt`、`evaluation_run_service` + `worker.execute_evaluation_run` |

每个 domain 自己的 canonical layout（`db/models.py` + `schemas.py` + `ports.py` + `service/` + `api/routes.py`）与跨 domain 的整体 layout 保持一致，详见 [`architecture.md §2 Domain map`](../../modularization/architecture.md#2-domain-map)。

## 2. Conversation domain

canonical 位置：`aperag/domains/conversation/`。

### 2.1 数据模型

| ORM 类 | 作用 |
| --- | --- |
| `Bot`（+ `BotStatus` / `BotType` 两个 enum） | 对话机器人定义；持有 system prompt / query prompt / 关联 Collection / LLM provider 等配置 |
| `Chat`（+ `ChatStatus` / `ChatPeerType`） | 对话线；聚合多个 Turn；维护最近消息、历史汇总 |
| `TurnFeedback`（+ `TurnFeedbackType` / `TurnFeedbackTag`） | 用户对单个 Agent 回答的点赞/点踩 / 打标 |

21 个 Pydantic schema（Bot / BotConfig / BotCreate / BotUpdate / Chat / ChatList / ChatCreate / ChatDetails / ChatMessage / Reference / File / TitleGenerateRequest / TurnFeedback / Feedback / TurnFeedbackWrite / Agent 等）定义在 `aperag/domains/conversation/schemas.py`，并通过 dual-hook（见 [`architecture.md §3.3`](../../modularization/architecture.md#33-dual-hook-scenario-a)）保留 pre-migration 的 `aperag.schema.view_models.*` import 路径。

### 2.2 Service 层拓扑

6 个 service singleton 位于 `aperag/domains/conversation/service/`。它们在 domain 内部全部以**sibling direct import** 交互，域内没有 Protocol seam：

- **叶子服务**：
  - `bot_service` — Bot CRUD + 与 `quota_service` 的 `_quota_ops` Protocol+DI seam 对接。
  - `turn_feedback_service` — TurnFeedback CRUD；导出 `turn_feedback_service_global` 单例。
- **引用型服务**：
  - `chat_title_service` — 读 `chat_service` 拿 chat 元数据；通过 cross-domain direct import 从 `model_platform.default_model_service` 取默认模型。
  - `chat_collection_service` — 创建"本 chat 专属 Collection"；cross-domain direct import 使用 `knowledge_base.collection_service`、`knowledge_base.schemas.CollectionCreate`、`model_platform.llm_available_model_service`；对 `User.chat_collection_id` 字段的写入走 `identity.service.identity_user_ops.set_chat_collection` **facade**（lesson 9a-sexdec hierarchy-1 终态，见 [`architecture.md §3.4`](../../modularization/architecture.md#34-user-write-hierarchy-lesson-9a-sexdec)）。
  - `chat_document_service` — 单 chat 上传文件管理；**sibling direct import** 调 `chat_collection_service`（Phase 6 entry 2 之后不再走 `ChatCollectionServiceOps` Protocol seam）；cross-domain direct import 使用 `knowledge_base.document_service` + `knowledge_base.schemas.Document`。
  - `chat_service` — 顶层 Chat CRUD；导出 `chat_service_global` 单例，以及 `ChatRow` alias（解决 ORM Row 与同名 Pydantic `Chat` 响应 schema 的命名冲突）。

### 2.3 API 路由

Router 文件：`aperag/domains/conversation/api/routes.py`。**导出两个 router object**：

- `chat_router` → `/api/v1/bots/{bot_id}/chats/*`（与 pre-migration 保持 byte-stable）
- `bots_router` → `/api/v2/bots/*`（新的统一 CRUD）

这是 Phase 4 在 `model_platform` 引入的"同域 v1+v2 并存必须 2-router split"模式的第二个使用者，详见 [`architecture.md §2.1 model_platform`](../../modularization/architecture.md#21-domain-inventory)。

### 2.4 Consumer-owned Protocols

`aperag/domains/conversation/ports.py` 现场 3 个 Protocol：

- `KnowledgeBaseCollectionView`（从 Phase 3 起即在；描述 conversation 消费 Collection 时读到的最小字段）
- `AuthenticatedUser`（local-decl per lesson 9a-ter，避免 import `User` ORM；见 [`architecture.md §3.4`](../../modularization/architecture.md#34-user-write-hierarchy-lesson-9a-sexdec)）
- `QuotaOps`（被 `bot_service` 消费；provider 是 standalone-infra `aperag.service.quota_service`；是两条永久 `CRITICAL_WIRINGS` 之一，见下文 §5）

### 2.5 跨 domain 消费者

- `agent_runtime.runtime` 在 async turn 里 late-import `chat_document_service.has_documents_in_chat`（late-import 为避免 module-import-time cycle 风险）。
- `evaluation.worker.dispatch_fn` late-import `chat_service_global.create_chat`（同样是 late-import 破 cycle）。

---

## 3. Agent Runtime domain

canonical 位置：`aperag/domains/agent_runtime/`。

### 3.1 数据模型

| ORM 类 | 作用 |
| --- | --- |
| `AgentTurn`（+ `AgentTurnStatus`） | 单次 turn 的执行状态；`queued / running / completed / failed / cancelled` |
| `AgentTimelineEvent`（+ `AgentEventActor`） | Turn 过程中的事件流；前端通过 SSE 订阅 |
| `AgentArtifact`（+ `AgentArtifactType`） | Turn 过程中产出的结构化内容（引用、图片等） |

`schemas.py` 里有 13 个 Pydantic schema：`AgentMessage`（在 Phase 5 5-S5a 单独 carve）、`CreateTurnRequest`、`VisibleAgentState`、以及 timeline event 与 artifact 的各种 envelope shapes。

### 3.2 模块组成

- `runtime.py` — `agent_runtime_manager` 单例 + `AgentRuntime` 抽象；turn 生命周期主流程。
- `services.py` — HTTP 边界的 turn CRUD / artifact 获取 / timeline 查询。
- `storage.py` — 基于 Redis 的 turn dispatch + lease 协调（`RedisConnectionManager`）。
- `api/routes.py` — 对外 REST + SSE endpoint（全部挂在 `/api/v2` 前缀）。

### 3.3 Turn 生命周期（核心流程）

```
1. HTTP POST /api/v2/turns          (api/routes.py)
2. TurnService.create_or_get_turn   (services.py)
3. agent_runtime_manager.claim_turn (通过 Redis lease 声明 owner)
4. launch_turn → AgentRuntime.run_turn (runtime.py)
   - History writer 构造历史上下文
   - 拉 chat_document_service.has_documents_in_chat (late-import conversation domain)
   - 走 PromptTemplateOps DI slot 解 system + query prompt
   - PydanticAI Agent 流式执行，emit timeline events
   - 产出 artifact、写回 DB
5. Turn 进入终态 (completed / failed / cancelled)，释放 lease
```

### 3.4 Consumer-owned Protocol — `PromptTemplateOps`（永久 seam）<a id="protocol-promptTemplateOps"></a>

`aperag/domains/agent_runtime/ports.py` 定义：

```python
class PromptTemplateOps(Protocol):
    async def resolve_agent_system_prompt(self, *, bot, user_id) -> str: ...
    async def resolve_agent_query_prompt(self, *, bot, user_id) -> str: ...
    def build_agent_query_prompt(self, chat_id, *, agent_message, user, template=None, has_chat_files=False) -> str: ...
```

provider 是 `aperag.service.prompt_template_service`（standalone-infra — prompts 的解析逻辑跨 agent_runtime / conversation / indexing / `/api/v2/prompts` REST 四处共享，没有自然 domain 归属。Phase 8 #49 G3 起 REST 路由碎入 `aperag/domains/model_platform/api/prompts_routes.py`，通过 model_platform 的 `PromptCRUDOps` Protocol wire 同一单例；底层服务文件不动）。`aperag/app.py` 在启动时用 `_PromptTemplateOpsAdapter` 把具体服务 wire 进 runtime 的 `_prompt_template_ops` slot，详见 [`architecture.md §5.1`](../../modularization/architecture.md#51-the-phase-5-permanent-two-entry-registry-g18-alt)。

`_prompt_template_ops` 是 G18 alt 永久 `CRITICAL_WIRINGS` 的两条之一；`test_phase5_di_critical_wirings_at_app_startup` 在 CI 上守住 "import aperag.app 后 slot 必须非 None"。

### 3.5 跨 domain 直接引用（后 Phase 6 entry 4）

- `conversation.chat_document_service` — 在 `run_turn` 的 async scope 里 late-import；Phase 6 entry 4 删除了 `ChatDocumentOps` Protocol（它在 Phase 5 5-S1 被 seeded，但 conversation domain merge 后就变成 dead literal）。
- `aperag.domains.knowledge_base.schemas.Collection as KBCollectionSchema` — turn 过程中引用 Collection 上下文。

### 3.6 API

- `POST /api/v2/turns` 创建/获取 turn。
- `GET /api/v2/turns/{turn_id}` turn snapshot。
- `POST /api/v2/turns/{turn_id}:cancel` 取消 turn。
- `GET /api/v2/turns/{turn_id}/events` SSE 实时 timeline 事件流。
- `GET /api/v2/artifacts/{artifact_id}` 取 artifact。

所有 handler 都用 **local-decl `AuthenticatedUser(Protocol)`** 作为 auth 参数类型（lesson 9a-ter），不 import `User` ORM。

---

## 4. Evaluation domain

canonical 位置：`aperag/domains/evaluation/`。

### 4.1 数据模型

四张表 + 一组枚举：

- `EvaluationDataset` / `EvaluationDatasetItem`（+ `EvaluationDatasetSourceType`）
- `EvaluationRun` / `EvaluationRunItem` / `EvaluationRunItemAttempt`（+ `EvaluationRunStatus` / `EvaluationRunItemStatus` / `EvaluationRunItemAttemptStatus`）
- `EvaluationJudgeMode`

产品视角使用说明见 [`user-guide/evaluation-guide.md`](../user-guide/evaluation-guide.md)。

### 4.2 Worker 状态机（`worker.execute_evaluation_run`）

```
READ run.status
  |-- 已在终态 → 直接返回该状态
  |-- QUEUED / 其它 → 写 RUNNING，拉 items
LOOP items:
  |-- 重新读 run.status（允许用户中途 cancel）
  |-- 若已终态 → break
  |-- dispatch_fn(user_id, bot_id, input_message) → TurnDispatchOutcome
  |-- 持久化 RunItemAttempt + 推进 summary
FINAL:
  |-- 若 latest_run 已终态 → 沿用它
  |-- 否则用 summary 推导 COMPLETED / FAILED / CANCELLED
```

### 4.3 `EvaluationRunStatus.is_terminal()` classmethod

Phase 6 entry 5 把原来的 module-level `_TERMINAL_RUN_STATUSES` frozenset 提升为 enum classmethod：

```python
class EvaluationRunStatus(str, Enum):
    ...
    @classmethod
    def is_terminal(cls, status) -> bool:
        return status in (cls.COMPLETED, cls.FAILED, cls.CANCELLED)
```

**byte-identical semantics preservation**：该 classmethod 保留了 `54cd86b` (PR #1631, task #23) 修复 cancel→running TOCTOU race 时引入的 `frozenset` 判断逻辑，语义上与原 `status in _TERMINAL_RUN_STATUSES` 等价。`repositories/evaluation_v2.py::update_run_status` 仍然用 `is_terminal(run.status)` 守住"终态后不再接收 RUNNING 覆写"的不变量。

### 4.4 `dispatch_fn` test-injection seam

`dispatch_fn` 是 `worker.py` 里的 module-level **function reference**，默认值是 `dispatch_evaluation_turn`。测试可以通过 `execute_evaluation_run(dispatch_fn=fake)` 注入替身而不需要启动真实 runtime / Redis / broker。

这是**刻意设计的非 Protocol+DI seam**：

- 它不是 slot（模块启动时就有默认值，不是 `None`）。
- 它不需要 `aperag/app.py` wire-up。
- 它在测试 fixture 里 monkey-patch 即可。
- 因此它**不属于** G17 / G18 alt `CRITICAL_WIRINGS` 的覆盖范围，详见 [`architecture.md §5 "two patterns outside registries"`](../../modularization/architecture.md#5-runtime-seams--critical_wirings-steady-state)。

### 4.5 零 Protocol + DI seam

evaluation domain 不声明任何 consumer-owned Protocol + DI slot 给外部 wire。所有跨 domain 依赖都是 **late-import + direct cross-domain**：

```python
# in worker.py::dispatch_evaluation_turn function body
from aperag.domains.agent_runtime.db.models import AgentTurnStatus
from aperag.domains.agent_runtime.runtime import agent_runtime_manager
from aperag.domains.agent_runtime.schemas import CreateTurnRequest
from aperag.domains.conversation.service.chat_service import chat_service_global
```

Late-import 使 evaluation 模块本身的 import-time 依赖最小，避免 `evaluation → agent_runtime → (back) evaluation` 方向的模块环问题。evaluation domain 自己的 ports.py 里也有一对**残留的 dead Protocol class** `ChatSessionOps` / `AgentTurnDispatchOps` — 它们是 Phase 5 5-S1 seeded 的假设 chat_service + agent_runtime 仍 legacy 才需要的 Protocol，但随着这两个 provider 被 domain-move 完，seam 从未被 wire，class body 留在 ports.py 仅作 historical marker；删除方式与 `ChatDocumentOps` 一致（一次机械 cleanup），见 [`architecture.md §8 F14`](../../modularization/architecture.md#8-future-candidates)。

### 4.6 API

- Dataset CRUD：`GET/POST/PUT/DELETE /api/v2/evaluation-datasets{,/{id}}`
- Dataset items：`GET/POST/PUT/DELETE /api/v2/evaluation-datasets/{id}/items{,/{item_id}}`
- Run CRUD + cancel：`GET/POST /api/v2/evaluation-runs`, `/api/v2/evaluation-runs/{id}`, `/api/v2/evaluation-runs/{id}/cancel`
- Run items + attempts + retry：`/api/v2/evaluation-runs/{id}/items{,/{item_id}}`, `/api/v2/evaluation-runs/{id}/items/{item_id}/attempts`, `/api/v2/evaluation-runs/{id}/items/{item_id}/retry`

所有 handler 使用 local-decl `AuthenticatedUser(Protocol)`，不 import `User` ORM。

---

## 5. 跨三域 runtime flow（把 §2 + §3 + §4 串起来）

以下两条是实际运行时两类典型 flow：

### 5.1 用户对话 flow

```
1. Frontend POST /api/v2/turns (body: chat_id + user message)
2. agent_runtime.api.routes → TurnService.create_or_get_turn
3. DB: insert AgentTurn (status=queued), update Chat head
4. agent_runtime_manager.claim_turn (Redis lease)
5. launch async _runner()
       ↓ (在 async scope 里)
   - HistoryWriter 读 chat 历史
   - late-import conversation.chat_document_service 查 chat 是否有文件
   - 通过 _get_prompt_template_ops() 走 PromptTemplateOps DI slot
       → aperag.service.prompt_template_service.resolve_agent_{system,query}_prompt
   - PydanticAI Agent.run_stream()
   - emit AgentTimelineEvent (SSE 推到前端)
   - 写回 AgentArtifact + AgentTurn.status=completed
6. Frontend 通过 GET /api/v2/turns/{turn_id}/events SSE 订阅实时事件流
```

### 5.2 自动化评估 flow

```
1. User 在 Collection Evaluations 页面点 "Start evaluation"
2. POST /api/v2/evaluation-runs (body: dataset_id, optional bot_id)
3. evaluation_run_service.create_run:
   - 解析 bot_id（默认"Default Agent Bot" active → 最早 active bot）
   - 快照 bot_config + model_config
   - value-copy dataset_items → run_items
4. Celery task 拉起 worker.execute_evaluation_run(run_id)
5. worker loop：
   - 每次 iter 重新读 run.status，允许 cancel 中断
   - dispatch_fn per item (=dispatch_evaluation_turn by default)
       → 与 §5.1 相同的 runtime flow：late-import chat_service_global + agent_runtime_manager
       → 产出 chat + turn + run item attempt
   - 更新 run.summary
6. 所有 item 走完后 finalize run：
   - 若 worker 过程中 user 已 cancel → latest_run.status 已终态 → 沿用
   - 否则根据 summary 推导 COMPLETED / FAILED
```

两条 flow 都体现了 canonical rule："已 domain-move 的 provider → direct import；标为 standalone-infra 的 provider → consumer-owned Protocol + DI"，详见 [`architecture.md §3.1-3.2`](../../modularization/architecture.md#31-direct-import-vs-protocol--di)。

---

## 6. 边界与注意事项

- **不要在 evaluation domain 引入 Protocol+DI seam 给 runtime / chat**：它们都已 domain-move，直接 late-import 即可（canonical 规则见 architecture.md §3.1）。
- **不要在 conversation / agent_runtime / evaluation 任何模块 import `User` ORM 或 `Role` enum**（G15/G16）：读走 `AuthenticatedUser(Protocol)` local decl，写走 `identity_user_ops.*` facade。
- **不要在 api/routes.py 使用 `from __future__ import annotations`**（G19；lesson 9a-quatuordec 血泪教训）。
- **任何新加的 Protocol + DI seam**都必须同步登记到 G17 或 G18 alt 的 `CRITICAL_WIRINGS` 列表，并确保 `aperag/app.py` 在模块加载时 wire 完。详见 [`architecture.md §4.1 gate catalog`](../../modularization/architecture.md#41-backend-gate-catalog)。

## 7. 相关文档

- [`docs/modularization/architecture.md`](../../modularization/architecture.md) — 后端整体 canonical current-state；本文多处 cross-ref。
- [`docs/modularization/breaking-changes/phase5-conversation-agent-eval.md`](../../modularization/breaking-changes/phase5-conversation-agent-eval.md) — Phase 5 硬切分的 import surface migration table + rollback 说明。
- [`user-guide/chat-interaction.md`](../user-guide/chat-interaction.md) — 本三域的用户视角使用指南。
- [`user-guide/evaluation-guide.md`](../user-guide/evaluation-guide.md) — evaluation 产品使用说明。
- [`admin-guide/prompt-customization.md`](../admin-guide/prompt-customization.md) — prompt 三层覆盖机制。
- [`reference/prompt-api.md`](../reference/prompt-api.md) — `/prompts/*` REST 参考。
