# Agent Message Protocol — Final-State Design (Phase 8 D8 Canonical)

**Status**: Design canonical — implementation pending PM lane split
**Author**: 符炫炜 (总架构师)
**Date**: 2026-04-25
**Trigger**: earayu2 msg=5d3c428c — first-version Agent Runtime 重写时残留了 legacy fields (`StoredChatMessagePart.references / urls / metadata`)

## Problem Statement

ApeRAG 当前有两套并行 message storage 设计：
1. **Legacy parts-based** (`aperag/chat/history/message.py::StoredChatMessagePart`) — 仅 non-agent bot path 使用
2. **Modern artifact-based** (`aperag/domains/agent_runtime/`) — agent runtime 使用，与 legacy 并行存在但完全不交叉

这种 dual-track 是历史遗留：legacy 保留下来时附带的 `references / urls / metadata` 字段在 agent path 已被 `AgentArtifact` 取代，但仍占据 schema surface。

## Standards Survey Result

| 协议 | 评估 |
|---|---|
| OpenAI Responses API | de-facto lingua franca，但 vendor format |
| Anthropic Messages | typed citations 业界最佳设计 |
| **AI SDK v5 UI Message Stream Protocol** | **transport-only，开放 SSE spec，最佳 FE library 支持** |
| MCP | 不是 chat 协议（仅 tool RPC），与 chat 设计正交 |
| AG-UI | 多 framework adopted 但无 first-class citations |

> **Implementation note**: Pydantic AI already supports Vercel AI Data Stream Protocol via UIAdapter, so this wire choice remains compatible with a future PydanticAI-backed backend agent runtime.

## Canonical Decision

**采纳 AI SDK v5 UI Message Stream Protocol + Anthropic-style typed citations + MCP-ready tool lifecycle**

### 1. Wire format（FE-BE streaming）

SSE 框架 + `x-vercel-ai-ui-message-stream: v1` header

Stream parts (AI SDK v5):
- Lifecycle: `start` / `start-step` / `finish-step` / `finish` / `abort` / `error`
- Text: `text-start {id}` → `text-delta {id, delta}` → `text-end {id}`
- Reasoning: `reasoning-start/-delta/-end`
- Tools: `tool-input-start {toolCallId, toolName}` → `tool-input-delta` → `tool-input-available` → `tool-output-available` *(success)* | `tool-output-error {errorText}` *(failure, AI SDK v5 strict spec — D8.0c+ #89)*
- Sources: `source-url {sourceId, url, title?}` / `source-document {sourceId, mediaType, title}`
- Custom (ApeRAG 扩展):
  - `data-citation {cited_text, location: char_location | page_location | content_block_location | url_citation}`（Anthropic-shape）
  - `data-activity {intent: UserActivityIntent, label?, transient: true}`（保留现有 UX 价值）

### 2. Storage at-rest schema (UIMessage)

```typescript
type UIMessage = {
  id: string;
  role: "user" | "assistant" | "system"; // ChatML aligned, drop legacy "human"/"ai"
  parts: UIMessagePart[];
}

type UIMessagePart =
  | { type: "text"; text: string }
  | { type: "tool-<name>"; toolCallId: string; state: ToolState; input: Json; output?: Json; errorText?: string }
  | { type: "source-url"; sourceId: string; url: string; title?: string }
  | { type: "source-document"; sourceId: string; mediaType: string; title: string }
  | { type: "data-citation"; data: { cited_text: string; location: CitationLocation } }
  | { type: "data-activity"; data: { intent: UserActivityIntent; label?: string }; transient: true }
```

**Round-trip**: 流 events 与 stored UIMessage parts 同型 schema — 不存在"流 event vs stored message"双映射。

### 3. DB layer（保留）

- `AgentTurn` / `AgentTimelineEvent` / `AgentArtifact` 3-table model 已是 standards-aligned，保留
- 删除 `StoredChatMessagePart` / `RedisChatMessageHistory` legacy 路径

### 4. Tool integration

- 当前 RAG tools (search_collection / web_search / read_document) 暂保 internal Python calls
- Wire format 用 AI SDK v5 tool-call lifecycle（MCP-ready）
- Future（D9 独立任务）: tool layer 升级为 MCP server interface — 不影响 wire protocol 层

### §2.4 Tool naming convention (provider-safe + MCP traceability)

> **Updated by D9 final canonical (A1 + A6)** — supersedes the earlier proposal that the wire `tool-<name>` directly carry MCP dotted namespace. Many model providers require tool/function names in `[a-zA-Z0-9_-]`, so the wire name must be a **provider-safe canonical name**, while MCP server/tool identity is preserved as separate metadata.

`tool-<name>` parts 中的 `<name>` 必须是 **provider-safe `SafeToolName`**（仅 `[a-zA-Z0-9_-]`），而非 MCP dotted namespace 或 internal Python function name。MCP server / tool identity 通过 metadata 字段保留，保证 D9 把 RAG tools 升级为 MCP server interface 时 wire schema 无需 rename。

```typescript
// Wire / at-rest UIMessagePart shape (per D8 §2 + D9 A1)
{ type: `tool-${SafeToolName}`,
  toolCallId: string,
  metadata?: {
    mcpServer?: string,    // raw MCP server name (e.g. "aperag-knowledge-base")
    mcpToolName?: string,  // raw MCP tool name (e.g. "search_collection")
  },
  state: ToolState,
  input: Json,
  output?: Json,
  errorText?: string,
}
```

**SafeToolName generation rule (D9 A6)** — stable, unique, collision-resistant:

```typescript
function safeToolName(mcpServer: string, mcpToolName: string): string {
  const naive = sanitize(`${mcpServer}_${mcpToolName}`);
  if (registry.hasNoCollisionWith(naive, currentEntity)) {
    return naive;
  }
  // collision → append stable hash suffix
  const suffix = sha256(`${mcpServer}|${mcpToolName}`).slice(0, 6);
  return `${naive}__${suffix}`; // double underscore separator marks hash suffix
}

function sanitize(s: string): string {
  return s.replace(/[^a-zA-Z0-9_-]/g, "_");
}
```

Properties:
- **Stable**: same `(mcpServer, mcpToolName)` 始终生成同一 safe name
- **Unique**: collision via stable `sha256` hash suffix
- **Inspectable**: double underscore `__` 是明显的 collision marker
- **Provider-safe**: 仅 `[a-zA-Z0-9_-]`

Registry 必须存 `(mcpServer, mcpToolName, safeName)` 三元组以支持 reverse lookup（safeName → MCP identity）。

**Examples**:
- ApeRAG built-in (system): `tool-aperag_knowledge_base_search_collection` + `metadata: { mcpServer: "aperag-knowledge-base", mcpToolName: "search_collection" }`
- 用户 personal MCP server: `tool-user_notes_create_note` + `metadata: { mcpServer: "user-notes", mcpToolName: "create_note" }`
- collision case: `tool-foo_bar__a1b2c3` + `metadata: { mcpServer: "foo.bar", mcpToolName: "..." }`

落地时机：D8.3 (citations + tools) 实施时按此命名规则展开。详见 [`agent-runtime-mcp-design.md`](./agent-runtime-mcp-design.md) §A1 + §A6（D9 canonical）。

### §2.5 Web-backend MCP-capable agent target architecture (out of D8 implementation scope; D9 design-only)

> **Updated by D9 final canonical (A5 + A7)** — D9 design has now closed; this section cross-references the locked D9 deliverables. Full canonical detail lives in [`agent-runtime-mcp-design.md`](./agent-runtime-mcp-design.md).

为避免 D8.3 tool lifecycle 实施时出现 "looks MCP-ready 但命名 / 权限 / consent 不到位" 的 retrofitting 成本，本节明确 web-backend MCP-capable agent 的目标边界（actual implementation 走 D9 lane）：

- **FE protocol layer**: 只认 UIMessage stream / UIMessage storage，**不感知 MCP**。Wire schema 的 `tool-<SafeToolName>` parts (per §2.4) 与 MCP server/tool identity 解耦，不直接表达 MCP `resources/read` / `prompts/get` / `sampling/createMessage` / `elicitation` RPCs。
- **Backend agent runtime layer (D9 territory)**: 负责 orchestrate agent loop（D9 §A3 推荐 default backbone = **PydanticAI**，per Weston PydanticAI 调研 + UIAdapter 兼容性）；将 MCP RPC 内部事件投影成 UIMessage parts 透传给 FE。
- **Tool boundary**: 当前 RAG tools (search_collection / web_search / read_document) 是 Python internal calls；未来 D9 切到 MCP client/server interface 时，对 FE 仍表现为同一套 `tool-<SafeToolName>` parts（per §2.4 + D9 §A1/§A6）。
- **MCP auth / registry / consent / sampling / elicitation 边界**: 属于 D9 design scope，**必须在 D8.3 tool lifecycle 实施前完成 D9 design**（design-only 不 block D8.0 doc landing 也不 block D8.1/D8.2 stream/storage implementation）。

### D9 design canonical deliverables（locked per architect msg=2dcae724 + Weston msg=f56b2ae1 final ack）

| § | Topic | Status |
|---|---|---|
| §1 + A5 | MCP server registry — three-tier (system/bot/user) + **no silent override on system namespace**（admin alias/disable 必须 explicit + audit-logged）| ✅ locked |
| §2 | Three-level authorization (visibility / invocation / consent) | ✅ locked |
| §3 + A7 | `data-tool-consent` part with **`argsPreview` + `argsHash` (raw args backend-private)** — wire 不暴露 raw args，避免 token / secret 进 persisted UIMessage history | ✅ locked |
| §4 | Sampling — backend-internal, NOT surfaced to wire | ✅ locked |
| §5 | `data-elicitation` part — surfaced with schema-validated input | ✅ locked |
| §6 | D8.3 tool lifecycle interface contract (含 consent + elicitation lifecycle) | ✅ locked |
| §7 + A3 | PydanticAI evaluation — runtime candidate (default backbone) | ✅ locked |
| A1 + A6 | SafeToolName naming with collision-resistant hash suffix (per §2.4) | ✅ locked |
| A2 | AI SDK v5 + 自定义 consent (lock v5, NOT v6) | ✅ locked |
| A4 | D8.3 前置 7 decisions checklist | ✅ locked |

详见 [`agent-runtime-mcp-design.md`](./agent-runtime-mcp-design.md) — D9 canonical SSoT（landed by Phase 8 #72 / D9.0）。

D8.3 implementation owner 必须在 PR description 显式 confirm 以下 7 项 D9 contract 全部 enforce：
1. SafeToolName + MCP metadata (A1 + A6)
2. AI SDK v5 + 自定义 `data-tool-consent` (A2)
3. `data-tool-consent` `argsPreview + argsHash` — raw args backend-private (A7)
4. Registry no silent system override (A5)
5. `data-elicitation` schema-validated input (§5)
6. Three-level authorization (§2)
7. PydanticAI as default runtime backbone (A3)

## Field-Level Disposition Table

### Current → Final-state mapping

| 当前 (legacy + v3.1 mixed) | Final state | Action |
|---|---|---|
| `StoredChatMessagePart.role: human/ai/system` | `UIMessage.role: user/assistant/system` | RENAME (ChatML alignment) |
| `StoredChatMessagePart.content: str` | `UIMessagePart{type: "text", text}` | REPLACE |
| `StoredChatMessagePart.references: List[Dict]` | `UIMessagePart{type: "data-citation", data: {cited_text, location}}` | REPLACE typed |
| `StoredChatMessagePart.urls: List[str]` | `UIMessagePart{type: "source-url", sourceId, url, title?}` | REPLACE typed |
| `StoredChatMessagePart.metadata` | — | **DELETE** (pure dead code, 0 consumer) |
| `StoredChatMessagePart.part_id` | UIMessagePart 自身 first-class | DELETE (redundant) |
| `AgentTimelineEventEnvelope.type: "tool.started/.finished"` | `tool-<name>` part with state lifecycle | REPLACE |
| `AgentTimelineEventEnvelope.type: "text.delta"` | `text-delta` part | REPLACE |
| `AgentTimelineEventEnvelope.user_activity` | `data-activity` transient part | REPLACE typed |
| `AgentArtifact{type: "answer"}` | aggregated `text` parts | INLINE (no separate artifact) |
| `AgentArtifact{type: "reference_bundle"}` | aggregated `data-citation` parts | INLINE |
| `AgentArtifact{type: "tool_result_summary"}` | `tool-<name>` part 的 `output` field | INLINE |

### Must-delete history残留清单

| Field | File:Line | 状态 | Disposition |
|---|---|---|---|
| `StoredChatMessagePart.metadata` | `aperag/chat/history/message.py:30` | accepted by helpers, never populated | **DELETE** |
| `StoredChatMessagePart.urls` | `aperag/chat/history/message.py:29` | legacy chat 用 | **REPLACE** by `source-url` |
| `StoredChatMessagePart.references` | `aperag/chat/history/message.py:28` | legacy chat 用 | **REPLACE** by `data-citation` |
| `ChatMessage.references / urls / metadata` | `aperag/domains/conversation/schemas.py:135,161,162` | agent path 不用 | **DELETE** |
| Entire `StoredChatMessagePart` parts model | `aperag/chat/history/message.py:10-31` | 仅 legacy non-agent 用 | **DEPRECATE** + 全量迁 UIMessage |
| `RedisChatMessageHistory` | `aperag/utils/history.py:103` | 0 current refs | **DELETE** |

## Migration path

按 Phase 8 hard-cut 哲学（earayu2 msg=78fdb6fc）— 零 backward compat shim：

1. **D8.1 Backend wire**: AI SDK v5 stream emitter（替代 `AgentTimelineEventEnvelope` wire）
2. **D8.2 Backend storage**: UIMessage data model + Redis store rewrite
3. **D8.3 Backend citations + tools**: typed `data-citation` (Anthropic-shape) + AI SDK v5 tool-call lifecycle
4. **D8.4 FE**: `@ai-sdk/react` 替换自研 SSE consumer
5. **D8.5 Backend non-agent**: legacy bot path 也迁到 UIMessage（hard-cut `StoredChatMessagePart`）
6. **D8.6 Cleanup**: 删除 D8 disposition 表中所有 DELETE 项
7. **D8.7 Doc**: 本文件作为 architect canonical SSoT

历史已存 Redis 数据：fresh DB strategy — 不做 data migration，accept data loss per earayu2 msg=78fdb6fc

## Implementation tasks（待 PM 拆）

D8.1-D8.7 各为独立 implementation task，可分 milestone 安排：
- **Milestone 1 (agent path)**: D8.1 + D8.2 + D8.3 + D8.4 — agent runtime + FE 完成切换
- **Milestone 2 (non-agent path + cleanup)**: D8.5 + D8.6 — legacy chat 迁移 + 残留清理
- **Doc**: D8.7 与每个 milestone 同步落 SSoT update

## Future scope (out of #68)

- **D9** (independent design lane): web-backend MCP-capable agent runtime boundaries — see [`agent-runtime-mcp-design.md`](./agent-runtime-mcp-design.md) (canonical landed by Phase 8 #72).
