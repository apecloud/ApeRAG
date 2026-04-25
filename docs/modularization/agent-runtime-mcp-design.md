# D9 — Web-Backend MCP-Capable Agent Runtime Boundaries (design-only)

**Status**: Design canonical — implementation deferred per PM lock
**Author**: 符炫炜 (总架构师), Weston counter-check
**Date**: 2026-04-25
**Scope**: 回答"web-backend MCP-capable agent 会额外要求哪些 runtime canonical 决策"，给 D8.3 tool lifecycle implementation 提前 lock 必要 contract。**不实施**。
**Cross-reference**: [`agent-message-protocol-design.md`](./agent-message-protocol-design.md) — D8 protocol/storage canonical SSoT.

## 1. MCP Server Registry / Discovery Boundary

### 1.1 三层 registry 架构 (per-system + per-bot + per-user)

| Tier | Source | Scope | Auth required |
|---|---|---|---|
| **System** | ApeRAG-built-in MCP servers (e.g., `aperag-knowledge-base`, `aperag-web-search`) | All users / all bots | None (deployed by admin) |
| **Bot** | Bot config 显式声明的 MCP server URLs | All users invoking that bot | Bot owner sets up |
| **User** | User-personal MCP servers (e.g., user's note-taking server) | Specific user only | User OAuth/token to remote MCP server |

**Resolution at agent invocation time**:
```
effective_servers(user, bot) = system_servers ∪ bot_servers ∪ user_servers
```

**冲突解决规则（D9 final, per A5）**:

1. **system namespace 是 reserved** — user/bot 不能 register 与 system namespace 重名的 tool
2. 如果 user/bot 尝试 register 与 system 重名的 tool，registry 必须 reject 该 server registration（或 quarantine 该 tool）
3. user-tool vs bot-tool 同名时：生成 distinct qualified safe name (per A1 + A6 collision rule)，**不做 override**
4. **admin override mechanism**: 仅 admin 可以显式 disable system tool 或者 alias 一个 user/bot tool 来 shadow system tool — 必须 audit-logged，**不能 silent**

**Implementation note** (D9 implementation owner 必须 enforce):
- registry insert 时检查 namespace conflict
- 生成 effective tool list 时按 system → bot → user 顺序，**遇到同名跳过非-system tool**（不 override）
- admin disable/alias 必须走 explicit API + audit log

### 1.2 Registry 存储 schema (D9 design only — D8 不实施)

新 domain `aperag/domains/mcp/`（D9 boundary）含 DB tables:

```sql
mcp_server (
  id PK,
  scope ENUM('system','bot','user'),
  scope_ref FK -- bot_id or user_id, NULL for system
  name TEXT NOT NULL, -- 'aperag-knowledge-base' / 'user-notes'
  url TEXT NOT NULL,
  auth_config JSON, -- per-server auth secrets (encrypted)
  enabled BOOL,
  ...
)

mcp_tool_cache ( -- cached MCP tools/list result, refreshed periodically
  server_id FK,
  tool_name TEXT, -- raw MCP tool name
  safe_name TEXT, -- provider-safe canonical name per A1+A6
  schema JSON, -- input schema from MCP
  description TEXT,
  ...
)
```

Registry 必须存 `(mcpServer, mcpToolName, safeName)` 三元组以保证 safeName → MCP identity 的 reverse lookup（per A6）。

**Discovery**: backend resolves `effective_servers(user, bot)` at turn start, calls `tools/list` on each (with cache TTL ~60s)，aggregates 成 unified tool set 给 agent runtime。

## 2. Tool Authorization Scope Model

### 2.1 三级权限

| Level | Check | Outcome |
|---|---|---|
| **Visibility** | `can_user_see_tool(user, tool)` | tool 是否出现在 agent 的 available_tools list（影响 LLM 是否知道该 tool 存在） |
| **Invocation** | `can_user_invoke_tool(user, tool, args)` | tool 是否可以**自动**被调用，无需用户确认 |
| **Consent gate** | `requires_consent(tool, args)` | tool 调用必须通过 user-explicit approval 才能执行 |

### 2.2 默认策略

| Tool category | Visibility | Invocation | Consent |
|---|---|---|---|
| Read-only (`search_collection`, `web_search`, `read_document`) | All | Auto | None |
| Side-effect (`write_file`, `send_email`, `database_modify`) | All | **Blocked unless consent** | Required per-call |
| Admin-only (`system_config`, `user_management`) | Admin only | Auto (admin) | None |
| User-personal (user MCP server tools) | User-self only | Per-server policy | Optional |

策略实现：每条 MCP tool 在 cache 时附带 risk classification（来自 MCP server 自身 declared metadata 或 ApeRAG admin override）。

### 2.3 D8.3 contract 要求

D8.3 tool lifecycle implementation 必须 enforce：
- **Visibility filter**: 不 visible 的 tool 不进入 agent 的 system prompt tool list
- **Consent gate hook**: agent 决定调用 consent-required tool 时，不直接发 `tool-input-start`，而是先发 `data-tool-consent` part，等待用户 decision；approved 后发 `tool-input-start` continue
- **Invocation block**: agent 调用 blocked tool 时直接发 `tool-output-error` (AI SDK v5 strict spec, post D8.0c+ #89) with `errorText: "Tool invocation denied: requires user consent"`

## 3. Tool Consent UI in UIMessage Parts

### 3.1 `data-tool-consent` part shape (D9 final, per A7)

> **Updated by A7** — supersedes the earlier proposal that the part carry full raw `args`. Raw args may contain user tokens / secrets / sensitive payload; persisting full args in UIMessage history is a privacy/security risk. The wire now carries a redacted preview + sha256 hash; raw args stay backend-private.

```typescript
{ type: "data-tool-consent", data: {
    toolCallId: string,                 // stable across consent → tool-input-start
    toolName: string,                   // SafeToolName per A1+A6
    metadata: { mcpServer?: string, mcpToolName?: string },
    argsPreview: string,                // redacted human-readable preview，max 500 chars
                                        // e.g. "write_file(path='/tmp/...truncated...', content=<redacted>)"
    argsHash: string,                   // sha256 of canonical-serialized raw args，用于 audit + correlation
    risk: "writes_user_data" | "calls_external_api" | "modifies_system" | "admin_only",
    requestedAt: string,                // ISO timestamp
    state: "pending" | "approved" | "denied" | "expired"
}}
```

**Wire 不暴露 raw args**。Backend 持有 raw args（in agent runtime memory / short-TTL Redis），等待 user consent decision，approved → 用 raw args 调 tool，denied → 丢弃 raw args + emit tool-output-error。

**Persisted**（不是 transient — consent decision 是 audit-trail relevant）。

**`argsPreview` 生成规则** (D9 implementation owner 必须 enforce):
- 字符串 args > 100 chars 截断为 `"<first-50-chars>...<truncated>..."`
- 任何被 risk classification 标 `secret` 的 field（如 password / api_key / token） → `<redacted>`
- 嵌套 object 展开 1 level，更深 → `{...}`
- 总长度 cap 500 chars

**`argsHash`**: `sha256(JSON.stringify(args, sortedKeys))` — 用于 audit log + 防止 replay attack。

Frontend 渲染 consent UI 用 `toolName + argsPreview + risk` 给用户决策；不展示 raw args。如果用户想看 raw args（debug），admin-only endpoint `GET /api/v2/agent/turns/{id}/consent/{toolCallId}/raw-args` 可单独取（with audit log）。

### 3.2 Consent flow

```
1. agent → backend: "I want to call tool-aperag_fs_write_file with args X"
2. backend authorization check: requires_consent
   → emit data-tool-consent part state="pending" (with argsPreview + argsHash)
3. FE renders inline approve/deny prompt (toolName + argsPreview + risk only)
4. user → backend (POST /api/v2/agent/turns/{id}/consent):
   { toolCallId, decision: "approved" | "denied" }
5. backend updates data-tool-consent part state → "approved" or "denied"
6. backend → agent runtime:
   - approved: continue with tool-input-start lifecycle (raw args used internally)
   - denied: emit tool-output-error with errorText "denied by user"  (AI SDK v5 strict spec, post D8.0c+ #89)
```

### 3.3 Timeout policy

Consent pending > 5 min → `state: "expired"`，agent 收到 timeout 信号 → 跳过该 tool call 继续 reasoning。

## 4. Sampling — Backend-internal (NOT surfaced to wire)

MCP `sampling/createMessage` = MCP server requests host run an LLM call internally。这是 server↔host mechanic，不是 user-facing flow。

**Decision**: backend handles sampling 完全内部 — backend agent 收到 sampling request → recursive LLM call → return result to MCP server。流向 user 的只有最终 agent 输出（普通 text-delta parts）。

**Optional debug visibility**: 用现有 `data-activity` part with `intent: "sampling_in_progress"` 表达 — 仅做 UI 状态指示，不暴露 sampling 细节。

## 5. Elicitation — Surfaced via `data-elicitation` part

MCP `elicitation` = server asks user for input mid-tool (e.g., "specify which file path to write")。这是 user-facing flow，**必须 surface 到 wire**。

### 5.1 `data-elicitation` part shape

```typescript
{ type: "data-elicitation", data: {
    elicitationId: string,        // stable for response correlation
    serverName: string,           // MCP server requesting input
    prompt: string,               // human-readable question
    schema: JsonSchema,           // input validation schema
    state: "pending" | "answered" | "cancelled"
}}
```

**Persisted** (non-transient — part of message history)。

### 5.2 Elicitation flow

```
1. MCP server → tool-input → tool execution → mid-execution elicitation request
2. backend agent runtime receives elicitation → emit data-elicitation part
3. FE renders form based on schema
4. user → backend (POST /api/v2/agent/turns/{id}/elicit/{elicitationId}): { value }
5. backend resumes tool execution with user input
6. tool completes → tool-output-available
```

## 6. D8.3 Tool Lifecycle Interface Contract

D8.3 实施时 wire emitter 必须支持以下 lifecycle，**否则后续 retrofit 成本高**：

### 6.1 Standard lifecycle (no consent, no elicitation)

```
text-start (agent reasoning) → text-delta → text-end
tool-input-start { toolCallId, toolName: "tool-aperag_knowledge_base_search_collection",
                   metadata: { mcpServer: "aperag-knowledge-base", mcpToolName: "search_collection" } }
tool-input-delta { toolCallId, inputTextDelta: '{"q": "...' }
tool-input-available { toolCallId, input: { q: "..." } }
tool-output-available { toolCallId, output: { results: [...] } }
[ optional: data-citation parts × N ]
text-start (agent answer) → text-delta → text-end
finish-step / finish
```

### 6.2 Consent-gated lifecycle

```
text-start (agent reasoning) → ...
data-tool-consent { toolCallId, toolName, metadata, argsPreview, argsHash, risk, state: "pending" }
[ wait for user consent endpoint ]
data-tool-consent { toolCallId, ..., state: "approved" | "denied" }
[ if approved: continue with tool-input-start lifecycle ]
[ if denied: tool-output-available with errorText ]
```

### 6.3 Elicitation lifecycle

```
tool-input-start → tool-input-available
[ tool executes, hits elicitation ]
data-elicitation { elicitationId, prompt, schema, state: "pending" }
[ wait for user elicit endpoint ]
data-elicitation { ..., state: "answered" }
[ tool resumes ]
tool-output-available
```

## 7. PydanticAI as Default Runtime Backbone (per A3)

Per PM msg=e6c1b252 / 09ab99a2 + Weston msg=8bf09de6 + Weston PydanticAI 调研 (msg=6fb9104a)。

**PydanticAI 的定位**: web-backend agent runtime **default backbone**（不是协议替代）。

**Compatibility 评估**:

| Aspect | PydanticAI 支持度 | D8/D9 复用度 |
|---|---|---|
| **UI integration**: backend events → wire | ✅ `UIAdapter` + 内置 `VercelAIAdapter` / AG-UI adapter | ✅ 直接复用 D8 wire schema，无需重做 |
| **MCP client/server**: tool registry, sampling/elicitation | ✅ 完整支持 (per Weston PydanticAI 调研) | ✅ 直接覆盖 D9 §1, §4, §5 |
| **Toolset / tool prefixing** | ✅ 原生支持 toolset 和 prefix | ✅ 与 D9 A1+A6 SafeToolName + metadata 兼容 |
| **Tool approval (v5/v6)** | ⚠️ Vercel AI tool approval 需 AI SDK v6 | ⚠️ ApeRAG 选 v5 → consent 走 ApeRAG 自定义 part，PydanticAI 通过 callback hook 接入 |
| **Messages/history (model-call)** | ✅ 内部 history 支持 | ❌ 不取代 D8 UIMessage at-rest — UIMessage 是 FE-facing durable UI transcript，PydanticAI history 是 model-call internal |
| **Deferred tools / durable execution / agent graphs** | ✅ 完整支持 | ⚠️ D9 design 不覆盖 — 长 tool calls / 后台 continue 留 D10 (execution reliability) |

**理由 (PydanticAI 优于自建 MCP host)**:
- ✅ MCP support 完整覆盖 D9 §1 (registry) / §4 (sampling) / §5 (elicitation)
- ✅ UIAdapter / VercelAIAdapter 与 D8 wire schema 直接兼容
- ✅ Toolset / prefix 与 A1+A6 SafeToolName 兼容
- ⚠️ ApeRAG-specific consent (D9 §3) 通过 callback hook 接入（不是 native，但 well-supported pattern）
- ⚠️ ApeRAG-specific authorization (D9 §2) 也通过 hook 接入
- ✅ deferred tools / durable execution 即便 D9 不覆盖，未来 D10 时可以直接用 PydanticAI 现成支持

**自维护成本对比**：自建 MCP host 需要从 0 实现 MCP client + tool dispatch + sampling/elicitation event loop；PydanticAI 这些都是 free。

**D8.3 实施 owner 决定 final lock**（评估 hooks 接入复杂度后），但 D9 design **明确 default = PydanticAI**。

## A1. Tool Naming — Provider-Safe Canonical + MCP Metadata

> **Refined from earlier proposal that wire `tool-<name>` directly carry MCP dotted namespace.** Many model providers require tool/function names in `[a-zA-Z0-9_-]`, so the wire name must be a provider-safe canonical name; MCP server/tool identity is preserved as separate metadata. See A6 below for the collision-resistant generation rule.

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

**Naming convention examples**:
- ApeRAG built-in (system): `tool-aperag_knowledge_base_search_collection` + `metadata: { mcpServer: "aperag-knowledge-base", mcpToolName: "search_collection" }`
- 用户 personal MCP server: `tool-user_notes_create_note` + `metadata: { mcpServer: "user-notes", mcpToolName: "create_note" }`

This means D8 doc §2.4 — see [`agent-message-protocol-design.md`](./agent-message-protocol-design.md) §2.4 (retroactively updated by D8.0b to match A1 + A6).

## A2. AI SDK v5 + 自定义 `data-tool-consent` (lock v5, NOT v6)

| 选项 | Pros | Cons |
|---|---|---|
| **AI SDK v5 + 自定义 `data-tool-consent`** | v5 已稳定 release；不依赖 alpha；自由定义 ApeRAG-specific consent shape | 需要自维护 consent part schema + FE 适配 |
| **AI SDK v6 native tool approval** | 无需自定义 consent part；FE 用 SDK built-in approval UI | v6 是较新 release（截至 2026-04 可能尚未 GA）；ApeRAG 绑定 v6 demo path |

**D9 lock**: **v5 + 自定义 `data-tool-consent`**（per D9 §3, with A7 redacted preview）。

理由：
1. v6 的 tool approval shape 即便 GA，仍可能与 ApeRAG `risk` classification + `consent_state` 流程不完全对齐 — 自定义 part 给 ApeRAG 自己的边界
2. v5 现成稳定 + FE library 工具链成熟
3. 未来如果 v6 GA + ApeRAG 选择 migrate，consent part shape 可以平滑映射到 v6 native approval（不会 wire schema rewrite）

**D8.3 实施 owner 必须显式 confirm 选择 v5**（PR description 中），不允许自行升级到 v6。

## A4. D8.3 前置 D9 decisions checklist

D8.3 实施前 D9 必须 lock 的决策点（all locked here）：

| Decision | D9 lock |
|---|---|
| Tool naming convention | A1 + A6: `tool-<SafeToolName>` + `metadata: {mcpServer, mcpToolName}` + collision hash suffix |
| AI SDK version | A2: v5 + 自定义 `data-tool-consent` |
| Consent part schema | §3.1 + A7: `argsPreview` + `argsHash`, raw args backend-private |
| Elicitation part schema | §5.1: `data-elicitation` shape |
| Sampling visibility | §4: backend-internal NOT surfaced |
| Authorization model | §2: 三级权限 (visibility / invocation / consent) |
| Registry override policy | §1.1 + A5: system namespace reserved，no silent override |
| Runtime backbone (default) | §7 + A3: PydanticAI |

D8.3 实施 PR description 必须显式 confirm 这 8 项全部 covered。

## A5. Registry Override → No Silent Override on system namespace

See §1.1 above — final canonical 已 inline。

Key invariant: **system namespace 是 reserved**，user/bot 不能 silent shadow；admin override 必须 explicit + audit-logged。

## A6. SafeToolName Collision Rule (stable unique with hash suffix)

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

**Properties**:
- **Stable**: same `(mcpServer, mcpToolName)` 始终生成同一 safe name
- **Unique**: collision via stable `sha256` hash suffix
- **Inspectable**: double underscore `__` 是明显的 collision marker
- **Provider-safe**: 仅 `[a-zA-Z0-9_-]`

Registry 必须存 `(mcpServer, mcpToolName, safeName)` 三元组以保证 reverse lookup（safeName → MCP identity）。

## A7. `data-tool-consent` argsPreview + argsHash

See §3.1 above — final canonical 已 inline。

Key invariant: wire/at-rest **不暴露 raw args**；backend 私有持有；audit-trail 通过 `argsHash` 关联；FE 仅渲染 `toolName + argsPreview + risk`。

## D9 final canonical deliverables (locked per Weston msg=f56b2ae1 final ack)

| § | Topic | Status |
|---|---|---|
| §1 + A5 | MCP server registry — three-tier (system/bot/user) + **no silent override on system namespace** | ✅ locked |
| §2 | Three-level authorization (visibility/invocation/consent) | ✅ locked |
| §3 + A7 | `data-tool-consent` part with **`argsPreview` + `argsHash` (raw args backend-private)** | ✅ locked |
| §4 | Sampling — backend-internal, NOT surfaced | ✅ locked |
| §5 | `data-elicitation` part — surfaced with schema-validated input | ✅ locked |
| §6 | D8.3 tool lifecycle interface contract (含 consent + elicitation) | ✅ locked |
| §7 + A3 | PydanticAI evaluation — runtime candidate (default backbone) | ✅ locked |
| A1 + A6 | SafeToolName naming with collision-resistant hash suffix | ✅ locked |
| A2 | AI SDK v5 + 自定义 consent (lock v5, NOT v6) | ✅ locked |
| A4 | D8.3 前置 8 decisions checklist | ✅ locked |

## Out of D9 scope

- **MCP host implementation 选型 lock**（PydanticAI vs 自建）— D9 default = PydanticAI；final lock 由 D8.3 implementation owner 评估 hook 接入复杂度后决定
- **Per-bot MCP server 配置 UI**（admin / bot owner 如何 add server）— 留 D10 (UI design) 后续任务
- **MCP `roots`** (filesystem path scoping) — Phase 9 候选
- **Multi-agent orchestration** (agent calls another agent) — Phase 9 候选
- **Deferred tools / durable execution / agent graphs** (long tool calls / 后台 continue) — 留 D10 (execution reliability)

## D8.3 implementation contract (必须前置 enforce)

D8.3 (tool lifecycle backend) 实施 owner 必须在 PR description 显式 confirm 以下 contract 全部 covered：

1. SafeToolName + MCP metadata (A1 + A6)
2. AI SDK v5 + 自定义 consent (A2)
3. `data-tool-consent` `argsPreview + argsHash` — raw args backend-private (A7)
4. Registry no silent system override (A5)
5. `data-elicitation` schema-validated input (§5)
6. Three-level authorization (§2)
7. PydanticAI as default backbone (A3)
