---
title: Agent Runtime V3 Design
description: Detailed design for ApeRAG Agent Runtime V3, including the Turn, TimelineEvent, Artifact, SSE contract, and migration boundary
keywords: Agent Runtime, SSE, Turn, TimelineEvent, Artifact, PydanticAI, MCP
position: 4
---

# ApeRAG Agent Runtime V3 Design

## 1. Background and Goal

The current ApeRAG agent chat path is built on top of `mcp-agent`. It has proven that the system can work, but it is not a good long-term fit for ApeRAG's product and delivery goals.

The unstable part is no longer the business API surface. The weakest layer is the runtime glue:

- event dispatch
- streaming output
- frontend/backend message grammar
- session cache shape
- tool-result propagation

These are coupled too tightly to third-party runtime internals, which directly turns into support cost, debugging cost, and deployment risk in private environments.

This design therefore does **not** aim to find a more powerful agent framework. Its purpose is to rebuild ApeRAG's `agent product layer` around private deployment requirements:

1. private-deployment friendly
2. simple and reliable
3. low maintenance cost
4. minimal post-delivery support burden

## 2. Final Decision

Agent Runtime V3 makes the following decisions official:

1. `mcp-agent` will not remain the long-term runtime core
2. `FastAPI + FastMCP + existing business API/provider integrations` remain the stable business surface
3. `SSE` becomes the only primary transport
4. the product contract is rebuilt around `Turn + TimelineEvent + Artifact`
5. `PydanticAI adapter` is used as the Phase 1 runtime implementation
6. the system keeps a long-term path to collapse into a self-owned thin orchestration layer
7. the main runtime will not be rebuilt around `Vercel AI SDK`, `OpenAI Agents SDK`, or `LangGraph`

External libraries may help with implementation, but they do not define the product contract.

## 3. Design Principles

### 3.1 Private deployment first

All decisions optimize for:

- deterministic defaults
- diagnosable failures
- clear rollback boundaries
- minimal hidden assumptions
- minimal compatibility baggage

### 3.2 ApeRAG owns the contract

The following are owned by ApeRAG itself:

- API entrypoints
- SSE event protocol
- user-visible status vocabulary
- TimelineEvent schema
- Artifact schema
- history commit policy

Third-party runtimes must adapt to this contract, not the other way around.

### 3.3 Final answer and process events must be separated

The final answer, process timeline, references, and tool results are no longer packed into one assistant message.

The layering is:

- `answer` is an answer artifact
- `timeline` is a process-event stream
- `references` are separate artifacts
- `tool result` is surfaced via summary events and artifact references

### 3.4 Phase 1 stays intentionally narrow

Phase 1 supports only:

- single agent
- serial tool loop
- single MCP server view
- multiple internal loops inside a single turn

Phase 1 explicitly does not support:

- multi-agent
- parallel tool fan-out
- workflow/graph orchestration
- long-running orchestration

## 4. Core Object Model

## 4.1 Turn

A `Turn` is one complete agent execution for one user query.

It is important to clarify that a turn is **not** a single-step answer. A turn may contain many rounds of:

- thinking
- web search
- tool calls
- result reading
- internal reasoning

The turn is the outer execution boundary.

### 4.1.1 Suggested fields

```text
schema_version
turn_id
chat_id
user_id
request_id
client_idempotency_key
status
input_text
model_profile
started_at
finished_at
error_code
error_message
answer_artifact_id
reference_bundle_artifact_id
timeline_cursor
```

### 4.1.2 State machine

```text
queued -> running -> completed
queued -> running -> failed
queued -> running -> cancelled
```

### 4.1.3 Hard rules

1. one turn must have exactly one final answer artifact
2. one turn must never execute twice
3. one `chat_id + client_idempotency_key` must not create multiple valid turns

## 4.2 TimelineEvent

A `TimelineEvent` is the standardized event stream for one turn.

It serves three roles:

- frontend timeline rendering model
- SSE transport model
- replay and diagnosis model

It is **not** a raw debug-log dump.

### 4.2.1 Required fields

```text
schema_version
event_id
turn_id
sequence
timestamp
type
label
status
actor
data
```

### 4.2.2 Hard rules

1. `sequence` must be strictly monotonic inside one turn
2. the frontend must not infer ordering from timestamps
3. `actor` is limited to `agent | tool | system`
4. `data` carries only the minimum payload
5. the timeline must be replayable

### 4.2.3 Event types

Phase 1 event types are:

- `turn.started`
- `agent.state.changed`
- `tool.started`
- `tool.progress`
- `tool.finished`
- `external_action.started`
- `external_action.finished`
- `text.delta`
- `artifact.created`
- `turn.completed`
- `turn.failed`
- `turn.cancelled`
- `heartbeat`

### 4.2.4 Layering rules

- `tool.*` is for the standard tool loop
- `external_action.*` is only for user-visible external actions such as `web_search`
- not every internal runtime step should appear in the timeline

## 4.3 Artifact

An `Artifact` is a persisted, re-readable, reusable object.

### 4.3.1 Suggested artifact types

- `answer`
- `reference_bundle`
- `tool_result_summary`
- `search_result_summary`
- `error_summary`

### 4.3.2 Suggested fields

```text
schema_version
artifact_id
turn_id
artifact_type
created_at
summary
storage_ref | payload
```

### 4.3.3 Hard rules

1. the stream must not carry large bodies
2. the stream only carries summaries, artifact ids, and minimal metadata
3. references must be materialized as a separate artifact

## 5. User-visible status vocabulary

The frontend must not expose raw runtime event names.

Phase 1 user-facing status vocabulary is fixed to:

- `Thinking`
- `Searching`
- `Calling Tool`
- `Reading Result`
- `Streaming Answer`
- `Completed`
- `Failed`

This keeps the UI stable even if backend internals evolve later.

## 6. API and Transport Design

## 6.1 Primary transport

The primary transport is `SSE` only.

The system should not keep a long-lived `WebSocket + SSE` dual stack.

## 6.2 API surface

### 6.2.1 Create a turn

```text
POST /api/v2/agent/chats/{chat_id}/turns
```

Suggested request fields:

- `query`
- `context`
- `model_profile`
- `client_idempotency_key`

Suggested response fields:

- `turn_id`
- `status`
- `stream_url`

### 6.2.2 Subscribe to turn events

```text
GET /api/v2/agent/chats/{chat_id}/turns/{turn_id}/events
```

Response type:

```text
Content-Type: text/event-stream
```

### 6.2.3 Get turn snapshot

```text
GET /api/v2/agent/chats/{chat_id}/turns/{turn_id}
```

Used for:

- page refresh recovery
- fallback after failed SSE reconnect
- diagnosis

### 6.2.4 Cancel a turn

```text
POST /api/v2/agent/chats/{chat_id}/turns/{turn_id}/cancel
```

### 6.2.5 Get an artifact

```text
GET /api/v2/agent/artifacts/{artifact_id}
```

### 6.2.6 OpenAI-compatible adapter

```text
POST /v1/chat/completions
```

This endpoint is a compatibility adapter for OpenAI-shaped clients. It is not
the primary UI contract. The implementation must translate each request into an
Agent Runtime V3 turn and then format the result as either:

- `chat.completion` JSON when `stream=false`
- `text/event-stream` `chat.completion.chunk` frames when `stream=true`

The adapter contract is:

- `bot_id` is required as a query parameter
- `chat_id` is optional; if omitted, the backend creates and later deletes an
  ephemeral chat
- `language` is optional and defaults to `en-US`
- `Idempotency-Key` / `X-Idempotency-Key` maps to
  `client_idempotency_key`

## 6.3 Idempotency and reconnect

### 6.3.1 Idempotency

- `POST turn` must support a client idempotency key
- repeated requests with the same `chat_id + client_idempotency_key` must not create multiple turns
- one turn must never execute twice

### 6.3.2 SSE reconnect

- reconnect should use `Last-Event-ID` or an explicit offset
- if the event buffer has expired:
  1. the client fetches the turn snapshot
  2. the client resumes from the newest available cursor

## 6.4 Heartbeat, backpressure, and timeout

The SSE layer must define:

- heartbeat behavior
- event buffer limits
- delta merge policy
- overload summarization/truncation behavior

It must also distinguish:

- single tool timeout
- total runtime timeout for one turn
- stream idle timeout

## 7. Permission Boundary

The new runtime entrypoint must re-check permissions explicitly. It must not inherit assumptions from the old WebSocket path.

Every turn creation must validate:

- chat ownership
- collection/file context visibility
- tool visibility scope

Artifact retrieval endpoints must also enforce permissions.

## 8. Storage Design

## 8.1 Redis responsibility

Redis handles only short-lived runtime and stream recovery state:

- `turn runtime state`
- `stream cursor`
- `transient event buffer`
- `in-flight text buffer`

Redis no longer owns:

- the old message grammar
- the product-level message contract
- the long-term history representation

## 8.2 DB / persistent responsibility

Persistent storage holds:

- `conversation_turn`
- `timeline_event` (at least key events)
- `artifact`
- `reference_bundle`
- `error_summary`

The timeline must be replayable after the stream ends.

## 8.3 History commit policy

The final history is not written token-by-token.

Instead:

1. only transient runtime state is updated during streaming
2. the standardized turn record is committed only after `done` or explicit `error`

This prevents half-written history after cancellation, reconnect, or rollback.

## 9. Frontend Experience Model

The frontend should no longer treat one assistant bubble as the carrier of everything.

The recommended layout is:

1. `Turn Header`
2. `Timeline`
3. `Final Answer Panel`
4. `References Panel`
5. `Diagnostics Drawer`

Where:

- Timeline shows process only
- Final Answer Panel shows the final answer only
- References Panel shows sources only
- Diagnostics Drawer is expandable and opt-in

## 10. Runtime Path

## 10.1 Phase 1: PydanticAI adapter

Phase 1 uses `PydanticAI` as the runtime implementation because it lowers implementation cost for:

- single-turn internal loops
- tool calling
- provider calling
- state mapping

But it does not define:

- public API
- timeline schema
- artifact schema
- history commit policy

## 10.2 Long-term path

If the `PydanticAI adapter` stays stable and low-maintenance, it may remain.

If its runtime behavior still constrains ApeRAG too much, the internals can later be replaced by a self-built thin orchestration layer.

Because the contract is already separated, that later replacement changes the implementation only, not the product boundary.

## 11. Replacement Boundary

This rewrite:

- keeps `FastAPI`, `FastMCP`, business APIs, provider integrations, and business entities
- replaces `mcp-agent runtime glue`, the old WebSocket grammar, the old Redis message shape, the old frontend rendering model, and the old tool-result/event pushback mechanism

That means business value is preserved while the most fragile product/runtime coupling is rebuilt cleanly.

## 12. Migration and Rollback Principles

### 12.1 Migration

- a short-lived feature flag is allowed for migration safety
- a long-lived dual stack is not allowed
- once the new path is stable, the old WebSocket grammar and old runtime glue should be removed

### 12.2 Rollback conditions

Rollback is allowed if:

- SSE is unstable behind enterprise proxies
- timeline replay/reconnect is unreliable
- turn/history/artifact compatibility is broken
- provider/tool failures are not diagnosable

### 12.3 Rollback requirements

After rollback:

- history must remain readable
- turn records must not become orphaned
- artifacts must remain traceable

## 13. Phase 1 Task Outline

Phase 1 should focus on the minimum viable end-to-end path rather than the final long-term shape.

Suggested tasks:

1. create `aperag/agent_runtime/`
2. define `Turn / TimelineEvent / Artifact` schemas
3. implement `TurnService`
4. implement `EventService`
5. implement `ArtifactService`
6. implement `HistoryWriter`
7. define `AgentRuntime`
8. implement `PydanticAIRuntime`
9. implement the MCP client adapter
10. implement `SSE StreamEmitter`
11. add v2 agent APIs
12. build the new timeline frontend
13. build answer/references/diagnostics panels
14. implement snapshot recovery and cancel
15. add contract-level E2E coverage

## 14. Acceptance Criteria

Phase 1 should satisfy:

1. the new API can create turns, stream events, fetch snapshots, fetch artifacts, and cancel turns
2. a single turn can complete multiple search/tool/thinking loops
3. the timeline is reconnectable and replayable
4. the final answer and process events are fully separated
5. history commit policy leaves no half-written final records
6. `mcp-agent` is no longer on the primary chat execution path

## 15. Final Decision

Agent Runtime V3 makes the following official:

- stop patching `mcp-agent`
- stop patching the old WebSocket grammar
- rebuild the runtime contract around `Turn + TimelineEvent + Artifact + SSE`
- use `PydanticAI adapter` for Phase 1
- let implementation follow this document, while architecture remains responsible for the contract boundary and long-term direction

In one sentence:

This is not just a runtime swap. It is a product-layer rebuild of ApeRAG's agent runtime for private, low-support delivery.
