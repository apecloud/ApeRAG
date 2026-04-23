---
title: Agent Runtime V3 设计方案
description: ApeRAG Agent Runtime V3 的详细设计文档，定义新的 Turn、TimelineEvent、Artifact、SSE 协议与迁移边界
keywords: Agent Runtime, SSE, Turn, TimelineEvent, Artifact, PydanticAI, MCP
position: 4
---

# ApeRAG Agent Runtime V3 设计方案

## 1. 背景与目标

当前 ApeRAG 的 agent chat 主链路基于 `mcp-agent` 运行。它已经证明“能工作”，但对 ApeRAG 的长期产品目标并不合适：

- 当前最脆弱的部分不是业务 API，而是 runtime 胶水层
- 事件分发、流式输出、前后端消息格式、会话缓存、工具结果回推都与第三方 runtime 的内部语义耦合过深
- 这类耦合会直接转化成私有化交付后的维护成本、排障成本和答疑成本

因此，这次设计的目标不是“换一个更强的 agent 框架”，而是：

1. 重做 ApeRAG 的 `agent 产品层`
2. 保持 ApeRAG 的 `业务能力面` 稳定
3. 建立一个更适合 `私有化部署`、`简单可靠`、`低维护成本`、`低答疑成本` 的单 agent runtime

本设计的约束前提如下：

- 优先面向 `私有化交付`
- 假设客户普遍缺乏强运维能力
- 功能可以保守，但默认行为必须稳健
- agent 层允许整体切换，不为旧 WebSocket grammar、旧 Redis message shape、旧 message part 语义做长期兼容

## 2. 设计结论

Agent Runtime V3 的正式结论如下：

1. `mcp-agent` 不再作为 ApeRAG 的长期核心 runtime
2. `FastAPI + FastMCP + 现有业务 API/provider` 继续作为稳定业务面保留
3. 主传输协议统一为 `SSE`
4. 核心产品契约统一为 `Turn + TimelineEvent + Artifact`
5. 第一阶段 runtime 实现使用 `PydanticAI adapter`
6. 长期保留进一步收敛到 `自研薄编排层` 的空间
7. 不把核心 runtime 迁移到 `Vercel AI SDK`、`OpenAI Agents SDK` 或 `LangGraph`

这意味着：外部库只负责帮助实现，不再定义 ApeRAG 的产品语义。

## 3. 设计原则

### 3.1 私有化优先

所有设计优先满足以下诉求：

- 默认配置可运行
- 失败行为可诊断
- 升级和回退边界清晰
- 少依赖隐式前提
- 少引入双栈和兼容包袱

### 3.2 产品契约归 ApeRAG 自己拥有

新的对外契约由 ApeRAG 自己定义，包括：

- API 入口
- SSE 事件流
- 前端可见状态词表
- TimelineEvent schema
- Artifact schema
- History commit policy

第三方 runtime 只能适配这套契约，不能反向决定它。

### 3.3 最终回答与过程事件彻底分离

最终 answer、运行过程、references、tool result 不再混塞进一条 assistant message。

分层原则如下：

- `answer` 是 answer artifact
- `timeline` 是过程事件流
- `references` 是独立 artifact
- `tool result` 通过摘要事件 + artifact 引用暴露

### 3.4 明确缩小 Phase 1 能力边界

Phase 1 只支持：

- 单 agent
- 串行 tool loop
- 单 MCP server 视图
- 单个 turn 内多轮 internal loop

Phase 1 不支持：

- 多 agent
- 并发 tool fan-out
- workflow/graph orchestration
- 长任务编排

## 4. 核心对象模型

## 4.1 Turn

`Turn` 表示一次用户 query 对应的一次完整 agent execution。

需要特别澄清的是：

- 一个 turn 不是“一步回答”
- 一个 turn 内允许多次 thinking、多次 web search、多次 tool call、多次读取结果和多轮内部推理
- turn 是外层执行边界，不是内部 loop 次数的限制器

### 4.1.1 设计目标

Turn 统一承担以下职责：

- 幂等边界
- 取消边界
- 超时边界
- 恢复边界
- 最终历史提交边界
- 回放与评测边界

### 4.1.2 建议字段

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

### 4.1.3 状态机

```text
queued -> running -> completed
queued -> running -> failed
queued -> running -> cancelled
```

### 4.1.4 硬约束

1. 一个 turn 只允许一个最终 `answer_artifact_id`
2. 同一个 turn 不允许被执行两次
3. 同一个 `chat_id + client_idempotency_key` 只能创建一个有效 turn

## 4.2 TimelineEvent

`TimelineEvent` 表示 turn 执行过程中的标准化事件流。

它既是：

- 前端时间线展示模型
- SSE 传输模型
- 诊断与回放模型

但它不是：

- runtime 原始内部日志 dump
- debug event 任意透出层

### 4.2.1 必备字段

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

### 4.2.2 硬约束

1. `sequence` 在 turn 内必须严格单调递增
2. 不允许前端按时间戳猜顺序
3. `actor` 只允许：`agent | tool | system`
4. `data` 只携带最小必要 payload
5. timeline 必须支持重放

### 4.2.3 事件类型

Phase 1 标准事件类型定义如下：

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

### 4.2.4 事件分层约束

- `tool.*` 用于标准 tool loop
- `external_action.*` 只用于少数用户可感知外部动作，例如 `web_search`
- 不允许把所有内部小步骤都提升到 timeline 层

## 4.3 Artifact

`Artifact` 表示需要持久化、可重读、可复用、可排障的大对象。

### 4.3.1 建议类型

- `answer`
- `reference_bundle`
- `tool_result_summary`
- `search_result_summary`
- `error_summary`

### 4.3.2 建议字段

```text
schema_version
artifact_id
turn_id
artifact_type
created_at
summary
storage_ref | payload
```

### 4.3.3 硬约束

1. stream 中不直接推送大正文
2. stream 只推送摘要、artifact id 和必要元数据
3. references 必须 materialize 成独立 artifact

## 5. 用户可见状态词表

前端不直接显示 runtime 原始事件名，而是统一映射成稳定的用户可见状态词表。

Phase 1 固定为：

- `Thinking`
- `Searching`
- `Calling Tool`
- `Reading Result`
- `Streaming Answer`
- `Completed`
- `Failed`

这样做的目的有两个：

1. 避免后端内部状态演进反复影响前端展示
2. 降低用户理解成本与答疑成本

## 6. 前后端协议设计

## 6.1 主传输协议

主链路只保留 `SSE`。

不长期保留 `WebSocket + SSE` 双栈共存。

## 6.2 API 设计

### 6.2.1 创建 turn

```text
POST /api/v2/agent/chats/{chat_id}/turns
```

请求体建议包含：

- `query`
- `context`
- `model_profile`
- `client_idempotency_key`

响应建议包含：

- `turn_id`
- `status`
- `stream_url`

### 6.2.2 订阅 turn 事件流

```text
GET /api/v2/agent/chats/{chat_id}/turns/{turn_id}/events
```

返回类型：

```text
Content-Type: text/event-stream
```

### 6.2.3 获取 turn snapshot

```text
GET /api/v2/agent/chats/{chat_id}/turns/{turn_id}
```

用于：

- 刷新页面恢复
- SSE 重连失败时兜底
- 调试和诊断

### 6.2.4 取消 turn

```text
POST /api/v2/agent/chats/{chat_id}/turns/{turn_id}/cancel
```

### 6.2.5 获取 artifact

```text
GET /api/v2/agent/artifacts/{artifact_id}
```

### 6.2.6 OpenAI-compatible adapter

```text
POST /v1/chat/completions
```

这个接口是给 OpenAI 形状客户端使用的兼容 adapter，不是前端主 UI
contract。实现必须把每个请求转换成 Agent Runtime V3 turn，再按
OpenAI 形状格式化输出：

- `stream=false` 时返回 `chat.completion` JSON
- `stream=true` 时返回 `text/event-stream` 的 `chat.completion.chunk`
  帧

adapter contract 固定为：

- `bot_id` 是必填 query 参数
- `chat_id` 可选；不传时后端创建并在请求结束后删除 ephemeral chat
- `language` 可选，默认 `en-US`
- `Idempotency-Key` / `X-Idempotency-Key` 映射为
  `client_idempotency_key`

## 6.3 幂等与重连

### 6.3.1 幂等策略

- `POST turn` 必须支持客户端幂等键
- 同一 `chat_id + client_idempotency_key` 下重复请求不得创建多个 turn
- 同一个 turn 一旦创建成功，就不允许重复执行

### 6.3.2 SSE 重连策略

- 默认支持 `Last-Event-ID` 或 offset 续传
- 如果服务端 event buffer 已过期，则：
  1. 客户端先拉 `turn snapshot`
  2. 再从当前最新游标继续订阅

## 6.4 心跳、背压与超时

SSE 层必须定义以下行为：

- heartbeat 事件
- event buffer 上限
- delta 合并策略
- 过载时摘要/截断策略

同时区分以下超时：

- 单 tool timeout
- 单轮 total runtime timeout
- stream idle timeout

## 7. 权限与安全边界

新 runtime 入口必须重新做完整鉴权，不能依赖旧 WebSocket 路径中的隐式前提。

每个 turn 创建时必须重新校验：

- `chat_id` ownership
- collection/file context visibility
- tool 可见范围

artifact 读取接口也必须重新做权限校验，防止通过 artifact id 越权读取。

## 8. 存储设计

## 8.1 Redis 职责

Redis 只负责短期运行态与流式恢复：

- `turn runtime state`
- `stream cursor`
- `transient event buffer`
- `in-flight text buffer`

Redis 不再承担：

- 旧 message grammar 兼容
- 最终产品层消息协议
- 长期历史表达

## 8.2 DB / Persistent Store 职责

持久化层负责保存：

- `conversation_turn`
- `timeline_event`（至少关键事件）
- `artifact`
- `reference_bundle`
- `error_summary`

Timeline 必须可重放，不能只存在于流式阶段。

## 8.3 History Commit Policy

最终 history 不按 token 流实时写入。

策略如下：

1. stream 期间只写运行态/缓存态
2. 只有在 `done` 或明确 `error` 后，才一次性提交标准化 turn 记录

这样可以避免：

- 半截输出污染 history
- 取消后残留脏记录
- 重连或回退留下不可解释状态

## 9. 前端体验模型

新的前端展示不再以“一个 assistant bubble 包含一切”为核心。

建议拆成五个视图层：

1. `Turn Header`
2. `Timeline`
3. `Final Answer Panel`
4. `References Panel`
5. `Diagnostics Drawer`

其中：

- Timeline 只展示过程
- Final Answer Panel 只展示最终 answer
- References Panel 只展示引用和来源
- Diagnostics Drawer 只在需要时展开

## 10. Runtime 路线

## 10.1 Phase 1：PydanticAI Adapter

第一阶段 runtime 实现采用 `PydanticAI`，原因不是它定义契约，而是它能降低第一阶段实现成本。

它适合用来实现：

- 单 turn 内部 loop
- tool 调用
- provider 调用
- 状态映射

但不负责定义：

- 对外 API
- TimelineEvent schema
- Artifact schema
- History commit policy

## 10.2 长期路线

如果 `PydanticAI adapter` 运行稳定、维护成本可接受，可以继续保留。

如果后续仍发现第三方 runtime 对行为边界约束太多，则继续把底层收成完全自研薄编排层。

由于契约层已经独立，届时只替换 runtime 实现，不需要再次重写前后端协议。

## 11. 替换边界

本次重写：

- 保留：`FastAPI`、`FastMCP`、业务 API、provider 接入、业务数据实体
- 替换：`mcp-agent runtime glue`、旧 WebSocket grammar、旧 Redis message shape、旧前端消息渲染模型、旧事件回推机制

这意味着：

- 业务价值复用
- 产品层和运行时耦合重做

## 12. 迁移与回退原则

### 12.1 迁移原则

- 可以保留短期 feature flag 灰度
- 不允许长期双栈共存
- 一旦新链路稳定，旧 WebSocket grammar 和旧 runtime glue 直接下线

### 12.2 回退条件

以下情况允许回退：

- SSE 在企业代理环境中明显不稳定
- timeline 重放和恢复不可靠
- turn/history/artifact 兼容性不成立
- tool/provider 错误不可诊断

### 12.3 回退要求

回退后必须保证：

- 历史记录可读
- turn 记录不变成孤儿
- artifact 不变成不可追踪残留

## 13. Phase 1 实施清单

Phase 1 的实施目标是跑通新的最小主链路，而不是一次性追求最终形态。

建议的 Phase 1 任务：

1. 新建 `aperag/agent_runtime/` 模块
2. 定义 `Turn / TimelineEvent / Artifact` schema
3. 实现 `TurnService`
4. 实现 `EventService`
5. 实现 `ArtifactService`
6. 实现 `HistoryWriter`
7. 定义 `AgentRuntime` 抽象
8. 实现 `PydanticAIRuntime`
9. 实现 MCP client adapter
10. 实现 `SSE StreamEmitter`
11. 新增 v2 agent API
12. 新增前端 timeline 组件
13. 新增 answer/references/diagnostics 分层面板
14. 实现 snapshot 恢复与 cancel
15. 补充契约级 E2E 覆盖

## 14. 验收标准

Phase 1 完成时，至少应满足：

1. 新 API 可以创建 turn、订阅 SSE、拉 snapshot、读取 artifact、取消 turn
2. 单 turn 内部可以完成多轮 search/tool/thinking loop
3. Timeline 可重连、可重放
4. 最终 answer 与过程事件完全分离
5. 历史提交策略不产生半截脏记录
6. `mcp-agent` 已退出主 chat 运行路径

## 15. 正式拍板

Agent Runtime V3 的正式拍板如下：

- 不再继续修补 `mcp-agent`
- 不再继续修补旧 WebSocket grammar
- 新 runtime 契约统一为 `Turn + TimelineEvent + Artifact + SSE`
- 第一阶段采用 `PydanticAI adapter`
- 后续由实现同学按本设计文档推进，架构侧负责监督契约边界与长期方向

一句话总结：

这次不是“把某个 agent 库换掉”，而是为 ApeRAG 重建一个更适合私有化交付的 agent runtime 产品层。
