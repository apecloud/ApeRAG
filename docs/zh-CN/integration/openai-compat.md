# OpenAI 兼容接口

> **读者定位**：想把 ApeRAG Bot 当作 OpenAI Chat Completion endpoint 接入第三方工具（SDK / Dify / Zapier / 自研应用）的用户。
>
> **范围**：`POST /api/v1/chat/completions` 端点的调用方式、鉴权、query 参数、stream / non-stream 差异、局限性。

## 能做什么

ApeRAG 暴露一个与 OpenAI Chat Completion 协议**部分兼容**的端点，用户可以：

- 用现有的 OpenAI SDK（只需改 `base_url`）把 ApeRAG Bot 当 LLM 调用
- 在 Dify、LobeChat、Cherry Studio 等支持 OpenAI 兼容后端的工具里接入 ApeRAG
- 保留 ApeRAG 的能力（RAG 检索、自定义 prompt、工具调用），同时享受 OpenAI 协议的生态

端点本身不是一个 LLM — 它是 ApeRAG 某个 Bot 的 HTTP 封装。实际的 LLM 调用、RAG 检索、工具执行都走 Agent Runtime V3 的完整 pipeline。

## 基本调用

### 端点

```
POST /api/v1/chat/completions
```

### 鉴权

Bearer API Key（与普通 ApeRAG API 相同，见 [`admin-guide/api-keys.md`](../admin-guide/api-keys.md)）：

```
Authorization: Bearer sk-xxxxxxxxxxxxxxxx
```

### Query 参数

| 参数 | 必填 | 说明 |
| --- | --- | --- |
| `bot_id` | 推荐 | 绑定具体的 Agent Bot。省略时当前版本会失败（没有默认 Bot 概念，该字段名义上 optional 是为接口签名对齐） |
| `chat_id` | 否 | 复用已有会话，便于保留历史上下文。省略则创建一个临时（ephemeral）会话，不持久化历史 |
| `language` | 否 | 响应语言，透传给 Agent Runtime。默认 `en-US`。支持：`en-US` / `zh-CN` / `zh-TW` / `ja-JP` / `ko-KR` / `fr-FR` / `de-DE` / `es-ES` / `it-IT` / `pt-BR` / `ru-RU` |

### Request Body

```json
{
  "model": "aperag",
  "messages": [
    {"role": "user", "content": "What is RAG?"}
  ],
  "stream": false,
  "temperature": 0.7,
  "max_tokens": 1024
}
```

字段说明：

- `model`：**当前被忽略**，Bot 配置里的 LLM 才是真正生效的模型。为了 OpenAI SDK 的兼容，请随便填一个合法字符串（例如 `"aperag"`）。
- `messages`：OpenAI 格式的消息数组。只有最后一条 `user` 消息会触发新一轮 turn；前面的 messages 只在 `chat_id` 为空时被当作历史上下文的简化形式。若 `chat_id` 不为空，以 ApeRAG 自己的 chat history 为准。
- `stream`：`true` 返回 SSE 流（`text/event-stream`），`false` 返回完整 JSON。
- `temperature` / `max_tokens` / `max_completion_tokens` / `timeout`：**目前被忽略**。真正生效的是 Bot 配置里的参数。保留在 schema 里是为了 OpenAI SDK 不报错。

### 非流式响应

```json
{
  "id": "<msg-id>",
  "object": "chat.completion",
  "created": 1714000000,
  "model": "aperag",
  "choices": [
    {
      "index": 0,
      "message": {"role": "assistant", "content": "RAG (Retrieval-Augmented Generation) is ..."},
      "finish_reason": "stop"
    }
  ],
  "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
}
```

> ⚠️ **`usage` 永远返回 0**。当前版本没有实现 token 统计（原因：ApeRAG 内部 turn 可能经过多个 LLM / tool，单一 token 计数没有清晰语义）。依赖 usage 做计费的下游请走 Bot 级的 audit log 而非这个字段。

### 流式响应

设置 `"stream": true` 时返回 `text/event-stream`：

```
data: {"id":"...","object":"chat.completion.chunk","created":...,"model":"aperag","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}

data: {"id":"...","object":"chat.completion.chunk","created":...,"model":"aperag","choices":[{"index":0,"delta":{"content":"RAG"},"finish_reason":null}]}

data: {"id":"...","object":"chat.completion.chunk","created":...,"model":"aperag","choices":[{"index":0,"delta":{"content":" is ..."},"finish_reason":null}]}

data: {"id":"...","object":"chat.completion.chunk","created":...,"model":"aperag","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}
```

格式与 OpenAI 的 `chat.completion.chunk` 一致：首条含 `role: assistant`，中间若干条 `delta.content` 增量，最后一条 `finish_reason: stop`。**目前没有发 `data: [DONE]` 行**，下游 SDK 若严格依赖该 sentinel 需要额外处理。

## 错误响应

错误按 OpenAI 格式返回：

```json
{
  "error": {
    "message": "Invalid JSON request body",
    "type": "server_error",
    "code": "internal_error"
  }
}
```

常见错误：

| HTTP | message | 原因 |
| --- | --- | --- |
| 400 | `Invalid JSON request body` | body 不是合法 JSON |
| 401 | fastapi-users 默认 | Bearer token 缺失 / 失效 / 已吊销 |
| 403 | quota message | 触发 quota 限制（见 [`admin-guide/quota-system.md`](../admin-guide/quota-system.md)） |
| 404 | bot not found | `bot_id` 不存在或无访问权限 |
| 500 | internal_error | Agent Runtime 异常 |

## 在 OpenAI SDK 里调用

### Python

```python
from openai import OpenAI

client = OpenAI(
    api_key="sk-xxxxxxxxxxxxxxxx",
    base_url="https://<your-aperag-host>/api/v1",
)

# 非流式
resp = client.chat.completions.create(
    model="aperag",
    messages=[{"role": "user", "content": "What is RAG?"}],
    extra_query={"bot_id": "bot-xxx"},
)
print(resp.choices[0].message.content)

# 流式
stream = client.chat.completions.create(
    model="aperag",
    messages=[{"role": "user", "content": "What is RAG?"}],
    stream=True,
    extra_query={"bot_id": "bot-xxx", "chat_id": "chat-xxx"},
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

OpenAI SDK 不直接支持自定义 query param，用 `extra_query={"bot_id": ..., "chat_id": ..., "language": ...}` 即可。

### cURL

```bash
curl -X POST "https://<your-aperag-host>/api/v1/chat/completions?bot_id=bot-xxx" \
  -H "Authorization: Bearer sk-xxxxxxxxxxxxxxxx" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "aperag",
    "messages": [{"role": "user", "content": "What is RAG?"}],
    "stream": false
  }'
```

### 在 Dify / LobeChat 里接入

把 OpenAI-compatible provider 的 `base_url` 设为 `https://<your-aperag-host>/api/v1`，`api_key` 用 ApeRAG API Key。然后在每条消息 URL 里拼 `?bot_id=<bot_id>` — 具体配置取决于工具侧 UI。

## 不兼容点与注意事项

### 不支持

- **`tools` / `tool_calls` 字段**：ApeRAG Bot 的 tool 调用是在 runtime 内部完成的，不透传到 HTTP 响应里。
- **`functions`（旧版 function calling）**：同上。
- **`logprobs` / `top_logprobs`**：不支持。
- **`n > 1`（多 choice）**：不支持，永远返回单个 choice。
- **`response_format: json_object`**：不支持，返回纯文本。
- **`seed` / `logit_bias`**：不支持。

### 行为差异

- **`model` 字段被忽略**：真正生效的 LLM 由 Bot 配置决定，`model` 字段仅作 SDK 兼容占位。
- **`usage` 全为 0**：当前版本不做 token 统计。
- **流式无 `data: [DONE]`**：严格依赖 SDK 需手工兜底。
- **没有 `moderation` endpoint**：只开放 `/chat/completions`。
- **rate limit header 未实现**：限流只通过 quota system，没有 `X-RateLimit-*` header。
- **timeout**：Agent Runtime 默认 300s 超时；Bot 里配置的 timeout 以 Bot 为准，请求层面的 `timeout` 字段被忽略。

### 与原生 ApeRAG 接口的区别

ApeRAG 自身的对话接口是 `/api/v2/bots/{bot_id}/chats/{chat_id}/messages`（见 [`reference/prompt-api.md`](../reference/prompt-api.md)），返回更丰富的结构（turn_id / timeline events / artifacts / references）。OpenAI 兼容接口把这些都**摊平成纯文本**，丢失结构化能力。

**什么时候用 OpenAI 兼容接口**：

- 接第三方工具（Dify / LobeChat / 自己的 ChatGPT UI）
- 已有大量 OpenAI SDK 代码要迁移

**什么时候用原生接口**：

- 需要展示引用（references）
- 需要 timeline / artifact 细节
- 需要访问 turn 级元数据

## 相关文档

- [`admin-guide/api-keys.md`](../admin-guide/api-keys.md) — 获取调用用的 Bearer Token
- [`admin-guide/quota-system.md`](../admin-guide/quota-system.md) — quota 限制
- [`reference/prompt-api.md`](../reference/prompt-api.md) — ApeRAG 原生对话接口
- [`user-guide/chat-interaction.md`](../user-guide/chat-interaction.md) — 用户交互流程
