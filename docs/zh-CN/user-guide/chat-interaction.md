# Chat 交互使用指南

本文面向最终用户，说明 ApeRAG 的 Agent Chat 是什么样的交互体验、如何创建 Bot、如何让 Agent 用上你的 Collection 与文件、以及常见的运行时状态含义。后端架构面（Turn / TimelineEvent / Artifact / SSE / runtime DI）见 [`architecture/conversation-agent-evaluation.md`](../architecture/conversation-agent-evaluation.md)。

## 1. 核心概念

- **Bot**：一次对话的"人格 + 工具箱"。决定 Agent 的系统提示、能使用的 Collection、以及默认模型。
- **Chat**：一条对话线（一个 Bot 下可有多条 Chat）。Chat 保留历史消息，重启时可继续之前的上下文。
- **Turn**：一次完整的问答（用户发一条消息 → Agent 完成应答）。Turn 有明确的生命周期：`queued → running → completed / failed / cancelled`。
- **Artifact**：Turn 过程中产出的结构化内容，例如引用、图片、表格。前端会自动渲染，不需要手动处理。
- **Timeline Event**：Turn 过程中的细粒度事件流（agent 思考、工具调用、流式 token 产出等），通过 SSE 实时推到前端。

## 2. 创建 Bot 与 Chat

1. 进入 `Bots` 页面 → "创建 Bot"。
   - 填名字与描述。
   - **Type = Agent**：选择 Agent 模式（这是当前推荐的默认）。
   - **选择 Collection**（可选）：Agent 在回答时可以引用这些知识库。
   - **选择模型**：省略时使用当前用户的默认 LLM provider；在 Bot 配置中可以 override。
2. Bot 创建成功后进入 Bot 详情页，点"开始对话"创建第一条 Chat。
3. 一条 Bot 下可以有多条 Chat，支持在不同话题间切换。

新注册用户会自动获得一个名为 `Default Agent Bot` 的默认 Bot，绑定系统级默认模型，便于立即尝试。

## 3. 在 Chat 里发起一次对话

1. 在 Chat 页面的输入区敲入问题，或者上传文件（PDF / Word / 图片，详见 [`user-guide/document-upload.md`](./document-upload.md)）。
2. 按回车发送 → 进入 **Turn 生命周期**：
   - **queued**：请求已入队，等待 runtime 认领。通常几百毫秒内进入 running。
   - **running**：Agent 正在思考、调用工具（检索 Collection、读网页、调 LLM）。Timeline 会以流式方式把中间步骤显示在消息下方，包括思考短摘、当前步骤的可见状态（"Thinking" / "Searching" / "Answering"）。
   - **completed**：回答已完成，Artifact（引用 / 图片 / 表格）加载到消息下方。
   - **failed**：回答因错误终止（LLM provider 故障、超时、外部工具错误等）；前端会显示错误分类 + 可再次尝试的操作。
   - **cancelled**：用户主动取消，或者上游配置（如 chat collection 权限）在 turn 开始前被禁用。
3. **可中途 Cancel**：在 turn 运行中，消息右上角会出现取消按钮。点击后 runtime 会尝试优雅中止：当前已生成的内容会保留，后续的工具调用与流式 token 会被终止。

## 4. Agent 能用到的能力

以下来源都可能被 Agent 调用来回答问题；具体启用哪些由 Bot 的配置决定：

| 能力 | 何时使用 | 对应配置 |
| --- | --- | --- |
| Collection 检索 | Bot 关联了一个或多个 Collection | Bot Collection 绑定（创建/编辑 Bot 时） |
| 单次 Chat 文件 | 用户在 Chat 里上传了 PDF/Doc/图片 | 上传 button（详见 document-upload.md） |
| Web 搜索 + 网页阅读 | Bot 开启 Web Access | Bot 配置的 `web_search_enabled` 字段 |
| MCP 工具 | 接入了 MCP server | 系统级 MCP 集成（见 [`integration/mcp.md`](../integration/mcp.md)） |
| 知识图谱查询 | Collection 启用了图谱索引 | Collection 配置 `enable_knowledge_graph` |

Agent 会自动判断什么时候去 Collection 检索、什么时候上 Web、什么时候读 Chat 上传的文件。大多数情况下用户不需要手动选择。

## 5. 常见运行时状态解读

### 5.1 进度指示

- "Thinking..." — Agent 在消化刚收到的问题，还没决定下一步做什么。
- "Searching collection..." — 正在检索某个 Collection 的向量 / 全文 / 图谱索引。
- "Reading chat file..." — 正在读 Chat 上传的文件内容。
- "Calling MCP tool..." — 正在调用外部 MCP 集成。
- "Answering..." — Agent 已经有足够信息，开始流式回答。

### 5.2 消息右上角标记

- 灰色齿轮 → Turn 仍在运行。
- 红色感叹号 → Turn 失败；hover 可看错误类型。
- 绿色勾 → Turn completed；同一组消息下方会显示引用来源（Artifact）。

### 5.3 "我已经取消这次回答，但流还在继续？"

- Cancel 是"请求 runtime 停下来"而不是"立即切断网络流"。在极少数情况下会有 1-2 秒延迟；收到 cancel 后 runtime 会结束当前工具调用再关闭流。
- 如果 cancel 后依然看到新 token，一般是已经在 pipeline 里的数据；这种情况下后端会保证 `turn.status = cancelled`，而不会回写 `running`。

## 6. 聊天历史的持久化

- 同一 Chat 下的所有 Turn 与消息都会落入 `chats` / `turns` / `artifacts` 等表；刷新或登出再登录都能恢复。
- 历史消息对应的引用（Artifact）也持久化；回查历史时引用仍可点开。
- **保留范围**：默认不设 TTL，除非系统管理员在 quota 管理里限制每用户的 chat 保留条数（详见 [`admin-guide/quota-system.md`](../admin-guide/quota-system.md)）。

## 7. 相关文档

- [`user-guide/document-upload.md`](./document-upload.md) — 文件上传与 chat document 的使用。
- [`admin-guide/prompt-customization.md`](../admin-guide/prompt-customization.md) — 如何定制 Agent 的系统提示 / 查询模板。
- [`user-guide/evaluation-guide.md`](./evaluation-guide.md) — 如何对 Bot + Collection 组合做自动化评估。
- [`architecture/conversation-agent-evaluation.md`](../architecture/conversation-agent-evaluation.md) — 后端架构面（Turn 生命周期、Artifact 存储、SSE 协议）。
