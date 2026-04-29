---
title: MCP 集成指南
description: ApeRAG 内置 MCP Server，用于让 Claude Desktop、Cursor、Dify 等 AI 客户端直接调用知识库、Graph RAG、文档读取与网页工具
position: 1
---

# MCP 集成指南

ApeRAG 内置了一个 [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) Server，与主 API 进程共生，不需要额外部署。任何支持 MCP 协议的客户端（Claude Desktop、Cursor、Dify、Pydantic AI Agent 等）都可以通过 HTTP 直接调用 ApeRAG 的工具：集合/文档元数据、向量/全文/图谱检索、Graph 实体/关系查询、文档读取、网页搜索与网页抓取。

本文只描述「怎么接入 ApeRAG MCP」和「底层是怎么跑起来的」。每个工具的完整参数、返回 schema 和调优建议，见 [MCP API 参考](./mcp-api.md)。

## 架构一览

- **进程模型**：MCP Server 是 FastAPI 主进程的子挂载，路径为 `/mcp/`；它通过 [FastMCP](https://github.com/jlowin/fastmcp) 框架实现，运行在 stateless HTTP 模式（与 SSE 兼容），复用同一个 `uvicorn` 进程、同一份 8000 端口。
- **业务边界**：MCP Server 自己不持有数据层，所有工具都是通过 `httpx` 回调 ApeRAG 自身的 `/api/v1/...` 和 `/api/v2/...` 路由，或通过租户校验后的 domain service 读取数据；这样客户端走 MCP 和走 REST API 看到的语义一致。
- **共享基础设施**：`aperag/mcp/` 在后端模块化划分里属于 "shared infrastructure"，不是独立 domain。详见 [`docs/modularization/architecture.md`](../../modularization/architecture.md) 第 F10 条「跨域共享基础设施」相关条目。
- **内部复用**：ApeRAG 的 Agent Runtime V3 自身也会把同一 MCP Server 当作 toolset 注入到 Agent Turn 里，启动时通过环境变量 `APERAG_MCP_URL`（默认 `http://localhost:8000/mcp/`）引用。

## 开箱即用

默认部署无需额外配置即可启用 MCP：

- **Docker Compose**：`docker-compose up -d` 即可，`api` 服务的 `8000:8000` 端口同时暴露 REST API 和 MCP Server。
- **健康检查**：`curl http://localhost:8000/health/ready` 正常即说明主进程 HTTP 入口和 MCP 子挂载都已启动；`/health` 保留为旧探针兼容入口。
- **关闭 MCP**：目前没有提供关闭开关；MCP Server 与主进程共生。

可选环境变量：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `APERAG_API_KEY` | — | MCP 认证备用凭证；仅当客户端没带 `Authorization` 头时才会读到 |
| `OTEL_MCP_ENABLED` | `true` | 是否为 MCP 开启 OpenTelemetry 追踪；排查性能问题时可关掉 |
| `APERAG_MCP_URL` | `http://localhost:8000/mcp/` | Agent Runtime 内部回连 MCP 的地址，仅当你把 MCP 反向代理到非默认路径时才需要改 |

## 认证

MCP Server 有两条认证通道，按优先级读取：

1. **HTTP Authorization 头（推荐，生产环境唯一可用方式）**：客户端在每次请求里带 `Authorization: Bearer <api-key>`。API Key 在 ApeRAG Web 控制台创建。
2. **环境变量 `APERAG_API_KEY`（备用，单租户场景）**：当 Authorization 头缺失时才回退到这个。多用户同时使用时会导致互相串号，不要在生产环境启用。

两种方式都没命中时，MCP Server 会直接报错拒绝请求。

## 客户端接入

### Claude Desktop

修改 Claude Desktop 的 `claude_desktop_config.json`：

```json
{
  "mcpServers": {
    "aperag": {
      "url": "http://localhost:8000/mcp/",
      "headers": {
        "Authorization": "Bearer your-api-key-here"
      }
    }
  }
}
```

远程部署时把 `http://localhost:8000/mcp/` 替换成你的实际地址（`https://<你的域名>/mcp/`）。保存后重启 Claude Desktop 即可在对话中看到 ApeRAG 暴露的集合、检索、Graph、文档读取和网页工具。

### Cursor

在 `~/.cursor/mcp.json` 或项目级 `.cursor/mcp.json` 里加入与 Claude Desktop 一致的 `mcpServers` 段即可。Cursor 会在 Agent / Composer 模式下自动调用。

### Dify

Dify 的工作流 Agent 节点支持 "MCP Server" 工具类型。配置方法详见 [Dify 集成](./dify.md) — 核心步骤是在 Dify 工具面板里填写 `http://<ApeRAG 地址>/mcp/` 和 API Key，然后把对应 Agent 挂上即可。

### 其他 MCP 客户端

所有遵循 MCP HTTP 传输规范的客户端都可以直接对接：只需要提供一个支持 `Bearer` 认证头的 HTTP client 端即可。如果客户端只支持 stdio 传输，需要在本地起一个 MCP HTTP-to-stdio 代理把 `/mcp/` 桥接过去（这部分由客户端侧完成，不需要 ApeRAG 做改造）。

## 可用工具速览

完整参数与返回 schema 见 [MCP API 参考](./mcp-api.md)。概览如下：

| 工具 | 粒度 | 作用 | 典型场景 |
|------|------|------|---------|
| `list_collections` | collection | 列出当前 API Key 可访问的知识库 | 让 Agent 在多个知识库之间自动选择 |
| `get_collection_metadata` | collection | 读取单个知识库的索引模式、文档数等元数据 | 判断是否启用了 vector / fulltext / graph |
| `list_documents` | document | 分页列出知识库文档 | 让 Agent 先盘点文档，再决定读取或检索 |
| `get_document_metadata` | document | 读取单个文档的索引状态、chunk 数、媒体类型 | 确认文档是否已可检索 |
| `vector_search` | chunk | 向量语义检索，返回可读取的 chunk evidence | 模糊语义问题、自然语言问答 |
| `fulltext_search` | chunk | 全文/关键词检索，返回可读取的 chunk evidence | 专有名词、编号、精确短语 |
| `graph_search` | chunk | 图谱相关 chunk 检索，返回可读取的 chunk evidence | 需要实体/关系语义但最终要原文证据 |
| `query_graph_entities` | entity | 按自然语言查找相关 Graph 实体 | 先找候选实体，再扩展关系 |
| `expand_graph_subgraph` | entity/relation | 从实体名扩展邻居和关系 | 做 Graph reasoning 或关系探索 |
| `get_entity_detail` | entity | 读取单个实体详情 | 已知实体名时查类型、描述和证据引用 |
| `read_document_chunk` | chunk | 按 `document_id + chunk_id` 读取 chunk 原文 | 读取 search / Graph evidence 的原文 |
| `read_document` | document | 读取整篇解析后的 Markdown，可带 byte range | 长上下文模型或人工检查全文 |
| `read_document_outline` | document/section | 读取文档标题树 | 先导航，再选 section |
| `read_document_section` | section | 按 section path 或 heading anchor 读取章节 | 精读某一章节 |
| `web_search` | web | 互联网搜索（JINA / DuckDuckGo） | 补充时效性信息 |
| `web_read` | web | 抓取给定 URL 列表并返回正文 | 读取引用链接的原文 |

## 检索与读取的组合关系

- `vector_search`、`fulltext_search`、`graph_search` 都是 **chunk-level search**：返回的 `items[*].metadata` 应包含 `collection_id`、`document_id`、`chunk_id` 等证据定位信息。拿到命中后，继续调用 `read_document_chunk(collection_id, document_id, chunk_id)` 读取原文。
- `query_graph_entities`、`expand_graph_subgraph`、`get_entity_detail` 是 **Graph element-level tools**：返回实体 / 关系，而不是直接返回原文 chunk。task #32 Phase A 会让这些响应携带 `evidence_refs`，每个 ref 包含 `document_id`、`chunk_id` 和可选 `parse_version`，这样 Agent 可以直接跳到 `read_document_chunk`。
- `search_graph` 与 `query_graph_entities` 不等价：前者服务「我要相关原文证据」，后者服务「我要理解哪些实体/关系相关」。复杂问题通常先用 `query_graph_entities` 找实体，再用 `expand_graph_subgraph` 看关系，最后用 `evidence_refs` 或 `graph_search` 跳回 chunk 证据。
- `list_documents` / `get_document_metadata` 用于盘点和确认索引状态；`read_document_outline` / `read_document_section` 用于按结构导航；`read_document` 适合长上下文模型读取全文。
- rerank 已从 MCP 接口面删除。各索引工具保留自己的排序语义：vector 用相似度，fulltext 用关键词相关性，graph 用图谱证据。

附带一项 MCP Resource `aperag://usage-guide` 提供给客户端作为 Agent 使用的提示词素材；以及一项 MCP Prompt `search_assistant`，供客户端复用。

## 常见问题

**Q：MCP 可以独立部署吗？**
不行。MCP Server 与 ApeRAG 主进程绑定在同一个 FastAPI 应用上，所有工具都通过内部回调走 REST API 或复用主进程 domain service，拆出来反而需要额外的服务发现。

**Q：启动时没看到 MCP Server 挂上？**
启动日志里会打印 FastMCP 的初始化信息；如果只看到 uvicorn 启动但没有 FastMCP 输出，检查是否能访问 `http://localhost:8000/mcp/`（正常会返回 MCP 协议响应，不是 404）。

**Q：我希望只允许部分工具？**
目前没有基于工具粒度的开关；授权粒度走 API Key，所有调用会落到这个 Key 对应的用户身份，用户看不到的知识库不会出现在 `list_collections` 里。

**Q：搜到图片时 URL 以 `asset://` 开头怎么办？**
这是 ApeRAG 自定义协议，客户端需要再请求 `GET /api/v1/assets/...` 换成真实 HTTP URL。`asset_url` 的构造见 [MCP API 参考](./mcp-api.md) 的 "图片处理" 小节。

## 相关链接

- [MCP 协议规范](https://modelcontextprotocol.io/)
- [MCP API 参考](./mcp-api.md) — 工具的完整参数/返回 schema
- [Dify 集成](./dify.md) — 在 Dify Agent 里接入 ApeRAG MCP
- [`docs/modularization/architecture.md`](../../modularization/architecture.md) — 后端模块化边界（MCP 归属「共享基础设施」）
