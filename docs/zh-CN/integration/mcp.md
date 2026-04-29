---
title: MCP 集成指南
description: ApeRAG 内置 MCP Server，用于让 Claude Desktop、Cursor、Dify 等 AI 客户端直接调用知识库检索与网页抓取能力
position: 1
---

# MCP 集成指南

ApeRAG 内置了一个 [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) Server，与主 API 进程共生，不需要额外部署。任何支持 MCP 协议的客户端（Claude Desktop、Cursor、Dify、Pydantic AI Agent 等）都可以通过 HTTP 直接调用 ApeRAG 的 5 个工具：知识库检索、聊天临时文件检索、网页搜索、网页内容抓取、知识库列表。

本文只描述「怎么接入 ApeRAG MCP」和「底层是怎么跑起来的」。每个工具的完整参数、返回 schema 和调优建议，见 [MCP API 参考](./mcp-api.md)。

## 架构一览

- **进程模型**：MCP Server 是 FastAPI 主进程的子挂载，路径为 `/mcp/`；它通过 [FastMCP](https://github.com/jlowin/fastmcp) 框架实现，运行在 stateless HTTP 模式（与 SSE 兼容），复用同一个 `uvicorn` 进程、同一份 8000 端口。
- **业务边界**：MCP Server 自己不持有数据层，所有工具都是通过 `httpx` 回调 ApeRAG 自身的 `/api/v1/...` 和 `/api/v2/...` 路由，最终走到 `retrieval` / `knowledge_base` / `web_access` 等 domain；这样客户端走 MCP 和走 REST API 看到的语义一致。
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

远程部署时把 `http://localhost:8000/mcp/` 替换成你的实际地址（`https://<你的域名>/mcp/`）。保存后重启 Claude Desktop 即可在对话中看到 `list_collections`、`search_collection` 等工具。

### Cursor

在 `~/.cursor/mcp.json` 或项目级 `.cursor/mcp.json` 里加入与 Claude Desktop 一致的 `mcpServers` 段即可。Cursor 会在 Agent / Composer 模式下自动调用。

### Dify

Dify 的工作流 Agent 节点支持 "MCP Server" 工具类型。配置方法详见 [Dify 集成](./dify.md) — 核心步骤是在 Dify 工具面板里填写 `http://<ApeRAG 地址>/mcp/` 和 API Key，然后把对应 Agent 挂上即可。

### 其他 MCP 客户端

所有遵循 MCP HTTP 传输规范的客户端都可以直接对接：只需要提供一个支持 `Bearer` 认证头的 HTTP client 端即可。如果客户端只支持 stdio 传输，需要在本地起一个 MCP HTTP-to-stdio 代理把 `/mcp/` 桥接过去（这部分由客户端侧完成，不需要 ApeRAG 做改造）。

## 可用工具速览

完整参数与返回 schema 见 [MCP API 参考](./mcp-api.md)。概览如下：

| 工具 | 作用 | 典型场景 |
|------|------|---------|
| `list_collections` | 列出当前 API Key 可访问的知识库 | 让 Agent 在多个知识库之间自动选择 |
| `search_collection` | 在指定知识库中做混合检索（向量 / 全文 / 图谱 / 摘要 / 视觉） | 主流 RAG 问答 |
| `search_chat_files` | 在一次对话中临时上传的文件里检索 | 即时分析用户上传的简历、论文等 |
| `web_search` | 互联网搜索（JINA / DuckDuckGo） | 补充时效性信息 |
| `web_read` | 抓取给定 URL 列表并返回正文 | 读取引用链接的原文 |

附带一项 MCP Resource `aperag://usage-guide` 提供给客户端作为 Agent 使用的提示词素材；以及一项 MCP Prompt `search_assistant`，供客户端复用。

## 常见问题

**Q：MCP 可以独立部署吗？**
不行。MCP Server 与 ApeRAG 主进程绑定在同一个 FastAPI 应用上，所有工具都通过内部回调走 REST API，拆出来反而需要额外的服务发现。

**Q：启动时没看到 MCP Server 挂上？**
启动日志里会打印 FastMCP 的初始化信息；如果只看到 uvicorn 启动但没有 FastMCP 输出，检查是否能访问 `http://localhost:8000/mcp/`（正常会返回 MCP 协议响应，不是 404）。

**Q：我希望只允许部分工具？**
目前没有基于工具粒度的开关；授权粒度走 API Key，所有调用会落到这个 Key 对应的用户身份，用户看不到的知识库不会出现在 `list_collections` 里。

**Q：搜到图片时 URL 以 `asset://` 开头怎么办？**
这是 ApeRAG 自定义协议，客户端需要再请求 `GET /api/v1/assets/...` 换成真实 HTTP URL。`asset_url` 的构造见 [MCP API 参考](./mcp-api.md) 的 "图片处理" 小节。

## 相关链接

- [MCP 协议规范](https://modelcontextprotocol.io/)
- [MCP API 参考](./mcp-api.md) — 5 个工具的完整参数/返回 schema
- [Dify 集成](./dify.md) — 在 Dify Agent 里接入 ApeRAG MCP
- [`docs/modularization/architecture.md`](../../modularization/architecture.md) — 后端模块化边界（MCP 归属「共享基础设施」）
