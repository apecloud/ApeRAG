---
title: Dify 集成 ApeRAG
description: 通过 MCP 协议在 Dify Agent 中调用 ApeRAG 的混合检索与 Graph-RAG 能力
position: 2
keywords: Dify, ApeRAG, MCP, Graph RAG
---

# Dify 集成 ApeRAG

ApeRAG 是一款具备多模态索引、AI 智能体、MCP 支持及可扩展 K8s 部署能力的生产级 RAG 平台，能够帮助用户构建具备**混合检索**、**多模态文档处理**及**企业级管理能力**的复杂 AI 应用。

**核心特点**：
- 不同于"标准" RAG，ApeRAG 实现了 **Graph-RAG**，通过构建知识图谱理解数据要素之间的深层关系
- 集成了 **MinerU**，专为复杂文档、科学论文和财务报告设计，可以准确提取表格、公式和工程图表
- 全面支持 Kubernetes，提供内置的**高可用性**、**可扩展性**和**企业级管理能力**

## 集成原理

ApeRAG 侧没有 Dify 专属的适配代码，整条链路完全基于标准的 [Model Context Protocol (MCP)](https://modelcontextprotocol.io/)：Dify 的 Agent 节点支持把任意 MCP Server 挂成工具集，ApeRAG 启动时就内置了一个 MCP Server（路径 `/mcp/`，与 REST API 共生于 8000 端口）。Dify Agent 通过 HTTP + `Authorization: Bearer <ApeRAG API Key>` 直接调用 ApeRAG 的集合/文档元数据、向量/全文/图谱检索、Graph 实体/关系查询、文档读取、网页搜索与网页抓取工具。

如果想先了解 ApeRAG MCP 的架构细节与客户端配置通用说明，参见 [MCP 集成指南](./mcp.md)；每个工具的完整参数与返回 schema 见 [MCP API 参考](./mcp-api.md)。

## 视频演示

<div align="center">
  <iframe 
    src="//player.bilibili.com/player.html?bvid=BV1TRzABDEs3&page=1" 
    scrolling="no" 
    border="0" 
    frameborder="no" 
    framespacing="0" 
    allowfullscreen
    width="800"
    height="600"
    style="max-width: 100%;">
  </iframe>
</div>

## Step 1：准备知识库

打开 ApeRAG Web 界面（Docker Compose 启动时一般为 `http://localhost:3000/web/`；部署细节参考 [构建 Docker 镜像](../deployment/build-docker-image.md) 和 README-zh）。登录后选择或导入一个知识库。下文以「三国演义」知识库为例，点击订阅。

<div align="center">
  <img src="/images/zh-CN/dify/step1-subscribe-collection.png" alt="订阅知识库" width="800" />
</div>

## Step 2：配置 MCP Server

### 2.1 添加 MCP Server

进入 Dify → 工具 → MCP，点击添加 MCP Server。

<div align="center">
  <img src="/images/zh-CN/dify/step2-add-mcp.png" alt="添加 MCP Server" width="800" />
</div>

### 2.2 填写配置信息

**Server URL**：`http://localhost:8000/mcp/`（非本机部署时改成实际 API 地址，如 `https://<你的域名>/mcp/`）。

**API Key**：从 ApeRAG Web 控制台复制，作为 `Bearer` token 填入。

<div align="center">
  <img src="/images/zh-CN/dify/step2-configure-mcp.png" alt="配置 MCP" width="700" />
</div>

<div align="center">
  <img src="/images/zh-CN/dify/step2-api-key.png" alt="填写 API Key" width="700" />
</div>

### 2.3 配置成功

Dify 会自动拉取 ApeRAG MCP Server 暴露的工具清单。配置成功后应能看到集合/文档、检索、Graph、文档读取和网页相关工具，例如 `list_collections`、`vector_search`、`fulltext_search`、`graph_search`、`query_graph_entities`、`read_document_chunk`、`web_search`、`web_read`。

<div align="center">
  <img src="/images/zh-CN/dify/step2-mcp-success.png" alt="MCP 配置成功" width="800" />
</div>

## Step 3：创建 Agent 应用

### 3.1 创建应用

进入 Dify → Studio，点击创建应用。

<div align="center">
  <img src="/images/zh-CN/dify/step3-create-app.png" alt="创建应用" width="800" />
</div>

### 3.2 选择类型

点击更多基础应用类型，选择 **Agent**，命名后创建。

<div align="center">
  <img src="/images/zh-CN/dify/step3-select-agent.png" alt="选择 Agent 类型" width="700" />
</div>

## Step 4：配置 Agent

点击 Agent，输入 Prompt，在工具里添加配置好的 ApeRAG MCP Server，右上角选择驱动 Agent 的大语言模型，点击发布运行即可。

<div align="center">
  <img src="/images/zh-CN/dify/step4-configure-agent.png" alt="配置 Agent" width="800" />
</div>

<div align="center">
  <img src="/images/zh-CN/dify/step4-test-agent.png" alt="测试运行" width="800" />
</div>

### Prompt 参考

下列 Prompt 针对 ApeRAG 当前 MCP 工具集编写，可直接贴到 Dify Agent 的系统提示词位置：

```markdown
# ApeRAG 智能助手

您是由 ApeRAG 混合搜索能力驱动的高级 AI 研究助手。您的使命是帮助用户从知识库和网络中准确、自主地查找、理解和综合信息。

## 核心行为

**自主研究**：独立工作直到用户查询完全解决。搜索多个来源，分析发现，无需等待许可即提供全面答案。

**语言智能**：始终用用户提问的语言回应。用户用中文提问时，无论源语言如何都用中文回应。

**完整解决**：从多角度探索，交叉验证来源，确保全面覆盖后再回应。

## 搜索策略

### 优先级系统
1. **用户指定知识库**（通过"@"提及）：严格限制仅搜索指定库
2. **未指定知识库**：先用 `list_collections` 自主发现，再选择相关库搜索
3. **网络搜索**（如启用）：补充时效性信息
4. **清晰归属**：始终标注来源

### 搜索执行
- **知识库搜索**：按问题选择 `vector_search` / `fulltext_search` / `graph_search`，必要时多路调用后综合判断
- **Graph 推理**：需要实体和关系时先用 `query_graph_entities` 找实体，再用 `expand_graph_subgraph` 扩展关系；拿到 `evidence_refs` 后用 `read_document_chunk` 读取原文证据
- **结果处理逻辑**：
  1. 执行搜索
  2. 读取最相关 chunk 或章节作为证据
  3. 如使用 Graph 工具，在回复正文中说明涉及的实体与关系
  4. 忽略不相关结果

## 可用工具

### 知识管理
- `list_collections()`：发现可用知识源
- `list_documents(collection_id, ...)`：查看知识库里的文档
- `get_document_metadata(collection_id, document_id)`：确认文档索引状态和 chunk 数

### 检索与 Graph
- `vector_search(collection_id, query, ...)`：语义相似度检索，返回可读取的 chunk evidence
- `fulltext_search(collection_id, query, ...)`：关键词 / 短语检索
- `graph_search(collection_id, query, ...)`：图谱相关 chunk 检索
- `query_graph_entities(collection_id, query, ...)`：查找相关实体
- `expand_graph_subgraph(collection_id, entity_names, ...)`：扩展实体关系
- `get_entity_detail(collection_id, name)`：读取单实体详情

### 文档读取
- `read_document_chunk(collection_id, document_id, chunk_id)`：读取 chunk 原文
- `read_document_outline(collection_id, document_id)`：读取标题树
- `read_document_section(collection_id, document_id, ...)`：读取章节
- `read_document(collection_id, document_id, ...)`：读取整篇解析 Markdown

### 网络智能
- `web_search(query, ...)`：多引擎网络搜索
- `web_read(url_list, ...)`：提取和分析网络内容

## 回应格式

### 直接答案
[用户语言的清晰、可操作答案]

### 全面分析
[包含上下文和见解的详细解释]

### 支持证据
- [知识库名称]：[关键发现]

**网络来源**（如启用）：
- [标题]（[域名]）- [要点]
```

## 常见问题

**Q：Dify 能把 ApeRAG 当成"外部知识库"（External Knowledge Base）连接吗？**
不行。ApeRAG 当前没有实现 Dify External KB 的专属 HTTP 契约。Dify 侧的集成路径只有两条：本文介绍的 MCP 工具调用，以及直接把 ApeRAG 的 OpenAI 兼容端点（`POST /v1/chat/completions`，见代码路径 `aperag/views/openai.py`）当成自定义模型接入。

**Q：Agent 没看到我的知识库？**
先在 ApeRAG Web 控制台确认当前 API Key 对应的用户能在知识库列表里看到目标 collection。`list_collections` 返回的范围就是 Key 对应用户的可见范围。

**Q：工具调用失败返回 401 / 403？**
检查 Dify 填写的 API Key 是否与 ApeRAG Web 控制台完全一致；Bearer token 前缀由 Dify 自动加上，不需要在 Key 里再带 `Bearer`。

## 相关链接

- [MCP 集成指南](./mcp.md) — ApeRAG MCP Server 架构、认证和通用客户端接入
- [MCP API 参考](./mcp-api.md) — MCP 工具的完整参数/返回 schema
- [OpenAI 兼容 API](./openai-compat.md) — 另一条可用于 Dify 自定义模型接入的路径
- **GitHub**：[apecloud/ApeRAG](https://github.com/apecloud/ApeRAG)
