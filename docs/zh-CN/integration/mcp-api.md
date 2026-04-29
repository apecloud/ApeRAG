---
title: MCP API
description: Model Context Protocol API 文档
---

# MCP API

ApeRAG 通过 [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) 对外提供标准化工具接口，让 AI 助手（Claude Desktop、Cursor、Dify 等）能够直接访问知识库、Graph RAG、文档读取和网页工具。

## 快速开始

### 配置示例

以 Claude Desktop 为例，在配置文件中添加：

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

### 认证方式

支持两种认证方式（按优先级）：

1. **HTTP Authorization 头**（推荐）：`Authorization: Bearer your-api-key`
2. **环境变量**（备用）：`APERAG_API_KEY=your-api-key`

> **获取 API Key**：登录 ApeRAG 后，在设置页面创建或复制你的 API Key。

## 工具总览

| 分类 | 工具 | 粒度 | 主要用途 |
|------|------|------|----------|
| 集合元数据 | `list_collections` | collection | 列出可访问知识库 |
| 集合元数据 | `get_collection_metadata` | collection | 查看知识库索引模式、文档数等 |
| 文档元数据 | `list_documents` | document | 分页列出知识库内文档 |
| 文档元数据 | `get_document_metadata` | document | 查看单文档索引状态、chunk 数、媒体类型 |
| 检索 | `vector_search` | chunk | 语义相似度检索 |
| 检索 | `fulltext_search` | chunk | 关键词 / 短语检索 |
| 检索 | `graph_search` | chunk | 图谱相关 chunk 证据检索 |
| Graph | `query_graph_entities` | entity | 根据自然语言查询相关实体 |
| Graph | `expand_graph_subgraph` | entity/relation | 从实体扩展邻居与关系 |
| Graph | `get_entity_detail` | entity | 获取单个实体详情 |
| 文档读取 | `read_document_chunk` | chunk | 按 `document_id + chunk_id` 读取 chunk 原文 |
| 文档读取 | `read_document` | document | 读取整篇解析 Markdown，可带 byte range |
| 文档读取 | `read_document_outline` | document/section | 读取文档标题树 |
| 文档读取 | `read_document_section` | section | 读取指定章节 |
| 网络 | `web_search` | web | 搜索互联网 |
| 网络 | `web_read` | web | 读取 URL 正文 |

## 组合原则

### Chunk-level search

`vector_search`、`fulltext_search`、`graph_search` 都返回 chunk-level `SearchResult`。返回项的 `metadata` 用于继续读取原文：

```python
hits = vector_search(collection_id="col_1", query="部署策略", top_k=5)
top = hits["items"][0]

chunk = read_document_chunk(
    collection_id=top["metadata"]["collection_id"],
    document_id=top["metadata"]["document_id"],
    chunk_id=top["metadata"]["chunk_id"],
)
```

- `vector_search`：适合自然语言、同义表达、模糊语义问题。
- `fulltext_search`：适合专有名词、编号、精确短语。
- `graph_search`：适合需要图谱语义但最终要拿原文证据的问题。

### Graph element tools

`query_graph_entities`、`expand_graph_subgraph`、`get_entity_detail` 返回 Graph 元素，不等同于 chunk-level search：

- `query_graph_entities`：按语义查找实体，适合先找候选实体。
- `expand_graph_subgraph`：从实体名扩展邻居和关系，适合关系探索。
- `get_entity_detail`：已知实体名时获取详情。

task #32 Phase A 后，Graph 元素响应会携带 `evidence_refs`。每个 ref 至少包含：

```json
{
  "document_id": "doc_abc",
  "chunk_id": "chunk_001",
  "parse_version": "optional"
}
```

这让 Agent 可以直接调用 `read_document_chunk(collection_id, document_id, chunk_id)` 读取证据原文。不要只依赖裸 `chunk_id`：chunk id 不是全局唯一，必须和 `document_id` 一起使用。

## 工具详情

### list_collections

列出当前 API Key 可访问的知识库。

**参数**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `cursor` | string | null | 分页游标 |
| `limit` | int | 50 | 每页数量 |
| `sort_by` | string | `created_at` | 排序字段：`created_at` / `updated_at` / `title` |
| `sort_order` | string | `desc` | `asc` / `desc` |
| `title_filter` | string | null | 标题过滤 |

### get_collection_metadata

读取单个知识库元数据，包括可用索引模式和文档数。

**参数**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `collection_id` | string | 必需 | 知识库 ID |

### list_documents

分页列出知识库内文档。

**参数**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `collection_id` | string | 必需 | 知识库 ID |
| `cursor` | string | null | 分页游标 |
| `limit` | int | 50 | 每页数量 |
| `sort_by` | string | `created_at` | `created_at` / `title` / `size_bytes` |
| `sort_order` | string | `desc` | `asc` / `desc` |
| `title_filter` | string | null | 标题过滤 |
| `type_filter` | list[string] | null | MIME type 过滤 |
| `indexed_only` | bool | false | 只返回已完成索引的文档 |

### get_document_metadata

读取单个文档元数据，包括索引状态、chunk 数、媒体类型。

**参数**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `collection_id` | string | 必需 | 知识库 ID |
| `document_id` | string | 必需 | 文档 ID |

### vector_search

向量语义检索，返回 chunk-level `SearchResult`。

**参数**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `collection_id` | string | 必需 | 知识库 ID |
| `query` | string | 必需 | 搜索问题 |
| `top_k` | int | 5 | 最大返回数量 |
| `similarity_threshold` | float | null | 最小相似度，null 表示使用知识库默认阈值 |

### fulltext_search

全文关键词检索，返回 chunk-level `SearchResult`。

**参数**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `collection_id` | string | 必需 | 知识库 ID |
| `query` | string | 必需 | 搜索问题 |
| `top_k` | int | 5 | 最大返回数量 |
| `keywords` | list[string] | null | 显式关键词；为空时由后端从 query 提取 |

### graph_search

图谱相关 chunk 检索，返回 chunk-level `SearchResult`。它适合直接获取原文证据，不适合作为实体/关系浏览工具。

**参数**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `collection_id` | string | 必需 | 知识库 ID |
| `query` | string | 必需 | 搜索问题 |
| `top_k` | int | 5 | 最大返回数量 |

### query_graph_entities

按自然语言在 Graph 实体上做语义查询，返回实体列表。

**参数**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `collection_id` | string | 必需 | 知识库 ID |
| `query` | string | 必需 | 实体查询 |
| `top_k` | int | 10 | 最大返回实体数 |

### expand_graph_subgraph

从一个或多个实体名扩展邻居和关系。

**参数**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `collection_id` | string | 必需 | 知识库 ID |
| `entity_names` | list[string] | 必需 | 起点实体名 |
| `hops` | int | 1 | 扩展跳数，后端会限制最大值和结果量 |

### get_entity_detail

按实体名读取单个 Graph 实体详情。

**参数**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `collection_id` | string | 必需 | 知识库 ID |
| `name` | string | 必需 | 实体规范名 |

### read_document_chunk

按 stable chunk id 读取原文 chunk。调用时必须同时提供 `document_id`，因为 `chunk_id` 不保证全局唯一。

**参数**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `collection_id` | string | 必需 | 知识库 ID |
| `document_id` | string | 必需 | 文档 ID |
| `chunk_id` | string | 必需 | Chunk ID |

### read_document

读取整篇解析后的 Markdown。可选 byte range 只是 best-effort，不保证跨 parse version 稳定。

**参数**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `collection_id` | string | 必需 | 知识库 ID |
| `document_id` | string | 必需 | 文档 ID |
| `range_start` | int | null | 起始 byte offset |
| `range_end` | int | null | 结束 byte offset |

### read_document_outline

读取文档标题树，用于章节导航。

**参数**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `collection_id` | string | 必需 | 知识库 ID |
| `document_id` | string | 必需 | 文档 ID |

### read_document_section

按 section path 或 heading anchor 读取章节。

**参数**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `collection_id` | string | 必需 | 知识库 ID |
| `document_id` | string | 必需 | 文档 ID |
| `section_path` | string | null | 章节路径，优先使用 |
| `heading_anchor` | string | null | 标题 anchor |

### web_search

搜索互联网内容。

**参数**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `query` | string | 必需 | 搜索关键词 |
| `top_k` | int | 5 | 返回结果数 |
| `source` | string | null | 指定域名或 URL |
| `timeout` | int | 30 | 超时时间（秒） |
| `locale` | string | `en-US` | 语言地区 |

### web_read

读取网页内容。

**参数**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `url_list` | list[string] | 必需 | URL 列表 |
| `timeout` | int | 30 | 超时时间（秒） |
| `locale` | string | `en-US` | 浏览器语言地区 |
| `max_concurrent` | int | 5 | 最大并发数 |

## 实战链路

### 从 Graph 实体回到原文证据

```python
# 1. 找到相关实体
entities = query_graph_entities(
    collection_id="col_1",
    query="哪些服务和 Kubernetes 部署有关？",
    top_k=5,
)

# 2. 扩展实体关系
subgraph = expand_graph_subgraph(
    collection_id="col_1",
    entity_names=[entities["entities"][0]["name"]],
    hops=1,
)

# 3. 读取实体或关系的原文证据
ref = entities["entities"][0]["evidence_refs"][0]
chunk = read_document_chunk(
    collection_id="col_1",
    document_id=ref["document_id"],
    chunk_id=ref["chunk_id"],
)
```

### 组合多种检索方式

```python
vector_hits = vector_search(collection_id="col_1", query="部署策略")
keyword_hits = fulltext_search(collection_id="col_1", query="kubelet readiness")
graph_hits = graph_search(collection_id="col_1", query="API 和 indexing worker 的关系")
```

Agent 可以同时看三类结果：vector 覆盖语义相近内容，fulltext 覆盖精确词，graph_search 覆盖图谱相关证据。ApeRAG 不再提供 rerank 参数；调用方应根据各工具返回的 `score`、`recall_type` 和证据内容自行组合。

## 注意事项

### 性能优化

1. **合理设置 top_k**：
   - 太大会增加上下文消耗
   - 太小可能遗漏重要证据
   - 推荐：5-10

2. **先检索，再精读**：
   - 先用 `vector_search` / `fulltext_search` / `graph_search` 找 chunk
   - 再用 `read_document_chunk` 或 `read_document_section` 读取原文
   - 只有长上下文模型或人工检查全文时才用 `read_document`

3. **Graph 工具按粒度选择**：
   - 要原文证据：用 `graph_search`
   - 要实体和关系：用 `query_graph_entities` + `expand_graph_subgraph`
   - 已知实体名：用 `get_entity_detail`

4. **超时设置**：
   - 图谱检索和网页读取可能较慢
   - 网络搜索建议 30-60 秒
   - 批量 URL 读取建议 60 秒以上

### 常见问题

**Q：搜索没有结果？**
- 检查知识库 ID 是否正确
- 确认知识库已完成索引构建
- 换一种检索方式交叉验证，例如 vector 无结果时尝试 fulltext 或 graph

**Q：Graph 实体结果如何读取原文？**
- 使用返回的 `evidence_refs[*].document_id` 和 `evidence_refs[*].chunk_id`
- 调用 `read_document_chunk(collection_id, document_id, chunk_id)`
- 不要只保存 `chunk_id`，它不是全局唯一

**Q：图片显示不了？**
- 检查 `metadata.indexer == "vision"`
- 使用 `asset://` 协议构建 URL
- 确保包含所有必需参数（asset_id、document_id、collection_id）

## 工具对比

| 工具 | 用途 | 适用场景 |
|------|------|---------|
| `list_collections` | 列出知识库 | 查看有哪些可用资源 |
| `vector_search` | 语义检索 | 模糊自然语言问题 |
| `fulltext_search` | 关键词检索 | 精确词、编号、短语 |
| `graph_search` | 图谱相关 chunk 检索 | 需要原文证据的 Graph RAG |
| `query_graph_entities` | 查询实体 | 找候选实体 |
| `expand_graph_subgraph` | 扩展关系 | 关系探索 |
| `read_document_chunk` | 读取 chunk | 读取证据原文 |
| `web_search` | 搜索互联网 | 获取实时信息或外部资料 |
| `web_read` | 读取网页 | 提取网页完整内容 |

## 相关链接

- **MCP 协议官网**: https://modelcontextprotocol.io/
- **ApeRAG GitHub**: https://github.com/apecloud/ApeRAG
- **API 文档**: http://localhost:8000/docs （本地部署）
