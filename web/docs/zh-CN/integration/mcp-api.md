---
title: MCP API
description: Model Context Protocol API 文档
---

# MCP API

ApeRAG 通过 [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) 对外提供标准化的工具接口，让 AI 助手（Claude Desktop、Cursor、Dify 等）能够直接访问你的知识库。

## 快速开始

### 配置示例

以 Claude Desktop 为例，在配置文件中添加：

```json
{
  "mcpServers": {
    "aperag": {
      "url": "https://rag.apecloud.com/mcp/",
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

> **获取 API Key**：登录 ApeRAG 后，在设置页面创建或复制你的 API Key

## 可用工具

### 1. list_collections

列出所有可访问的知识库。

**参数**：无

**返回**：
```json
{
  "items": [
    {
      "id": "collection-id",
      "title": "知识库标题",
      "description": "知识库描述"
    }
  ]
}
```

### 2. search_collection

在知识库中搜索，支持多种检索方式。

**核心参数**：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `collection_id` | string | 必需 | 知识库 ID |
| `query` | string | 必需 | 搜索问题 |
| `use_vector_index` | bool | true | 向量检索（语义搜索） |
| `use_fulltext_index` | bool | true | 全文检索（关键词匹配） |
| `use_graph_index` | bool | true | 图谱检索（关系查询） |
| `use_summary_index` | bool | true | 摘要检索 |
| `use_vision_index` | bool | true | 视觉检索（图片搜索） |
| `rerank` | bool | true | AI 重排序 |
| `topk` | int | 5 | 每种方式返回的结果数 |

**返回格式**：
```json
{
  "query": "你的问题",
  "items": [
    {
      "rank": 1,
      "score": 0.95,
      "content": "相关内容片段",
      "source": "文档名称",
      "recall_type": "vector_search|graph_search|fulltext_search|summary_search",
      "metadata": {
        "page_idx": 0,
        "document_id": "doc-id",
        "collection_id": "col-id",
        "indexer": "text|vision"
      }
    }
  ]
}
```

**图片处理**：

如果 `metadata.indexer == "vision"`，表示这是一张图片：
- `content` 为空：通过多模态向量检索
- `content` 不为空：包含图片的文字描述

显示图片的 URL 格式：
```python
m = item.metadata
asset_url = f"asset://{m['asset_id']}?document_id={m['document_id']}&collection_id={m['collection_id']}&mime_type={m['mimetype']}"
```

**使用示例**：

```python
# 默认搜索（推荐）- 启用所有检索方式
results = search_collection(
    collection_id="abc123",
    query="如何部署应用？"
)

# 仅向量+图谱检索
results = search_collection(
    collection_id="abc123",
    query="部署策略",
    use_vector_index=True,
    use_fulltext_index=False,
    use_graph_index=True,
    use_summary_index=False,
    topk=10
)
```

### 3. search_chat_files

在聊天会话的临时文件中搜索。

**何时使用**：
- ✅ 用户在当前对话中上传了文件
- ✅ 需要分析聊天中的临时文档
- ❌ 不要用于搜索持久化的知识库（应该用 `search_collection`）

**参数**：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `chat_id` | string | 必需 | 聊天 ID |
| `query` | string | 必需 | 搜索问题 |
| `use_vector_index` | bool | true | 向量检索 |
| `use_fulltext_index` | bool | true | 全文检索 |
| `rerank` | bool | true | 重排序 |
| `topk` | int | 5 | 返回结果数 |

**返回格式**：与 `search_collection` 相同

### 4. web_search

搜索互联网内容。

**参数**：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `query` | string | "" | 搜索关键词 |
| `max_results` | int | 5 | 返回结果数 |
| `source` | string | "" | 指定域名（如 `vercel.com`） |
| `search_llms_txt` | string | "" | 搜索域名的 LLM.txt 文件 |
| `timeout` | int | 30 | 超时时间（秒） |
| `locale` | string | "en-US" | 语言地区 |

**使用模式**：

```python
# 常规搜索
web_search(query="ApeRAG 2025")

# 指定网站搜索
web_search(query="部署文档", source="vercel.com")

# LLM.txt 发现
web_search(search_llms_txt="anthropic.com")
```

### 5. web_read

读取网页内容。

**参数**：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `url_list` | list[str] | 必需 | URL 列表 |
| `timeout` | int | 30 | 超时时间（秒） |
| `max_concurrent` | int | 5 | 最大并发数 |

**返回**：
```json
{
  "results": [
    {
      "status": "success",
      "url": "https://example.com",
      "title": "页面标题",
      "content": "提取的文本内容",
      "word_count": 1234
    }
  ]
}
```

**示例**：
```python
# 读取单个页面
web_read(url_list=["https://example.com/article"])

# 批量读取
web_read(
    url_list=["https://example.com/page1", "https://example.com/page2"],
    max_concurrent=2
)
```

## 实战示例

### 示例 1：知识库问答

```python
# 1. 列出所有知识库
collections = list_collections()

# 2. 选择一个知识库
collection_id = collections.items[0].id

# 3. 搜索（默认启用所有检索方式）
results = search_collection(
    collection_id=collection_id,
    query="如何优化性能？"
)

# 4. 处理结果
for item in results.items:
    print(f"[{item.recall_type}] {item.content}")
    print(f"来源: {item.source}, 得分: {item.score}\n")
```

### 示例 2：图谱可视化

```python
# 搜索并获取实体关系数据
results = search_collection(
    collection_id="abc123",
    query="刘备和诸葛亮的关系",
    use_graph_index=True  # 确保启用图谱检索
)

# 检查是否包含图谱数据
if results.graph_search and results.graph_search.entities:
    print("实体:", results.graph_search.entities)
    print("关系:", results.graph_search.relationships)
    # 可以用这些数据生成知识图谱可视化
```

### 示例 3：混合搜索（知识库 + 互联网）

```python
# 1. 搜索互联网
web_results = web_search(query="最新 AI 发展", max_results=3)
urls = [r.url for r in web_results.results]

# 2. 读取网页内容
web_content = web_read(url_list=urls)

# 3. 搜索内部知识库
kb_results = search_collection(
    collection_id="ai-knowledge",
    query="AI 发展趋势"
)

# 4. 综合分析
print("=== 互联网资讯 ===")
for r in web_results.results:
    print(f"{r.title}: {r.url}")

print("\n=== 内部知识 ===")
for item in kb_results.items:
    print(f"{item.content[:100]}...")
```

## 注意事项

### 性能优化

1. **合理设置 topk**：
   - 太大会增加 LLM 上下文消耗
   - 太小可能遗漏重要信息
   - 推荐：5-10

2. **选择性启用检索**：
   - 不是所有查询都需要全文检索
   - 全文检索可能返回大量文本
   - 根据问题类型选择合适的检索方式

3. **超时设置**：
   - 图谱检索可能较慢（默认 120 秒）
   - 网络搜索建议 30-60 秒
   - 批量 URL 读取建议 60 秒以上

### 常见问题

**Q: 搜索没有结果？**
- 检查知识库 ID 是否正确
- 确认知识库已完成索引构建
- 尝试不同的检索方式组合

**Q: 图谱数据为空？**
- 确认知识库启用了 Graph 索引
- 某些简单文档可能不包含明显的实体关系

**Q: 图片显示不了？**
- 检查 `metadata.indexer == "vision"`
- 使用 `asset://` 协议构建 URL
- 确保包含所有必需参数（asset_id、document_id、collection_id）

## 工具对比

| 工具 | 用途 | 适用场景 |
|------|------|---------|
| `list_collections` | 列出知识库 | 查看有哪些可用资源 |
| `search_collection` | 搜索知识库 | 主要搜索工具，用于持久化知识 |
| `search_chat_files` | 搜索聊天文件 | 分析用户在对话中上传的临时文件 |
| `web_search` | 搜索互联网 | 获取实时信息或外部资料 |
| `web_read` | 读取网页 | 提取网页完整内容 |

## 相关链接

- **MCP 协议官网**: https://modelcontextprotocol.io/
- **ApeRAG GitHub**: https://github.com/apecloud/ApeRAG
- **API 文档**: http://localhost:8000/docs （本地部署）
