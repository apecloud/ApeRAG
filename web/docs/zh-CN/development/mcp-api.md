# MCP API 文档

## 简介

ApeRAG 提供全面的 [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) 支持，使 AI 助手和工具能够直接与你的知识库交互。MCP 服务器通过标准化协议提供强大的搜索功能、集合管理和网页浏览功能。

## 快速开始

### 配置

配置你的 MCP 客户端以连接到 ApeRAG：

```json
{
  "mcpServers": {
    "aperag-mcp": {
      "url": "https://rag.apecloud.com/mcp/",
      "headers": {
        "Authorization": "Bearer your-api-key-here"
      }
    }
  }
}
```

**重要提示**：
- 将 `https://rag.apecloud.com` 替换为你实际的 ApeRAG API URL
- 将 `your-api-key-here` 替换为从 ApeRAG 设置中获取的有效 API 密钥

### 身份验证

API 身份验证通过以下方法之一自动处理（按优先级顺序）：

1. **HTTP Authorization 头**：`Authorization: Bearer your-api-key`（HTTP 传输首选方式）
2. **环境变量**：`APERAG_API_KEY=your-api-key`（备用方法）

确保在 MCP 客户端中至少正确配置了一种身份验证方法。

## 可用工具

ApeRAG MCP 服务器提供以下工具：

### 1. list_collections

列出用户可访问的所有集合。

**参数**：无

**返回值**：
```json
{
  "items": [
    {
      "id": "collection-id",
      "title": "集合标题",
      "description": "集合描述"
    }
  ]
}
```

**示例**：
```python
collections = list_collections()
for collection in collections.items:
    print(f"{collection.title}: {collection.description}")
```

---

### 2. search_collection

使用向量搜索、全文搜索、图搜索、摘要搜索和/或视觉搜索在持久集合/知识库中搜索知识。

**主要用途**：这是搜索永久知识库的主要工具。用于一般问答、知识检索和访问有组织的知识集合。

**参数**：

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `collection_id` | string | *必需* | 要搜索的集合 ID |
| `query` | string | *必需* | 搜索查询 |
| `query_keywords` | list[str] | None | 从查询中提取的关键词用于全文搜索（可选） |
| `use_vector_index` | bool | True | 启用使用嵌入的向量/语义搜索 |
| `use_fulltext_index` | bool | True | 启用全文关键词搜索 |
| `use_graph_index` | bool | True | 启用知识图谱搜索 |
| `use_summary_index` | bool | True | 启用摘要搜索 |
| `use_vision_index` | bool | True | 启用视觉搜索 |
| `rerank` | bool | True | 启用 AI 驱动的重排序以提高相关性 |
| `topk` | int | 5 | 每种搜索类型返回的最大结果数 |

**返回值**：SearchResult 格式，结构如下：
```json
{
  "id": "search-id",
  "query": "你的查询",
  "items": [
    {
      "rank": 1,
      "score": 0.95,
      "content": "相关内容",
      "source": "文档名称",
      "recall_type": "vector_search",
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

**重要说明**：

- `metadata["page_idx"]` 表示内容来自文档的第 `page_idx` 页（从 0 开始）
- 全文搜索可能返回大量文本，可能导致较小 LLM 模型的上下文溢出
- 向量搜索结果可能包含通过多模态嵌入或视觉 LLM 描述索引的图像
- 如果 `metadata["indexer"]` 是 "vision"，该项是图像：
  - 空的 `content`：通过多模态嵌入检索
  - 非空的 `content`：包含图像的视觉描述
- 要在 markdown 中显示图像，构造 asset URL：
  ```python
  m = result.items[0].metadata
  if m.get("asset_id") and m.get("document_id") and m.get("collection_id") and m.get("mimetype"):
      asset_url = f"asset://{m['asset_id']}?document_id={m['document_id']}&collection_id={m['collection_id']}&mime_type={m['mimetype']}"
  ```

**示例**：
```python
# 启用所有方法搜索（默认）
results = search_collection(
    collection_id="abc123",
    query="如何部署应用程序？",
    use_vector_index=True,
    use_fulltext_index=True,
    use_graph_index=True,
    use_summary_index=True,
    use_vision_index=True,
    rerank=True,
    topk=5
)

# 仅使用向量和图搜索
results = search_collection(
    collection_id="abc123",
    query="部署策略",
    use_vector_index=True,
    use_fulltext_index=False,
    use_graph_index=True,
    use_summary_index=False,
    use_vision_index=False,
    rerank=True,
    topk=10
)
```

---

### 3. search_chat_files

仅在特定聊天会话中用户临时上传的文件中搜索。

**重要提示 - 何时使用此工具**：
- 仅当搜索在当前聊天对话中明确上传的文件时使用
- 用于临时的、会话特定的文档分析
- 当用户引用他们在当前聊天中分享的文档时

**不要使用此工具的情况**：
- 搜索一般知识库或集合（请改用 `search_collection`）
- 访问持久/永久知识库
- 不涉及聊天上传文件的一般问答

**参数**：

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `chat_id` | string | *必需* | 要搜索文件的聊天 ID |
| `query` | string | *必需* | 搜索查询 |
| `use_vector_index` | bool | True | 启用向量/语义搜索 |
| `use_fulltext_index` | bool | True | 启用全文关键词搜索 |
| `rerank` | bool | True | 启用重排序以提高相关性 |
| `topk` | int | 5 | 返回的最大结果数 |

**返回值**：SearchResult 格式（与 `search_collection` 结构相同）

**示例**：
```python
# 在聊天上传的文件中搜索
results = search_chat_files(
    chat_id="chat-123",
    query="预算分析",
    use_vector_index=True,
    use_fulltext_index=True,
    rerank=True,
    topk=5
)
```

---

### 4. web_search

使用多种搜索引擎执行网络搜索，支持高级域名定位和 LLM.txt 发现。

**参数**：

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `query` | string | "" | 常规网络搜索的搜索查询 |
| `max_results` | int | 5 | 返回的最大结果数 |
| `timeout` | int | 30 | 请求超时时间（秒） |
| `locale` | string | "en-US" | 浏览器语言环境 |
| `source` | string | "" | 用于站点特定过滤的可选域名或 URL |
| `search_llms_txt` | string | "" | LLM.txt 发现搜索的域名 |

**返回值**：
```json
{
  "results": [
    {
      "title": "页面标题",
      "url": "https://example.com/page",
      "snippet": "页面描述或摘录",
      "domain": "example.com"
    }
  ]
}
```

**搜索模式**：

1. **常规搜索**：提供 `query` 参数
   ```python
   results = web_search(query="ApeRAG RAG 系统 2025", max_results=5)
   ```

2. **站点特定搜索**：同时提供 `query` 和 `source`
   ```python
   results = web_search(
       query="部署文档",
       source="vercel.com",
       max_results=10
   )
   ```

3. **LLM.txt 发现**：提供 `search_llms_txt` 参数
   ```python
   results = web_search(
       search_llms_txt="anthropic.com",
       max_results=5
   )
   ```

4. **组合搜索**：同时使用常规和 LLM.txt 发现
   ```python
   results = web_search(
       query="机器学习教程",
       source="docs.python.org",
       search_llms_txt="openai.com",
       max_results=8
   )
   ```

---

### 5. web_read

读取并从网页中提取内容。

**参数**：

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `url_list` | list[str] | *必需* | 要读取的 URL 列表（单个 URL：使用包含一个元素的数组） |
| `timeout` | int | 30 | 请求超时时间（秒） |
| `locale` | string | "en-US" | 浏览器语言环境 |
| `max_concurrent` | int | 5 | 最大并发请求数 |

**返回值**：
```json
{
  "results": [
    {
      "status": "success",
      "url": "https://example.com/article",
      "title": "文章标题",
      "content": "提取的内容...",
      "word_count": 1234
    }
  ]
}
```

**示例**：
```python
# 读取单个 URL
content = web_read(
    url_list=["https://example.com/article"],
    timeout=30
)

# 读取多个 URL
content = web_read(
    url_list=[
        "https://example.com/page1",
        "https://example.com/page2"
    ],
    max_concurrent=2
)

# 处理结果
for result in content.results:
    if result.status == "success":
        print(f"标题: {result.title}")
        print(f"内容: {result.content}")
        print(f"字数: {result.word_count}")
```

---

## 可用资源

### aperag://usage-guide

提供 ApeRAG 搜索功能的综合使用指南，包括身份验证设置、搜索策略和示例工作流程。

访问此资源以获取以下详细文档：
- 可用操作
- 身份验证方法
- 快速入门指南
- 搜索类型和参数
- 示例工作流程
- 网络搜索和内容读取示例

---

## 可用提示

### search_assistant

提供有效 ApeRAG 搜索指导的帮助提示，包括：
- 如何使用搜索助手
- 可用功能
- 搜索技巧和最佳实践
- 身份验证设置
- 组合网络和内部搜索策略

---

## 完整工作流程示例

### 示例 1：基本知识搜索

```python
# 步骤 1：获取可用集合
collections = list_collections()

# 步骤 2：选择一个集合
collection_id = collections.items[0].id

# 步骤 3：使用默认设置搜索
results = search_collection(
    collection_id=collection_id,
    query="如何部署应用程序？",
    topk=5
)

# 步骤 4：处理结果
for item in results.items:
    print(f"[{item.recall_type}] {item.content[:100]}...")
    print(f"得分: {item.score}, 来源: {item.source}\n")
```

### 示例 2：组合网络和内部搜索

```python
# 1. 搜索网络以获取最新信息
web_results = web_search(
    query="最新 AI 发展 2025",
    source="anthropic.com",
    search_llms_txt="anthropic.com",
    max_results=3
)

# 2. 从搜索结果中提取 URL
urls = [result.url for result in web_results.results]

# 3. 读取这些页面的完整内容
web_content = web_read(url_list=urls, max_concurrent=2)

# 4. 搜索内部知识库
collections = list_collections()
if collections.items:
    internal_results = search_collection(
        collection_id=collections.items[0].id,
        query="AI 发展",
        rerank=True,
        topk=5
    )

# 5. 组合和分析结果
print("=== 网络结果 ===")
for result in web_results.results:
    print(f"[{result.domain}] {result.title}: {result.url}")

print("\n=== 网页内容 ===")
for content in web_content.results:
    if content.status == "success":
        print(f"📄 {content.title} ({content.word_count} 字)")

print("\n=== 内部知识 ===")
for item in internal_results.items:
    print(f"🔍 {item.content[:100]}...")
```

### 示例 3：带图像的多模态搜索

```python
# 启用视觉索引搜索
results = search_collection(
    collection_id="collection-id",
    query="架构图",
    use_vector_index=True,
    use_vision_index=True,
    topk=10
)

# 处理结果并显示图像
for item in results.items:
    if item.metadata.get("indexer") == "vision":
        m = item.metadata
        if m.get("asset_id") and m.get("document_id") and m.get("collection_id"):
            asset_url = f"asset://{m['asset_id']}?document_id={m['document_id']}&collection_id={m['collection_id']}&mime_type={m['mimetype']}"
            print(f"找到图像: {asset_url}")
            if item.content:
                print(f"描述: {item.content}")
```

---

## 最佳实践

### 搜索策略

1. **从广泛开始**：启用所有搜索类型以获得全面的结果
2. **根据需要优化**：如果需要聚焦结果，禁用特定搜索类型
3. **使用重排序**：保持 `rerank=True` 以获得更好的结果相关性（AI 驱动）
4. **调整 topk**：在结果数量和上下文窗口大小之间取得平衡
5. **注意上下文大小**：全文搜索可能返回大量文本

### 性能技巧

1. **并行操作**：组合网络搜索和内部搜索以获得全面覆盖
2. **批量 URL 读取**：使用 `web_read` 处理多个 URL 并设置 `max_concurrent` 参数
3. **缓存集合**：列出集合一次并重用集合 ID
4. **超时管理**：根据网络和内容大小调整超时值

### 安全考虑

1. **API 密钥保护**：永远不要在客户端代码中暴露 API 密钥
2. **授权头**：使用 HTTP Authorization 头进行安全传输
3. **集合访问**：用户只能访问他们有权限的集合
4. **速率限制**：在生产部署中遵守 API 速率限制

---

## 故障排除

### 身份验证问题

如果遇到身份验证错误：
1. 在 ApeRAG 设置中验证你的 API 密钥是否有效
2. 检查 HTTP Authorization 头格式：`Bearer your-api-key`
3. 如果不使用头，确保设置了环境变量 `APERAG_API_KEY`

### 搜索问题

如果搜索结果不符合预期：
1. 尝试不同的搜索类型组合
2. 启用/禁用重排序以查看差异
3. 调整 `topk` 值以获得更多或更少的结果
4. 使用更具体的查询以获得更好的结果

### 超时错误

如果遇到超时错误：
1. 增加网络操作的 `timeout` 参数
2. 图搜索可能耗时较长 - 请耐心等待
3. 减少 `web_read` 中的并发请求数

---

## API 参考摘要

| 工具 | 主要用途 | 关键参数 |
|------|---------|---------|
| `list_collections` | 浏览可用集合 | 无 |
| `search_collection` | 搜索持久知识库 | `collection_id`、`query`、搜索类型、`topk` |
| `search_chat_files` | 搜索聊天会话文件 | `chat_id`、`query`、搜索类型 |
| `web_search` | 搜索网络 | `query`、`source`、`search_llms_txt` |
| `web_read` | 提取网页内容 | `url_list`、`timeout` |

更多示例和详细的 API 文档，请访问：http://localhost:8000/docs
