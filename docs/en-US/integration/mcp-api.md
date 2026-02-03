---
title: MCP API
description: Model Context Protocol API Documentation
---

# MCP API

ApeRAG provides standardized tool interfaces through [Model Context Protocol (MCP)](https://modelcontextprotocol.io/), allowing AI assistants (Claude Desktop, Cursor, Dify, etc.) to directly access your knowledge bases.

## Quick Start

### Configuration Example

For Claude Desktop, add to configuration file:

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

### Authentication

Two authentication methods supported (by priority):

1. **HTTP Authorization Header** (Recommended): `Authorization: Bearer your-api-key`
2. **Environment Variable** (Fallback): `APERAG_API_KEY=your-api-key`

> **Get API Key**: Login to ApeRAG, create or copy your API Key from settings

## Available Tools

### 1. list_collections

List all accessible knowledge bases.

**Parameters**: None

**Returns**:
```json
{
  "items": [
    {
      "id": "collection-id",
      "title": "Collection Title",
      "description": "Collection Description"
    }
  ]
}
```

### 2. search_collection

Search in knowledge bases with multiple retrieval methods.

**Core Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `collection_id` | string | Required | Knowledge base ID |
| `query` | string | Required | Search query |
| `use_vector_index` | bool | true | Vector retrieval (semantic search) |
| `use_fulltext_index` | bool | true | Full-text retrieval (keyword matching) |
| `use_graph_index` | bool | true | Graph retrieval (relation query) |
| `use_summary_index` | bool | true | Summary retrieval |
| `use_vision_index` | bool | true | Vision retrieval (image search) |
| `rerank` | bool | true | AI reranking |
| `topk` | int | 5 | Results per method |

**Return Format**:
```json
{
  "query": "your question",
  "items": [
    {
      "rank": 1,
      "score": 0.95,
      "content": "relevant content",
      "source": "document name",
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

**Image Handling**:

If `metadata.indexer == "vision"`, it's an image:
- Empty `content`: Retrieved via multimodal vector
- Non-empty `content`: Contains image description

Image URL format:
```python
m = item.metadata
asset_url = f"asset://{m['asset_id']}?document_id={m['document_id']}&collection_id={m['collection_id']}&mime_type={m['mimetype']}"
```

**Usage Examples**:

```python
# Default search (recommended) - all methods enabled
results = search_collection(
    collection_id="abc123",
    query="How to deploy applications?"
)

# Vector + Graph only
results = search_collection(
    collection_id="abc123",
    query="deployment strategies",
    use_vector_index=True,
    use_fulltext_index=False,
    use_graph_index=True,
    use_summary_index=False,
    topk=10
)
```

### 3. search_chat_files

Search in temporary files from chat session.

**When to Use**:
- ✅ User uploaded files in current conversation
- ✅ Analyzing temporary documents in chat
- ❌ Don't use for persistent knowledge bases (use `search_collection`)

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `chat_id` | string | Required | Chat ID |
| `query` | string | Required | Search query |
| `use_vector_index` | bool | true | Vector retrieval |
| `use_fulltext_index` | bool | true | Full-text retrieval |
| `rerank` | bool | true | Reranking |
| `topk` | int | 5 | Results count |

**Return Format**: Same as `search_collection`

### 4. web_search

Search the internet.

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | string | "" | Search keywords |
| `max_results` | int | 5 | Results count |
| `source` | string | "" | Specific domain (e.g., `vercel.com`) |
| `search_llms_txt` | string | "" | Search domain's LLM.txt file |
| `timeout` | int | 30 | Timeout (seconds) |
| `locale` | string | "en-US" | Language locale |

**Usage Patterns**:

```python
# Regular search
web_search(query="ApeRAG 2025")

# Site-specific search
web_search(query="deployment docs", source="vercel.com")

# LLM.txt discovery
web_search(search_llms_txt="anthropic.com")
```

### 5. web_read

Read webpage content.

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `url_list` | list[str] | Required | URL list |
| `timeout` | int | 30 | Timeout (seconds) |
| `max_concurrent` | int | 5 | Max concurrent requests |

**Returns**:
```json
{
  "results": [
    {
      "status": "success",
      "url": "https://example.com",
      "title": "Page Title",
      "content": "Extracted text",
      "word_count": 1234
    }
  ]
}
```

**Example**:
```python
# Read single page
web_read(url_list=["https://example.com/article"])

# Batch read
web_read(
    url_list=["https://example.com/page1", "https://example.com/page2"],
    max_concurrent=2
)
```

## Practical Examples

### Example 1: Knowledge Base Q&A

```python
# 1. List all knowledge bases
collections = list_collections()

# 2. Select a knowledge base
collection_id = collections.items[0].id

# 3. Search (all methods enabled by default)
results = search_collection(
    collection_id=collection_id,
    query="How to optimize performance?"
)

# 4. Process results
for item in results.items:
    print(f"[{item.recall_type}] {item.content}")
    print(f"Source: {item.source}, Score: {item.score}\n")
```

### Example 2: Graph Visualization

```python
# Search with graph retrieval
results = search_collection(
    collection_id="abc123",
    query="relationship between Liu Bei and Zhuge Liang",
    use_graph_index=True
)

# Check for graph data
if results.graph_search and results.graph_search.entities:
    print("Entities:", results.graph_search.entities)
    print("Relationships:", results.graph_search.relationships)
    # Use this data to generate knowledge graph visualization
```

### Example 3: Hybrid Search (KB + Web)

```python
# 1. Search web
web_results = web_search(query="latest AI developments", max_results=3)
urls = [r.url for r in web_results.results]

# 2. Read web content
web_content = web_read(url_list=urls)

# 3. Search internal KB
kb_results = search_collection(
    collection_id="ai-knowledge",
    query="AI development trends"
)

# 4. Synthesize
print("=== Web Results ===")
for r in web_results.results:
    print(f"{r.title}: {r.url}")

print("\n=== Internal Knowledge ===")
for item in kb_results.items:
    print(f"{item.content[:100]}...")
```

## Best Practices

### Performance Tips

1. **Reasonable topk**:
   - Too large increases LLM context consumption
   - Too small may miss important information
   - Recommended: 5-10

2. **Selective Retrieval**:
   - Not all queries need full-text search
   - Full-text may return large amounts of text
   - Choose methods based on query type

3. **Timeout Settings**:
   - Graph retrieval may be slow (default 120s)
   - Web search: 30-60s recommended
   - Batch URL read: 60s+ recommended

### Common Issues

**Q: No search results?**
- Check if collection ID is correct
- Confirm knowledge base indexing is complete
- Try different retrieval method combinations

**Q: Graph data empty?**
- Confirm knowledge base has Graph index enabled
- Simple documents may not contain obvious entity relationships

**Q: Images not showing?**
- Check `metadata.indexer == "vision"`
- Use `asset://` protocol for URL
- Ensure all required parameters included (asset_id, document_id, collection_id)

## Tool Comparison

| Tool | Purpose | Use Case |
|------|---------|----------|
| `list_collections` | List knowledge bases | See available resources |
| `search_collection` | Search knowledge base | Primary search tool for persistent knowledge |
| `search_chat_files` | Search chat files | Analyze temporary files uploaded in conversation |
| `web_search` | Search internet | Get real-time or external information |
| `web_read` | Read webpage | Extract full webpage content |

## Related Links

- **MCP Protocol**: https://modelcontextprotocol.io/
- **ApeRAG GitHub**: https://github.com/apecloud/ApeRAG
- **API Docs**: http://localhost:8000/docs (local deployment)
