# MCP API Documentation

## Introduction

ApeRAG provides comprehensive [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) support, enabling AI assistants and tools to interact directly with your knowledge base. The MCP server exposes powerful search capabilities, collection management, and web browsing features through a standardized protocol.

## Getting Started

### Configuration

Configure your MCP client to connect to ApeRAG:

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

**Important**: 
- Replace `https://rag.apecloud.com` with your actual ApeRAG API URL
- Replace `your-api-key-here` with a valid API key from your ApeRAG settings

### Authentication

API authentication is handled automatically through one of these methods (in priority order):

1. **HTTP Authorization header**: `Authorization: Bearer your-api-key` (preferred for HTTP transport)
2. **Environment variable**: `APERAG_API_KEY=your-api-key` (fallback method)

Make sure at least one authentication method is properly configured in your MCP client.

## Available Tools

The ApeRAG MCP server provides the following tools:

### 1. list_collections

List all collections available to the user.

**Parameters**: None

**Returns**:
```json
{
  "items": [
    {
      "id": "collection-id",
      "title": "Collection Title",
      "description": "Collection description"
    }
  ]
}
```

**Example**:
```python
collections = list_collections()
for collection in collections.items:
    print(f"{collection.title}: {collection.description}")
```

---

### 2. search_collection

Search for knowledge in a persistent collection/knowledge base using vector, full-text, graph, summary, and/or vision search.

**PRIMARY USE CASE**: This is the main tool for searching permanent knowledge repositories. Use this for general Q&A, knowledge retrieval, and accessing organized knowledge collections.

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `collection_id` | string | *required* | The ID of the collection to search in |
| `query` | string | *required* | The search query |
| `query_keywords` | list[str] | None | Keywords extracted from query for fulltext search (optional) |
| `use_vector_index` | bool | True | Enable vector/semantic search using embeddings |
| `use_fulltext_index` | bool | True | Enable full-text keyword search |
| `use_graph_index` | bool | True | Enable knowledge graph search |
| `use_summary_index` | bool | True | Enable summary search |
| `use_vision_index` | bool | True | Enable vision search |
| `rerank` | bool | True | Enable AI-powered reranking for better relevance |
| `topk` | int | 5 | Maximum number of results to return per search type |

**Returns**: SearchResult format with the following structure:
```json
{
  "id": "search-id",
  "query": "your query",
  "items": [
    {
      "rank": 1,
      "score": 0.95,
      "content": "relevant content",
      "source": "document name",
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

**Important Notes**:

- `metadata["page_idx"]` indicates the content is from page `page_idx` of the document (0-indexed)
- Full-text search can return large amounts of text which may cause context overflow with smaller LLM models
- Vector search results may include images indexed via multimodal embeddings or vision LLM descriptions
- If `metadata["indexer"]` is "vision", the item is an image:
  - Empty `content`: retrieved via multimodal embedding
  - Non-empty `content`: contains visual description of the image
- To display images in markdown, construct asset URLs:
  ```python
  m = result.items[0].metadata
  if m.get("asset_id") and m.get("document_id") and m.get("collection_id") and m.get("mimetype"):
      asset_url = f"asset://{m['asset_id']}?document_id={m['document_id']}&collection_id={m['collection_id']}&mime_type={m['mimetype']}"
  ```

**Example**:
```python
# Search with all methods enabled (default)
results = search_collection(
    collection_id="abc123",
    query="How to deploy applications?",
    use_vector_index=True,
    use_fulltext_index=True,
    use_graph_index=True,
    use_summary_index=True,
    use_vision_index=True,
    rerank=True,
    topk=5
)

# Search with only vector and graph methods
results = search_collection(
    collection_id="abc123",
    query="deployment strategies",
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

Search ONLY within files temporarily uploaded by the user in a specific chat session.

**IMPORTANT - When to Use This Tool**:
- ONLY when searching files explicitly uploaded in the current chat conversation
- For temporary, session-specific document analysis
- When the user references documents they shared in the current chat

**DO NOT Use This Tool For**:
- Searching general knowledge bases or collections (use `search_collection` instead)
- Accessing persistent/permanent knowledge repositories
- General Q&A that doesn't involve chat-uploaded files

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `chat_id` | string | *required* | The ID of the chat to search files in |
| `query` | string | *required* | The search query |
| `use_vector_index` | bool | True | Enable vector/semantic search |
| `use_fulltext_index` | bool | True | Enable full-text keyword search |
| `rerank` | bool | True | Enable reranking for better relevance |
| `topk` | int | 5 | Maximum number of results to return |

**Returns**: SearchResult format (same structure as `search_collection`)

**Example**:
```python
# Search in chat-uploaded files
results = search_chat_files(
    chat_id="chat-123",
    query="budget analysis",
    use_vector_index=True,
    use_fulltext_index=True,
    rerank=True,
    topk=5
)
```

---

### 4. web_search

Perform web search using various search engines with advanced domain targeting and LLM.txt discovery.

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | string | "" | Search query for regular web search |
| `max_results` | int | 5 | Maximum number of results to return |
| `timeout` | int | 30 | Request timeout in seconds |
| `locale` | string | "en-US" | Browser locale |
| `source` | string | "" | Optional domain or URL for site-specific filtering |
| `search_llms_txt` | string | "" | Domain for LLM.txt discovery search |

**Returns**:
```json
{
  "results": [
    {
      "title": "Page Title",
      "url": "https://example.com/page",
      "snippet": "Page description or excerpt",
      "domain": "example.com"
    }
  ]
}
```

**Search Modes**:

1. **Regular Search**: Provide `query` parameter
   ```python
   results = web_search(query="ApeRAG RAG system 2025", max_results=5)
   ```

2. **Site-specific Search**: Provide both `query` and `source`
   ```python
   results = web_search(
       query="deployment documentation",
       source="vercel.com",
       max_results=10
   )
   ```

3. **LLM.txt Discovery**: Provide `search_llms_txt` parameter
   ```python
   results = web_search(
       search_llms_txt="anthropic.com",
       max_results=5
   )
   ```

4. **Combined Search**: Use both regular and LLM.txt discovery
   ```python
   results = web_search(
       query="machine learning tutorials",
       source="docs.python.org",
       search_llms_txt="openai.com",
       max_results=8
   )
   ```

---

### 5. web_read

Read and extract content from web pages.

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `url_list` | list[str] | *required* | List of URLs to read (single URL: use array with one element) |
| `timeout` | int | 30 | Request timeout in seconds |
| `locale` | string | "en-US" | Browser locale |
| `max_concurrent` | int | 5 | Maximum concurrent requests |

**Returns**:
```json
{
  "results": [
    {
      "status": "success",
      "url": "https://example.com/article",
      "title": "Article Title",
      "content": "Extracted content...",
      "word_count": 1234
    }
  ]
}
```

**Example**:
```python
# Read single URL
content = web_read(
    url_list=["https://example.com/article"],
    timeout=30
)

# Read multiple URLs
content = web_read(
    url_list=[
        "https://example.com/page1",
        "https://example.com/page2"
    ],
    max_concurrent=2
)

# Process results
for result in content.results:
    if result.status == "success":
        print(f"Title: {result.title}")
        print(f"Content: {result.content}")
        print(f"Word Count: {result.word_count}")
```

---

## Available Resources

### aperag://usage-guide

Provides a comprehensive usage guide for ApeRAG search capabilities, including authentication setup, search strategies, and example workflows.

Access this resource to get detailed documentation about:
- Available operations
- Authentication methods
- Quick start guide
- Search types and parameters
- Example workflows
- Web search and content reading examples

---

## Available Prompts

### search_assistant

A helper prompt that provides guidance for effective ApeRAG searching, including:
- How to use the search assistant
- Available capabilities
- Search tips and best practices
- Authentication setup
- Combined web and internal search strategies

---

## Complete Workflow Examples

### Example 1: Basic Knowledge Search

```python
# Step 1: Get available collections
collections = list_collections()

# Step 2: Choose a collection
collection_id = collections.items[0].id

# Step 3: Search with default settings
results = search_collection(
    collection_id=collection_id,
    query="How to deploy applications?",
    topk=5
)

# Step 4: Process results
for item in results.items:
    print(f"[{item.recall_type}] {item.content[:100]}...")
    print(f"Score: {item.score}, Source: {item.source}\n")
```

### Example 2: Combined Web and Internal Search

```python
# 1. Search web for recent information
web_results = web_search(
    query="latest AI developments 2025",
    source="anthropic.com",
    search_llms_txt="anthropic.com",
    max_results=3
)

# 2. Extract URLs from search results
urls = [result.url for result in web_results.results]

# 3. Read full content from those pages
web_content = web_read(url_list=urls, max_concurrent=2)

# 4. Search internal knowledge base
collections = list_collections()
if collections.items:
    internal_results = search_collection(
        collection_id=collections.items[0].id,
        query="AI developments",
        rerank=True,
        topk=5
    )

# 5. Combine and analyze results
print("=== Web Results ===")
for result in web_results.results:
    print(f"[{result.domain}] {result.title}: {result.url}")

print("\n=== Web Content ===")
for content in web_content.results:
    if content.status == "success":
        print(f"📄 {content.title} ({content.word_count} words)")

print("\n=== Internal Knowledge ===")
for item in internal_results.items:
    print(f"🔍 {item.content[:100]}...")
```

### Example 3: Multi-Modal Search with Images

```python
# Search with vision index enabled
results = search_collection(
    collection_id="collection-id",
    query="architecture diagrams",
    use_vector_index=True,
    use_vision_index=True,
    topk=10
)

# Process results and display images
for item in results.items:
    if item.metadata.get("indexer") == "vision":
        m = item.metadata
        if m.get("asset_id") and m.get("document_id") and m.get("collection_id"):
            asset_url = f"asset://{m['asset_id']}?document_id={m['document_id']}&collection_id={m['collection_id']}&mime_type={m['mimetype']}"
            print(f"Image found: {asset_url}")
            if item.content:
                print(f"Description: {item.content}")
```

---

## Best Practices

### Search Strategy

1. **Start Broad**: Use all search types enabled for comprehensive results
2. **Refine as Needed**: Disable specific search types if you need focused results
3. **Use Reranking**: Keep `rerank=True` for better result relevance (AI-powered)
4. **Adjust topk**: Balance between result quantity and context window size
5. **Watch Context Size**: Full-text search can return large amounts of text

### Performance Tips

1. **Parallel Operations**: Combine web search and internal search for comprehensive coverage
2. **Batch URL Reading**: Use `web_read` with multiple URLs and `max_concurrent` parameter
3. **Cache Collections**: List collections once and reuse collection IDs
4. **Timeout Management**: Adjust timeout values based on your network and content size

### Security Considerations

1. **API Key Protection**: Never expose API keys in client-side code
2. **Authorization Headers**: Use HTTP Authorization header for secure transmission
3. **Collection Access**: Users can only access collections they have permission for
4. **Rate Limiting**: Respect API rate limits for production deployments

---

## Troubleshooting

### Authentication Issues

If you encounter authentication errors:
1. Verify your API key is valid in ApeRAG settings
2. Check HTTP Authorization header format: `Bearer your-api-key`
3. Ensure environment variable `APERAG_API_KEY` is set if not using headers

### Search Issues

If search results are not as expected:
1. Try different combinations of search types
2. Enable/disable reranking to see the difference
3. Adjust `topk` values to get more or fewer results
4. Use more specific queries for better results

### Timeout Errors

If you encounter timeout errors:
1. Increase the `timeout` parameter for web operations
2. Graph search can be time-consuming - be patient
3. Reduce the number of concurrent requests in `web_read`

---

## API Reference Summary

| Tool | Primary Use | Key Parameters |
|------|-------------|----------------|
| `list_collections` | Browse available collections | None |
| `search_collection` | Search persistent knowledge base | `collection_id`, `query`, search types, `topk` |
| `search_chat_files` | Search chat session files | `chat_id`, `query`, search types |
| `web_search` | Search the web | `query`, `source`, `search_llms_txt` |
| `web_read` | Extract web content | `url_list`, `timeout` |

For more examples and detailed API documentation, visit: http://localhost:8000/docs
