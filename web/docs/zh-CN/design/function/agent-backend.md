---
title: Agent接口设计
description:
keywords:
---

# ApeRAG Agent 后端接口设计方案

## 1. 设计概述

基于现有ApeRAG项目架构，为Agent功能设计一套完整的后端接口系统。Agent作为一个独立的智能对话助手，需要支持Web搜索、模型切换等功能，并提供流畅的对话体验。集成现有的MCP接口进行collection搜索和管理。

## 2. 接口架构设计

### 2.1 接口路径规划

根据现有API设计模式，分为两大模块：

```
/api/v1/agent/
└── chats/                          # Agent对话管理

/api/v1/web/
├── search                          # Web搜索
└── read                            # Web内容读取
```

### 2.2 数据流架构

```
Frontend → Agent API → Agent Service → [
    MCP Service (collection搜索，由Agent后端调用)
    Web Service (网络搜索和内容读取)
    LLM Service (模型调用)
    Chat Service (对话历史)
]
```

## 3. 接口详细设计

### 3.1 Agent对话管理接口

#### 3.1.1 创建Agent对话

```
POST /api/v1/agent/chats
```

**请求体**：

```json
{
  "title": "新对话" // 可选，默认自动生成
}
```

**响应**：

```json
{
  "id": "chat_12345",
  "title": "新对话",
  "created": "2025-01-07T10:00:00Z",
  "updated": "2025-01-07T10:00:00Z"
}
```

#### 3.1.2 获取对话列表

```
GET /api/v1/agent/chats
```

**响应**：

```json
{
  "items": [
    {
      "id": "chat_12345",
      "title": "技术问题讨论",
      "created": "2025-01-07T10:00:00Z",
      "updated": "2025-01-07T10:00:00Z"
    }
  ]
}
```

#### 3.1.3 获取对话详情

```
GET /api/v1/agent/chats/{chat_id}
```

**响应**：

```json
{
  "id": "chat_12345",
  "title": "技术问题讨论",
  "created": "2025-01-07T10:00:00Z",
  "updated": "2025-01-07T10:00:00Z",
  "messages": [
    {
      "id": "msg_67890",
      "type": "user",
      "content": "请介绍一下ApeRAG的架构",
      "web_search_enabled": false,
      "model_used": "claude-3-5-sonnet",
      "timestamp": "2025-01-07T10:01:00Z"
    },
    {
      "id": "msg_67891",
      "type": "assistant",
      "content": "ApeRAG是一个...",
      "sources": [
        {
          "collection_id": "col_1",
          "collection_name": "技术文档",
          "score": 0.95,
          "text": "相关文档内容...",
          "metadata": { "source": "doc1.pdf" }
        }
      ],
      "web_search_results": null,
      "model_used": "claude-3-5-sonnet",
      "timestamp": "2025-01-07T10:01:05Z"
    }
  ]
}
```

#### 3.1.4 发送消息

```
POST /api/v1/agent/chats/{chat_id}/messages
```

**请求体**：

```json
{
  "content": "请介绍一下ApeRAG的架构",
  "model_id": "claude-3-5-sonnet", // 可选，使用默认模型
  "web_search_enabled": false, // 可选，默认false
  "stream": true // 可选，是否流式响应
}
```

**流式响应**（Server-Sent Events）：

```
data: {"type": "start", "message_id": "msg_67890"}

data: {"type": "content", "content": "ApeRAG是一个"}

data: {"type": "content", "content": "强大的"}

data: {"type": "sources", "sources": [...]}

data: {"type": "end", "message_id": "msg_67890"}
```

**非流式响应**：

```json
{
  "id": "msg_67890",
  "content": "ApeRAG是一个强大的RAG系统...",
  "sources": [
    {
      "collection_id": "col_1",
      "collection_name": "技术文档",
      "score": 0.95,
      "text": "相关文档内容...",
      "metadata": { "source": "doc1.pdf" }
    }
  ],
  "web_search_results": null,
  "model_used": "claude-3-5-sonnet",
  "created": "2025-01-07T10:01:05Z"
}
```

### 3.2 Web服务接口

#### 3.2.1 Web搜索接口

**HTTP接口**：

```
POST /api/v1/web/search
```

**MCP工具**：

```
web_search(query, max_results, search_engine, ...)
```

参考[JINA Reader API](https://jina.ai/reader)的`s.jina.ai`设计。

**请求体**：

```json
{
  "query": "ApeRAG 2025年最新发展",
  "max_results": 5, // 可选，默认5
  "search_engine": "google", // 可选，默认google
  "timeout": 30, // 可选，超时时间（秒）
  "locale": "zh-CN" // 可选，浏览器语言
}
```

**响应**：

```json
{
  "query": "ApeRAG 2025年最新发展",
  "results": [
    {
      "rank": 1,
      "title": "ApeRAG 2025年技术路线图",
      "url": "https://example.com/aperag-2025-roadmap",
      "snippet": "ApeRAG在2025年将重点发展...",
      "domain": "example.com",
      "timestamp": "2025-01-01T00:00:00Z"
    }
  ],
  "search_engine": "google",
  "total_results": 1250,
  "search_time": 1.2
}
```

#### 3.2.2 Web内容读取接口

**HTTP接口**：

```
POST /api/v1/web/read
```

**MCP工具**：

```
web_read(urls, timeout, ...)
```

参考[JINA Reader API](https://jina.ai/reader)的`r.jina.ai`设计。

**请求体**：

```json
{
  "urls": [
    // 支持单个URL字符串或URL数组
    "https://example.com/aperag-2025-roadmap",
    "https://example.com/another-page"
  ],
  "timeout": 30, // 可选，超时时间
  "css_selector": null, // 可选，CSS选择器，提取特定内容
  "wait_for_selector": null, // 可选，等待选择器，适用于SPA页面
  "exclude_selector": null, // 可选，排除选择器，去掉广告等无用内容
  "bypass_cache": false, // 可选，绕过缓存，获取最新内容
  "locale": "zh-CN", // 可选，浏览器语言
  "max_concurrent": 3 // 可选，最大并发数（仅对多个URL有效）
}
```

**响应**：

```json
{
  "results": [
    {
      "url": "https://example.com/aperag-2025-roadmap",
      "status": "success",
      "title": "ApeRAG 2025年技术路线图",
      "content": "# ApeRAG 2025年技术路线图\n\nApeRAG在2025年将...",
      "extracted_at": "2025-01-07T10:01:00Z",
      "word_count": 1250,
      "token_count": 3200
    },
    {
      "url": "https://example.com/another-page",
      "status": "error",
      "error": "页面无法访问",
      "error_code": "TIMEOUT"
    }
  ],
  "total_urls": 2,
  "successful": 1,
  "failed": 1,
  "processing_time": 5.2
}
```

## 4. 前后端交互流程

### 4.1 对话创建流程

```mermaid
sequenceDiagram
    participant F as Frontend
    participant A as Agent API
    participant S as Agent Service
    participant D as Database

    F->>A: POST /api/v1/agent/chats
    A->>S: agent_service.create_chat()
    S->>D: 创建对话记录
    D-->>S: 返回chat_id
    S-->>A: 返回chat对象
    A-->>F: 返回JSON响应
```

### 4.2 消息发送流程

```mermaid
sequenceDiagram
    participant F as Frontend
    participant A as Agent API
    participant S as Agent Service
    participant MCP as MCP Service
    participant WS as Web Service
    participant LLM as LLM Service

    F->>A: POST /api/v1/agent/chats/{chat_id}/messages
    A->>S: agent_service.send_message()

    Note over S: Agent后端智能选择collections
    S->>MCP: 调用MCP搜索接口
    MCP-->>S: 返回搜索结果

    alt 如果启用web搜索
        S->>WS: 调用Web搜索/读取接口
        WS-->>S: 返回web结果
    end

    S->>LLM: 调用LLM生成回答
    LLM-->>S: 返回流式响应
    S-->>A: 返回流式响应
    A-->>F: SSE流式响应
```

### 4.3 Web服务调用流程

```mermaid
sequenceDiagram
    participant Client as 客户端
    participant HTTP as HTTP API
    participant MCP as MCP Tools
    participant WS as Web Service
    participant JINA as JINA API (初期)

    alt HTTP调用
        Client->>HTTP: POST /api/v1/web/search
        HTTP->>WS: 执行搜索
        WS->>JINA: 调用JINA API
        JINA-->>WS: 返回结果
        WS-->>HTTP: 格式化响应
        HTTP-->>Client: 返回标准格式
    else MCP调用
        Client->>MCP: web_search(query, ...)
        MCP->>WS: 执行搜索
        WS->>JINA: 调用JINA API
        JINA-->>WS: 返回结果
        WS-->>MCP: 格式化响应
        MCP-->>Client: 返回MCP格式
    end
```

## 5. 数据模型设计

### 5.1 Agent对话模型

```python
class AgentChat:
    id: str
    title: str
    created: datetime
    updated: datetime

class AgentMessage:
    id: str
    chat_id: str
    type: Literal["user", "assistant"]
    content: str
    model_used: str
    web_search_enabled: bool
    sources: List[SearchSource]
    web_search_results: List[WebSearchResult]
    created: datetime
```

### 5.2 Web服务模型

```python
class WebSearchRequest:
    query: str
    max_results: Optional[int] = 5
    search_engine: Optional[str] = "google"
    include_content: Optional[bool] = False
    timeout: Optional[int] = 30
    format: Optional[str] = "markdown"
    # ... 其他JINA参数

class WebReadRequest:
    urls: Union[str, List[str]]
    format: Optional[str] = "markdown"
    timeout: Optional[int] = 30
    max_concurrent: Optional[int] = 3
    # ... 其他JINA参数

class WebResult:
    url: str
    status: Literal["success", "error"]
    title: Optional[str]
    content: Optional[str]
    format: str
    word_count: Optional[int]
    token_count: Optional[int]
    images: Optional[List[WebImage]]
    links: Optional[List[WebLink]]
    error: Optional[str]
    error_code: Optional[str]
```

## 6. 实现策略

### 6.1 分阶段实现

**第一阶段：JINA集成**

- 直接集成[JINA Reader API](https://jina.ai/reader)作为后端
- 提供标准化的HTTP和MCP接口
- 支持JINA的主要参数和功能

**第二阶段：自研实现**

- 实现自己的web搜索引擎集成
- 实现自己的网页内容提取
- 逐步替换JINA依赖

**第三阶段：优化增强**

- 添加缓存机制
- 实现智能内容摘要
- 支持更多搜索引擎

### 6.2 接口兼容性

无论使用JINA还是自研实现，都保持相同的接口格式，确保：

- HTTP API接口不变
- MCP工具接口不变
- 响应格式保持一致
- 参数含义保持兼容

## 7. 错误处理设计

### 7.1 标准错误响应格式

```json
{
  "error": "WEB_SEARCH_FAILED",
  "message": "网络搜索服务暂时不可用",
  "details": {
    "search_engine": "google",
    "retry_after": 30
  }
}
```

### 7.2 常见错误码

| 错误码              | HTTP状态码 | 描述         |
| ------------------- | ---------- | ------------ |
| CHAT_NOT_FOUND      | 404        | 对话不存在   |
| MODEL_NOT_AVAILABLE | 400        | 模型不可用   |
| WEB_SEARCH_FAILED   | 500        | Web搜索失败  |
| WEB_READ_FAILED     | 500        | 网页读取失败 |
| URL_NOT_ACCESSIBLE  | 400        | URL无法访问  |
| QUOTA_EXCEEDED      | 429        | 超过配额限制 |
| INVALID_URL_FORMAT  | 400        | URL格式错误  |
| TIMEOUT_ERROR       | 408        | 请求超时     |

## 8. 性能优化设计

### 8.1 缓存策略

- **Web搜索结果缓存**：相同查询的搜索结果缓存1小时
- **Web页面内容缓存**：页面内容缓存6小时，支持ETags
- **模型配置缓存**：缓存用户可用的模型列表

### 8.2 异步处理

- **流式响应**：使用异步生成器实现流式输出
- **并发Web读取**：支持批量并发读取多个页面
- **超时控制**：所有外部请求都有超时限制

### 8.3 限流和配额

- **用户级限流**：每个用户每分钟最多10次web搜索
- **IP级限流**：防止滥用
- **内容大小限制**：单个页面内容最大5MB

## 9. 安全性设计

### 9.1 认证授权

- **API认证**：支持Bearer Token和Cookie认证
- **权限控制**：用户只能访问自己的对话
- **API限流**：防止接口滥用

### 9.2 数据安全

- **URL验证**：验证URL格式和域名白名单
- **内容过滤**：过滤恶意内容和敏感信息
- **XSS防护**：对Web内容进行sanitization

## 10. 实现优先级

### 10.1 第一阶段（核心功能）

1. Agent对话管理接口
2. 基础消息发送（非流式）
3. MCP接口集成（collection搜索）
4. Web服务接口（HTTP + MCP）
5. JINA Reader API集成

### 10.2 第二阶段（高级功能）

1. 流式响应支持
2. 缓存优化
3. 错误处理完善
4. 自研Web服务开发

### 10.3 第三阶段（优化功能）

1. 高级错误处理
2. 性能监控
3. 审计日志集成
4. 智能内容摘要

## 11. 总结

这个设计方案基于现有ApeRAG架构，通过新增Agent专用接口和独立的Web服务来支持智能对话功能。主要特点：

1. **架构兼容**：复用现有的service层和数据模型
2. **MCP集成**：通过MCP接口进行collection搜索，Agent后端智能选择
3. **Web服务独立**：参考[JINA Reader API](https://jina.ai/reader)设计，提供HTTP和MCP双接口
4. **参数兼容**：完整支持JINA的参数体系，便于初期集成
5. **渐进替换**：先用JINA实现，后续自研替换，接口保持兼容
6. **扩展性强**：接口设计支持未来功能扩展
7. **性能优化**：考虑缓存、异步处理等性能优化

前端可以通过这些接口实现类似Cursor的对话体验，用户可以轻松切换模型，并获得智能的搜索和问答服务。Agent后端会智能调用MCP接口进行collection搜索，同时Web服务提供强大的网络信息获取能力。
