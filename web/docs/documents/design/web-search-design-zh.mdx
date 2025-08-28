# ApeRAG Web搜索与内容读取服务设计文档

## 1. 设计概述

### 1.1 设计理念

ApeRAG Web搜索模块采用**Provider抽象模式**，参考现有LLM服务架构（EmbeddingService、RerankService等），提供统一的Web搜索和内容读取能力。

**核心特性**：
- **统一接口**：上层Service统一调用，底层可切换Provider
- **插件化架构**：新增搜索引擎或内容提取器只需实现Provider接口
- **资源安全管理**：完整的异步资源生命周期管理
- **双路供给**：同时提供HTTP API和MCP工具支持
- **生产就绪**：内置错误处理、超时控制、并发限制

### 1.2 已实现架构

```
┌─────────────────────────────────────────────────────────┐
│                   API Layer                             │
│  ┌─────────────────┐    ┌─────────────────┐            │
│  │  HTTP Views     │    │   MCP Tools     │            │
│  │ /api/v1/web/*   │    │   web_search    │            │
│  └─────────────────┘    │   web_read      │            │
└──────────────┬──────────└─────────────────┘────────────┘
               │                  │
        ┌──────▼──────┐    ┌──────▼──────┐
        │SearchService│    │ReaderService│
        │+ async with │    │+ async with │
        └──────┬──────┘    └──────┬──────┘
               │                  │
     ┌─────────▼─────────┐ ┌─────▼─────────┐
     │BaseSearchProvider│ │BaseReaderProvider│
     │+ close() support │ │+ close() support│
     └─────────┬─────────┘ └─────┬─────────┘
               │                 │
    ┌──────────▼──────────┐ ┌────▼──────────┐
    │  DuckDuckGoProvider │ │TrafilaturaProvider│
    │  JinaSearchProvider │ │JinaReaderProvider │
    └─────────────────────┘ └─────────────────┘
```

## 2. 实际目录结构

### 2.1 已实现的模块结构

```
aperag/websearch/                       # Web搜索模块根目录
├── __init__.py                         # 导出SearchService, ReaderService
├── search/                             # 搜索功能模块
│   ├── __init__.py                     # 导出SearchService, BaseSearchProvider
│   ├── base_search.py                  # 搜索Provider抽象基类
│   ├── search_service.py               # 搜索服务（支持上下文管理器）
│   └── providers/                      # 搜索Provider实现
│       ├── __init__.py                 # 导出所有Provider
│       ├── duckduckgo_search_provider.py   # DuckDuckGo实现（默认）
│       └── jina_search_provider.py     # JINA搜索实现
├── reader/                             # 内容读取功能模块
│   ├── __init__.py                     # 导出ReaderService, BaseReaderProvider
│   ├── base_reader.py                  # 读取Provider抽象基类
│   ├── reader_service.py               # 读取服务（支持上下文管理器）
│   └── providers/                      # 读取Provider实现
│       ├── __init__.py                 # 导出所有Provider
│       ├── trafilatura_read_provider.py    # Trafilatura实现（默认）
│       └── jina_read_provider.py       # JINA读取实现
├── utils/                              # 工具模块
│   ├── __init__.py                     # 导出工具类
│   ├── url_validator.py                # URL验证和规范化
│   └── content_processor.py            # 内容处理和清理
└── README-zh.md                        # 模块使用文档
```

### 2.2 集成的系统模块

```
aperag/views/web.py                     # HTTP API视图（已实现）
aperag/mcp/server.py                    # MCP工具注册（待集成）
aperag/schema/view_models.py            # 数据模型（已扩展）
```

## 3. API接口设计

### 3.1 HTTP API接口

**已实现的RESTful API**：

```yaml
# OpenAPI规范片段
/api/v1/web/search:
  post:
    summary: 执行Web搜索
    requestBody:
      required: true
      content:
        application/json:
          schema:
            $ref: '#/components/schemas/WebSearchRequest'
    responses:
      '200':
        description: 搜索成功
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/WebSearchResponse'

/api/v1/web/read:
  post:
    summary: 读取Web页面内容
    requestBody:
      required: true
      content:
        application/json:
          schema:
            $ref: '#/components/schemas/WebReadRequest'
    responses:
      '200':
        description: 读取成功
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/WebReadResponse'
```

### 3.2 请求/响应数据模型

**WebSearchRequest**:
```python
class WebSearchRequest(BaseModel):
    query: str                          # 搜索查询
    max_results: int = 5               # 最大结果数
    search_engine: str = "duckduckgo"  # 搜索引擎
    timeout: int = 30                  # 超时时间（秒）
    locale: str = "zh-CN"             # 语言地区
```

**WebSearchResponse**:
```python
class WebSearchResponse(BaseModel):
    query: str                         # 原始查询
    results: List[WebSearchResultItem] # 搜索结果列表
    search_engine: str                 # 使用的搜索引擎
    total_results: int                 # 结果总数
    search_time: float                 # 搜索耗时（秒）
```

**WebReadRequest**:
```python
class WebReadRequest(BaseModel):
    urls: Union[str, List[str]]        # 单个或多个URL
    timeout: int = 30                  # 超时时间（秒）
    locale: str = "zh-CN"             # 语言地区
    max_concurrent: int = 3            # 最大并发数（批量读取）
```

> **注意**：统一的接口设计确保了所有provider都使用相同的参数。JINA等高级provider的特有功能（如CSS选择器、SPA支持等）在provider内部使用合理的默认值自动处理。

**WebReadResponse**:
```python
class WebReadResponse(BaseModel):
    results: List[WebReadResultItem]   # 读取结果列表
    total_urls: int                    # 总URL数量
    successful: int                    # 成功数量
    failed: int                        # 失败数量
    processing_time: float             # 处理耗时（秒）
```

## 4. 核心组件实现

### 4.1 SearchService实现特性

**参考架构**：`aperag/llm/embed/embedding_service.py`

**实现特性**：
```python
class SearchService:
    # 支持的Provider切换
    def __init__(self, provider_name: str = None, provider_config: Dict = None)
    
    # 异步搜索接口
    async def search(self, request: WebSearchRequest) -> WebSearchResponse
    
    # 简化搜索接口
    async def search_simple(self, query: str, **kwargs) -> List[WebSearchResultItem]
    
    # 资源管理
    async def close(self)
    async def __aenter__(self) / __aexit__(self)  # 上下文管理器支持
    
    # 工厂方法
    @classmethod
    def create_default(cls) -> "SearchService"
    @classmethod
    def create_with_provider(cls, provider_name: str, **config) -> "SearchService"
```

**资源安全使用**：
```python
# 推荐使用方式：上下文管理器
async with SearchService() as search_service:
    response = await search_service.search(request)
    # 资源自动清理

# 或手动管理
search_service = SearchService()
try:
    response = await search_service.search(request)
finally:
    await search_service.close()
```

### 4.2 ReaderService实现特性

**参考架构**：`aperag/llm/rerank/rerank_service.py`

**实现特性**：
```python
class ReaderService:
    # 支持单个和批量读取
    async def read(self, request: WebReadRequest) -> WebReadResponse
    async def read_simple(self, url: str, **kwargs) -> WebReadResultItem
    async def read_batch_simple(self, urls: List[str], **kwargs) -> List[WebReadResultItem]
    
    # 完整的资源管理
    async def close(self)
    async def cleanup(self)  # 别名
    async def __aenter__(self) / __aexit__(self)
```

**并发控制实现**：
```python
# 内部使用asyncio.Semaphore控制并发
async def read_batch(self, urls: List[str], max_concurrent: int = 3):
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def read_single(url: str):
        async with semaphore:
            return await self.read(url)
    
    results = await asyncio.gather(*[read_single(url) for url in urls])
```

### 4.3 Provider接口规范

**BaseSearchProvider接口**：
```python
class BaseSearchProvider(ABC):
    @abstractmethod
    async def search(self, query: str, **kwargs) -> List[WebSearchResultItem]
    
    @abstractmethod
    def get_supported_engines(self) -> List[str]
    
    def validate_search_engine(self, search_engine: str) -> bool
    
    async def close(self):  # 资源清理
        pass
```

**BaseReaderProvider接口**：
```python
class BaseReaderProvider(ABC):
    @abstractmethod
    async def read(self, url: str, **kwargs) -> WebReadResultItem
    
    @abstractmethod
    async def read_batch(self, urls: List[str], **kwargs) -> List[WebReadResultItem]
    
    def validate_url(self, url: str) -> bool
    
    async def close(self):  # 资源清理
        pass
```

## 5. Provider实现详情

### 5.1 DuckDuckGoProvider（默认搜索）

**实现特性**：
- ✅ 免费使用，无需API密钥
- ✅ 基于`duckduckgo-search`库
- ✅ 支持地区和语言定制
- ✅ 异步包装（使用`loop.run_in_executor`）

**配置示例**：
```python
# 无需配置，开箱即用
service = SearchService()  # 默认使用DuckDuckGo

# 显式指定
service = SearchService(provider_name="duckduckgo")
```

### 5.2 JinaSearchProvider

**实现特性**：
- 🚀 专为LLM优化的搜索结果
- 🔧 支持多搜索引擎后端（Google、Bing）
- 📊 提供引用信息和结构化输出
- 🌐 基于JINA s.jina.ai API

**配置示例**：
```python
service = SearchService(
    provider_name="jina",
    provider_config={"api_key": "jina_xxxxxxxxxxxxxxxx"}
)
```

### 5.3 TrafilaturaProvider（默认读取）

**实现特性**：
- ⚡ 高性能本地处理，无需外部API
- 🎯 基于`trafilatura`库的准确正文提取
- 📱 支持多种网页格式
- 💰 完全免费

**配置示例**：
```python
# 无需配置，开箱即用
service = ReaderService()  # 默认使用Trafilatura
```

### 5.4 JinaReaderProvider

**实现特性**：
- 🤖 LLM优化的内容提取
- 📝 Markdown格式输出
- 🎯 智能内容识别
- 🌐 基于JINA r.jina.ai API

**配置示例**：
```python
service = ReaderService(
    provider_name="jina",
    provider_config={"api_key": "jina_xxxxxxxxxxxxxxxx"}
)
```

## 6. 使用示例

### 6.1 基础搜索示例

```python
from aperag.websearch import SearchService
from aperag.schema.view_models import WebSearchRequest

async def basic_search():
    async with SearchService() as search_service:
        request = WebSearchRequest(
            query="ApeRAG RAG系统架构",
            max_results=5,
            search_engine="duckduckgo"
        )
        
        response = await search_service.search(request)
        
        for result in response.results:
            print(f"标题: {result.title}")
            print(f"URL: {result.url}")
            print(f"摘要: {result.snippet}")
            print(f"域名: {result.domain}")
            print("---")
```

### 6.2 内容读取示例

```python
from aperag.websearch import ReaderService
from aperag.schema.view_models import WebReadRequest

async def basic_read():
    async with ReaderService() as reader_service:
        # 单个URL读取
        request = WebReadRequest(urls="https://example.com/article")
        response = await reader_service.read(request)
        
        result = response.results[0]
        if result.status == "success":
            print(f"标题: {result.title}")
            print(f"内容长度: {result.word_count} 词")
            print(f"内容预览: {result.content[:200]}...")
```

### 6.3 批量处理示例

```python
async def batch_processing():
    """搜索并批量读取内容的完整示例"""
    
    # 1. 执行搜索
    async with SearchService(provider_name="jina", 
                           provider_config={"api_key": "your_key"}) as search_service:
        search_request = WebSearchRequest(
            query="人工智能最新发展",
            max_results=5
        )
        search_response = await search_service.search(search_request)
        urls = [result.url for result in search_response.results]
    
    # 2. 批量读取内容
    async with ReaderService() as reader_service:
        read_request = WebReadRequest(
            urls=urls,
            max_concurrent=3,
            timeout=30
        )
        read_response = await reader_service.read(read_request)
    
    # 3. 整合结果
    for i, search_result in enumerate(search_response.results):
        read_result = read_response.results[i]
        
        print(f"\n=== {search_result.title} ===")
        print(f"URL: {search_result.url}")
        print(f"搜索摘要: {search_result.snippet}")
        
        if read_result.status == "success":
            print(f"完整内容: {read_result.content[:300]}...")
            print(f"字数: {read_result.word_count}")
        else:
            print(f"内容读取失败: {read_result.error}")
```

### 6.4 HTTP API使用示例

```bash
# 搜索API调用
curl -X POST "http://localhost:8000/api/v1/web/search" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your_token" \
  -d '{
    "query": "ApeRAG RAG系统",
    "max_results": 5,
    "search_engine": "duckduckgo"
  }'

# 读取API调用
curl -X POST "http://localhost:8000/api/v1/web/read" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your_token" \
  -d '{
    "urls": ["https://example.com/article1", "https://example.com/article2"],
    "max_concurrent": 2,
    "timeout": 30
  }'
```

## 7. 资源管理与安全

### 7.1 资源泄漏解决方案

**问题**：在异步环境中，未正确关闭的资源会导致`ResourceWarning: Unclosed <MemoryObjectReceiveStream>`。

**解决方案**：
1. **所有Provider实现close()方法**
2. **Service层支持上下文管理器**
3. **HTTP视图层使用上下文管理器**

**实现细节**：
```python
# 所有Provider基类都实现close()
async def close(self):
    # 子类可重写进行资源清理
    pass

# Service层上下文管理器
async def __aenter__(self):
    return self

async def __aexit__(self, exc_type, exc_val, exc_tb):
    await self.close()

# HTTP视图层安全使用
async def web_search(request: WebSearchRequest):
    async with SearchService() as search_service:
        response = await search_service.search(request)
        return response
    # 资源自动清理，避免泄漏
```

### 7.2 错误处理机制

**多层错误处理**：
```python
# Provider层：具体错误
class SearchProviderError(Exception):
    pass

class ReaderProviderError(Exception):
    pass

# Service层：统一包装
try:
    results = await self.provider.search(...)
except SearchProviderError:
    raise  # 重新抛出Provider错误
except Exception as e:
    raise SearchProviderError(f"Search service error: {str(e)}")

# HTTP视图层：用户友好的错误响应
except SearchProviderError as e:
    raise HTTPException(status_code=500, detail=f"搜索失败: {str(e)}")
```

## 8. 性能优化

### 8.1 并发控制

**批量处理优化**：
```python
# 使用Semaphore控制并发数
semaphore = asyncio.Semaphore(max_concurrent)

async def read_single(url: str):
    async with semaphore:
        return await self.provider.read(url)

# 并发执行，自动限制并发数
results = await asyncio.gather(*[read_single(url) for url in urls])
```

### 8.2 超时控制

**多层超时保护**：
```python
# Provider层：网络请求超时
async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
    async with session.post(url, json=payload) as response:
        # 自动超时保护

# Service层：整体操作超时
response = await asyncio.wait_for(
    self.provider.search(query),
    timeout=request.timeout
)
```

## 9. 依赖管理

### 9.1 已集成依赖

```python
# 搜索相关
duckduckgo-search>=6.0.0      # DuckDuckGo搜索
aiohttp>=3.9.0                # 异步HTTP客户端

# 内容读取相关
trafilatura>=1.12.0           # 内容提取
markdownify>=0.11.0           # HTML转Markdown

# 内容处理
beautifulsoup4>=4.12.0        # HTML解析（可选）
lxml>=5.0.0                   # 解析器（可选）
```

### 9.2 安装方法

```bash
# 通过项目Makefile安装
make install

# 或直接pip安装
pip install duckduckgo-search trafilatura markdownify aiohttp
```

## 10. 配置管理

### 10.1 推荐配置方式

**代码配置（推荐）**：
```python
# 直接传递配置，最灵活
search_config = {
    "api_key": "your_jina_api_key",
    "timeout": 30,
    "max_retries": 3
}

service = SearchService(provider_name="jina", provider_config=search_config)
```

**环境变量配置（可选）**：
```bash
# .env文件
JINA_API_KEY=jina_xxxxxxxxxxxxxxxx
WEB_SEARCH_TIMEOUT=30
WEB_READ_MAX_CONCURRENT=3
```

### 10.2 Provider选择策略

**智能降级**：
```python
class SmartWebService:
    async def search_with_fallback(self, query: str):
        # 主Provider：JINA（如果有API Key）
        if self.has_jina_key():
            try:
                async with SearchService("jina", {"api_key": self.jina_key}) as service:
                    return await service.search_simple(query)
            except Exception as e:
                logger.warning(f"JINA搜索失败，降级到DuckDuckGo: {e}")
        
        # 降级Provider：DuckDuckGo
        async with SearchService("duckduckgo") as service:
            return await service.search_simple(query)
```

## 11. 测试与监控

### 11.1 单元测试覆盖

```
tests/unit_test/websearch/
├── test_search_service.py           # SearchService测试
├── test_reader_service.py           # ReaderService测试
├── test_duckduckgo_provider.py      # DuckDuckGo Provider测试
├── test_jina_providers.py           # JINA Providers测试
└── test_web_views.py                # HTTP接口测试
```

### 11.2 性能监控

**关键指标**：
- 搜索响应时间
- 内容读取成功率
- 并发处理能力
- 资源使用情况

**监控实现**：
```python
# 在Service层添加指标收集
import time
import logging

logger = logging.getLogger(__name__)

async def search(self, request: WebSearchRequest):
    start_time = time.time()
    try:
        result = await self.provider.search(...)
        search_time = time.time() - start_time
        
        # 记录成功指标
        logger.info(f"搜索成功: query={request.query}, time={search_time:.2f}s, results={len(result)}")
        return result
        
    except Exception as e:
        # 记录失败指标
        logger.error(f"搜索失败: query={request.query}, error={str(e)}")
        raise
```

## 12. 集成与部署

### 12.1 MCP工具集成（待完成）

**计划集成的MCP工具**：
```python
# aperag/mcp/server.py 扩展
@server.tool()
async def web_search(query: str, max_results: int = 5) -> dict:
    """执行Web搜索"""
    async with SearchService() as service:
        request = WebSearchRequest(query=query, max_results=max_results)
        response = await service.search(request)
        return response.dict()

@server.tool()  
async def web_read(urls: List[str], max_concurrent: int = 3) -> dict:
    """读取Web页面内容"""
    async with ReaderService() as service:
        request = WebReadRequest(urls=urls, max_concurrent=max_concurrent)
        response = await service.read(request)
        return response.dict()
```

### 12.2 生产环境配置

**Docker配置**：
```dockerfile
# Dockerfile中添加依赖
RUN pip install duckduckgo-search trafilatura markdownify

# 环境变量
ENV WEB_SEARCH_PROVIDER=duckduckgo
ENV WEB_READ_PROVIDER=trafilatura
ENV WEB_REQUEST_TIMEOUT=30
```

**Kubernetes配置**：
```yaml
# 通过ConfigMap管理配置
apiVersion: v1
kind: ConfigMap
metadata:
  name: websearch-config
data:
  JINA_API_KEY: "your_jina_api_key"
  WEB_SEARCH_TIMEOUT: "30"
  WEB_READ_MAX_CONCURRENT: "3"
```

## 13. 总结

### 13.1 实现成果

✅ **已完成功能**：
- 完整的Provider抽象架构
- DuckDuckGo和JINA搜索Provider
- Trafilatura和JINA读取Provider
- HTTP API接口实现
- 资源安全管理机制
- 并发控制和错误处理
- 完整的单元测试

✅ **技术优势**：
- **架构统一**：完全遵循ApeRAG现有LLM服务设计模式
- **资源安全**：解决了异步资源泄漏问题，生产环境可靠
- **性能优化**：内置并发控制、超时保护、智能降级
- **易于扩展**：新增Provider只需实现标准接口
- **开箱即用**：DuckDuckGo和Trafilatura无需配置即可使用

### 13.2 待完成集成

⏳ **计划中功能**：
- MCP工具注册和集成
- 缓存机制实现
- 监控指标完善
- 更多Provider支持（Bing、Google等）

### 13.3 最佳实践总结

1. **始终使用上下文管理器**：避免资源泄漏
2. **合理设置并发数**：防止外部服务过载
3. **实现智能降级**：提高服务可用性
4. **监控关键指标**：确保服务质量
5. **遵循统一接口**：便于Provider切换

这个Web搜索模块为ApeRAG Agent提供了强大的Web信息获取能力，架构设计完全符合系统整体设计理念，实现了生产级的稳定性和可扩展性。