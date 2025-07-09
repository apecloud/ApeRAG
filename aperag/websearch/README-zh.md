# ApeRAG WebSearch 模块

## 概述

ApeRAG WebSearch模块提供统一的Web搜索和内容读取能力，支持多种搜索引擎和内容提取器。模块采用Provider模式设计，易于扩展和维护。

## 架构设计

```
aperag/websearch/
├── search/                     # 搜索功能
│   ├── base_search.py         # 搜索基类
│   ├── search_service.py      # 搜索服务
│   └── providers/             # 搜索提供商
│       ├── duckduckgo_search_provider.py
│       └── jina_search_provider.py
├── reader/                     # 内容读取功能
│   ├── base_reader.py         # 读取基类
│   ├── reader_service.py      # 读取服务
│   └── providers/             # 读取提供商
│       ├── trafilatura_read_provider.py
│       └── jina_read_provider.py
└── utils/                      # 工具模块
    ├── url_validator.py       # URL验证
    └── content_processor.py   # 内容处理
```

## Search Providers

### 1. DuckDuckGoProvider

基于DuckDuckGo搜索引擎的搜索provider，免费且无需API密钥。

#### 特点
- ✅ 免费使用，无需配置
- ✅ 支持多语言搜索
- ✅ 隐私友好，不追踪用户
- ✅ 结果质量稳定

#### 基础用法

```python
from aperag.websearch.search.search_service import SearchService

# 创建搜索服务（默认使用DuckDuckGo）
search_service = SearchService()

# 或显式指定DuckDuckGo
search_service = SearchService(provider_name="duckduckgo")

# 执行搜索
request = WebSearchRequest(
    query="ApeRAG RAG系统",
    max_results=5
)

response = await search_service.search(request)
for result in response.results:
    print(f"标题: {result.title}")
    print(f"URL: {result.url}")
    print(f"摘要: {result.snippet}")
```

#### 配置选项

```python
# DuckDuckGo无需特殊配置，支持的参数包括：
request = WebSearchRequest(
    query="搜索关键词",
    max_results=10,        # 最大结果数量
    locale="zh-CN",       # 搜索语言
    timeout=30            # 请求超时时间
)
```

### 2. JinaSearchProvider

基于JINA AI的LLM优化搜索provider，专为AI应用设计。

#### 特点
- 🚀 LLM优化的搜索结果
- 🔍 支持多搜索引擎（Google、Bing）
- 📊 提供引用信息和相关性评分
- 🌍 支持多语言和地区定制
- ⚡ 专为AI Agent设计

#### 基础用法

```python
from aperag.websearch.search.search_service import SearchService

# 创建JINA搜索服务
search_service = SearchService(
    provider_name="jina",
    provider_config={
        "api_key": "your_jina_api_key"
    }
)

# 执行搜索
request = WebSearchRequest(
    query="ApeRAG架构设计",
    max_results=5,
    search_engine="google",  # 或 "bing", "jina"
    locale="zh-CN"
)

response = await search_service.search(request)
for result in response.results:
    print(f"标题: {result.title}")
    print(f"URL: {result.url}")
    print(f"摘要: {result.snippet}")
    print(f"域名: {result.domain}")
```

#### 高级特性

```python
# JINA搜索使用标准的WebSearchRequest接口
# 高级特性（引用信息、图片、时间范围等）在provider内部自动处理
request = WebSearchRequest(
    query="机器学习最新发展",
    max_results=10,
    search_engine="google",      # 支持: "google", "bing", "jina"
    locale="zh-CN",             # 语言地区
    timeout=30                  # 超时时间
)

# JINA provider内部自动启用以下特性：
# - 引用信息提取 (include_citations=True)
# - LLM优化的结果格式
# - 相关性评分
# - 智能内容摘要
```

#### 支持的搜索引擎

```python
# 获取支持的搜索引擎列表
engines = search_service.get_supported_engines()
print(engines)  # ['jina', 'google', 'bing']
```

## Reader Providers

### 1. TrafilaturaProvider

基于Trafilatura库的内容提取器，快速高效的本地处理。

#### 特点
- ⚡ 高性能本地处理
- 🎯 准确的正文提取
- 📱 支持多种网页格式
- 🔧 可自定义提取规则
- 💰 完全免费

#### 基础用法

```python
from aperag.websearch.reader.reader_service import ReaderService

# 创建读取服务（默认使用Trafilatura）
reader_service = ReaderService()

# 或显式指定Trafilatura
reader_service = ReaderService(provider_name="trafilatura")

# 读取单个URL
request = WebReadRequest(
    urls="https://example.com/article"
)

response = await reader_service.read(request)
for result in response.results:
    if result.status == "success":
        print(f"标题: {result.title}")
        print(f"内容: {result.content}")
        print(f"字数: {result.word_count}")
```

#### 批量处理

```python
# 批量读取多个URL
request = WebReadRequest(
    urls=[
        "https://example.com/article1",
        "https://example.com/article2",
        "https://example.com/article3"
    ],
    max_concurrent=3,  # 最大并发数
    timeout=30
)

response = await reader_service.read(request)
print(f"成功: {response.successful}/{response.total_urls}")

for result in response.results:
    if result.status == "success":
        print(f"✅ {result.url}: {result.title}")
    else:
        print(f"❌ {result.url}: {result.error}")
```

### 2. JinaReaderProvider

基于JINA AI的LLM优化内容提取器，专为AI应用优化。

#### 特点
- 🤖 LLM优化的内容提取
- 📝 Markdown格式输出
- 🎯 智能CSS选择器支持
- 🔄 SPA页面支持
- 📊 详细的元数据信息

#### 基础用法

```python
from aperag.websearch.reader.reader_service import ReaderService

# 创建JINA读取服务
reader_service = ReaderService(
    provider_name="jina",
    provider_config={
        "api_key": "your_jina_api_key"
    }
)

# 读取网页内容
request = WebReadRequest(
    urls="https://example.com/article",
    timeout=30,                # 请求超时时间
    locale="zh-CN"             # 语言地区
)

response = await reader_service.read(request)
for result in response.results:
    print(f"标题: {result.title}")
    print(f"内容: {result.content}")  # Markdown格式
    print(f"Token数: {result.token_count}")
```

#### 高级特性

```python
# JINA读取服务使用标准的WebReadRequest接口
# 高级特性（CSS选择器、SPA支持、缓存控制等）在provider内部自动处理
request = WebReadRequest(
    urls="https://news.example.com/article",
    timeout=45,                # 适当增加超时用于复杂页面
    locale="zh-CN",           # 语言地区
    max_concurrent=2          # 控制并发数
)

response = await reader_service.read(request)
result = response.results[0]

if result.status == "success":
    print(f"标题: {result.title}")
    print(f"内容: {result.content}")  # 已优化的Markdown格式
    print(f"字数: {result.word_count}")
    print(f"Token数: {result.token_count}")

# JINA provider内部自动启用以下特性：
# - 智能内容提取 (target_selector自动识别)
# - 广告和无关内容过滤 (exclude_selector自动处理)
# - SPA页面支持 (wait_for_selector自动处理)
# - LLM优化的Markdown输出
# - 元数据和结构化信息提取
```

## 服务使用指南

### 统一的服务接口

SearchService和ReaderService都提供统一的接口，便于在不同provider间切换：

```python
# 搜索服务示例
from aperag.websearch.search.search_service import SearchService

# 方式1：使用默认provider
service = SearchService()

# 方式2：指定provider名称
service = SearchService(provider_name="jina")

# 方式3：指定provider和配置
service = SearchService(
    provider_name="jina",
    provider_config={"api_key": "your_key"}
)

# 获取当前provider信息
print(f"当前provider: {service.provider_name}")
```

### 错误处理

```python
# 错误类从具体provider导入
from aperag.websearch.search.providers.duckduckgo_search_provider import SearchProviderError
from aperag.websearch.reader.providers.trafilatura_read_provider import ReaderProviderError

try:
    response = await search_service.search(request)
except SearchProviderError as e:
    print(f"搜索失败: {e}")
except Exception as e:
    print(f"未知错误: {e}")

try:
    response = await reader_service.read(request)
    for result in response.results:
        if result.status == "error":
            print(f"读取失败 {result.url}: {result.error}")
except ReaderProviderError as e:
    print(f"读取服务失败: {e}")

# 对于JINA providers，可以导入对应的错误类
# from aperag.websearch.search.providers.jina_search_provider import SearchProviderError
# from aperag.websearch.reader.providers.jina_read_provider import ReaderProviderError
```

### 异步批处理

```python
import asyncio

async def batch_search_and_read():
    """批量搜索并读取内容的完整示例"""
    
    # 初始化服务
    search_service = SearchService(provider_name="jina", 
                                 provider_config={"api_key": "your_key"})
    reader_service = ReaderService(provider_name="jina",
                                 provider_config={"api_key": "your_key"})
    
    # 1. 执行搜索
    search_request = WebSearchRequest(
        query="ApeRAG RAG系统架构",
        max_results=5
    )
    
    search_response = await search_service.search(search_request)
    urls = [result.url for result in search_response.results]
    
    # 2. 批量读取内容
    read_request = WebReadRequest(
        urls=urls,
        max_concurrent=3
    )
    
    read_response = await reader_service.read(read_request)
    
    # 3. 整合结果
    for i, search_result in enumerate(search_response.results):
        read_result = read_response.results[i]
        
        print(f"\n=== {search_result.title} ===")
        print(f"URL: {search_result.url}")
        print(f"搜索摘要: {search_result.snippet}")
        
        if read_result.status == "success":
            print(f"完整内容: {read_result.content[:200]}...")
            print(f"字数: {read_result.word_count}")
        else:
            print(f"内容读取失败: {read_result.error}")

# 运行示例
asyncio.run(batch_search_and_read())
```

## 配置说明

### 环境变量（可选）

虽然支持通过环境变量配置，但推荐直接在代码中传递配置：

```bash
# .env 文件（可选）
JINA_API_KEY=your_jina_api_key_here
```

### 推荐配置方式

```python
# 推荐：直接传递配置参数
config = {
    "api_key": "your_jina_api_key",
    "timeout": 30,
    "max_retries": 3
}

service = SearchService(provider_name="jina", provider_config=config)
```

## 最佳实践

### 1. Provider选择建议

**搜索Provider选择:**
- **DuckDuckGo**: 适用于简单搜索需求，免费稳定
- **JINA**: 适用于AI应用，需要高质量结果和引用信息

**读取Provider选择:**
- **Trafilatura**: 适用于大量本地处理，高性能需求
- **JINA**: 适用于需要结构化输出和AI优化的场景

### 2. 性能优化

```python
# 并发控制，避免过载
request = WebReadRequest(
    urls=url_list,
    max_concurrent=3,  # 控制并发数
    timeout=30         # 设置合理超时
)

# 批量处理，提高效率
batch_size = 10
for i in range(0, len(urls), batch_size):
    batch_urls = urls[i:i+batch_size]
    # 处理批次
```

### 3. 错误处理策略

```python
async def robust_web_operation(service, request, max_retries=3):
    """带重试机制的Web操作"""
    for attempt in range(max_retries):
        try:
            return await service.search(request)  # 或 service.read(request)
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            await asyncio.sleep(2 ** attempt)  # 指数退避
```

### 4. 缓存策略

```python
from functools import lru_cache
import hashlib

class CachedWebService:
    def __init__(self):
        self.search_service = SearchService()
        self.reader_service = ReaderService()
    
    @lru_cache(maxsize=100)
    async def cached_search(self, query: str, max_results: int = 5):
        """带缓存的搜索"""
        request = WebSearchRequest(query=query, max_results=max_results)
        return await self.search_service.search(request)
```

## 依赖说明

```python
# 核心依赖
pip install duckduckgo-search  # DuckDuckGo搜索
pip install trafilatura       # 内容提取
pip install aiohttp           # HTTP客户端（JINA providers）

# 可选依赖（根据使用的provider安装）
pip install beautifulsoup4    # HTML解析增强
pip install lxml             # XML/HTML解析器
```

## 故障排除

### 常见问题

1. **JINA API密钥问题**
   ```python
   # 确保API密钥正确传递
   config = {"api_key": "jina_xxxxxxxxxxxx"}
   service = SearchService(provider_name="jina", provider_config=config)
   ```

2. **网络超时**
   ```python
   # 增加超时时间
   request = WebSearchRequest(query="...", timeout=60)
   ```

3. **并发限制**
   ```python
   # 减少并发数
   request = WebReadRequest(urls=urls, max_concurrent=2)
   ```

4. **内容提取失败**
   ```python
   # 增加超时时间，让provider有更多时间处理复杂页面
   request = WebReadRequest(
       urls=url,
       timeout=60,           # 增加超时时间
       max_concurrent=1      # 降低并发数
   )
   ```

---

**更多信息请参考：**
- [Agent后端设计文档](../../docs/design/agent-backend-zh.md)
- [JINA API文档](https://jina.ai/reader)
- [DuckDuckGo Search文档](https://pypi.org/project/duckduckgo-search/)
- [Trafilatura文档](https://trafilatura.readthedocs.io/) 