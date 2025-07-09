# ApeRAG Web搜索与内容读取服务设计文档

## 1. 设计概述

### 1.1 设计思想

基于现有LLM服务架构（EmbeddingService、RerankService等），采用**Provider抽象模式**设计Web搜索和内容读取服务。核心思想：

- **统一接口**：上层Service统一调用，底层可切换Provider
- **插件化**：新增搜索引擎或内容提取器只需实现Provider接口
- **双路供给**：同时提供HTTP API和MCP工具
- **渐进替换**：初期使用Crawl4AI/JINA，后续可无缝切换自研实现

### 1.2 技术架构决策

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
        │WebSearchSvc │    │ WebReadSvc  │
        └──────┬──────┘    └──────┬──────┘
               │                  │
     ┌─────────▼─────────┐ ┌─────▼─────────┐
     │SearchProviderBase│ │ReadProviderBase│
     └─────────┬─────────┘ └─────┬─────────┘
               │                 │
    ┌──────────▼──────────┐ ┌────▼──────────┐
    │  DuckDuckGoProvider │ │ Crawl4AIProvider│
    │  BingProvider       │ │ JINAProvider    │
    │  GoogleProvider     │ │ TrafilaturaProvider│
    └─────────────────────┘ └─────────────────┘
```

## 2. 目录结构设计

### 2.1 新增目录结构

```
aperag/
├── websearch/                          # 新增：Web搜索模块
│   ├── __init__.py
│   ├── search/                         # 搜索相关
│   │   ├── __init__.py
│   │   ├── base_search.py              # 搜索Provider抽象基类
│   │   ├── search_service.py           # 搜索服务类
│   │   └── providers/                  # 搜索Provider实现
│   │       ├── __init__.py
│   │       ├── duckduckgo.py          # DuckDuckGo搜索实现
│   │       ├── bing.py                # Bing搜索实现（可选）
│   │       └── google.py              # Google搜索实现（可选）
│   ├── reader/                         # 内容读取相关
│   │   ├── __init__.py
│   │   ├── base_reader.py              # 内容读取Provider抽象基类
│   │   ├── reader_service.py           # 内容读取服务类
│   │   └── providers/                  # 内容读取Provider实现
│   │       ├── __init__.py
│   │       ├── crawl4ai.py            # Crawl4AI实现（主力）
│   │       ├── jina.py                # JINA实现（备选）
│   │       └── trafilatura.py         # Trafilatura实现（轻量级）
│   └── utils/                          # 工具类
│       ├── __init__.py
│       ├── url_validator.py           # URL验证
│       └── content_processor.py       # 内容处理
```

### 2.2 现有目录改动

```
aperag/
├── views/                              # 改动：新增web相关视图
│   ├── web.py                         # 新增：Web服务HTTP接口
│   └── __init__.py                    # 修改：导入web视图
├── mcp/                               # 改动：新增web相关MCP工具
│   ├── server.py                      # 修改：注册web_search和web_read工具
│   └── __init__.py                    # 可能需要修改
└── schema/
    └── view_models.py                 # 改动：新增Web相关数据模型
```

## 3. API接口设计

### 3.1 HTTP API路径规划

```
/api/v1/web/
├── search                             # POST - Web搜索
└── read                               # POST - Web内容读取
```

**设计决策**：
- 独立于`/api/v1/agent/`路径，体现Web服务的通用性
- 只有两个端点，简洁清晰
- 使用POST方法，支持复杂参数传递

### 3.2 MCP工具接口

```
MCP Tools:
├── web_search(query, max_results, search_engine, ...)
└── web_read(urls, timeout, css_selector, ...)
```

## 4. 核心组件设计

### 4.1 SearchService设计思想

**参考**：`aperag/llm/embed/embedding_service.py`

**核心特性**：
- Provider抽象：支持多种搜索引擎切换
- 统一接口：`async def search(query, **kwargs) -> SearchResult`
- 配置驱动：通过环境变量选择Provider
- 错误处理：统一的异常处理和降级策略

### 4.2 ReaderService设计思想

**参考**：`aperag/llm/rerank/rerank_service.py`

**核心特性**：
- Provider抽象：支持多种内容提取库切换
- 批量处理：支持并发读取多个URL
- 格式统一：输出标准化的Markdown格式
- 智能降级：主Provider失败时自动切换备用Provider

### 4.3 Provider接口设计

**搜索Provider接口**：
```python
class BaseSearchProvider(ABC):
    @abstractmethod
    async def search(self, query: str, **kwargs) -> List[SearchResult]
    
    @abstractmethod
    def get_supported_engines(self) -> List[str]
```

**读取Provider接口**：
```python
class BaseReaderProvider(ABC):
    @abstractmethod
    async def read(self, url: str, **kwargs) -> ReaderResult
    
    @abstractmethod
    async def read_batch(self, urls: List[str], **kwargs) -> List[ReaderResult]
```

## 5. 改动范围明确

### 5.1 新增文件（约15个文件）

```
aperag/websearch/ - 完全新增目录
├── 搜索模块：5个文件
├── 读取模块：5个文件  
├── 工具模块：3个文件
└── 其他：2个文件
```

### 5.2 修改现有文件（约4个文件）

```
1. aperag/views/__init__.py           - 导入web视图
2. aperag/views/web.py                - 新增Web HTTP接口
3. aperag/mcp/server.py               - 注册MCP工具
4. aperag/schema/view_models.py       - 新增数据模型
```

### 5.3 配置文件改动

```
1. envs/env.template                  - 新增Web服务配置
2. requirements.txt                   - 新增依赖包
```

## 6. 关键技术决策

### 6.1 Provider选择策略

**第一阶段Provider组合**：
- **搜索**：DuckDuckGo（免费、无限制、隐私友好）
- **读取**：Crawl4AI（专为LLM设计、功能强大、维护活跃）

**第二阶段扩展**：
- 搜索：新增Bing、Google Provider
- 读取：新增JINA、Trafilatura Provider

### 6.2 错误处理策略

**多层降级机制**：
1. 主Provider失败 → 自动切换备用Provider
2. 所有Provider失败 → 返回标准错误响应
3. 部分URL失败 → 返回成功和失败混合结果

### 6.3 配置管理策略

**环境变量配置**：
```bash
# 搜索配置
WEB_SEARCH_PROVIDER=duckduckgo
WEB_SEARCH_FALLBACK_PROVIDER=bing

# 读取配置  
WEB_READER_PROVIDER=crawl4ai
WEB_READER_FALLBACK_PROVIDER=jina
WEB_READER_TIMEOUT=30
WEB_READER_MAX_CONCURRENT=3
```

### 6.4 性能优化策略

**核心优化点**：
- **异步处理**：所有网络请求使用asyncio
- **并发控制**：读取多个URL时限制并发数
- **超时控制**：所有外部请求设置超时
- **连接复用**：Provider内部使用连接池

## 7. 实现优先级

### 7.1 第一阶段（核心MVP）

**目标**：提供基础的搜索和读取功能

1. **基础架构搭建**
   - Provider抽象接口定义
   - Service层实现
   - 基础数据模型

2. **DuckDuckGo搜索Provider**
   - 免费、无需API Key
   - 基础搜索功能

3. **Crawl4AI读取Provider**
   - 安装和基础配置
   - 单URL读取功能

4. **HTTP API接口**
   - `/api/v1/web/search`
   - `/api/v1/web/read`

5. **MCP工具接口**
   - `web_search`工具
   - `web_read`工具

### 7.2 第二阶段（功能完善）

1. **批量读取功能**
2. **错误处理完善**
3. **参数验证和安全检查**
4. **基础性能优化**

### 7.3 第三阶段（扩展增强）

1. **多Provider支持**
2. **智能降级机制**
3. **缓存和性能优化**
4. **监控和日志集成**

## 8. 依赖管理

### 8.1 新增核心依赖

```bash
# 搜索相关
duckduckgo-search>=6.0.0           # DuckDuckGo搜索

# 内容读取相关  
crawl4ai>=0.3.0                    # 主力内容读取库
trafilatura>=1.12.0                # 轻量级备选
requests>=2.31.0                   # HTTP请求
aiohttp>=3.9.0                     # 异步HTTP请求

# 内容处理
beautifulsoup4>=4.12.0             # HTML解析
lxml>=5.0.0                        # XML/HTML解析器
```

### 8.2 可选依赖（按需安装）

```bash
# JINA Reader集成（可选）
jina>=3.0.0

# 高级搜索API（可选）
bing-search-api>=1.0.0
google-search-results>=2.4.0
```

## 9. 测试策略

### 9.1 单元测试范围

```
tests/unit_test/websearch/
├── test_search_service.py          # 搜索服务测试
├── test_reader_service.py          # 读取服务测试
├── test_duckduckgo_provider.py     # DuckDuckGo Provider测试
├── test_crawl4ai_provider.py       # Crawl4AI Provider测试
└── test_web_views.py               # HTTP接口测试
```

### 9.2 集成测试重点

1. **真实网络请求测试**：使用实际URL验证功能
2. **MCP工具集成测试**：验证MCP调用流程
3. **错误场景测试**：网络超时、URL无效等
4. **并发性能测试**：批量读取压力测试

## 10. 部署和配置

### 10.1 环境配置

**开发环境**：
```bash
# 安装开发依赖
make install  # 已有命令，会安装新依赖

# 配置环境变量
cp envs/env.template .env
# 编辑.env添加Web服务配置
```

**生产环境**：
- 容器化部署：更新Dockerfile包含新依赖
- 环境变量：通过K8s ConfigMap管理配置
- 监控：集成现有监控体系

### 10.2 配置管理

**配置优先级**：
1. 环境变量（最高优先级）
2. 配置文件
3. 代码默认值（最低优先级）

## 11. 总结

### 11.1 设计优势

1. **架构统一**：完全遵循现有LLM服务设计模式
2. **扩展性强**：Provider模式支持无限扩展
3. **实现渐进**：可从简单实现开始，逐步完善
4. **接口双重**：HTTP + MCP双接口满足不同需求
5. **技术领先**：Crawl4AI是当前最适合LLM的方案

### 11.2 技术债务控制

- **代码复用**：最大化复用现有架构和工具
- **依赖管理**：仅引入必要的核心依赖
- **测试覆盖**：从设计阶段就考虑测试策略
- **文档同步**：接口文档与实现同步更新

### 11.3 风险控制

- **Provider切换**：架构设计支持无缝切换
- **服务降级**：多层错误处理确保服务可用性
- **性能监控**：关键指标监控防止性能问题
- **安全防护**：URL验证和内容过滤防止安全问题

这个设计方案在保持架构一致性的同时，为ApeRAG提供了强大的Web搜索和内容读取能力，为Agent功能奠定了坚实的技术基础。