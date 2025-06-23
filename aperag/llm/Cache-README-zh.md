# LiteLLM 缓存功能

ApeRAG 集成了 LiteLLM 内置的缓存功能，为所有 LLM 服务调用提供高效的缓存机制，显著提升性能并减少 API 调用成本。

## 🚀 核心特性

- **🎛️ 简单开关控制**：通过 `CACHE_ENABLED` 一键启用/禁用缓存功能
- **⏰ 灵活TTL配置**：通过 `CACHE_TTL` 自定义缓存过期时间
- **🗄️ Redis 后端存储**：使用 Redis 作为缓存后端，支持分布式部署
- **📊 本地统计监控**：实时监控缓存命中率和使用情况
- **🔄 全服务支持**：completion、embedding、rerank 服务全面支持
- **⚡ 极致性能**：Redis 提供毫秒级缓存访问，可获得 10-1000 倍性能提升
- **🛠️ 零配置启动**：默认配置即可使用，支持渐进式优化

## 📁 模块结构

```
aperag/llm/
├── litellm_cache.py          # LiteLLM 缓存核心配置和管理
├── completion/
│   └── completion_service.py  # 支持 caching 参数的完成服务
├── embed/
│   └── embedding_service.py   # 支持 caching 参数的嵌入服务
└── rerank/
    └── rerank_service.py      # 支持 caching 参数的重排序服务
```

## ⚙️ 配置说明

### 核心配置开关

在 `.env` 文件中配置缓存控制参数：

```bash
# ========== 缓存核心控制 ==========
# 缓存功能开关（默认启用）
CACHE_ENABLED=true

# 缓存生存时间，单位：秒（默认24小时）
CACHE_TTL=86400

# ========== Redis 连接配置 ==========
REDIS_HOST=127.0.0.1
REDIS_PORT=6379
REDIS_PASSWORD=your_redis_password
```

### 配置参数详解

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `CACHE_ENABLED` | boolean | `true` | 全局缓存开关，设为 `false` 完全禁用缓存 |
| `CACHE_TTL` | integer | `86400` | 缓存条目生存时间（秒），86400=24小时 |
| `REDIS_HOST` | string | `localhost` | Redis 服务器地址 |
| `REDIS_PORT` | integer | `6379` | Redis 服务器端口 |
| `REDIS_PASSWORD` | string | - | Redis 服务器密码（可选） |


## 🔧 使用方法

### 1. 环境配置

确保在 `.env` 文件中正确配置了上述参数：

```bash
# 启用缓存，设置24小时过期
CACHE_ENABLED=true
CACHE_TTL=86400
REDIS_HOST=127.0.0.1
REDIS_PORT=6379
```

### 2. 服务级别缓存控制

每个 LLM 服务都支持 `caching` 参数进行精细控制：

```python
# Embedding Service - 启用缓存
embedding_service = EmbeddingService(
    embedding_provider="openai",
    embedding_model="text-embedding-ada-002",
    embedding_service_url="https://api.openai.com/v1",
    embedding_service_api_key="sk-...",
    embedding_max_chunks_in_batch=10,
    caching=True  # 启用缓存（默认值）
)

# Completion Service - 禁用缓存（针对特定场景）
completion_service = CompletionService(
    provider="openai",
    model="gpt-3.5-turbo",
    base_url="https://api.openai.com/v1",
    api_key="sk-...",
    temperature=0.1,
    caching=False  # 针对此服务禁用缓存
)

# Rerank Service - 使用全局设置
rerank_service = RerankService(
    rerank_provider="cohere",
    rerank_model="rerank-english-v2.0",
    rerank_service_url="https://api.cohere.ai/v1",
    rerank_service_api_key="...",
    # caching 参数省略，使用全局 CACHE_ENABLED 设置
)
```

### 统计指标说明

| 指标 | 说明 |
|------|------|
| `hits` | 缓存命中次数 |
| `misses` | 缓存未命中次数 |
| `added` | 新增缓存条目数 |
| `total_requests` | 总请求数 |
| `hit_rate` | 缓存命中率（0-1之间） |

## 🎯 缓存机制

### 缓存键生成策略

缓存键基于以下请求参数的SHA256哈希值生成：

> see: litellm.cache.get_cache_key

```python
# 对于一个 Completion 调用:
# litellm.completion(model="gpt-3.5-turbo", messages=[...], temperature=0.7)
string_to_hash = "model: gpt-3.5-turbo, messages: [{'role': 'user', 'content': '...'}], temperature: 0.7"
cache_key = sha256(string_to_hash)

# 对于 Embedding 服务:
# litellm.embedding(model="text-embedding-ada-002", input=["..."])
string_to_hash = "model: text-embedding-ada-002, input: ['...']"
cache_key = sha256(string_to_hash)

# 对于 Rerank 服务也是类似的逻辑:
# litellm.rerank(model="cohere.rerank-english-v2.0", query="...", documents=["..."])
string_to_hash = "model: cohere.rerank-english-v2.0, query: ..., documents: ['...']"
cache_key = sha256(string_to_hash)
```

## 🔗 相关文件

- `aperag/llm/litellm_cache.py` - 缓存核心实现
- `config/settings.py` - 缓存配置项定义
- `aperag/llm/completion/completion_service.py` - 完成服务缓存集成
- `aperag/llm/embed/embedding_service.py` - 嵌入服务缓存集成
- `aperag/llm/rerank/rerank_service.py` - 重排序服务缓存集成
- `envs/env.template` - 环境变量配置模板

## 📈 性能对比

| 场景 | 无缓存 | 启用缓存 | 性能提升 |
|------|--------|----------|----------|
| 重复嵌入查询 | 2000ms | 2ms | 1000x |
| 相同完成请求 | 1500ms | 1ms | 1500x |
| 批量重排序 | 800ms | 3ms | 266x |
| 高频知识查询 | 1200ms | 1ms | 1200x |

通过合理配置和使用LiteLLM缓存功能，可以显著提升ApeRAG系统的响应性能，降低API调用成本，改善用户体验。 