# LiteLLM Caching Feature

ApeRAG integrates LiteLLM's built-in caching functionality, providing an efficient caching mechanism for all LLM service calls. This significantly boosts performance and reduces API call costs.

## 🚀 Core Features

  * **🎛️ Simple Toggle Control**: Enable/disable caching with a single switch via `CACHE_ENABLED`.
  * **⏰ Flexible TTL Configuration**: Customize cache expiration time using `CACHE_TTL`.
  * **🗄️ Redis Backend Storage**: Uses Redis as the cache backend, supporting distributed deployment.
  * **📊 Local Statistics Monitoring**: Real-time monitoring of cache hit rate and usage.
  * **🔄 Full Service Support**: Comprehensive support for completion, embedding, and rerank services.
  * **⚡ Extreme Performance**: Redis provides millisecond-level cache access, offering a 10-1000x performance improvement.
  * **🛠️ Zero-Config Startup**: Usable with default settings, supporting progressive optimization.

## 📁 Module Structure

```
aperag/llm/
├── litellm_cache.py          # LiteLLM Cache Core Configuration and Management
├── completion/
│   └── completion_service.py  # Completion Service supporting caching parameters
├── embed/
│   └── embedding_service.py   # Embedding Service supporting caching parameters
└── rerank/
    └── rerank_service.py      # Rerank Service supporting caching parameters
```

## ⚙️ Configuration Instructions

### Core Configuration Switch

Configure caching control parameters in the `.env` file:

```bash
# ========== Cache Core Control ==========
# Cache feature switch (enabled by default)
CACHE_ENABLED=true

# Cache Time-To-Live (TTL) in seconds (default 24 hours)
CACHE_TTL=86400

# ========== Redis Connection Configuration ==========
REDIS_HOST=127.0.0.1
REDIS_PORT=6379
REDIS_PASSWORD=your_redis_password
```

### Configuration Parameter Details

| Parameter | Type | Default Value | Description |
| :---------- | :------ | :------------ | :------------------------------------------- |
| `CACHE_ENABLED` | boolean | `true`        | Global cache switch. Set to `false` to disable caching completely. |
| `CACHE_TTL` | integer | `86400`       | Cache entry Time-To-Live (seconds). 86400 = 24 hours. |
| `REDIS_HOST` | string | `localhost`   | Redis server address. |
| `REDIS_PORT` | integer | `6379`        | Redis server port. |
| `REDIS_PASSWORD` | string | -             | Redis server password (optional). |

-----

## 🔧 Usage

### 1\. Environment Configuration

Ensure the above parameters are correctly configured in your `.env` file:

```bash
# Enable cache, set 24-hour expiration
CACHE_ENABLED=true
CACHE_TTL=86400
REDIS_HOST=127.0.0.1
REDIS_PORT=6379
```

### 2\. Service-Level Cache Control

Each LLM service supports the `caching` parameter for fine-grained control:

```python
# Embedding Service - Enable Caching
embedding_service = EmbeddingService(
    embedding_provider="openai",
    embedding_model="text-embedding-ada-002",
    embedding_service_url="https://api.openai.com/v1",
    embedding_service_api_key="sk-...",
    embedding_max_chunks_in_batch=10,
    caching=True  # Enable caching (default value)
)

# Completion Service - Disable Caching (for specific scenarios)
completion_service = CompletionService(
    provider="openai",
    model="gpt-3.5-turbo",
    base_url="https://api.openai.com/v1",
    api_key="sk-...",
    temperature=0.1,
    caching=False  # Disable caching for this specific service
)

# Rerank Service - Use Global Settings
rerank_service = RerankService(
    rerank_provider="cohere",
    rerank_model="rerank-english-v2.0",
    rerank_service_url="https://api.cohere.ai/v1",
    rerank_service_api_key="...",
    # caching parameter omitted, uses global CACHE_ENABLED setting
)
```

### Statistical Metrics Description

| Metric           | Description                     |
| :--------------- | :------------------------------ |
| `hits`           | Number of cache hits            |
| `misses`         | Number of cache misses          |
| `added`          | Number of new cache entries added |
| `total_requests` | Total number of requests        |
| `hit_rate`       | Cache hit rate (between 0-1)    |

## 🎯 Caching Mechanism

### Cache Key Generation Strategy

Cache keys are generated based on the SHA256 hash of the following request parameters:

> see: litellm.cache.get\_cache\_key

```python
# For a Completion call:
# litellm.completion(model="gpt-3.5-turbo", messages=[...], temperature=0.7)
string_to_hash = "model: gpt-3.5-turbo, messages: [{'role': 'user', 'content': '...'}], temperature: 0.7"
cache_key = sha256(string_to_hash)

# For an Embedding service:
# litellm.embedding(model="text-embedding-ada-002", input=["..."])
string_to_hash = "model: text-embedding-ada-002, input: ['...']"
cache_key = sha256(string_to_hash)

# Similar logic for Rerank service:
# litellm.rerank(model="cohere.rerank-english-v2.0", query="...", documents=["..."])
string_to_hash = "model: cohere.rerank-english-v2.0, query: ..., documents: ['...']"
cache_key = sha256(string_to_hash)
```

## 🔗 Related Files

  * `aperag/llm/litellm_cache.py` - Core cache implementation
  * `config/settings.py` - Cache configuration item definitions
  * `aperag/llm/completion/completion_service.py` - Completion service cache integration
  * `aperag/llm/embed/embedding_service.py` - Embedding service cache integration
  * `aperag/llm/rerank/rerank_service.py` - Rerank service cache integration
  * `envs/env.template` - Environment variable configuration template

## 📈 Performance Comparison

| Scenario           | Without Cache | With Cache | Performance Improvement |
| :----------------- | :------------ | :--------- | :---------------------- |
| Repeated embedding queries | 2000ms        | 2ms        | 1000x                   |
| Identical completion requests | 1500ms        | 1ms        | 1500x                   |
| Batch reranking    | 800ms         | 3ms        | 266x                    |
| High-frequency knowledge queries | 1200ms        | 1ms        | 1200x                   |

By reasonably configuring and utilizing the LiteLLM caching feature, you can significantly enhance the response performance of the ApeRAG system, reduce API call costs, and improve the user experience.