# LLM 缓存迁移说明

ApeRAG 不再使用 LiteLLM 内置缓存。

LLM、embedding、rerank 的缓存由 `aperag/cache` 中的 Redis-backed 应用级缓存负责。当前设计请参考 `aperag/cache/README.md`。

LLM 服务上的 `caching` 参数仍然保留：

- `caching=True`：在调用类型可缓存时使用 ApeRAG 应用级缓存。
- `caching=False`：绕过 ApeRAG 应用级缓存。
- 调用 LiteLLM 时统一传 `caching=False`，避免双缓存。

`aperag/llm/litellm_cache.py` 仅作为旧 import 的兼容 facade 保留。
