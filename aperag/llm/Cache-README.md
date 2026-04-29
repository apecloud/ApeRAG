# LLM Cache Migration Note

ApeRAG no longer uses LiteLLM's built-in cache.

LLM and embedding caching is owned by the Redis-backed application
cache in `aperag/cache`. See `aperag/cache/README.md` for the current design.

The service-level `caching` flag is still supported:

- `caching=True` uses ApeRAG's application cache when the call type is cacheable.
- `caching=False` bypasses ApeRAG's application cache.
- Calls into LiteLLM pass `caching=False` to avoid double caching.

`aperag/llm/litellm_cache.py` remains only as a compatibility facade for older imports.
