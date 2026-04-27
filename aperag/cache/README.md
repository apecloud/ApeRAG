# ApeRAG Cache

ApeRAG has two cache families with separate responsibilities:

- **Read-primitive cache**: parse-version-keyed MCP document read results. It keeps the existing L1 + Redis L2 design because read primitives are small, authorization-gated, and repeatedly read within one worker.
- **Application cache**: Redis-backed expensive provider/parser call results. It intentionally has no default L1 so private deployments have one shared cache to observe and clear.

## Application Cache

Application cache keys use:

```text
aperag:cache:v1:{namespace}:{sha256(canonical-json)}
```

Raw prompts, URLs, documents, and file bytes are not embedded in Redis key names.

Namespaces:

| Namespace | Caller |
| --- | --- |
| `llm_completion` | non-streaming completion |
| `embedding` | per-input embedding vectors |
| `embedding_dimension` | embedding dimension probes |
| `rerank` | query/documents rerank calls |
| `web_search` | search responses, excluding unavailable results |
| `web_read` | per-URL successful read results |
| `parser_preflight` | parser health probes |
| `remote_parser` | OCR/ASR/MinerU remote parser results |

## Read-Primitive Cache

Read-primitive cache keys use:

```text
read_primitive:{namespace}:{parts...}
```

The public configuration uses semantic names:

```bash
READ_PRIMITIVE_CACHE_L1_SIZE=256
READ_PRIMITIVE_CACHE_L2_TTL_SECONDS=3600
```

## Shared Configuration

```bash
CACHE_ENABLED=True
CACHE_TTL=86400
CACHE_REDIS_URL=
CACHE_LLM_TTL_SECONDS=
CACHE_EMBEDDING_TTL_SECONDS=
CACHE_RERANK_TTL_SECONDS=
CACHE_WEB_SEARCH_TTL_SECONDS=600
CACHE_WEB_READ_TTL_SECONDS=3600
CACHE_PARSER_PREFLIGHT_TTL_SECONDS=60
CACHE_REMOTE_PARSER_TTL_SECONDS=604800
CACHE_MAX_VALUE_BYTES=8388608
```

`CACHE_REDIS_URL` is optional. When unset, the application cache uses the existing memory Redis URL.

## Non-Goals

- Agent runtime Redis state is not application cache.
- Chat message history is not application cache.
- Redis locks and Celery broker data are infrastructure.
- API key authentication is not cached because deletion/rotation and `last_used` semantics need immediate database truth.
