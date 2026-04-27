import asyncio
import logging

import redis
import redis.asyncio as async_redis

from aperag.cache.application import (
    NAMESPACE_EMBEDDING,
    NAMESPACE_EMBEDDING_DIMENSION,
    NAMESPACE_LLM_COMPLETION,
    NAMESPACE_PARSER_PREFLIGHT,
    NAMESPACE_REMOTE_PARSER,
    NAMESPACE_RERANK,
    NAMESPACE_WEB_READ,
    NAMESPACE_WEB_SEARCH,
    ApplicationCache,
    ApplicationCachePolicy,
    ApplicationRedisCacheBackend,
    NoopApplicationCacheBackend,
    SyncApplicationCache,
    SyncApplicationRedisCacheBackend,
    SyncNoopApplicationCacheBackend,
    application_cache_metrics,
)
from aperag.config import settings
from aperag.db.redis_manager import RedisConnectionManager

logger = logging.getLogger(__name__)

_async_cache: ApplicationCache | None = None
_async_cache_loop: asyncio.AbstractEventLoop | None = None
_sync_cache: SyncApplicationCache | None = None


def application_cache_policy(namespace: str, *, enabled: bool | None = None) -> ApplicationCachePolicy:
    return ApplicationCachePolicy(
        namespace=namespace,
        ttl_seconds=_ttl_for_namespace(namespace),
        enabled=settings.cache_enabled if enabled is None else enabled,
        cache_empty=namespace == NAMESPACE_WEB_SEARCH,
        max_value_bytes=settings.cache_max_value_bytes,
    )


async def get_application_cache() -> ApplicationCache:
    global _async_cache, _async_cache_loop
    loop = asyncio.get_running_loop()
    if _async_cache is not None and _async_cache_loop is loop:
        return _async_cache
    if _async_cache is not None and _async_cache_loop is not loop:
        _async_cache = ApplicationCache(
            backend=NoopApplicationCacheBackend(),
            default_policy=application_cache_policy("default", enabled=False),
        )
        _async_cache_loop = loop
        return _async_cache

    if not settings.cache_enabled:
        _async_cache = ApplicationCache(
            backend=NoopApplicationCacheBackend(),
            default_policy=application_cache_policy("default", enabled=False),
        )
        _async_cache_loop = loop
        return _async_cache

    try:
        if settings.cache_redis_url:
            redis_client = async_redis.Redis.from_url(
                settings.cache_redis_url,
                encoding="utf-8",
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True,
                health_check_interval=30,
            )
            await redis_client.ping()
        else:
            redis_client = await RedisConnectionManager.get_async_client()
        backend = ApplicationRedisCacheBackend(redis=redis_client)
    except Exception as e:
        logger.warning("Application cache disabled for async caller because Redis is unavailable: %s", e)
        backend = NoopApplicationCacheBackend()

    _async_cache = ApplicationCache(backend=backend, default_policy=application_cache_policy("default"))
    _async_cache_loop = loop
    return _async_cache


def get_sync_application_cache() -> SyncApplicationCache:
    global _sync_cache
    if _sync_cache is not None:
        return _sync_cache

    if not settings.cache_enabled:
        _sync_cache = SyncApplicationCache(
            backend=SyncNoopApplicationCacheBackend(),
            default_policy=application_cache_policy("default", enabled=False),
        )
        return _sync_cache

    try:
        if settings.cache_redis_url:
            redis_client = redis.Redis.from_url(
                settings.cache_redis_url,
                encoding="utf-8",
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True,
                health_check_interval=30,
            )
            redis_client.ping()
        else:
            redis_client = RedisConnectionManager.get_sync_client()
        backend = SyncApplicationRedisCacheBackend(redis=redis_client)
    except Exception as e:
        logger.warning("Application cache disabled for sync caller because Redis is unavailable: %s", e)
        backend = SyncNoopApplicationCacheBackend()

    _sync_cache = SyncApplicationCache(backend=backend, default_policy=application_cache_policy("default"))
    return _sync_cache


def reset_application_cache_for_tests() -> None:
    global _async_cache, _async_cache_loop, _sync_cache
    _async_cache = None
    _async_cache_loop = None
    _sync_cache = None
    application_cache_metrics.clear()


def _ttl_for_namespace(namespace: str) -> int:
    if namespace == NAMESPACE_LLM_COMPLETION:
        return settings.cache_llm_ttl_seconds or settings.cache_ttl
    if namespace in (NAMESPACE_EMBEDDING, NAMESPACE_EMBEDDING_DIMENSION):
        return settings.cache_embedding_ttl_seconds or settings.cache_ttl
    if namespace == NAMESPACE_RERANK:
        return settings.cache_rerank_ttl_seconds or settings.cache_ttl
    if namespace == NAMESPACE_WEB_SEARCH:
        return settings.cache_web_search_ttl_seconds
    if namespace == NAMESPACE_WEB_READ:
        return settings.cache_web_read_ttl_seconds
    if namespace == NAMESPACE_PARSER_PREFLIGHT:
        return settings.cache_parser_preflight_ttl_seconds
    if namespace == NAMESPACE_REMOTE_PARSER:
        return settings.cache_remote_parser_ttl_seconds
    return settings.cache_ttl
