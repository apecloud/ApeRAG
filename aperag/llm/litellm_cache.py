# Copyright 2025 ApeCloud, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
LiteLLM Cache Configuration

This module configures LiteLLM's built-in caching functionality with:
- Redis cache storage
- Custom cache key generation
- Cache hit/miss tracking
- Cache statistics
"""

import logging
from typing import Any, Dict

import litellm
from litellm.types.caching import LiteLLMCacheType

logger = logging.getLogger(__name__)

# Local in-memory statistics
# Note: These are simple integer operations that may not be thread-safe
# in multi-threaded environments, but are acceptable for monitoring purposes.
# In multi-process environments (e.g., Celery prefork), each process maintains its own stats.
_cache_stats = {
    "hits": 0,
    "misses": 0,
    "added": 0,
    "total_requests": 0,
}


# doc: https://docs.litellm.ai/docs/caching/all_caches#enabling-cache
# All parameters for cache: https://docs.litellm.ai/docs/caching/all_caches#cache-initialization-parameters
def setup_litellm_cache(default_type=LiteLLMCacheType.DISK):
    from litellm.caching.caching import CacheMode

    from config.settings import CACHE_ENABLED, CACHE_TTL, REDIS_HOST, REDIS_PASSWORD, REDIS_PORT

    if not CACHE_ENABLED:
        return

    litellm.enable_cache(
        type=default_type,
        mode=CacheMode.default_on,
        host=REDIS_HOST,
        port=REDIS_PORT,
        password=REDIS_PASSWORD,
        ttl=CACHE_TTL,
        disk_cache_dir="/tmp/litellm_cache",
    )
    # Setup custom cache handlers with local stats tracking
    # Note: Only setup if cache was successfully initialized
    if litellm.cache is not None:
        setup_custom_get_cache()
        setup_custom_add_cache()
        logger.info("LiteLLM cache with local statistics initialized")


def disable_litellm_cache():
    litellm.disable_cache()


def setup_custom_get_cache_key():
    def custom_get_cache_key(*args, **kwargs):
        # return key to use for your cache:
        key = (
            kwargs.get("model", "")
            + str(kwargs.get("messages", ""))
            + str(kwargs.get("temperature", ""))
            + str(kwargs.get("logit_bias", ""))
        )
        print("key for cache", key)
        return key

    if litellm.cache is not None:
        litellm.cache.get_cache_key = custom_get_cache_key


def setup_custom_add_cache():
    """
    Wraps litellm.cache.add_cache to include local statistics for cache additions.
    """
    if litellm.cache is None:
        return

    # Store the original method
    original_add_cache = litellm.cache.add_cache

    def custom_add_cache(result, *args, **kwargs):
        # Update local stats - simple increment, may not be atomic in multi-threaded env
        global _cache_stats
        _cache_stats["added"] += 1
        logger.debug("LiteLLM Cache ADD")

        # Call the original caching function
        return original_add_cache(result, *args, **kwargs)

    # Replace the method
    litellm.cache.add_cache = custom_add_cache


def setup_custom_get_cache():
    """
    Wraps litellm.cache.get_cache to include local hit/miss statistics.
    """
    if litellm.cache is None:
        return

    # Store the original method
    original_get_cache = litellm.cache.get_cache

    def custom_get_cache(*args, **kwargs):
        # Call the original function to get the result from cache
        result = original_get_cache(*args, **kwargs)

        # Update local stats - simple increment, may not be atomic in multi-threaded env
        global _cache_stats
        _cache_stats["total_requests"] += 1
        if result is not None:
            _cache_stats["hits"] += 1
            logger.info("LiteLLM Cache HIT")
        else:
            _cache_stats["misses"] += 1
            logger.debug("LiteLLM Cache MISS")

        return result

    # Replace the method
    litellm.cache.get_cache = custom_get_cache


def get_cache_stats() -> Dict[str, Any]:
    """
    Get local in-memory cache statistics for the current process.

    Returns:
        Dict containing cache statistics including hit rate calculation.
    """
    # Create a copy to avoid modification during read
    stats = _cache_stats.copy()

    # Calculate hit rate
    if stats["total_requests"] > 0:
        stats["hit_rate"] = round(stats["hits"] / stats["total_requests"], 4)
    else:
        stats["hit_rate"] = 0.0

    # Add metadata
    stats["cache_type"] = "local_memory"
    stats["note"] = "Process-specific stats, not thread-safe"

    return stats


def clear_cache_stats() -> None:
    """Reset local in-memory cache statistics for the current process."""
    global _cache_stats
    _cache_stats = {
        "hits": 0,
        "misses": 0,
        "added": 0,
        "total_requests": 0,
    }
    logger.info("Local cache statistics cleared")
