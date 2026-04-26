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

"""Memory-Redis client accessor.

Phase 8 D8.6 (#80) hard-cut removed the legacy
``RedisChatMessageHistory`` class and the streaming-protocol JSON
helpers (``success_response`` / ``fail_response`` / ``start_response``
/ ``references_response`` / ``stop_response``). Chat history is now
canonical at-rest in the ``agent_message`` table (D8.2 #74) and the
streaming protocol is the AI SDK v5 wire emit path (D8.1 #73).

This module is reduced to one function — :func:`get_async_redis_client`
— which is still used outside the chat-history context (e.g.,
:mod:`aperag.utils.weixin.client`).
"""

from __future__ import annotations

_async_redis_client = None


def get_async_redis_client():
    """Return the lazily-instantiated process-wide async Redis client.

    Backed by :attr:`aperag.config.Config.memory_redis_url`. The client
    is shared across modules that need a non-locking Redis connection;
    :class:`aperag.concurrent_control.redis_lock.RedisLock` builds its
    own connection separately.
    """

    global _async_redis_client
    if _async_redis_client is None:
        import redis.asyncio as redis

        from aperag.config import settings

        _async_redis_client = redis.Redis.from_url(settings.memory_redis_url)
    return _async_redis_client


__all__ = ["get_async_redis_client"]
