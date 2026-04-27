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

import json
import os
from typing import Any

from aperag.db.redis_manager import RedisConnectionManager
from aperag.domains.agent_runtime.schemas import AgentTimelineEventEnvelope

DEFAULT_AGENT_TURN_LEASE_TTL_SECONDS = int(os.getenv("APERAG_AGENT_TURN_LEASE_TTL_SECONDS", "300"))

_RENEW_TURN_LEASE_SCRIPT = """
if redis.call('GET', KEYS[1]) == ARGV[1] then
  return redis.call('EXPIRE', KEYS[1], tonumber(ARGV[2]))
end
return 0
"""

_RELEASE_TURN_LEASE_SCRIPT = """
if redis.call('GET', KEYS[1]) == ARGV[1] then
  return redis.call('DEL', KEYS[1])
end
return 0
"""


class AgentRuntimeRedisStore:
    def __init__(self, prefix: str = "agent_runtime", ttl_seconds: int = 86400):
        self.prefix = prefix
        self.ttl_seconds = ttl_seconds

    def _events_key(self, turn_id: str) -> str:
        return f"{self.prefix}:turn:{turn_id}:events"

    def _state_key(self, turn_id: str) -> str:
        return f"{self.prefix}:turn:{turn_id}:state"

    def _cancel_key(self, turn_id: str) -> str:
        return f"{self.prefix}:turn:{turn_id}:cancelled"

    def _lease_key(self, turn_id: str) -> str:
        return f"{self.prefix}:turn:{turn_id}:lease"

    async def append_event(self, event: AgentTimelineEventEnvelope) -> None:
        client = await RedisConnectionManager.get_async_client()
        await client.rpush(self._events_key(event.turn_id), event.model_dump_json())
        await client.expire(self._events_key(event.turn_id), self.ttl_seconds)

    async def get_events_after(self, turn_id: str, after_sequence: int = 0, limit: int = 500) -> list[dict[str, Any]]:
        client = await RedisConnectionManager.get_async_client()
        start_index = max(after_sequence, 0)
        raw_events = await client.lrange(self._events_key(turn_id), start_index, start_index + limit - 1)
        return [json.loads(item) for item in raw_events]

    async def get_all_events(self, turn_id: str) -> list[dict[str, Any]]:
        client = await RedisConnectionManager.get_async_client()
        raw_events = await client.lrange(self._events_key(turn_id), 0, -1)
        return [json.loads(item) for item in raw_events]

    async def update_runtime_state(self, turn_id: str, state: dict[str, Any]) -> None:
        client = await RedisConnectionManager.get_async_client()
        await client.set(self._state_key(turn_id), json.dumps(state), ex=self.ttl_seconds)

    async def merge_runtime_state(self, turn_id: str, updates: dict[str, Any]) -> dict[str, Any]:
        state = await self.get_runtime_state(turn_id) or {}
        state.update(updates)
        await self.update_runtime_state(turn_id, state)
        return state

    async def get_runtime_state(self, turn_id: str) -> dict[str, Any] | None:
        client = await RedisConnectionManager.get_async_client()
        raw = await client.get(self._state_key(turn_id))
        return json.loads(raw) if raw else None

    async def mark_cancelled(self, turn_id: str) -> None:
        client = await RedisConnectionManager.get_async_client()
        await client.set(self._cancel_key(turn_id), "1", ex=self.ttl_seconds)

    async def clear_cancelled(self, turn_id: str) -> None:
        client = await RedisConnectionManager.get_async_client()
        await client.delete(self._cancel_key(turn_id))

    async def is_cancelled(self, turn_id: str) -> bool:
        client = await RedisConnectionManager.get_async_client()
        return await client.exists(self._cancel_key(turn_id)) > 0

    async def try_claim_turn(self, turn_id: str, owner_token: str, ttl_seconds: int | None = None) -> bool:
        client = await RedisConnectionManager.get_async_client()
        ttl = max(ttl_seconds or DEFAULT_AGENT_TURN_LEASE_TTL_SECONDS, 1)
        claimed = await client.set(self._lease_key(turn_id), owner_token, ex=ttl, nx=True)
        return bool(claimed)

    async def renew_turn_claim(self, turn_id: str, owner_token: str, ttl_seconds: int | None = None) -> bool:
        client = await RedisConnectionManager.get_async_client()
        ttl = max(ttl_seconds or DEFAULT_AGENT_TURN_LEASE_TTL_SECONDS, 1)
        renewed = await client.eval(_RENEW_TURN_LEASE_SCRIPT, 1, self._lease_key(turn_id), owner_token, ttl)
        return bool(renewed)

    async def release_turn_claim(self, turn_id: str, owner_token: str) -> bool:
        client = await RedisConnectionManager.get_async_client()
        released = await client.eval(_RELEASE_TURN_LEASE_SCRIPT, 1, self._lease_key(turn_id), owner_token)
        return bool(released)
