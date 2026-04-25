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

"""Backend-private raw-args cache -- D9 §3.1 / §A7.

When a tool call is gated by consent, the agent runtime needs to:

1. Compute a redacted ``argsPreview`` and stable ``argsHash`` over
   the raw args, surface them in the ``data-tool-consent`` part on
   the wire / at-rest store.
2. **NOT** put the raw args anywhere durable. Raw args may carry
   secrets (api keys, file paths, message bodies) that must never
   reach the persisted UIMessage history or the FE wire.
3. After the user approves, fetch the raw args back to actually
   execute the tool. Denial -> drop the raw args.

This module owns the in-process raw-args store. The wire-side
``argsPreview`` and ``argsHash`` are produced via
:func:`aperag.domains.agent_runtime.uimessage.args_preview` and
:func:`aperag.domains.agent_runtime.uimessage.args_hash` so the wire
emitter and at-rest store share one canonical hashing function (no
double implementation, no drift).

Storage choice: in-process is the default; the protocol /
:class:`InMemoryRawArgsCache` is structured so a Redis-backed
implementation can drop in by satisfying the same async surface. We
explicitly do NOT default to Redis here -- single-process turn
runtime keeps raw-args local to the worker that holds the turn
lease, eliminating one network round-trip from the consent hot path
and avoiding any temptation to persist the raw args by accident.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Optional, Protocol

from aperag.domains.agent_runtime.uimessage import args_hash, args_preview

# D9 §3.3 says consent-pending > 5 min -> ``state: "expired"``. The
# cache TTL must outlive that policy window so the consent decision
# can still resolve to raw args even on the boundary.
DEFAULT_RAW_ARGS_TTL_SECONDS = 360


class RawArgsCache(Protocol):
    """Interface the consent flow uses to store + retrieve raw args.

    All methods are async so a Redis implementation drops in
    transparently. The in-memory default below uses ``asyncio`` locks
    rather than ``threading`` because the agent runtime is async-end-
    to-end; mixing sync locks would block the event loop.
    """

    async def put(self, consent_id: str, raw_args: Any, *, ttl_seconds: int = DEFAULT_RAW_ARGS_TTL_SECONDS) -> None: ...

    async def get(self, consent_id: str) -> Optional[Any]: ...

    async def delete(self, consent_id: str) -> bool: ...


class InMemoryRawArgsCache:
    """Default :class:`RawArgsCache` -- per-process dict + lazy TTL eviction.

    The store is keyed by ``consent_id`` (stable for the lifetime of
    the consent prompt -- typically the tool call id) and entries
    expire at ``put`` + ``ttl_seconds``. Evictions happen lazily on
    each ``get`` to avoid a separate sweeper task; this is fine
    because the cache is small (one entry per pending consent in a
    process) and high-traffic eviction would only happen under load
    where ``get`` is also frequent.
    """

    def __init__(self, *, clock: Optional[callable] = None):  # type: ignore[valid-type]
        self._lock = asyncio.Lock()
        self._entries: dict[str, tuple[float, Any]] = {}
        # Tests inject a deterministic clock; production uses
        # ``time.monotonic`` so wall-clock skew (NTP jumps, manual
        # adjustments) cannot affect TTL maths.
        self._clock = clock or time.monotonic

    async def put(
        self,
        consent_id: str,
        raw_args: Any,
        *,
        ttl_seconds: int = DEFAULT_RAW_ARGS_TTL_SECONDS,
    ) -> None:
        if not consent_id:
            raise ValueError("consent_id must be non-empty")
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        expiry = self._clock() + ttl_seconds
        async with self._lock:
            self._entries[consent_id] = (expiry, raw_args)

    async def get(self, consent_id: str) -> Optional[Any]:
        async with self._lock:
            entry = self._entries.get(consent_id)
            if entry is None:
                return None
            expiry, value = entry
            if self._clock() >= expiry:
                self._entries.pop(consent_id, None)
                return None
            return value

    async def delete(self, consent_id: str) -> bool:
        async with self._lock:
            return self._entries.pop(consent_id, None) is not None

    async def cleanup_expired(self) -> int:
        """Sweep expired entries proactively.

        Useful for test determinism + ad-hoc admin ops. Returns the
        count of evicted entries.
        """

        now = self._clock()
        async with self._lock:
            expired = [cid for cid, (expiry, _) in self._entries.items() if now >= expiry]
            for cid in expired:
                self._entries.pop(cid, None)
            return len(expired)


__all__ = [
    "DEFAULT_RAW_ARGS_TTL_SECONDS",
    "InMemoryRawArgsCache",
    "RawArgsCache",
    "args_hash",
    "args_preview",
]
