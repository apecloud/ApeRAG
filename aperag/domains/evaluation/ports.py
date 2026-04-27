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

"""Cross-domain contracts owned by the ``evaluation`` domain.

Phase 5 canonical ``msg=4a93c97e`` Q3 (ruling A) formalises the
existing ``dispatch_fn`` testability seam in
``aperag.evaluation_v2.worker.dispatch_evaluation_turn`` into two
consumer-owned Protocols:

* ``ChatSessionOps`` replaces the ``chat_service_global`` import at
  ``worker.py`` line 107 (``await chat_service_global.create_chat``).
* ``AgentTurnDispatchOps`` wraps the ``agent_runtime_manager`` fan-out
  that today reaches ``claim_turn`` / ``launch_turn`` / ``cancel_turn``
  plus the turn-service reads (``create_or_get_turn``,
  ``query_agent_turn``, ``get_turn_snapshot``). The Protocol keeps the
  surface at the method-level — ``AgentTurnDispatchOps`` hides the
  manager handle so the worker never has to know which domain owns
  the runtime.

``AuthenticatedUser`` is defined per-domain (lesson 9a-ter) with only
``id``; evaluation runs are driven by ``user_id`` strings, not by
role-aware auth context.

All Protocols are ``@runtime_checkable`` so the G18 alt runtime smoke
in ``5-S8`` can assert the wire-up against the concrete instances
plugged in at ``aperag/app.py`` startup.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class AuthenticatedUser(Protocol):
    """Per-domain auth context (lesson 9a-ter).

    Only ``id`` is pinned — evaluation never inspects ``role`` once the
    router authenticates the caller.
    """

    id: Any


@runtime_checkable
class ChatSessionOps(Protocol):
    """``worker.py``'s view of the legacy chat service.

    ``dispatch_evaluation_turn`` allocates a fresh chat for each
    evaluation turn so the snapshot-on-run audit trail can pin the
    agent-runtime output back to a discrete ``chat_id``. The single
    call site is:

        chat_view = await chat_service_global.create_chat(
            user_id, bot_id
        )

    Phase 5 canonical Q3 (ruling A) swaps the hard-import for a DI
    slot wired at startup; conversation's concrete ``ChatService``
    satisfies this Protocol structurally.
    """

    async def create_chat(self, user_id: str, bot_id: str) -> Any: ...


@runtime_checkable
class AgentTurnDispatchOps(Protocol):
    """Formalised replacement for the ``dispatch_fn`` testability seam.

    Today ``dispatch_evaluation_turn`` grabs
    ``aperag.agent_runtime.runtime.agent_runtime_manager`` and reaches
    through it for four distinct concerns:

    * ``turn_service.create_or_get_turn(user_id, chat_id, request)``
    * ``claim_turn(turn_id)`` / ``launch_turn(...)``
    * ``turn_service.db_ops.query_agent_turn(user_id, chat_id, turn_id)``
      (poll loop)
    * ``cancel_turn(turn_id)`` (timeout) and
      ``turn_service.get_turn_snapshot(user_id, chat_id, turn_id)``
      (post-run reconciliation)

    Phase 5 canonical Q3 (ruling A) hoists all of that into a single
    consumer-owned Protocol so the worker never imports the runtime
    directly. ``agent_runtime`` provides a thin facade that satisfies
    the Protocol; ``dispatch_fn`` stays as an optional test override
    parameter on ``dispatch_evaluation_turn`` for backward-compat.

    ``request`` is typed ``Any`` because the concrete
    ``CreateTurnRequest`` class lives in the agent_runtime domain —
    pinning it here would pull the class into evaluation's import
    graph and defeat the whole point of the Protocol. Callers
    construct the request via the facade; the evaluation worker
    passes it through opaquely.
    """

    async def create_or_get_turn(
        self,
        user_id: str,
        chat_id: str,
        request: Any,
    ) -> Any: ...

    async def claim_turn(self, turn_id: str) -> Any: ...

    def launch_turn(
        self,
        *,
        turn: Any,
        chat: Any,
        bot: Any,
        user: str,
        request: Any,
        lease_owner: Any,
    ) -> None: ...

    async def query_agent_turn(
        self,
        user_id: str,
        chat_id: str,
        turn_id: str,
    ) -> Any: ...

    async def cancel_turn(self, turn_id: str) -> Any: ...

    async def get_turn_snapshot(
        self,
        user_id: str,
        chat_id: str,
        turn_id: str,
    ) -> Any: ...


__all__ = [
    "AuthenticatedUser",
    "ChatSessionOps",
    "AgentTurnDispatchOps",
]
