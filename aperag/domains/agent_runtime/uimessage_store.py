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

"""DB + Redis at-rest persistence for UIMessage (Phase 8 D8.2).

This module wraps the read/write surface for the new ``agent_message``
table and the matching Redis snapshot key
(``agent_runtime:turn:<id>:message``) introduced in D8.2. Callers go
through :class:`UIMessageStore` rather than touching the SQLAlchemy
ORM directly so the wire emitter (D8.1, #73) and the future FE
reload endpoint share a single canonical write/read contract:

* :meth:`UIMessageStore.write` filters transient parts via
  :func:`uimessage.persistable_parts`, persists the at-rest envelope
  to both the ``agent_message`` row and the Redis snapshot, and
  refreshes ``gmt_updated``. Idempotent on ``turn_id``.
* :meth:`UIMessageStore.read` returns the at-rest UIMessage by
  ``turn_id``; Redis is consulted first as a low-latency cache,
  falling back to the DB row.
* :meth:`UIMessageStore.delete` is intended for D8.6 cleanup; it
  removes both the DB row and the Redis snapshot.

The contract test in
``tests/unit_test/agent_runtime/test_uimessage_at_rest.py`` pins the
following invariants (the prerequisites Weston named for D8.4b
unblocking, msg=50c90f6f / msg=cef89ed8):

1. Round-trip — every persistable variant of ``UIMessagePart``
   survives a ``write`` → ``read`` cycle byte-for-byte.
2. Transient exclusion — ``data-activity`` parts are stripped before
   write and never reappear on read.
3. Snapshot consistency — Redis snapshot and DB row return the same
   UIMessage shape; deletion of either is recoverable from the
   other.
"""

from __future__ import annotations

from typing import Any, Optional

from sqlalchemy import select

from aperag.domains.agent_runtime.db.models import AgentMessage
from aperag.domains.agent_runtime.storage import AgentRuntimeRedisStore
from aperag.domains.agent_runtime.uimessage import UIMessage, persistable_parts
from aperag.utils.utils import utc_now


class UIMessageStore:
    """Combined DB + Redis store for at-rest UIMessage snapshots.

    The DB row in ``agent_message`` is the durable authority; the
    Redis snapshot is a low-latency reload cache that mirrors the
    same shape. Writes go to both; reads prefer Redis but fall back
    to DB when Redis is cold (e.g. after TTL expiry on long-idle
    turns).
    """

    def __init__(self, db_ops: Any, redis_store: AgentRuntimeRedisStore):
        self._db_ops = db_ops
        self._redis_store = redis_store

    async def write(
        self,
        *,
        turn_id: str,
        chat_id: str,
        message: UIMessage,
    ) -> UIMessage:
        """Persist (or refresh) the UIMessage for ``turn_id``.

        Transient parts are filtered before persistence — the wire
        emitter still sees them on the live SSE stream, but they are
        never replayed on reload (consistent with the design of
        ``data-activity`` as ephemeral UX state).
        """

        cleaned = message.model_copy(update={"parts": persistable_parts(message.parts)})
        payload = cleaned.model_dump(mode="json", by_alias=True)
        await self._db_ops.upsert_agent_message(
            turn_id=turn_id,
            chat_id=chat_id,
            role=cleaned.role,
            schema_version=cleaned.schema_version,
            parts=payload["parts"],
            gmt_updated=utc_now(),
        )
        await self._redis_store.write_message_snapshot(turn_id, payload)
        return cleaned

    async def read(self, turn_id: str) -> Optional[UIMessage]:
        """Return the at-rest UIMessage for ``turn_id``, or ``None``."""

        snapshot = await self._redis_store.read_message_snapshot(turn_id)
        if snapshot is not None:
            return UIMessage.model_validate(snapshot)
        row = await self._db_ops.query_agent_message(turn_id)
        if row is None:
            return None
        return UIMessage.model_validate(
            {
                "id": row.id,
                "role": row.role,
                "schema_version": row.schema_version,
                "parts": row.parts,
            }
        )

    async def delete(self, turn_id: str) -> None:
        """Remove both the DB row and the Redis snapshot.

        Intended for D8.6 cleanup or test fixtures; production code
        should not call this on a turn that has been served to a
        client.
        """

        await self._db_ops.delete_agent_message(turn_id)
        await self._redis_store.delete_message_snapshot(turn_id)


class UIMessageDbOps:
    """SQLAlchemy-bound implementation of the operations
    :class:`UIMessageStore` needs.

    Kept separate from :class:`UIMessageStore` so unit tests can
    swap in a minimal in-memory fake (see
    ``test_uimessage_at_rest.py::_FakeDbOps``).
    """

    def __init__(self, session_factory: Any):
        self._session_factory = session_factory

    async def upsert_agent_message(
        self,
        *,
        turn_id: str,
        chat_id: str,
        role: str,
        schema_version: str,
        parts: list[dict[str, Any]],
        gmt_updated: Any,
    ) -> None:
        async for session in self._session_factory():
            existing = await session.execute(select(AgentMessage).where(AgentMessage.turn_id == turn_id))
            row = existing.scalars().first()
            if row is None:
                row = AgentMessage(
                    turn_id=turn_id,
                    chat_id=chat_id,
                    role=role,
                    schema_version=schema_version,
                    parts=parts,
                )
                session.add(row)
            else:
                row.role = role
                row.schema_version = schema_version
                row.parts = parts
                row.gmt_updated = gmt_updated
            await session.commit()
            return

    async def query_agent_message(self, turn_id: str) -> Optional[AgentMessage]:
        async for session in self._session_factory():
            result = await session.execute(select(AgentMessage).where(AgentMessage.turn_id == turn_id))
            return result.scalars().first()
        return None

    async def delete_agent_message(self, turn_id: str) -> None:
        async for session in self._session_factory():
            existing = await session.execute(select(AgentMessage).where(AgentMessage.turn_id == turn_id))
            row = existing.scalars().first()
            if row is not None:
                await session.delete(row)
                await session.commit()
            return
