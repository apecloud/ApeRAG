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

"""Cross-domain contracts owned by the ``agent_runtime`` domain.

Lesson 9a-quad canonical: the agent runtime reads one datum from
``conversation`` at request time — whether the current chat has any
attached documents, so ``build_agent_query_prompt`` can switch between
the "chat with attachments" and "plain prompt" branches. Phase 5
canonical (``msg=4a93c97e`` Q2, ruling A) replaces the existing
``from aperag.service.chat_document_service import chat_document_service``
hard-import in ``runtime.py`` with a ``ChatDocumentOps`` Protocol DI
slot wired at ``aperag/app.py`` startup.

Per ``msg=92f5788d`` Section 3 + ``msg=ce960fbc`` bless, the
``AuthenticatedUser`` Protocol is per-domain (lesson 9a-ter) and kept
minimal — ``id`` alone, since agent_runtime never inspects ``role``.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class AuthenticatedUser(Protocol):
    """Per-domain auth context (lesson 9a-ter).

    Runtime handlers only need ``id`` for turn ownership / lease
    checks. ``role`` is *not* pinned here because admin-vs-user
    distinctions are settled at the router layer before the runtime
    is even called.
    """

    id: Any


@runtime_checkable
class ChatDocumentOps(Protocol):
    """``runtime.py``'s view of the legacy chat-document service.

    Phase 5 canonical Q2 (``msg=4a93c97e`` ruling A) formalises the
    single call site at ``aperag.agent_runtime.runtime`` line 262:

        has_chat_files = await chat_document_service.has_documents_in_chat(
            chat.id, user
        )

    into a Protocol + DI slot owned by agent_runtime. The legacy
    ``chat_document_service`` singleton satisfies the Protocol
    structurally; once conversation physically moves the service
    (Phase 5 step 4, ``5-S4``) the DI wire-up at ``aperag/app.py``
    points at the new ``aperag.domains.conversation.service`` path.
    """

    async def has_documents_in_chat(
        self,
        chat_id: str,
        user_id: str,
    ) -> bool: ...


__all__ = [
    "AuthenticatedUser",
    "ChatDocumentOps",
]
