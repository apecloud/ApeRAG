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

Phase 5 step 5-S5b adds the ``PromptTemplateOps`` Protocol for the
legacy ``aperag.service.prompt_template_service`` provider —
``prompt_template_service`` stays in ``aperag/service/`` through
Phase 6 cleanup, so the agent_runtime domain cannot ``from
aperag.service.*`` import it directly without tripping G1.

The ``ChatDocumentOps`` Protocol originally seeded in Phase 5 step 1
has been **retired** per msg=940bd884 simplification: after
``chat_document_service`` physically moved into the conversation
domain (Phase 5 step 5-S4d), ``runtime.py`` can reach it via a
direct cross-domain import (domain→domain is allowed by G1). The
Protocol is kept here only in archived form (below the ``__all__``
for the live surface) so any in-flight branch still referencing it
resolves the same shape; Phase 6 removes it outright.

``AuthenticatedUser`` stays per-domain (lesson 9a-ter); runtime
handlers only need ``id`` for turn ownership / lease checks.
"""

from __future__ import annotations

from typing import Any, Optional, Protocol, runtime_checkable


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
class PromptTemplateOps(Protocol):
    """Consumer-owned view of the legacy
    ``aperag.service.prompt_template_service``.

    Exposes the three surfaces ``runtime.py`` actually uses:

    * ``resolve_agent_system_prompt(bot, user_id)`` — returns the
      fully-resolved system prompt for the bot + user combination.
    * ``resolve_agent_query_prompt(bot, user_id)`` — returns the
      query prompt template (caller performs the variable
      substitution via ``build_agent_query_prompt``).
    * ``build_agent_query_prompt(chat_id, *, agent_message, user,
      template, has_chat_files)`` — bound through the DI slot for
      symmetry so the runtime does not have to split its hard-import
      between a singleton and a module-level helper.

    The concrete ``aperag.service.prompt_template_service`` module
    exposes the two methods on the ``prompt_template_service``
    singleton plus the module-level ``build_agent_query_prompt``
    function, so ``aperag/app.py`` wires an adapter that fans out to
    both to satisfy the Protocol.

    Phase 6 cleanup will either move ``prompt_template_service`` into
    a canonical domain home (agent_runtime or model_platform candidate
    per msg=65a3b27d) or retire the Protocol in favour of a direct
    cross-domain import.
    """

    async def resolve_agent_system_prompt(self, *, bot: Any, user_id: str) -> str: ...

    async def resolve_agent_query_prompt(self, *, bot: Any, user_id: str) -> str: ...

    def build_agent_query_prompt(
        self,
        chat_id: str,
        *,
        agent_message: Any,
        user: str,
        template: Optional[str] = None,
        has_chat_files: bool = False,
    ) -> str: ...


__all__ = [
    "AuthenticatedUser",
    "PromptTemplateOps",
]


# ``ChatDocumentOps`` retained for any in-flight caller that still
# imports it. Phase 5 step 5-S5b replaced the runtime's DI seam with a
# direct cross-domain import of
# ``aperag.domains.conversation.service.chat_document_service`` now
# that the conversation domain is merged — the Protocol definition is
# no longer wired at app startup and is scheduled for removal in
# Phase 6.


@runtime_checkable
class ChatDocumentOps(Protocol):  # pragma: no cover - retired, Phase 6 deletion candidate
    async def has_documents_in_chat(self, chat_id: str, user_id: str) -> bool: ...
