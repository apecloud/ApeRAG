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

``PromptTemplateOps`` is the consumer-owned Protocol seam for
``aperag.service.prompt_template_service``. The service is a
standalone-infrastructure module (cross-cutting across agent_runtime
runtime calls, conversation bot-config resolution, indexing prompt
resolution, and a user-facing ``/prompts`` REST surface) with no
natural domain home — the Protocol + DI seam is its permanent
integration point under G1.

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
    """Consumer-owned view of ``aperag.service.prompt_template_service``.

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
