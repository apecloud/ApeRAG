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

"""Agent-runtime tool lifecycle subsystem (Phase 8 D8.3, #75).

This subpackage implements the seven-point D9 ``§A4`` contract that
``agent-runtime-mcp-design.md`` pins as a precondition for the D8.3
tool/citation/consent/elicitation backend lane:

1. ``safe_name`` -- D9 §A1+§A6 SafeToolName normalization with
   collision-resistant sha256 suffix + ``(mcpServer, mcpToolName,
   safeName)`` reverse lookup.
2. ``registry`` -- D9 §1.1+§A5 three-tier MCP registry
   (system/bot/user) with namespace reservation, no silent override,
   and audit-logged admin alias.
3. ``authorization`` -- D9 §2 three-level authorization
   (visibility / invocation / consent) with the §2.2 default policy
   table and per-tool risk overrides.
4. ``args_cache`` -- D9 §A7 backend-private raw-args cache with short
   TTL; the wire-side ``argsPreview`` + ``argsHash`` redaction is
   delegated to ``uimessage.args_preview`` / ``uimessage.args_hash``
   so the wire and at-rest stores share one canonical implementation.

Stage 2 of the lane (consent / elicitation emit hooks, tool lifecycle
adapter to AI SDK part shapes, REST endpoints) lives in sibling
modules added once the at-rest ``data-*`` wrapped shape lock from
#74 is in main.
"""

from aperag.domains.agent_runtime.tools.args_cache import (
    InMemoryRawArgsCache,
    RawArgsCache,
    args_hash,
    args_preview,
)
from aperag.domains.agent_runtime.tools.authorization import (
    AuthorizationDecision,
    ToolAuthorizationPolicy,
    ToolRiskClassification,
    default_policy,
)
from aperag.domains.agent_runtime.tools.registry import (
    MCPServerEntry,
    MCPServerRegistry,
    RegistryConflictError,
    RegistryScope,
)
from aperag.domains.agent_runtime.tools.safe_name import (
    SAFE_NAME_RE,
    SafeNameRegistry,
    SafeToolNameResult,
    sanitize_tool_name,
)

__all__ = [
    "AuthorizationDecision",
    "InMemoryRawArgsCache",
    "MCPServerEntry",
    "MCPServerRegistry",
    "RawArgsCache",
    "RegistryConflictError",
    "RegistryScope",
    "SAFE_NAME_RE",
    "SafeNameRegistry",
    "SafeToolNameResult",
    "ToolAuthorizationPolicy",
    "ToolRiskClassification",
    "args_hash",
    "args_preview",
    "default_policy",
    "sanitize_tool_name",
]
