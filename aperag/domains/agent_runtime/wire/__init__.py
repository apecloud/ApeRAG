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

"""AI SDK v5 UI Message Stream wire layer for the agent runtime.

This sub-package landed in Phase 8 / D8.1 (#73). It owns the
backend-side translation from ``AgentTimelineEventEnvelope`` events
(the runtime's internal timeline contract) into AI SDK v5 ``UI
Message Stream Protocol`` parts that the FE ``@ai-sdk/react``
consumer can parse.

Contract canonical: ``docs/modularization/agent-message-protocol-design.md``
(D8 protocol/storage canonical) + ``docs/modularization/agent-runtime-mcp-design.md``
(D9 MCP-capable agent runtime canonical).

Modules:
    parts       -- Pydantic models + ``StreamPart`` discriminated union
    translator  -- ``translate_envelope()`` pure function + per-turn
                   ``TranslatorState`` lifecycle bookkeeping
"""

from aperag.domains.agent_runtime.wire.parts import (
    StreamPart,
    StreamPartAdapter,
    dump_part,
    parse_part,
)
from aperag.domains.agent_runtime.wire.translator import (
    TranslatorState,
    translate_envelope,
)

__all__ = [
    "StreamPart",
    "StreamPartAdapter",
    "TranslatorState",
    "dump_part",
    "parse_part",
    "translate_envelope",
]
