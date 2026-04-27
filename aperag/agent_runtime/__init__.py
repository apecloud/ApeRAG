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

"""Legacy shim package for ``aperag/agent_runtime/`` after Phase 5 step 5-S5b.

Canonical home: ``aperag.domains.agent_runtime``. The top-level package
is retained as a re-export shim so pre-migration callers
(``aperag.evaluation_v2.worker``, ``aperag.views.agent_runtime`` shim,
any test that imports ``aperag.agent_runtime.runtime``) keep
resolving the same ``agent_runtime_manager`` + envelope Pydantic
classes without a rename sweep. Phase 6 cleanup removes the shim
after every caller has migrated.
"""

from aperag.domains.agent_runtime.runtime import agent_runtime_manager  # noqa: F401
from aperag.domains.agent_runtime.schemas import (  # noqa: F401
    AgentArtifactEnvelope,
    AgentTimelineEventEnvelope,
    AgentTurnEnvelope,
    AgentTurnSnapshot,
    CancelTurnResponse,
    CreateTurnRequest,
    CreateTurnResponse,
)

__all__ = [
    "AgentArtifactEnvelope",
    "AgentTimelineEventEnvelope",
    "AgentTurnEnvelope",
    "AgentTurnSnapshot",
    "CancelTurnResponse",
    "CreateTurnRequest",
    "CreateTurnResponse",
    "agent_runtime_manager",
]
