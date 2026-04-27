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

"""Legacy submodule shim — ``aperag/agent_runtime/runtime.py``.

Re-exports ``agent_runtime_manager`` + the rest of the runtime
surface from the canonical ``aperag.domains.agent_runtime.runtime``
home so callers that still do ``from aperag.agent_runtime.runtime
import …`` keep resolving the same objects after Phase 5 step 5-S5b.
Phase 6 cleanup removes the shim after every caller has migrated.
"""

from aperag.domains.agent_runtime.runtime import (  # noqa: F401
    DEFAULT_AGENT_TURN_LEASE_RENEW_INTERVAL_SECONDS,
    DEFAULT_AGENT_TURN_LEASE_TTL_SECONDS,
    AgentRuntime,
    AgentRuntimeTaskManager,
    PydanticAIRuntime,
    ResolvedAgentRequest,
    TurnLeaseGuard,
    TurnLeaseLostError,
    agent_runtime_manager,
)
