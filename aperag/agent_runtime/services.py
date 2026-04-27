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

"""Legacy submodule shim — ``aperag/agent_runtime/services.py``.

Re-exports the runtime-services surface from the canonical
``aperag.domains.agent_runtime.services`` home so callers that still
do ``from aperag.agent_runtime.services import …`` keep resolving the
same objects after Phase 5 step 5-S5b.
"""

from aperag.domains.agent_runtime.services import (  # noqa: F401
    ArtifactService,
    EventService,
    HistoryWriter,
    TurnService,
    _parse_bot_config,
)
