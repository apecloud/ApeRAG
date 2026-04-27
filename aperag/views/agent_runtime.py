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

"""Legacy shim for ``aperag/views/agent_runtime.py`` after Phase 5
step 5-S5b.

Canonical home: ``aperag.domains.agent_runtime.api.routes``. This
module re-exports the ``router`` so any caller that still does
``from aperag.views.agent_runtime import router`` keeps resolving the
same router object. The ``runtime_manager`` alias is also re-exported
so pre-migration tests that do
``monkeypatch.setattr(aperag.views.agent_runtime, "runtime_manager",
…)`` keep working until 5-S8 test-patch sweep migrates them to the
canonical domain path.

Phase 6 cleanup removes the shim after every caller has migrated.
"""

from aperag.domains.agent_runtime.api.routes import router  # noqa: F401
from aperag.domains.agent_runtime.runtime import agent_runtime_manager as runtime_manager  # noqa: F401

__all__ = ["router", "runtime_manager"]
