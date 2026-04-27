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

"""Legacy submodule shim — ``aperag/evaluation_v2/schemas.py``.

Re-exports the full schemas surface from the canonical
``aperag.domains.evaluation.schemas`` home so ``from
aperag.evaluation_v2.schemas import …`` callers keep working after
Phase 5 step 5-S6.
"""

from aperag.domains.evaluation.schemas import *  # noqa: F401,F403
