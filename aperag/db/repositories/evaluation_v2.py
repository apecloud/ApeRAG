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

"""Legacy shim — the canonical home is now
``aperag.domains.evaluation.db.repositories.evaluation_v2`` after
Phase 5 step 5-S6.

``aperag.db.ops`` pulls in the repository mixin from here so the
existing call sites keep working without a rename sweep. Phase 6
cleanup removes the shim.
"""

from aperag.domains.evaluation.db.repositories.evaluation_v2 import (  # noqa: F401
    AsyncEvaluationV2RepositoryMixin,
)

__all__ = ["AsyncEvaluationV2RepositoryMixin"]
