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

"""Legacy router-import shim — api_key + audit handlers merged into
governance domain router in Phase 4 Step 4-S5. See
``aperag/views/audit.py`` shim for the full context.
"""

from __future__ import annotations

from aperag.domains.governance.api.routes import router  # noqa: F401
