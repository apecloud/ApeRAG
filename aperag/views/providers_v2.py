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

"""Legacy router-import shim — the provider-CRUD handlers moved to
``aperag.domains.model_platform.api.providers_v2_routes`` (and are
composed into the model_platform aggregate
``aperag.domains.model_platform.api.routes.router``) in Phase 4 Step
4-S6.
"""

from __future__ import annotations

from aperag.domains.model_platform.api.providers_v2_routes import router  # noqa: F401
