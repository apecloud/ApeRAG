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

"""Compatibility shim for the governance quota service.

The canonical implementation lives in
``aperag.domains.governance.service.quota_service``. This module stays as
the legacy Protocol/DI import path until the remaining standalone-infra
seams are retired.
"""

from aperag.domains.governance.service.quota_service import QuotaService, quota_service

__all__ = ["QuotaService", "quota_service"]
