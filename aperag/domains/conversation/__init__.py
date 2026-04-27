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

"""Canonical ``conversation`` domain (bootstrap only).

Phase 3 only lays down the ``ports.py`` Protocol surface that
``conversation`` will expose in Phase 5 once the full Chat / Message /
Bot migration lands. The domain body is intentionally empty for now —
declaring the Protocol ahead of time lets legacy ``chat_*`` services
lift their Collection typing onto a narrow Protocol today rather than
against ``aperag.db.models.Collection``, so the Phase 3 DB split can
physically relocate the model without breaking the conversation
surface.
"""
