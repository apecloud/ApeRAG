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

"""Constants for the simplified evaluation module.

Centralising these strings keeps default-resolution behaviour testable and
prevents magic strings from scattering across service / repository / view
layers. See ``resolve_default_evaluation_bot`` in
``aperag.db.repositories.evaluation_v2``.
"""

# Title that identifies the system default Agent bot when a user does not
# pass an explicit ``bot_id`` on evaluation-run creation. The resolver prefers
# the user's active bot with this exact title; otherwise falls back to the
# oldest active bot (``gmt_created ASC``); if none exists the service raises
# a user-actionable ``ValidationException``.
DEFAULT_AGENT_BOT_TITLE = "Default Agent Bot"
