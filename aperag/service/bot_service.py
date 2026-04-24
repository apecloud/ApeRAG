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

"""Legacy shim for ``bot_service`` after Phase 5 step 5-S4c.

The canonical home is
``aperag.domains.conversation.service.bot_service``; this module
re-exports the ``BotService`` class, the ``bot_service`` singleton,
and the ``set_quota_ops`` setter so pre-migration callers (``views/bots_v2.py``,
``views/auth.py`` ``UserManager.on_after_register`` hook,
``aperag.domains.identity.service.user_manager`` after Phase 4 merge,
``evaluation_v2`` if it gains a bot surface) keep working without a
rename sweep. Phase 6 cleanup removes the shim after every caller has
migrated.
"""

from aperag.domains.conversation.service.bot_service import (  # noqa: F401
    BotService,
    bot_service,
    set_quota_ops,
)

__all__ = [
    "BotService",
    "bot_service",
    "set_quota_ops",
]
