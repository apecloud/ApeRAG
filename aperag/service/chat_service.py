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

"""Legacy shim for ``chat_service`` after Phase 5 step 5-S4b.

The canonical home is
``aperag.domains.conversation.service.chat_service``; this module
re-exports ``ChatService`` and the ``chat_service_global`` singleton
so pre-migration callers (``views/chat.py`` / ``views/bots_v2.py`` /
``agent_runtime`` lazy imports / ``evaluation_v2.worker``) keep
resolving the same class objects without a rename sweep. Phase 6
cleanup removes the shim after every caller has migrated.
"""

from aperag.domains.conversation.service.chat_service import (  # noqa: F401
    ChatService,
    chat_service_global,
)

__all__ = [
    "ChatService",
    "chat_service_global",
]
