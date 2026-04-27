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

"""Legacy shim for ``chat_collection_service`` after Phase 5 step 5-S4f.

The canonical home is
``aperag.domains.conversation.service.chat_collection_service``; this
module re-exports ``ChatCollectionService`` and the
``chat_collection_service`` singleton so pre-migration callers
(``views/chat.py``, the identity-domain ``_ChatInitOpsAdapter`` in
``aperag/app.py``, the ``aperag.service.chat_document_service``
shim's DI wire-up) keep resolving the same class objects. Phase 6
cleanup removes the shim after every caller has migrated.
"""

from aperag.domains.conversation.service.chat_collection_service import (  # noqa: F401
    ChatCollectionService,
    chat_collection_service,
)

__all__ = [
    "ChatCollectionService",
    "chat_collection_service",
]
