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

"""Legacy shim — canonical home is
``aperag.domains.conversation.service.chat_document_service``.

Re-exports the ``ChatDocumentService`` class and
``chat_document_service`` singleton so pre-migration callers keep
resolving the same class objects. Phase 6 cleanup drops this shim
after every caller has migrated.
"""

from aperag.domains.conversation.service.chat_document_service import (  # noqa: F401
    ChatDocumentService,
    chat_document_service,
)

__all__ = [
    "ChatDocumentService",
    "chat_document_service",
]
