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

"""Cross-domain contracts owned by the ``conversation`` domain.

Lesson 9a-quad (consumer-owned Protocol): the conversation domain
reads a handful of Collection fields while resolving "what knowledge
base does this chat attach to?" — title / id / description for the
UI response shape, and the raw ``config`` blob for downstream pipeline
wiring. Declaring ``KnowledgeBaseCollectionView`` here means the
Phase 3 DB split can physically relocate
``aperag.db.models.Collection`` into
``aperag.domains.knowledge_base.db.models`` without the legacy
``chat_collection_service`` having to change its type binding — the
real Collection row structurally satisfies this Protocol regardless of
where it lives.

Phase 5 adds the rest of the conversation Protocols (Chat / Message /
Bot views); this module is the first drop so Phase 3 step 7 can do the
Protocol bridge switchover for ``chat_collection_service`` cleanly.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class KnowledgeBaseCollectionView(Protocol):
    """Structural view of a Collection row readable by ``conversation``.

    The conversation domain consumes Collection rows at two points:

    * Chat-create / chat-update: the user picks a Collection id; the
      service dereferences it to surface ``title`` / ``description``
      in the chat payload and to gate access control on ``user``.
    * Pipeline wiring: the stored ``config`` JSON is parsed into the
      retrieval / flow engine's typed config so the assistant can run
      the right recall pipeline.

    ``type`` is included because conversation selects different bot
    wiring for ``document`` / ``graph`` collection types. ``config``
    is ``Any`` because the legacy model stores it as a JSON string
    hydrated by ``aperag.schema.utils.parseCollectionConfig`` at the
    point of use.
    """

    id: str
    user: str
    title: str
    description: Any  # Nullable Text column.
    type: Any  # EnumColumn(CollectionType), kept Any to avoid coupling.
    config: Any


__all__ = ["KnowledgeBaseCollectionView"]
