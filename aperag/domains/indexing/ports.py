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

"""Cross-domain contracts owned by the ``indexing`` domain.

Lesson 9a-quad (single-direction cross-domain bridge, consumer-owned
Protocol variant): the ``indexing`` domain needs a narrow view of a
Collection row when it wires per-document index workers, and its
consumers (``knowledge_base`` above all) need a way to trigger an
index spec update after a document state change without having to
import ``aperag.index.manager`` directly.

Two Protocols are declared here:

* ``CollectionIndexingView`` — the minimum Collection shape
  ``indexing`` reads to decide which index types to materialize. Owned
  by ``indexing`` because this domain is the **consumer** of Collection
  rows at the point of spec derivation. The real ``Collection`` SQLAlchemy
  model in ``aperag.domains.knowledge_base.db.models`` structurally
  satisfies it without either side importing the other.

* ``IndexingTrigger`` — the write-side contract ``knowledge_base``
  (and others) call to push a document into / out of the index pipeline.
  The provider domain (``indexing``) owns the Protocol because it is
  the only side that knows the operational semantics (index-type enum,
  idempotent upsert, deletion staging). ``knowledge_base`` structurally
  satisfies nothing — it just binds its dependency to the Protocol and
  calls the concrete manager in ``aperag.domains.indexing.manager`` via
  a small factory.

Phase 0 strict ban keeps this module free of ``aperag.db.models`` /
``aperag.service.*`` / ``aperag.schema.view_models`` imports; the
index-type parameter is typed as a string alias here and narrowed to
``DocumentIndexType`` at call sites once the enum physically moves to
``aperag.domains.indexing.db.models`` in Phase 3 step 5.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional, Protocol, runtime_checkable


@runtime_checkable
class CollectionIndexingView(Protocol):
    """Structural view of a Collection row readable by ``indexing``.

    Only the fields the domain touches are declared — every other
    column on the legacy model stays invisible so a careless caller
    cannot accidentally bind a larger surface. ``config`` is typed as
    ``Any`` because the legacy model stores it as a JSON string that
    ``aperag.schema.utils.parseCollectionConfig`` hydrates at the point
    of use; the indexing pipeline reads ``enable_vector`` /
    ``enable_fulltext`` / ``enable_knowledge_graph`` /
    ``enable_summary`` / ``enable_vision`` off that parsed object to
    decide which index types are live for the collection.
    """

    id: str
    user: str
    config: Any


@runtime_checkable
class IndexingTrigger(Protocol):
    """Write-side contract consumers call to push a document into / out
    of the per-document index pipeline.

    Signature pinned by the 符炫炜 Phase 3 design-lock adjustment
    (msg=226b2584): ``create_or_update_document_indexes`` accepts a
    ``CollectionIndexingView`` + the document id + an optional list of
    index types (``None`` means "derive the live set from
    ``collection.config``"). ``delete_document_indexes`` requires the
    caller to pass the exact set of index types to stage for deletion
    — no implicit "all" for the write path because a partial
    deletion is a legitimate state.
    """

    async def create_or_update_document_indexes(
        self,
        collection: CollectionIndexingView,
        document_id: str,
        index_types: Optional[Iterable[Any]] = None,
    ) -> None: ...

    async def delete_document_indexes(
        self,
        document_id: str,
        index_types: Iterable[Any],
    ) -> None: ...


__all__ = ["CollectionIndexingView", "IndexingTrigger"]
