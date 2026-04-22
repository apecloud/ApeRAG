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

"""Backend-neutral abstract interface for vector store connectors.

The goal of this module is to pin down the **smallest** surface that any
concrete backend (Qdrant today, pgvector / Milvus tomorrow) must implement
to be a drop-in replacement. Any type imported here must be backend-neutral
— in particular, **no** ``qdrant_client`` / ``psycopg`` / ``pymilvus``
imports are ever allowed in this file.

Return types use:

* ``QueryResult`` (``aperag.query.query``) for searches — already
  backend-neutral and existing callers depend on it.
* ``VectorPoint`` (defined here) for id-lookups — minimal DTO, just
  ``id`` + ``payload`` + optional ``vector``, enough to replace the
  Qdrant-specific ``Record`` at document-chunk preview sites.

``search(filter=...)`` accepts a ``VectorFilter`` DSL tree
(``aperag.vectorstore.filters``). Concrete connectors translate it into
their native filter representation. Passing raw backend filter objects
(e.g. ``qdrant_client.models.Filter``) is still tolerated for backwards
compatibility with the migration tooling, but new code must use the DSL.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.vector_stores.types import VectorStore

from aperag.query.query import QueryResult, QueryWithEmbedding
from aperag.vectorstore.filters import VectorFilter


@dataclass(frozen=True)
class VectorPoint:
    """Backend-neutral representation of a single stored point.

    * ``id`` is always a string because that's the lowest common
      denominator across backends (Qdrant allows int/uuid, pgvector uses
      uuid, Milvus allows int/varchar). We normalize on string at the
      connector boundary so callers don't have to branch.
    * ``payload`` is a plain dict. LlamaIndex conventions like
      ``_node_content`` live **inside** the dict; they are a detail of the
      Qdrant writer, not part of this contract.
    * ``vector`` is optional because most read-preview call sites don't
      need it, and shipping it across the wire costs real bytes.
    """

    id: str
    payload: Dict[str, Any]
    vector: Optional[List[float]] = None


class VectorStoreConnector(ABC):
    """Abstract contract every vector-DB backend implements.

    Lifecycle assumption: ``ctx`` carries everything the connector needs
    to locate / create its collection (URL, tenant id, vector size,
    distance, optimization knobs). Concrete subclasses read the keys
    they understand; unknown keys are ignored silently.
    """

    def __init__(self, ctx: Dict[str, Any], **kwargs: Any) -> None:
        self.ctx = ctx
        self.client = None
        self.embedding: BaseEmbedding = None
        self.store: VectorStore = None

    # -------------------------------------------------------------- search
    @abstractmethod
    def search(
        self,
        query: QueryWithEmbedding,
        *,
        filter: Optional[VectorFilter] = None,
        score_threshold: float = 0.1,
        **kwargs: Any,
    ) -> QueryResult:
        """Return the top-``query.top_k`` documents for the given embedding.

        ``filter`` accepts a ``VectorFilter`` DSL tree (preferred). For
        backward compatibility during the transition, implementations may
        also accept their native filter type, but new callers must use the
        DSL.
        """

    # -------------------------------------------------------------- writes
    @abstractmethod
    def delete(self, **delete_kwargs: Any) -> None:
        """Delete points by ``ids=[...]`` or other backend-specific key.

        Must enforce the tenant guard in multitenant-aware backends so a
        caller that accidentally passes another tenant's ids is a no-op
        rather than a data breach.
        """

    # --------------------------------------------------------------- reads
    @abstractmethod
    def retrieve(
        self,
        ids: Sequence[str],
        *,
        with_payload: bool = True,
        with_vectors: bool = False,
    ) -> List[VectorPoint]:
        """Fetch points by id.

        Implementations **must** apply the tenant guard in multitenant
        mode and silently drop any points whose tenant does not match;
        this matches ``delete()``'s behavior and prevents a curious
        caller from reading across tenants by guessing ids.
        """

    # ---------------------------------------------------------- collection
    @abstractmethod
    def create_collection(self, **create_kwargs: Any) -> None:
        """Ensure the physical collection / table exists and is correctly
        shaped for this connector's ``ctx``. Idempotent."""

    @abstractmethod
    def delete_collection(self, **delete_kwargs: Any) -> None:
        """Remove this tenant's data. See each backend's implementation
        for the exact semantics (per-tenant purge vs physical drop)."""
