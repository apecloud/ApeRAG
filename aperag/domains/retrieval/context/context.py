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

"""Vector retrieval orchestration.

``ContextManager`` wraps the configured ``VectorStoreConnectorAdaptor`` and
takes care of:

* Building query embeddings (if not pre-supplied by the caller).
* Translating business-level filter intent (index_types, chat_id) into the
  backend-neutral ``VectorFilter`` DSL.
* Issuing the actual search via the new DTO-based ``connector.search``.
* Adapting each backend's ``SearchHit`` back to the project-wide
  ``DocumentWithScore`` shape that ``search_pipeline_service`` consumes.

The file imports nothing from ``qdrant_client`` / ``psycopg`` / …; that
backend-ignorance is the whole point of the DSL + DTO layer.
"""

from abc import ABC
from typing import Any, Dict, List, Optional

from aperag.platform.query.query import DocumentWithScore
from aperag.vectorstore.connector import VectorStoreConnectorAdaptor
from aperag.vectorstore.dto import QueryRequest, SearchHit, flatten_node_payload
from aperag.vectorstore.filters import Eq, In, IsEmpty, VectorFilter, all_of, any_of


class ContextManager(ABC):
    def __init__(self, collection_name, embedding_model, vectordb_type, vectordb_ctx):
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        # Retained only for diagnostics / callers that still inspect the type.
        # Code paths in this class are backend-agnostic.
        self.vectordb_type = vectordb_type
        self.adaptor = VectorStoreConnectorAdaptor(vectordb_type, vectordb_ctx)

    def query(
        self,
        query,
        # ``score_threshold`` is on the normalized [0, 1] similarity scale
        # per task #61 P0-B (``aperag.vectorstore.base.normalize_score``).
        # 0.5 is a permissive cosine-grade default — non-cosine collections
        # may want to override.
        score_threshold: float = 0.5,
        topk: int = 3,
        vector: Optional[List[float]] = None,
        index_types: Optional[List[str]] = None,
        chat_id: Optional[str] = None,
    ) -> List[DocumentWithScore]:
        """Query vectors with optional filtering by index types and chat_id.

        Args:
            query: Query string.
            score_threshold: Similarity threshold (higher = stricter).
            topk: Number of results to return.
            vector: Pre-computed query vector (optional).
            index_types: List of index types to include
                (e.g. ``["vector", "vision", "summary"]``). If None, no index
                filter is applied.
            chat_id: Chat ID to include chat-scoped documents (optional).

        Returns:
            List of DocumentWithScore objects (project-wide neutral result
            type; built from each backend's SearchHit).
        """
        if vector is None:
            vector = self.embedding_model.embed_query(query)

        filter_condition = self._create_combined_filter(index_types, chat_id)

        request = QueryRequest(
            embedding=list(vector),
            top_k=int(topk),
            flt=filter_condition,
            score_threshold=float(score_threshold),
            with_vectors=True,
            hints=_default_search_hints(),
        )

        hits: List[SearchHit] = self.adaptor.connector.search(request)
        return [_hit_to_document(hit) for hit in hits]

    # ------------------------------------------------------------------ filter
    def _create_index_types_filter(self, index_types: Optional[List[str]]) -> Optional[VectorFilter]:
        """Include only points tagged with one of ``index_types``, OR points
        from before we started tagging (so old data stays searchable).

        Historical note: the ``indexer`` payload field was added in a later
        migration; older points don't carry it. We therefore OR the membership
        test with an ``IsEmpty`` guard — otherwise introducing an index_type
        filter silently hides pre-migration data.
        """
        if not index_types:
            return None
        return any_of(
            In(key="indexer", values=tuple(index_types)),
            IsEmpty(key="indexer"),
        )

    def _create_combined_filter(
        self,
        index_types: Optional[List[str]] = None,
        chat_id: Optional[str] = None,
    ) -> Optional[VectorFilter]:
        """Combine the index-type and chat-id filters into a single tree.

        Returns ``None`` when no constraints apply so the connector can
        short-circuit. The shape is always an AND of whichever of
        ``(index_types_filter, chat_id_eq)`` are present, matching the
        semantics of the pre-DSL implementation.
        """
        parts = []
        idx = self._create_index_types_filter(index_types)
        if idx is not None:
            parts.append(idx)
        if chat_id:
            parts.append(Eq(key="chat_id", value=chat_id))
        return all_of(*parts)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _default_search_hints() -> Dict[str, Any]:
    """Per-backend hints that improve retrieval quality at typical settings.

    Backends read only the keys they understand:
    * Qdrant: ``hnsw_ef``, ``exact``, ``consistency``.
    * pgvector: ``ef_search``.

    Having both present is safe — unknown keys are ignored.
    """
    return {
        # Qdrant
        "hnsw_ef": 128,
        "exact": False,
        "consistency": "majority",
        # pgvector
        "ef_search": 80,
    }


def _hit_to_document(hit: SearchHit) -> DocumentWithScore:
    """Flatten a backend-neutral ``SearchHit`` into the project's
    ``DocumentWithScore`` shape.

    The flattening helper in ``dto`` handles both the *new* payload
    shape (``{text, metadata}``) and the *old* LlamaIndex one
    (``{_node_content: json_str}``), so readers tolerate data written
    before / after the refactor without any per-caller branching.
    """
    flat = flatten_node_payload(hit.payload or {})
    return DocumentWithScore(
        text=flat.get("text"),
        metadata=flat.get("metadata"),
        score=hit.score,
    )
