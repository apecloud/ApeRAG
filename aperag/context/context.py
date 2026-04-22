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
* Issuing the actual search.

Before this module migrated to the DSL, filter construction branched on
``self.vectordb_type == "qdrant"`` and directly imported
``qdrant_client.models``. That made adding a second backend a per-site
refactor. The DSL path keeps this file backend-agnostic — the concrete
connector is the single place that knows Qdrant.
"""

from abc import ABC
from typing import List, Optional

from aperag.query.query import QueryWithEmbedding
from aperag.vectorstore.connector import VectorStoreConnectorAdaptor
from aperag.vectorstore.filters import Eq, In, IsEmpty, VectorFilter, all_of, any_of


class ContextManager(ABC):
    def __init__(self, collection_name, embedding_model, vectordb_type, vectordb_ctx):
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        # Retained only for diagnostics / callers that still inspect the type.
        # Code paths in this class are backend-agnostic.
        self.vectordb_type = vectordb_type
        self.adaptor = VectorStoreConnectorAdaptor(vectordb_type, vectordb_ctx)

    def query(self, query, score_threshold=0.5, topk=3, vector=None, index_types=None, chat_id=None):
        """Query vectors with optional filtering by index types and chat_id.

        Args:
            query: Query string.
            score_threshold: Similarity threshold.
            topk: Number of results to return.
            vector: Pre-computed query vector (optional).
            index_types: List of index types to include
                (e.g. ``["vector", "vision", "summary"]``). If None, no index
                filter is applied.
            chat_id: Chat ID to include chat-scoped documents (optional).

        Returns:
            List of DocumentWithScore objects.
        """
        if vector is None:
            vector = self.embedding_model.embed_query(query)

        # Build backend-neutral filter; concrete connector translates.
        filter_condition = self._create_combined_filter(index_types, chat_id)

        query_embedding = QueryWithEmbedding(query=query, top_k=topk, embedding=vector)
        results = self.adaptor.connector.search(
            query_embedding,
            collection_name=self.collection_name,
            query_vector=query_embedding.embedding,
            with_vectors=True,
            limit=query_embedding.top_k,
            consistency="majority",
            search_params={"hnsw_ef": 128, "exact": False},
            score_threshold=score_threshold,
            filter=filter_condition,
        )
        return results.results

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
