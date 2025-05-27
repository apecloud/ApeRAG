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

from abc import ABC

from aperag.query.query import QueryWithEmbedding
from aperag.vectorstore.connector import VectorStoreConnectorAdaptor


class ContextManager(ABC):
    def __init__(self, collection_name, embedding_model, vectordb_type, vectordb_ctx):
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.adaptor = VectorStoreConnectorAdaptor(vectordb_type, vectordb_ctx)

    def query(self, query, score_threshold=0.5, topk=3, vector=None):
        if vector is None:
            vector = self.embedding_model.embed_query(query)
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
        )
        return results.results
