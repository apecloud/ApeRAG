from abc import ABC

from deeprag.query.query import QueryWithEmbedding
from deeprag.vectorstore.connector import VectorStoreConnectorAdaptor


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
