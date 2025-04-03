from typing import Any, Dict

from llama_index.embeddings import google
from llama_index.vector_stores import opensearch
from llama_index.vector_stores.opensearch import OpensearchVectorStore

from deeprag.vectorstore.base import VectorStoreConnector


class OpensearchVectorStoreConnector(VectorStoreConnector):
    def __init__(self, ctx: Dict[str, Any], auth=None, **kwargs: Any) -> None:
        super().__init__(ctx, **kwargs)
        self.ctx = ctx
        self.index_name = ctx.get("index", "index")

        self.url = ctx.get("url", "https://localhost")
        self.port = ctx.get("port", 9200)
        self.https = ctx.get("https", True)
        self.timeout = ctx.get("timeout", 30)
        self.vector_size = ctx.get("vector_size", 300)

        self.client = opensearch.OpensearchVectorClient(
            endpoint=f"{self.url}:{self.port}",
            index=self.index_name,
            dim=self.vector_size,
            auth=auth,
        )

        self.embedding = ctx.get("embedding", google.GoogleUnivSentEncoderEmbedding)
        self.store = OpensearchVectorStore(client=self.client)
