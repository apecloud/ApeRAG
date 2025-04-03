from typing import Any, Dict

import chromadb
from chromadb.config import Settings
from llama_index.embeddings import google
from llama_index.vector_stores.chroma import ChromaVectorStore

from deeprag.vectorstore.base import VectorStoreConnector


class ChromaVectorStoreConnector(VectorStoreConnector):
    def __init__(self, ctx: Dict[str, Any], **kwargs: Any) -> None:
        super().__init__(ctx, **kwargs)
        self.ctx = ctx
        self.collection_name = ctx.get("collection_name", "test")
        self.chroma_db_impl = ctx.get("chroma_db_impl", "duckdb+parquet")
        self.persist_directory = ctx.get("persist_directory", ".chromadb/")
        self.chroma_api_impl = ctx.get("chroma_api_impl", "rest")
        self.chroma_server_host = ctx.get("chroma_server_host", None)
        self.chroma_server_http_port = ctx.get("chroma_server_http_port", None)

        # chroma in memory
        if self.chroma_server_host is None:
            self.client = chromadb.Client(
                Settings(
                    chroma_db_impl=self.chroma_db_impl,
                    persist_directory=self.persist_directory,
                    **kwargs,
                )
            )

        # chroma in client
        else:
            self.client = chromadb.Client(
                Settings(
                    chroma_db_impl=self.chroma_db_impl,
                    chroma_api_impl=self.chroma_api_impl,
                    persist_directory=self.persist_directory,
                    chroma_server_host=self.chroma_server_host,
                    chroma_server_http_port=self.chroma_server_http_port,
                    **kwargs,
                )
            )

        self.embedding = ctx.get("embedding", google.GoogleUnivSentEncoderEmbedding)
        self.store = ChromaVectorStore(
            client=self.client, chroma_collection=self.collection_name
        )
