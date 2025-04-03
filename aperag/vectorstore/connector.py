from typing import Any, Dict

from llama_index.vector_stores.registry import (
    VECTOR_STORE_TYPE_TO_VECTOR_STORE_CLASS,
    VectorStoreType,
)


class VectorStoreConnectorAdaptor:
    def __init__(self, vector_store_type, ctx: Dict[str, Any], **kwargs: Any) -> None:
        self.ctx = ctx
        self.vector_store_type = vector_store_type

        vector_store_class = VECTOR_STORE_TYPE_TO_VECTOR_STORE_CLASS[vector_store_type]
        if vector_store_class is None:
            raise ValueError("unsupported vector store type:", vector_store_type)

        # only import the connector class when it is needed
        match vector_store_type:
            case VectorStoreType.CHROMA:
                from aperag.vectorstore.chroma_connector import ChromaVectorStoreConnector

                self.connector = ChromaVectorStoreConnector(ctx, **kwargs)
            case VectorStoreType.WEAVIATE:
                from aperag.vectorstore.weaviate_connector import WeaviateVectorStoreConnector

                self.connector = WeaviateVectorStoreConnector(ctx, **kwargs)
            case VectorStoreType.QDRANT:
                from aperag.vectorstore.qdrant_connector import QdrantVectorStoreConnector

                self.connector = QdrantVectorStoreConnector(ctx, **kwargs)
            case VectorStoreType.MILVUS:
                from aperag.vectorstore.milvus_connector import MilvusVectorStoreConnector

                self.connector = MilvusVectorStoreConnector(ctx, **kwargs)
            case _:
                raise ValueError("unsupported vector store type:", vector_store_type)
