from typing import Any, Dict


class VectorStoreConnectorAdaptor:
    def __init__(self, vector_store_type, ctx: Dict[str, Any], **kwargs: Any) -> None:
        self.ctx = ctx
        self.vector_store_type = vector_store_type

        # only import the connector class when it is needed
        match vector_store_type:
            case "qdrant":
                from aperag.vectorstore.qdrant_connector import QdrantVectorStoreConnector

                self.connector = QdrantVectorStoreConnector(ctx, **kwargs)
            case _:
                raise ValueError("unsupported vector store type:", vector_store_type)
