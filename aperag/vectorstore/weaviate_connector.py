from typing import Any, Dict

from llama_index.core.vector_stores.utils import DEFAULT_TEXT_KEY
from llama_index.vector_stores.weaviate import WeaviateVectorStore
from weaviate import Client

from aperag.vectorstore.base import VectorStoreConnector


class WeaviateVectorStoreConnector(VectorStoreConnector):
    def __init__(self, ctx: Dict[str, Any], **kwargs: Any) -> None:
        super().__init__(ctx, **kwargs)

        self.collection_name = ctx.get("collection", "collection")

        self.url = ctx.get("url", "http://localhost:8080")
        self.auth_client_secret = ctx.get("auth_client_secret", None)
        self.timeout_config = ctx.get("timeout_config", (5, 15))
        self.additional_headers = ctx.get("additional_headers", None)

        self.class_prefix = ctx.get("class_prefix", None)
        self.index_name = ctx.get("index_name", None)
        self.text_key = ctx.get("text_key", DEFAULT_TEXT_KEY)

        self.client = Client(
            url=self.url,
            auth_client_secret=self.auth_client_secret,
            timeout_config=self.timeout_config,
            additional_headers=self.additional_headers,
            **kwargs,
        )

        self.store = WeaviateVectorStore(
            weaviate_client=self.client,
            class_prefix=self.class_prefix,
            index_name=self.index_name,
            text_key=self.text_key,
        )
