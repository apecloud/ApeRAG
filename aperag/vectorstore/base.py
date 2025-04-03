from abc import ABC, abstractmethod
from typing import Any, Dict

from llama_index.embeddings.base import BaseEmbedding
from llama_index.vector_stores.types import VectorStore

from deeprag.query.query import QueryResult, QueryWithEmbedding


class VectorStoreConnector(ABC):
    def __init__(self, ctx: Dict[str, Any], **kwargs: Any) -> None:
        self.ctx = ctx
        self.client = None
        self.embedding: BaseEmbedding = None
        self.store: VectorStore = None

    def search(self, query: QueryWithEmbedding, **kwargs) -> QueryResult:
        pass

    @abstractmethod
    def delete(self, **delete_kwargs: Any):
        pass

    @abstractmethod
    def create_collection(self, **create_kwargs: Any):
        pass

    @abstractmethod
    def delete_collection(self, **delete_kwargs: Any):
        pass
