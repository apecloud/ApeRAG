#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
from abc import ABC, abstractmethod
from threading import Lock
from typing import Any
from langchain_core.embeddings import Embeddings
from aperag.embed.embedding_service import EmbeddingService
from aperag.vectorstore.connector import VectorStoreConnectorAdaptor
from config.settings import (
    EMBEDDING_BACKEND,
    EMBEDDING_MODEL,
    EMBEDDING_DIM,
    EMBEDDING_SERVICE_API_KEY,
    EMBEDDING_SERVICE_URL,
)

mutex = Lock()

def synchronized(func):
    def wrapper(*args, **kwargs):
        with mutex:
            return func(*args, **kwargs)

    return wrapper


def loads_or_use_default_embedding_configs(config):
    """
    Load embedding configurations from the provided config dict,
    or use default values from settings if not present.

    Args:
        config (dict): The configuration dictionary to load from

    Returns:
        dict: The updated configuration dictionary with all embedding settings
    """
    config["embedding_backend"] = config.get("embedding_backend", EMBEDDING_BACKEND)
    config["embedding_model"] = config.get("embedding_model", EMBEDDING_MODEL)
    config["embedding_dim"] = config.get("embedding_dim", EMBEDDING_DIM)
    config["embedding_service_url"] = config.get("embedding_service_url", EMBEDDING_SERVICE_URL)
    config["embedding_service_api_key"] = config.get("embedding_service_api_key", EMBEDDING_SERVICE_API_KEY)
    return config

@synchronized
def get_embedding_model(
        embedding_backend: str = EMBEDDING_BACKEND,
        embedding_model: str = EMBEDDING_MODEL,
        embedding_dim: int = EMBEDDING_DIM,
        embedding_service_url: str = EMBEDDING_SERVICE_URL,
        embedding_service_api_key: str = EMBEDDING_SERVICE_API_KEY,
        **kwargs) -> tuple[Embeddings | None, int]:
    embedding_svc = EmbeddingService(embedding_backend, embedding_model, embedding_dim, embedding_service_url, embedding_service_api_key)
    return embedding_svc, embedding_dim


def get_collection_embedding_model(collection) -> tuple[Embeddings | None, int]:
    config = loads_or_use_default_embedding_configs(json.loads(collection.config))
    return get_embedding_model(
            embedding_backend=config["embedding_backend"],
            embedding_model=config["embedding_model"],
            embedding_dim=config["embedding_dim"],
            embedding_service_url=config["embedding_service_url"],
            embedding_service_api_key=config["embedding_service_api_key"]
        )

class DocumentBaseEmbedding(ABC):
    def __init__(
            self,
            vector_store_adaptor: VectorStoreConnectorAdaptor,
            embedding_model: Embeddings = None,
            vector_size: int = None,
            **kwargs: Any,
    ) -> None:
        self.connector = vector_store_adaptor.connector
        # Improved logic to handle optional embedding_model/vector_size
        if embedding_model is None or vector_size is None:
            loaded_embedding, loaded_vector_size = get_embedding_model()
            self.embedding = embedding_model if embedding_model is not None else loaded_embedding
            self.vector_size = vector_size if vector_size is not None else loaded_vector_size
        else:
            self.embedding, self.vector_size = embedding_model, vector_size

        # Ensure embedding is loaded if needed
        if self.embedding is None:
            raise ValueError("Embedding model could not be loaded or provided.")

        self.client = vector_store_adaptor.connector.client

    @abstractmethod
    def load_data(self, **kwargs: Any):
        pass