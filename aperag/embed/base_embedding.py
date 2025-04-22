#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
from abc import ABC, abstractmethod
from threading import Lock
from typing import Any
import logging

from langchain_core.embeddings import Embeddings

from aperag.embed.embedding_service import EmbeddingService

from aperag.db.models import(
    ModelServiceProvider,
)
from aperag.db.ops import (
    query_msp_dict,
)

from aperag.vectorstore.connector import VectorStoreConnectorAdaptor
from config.settings import (
    EMBEDDING_BACKEND,
    EMBEDDING_DIM,
    EMBEDDING_MAX_CHUNKS_IN_BATCH,
    EMBEDDING_MODEL,
    EMBEDDING_SERVICE_API_KEY,
    EMBEDDING_SERVICE_URL,
    EMBEDDING_DIMENSIONS,
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
    config["embedding_max_chunks_in_batch"] = config.get("embedding_max_chunks_in_batch", EMBEDDING_MAX_CHUNKS_IN_BATCH)
    return config

@synchronized
def get_embedding_model(
        embedding_backend: str = EMBEDDING_BACKEND,
        embedding_model: str = EMBEDDING_MODEL,
        embedding_dim: int = EMBEDDING_DIM,
        embedding_service_url: str = EMBEDDING_SERVICE_URL,
        embedding_service_api_key: str = EMBEDDING_SERVICE_API_KEY,
        embedding_max_chunks_in_batch: int = EMBEDDING_MAX_CHUNKS_IN_BATCH,
        **kwargs) -> tuple[Embeddings | None, int]:
    embedding_svc = EmbeddingService(embedding_backend, embedding_model, embedding_dim,
                                     embedding_service_url, embedding_service_api_key, embedding_max_chunks_in_batch)
    return embedding_svc, embedding_dim

def get_embedding_dimension(embedding_backend: str, model_type: str) -> int:
    rules = EMBEDDING_DIMENSIONS.get(embedding_backend, {})
    if embedding_backend == "openai":
        # Use service_model for openai, fallback to __default__ within openai rules
        return rules.get(model_type, rules.get("__default__", 1536))
    elif embedding_backend == "alibabacloud":
        return rules.get(model_type, rules.get("__default__", 1024))
    elif embedding_backend == "siliconflow":
        return rules.get(model_type, rules.get("__default__", 1024))

    # Use model_type for others, fallback to __default__ for that backend
    return rules.get(model_type, rules.get("__default__", 768))


async def get_collection_embedding_model(collection) -> tuple[Embeddings | None, int]:
    config = json.loads(collection.config)
    embedding_backend = config.get("embedding_model_service_provider", "")
    embedding_model_name = config.get("embedding_model_name", "")
    vector_size = get_embedding_dimension(embedding_backend, embedding_model_name)
    logging.info("get_collection_embedding_model %s %s %s", embedding_backend, embedding_model_name, vector_size)

    msp_dict = await query_msp_dict(collection.user)
    if embedding_backend in msp_dict:
        msp = msp_dict[embedding_backend]
        embedding_service_url = msp.base_url
        embedding_service_api_key = msp.api_key
        logging.info("get_collection_embedding_model %s %s", embedding_service_url, embedding_service_api_key)

        return get_embedding_model(
            embedding_backend=embedding_backend,
            embedding_model=embedding_model_name,
            embedding_dim=vector_size,
            embedding_service_url=embedding_service_url,
            embedding_service_api_key=embedding_service_api_key,
            embedding_max_chunks_in_batch=EMBEDDING_MAX_CHUNKS_IN_BATCH,
        )
    
    logging.warning("get_collection_embedding_model cannot find model service provider %s", embedding_backend)
    return None, 0


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
            raise ValueError("lacks embedding model or vector size")
        
        self.embedding, self.vector_size = embedding_model, vector_size
        self.client = vector_store_adaptor.connector.client

    @abstractmethod
    def load_data(self, **kwargs: Any):
        pass