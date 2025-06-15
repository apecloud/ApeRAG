#!/usr/bin/env python3
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

# -*- coding: utf-8 -*-
import logging
from threading import Lock

from langchain_core.embeddings import Embeddings

from aperag.config import settings
from aperag.db.ops import db_ops
from aperag.llm.embed.embedding_service import EmbeddingService
from aperag.llm.llm_error_types import (
    EmbeddingError,
    InvalidConfigurationError,
    ModelNotFoundError,
    ProviderNotFoundError,
)
from aperag.schema.utils import parseCollectionConfig

logger = logging.getLogger(__name__)

mutex = Lock()


def synchronized(func):
    def wrapper(*args, **kwargs):
        with mutex:
            return func(*args, **kwargs)

    return wrapper


_dimension_cache: dict[tuple[str, str], int] = {}


def _get_embedding_dimension(embedding_svc: EmbeddingService, embedding_provider: str, embedding_model) -> int:
    """
    Get embedding dimension by probing the embedding service.

    Args:
        embedding_svc: The embedding service instance
        embedding_provider: Provider name for caching
        embedding_model: Model name for caching

    Returns:
        int: The embedding dimension

    Raises:
        EmbeddingError: If dimension probing fails
    """
    cache_key = (embedding_provider, embedding_model)
    if cache_key in _dimension_cache:
        return _dimension_cache[cache_key]

    try:
        vec = embedding_svc.embed_query("dimension_probe")
        if not vec:
            raise EmbeddingError(
                "Failed to obtain embedding vector while probing dimension",
                {"provider": embedding_provider, "model": embedding_model},
            )
        if isinstance(vec[0], (list, tuple)):
            vec = vec[0]
        dim = len(vec)
        _dimension_cache[cache_key] = dim
        logger.info(f"Cached embedding dimension for {embedding_provider}/{embedding_model}: {dim}")
        return dim
    except Exception as e:
        logger.error(f"Failed to probe embedding dimension for {embedding_provider}/{embedding_model}: {str(e)}")
        raise EmbeddingError(
            f"Failed to probe embedding dimension: {str(e)}", {"provider": embedding_provider, "model": embedding_model}
        ) from e


@synchronized
def _get_embedding_model(
    embedding_provider: str,
    embedding_model: str,
    embedding_service_url: str,
    embedding_service_api_key: str,
    embedding_max_chunks_in_batch: int = settings.embedding_max_chunks_in_batch,
    **kwargs,
) -> tuple[Embeddings | None, int]:
    """
    Create and configure an embedding model instance.

    Args:
        embedding_provider: The embedding provider name
        embedding_model: The embedding model name
        embedding_service_url: The API base URL
        embedding_service_api_key: The API key
        embedding_max_chunks_in_batch: Maximum chunks per batch

    Returns:
        tuple: (EmbeddingService instance, embedding dimension)

    Raises:
        EmbeddingError: If model creation or dimension probing fails
    """
    try:
        embedding_svc = EmbeddingService(
            embedding_provider,
            embedding_model,
            embedding_service_url,
            embedding_service_api_key,
            embedding_max_chunks_in_batch,
        )
        embedding_dim = _get_embedding_dimension(embedding_svc, embedding_provider, embedding_model)
        return embedding_svc, embedding_dim
    except EmbeddingError:
        # Re-raise embedding errors
        raise
    except Exception as e:
        logger.error(f"Failed to create embedding model {embedding_provider}/{embedding_model}: {str(e)}")
        raise EmbeddingError(
            f"Failed to create embedding model: {str(e)}",
            {"provider": embedding_provider, "model": embedding_model, "api_base": embedding_service_url},
        ) from e


def get_collection_embedding_service_sync(collection) -> tuple[Embeddings, int]:
    """
    Get embedding service for a collection synchronously.

    Args:
        collection: The collection object with configuration

    Returns:
        tuple: (Embeddings instance, embedding dimension)

    Raises:
        ProviderNotFoundError: If the embedding provider is not found
        ModelNotFoundError: If the embedding model is not found
        InvalidConfigurationError: If configuration is invalid
        EmbeddingError: If embedding service creation fails
    """
    try:
        config = parseCollectionConfig(collection.config)
    except Exception as e:
        logger.error(f"Failed to parse collection config: {str(e)}")
        raise InvalidConfigurationError(
            "collection.config", collection.config, f"Invalid collection configuration: {str(e)}"
        ) from e

    embedding_msp = config.embedding.model_service_provider
    embedding_model_name = config.embedding.model
    custom_llm_provider = config.embedding.custom_llm_provider

    logger.info("get_collection_embedding_model_sync %s %s", embedding_msp, embedding_model_name)

    # Validate configuration fields
    if not embedding_msp:
        raise InvalidConfigurationError(
            "embedding.model_service_provider", embedding_msp, "Model service provider cannot be empty"
        )

    if not embedding_model_name:
        raise InvalidConfigurationError("embedding.model", embedding_model_name, "Model name cannot be empty")

    if not custom_llm_provider:
        raise InvalidConfigurationError(
            "embedding.custom_llm_provider", custom_llm_provider, "Custom LLM provider cannot be empty"
        )

    try:
        msp_dict = db_ops.query_msp_dict(collection.user)
    except Exception as e:
        logger.error(f"Failed to query model service providers: {str(e)}")
        raise EmbeddingError(f"Failed to query model service providers: {str(e)}") from e

    if embedding_msp not in msp_dict:
        available_providers = list(msp_dict.keys())
        logger.warning(
            "get_collection_embedding_model_sync cannot find model service provider %s. Available: %s",
            embedding_msp,
            available_providers,
        )
        raise ProviderNotFoundError(embedding_msp, "Embedding")

    msp = msp_dict[embedding_msp]

    if not msp.api_key:
        raise InvalidConfigurationError("api_key", None, f"API key not configured for provider '{embedding_msp}'")

    embedding_service_api_key = msp.api_key

    try:
        llm_provider = db_ops.query_llm_provider_by_name(embedding_msp)
        if not llm_provider:
            raise ModelNotFoundError(embedding_model_name, embedding_msp, "Embedding")
        embedding_service_url = llm_provider.base_url
    except Exception as e:
        logger.error(f"Failed to query LLM provider '{embedding_msp}': {str(e)}")
        raise ProviderNotFoundError(embedding_msp, "Embedding") from e

    if not embedding_service_url:
        raise InvalidConfigurationError(
            "base_url", embedding_service_url, f"Base URL not configured for provider '{embedding_msp}'"
        )

    logger.info("get_collection_embedding_model %s %s", embedding_service_url, embedding_service_api_key)

    try:
        return _get_embedding_model(
            embedding_provider=custom_llm_provider,
            embedding_model=embedding_model_name,
            embedding_service_url=embedding_service_url,
            embedding_service_api_key=embedding_service_api_key,
        )
    except EmbeddingError:
        # Re-raise embedding errors
        raise
    except Exception as e:
        logger.error(f"Failed to get embedding model for collection: {str(e)}")
        raise EmbeddingError(
            f"Failed to get embedding model for collection: {str(e)}",
            {
                "collection_id": getattr(collection, "id", "unknown"),
                "provider": embedding_msp,
                "model": embedding_model_name,
            },
        ) from e
