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
from abc import ABC, abstractmethod
from threading import Lock
from typing import Any

from langchain_core.embeddings import Embeddings
from sqlalchemy import select

from aperag.config import get_sync_session, settings
from aperag.db.models import ModelServiceProvider, ModelServiceProviderStatus
from aperag.db.ops import db_ops
from aperag.embed.embedding_service import EmbeddingService
from aperag.schema.utils import parseCollectionConfig
from aperag.vectorstore.connector import VectorStoreConnectorAdaptor

logger = logging.getLogger(__name__)

mutex = Lock()


def synchronized(func):
    def wrapper(*args, **kwargs):
        with mutex:
            return func(*args, **kwargs)

    return wrapper


_dimension_cache: dict[tuple[str, str], int] = {}


def _get_embedding_dimension(embedding_svc: EmbeddingService, embedding_provider: str, embedding_model) -> int:
    cache_key = (embedding_provider, embedding_model)
    if cache_key in _dimension_cache:
        return _dimension_cache[cache_key]
    vec = embedding_svc.embed_query("dimension_probe")
    if not vec:
        raise RuntimeError("Failed to obtain embedding vector while probing dimension.")
    if isinstance(vec[0], (list, tuple)):
        vec = vec[0]
    dim = len(vec)
    _dimension_cache[cache_key] = dim
    return dim


@synchronized
def get_embedding_model(
    embedding_provider: str,
    embedding_model: str,
    embedding_service_url: str,
    embedding_service_api_key: str,
    embedding_max_chunks_in_batch: int = settings.embedding_max_chunks_in_batch,
    **kwargs,
) -> tuple[Embeddings | None, int]:
    embedding_svc = EmbeddingService(
        embedding_provider,
        embedding_model,
        embedding_service_url,
        embedding_service_api_key,
        embedding_max_chunks_in_batch,
    )
    embedding_dim = _get_embedding_dimension(embedding_svc, embedding_provider, embedding_model)
    return embedding_svc, embedding_dim


async def get_collection_embedding_service(collection) -> tuple[Embeddings | None, int]:
    config = parseCollectionConfig(collection.config)
    embedding_msp = config.embedding.model_service_provider
    embedding_model_name = config.embedding.model
    custom_llm_provider = config.embedding.custom_llm_provider
    logging.info("get_collection_embedding_model %s %s", embedding_msp, embedding_model_name)

    msp_dict = await db_ops.query_msp_dict(collection.user)
    if embedding_msp in msp_dict:
        msp = msp_dict[embedding_msp]
        embedding_service_url = msp.base_url
        embedding_service_api_key = msp.api_key
        logging.info("get_collection_embedding_model %s %s", embedding_service_url, embedding_service_api_key)

        return get_embedding_model(
            embedding_provider=custom_llm_provider,
            embedding_model=embedding_model_name,
            embedding_service_url=embedding_service_url,
            embedding_service_api_key=embedding_service_api_key,
        )

    logging.warning("get_collection_embedding_model cannot find model service provider %s", embedding_msp)
    return None, 0


def get_collection_embedding_service_sync(collection) -> tuple[object, int]:
    """Synchronous version of get_collection_embedding_service for Celery tasks"""
    config = parseCollectionConfig(collection.config)
    embedding_msp = config.embedding.model_service_provider
    embedding_model_name = config.embedding.model
    custom_llm_provider = config.embedding.custom_llm_provider
    logger.info("get_collection_embedding_model_sync %s %s", embedding_msp, embedding_model_name)

    # Query model service providers using sync session
    for session in get_sync_session():
        stmt = select(ModelServiceProvider).where(
            ModelServiceProvider.user == collection.user,
            ModelServiceProvider.status == ModelServiceProviderStatus.ACTIVE,
        )
        result = session.execute(stmt)
        msps = result.scalars().all()

        msp_dict = {msp.name: msp for msp in msps}

    if embedding_msp in msp_dict:
        msp = msp_dict[embedding_msp]
        embedding_service_url = msp.base_url
        embedding_service_api_key = msp.api_key
        logger.info("get_collection_embedding_model_sync %s %s", embedding_service_url, embedding_service_api_key)

        return get_embedding_model(
            embedding_provider=custom_llm_provider,
            embedding_model=embedding_model_name,
            embedding_service_url=embedding_service_url,
            embedding_service_api_key=embedding_service_api_key,
        )

    logger.warning("get_collection_embedding_model_sync cannot find model service provider %s", embedding_msp)
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
