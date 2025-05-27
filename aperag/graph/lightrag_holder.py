import asyncio
import logging
from datetime import datetime
from typing import Any, AsyncIterator, Awaitable, Callable, Dict, List, Optional, Tuple

import numpy
from lightrag import LightRAG, QueryParam
from lightrag.base import DocStatus
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.utils import EmbeddingFunc

from aperag.db.models import Collection
from aperag.db.ops import (
    query_msp_dict,
)
from aperag.embed.base_embedding import get_collection_embedding_service
from aperag.schema.utils import parseCollectionConfig
from aperag.utils.utils import generate_lightrag_namespace_prefix

logger = logging.getLogger(__name__)


# Configuration constants
class LightRAGConfig:
    """Centralized configuration for LightRAG"""

    # Storage Configuration
    WORKING_DIR = "./documents"
    ENABLE_LLM_CACHE = False

    # Performance Configuration
    MAX_PARALLEL_INSERT = 2
    LLM_MODEL_MAX_ASYNC = 20
    ENTITY_EXTRACT_MAX_GLEANING = 0
    EMBEDDING_MAX_TOKEN_SIZE = 8192

    # Logging Configuration
    DEFAULT_LANGUAGE = "Simplified Chinese"


class LightRAGError(Exception):
    """Base exception for LightRAG operations"""

    pass


class LightRAGInitializationError(LightRAGError):
    """Exception raised during LightRAG initialization"""

    pass


class LightRAGServiceError(LightRAGError):
    """Exception raised during LightRAG service operations"""

    pass


class LightRagHolder:
    """
    Wrapper class holding a LightRAG instance and its llm / embedding implementations.
    """

    def __init__(
        self,
        rag: LightRAG,
        llm_func: Callable[..., Awaitable[str]],
        embed_impl: Callable[[List[str]], Awaitable[numpy.ndarray]],
        namespace_prefix: str,
    ) -> None:
        self.rag = rag
        self.llm_func = llm_func
        self.embed_impl = embed_impl
        self.namespace_prefix = namespace_prefix
        self._creation_time = datetime.now()

    @property
    def creation_time(self) -> datetime:
        """Get the creation time of this holder"""
        return self._creation_time

    async def ainsert(
        self,
        input: str | list[str],
        split_by_character: str | None = None,
        split_by_character_only: bool = False,
        ids: str | list[str] | None = None,
        file_paths: str | list[str] | None = None,
    ) -> None:
        """Insert documents into LightRAG with error handling"""
        try:
            logger.info(f"Inserting documents into LightRAG namespace: {self.namespace_prefix}")
            return await self.rag.ainsert(input, split_by_character, split_by_character_only, ids, file_paths)
        except Exception as e:
            logger.error(f"Failed to insert documents into LightRAG namespace {self.namespace_prefix}: {str(e)}")
            raise LightRAGServiceError(f"Document insertion failed: {str(e)}") from e

    async def get_processed_docs(self) -> dict[str, Any]:
        """Get processed documents with error handling"""
        try:
            return await self.rag.get_docs_by_status(DocStatus.PROCESSED)
        except Exception as e:
            logger.error(f"Failed to get processed docs from LightRAG namespace {self.namespace_prefix}: {str(e)}")
            raise LightRAGServiceError(f"Failed to get processed docs: {str(e)}") from e

    async def aget_docs_by_ids(self, ids: str | list[str]) -> dict[str, Any]:
        """Get documents by IDs with error handling"""
        try:
            return await self.rag.aget_docs_by_ids(ids)
        except Exception as e:
            logger.error(f"Failed to get docs by IDs from LightRAG namespace {self.namespace_prefix}: {str(e)}")
            raise LightRAGServiceError(f"Failed to get docs by IDs: {str(e)}") from e

    async def aquery(
        self, query: str, param: QueryParam = QueryParam(), system_prompt: str | None = None
    ) -> str | AsyncIterator[str]:
        """Query LightRAG with error handling"""
        try:
            logger.info(f"Querying LightRAG namespace {self.namespace_prefix} with query: {query[:100]}...")
            return await self.rag.aquery(query, param, system_prompt)
        except Exception as e:
            logger.error(f"Failed to query LightRAG namespace {self.namespace_prefix}: {str(e)}")
            raise LightRAGServiceError(f"Query failed: {str(e)}") from e

    async def adelete_by_doc_id(self, doc_id: str) -> None:
        """Delete document by ID with error handling"""
        try:
            logger.info(f"Deleting document {doc_id} from LightRAG namespace: {self.namespace_prefix}")
            return await self.rag.adelete_by_doc_id(doc_id)
        except Exception as e:
            logger.error(
                f"Failed to delete document {doc_id} from LightRAG namespace {self.namespace_prefix}: {str(e)}"
            )
            raise LightRAGServiceError(f"Document deletion failed: {str(e)}") from e


def log_llm_performance(start_time: datetime, end_time: datetime, prompt: str, response: str) -> None:
    """Log LLM performance metrics"""
    latency = (end_time - start_time).total_seconds() if start_time and end_time else 0.0
    logger.info(f"LLM Performance - Start: {start_time}, End: {end_time}, Latency: {latency:.2f}s")
    logger.debug(f"LLM PROMPT: {prompt[:200]}..." if len(prompt) > 200 else f"LLM PROMPT: {prompt}")
    logger.debug(f"LLM RESPONSE: {response[:200]}..." if len(response) > 200 else f"LLM RESPONSE: {response}")


async def create_lightrag_llm_func(collection: Collection, msp_dict: Dict[str, Any]) -> Callable[..., Awaitable[str]]:
    """Create LightRAG LLM function with proper error handling"""
    config = parseCollectionConfig(collection.config)
    lightrag_msp = config.completion.model_service_provider
    lightrag_model_name = config.completion.model

    logger.info(f"Creating LightRAG LLM function with MSP: {lightrag_msp}, Model: {lightrag_model_name}")

    if lightrag_msp not in msp_dict:
        raise LightRAGInitializationError(
            f"Model service provider '{lightrag_msp}' not found in user's MSP configuration"
        )

    msp = msp_dict[lightrag_msp]
    base_url = msp.base_url
    api_key = msp.api_key

    logger.info(f"Using base URL: {base_url}")

    async def lightrag_llm_func(
        prompt: str,
        system_prompt: Optional[str] = None,
        history_messages: List = [],
        **kwargs,
    ) -> str:
        start_time = datetime.now()

        try:
            merged_kwargs = {"api_key": api_key, "base_url": base_url, **config.completion.dict()}

            from aperag.llm.completion_service import CompletionService

            completion_service = CompletionService(**merged_kwargs)

            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})

            if history_messages:
                messages.extend(history_messages)

            # 使用 CompletionService 的公共 agenerate_stream 方法
            full_response = ""
            async for chunk in completion_service.agenerate_stream(
                history=messages, prompt=prompt, memory=True if messages else False
            ):
                if chunk:
                    full_response += chunk

            end_time = datetime.now()
            log_llm_performance(start_time, end_time, prompt, full_response)
            return full_response

        except Exception as e:
            end_time = datetime.now()
            logger.error(f"LLM function failed after {(end_time - start_time).total_seconds():.2f}s: {str(e)}")
            raise LightRAGServiceError(f"LLM completion failed: {str(e)}") from e

    return lightrag_llm_func


async def gen_lightrag_llm_func(collection: Collection) -> Callable[..., Awaitable[str]]:
    """Generate LightRAG LLM function with improved error handling"""
    try:
        msp_dict = await query_msp_dict(collection.user)
        return await create_lightrag_llm_func(collection, msp_dict)
    except Exception as e:
        logger.error(f"Failed to generate LightRAG LLM function for collection {collection.id}: {str(e)}")
        raise LightRAGInitializationError(f"Failed to create LLM function: {str(e)}") from e


async def gen_lightrag_embed_func(
    collection: Collection,
) -> Tuple[Callable[[list[str]], Awaitable[numpy.ndarray]], int]:
    """Generate LightRAG embedding function with proper error handling"""
    try:
        logger.info(f"Creating LightRAG embedding function for collection {collection.id}")
        embedding_svc, dim = await get_collection_embedding_service(collection)

        async def lightrag_embed_func(texts: list[str]) -> numpy.ndarray:
            try:
                embeddings = await embedding_svc.aembed_documents(texts)
                return numpy.array(embeddings)
            except Exception as e:
                logger.error(f"Embedding generation failed for {len(texts)} texts: {str(e)}")
                raise LightRAGServiceError(f"Embedding generation failed: {str(e)}") from e

        logger.info(f"Successfully created embedding function with dimension: {dim}")
        return lightrag_embed_func, dim

    except Exception as e:
        logger.error(f"Failed to generate LightRAG embedding function for collection {collection.id}: {str(e)}")
        raise LightRAGInitializationError(f"Failed to create embedding function: {str(e)}") from e


async def create_and_initialize_lightrag(
    namespace_prefix: str,
    llm_func: Callable[..., Awaitable[str]],
    embed_impl: Callable[[List[str]], Awaitable[numpy.ndarray]],
    embed_dim: int,
) -> LightRagHolder:
    """
    Creates the LightRAG dependencies, instantiates the object for a specific namespace,
    and runs its asynchronous initializers using supplied callable implementations.
    Returns a fully ready LightRagHolder for the given namespace.

    Args:
        namespace_prefix: The namespace prefix for this LightRAG instance.
        llm_func: Async callable that produces LLM completions.
        embed_impl: Async callable that produces embeddings.
        embed_dim: Embedding dimension.
    """
    logger.info(f"Creating and initializing LightRAG object for namespace: '{namespace_prefix}'")

    try:
        rag = LightRAG(
            namespace_prefix=namespace_prefix,
            working_dir=LightRAGConfig.WORKING_DIR,
            llm_model_func=llm_func,
            embedding_func=EmbeddingFunc(
                embedding_dim=embed_dim,
                max_token_size=LightRAGConfig.EMBEDDING_MAX_TOKEN_SIZE,
                func=embed_impl,
            ),
            enable_llm_cache=LightRAGConfig.ENABLE_LLM_CACHE,
            max_parallel_insert=LightRAGConfig.MAX_PARALLEL_INSERT,
            llm_model_max_async=LightRAGConfig.LLM_MODEL_MAX_ASYNC,
            entity_extract_max_gleaning=LightRAGConfig.ENTITY_EXTRACT_MAX_GLEANING,
            auto_manage_storages_states=False,
            addon_params={
                "language": LightRAGConfig.DEFAULT_LANGUAGE,
            },
        )

        await rag.initialize_storages()
        await initialize_pipeline_status()

        logger.info(f"LightRAG object for namespace '{namespace_prefix}' successfully initialized")
        return LightRagHolder(rag=rag, llm_func=llm_func, embed_impl=embed_impl, namespace_prefix=namespace_prefix)

    except Exception as e:
        logger.error(f"Failed to create and initialize LightRAG for namespace '{namespace_prefix}': {str(e)}")
        raise LightRAGInitializationError(f"LightRAG initialization failed: {str(e)}") from e


# Module-level cache management
class LightRAGCache:
    """Thread-safe cache for LightRAG instances"""

    def __init__(self):
        self._instances: Dict[str, LightRagHolder] = {}
        self._lock = asyncio.Lock()

    async def get(self, namespace_prefix: str) -> Optional[LightRagHolder]:
        """Get cached instance"""
        async with self._lock:
            return self._instances.get(namespace_prefix)

    async def set(self, namespace_prefix: str, holder: LightRagHolder) -> None:
        """Set cached instance"""
        async with self._lock:
            self._instances[namespace_prefix] = holder
            logger.info(f"Cached LightRAG instance for namespace '{namespace_prefix}'")

    async def remove(self, namespace_prefix: str) -> bool:
        """Remove cached instance"""
        async with self._lock:
            if namespace_prefix in self._instances:
                del self._instances[namespace_prefix]
                logger.info(f"Removed LightRAG instance from cache for namespace '{namespace_prefix}'")
                return True
            return False

    async def clear(self) -> None:
        """Clear all cached instances"""
        async with self._lock:
            count = len(self._instances)
            self._instances.clear()
            logger.info(f"Cleared {count} LightRAG instances from cache")

    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        async with self._lock:
            return {
                "total_instances": len(self._instances),
                "namespaces": list(self._instances.keys()),
                "creation_times": {ns: holder.creation_time.isoformat() for ns, holder in self._instances.items()},
            }


# Global cache instance
_cache = LightRAGCache()


async def get_lightrag_holder(collection: Collection) -> LightRagHolder:
    """
    Get or create a LightRAG holder for the given collection.
    Uses caching to avoid repeated initialization.
    """
    namespace_prefix: str = generate_lightrag_namespace_prefix(collection.id)

    if not namespace_prefix or not isinstance(namespace_prefix, str):
        raise ValueError("A valid namespace_prefix string must be provided.")

    # Try to get from cache first
    cached_holder = await _cache.get(namespace_prefix)
    if cached_holder:
        logger.debug(f"Using cached LightRAG instance for namespace '{namespace_prefix}'")
        return cached_holder

    logger.info(f"Initializing new LightRAG instance for namespace '{namespace_prefix}' (cache miss)")

    try:
        # Create new instance
        embed_func, dim = await gen_lightrag_embed_func(collection=collection)
        llm_func = await gen_lightrag_llm_func(collection=collection)

        if not llm_func:
            raise LightRAGInitializationError("Failed to create LLM function - no suitable MSP found")

        holder = await create_and_initialize_lightrag(namespace_prefix, llm_func, embed_func, embed_dim=dim)

        # Cache the new instance
        await _cache.set(namespace_prefix, holder)

        logger.info(f"LightRAG instance for namespace '{namespace_prefix}' initialized and cached successfully")
        return holder

    except Exception as e:
        # Clean up any partial cache entries
        await _cache.remove(namespace_prefix)
        logger.error(f"Failed to initialize LightRAG instance for namespace '{namespace_prefix}': {str(e)}")
        raise LightRAGInitializationError(
            f"Failed during LightRAG instance creation/initialization for namespace '{namespace_prefix}'"
        ) from e


async def reload_lightrag_holder(collection: Collection) -> LightRagHolder:
    """
    Reload a LightRAG holder by removing the cached instance and creating a new one.
    """
    namespace_prefix: str = generate_lightrag_namespace_prefix(collection.id)
    logger.info(f"Reloading LightRAG holder for namespace '{namespace_prefix}'")

    await _cache.remove(namespace_prefix)
    return await get_lightrag_holder(collection)


async def delete_lightrag_holder(collection: Collection) -> bool:
    """
    Delete a LightRAG holder from cache.
    Returns True if an instance was removed, False if it wasn't cached.
    """
    namespace_prefix: str = generate_lightrag_namespace_prefix(collection.id)
    if not namespace_prefix or not isinstance(namespace_prefix, str):
        return False

    return await _cache.remove(namespace_prefix)


async def get_cache_stats() -> Dict[str, Any]:
    """Get statistics about the LightRAG cache"""
    return await _cache.get_stats()


async def clear_all_cache() -> None:
    """Clear all cached LightRAG instances"""
    await _cache.clear()
