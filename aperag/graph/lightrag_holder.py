import asyncio
import logging
from typing import Optional, List, Dict, Callable, Awaitable, Tuple, AsyncIterator, Any

import json
import numpy
from lightrag import LightRAG, QueryParam
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.llm.openai import openai_complete_if_cache
from lightrag.utils import EmbeddingFunc
from lightrag.base import DocStatus

from aperag.db.models import Collection
from aperag.db.ops import (
    query_msp_dict,
)
from aperag.embed.base_embedding import get_collection_embedding_model
from aperag.utils.utils import generate_lightrag_namespace_prefix
from config.settings import (
    LIGHT_RAG_LLM_API_KEY,
    LIGHT_RAG_LLM_BASE_URL,
    LIGHT_RAG_LLM_MODEL,
    LIGHT_RAG_WORKING_DIR,
    LIGHT_RAG_ENABLE_LLM_CACHE,
    LIGHT_RAG_MAX_PARALLEL_INSERT,
)

# --- Configuration Parameters---
LLM_API_KEY = LIGHT_RAG_LLM_API_KEY
LLM_BASE_URL = LIGHT_RAG_LLM_BASE_URL
LLM_MODEL = LIGHT_RAG_LLM_MODEL
WORKING_DIR = LIGHT_RAG_WORKING_DIR
ENABLE_LLM_CACHE = LIGHT_RAG_ENABLE_LLM_CACHE
MAX_PARALLEL_INSERT = LIGHT_RAG_MAX_PARALLEL_INSERT
# --- End Configuration Parameters ---

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class LightRagHolder:
    """
    Wrapper class holding a LightRAG instance and its llm / embedding implementations.
    """

    def __init__(
        self,
        rag: LightRAG,
        llm_func: Callable[..., Awaitable[str]],
        embed_impl: Callable[[List[str]], Awaitable[numpy.ndarray]],
    ) -> None:
        self.rag = rag
        self.llm_func = llm_func
        self.embed_impl = embed_impl

    def insert(
        self,
        input: str | list[str],
        split_by_character: str | None = None,
        split_by_character_only: bool = False,
        ids: str | list[str] | None = None,
        file_paths: str | list[str] | None = None,
    ) -> None:
        return self.rag.insert(input, split_by_character, split_by_character_only, ids, file_paths)

    async def get_processed_docs(self) -> dict[str, Any]:
        return await self.rag.get_docs_by_status(DocStatus.PROCESSED)

    async def aquery(self, query: str, param: QueryParam = QueryParam(), system_prompt: str | None = None) -> str | AsyncIterator[str]:
        return await self.rag.aquery(query, param, system_prompt)

    async def adelete_by_doc_id(self, doc_id: str) -> None:
        return await self.rag.adelete_by_doc_id(doc_id)




# ---------- Default llm_func & embed_impl ---------- #
async def gen_lightrag_llm_func(collection: Collection) -> Callable[..., Awaitable[str]]:
    config = json.loads(collection.config)
    lightrag_backend = config.get("lightrag_model_service_provider", "")
    lightrag_model_name = config.get("lightrag_model_name", "")
    logging.info("gen_lightrag_llm_func %s %s", lightrag_backend, lightrag_model_name)

    msp_dict = await query_msp_dict(collection.user)
    if lightrag_backend in msp_dict:
        msp = msp_dict[lightrag_backend]
        lightrag_model_service_url = msp.base_url
        lightrag_model_service_api_key = msp.api_key
        logging.info("gen_lightrag_llm_func %s %s", lightrag_model_service_url, lightrag_model_service_api_key)

        async def lightrag_llm_func(
            prompt: str,
            system_prompt: Optional[str] = None,
            history_messages: List = [],
            **kwargs,
        ) -> str:
            merged_kwargs = {
                "api_key": lightrag_model_service_api_key,
                "base_url": lightrag_model_service_url,
                "model": lightrag_model_name,
                **kwargs,
            }
            return await openai_complete_if_cache(
                prompt=prompt,
                system_prompt=system_prompt,
                history_messages=history_messages,
                **merged_kwargs,
            )
        return lightrag_llm_func
    
    return None

# Module-level cache
_lightrag_instances: Dict[str, LightRagHolder] = {}
_initialization_lock = asyncio.Lock()


async def _create_and_initialize_lightrag(
    namespace_prefix: str,
    llm_func: Callable[..., Awaitable[str]],
    embed_impl: Callable[[List[str]], Awaitable[numpy.ndarray]],
    embed_dim: int
) -> LightRagHolder:
    """
    Creates the LightRAG dependencies, instantiates the object for a specific namespace,
    and runs its asynchronous initializers using supplied callable implementations.
    Returns a fully ready LightRagClient for the given namespace.

    Args:
        namespace_prefix: The namespace prefix for this LightRAG instance.
        llm_func: Async callable that produces LLM completions.
        embed_impl: Async callable that produces embeddings.
    """
    logger.debug(f"Creating and initializing LightRAG object for namespace: '{namespace_prefix}'...")

    rag = LightRAG(
        namespace_prefix=namespace_prefix,
        working_dir=WORKING_DIR,
        llm_model_func=llm_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=embed_dim,
            max_token_size=8192,
            func=embed_impl,
        ),
        enable_llm_cache=ENABLE_LLM_CACHE,
        max_parallel_insert=MAX_PARALLEL_INSERT,
    )

    await rag.initialize_storages()
    await initialize_pipeline_status()

    logger.debug(f"LightRAG object for namespace '{namespace_prefix}' fully initialized.")
    return LightRagHolder(rag=rag, llm_func=llm_func, embed_impl=embed_impl)

async def gen_lightrag_embed_func(collection: Collection) -> Tuple[
    Callable[[list[str]], Awaitable[numpy.ndarray]],
    int
]:
    embedding_svc, dim = await get_collection_embedding_model(collection)
    async def lightrag_embed_func(texts: list[str]) -> numpy.ndarray:
        embeddings = await embedding_svc.aembed_documents(texts)
        return numpy.array(embeddings)

    return lightrag_embed_func, dim

async def get_lightrag_holder(
    collection: Collection
) -> LightRagHolder:
    # Fixme: if lightrag_model changes, we need to re-initialize the lightrag instance
    namespace_prefix: str = generate_lightrag_namespace_prefix(collection.id)
    if not namespace_prefix or not isinstance(namespace_prefix, str):
        raise ValueError("A valid namespace_prefix string must be provided.")

    if namespace_prefix in _lightrag_instances:
        return _lightrag_instances[namespace_prefix]

    async with _initialization_lock:
        if namespace_prefix in _lightrag_instances:
            return _lightrag_instances[namespace_prefix]

        logger.info(f"Initializing LightRAG instance for namespace '{namespace_prefix}' (lazy loading)...")
        try:
            embed_func, dim = await gen_lightrag_embed_func(collection=collection)
            llm_func = await gen_lightrag_llm_func(collection=collection)
            client = await _create_and_initialize_lightrag(namespace_prefix, llm_func, embed_func, embed_dim=dim)
            _lightrag_instances[namespace_prefix] = client
            logger.info(f"LightRAG instance for namespace '{namespace_prefix}' initialized successfully.")
            return client
        except Exception as e:
            logger.exception(
                f"Failed during LightRAG instance creation/initialization for namespace '{namespace_prefix}'.",
                exc_info=e,
            )
            _lightrag_instances.pop(namespace_prefix, None)
            raise RuntimeError(
                f"Failed during LightRAG instance creation/initialization for namespace '{namespace_prefix}'"
            ) from e