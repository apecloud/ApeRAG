import asyncio
import logging
from typing import Optional, List, Any, Dict

import numpy
from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc
from lightrag.llm.openai import openai_embed, openai_complete_if_cache
from lightrag.kg.shared_storage import initialize_pipeline_status
from config.settings import (
    LIGHT_RAG_LLM_API_KEY,
    LIGHT_RAG_LLM_BASE_URL,
    LIGHT_RAG_LLM_MODEL,
    LIGHT_RAG_EMBEDDING_API_KEY,
    LIGHT_RAG_EMBEDDING_BASE_URL,
    LIGHT_RAG_EMBEDDING_MODEL,
    LIGHT_RAG_EMBEDDING_DIM,
    LIGHT_RAG_EMBEDDING_MAX_TOKENS,
    LIGHT_RAG_WORKING_DIR,
    LIGHT_RAG_ENABLE_LLM_CACHE,
    LIGHT_RAG_MAX_PARALLEL_INSERT,
)
# --- Configuration Parameters---
LLM_API_KEY = LIGHT_RAG_LLM_API_KEY
LLM_BASE_URL = LIGHT_RAG_LLM_BASE_URL
LLM_MODEL = LIGHT_RAG_LLM_MODEL
EMBEDDING_MODEL = LIGHT_RAG_EMBEDDING_MODEL
EMBEDDING_API_KEY = LIGHT_RAG_EMBEDDING_API_KEY
EMBEDDING_BASE_URL = LIGHT_RAG_EMBEDDING_BASE_URL
EMBEDDING_DIM = LIGHT_RAG_EMBEDDING_DIM
EMBEDDING_MAX_TOKENS = LIGHT_RAG_EMBEDDING_MAX_TOKENS
WORKING_DIR = LIGHT_RAG_WORKING_DIR
ENABLE_LLM_CACHE = LIGHT_RAG_ENABLE_LLM_CACHE
MAX_PARALLEL_INSERT = LIGHT_RAG_MAX_PARALLEL_INSERT
# --- End Configuration Parameters ---

logger = logging.getLogger(__name__)
# Set up basic logging if running standalone
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# Module-level dictionary to hold LightRAG instances keyed by namespace_prefix
_lightrag_instances: Dict[str, LightRAG] = {}
# Single lock to manage concurrent initialization *attempts* across all namespaces
# This prevents multiple coroutines trying to create the *same* namespace instance simultaneously.
_initialization_lock = asyncio.Lock()

async def _create_and_initialize_lightrag(namespace_prefix: str) -> LightRAG:
    """
    Creates the LightRAG dependencies, instantiates the object for a specific namespace,
    and runs its asynchronous initializers using pre-defined constants.
    Returns a fully ready instance for the given namespace.

    Args:
        namespace_prefix: The namespace prefix for this LightRAG instance,
                          acting like a collection name.
    """
    logger.debug(f"Creating and initializing LightRAG object for namespace: '{namespace_prefix}'...")

    # --- Define LLM function inline ---
    async def llm_func(
            prompt: str, system_prompt: Optional[str] = None, history_messages: List = [], **kwargs
    ) -> str:
        merged_kwargs = {
            "api_key": LLM_API_KEY,
            "base_url": LLM_BASE_URL,
            "model": LLM_MODEL,
            **kwargs
        }
        return await openai_complete_if_cache(
            prompt=prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            **merged_kwargs,
        )

    # --- Define Embedding function implementation inline ---
    async def embed_impl(texts: list[str]) -> numpy.ndarray:
        return await openai_embed(
            texts,
            model=EMBEDDING_MODEL,
            api_key=EMBEDDING_API_KEY,
            base_url=EMBEDDING_BASE_URL,
        )

    # --- Instantiate LightRAG with constants AND the namespace_prefix ---
    rag = LightRAG(
        namespace_prefix=namespace_prefix, # Pass the namespace prefix here
        working_dir=WORKING_DIR, # Note: All instances share the same base working dir
        llm_model_func=llm_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=EMBEDDING_DIM,
            max_token_size=EMBEDDING_MAX_TOKENS,
            func=embed_impl,
        ),
        enable_llm_cache=ENABLE_LLM_CACHE,
        max_parallel_insert=MAX_PARALLEL_INSERT,
    )
    logger.debug(f"LightRAG object created for namespace: '{namespace_prefix}'.")

    # --- Run async initializers ---
    logger.debug(f"Initializing LightRAG instance storages for namespace: '{namespace_prefix}'...")
    await rag.initialize_storages()
    logger.debug(f"LightRAG instance storages initialized for namespace: '{namespace_prefix}'.")

    # --- Initialize pipeline status (optional, check if it needs to be per-namespace or global) ---
    # Assuming it's okay or idempotent to call this potentially multiple times
    # (once per namespace initialization)
    await initialize_pipeline_status()
    logger.debug(f"Pipeline status initialized (called during init of namespace: '{namespace_prefix}').")

    logger.debug(f"LightRAG object for namespace '{namespace_prefix}' fully initialized.")
    return rag


async def get_lightrag_instance(namespace_prefix: str) -> LightRAG:
    """
    Gets the LightRAG instance for a specific namespace_prefix, performing
    lazy creation and initialization if needed. Uses an asyncio Lock to
    ensure thread-safe initialization for each specific namespace.

    Args:
        namespace_prefix: The namespace (like a collection name) for the
                          desired LightRAG instance.

    Returns:
        The initialized LightRAG instance for the given namespace.

    Raises:
        RuntimeError: If creation or initialization fails for the specific namespace.
        ValueError: If namespace_prefix is empty or invalid.
    """
    if not namespace_prefix or not isinstance(namespace_prefix, str):
         raise ValueError("A valid namespace_prefix string must be provided.")

    # Fast path: Check if already initialized without acquiring the lock
    if namespace_prefix in _lightrag_instances:
        return _lightrag_instances[namespace_prefix]

    # Acquire lock to handle concurrent initialization attempts for the *same* namespace
    async with _initialization_lock:
        # Double-check if another coroutine initialized it while waiting for the lock
        if namespace_prefix in _lightrag_instances:
            return _lightrag_instances[namespace_prefix]

        logger.info(f"Initializing LightRAG instance for namespace '{namespace_prefix}' (lazy loading)...")
        try:
            # --- Call the creation and initialization function for the specific namespace ---
            initialized_instance = await _create_and_initialize_lightrag(namespace_prefix)

            # Store the initialized instance in the dictionary
            _lightrag_instances[namespace_prefix] = initialized_instance

            logger.info(f"LightRAG instance for namespace '{namespace_prefix}' initialized successfully.")
            return initialized_instance

        except Exception as e:
            # Log the specific error during creation or initialization for this namespace
            logger.exception(f"Failed during LightRAG instance creation/initialization for namespace '{namespace_prefix}'.", exc_info=e)
            # Ensure the entry is not added on failure so subsequent calls retry
            if namespace_prefix in _lightrag_instances:
                 del _lightrag_instances[namespace_prefix] # Clean up potential partial state if needed
            raise RuntimeError(f"Failed during LightRAG instance creation/initialization for namespace '{namespace_prefix}'") from e


async def query_lightrag(query: str, param: QueryParam, namespace_prefix: str) -> Optional[Any]:
    """
    Performs a query using the initialized LightRAG instance for the specified namespace.

    Args:
        query: The query string.
        param: A LightRAG QueryParam object specifying query details (mode, etc.).
        namespace_prefix: The namespace of the LightRAG instance to use for the query.

    Returns:
        The result from LightRAG's aquery method, or None if an error occurs
        (including errors getting the instance).
    """
    try:
        # Get the specific instance for the requested namespace
        rag_instance = await get_lightrag_instance(namespace_prefix)
        result = await rag_instance.aquery(query, param=param)
        return result
    except (RuntimeError, ValueError) as e: # Catch initialization/lookup errors from get_lightrag_instance
         logger.error(f"Cannot query LightRAG for namespace '{namespace_prefix}': {e}")
         return None
    except Exception as e:
        logger.exception(f"Error during LightRAG query for namespace '{namespace_prefix}', query '{query[:50]}...'", exc_info=e)
        return None