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

"""
LightRAG Module for ApeRAG

This module is based on the original LightRAG project with extensive modifications.

Original Project:
- Repository: https://github.com/HKUDS/LightRAG
- Paper: "LightRAG: Simple and Fast Retrieval-Augmented Generation" (arXiv:2410.05779)
- Authors: Zirui Guo, Lianghao Xia, Yanhua Yu, Tu Ao, Chao Huang
- License: MIT License

Modifications by ApeRAG Team:
- Removed global state management for true concurrent processing
- Added stateless interfaces for Celery/Prefect integration
- Implemented instance-level locking mechanism
- Enhanced error handling and stability
- See changelog.md for detailed modifications
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import partial
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    final,
)

from aperag.graph.lightrag.constants import (
    DEFAULT_FORCE_LLM_SUMMARY_ON_MERGE,
    DEFAULT_MAX_TOKEN_SUMMARY,
)
from aperag.graph.lightrag.kg import (
    STORAGES,
    verify_storage_implementation,
)
from aperag.graph.lightrag.utils import get_env_value

from .base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    DocProcessingStatus,
    DocStatus,
    DocStatusStorage,
    QueryParam,
    StoragesStatus,
)
from .namespace import NameSpace
from .operate import (
    chunking_by_token_size,
    extract_entities,
    kg_query,
    merge_nodes_and_edges,
    naive_query,
)
from .prompt import GRAPH_FIELD_SEP, PROMPTS
from .types import KnowledgeGraph
from .utils import (
    EmbeddingFunc,
    TiktokenTokenizer,
    Tokenizer,
    always_get_an_event_loop,
    check_storage_env_vars,
    clean_text,
    compute_mdhash_id,
    create_lightrag_logger,
    get_content_summary,
    lazy_external_import,
    logger,
)
from .utils_graph import get_graph_db_lock


@final
@dataclass
class LightRAG:
    """LightRAG: Simple and Fast Retrieval-Augmented Generation."""

    # Storage
    # ---

    kv_storage: str = field(default="PGOpsSyncKVStorage")
    """Storage backend for key-value data."""

    vector_storage: str = field(default="PGOpsSyncVectorStorage")
    """Storage backend for vector embeddings."""

    graph_storage: str = field(default="Neo4JSyncStorage")
    """Storage backend for knowledge graphs."""

    doc_status_storage: str = field(default="PGOpsSyncDocStatusStorage")
    """Storage type for tracking document processing statuses."""

    # Entity extraction
    # ---

    entity_extract_max_gleaning: int = field(default=1)
    """Maximum number of entity extraction attempts for ambiguous content."""

    summary_to_max_tokens: int = field(default=get_env_value("MAX_TOKEN_SUMMARY", DEFAULT_MAX_TOKEN_SUMMARY, int))

    force_llm_summary_on_merge: int = field(
        default=get_env_value("FORCE_LLM_SUMMARY_ON_MERGE", DEFAULT_FORCE_LLM_SUMMARY_ON_MERGE, int)
    )

    # Text chunking
    # ---

    chunk_token_size: int = field(default=1200)
    """Maximum number of tokens per text chunk when splitting documents."""

    chunk_overlap_token_size: int = field(default=100)
    """Number of overlapping tokens between consecutive text chunks to preserve context."""

    tokenizer: Optional[Tokenizer] = field(default=None)
    """
    A function that returns a Tokenizer instance.
    If None, and a `tiktoken_model_name` is provided, a TiktokenTokenizer will be created.
    If both are None, the default TiktokenTokenizer is used.
    """

    tiktoken_model_name: str = field(default="gpt-4o-mini")
    """Model name used for tokenization when chunking text with tiktoken. Defaults to `gpt-4o-mini`."""

    chunking_func: Callable[
        [
            Tokenizer,
            str,
            Optional[str],
            bool,
            int,
            int,
        ],
        List[Dict[str, Any]],
    ] = field(default_factory=lambda: chunking_by_token_size)
    """
    Custom chunking function for splitting text into chunks before processing.

    The function should take the following parameters:

        - `tokenizer`: A Tokenizer instance to use for tokenization.
        - `content`: The text to be split into chunks.
        - `split_by_character`: The character to split the text on. If None, the text is split into chunks of `chunk_token_size` tokens.
        - `split_by_character_only`: If True, the text is split only on the specified character.
        - `chunk_token_size`: The maximum number of tokens per chunk.
        - `chunk_overlap_token_size`: The number of overlapping tokens between consecutive chunks.

    The function should return a list of dictionaries, where each dictionary contains the following keys:
        - `tokens`: The number of tokens in the chunk.
        - `content`: The text content of the chunk.

    Defaults to `chunking_by_token_size` if not specified.
    """

    # Embedding
    # ---

    embedding_func: EmbeddingFunc | None = field(default=None)
    """Function for computing text embeddings. Must be set before use."""

    embedding_batch_num: int = field(default=32)
    """Batch size for embedding computations."""

    embedding_func_max_async: int = field(default=16)
    """Maximum number of concurrent embedding function calls."""

    embedding_cache_config: dict[str, Any] = field(
        default_factory=lambda: {
            "enabled": False,
            "similarity_threshold": 0.95,
            "use_llm_check": False,
        }
    )
    """Configuration for embedding cache.
    - enabled: If True, enables caching to avoid redundant computations.
    - similarity_threshold: Minimum similarity score to use cached embeddings.
    - use_llm_check: If True, validates cached embeddings using an LLM.
    """

    cosine_better_than_threshold: float = field(default=0.2)
    """
    Threshold for cosine similarity to determine if two embeddings are similar enough to be considered the same.
    """

    max_batch_size: int = field(default=32)
    """
    Maximum batch size for embedding computations.
    """

    # LLM Configuration
    # ---

    llm_model_func: Callable[..., object] | None = field(default=None)
    """Function for interacting with the large language model (LLM). Must be set before use."""

    llm_model_name: str = field(default="gpt-4o-mini")
    """Name of the LLM model used for generating responses."""

    llm_model_max_token_size: int = field(default=32768)
    """Maximum number of tokens allowed per LLM response."""

    llm_model_max_async: int = field(default=4)
    """Maximum number of concurrent LLM calls."""

    llm_model_kwargs: dict[str, Any] = field(default_factory=dict)
    """Additional keyword arguments passed to the LLM model function."""

    # Storage
    # ---

    vector_db_storage_cls_kwargs: dict[str, Any] = field(default_factory=dict)
    """Additional parameters for vector database storage."""

    workspace: str = field(default="default")
    """Workspace identifier for data isolation across different collections/tenants."""

    # Extensions
    # ---

    addon_params: dict[str, Any] = field(
        default_factory=lambda: {"language": get_env_value("SUMMARY_LANGUAGE", "English", str)}
    )

    # Storages Management
    # ---

    _storages_status: StoragesStatus = field(default=StoragesStatus.NOT_CREATED)

    def __post_init__(self):
        self._graph_db_lock = get_graph_db_lock(self.workspace)
        logger.info(f"Using universal concurrent control for graph operations: workspace={self.workspace}")

        # Verify storage implementation compatibility and environment variables
        storage_configs = [
            ("KV_STORAGE", self.kv_storage),
            ("VECTOR_STORAGE", self.vector_storage),
            ("GRAPH_STORAGE", self.graph_storage),
            ("DOC_STATUS_STORAGE", self.doc_status_storage),
        ]

        for storage_type, storage_name in storage_configs:
            # Verify storage implementation compatibility
            verify_storage_implementation(storage_type, storage_name)
            # Check environment variables
            check_storage_env_vars(storage_name)

        # Init Tokenizer
        # Post-initialization hook to handle backward compatabile tokenizer initialization based on provided parameters
        if self.tokenizer is None:
            if self.tiktoken_model_name:
                self.tokenizer = TiktokenTokenizer(self.tiktoken_model_name)
            else:
                self.tokenizer = TiktokenTokenizer()

        # Initialize all storages
        self.key_string_value_json_storage_cls: type[BaseKVStorage] = self._get_storage_class(self.kv_storage)  # type: ignore
        self.vector_db_storage_cls: type[BaseVectorStorage] = self._get_storage_class(self.vector_storage)  # type: ignore
        self.graph_storage_cls: type[BaseGraphStorage] = self._get_storage_class(self.graph_storage)  # type: ignore
        self.key_string_value_json_storage_cls = partial(  # type: ignore
            self.key_string_value_json_storage_cls
        )
        self.vector_db_storage_cls = partial(  # type: ignore
            self.vector_db_storage_cls
        )
        self.graph_storage_cls = partial(  # type: ignore
            self.graph_storage_cls
        )

        # Initialize document status storage
        self.doc_status_storage_cls = self._get_storage_class(self.doc_status_storage)

        self.full_docs: BaseKVStorage = self.key_string_value_json_storage_cls(  # type: ignore
            namespace=NameSpace.KV_STORE_FULL_DOCS,
            workspace=self.workspace,
            embedding_func=self.embedding_func,
        )

        # TODO: deprecating, text_chunks is redundant with chunks_vdb
        self.text_chunks: BaseKVStorage = self.key_string_value_json_storage_cls(  # type: ignore
            namespace=NameSpace.KV_STORE_TEXT_CHUNKS,
            workspace=self.workspace,
            embedding_func=self.embedding_func,
        )
        self.chunk_entity_relation_graph: BaseGraphStorage = self.graph_storage_cls(  # type: ignore
            namespace=NameSpace.GRAPH_STORE_CHUNK_ENTITY_RELATION,
            workspace=self.workspace,
            embedding_func=self.embedding_func,
        )

        self.entities_vdb: BaseVectorStorage = self.vector_db_storage_cls(  # type: ignore
            namespace=NameSpace.VECTOR_STORE_ENTITIES,
            workspace=self.workspace,
            embedding_func=self.embedding_func,
            cosine_better_than_threshold=self.cosine_better_than_threshold,
            _max_batch_size=self.max_batch_size,
            meta_fields={"entity_name", "source_id", "content", "file_path"},
        )
        self.relationships_vdb: BaseVectorStorage = self.vector_db_storage_cls(  # type: ignore
            namespace=NameSpace.VECTOR_STORE_RELATIONSHIPS,
            workspace=self.workspace,
            embedding_func=self.embedding_func,
            cosine_better_than_threshold=self.cosine_better_than_threshold,
            _max_batch_size=self.max_batch_size,
            meta_fields={"src_id", "tgt_id", "source_id", "content", "file_path"},
        )
        self.chunks_vdb: BaseVectorStorage = self.vector_db_storage_cls(  # type: ignore
            namespace=NameSpace.VECTOR_STORE_CHUNKS,
            workspace=self.workspace,
            embedding_func=self.embedding_func,
            cosine_better_than_threshold=self.cosine_better_than_threshold,
            _max_batch_size=self.max_batch_size,
            meta_fields={"full_doc_id", "content", "file_path"},
        )

        # Initialize document status storage
        self.doc_status: DocStatusStorage = self.doc_status_storage_cls(
            namespace=NameSpace.DOC_STATUS,
            workspace=self.workspace,
            embedding_func=None,
        )

        self._storages_status = StoragesStatus.CREATED

    async def initialize_storages(self):
        """Asynchronously initialize the storages"""
        if self._storages_status == StoragesStatus.CREATED:
            tasks = []

            for storage in (
                self.full_docs,
                self.text_chunks,
                self.entities_vdb,
                self.relationships_vdb,
                self.chunks_vdb,
                self.chunk_entity_relation_graph,
                self.doc_status,
            ):
                if storage:
                    tasks.append(storage.initialize())

            await asyncio.gather(*tasks)

            self._storages_status = StoragesStatus.INITIALIZED
            logger.debug("Initialized Storages")

    async def finalize_storages(self):
        """Asynchronously finalize the storages"""
        if self._storages_status == StoragesStatus.INITIALIZED:
            tasks = []

            for storage in (
                self.full_docs,
                self.text_chunks,
                self.entities_vdb,
                self.relationships_vdb,
                self.chunks_vdb,
                self.chunk_entity_relation_graph,
                self.doc_status,
            ):
                if storage:
                    tasks.append(storage.finalize())

            await asyncio.gather(*tasks)

            self._storages_status = StoragesStatus.FINALIZED
            logger.debug("Finalized Storages")

    async def get_graph_labels(self):
        text = await self.chunk_entity_relation_graph.get_all_labels()
        return text

    async def get_knowledge_graph(
        self,
        node_label: str,
        max_depth: int = 3,
        max_nodes: int = 1000,
    ) -> KnowledgeGraph:
        """Get knowledge graph for a given label

        Args:
            node_label (str): Label to get knowledge graph for
            max_depth (int): Maximum depth of graph
            max_nodes (int, optional): Maximum number of nodes to return. Defaults to 1000.

        Returns:
            KnowledgeGraph: Knowledge graph containing nodes and edges
        """

        return await self.chunk_entity_relation_graph.get_knowledge_graph(node_label, max_depth, max_nodes)

    def _get_storage_class(self, storage_name: str) -> Callable[..., Any]:
        import_path = STORAGES[storage_name]
        storage_class = lazy_external_import(import_path, storage_name)
        return storage_class

    # ============= New Stateless Interfaces =============

    async def ainsert_and_chunk_document(
        self,
        documents: list[str],
        doc_ids: list[str] | None = None,
        file_paths: list[str] | None = None,
        split_by_character: str | None = None,
        split_by_character_only: bool = False,
    ) -> dict[str, Any]:
        """
        Stateless document insertion and chunking - inserts documents and performs chunking in one step.

        Args:
            documents: List of document contents
            doc_ids: Optional list of document IDs (will generate if not provided)
            file_paths: Optional list of file paths for citation
            split_by_character: Optional character to split by
            split_by_character_only: If True, only split by character

        Returns:
            Dict with document metadata and chunks
        """
        # Ensure lists
        if isinstance(documents, str):
            documents = [documents]
        if isinstance(doc_ids, str):
            doc_ids = [doc_ids]
        if isinstance(file_paths, str):
            file_paths = [file_paths]

        # Validate inputs
        if file_paths and len(file_paths) != len(documents):
            raise ValueError("Number of file paths must match number of documents")
        if doc_ids and len(doc_ids) != len(documents):
            raise ValueError("Number of doc IDs must match number of documents")

        # Use default file paths if not provided
        if not file_paths:
            file_paths = ["unknown_source"] * len(documents)

        # Generate or validate IDs
        if not doc_ids:
            doc_ids = [compute_mdhash_id(clean_text(doc), prefix="doc-") for doc in documents]
        else:
            # Check uniqueness
            if len(doc_ids) != len(set(doc_ids)):
                raise ValueError("Document IDs must be unique")

        results = []

        for doc_id, content, file_path in zip(doc_ids, documents, file_paths):
            cleaned_content = clean_text(content)

            if not cleaned_content.strip():
                raise ValueError(f"Document {doc_id} content is empty after cleaning")

            # Perform chunking
            chunk_list = self.chunking_func(
                self.tokenizer,
                cleaned_content,
                split_by_character,
                split_by_character_only,
                self.chunk_overlap_token_size,
                self.chunk_token_size,
            )

            # Validate chunk_list format
            if not chunk_list:
                raise ValueError(f"Chunking returned empty list for document {doc_id}")

            # Create chunk data with IDs and validate each chunk
            chunks = {}
            for i, chunk_data in enumerate(chunk_list):
                # Validate chunk_data format
                if not isinstance(chunk_data, dict):
                    raise ValueError(f"Chunk {i} is not a dictionary: {type(chunk_data)}")
                if "content" not in chunk_data:
                    raise ValueError(f"Chunk {i} missing 'content' key: {chunk_data.keys()}")
                if not chunk_data["content"]:
                    logger.warning(f"Chunk {i} has empty content, skipping")
                    continue

                chunk_id = compute_mdhash_id(chunk_data["content"], prefix="chunk-")
                chunks[chunk_id] = {
                    **chunk_data,
                    "full_doc_id": doc_id,
                    "file_path": file_path,
                }

            if not chunks:
                raise ValueError(f"No valid chunks created for document {doc_id}")

            # Prepare complete status data
            status_data = {
                "status": DocStatus.PROCESSING,
                "content": cleaned_content,
                "content_summary": get_content_summary(cleaned_content),
                "content_length": len(cleaned_content),
                "chunks_count": len(chunks),
                "created_at": datetime.now(timezone.utc).isoformat(),
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "file_path": file_path,
            }

            # Write to storage (avoid concurrent operations on same connection)
            doc_data = {doc_id: {"content": cleaned_content}}
            logger.info(f"LightRAG: About to upsert {len(chunks)} chunks to storages")
            await self.full_docs.upsert(doc_data)
            logger.info(f"LightRAG: Calling chunks_vdb.upsert with {len(chunks)} chunks")
            await self.chunks_vdb.upsert(chunks)
            logger.info(f"LightRAG: Calling text_chunks.upsert with {len(chunks)} chunks")
            await self.text_chunks.upsert(chunks)
            await self.doc_status.upsert({doc_id: status_data})
            logger.info(f"LightRAG: Completed all upsert operations for {doc_id}")

            logger.info(f"Inserted and chunked document {doc_id}: {len(chunks)} chunks")

            results.append(
                {
                    "doc_id": doc_id,
                    "chunks": list(chunks.keys()),
                    "chunk_count": len(chunks),
                    "chunks_data": chunks,
                    "status": "processed",
                }
            )

        return {
            "results": results,
            "total_documents": len(doc_ids),
            "total_chunks": sum(len(r["chunks"]) for r in results),
            "status": "success",
        }

    async def aprocess_graph_indexing(
        self,
        chunks: dict[str, Any],
        collection_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Stateless graph indexing - extracts entities/relations and builds graph index.

        Args:
            chunks: Dict of chunk_id -> chunk_data
            collection_id: Optional collection ID for logging

        Returns:
            Dict with extraction results
        """
        # Create logger for this processing session
        workspace = self.workspace if self.workspace else "default"
        lightrag_logger = create_lightrag_logger(prefix="LightRAG-GraphIndex", workspace=workspace)

        try:
            # Validate chunks input
            if not chunks:
                raise ValueError("No chunks provided for graph indexing")

            # Validate each chunk has required fields
            for chunk_id, chunk_data in chunks.items():
                if not isinstance(chunk_data, dict):
                    raise ValueError(f"Chunk {chunk_id} is not a dictionary")
                if "content" not in chunk_data:
                    raise ValueError(f"Chunk {chunk_id} missing 'content' key")
                if not chunk_data["content"]:
                    raise ValueError(f"Chunk {chunk_id} has empty content")

            lightrag_logger.info(f"Starting graph indexing for {len(chunks)} chunks")

            # 1. Extract entities and relations from chunks
            chunk_results = await extract_entities(
                chunks,
                use_llm_func=self.llm_model_func,
                entity_extract_max_gleaning=self.entity_extract_max_gleaning,
                addon_params=self.addon_params,
                llm_model_max_async=self.llm_model_max_async,
                lightrag_logger=lightrag_logger,
            )

            lightrag_logger.info(f"Extracted entities and relations from {len(chunks)} chunks")

            # 2. Merge nodes and edges (using instance-level lock)
            await merge_nodes_and_edges(
                chunk_results=chunk_results,
                knowledge_graph_inst=self.chunk_entity_relation_graph,
                entity_vdb=self.entities_vdb,
                relationships_vdb=self.relationships_vdb,
                llm_model_func=self.llm_model_func,
                tokenizer=self.tokenizer,
                llm_model_max_token_size=self.llm_model_max_token_size,
                summary_to_max_tokens=self.summary_to_max_tokens,
                addon_params=self.addon_params or PROMPTS["DEFAULT_LANGUAGE"],
                force_llm_summary_on_merge=self.force_llm_summary_on_merge,
                current_file_number=0,
                total_files=0,
                file_path="stateless_processing",
                lightrag_logger=lightrag_logger,
                graph_db_lock=self._graph_db_lock,
            )

            lightrag_logger.info("Completed merging entities and relations")

            # Count results
            entity_count = sum(len(nodes) for nodes, _ in chunk_results)
            relation_count = sum(len(edges) for _, edges in chunk_results)

            lightrag_logger.info(f"Graph indexing completed: {entity_count} entities, {relation_count} relations")

            return {
                "status": "success",
                "chunks_processed": len(chunks),
                "entities_extracted": entity_count,
                "relations_extracted": relation_count,
                "collection_id": collection_id,
            }

        except Exception as e:
            if "lightrag_logger" in locals():
                lightrag_logger.error(f"Graph indexing failed: {str(e)}")
            else:
                logger.error(f"Graph indexing failed: {str(e)}", exc_info=True)
            return {
                "status": "error",
                "error": str(e),
                "chunks_processed": 0,
                "collection_id": collection_id,
            }

    # ============= End of New Stateless Interfaces =============

    def query(
        self,
        query: str,
        param: QueryParam = QueryParam(),
        system_prompt: str | None = None,
    ) -> str | Iterator[str]:
        """
        Perform a sync query.

        Args:
            query (str): The query to be executed.
            param (QueryParam): Configuration parameters for query execution.
            prompt (Optional[str]): Custom prompts for fine-tuned control over the system's behavior. Defaults to None, which uses PROMPTS["rag_response"].

        Returns:
            str: The result of the query execution.
        """
        loop = always_get_an_event_loop()

        return loop.run_until_complete(self.aquery(query, param, system_prompt))  # type: ignore

    async def aquery(
        self,
        query: str,
        param: QueryParam = QueryParam(),
        system_prompt: str | None = None,
    ) -> str | AsyncIterator[str]:
        """
        Perform a async query.

        Args:
            query (str): The query to be executed.
            param (QueryParam): Configuration parameters for query execution.
                If param.model_func is provided, it will be used instead of the global model.
            prompt (Optional[str]): Custom prompts for fine-tuned control over the system's behavior. Defaults to None, which uses PROMPTS["rag_response"].

        Returns:
            str: The result of the query execution.
        """
        # If a custom model is provided in param, temporarily update global config
        # Save original query for vector search
        param.original_query = query

        if param.mode in ["local", "global", "hybrid", "mix"]:
            response = await kg_query(
                query.strip(),
                self.chunk_entity_relation_graph,
                self.entities_vdb,
                self.relationships_vdb,
                self.text_chunks,
                param,
                self.tokenizer,
                self.llm_model_func,
                self.addon_params,
                system_prompt=system_prompt,
                chunks_vdb=self.chunks_vdb,
            )
        elif param.mode == "naive":
            response = await naive_query(
                query.strip(),
                self.chunks_vdb,
                param,
                self.llm_model_func,
                self.tokenizer,
                system_prompt=system_prompt,
            )
        elif param.mode == "bypass":
            # Bypass mode: directly use LLM without knowledge retrieval

            param.stream = True if param.stream is None else param.stream
            response = await self.llm_model_func(
                query.strip(),
                system_prompt=system_prompt,
                history_messages=param.conversation_history,
                stream=param.stream,
            )
        else:
            raise ValueError(f"Unknown mode {param.mode}")
        return response

    async def get_docs_by_status(self, status: DocStatus) -> dict[str, DocProcessingStatus]:
        """Get documents by status

        Returns:
            Dict with document id is keys and document status is values
        """
        return await self.doc_status.get_docs_by_status(status)

    async def aget_docs_by_ids(self, ids: str | list[str]) -> dict[str, DocProcessingStatus]:
        """Retrieves the processing status for one or more documents by their IDs.

        Args:
            ids: A single document ID (string) or a list of document IDs (list of strings).

        Returns:
            A dictionary where keys are the document IDs for which a status was found,
            and values are the corresponding DocProcessingStatus objects. IDs that
            are not found in the storage will be omitted from the result dictionary.
        """
        if isinstance(ids, str):
            # Ensure input is always a list of IDs for uniform processing
            id_list = [ids]
        elif ids is None:  # Handle potential None input gracefully, although type hint suggests str/list
            logger.warning("aget_docs_by_ids called with None input, returning empty dict.")
            return {}
        else:
            # Assume input is already a list if not a string
            id_list = ids

        # Return early if the final list of IDs is empty
        if not id_list:
            logger.debug("aget_docs_by_ids called with an empty list of IDs.")
            return {}

        # Create tasks to fetch document statuses concurrently using the doc_status storage
        tasks = [self.doc_status.get_by_id(doc_id) for doc_id in id_list]
        # Execute tasks concurrently and gather the results. Results maintain order.
        # Type hint indicates results can be DocProcessingStatus or None if not found.
        results_list: list[Optional[DocProcessingStatus]] = await asyncio.gather(*tasks)

        # Build the result dictionary, mapping found IDs to their statuses
        found_statuses: dict[str, DocProcessingStatus] = {}
        # Keep track of IDs for which no status was found (for logging purposes)
        not_found_ids: list[str] = []

        # Iterate through the results, correlating them back to the original IDs
        for i, status_obj in enumerate(results_list):
            doc_id = id_list[i]  # Get the original ID corresponding to this result index
            if status_obj:
                # If a status object was returned (not None), add it to the result dict
                found_statuses[doc_id] = status_obj
            else:
                # If status_obj is None, the document ID was not found in storage
                not_found_ids.append(doc_id)

        # Log a warning if any of the requested document IDs were not found
        if not_found_ids:
            logger.warning(f"Document statuses not found for the following IDs: {not_found_ids}")

        # Return the dictionary containing statuses only for the found document IDs
        return found_statuses

    # TODO: Deprecated (Deleting documents can cause hallucinations in RAG.)
    # Document delete is not working properly for most of the storage implementations.
    async def adelete_by_doc_id(self, doc_id: str) -> None:
        """Delete a document and all its related data

        Args:
            doc_id: Document ID to delete
        """
        try:
            # 1. Get the document status and related data
            if not await self.doc_status.get_by_id(doc_id):
                logger.warning(f"Document {doc_id} not found")
                return

            logger.debug(f"Starting deletion for document {doc_id}")

            # 2. Get all chunks related to this document
            # Find all chunks where full_doc_id equals the current doc_id
            all_chunks = await self.text_chunks.get_all()
            related_chunks = {
                chunk_id: chunk_data
                for chunk_id, chunk_data in all_chunks.items()
                if isinstance(chunk_data, dict) and chunk_data.get("full_doc_id") == doc_id
            }

            if not related_chunks:
                logger.warning(f"No chunks found for document {doc_id}")
                return

            # Get all related chunk IDs
            chunk_ids = set(related_chunks.keys())
            logger.debug(f"Found {len(chunk_ids)} chunks to delete")

            # TODO: self.entities_vdb.client_storage only works for local storage, need to fix this

            # 3. Before deleting, check the related entities and relationships for these chunks
            for chunk_id in chunk_ids:
                # Check entities
                entities_storage = await self.entities_vdb.client_storage
                entities = [dp for dp in entities_storage["data"] if chunk_id in dp.get("source_id")]
                logger.debug(f"Chunk {chunk_id} has {len(entities)} related entities")

                # Check relationships
                relationships_storage = await self.relationships_vdb.client_storage
                relations = [dp for dp in relationships_storage["data"] if chunk_id in dp.get("source_id")]
                logger.debug(f"Chunk {chunk_id} has {len(relations)} related relations")

            # Continue with the original deletion process...

            # 4. Delete chunks from vector database
            if chunk_ids:
                await self.chunks_vdb.delete(chunk_ids)
                await self.text_chunks.delete(chunk_ids)

            # 5. Find and process entities and relationships that have these chunks as source
            # Get all nodes and edges from the graph storage using storage-agnostic methods
            entities_to_delete = set()
            entities_to_update = {}  # entity_name -> new_source_id
            relationships_to_delete = set()
            relationships_to_update = {}  # (src, tgt) -> new_source_id

            # Process entities - use storage-agnostic methods
            all_labels = await self.chunk_entity_relation_graph.get_all_labels()
            for node_label in all_labels:
                node_data = await self.chunk_entity_relation_graph.get_node(node_label)
                if node_data and "source_id" in node_data:
                    # Split source_id using GRAPH_FIELD_SEP
                    sources = set(node_data["source_id"].split(GRAPH_FIELD_SEP))
                    sources.difference_update(chunk_ids)
                    if not sources:
                        entities_to_delete.add(node_label)
                        logger.debug(f"Entity {node_label} marked for deletion - no remaining sources")
                    else:
                        new_source_id = GRAPH_FIELD_SEP.join(sources)
                        entities_to_update[node_label] = new_source_id
                        logger.debug(f"Entity {node_label} will be updated with new source_id: {new_source_id}")

            # Process relationships
            for node_label in all_labels:
                node_edges = await self.chunk_entity_relation_graph.get_node_edges(node_label)
                if node_edges:
                    for src, tgt in node_edges:
                        edge_data = await self.chunk_entity_relation_graph.get_edge(src, tgt)
                        if edge_data and "source_id" in edge_data:
                            # Split source_id using GRAPH_FIELD_SEP
                            sources = set(edge_data["source_id"].split(GRAPH_FIELD_SEP))
                            sources.difference_update(chunk_ids)
                            if not sources:
                                relationships_to_delete.add((src, tgt))
                                logger.debug(f"Relationship {src}-{tgt} marked for deletion - no remaining sources")
                            else:
                                new_source_id = GRAPH_FIELD_SEP.join(sources)
                                relationships_to_update[(src, tgt)] = new_source_id
                                logger.debug(
                                    f"Relationship {src}-{tgt} will be updated with new source_id: {new_source_id}"
                                )

            # Delete entities
            if entities_to_delete:
                for entity in entities_to_delete:
                    await self.entities_vdb.delete_entity(entity)
                    logger.debug(f"Deleted entity {entity} from vector DB")
                await self.chunk_entity_relation_graph.remove_nodes(list(entities_to_delete))
                logger.debug(f"Deleted {len(entities_to_delete)} entities from graph")

            # Update entities
            for entity, new_source_id in entities_to_update.items():
                node_data = await self.chunk_entity_relation_graph.get_node(entity)
                if node_data:
                    node_data["source_id"] = new_source_id
                    await self.chunk_entity_relation_graph.upsert_node(entity, node_data)
                    logger.debug(f"Updated entity {entity} with new source_id: {new_source_id}")

            # Delete relationships
            if relationships_to_delete:
                for src, tgt in relationships_to_delete:
                    rel_id_0 = compute_mdhash_id(src + tgt, prefix="rel-")
                    rel_id_1 = compute_mdhash_id(tgt + src, prefix="rel-")
                    await self.relationships_vdb.delete([rel_id_0, rel_id_1])
                    logger.debug(f"Deleted relationship {src}-{tgt} from vector DB")
                await self.chunk_entity_relation_graph.remove_edges(list(relationships_to_delete))
                logger.debug(f"Deleted {len(relationships_to_delete)} relationships from graph")

            # Update relationships
            for (src, tgt), new_source_id in relationships_to_update.items():
                edge_data = await self.chunk_entity_relation_graph.get_edge(src, tgt)
                if edge_data:
                    edge_data["source_id"] = new_source_id
                    await self.chunk_entity_relation_graph.upsert_edge(src, tgt, edge_data)
                    logger.debug(f"Updated relationship {src}-{tgt} with new source_id: {new_source_id}")

            # 6. Delete original document and status
            await self.full_docs.delete([doc_id])
            await self.doc_status.delete([doc_id])

            logger.info(
                f"Successfully deleted document {doc_id} and related data. "
                f"Deleted {len(entities_to_delete)} entities and {len(relationships_to_delete)} relationships. "
                f"Updated {len(entities_to_update)} entities and {len(relationships_to_update)} relationships."
            )

            async def process_data(data_type, vdb, chunk_id):
                # Check data (entities or relationships)
                storage = await vdb.client_storage
                data_with_chunk = [
                    dp for dp in storage["data"] if chunk_id in (dp.get("source_id") or "").split(GRAPH_FIELD_SEP)
                ]

                data_for_vdb = {}
                if data_with_chunk:
                    logger.warning(f"found {len(data_with_chunk)} {data_type} still referencing chunk {chunk_id}")

                    for item in data_with_chunk:
                        old_sources = item["source_id"].split(GRAPH_FIELD_SEP)
                        new_sources = [src for src in old_sources if src != chunk_id]

                        if not new_sources:
                            logger.info(
                                f"{data_type} {item.get('entity_name', 'N/A')} is deleted because source_id is not exists"
                            )
                            await vdb.delete_entity(item)
                        else:
                            item["source_id"] = GRAPH_FIELD_SEP.join(new_sources)
                            item_id = item["__id__"]
                            data_for_vdb[item_id] = item.copy()
                            if data_type == "entities":
                                data_for_vdb[item_id]["content"] = data_for_vdb[item_id].get("content") or (
                                    item.get("entity_name", "") + (item.get("description") or "")
                                )
                            else:  # relationships
                                data_for_vdb[item_id]["content"] = data_for_vdb[item_id].get("content") or (
                                    (item.get("keywords") or "")
                                    + (item.get("src_id") or "")
                                    + (item.get("tgt_id") or "")
                                    + (item.get("description") or "")
                                )

                    if data_for_vdb:
                        await vdb.upsert(data_for_vdb)
                        logger.info(f"Successfully updated {data_type} in vector DB")

            # Add verification step
            async def verify_deletion():
                # Verify if the document has been deleted
                if await self.full_docs.get_by_id(doc_id):
                    logger.warning(f"Document {doc_id} still exists in full_docs")

                # Verify if chunks have been deleted
                all_remaining_chunks = await self.text_chunks.get_all()
                remaining_related_chunks = {
                    chunk_id: chunk_data
                    for chunk_id, chunk_data in all_remaining_chunks.items()
                    if isinstance(chunk_data, dict) and chunk_data.get("full_doc_id") == doc_id
                }

                if remaining_related_chunks:
                    logger.warning(f"Found {len(remaining_related_chunks)} remaining chunks")

                # Verify entities and relationships
                for chunk_id in chunk_ids:
                    await process_data("entities", self.entities_vdb, chunk_id)
                    await process_data("relationships", self.relationships_vdb, chunk_id)

            await verify_deletion()

        except Exception as e:
            logger.error(f"Error while deleting document {doc_id}: {e}")

    async def adelete_by_entity(self, entity_name: str) -> None:
        """Asynchronously delete an entity and all its relationships.

        Args:
            entity_name: Name of the entity to delete
        """
        from .utils_graph import adelete_by_entity

        return await adelete_by_entity(
            self.chunk_entity_relation_graph,
            self.entities_vdb,
            self.relationships_vdb,
            entity_name,
            graph_db_lock=self._graph_db_lock,
        )

    def delete_by_entity(self, entity_name: str) -> None:
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.adelete_by_entity(entity_name))

    async def adelete_by_relation(self, source_entity: str, target_entity: str) -> None:
        """Asynchronously delete a relation between two entities.

        Args:
            source_entity: Name of the source entity
            target_entity: Name of the target entity
        """
        from .utils_graph import adelete_by_relation

        return await adelete_by_relation(
            self.chunk_entity_relation_graph,
            self.relationships_vdb,
            source_entity,
            target_entity,
            graph_db_lock=self._graph_db_lock,
        )

    def delete_by_relation(self, source_entity: str, target_entity: str) -> None:
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.adelete_by_relation(source_entity, target_entity))

    async def get_entity_info(
        self, entity_name: str, include_vector_data: bool = False
    ) -> dict[str, str | None | dict[str, str]]:
        """Get detailed information of an entity"""
        from .utils_graph import get_entity_info

        return await get_entity_info(
            self.chunk_entity_relation_graph,
            self.entities_vdb,
            entity_name,
            include_vector_data,
        )

    async def get_relation_info(
        self, src_entity: str, tgt_entity: str, include_vector_data: bool = False
    ) -> dict[str, str | None | dict[str, str]]:
        """Get detailed information of a relationship"""
        from .utils_graph import get_relation_info

        return await get_relation_info(
            self.chunk_entity_relation_graph,
            self.relationships_vdb,
            src_entity,
            tgt_entity,
            include_vector_data,
        )

    async def aedit_entity(
        self, entity_name: str, updated_data: dict[str, str], allow_rename: bool = True
    ) -> dict[str, Any]:
        """Asynchronously edit entity information.

        Updates entity information in the knowledge graph and re-embeds the entity in the vector database.

        Args:
            entity_name: Name of the entity to edit
            updated_data: Dictionary containing updated attributes, e.g. {"description": "new description", "entity_type": "new type"}
            allow_rename: Whether to allow entity renaming, defaults to True

        Returns:
            Dictionary containing updated entity information
        """
        from .utils_graph import aedit_entity

        return await aedit_entity(
            self.chunk_entity_relation_graph,
            self.entities_vdb,
            self.relationships_vdb,
            entity_name,
            updated_data,
            allow_rename,
            graph_db_lock=self._graph_db_lock,
        )

    def edit_entity(self, entity_name: str, updated_data: dict[str, str], allow_rename: bool = True) -> dict[str, Any]:
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.aedit_entity(entity_name, updated_data, allow_rename))

    async def aedit_relation(
        self, source_entity: str, target_entity: str, updated_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Asynchronously edit relation information.

        Updates relation (edge) information in the knowledge graph and re-embeds the relation in the vector database.

        Args:
            source_entity: Name of the source entity
            target_entity: Name of the target entity
            updated_data: Dictionary containing updated attributes, e.g. {"description": "new description", "keywords": "new keywords"}

        Returns:
            Dictionary containing updated relation information
        """
        from .utils_graph import aedit_relation

        return await aedit_relation(
            self.chunk_entity_relation_graph,
            self.entities_vdb,
            self.relationships_vdb,
            source_entity,
            target_entity,
            updated_data,
            graph_db_lock=self._graph_db_lock,
        )

    def edit_relation(self, source_entity: str, target_entity: str, updated_data: dict[str, Any]) -> dict[str, Any]:
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.aedit_relation(source_entity, target_entity, updated_data))

    async def acreate_entity(self, entity_name: str, entity_data: dict[str, Any]) -> dict[str, Any]:
        """Asynchronously create a new entity.

        Creates a new entity in the knowledge graph and adds it to the vector database.

        Args:
            entity_name: Name of the new entity
            entity_data: Dictionary containing entity attributes, e.g. {"description": "description", "entity_type": "type"}

        Returns:
            Dictionary containing created entity information
        """
        from .utils_graph import acreate_entity

        return await acreate_entity(
            self.chunk_entity_relation_graph,
            self.entities_vdb,
            self.relationships_vdb,
            entity_name,
            entity_data,
            graph_db_lock=self._graph_db_lock,
        )

    def create_entity(self, entity_name: str, entity_data: dict[str, Any]) -> dict[str, Any]:
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.acreate_entity(entity_name, entity_data))

    async def acreate_relation(
        self, source_entity: str, target_entity: str, relation_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Asynchronously create a new relation between entities.

        Creates a new relation (edge) in the knowledge graph and adds it to the vector database.

        Args:
            source_entity: Name of the source entity
            target_entity: Name of the target entity
            relation_data: Dictionary containing relation attributes, e.g. {"description": "description", "keywords": "keywords"}

        Returns:
            Dictionary containing created relation information
        """
        from .utils_graph import acreate_relation

        return await acreate_relation(
            self.chunk_entity_relation_graph,
            self.entities_vdb,
            self.relationships_vdb,
            source_entity,
            target_entity,
            relation_data,
            graph_db_lock=self._graph_db_lock,
        )

    def create_relation(self, source_entity: str, target_entity: str, relation_data: dict[str, Any]) -> dict[str, Any]:
        loop = always_get_an_event_loop()
        return loop.run_until_complete(self.acreate_relation(source_entity, target_entity, relation_data))

    async def amerge_entities(
        self,
        source_entities: list[str],
        target_entity: str,
        merge_strategy: dict[str, str] = None,
        target_entity_data: dict[str, Any] = None,
    ) -> dict[str, Any]:
        """Asynchronously merge multiple entities into one entity.

        Merges multiple source entities into a target entity, handling all relationships,
        and updating both the knowledge graph and vector database.

        Args:
            source_entities: List of source entity names to merge
            target_entity: Name of the target entity after merging
            merge_strategy: Merge strategy configuration, e.g. {"description": "concatenate", "entity_type": "keep_first"}
                Supported strategies:
                - "concatenate": Concatenate all values (for text fields)
                - "keep_first": Keep the first non-empty value
                - "keep_last": Keep the last non-empty value
                - "join_unique": Join all unique values (for fields separated by delimiter)
            target_entity_data: Dictionary of specific values to set for the target entity,
                overriding any merged values, e.g. {"description": "custom description", "entity_type": "PERSON"}

        Returns:
            Dictionary containing the merged entity information
        """
        from .utils_graph import amerge_entities

        return await amerge_entities(
            self.chunk_entity_relation_graph,
            self.entities_vdb,
            self.relationships_vdb,
            source_entities,
            target_entity,
            merge_strategy,
            target_entity_data,
            graph_db_lock=self._graph_db_lock,
        )

    def merge_entities(
        self,
        source_entities: list[str],
        target_entity: str,
        merge_strategy: dict[str, str] = None,
        target_entity_data: dict[str, Any] = None,
    ) -> dict[str, Any]:
        loop = always_get_an_event_loop()
        return loop.run_until_complete(
            self.amerge_entities(source_entities, target_entity, merge_strategy, target_entity_data)
        )
