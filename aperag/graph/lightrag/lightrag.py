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
from functools import partial
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
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
    QueryParam,
    StoragesStatus,
)
from .namespace import NameSpace
from .operate import (
    build_query_context,
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
    check_storage_env_vars,
    clean_text,
    compute_mdhash_id,
    create_lightrag_logger,
    lazy_external_import,
    logger,
)


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

    llm_model_max_async: int = field(default=8)
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
        # Verify storage implementation compatibility and environment variables
        storage_configs = [
            ("KV_STORAGE", self.kv_storage),
            ("VECTOR_STORAGE", self.vector_storage),
            ("GRAPH_STORAGE", self.graph_storage),
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

        self._storages_status = StoragesStatus.CREATED

    async def initialize_storages(self):
        """Asynchronously initialize the storages"""
        if self._storages_status == StoragesStatus.CREATED:
            tasks = []

            for storage in (
                self.text_chunks,
                self.entities_vdb,
                self.relationships_vdb,
                self.chunks_vdb,
                self.chunk_entity_relation_graph,
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
                self.text_chunks,
                self.entities_vdb,
                self.relationships_vdb,
                self.chunks_vdb,
                self.chunk_entity_relation_graph,
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

    def _find_connected_components(self, chunk_results: List[tuple[dict, dict]]) -> List[List[str]]:
        """
        Find connected components in the extracted entities and relationships.

        Args:
            chunk_results: List of (nodes_dict, edges_dict) tuples from entity extraction

        Returns:
            List of entity groups, where each group is a list of connected entity names
        """
        # Build adjacency list
        adjacency: Dict[str, set[str]] = {}

        for nodes, edges in chunk_results:
            # Add all nodes to adjacency list
            for entity_name in nodes.keys():
                if entity_name not in adjacency:
                    adjacency[entity_name] = set()

            # Add edges to create connections
            for src, tgt in edges.keys():
                if src not in adjacency:
                    adjacency[src] = set()
                if tgt not in adjacency:
                    adjacency[tgt] = set()
                adjacency[src].add(tgt)
                adjacency[tgt].add(src)

        # Find connected components using BFS
        visited = set()
        components = []

        for node in adjacency:
            if node not in visited:
                # Start BFS from this node
                component = []
                queue = [node]
                visited.add(node)

                while queue:
                    current = queue.pop(0)
                    component.append(current)

                    # Visit all neighbors
                    for neighbor in adjacency.get(current, set()):
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append(neighbor)

                components.append(component)

        logger.info(f"Found {len(components)} connected components from {len(adjacency)} entities")
        return components

    async def _grouping_process_chunk_results(
        self,
        chunk_results: List[tuple[dict, dict]],
        collection_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Process entities and relationships in groups based on connected components.

        Args:
            chunk_results: List of (nodes_dict, edges_dict) from entity extraction
            collection_id: Optional collection ID for logging

        Returns:
            Dict with processing results
        """
        components = self._find_connected_components(chunk_results)

        workspace = self.workspace if self.workspace else "default"
        lightrag_logger = create_lightrag_logger(prefix="LightRAG-GraphIndex", workspace=workspace)

        # Prepare component data for parallel processing
        component_tasks = []

        for i, component in enumerate(components):
            # Create a set for quick lookup
            component_entities = set(component)

            # Filter chunk results for this component
            component_chunk_results = []

            for nodes, edges in chunk_results:
                # Filter nodes belonging to this component
                filtered_nodes = {
                    entity_name: entity_data
                    for entity_name, entity_data in nodes.items()
                    if entity_name in component_entities
                }

                # Filter edges where both endpoints belong to this component
                filtered_edges = {
                    (src, tgt): edge_data
                    for (src, tgt), edge_data in edges.items()
                    if src in component_entities and tgt in component_entities
                }

                if filtered_nodes or filtered_edges:
                    component_chunk_results.append((filtered_nodes, filtered_edges))

            if not component_chunk_results:
                continue

            # Add task data for this component
            component_tasks.append(
                {
                    "index": i,
                    "component": component,
                    "component_chunk_results": component_chunk_results,
                    "total_components": len(components),
                }
            )

        # Process components concurrently with semaphore
        semaphore = asyncio.Semaphore(1)

        async def _process_component_with_semaphore(task_data):
            async with semaphore:
                lightrag_logger.info(
                    f"Processing component {task_data['index'] + 1}/{task_data['total_components']} "
                    f"with {len(task_data['component'])} entities"
                )

                # Call merge_nodes_and_edges with component information
                result = await merge_nodes_and_edges(
                    chunk_results=task_data["component_chunk_results"],
                    component=task_data["component"],
                    workspace=self.workspace,
                    knowledge_graph_inst=self.chunk_entity_relation_graph,
                    entity_vdb=self.entities_vdb,
                    relationships_vdb=self.relationships_vdb,
                    llm_model_func=self.llm_model_func,
                    tokenizer=self.tokenizer,
                    llm_model_max_token_size=self.llm_model_max_token_size,
                    summary_to_max_tokens=self.summary_to_max_tokens,
                    addon_params=self.addon_params or PROMPTS["DEFAULT_LANGUAGE"],
                    force_llm_summary_on_merge=self.force_llm_summary_on_merge,
                    lightrag_logger=lightrag_logger,
                )

                lightrag_logger.info(
                    f"Completed component {task_data['index'] + 1}: "
                    f"{result['entity_count']} entities, {result['relation_count']} relations"
                )

                return result

        # Create tasks for concurrent processing
        tasks = []
        for task_data in component_tasks:
            task = asyncio.create_task(_process_component_with_semaphore(task_data))
            tasks.append(task)

        # Wait for all tasks to complete or for the first exception
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)

        # Check if any task raised an exception
        for task in done:
            if task.exception():
                # Cancel all pending tasks
                for pending_task in pending:
                    pending_task.cancel()

                # Wait for cancellation to complete
                if pending:
                    await asyncio.wait(pending)

                # Re-raise the exception
                raise task.exception()

        # Collect results from all tasks
        results = [task.result() for task in tasks]

        # Calculate totals
        total_entities = sum(r["entity_count"] for r in results)
        total_relations = sum(r["relation_count"] for r in results)
        processed_groups = len(results)

        return {
            "groups_processed": processed_groups,
            "total_entities": total_entities,
            "total_relations": total_relations,
            "collection_id": collection_id,
        }

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

            # Write to storage (avoid concurrent operations on same connection)
            logger.info(f"LightRAG: About to upsert {len(chunks)} chunks to storages")
            logger.info(f"LightRAG: Calling chunks_vdb.upsert with {len(chunks)} chunks")
            await self.chunks_vdb.upsert(chunks)
            logger.info(f"LightRAG: Calling text_chunks.upsert with {len(chunks)} chunks")
            await self.text_chunks.upsert(chunks)
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

            # 1. Extract entities and relations from chunks (completely parallel, no lock)
            chunk_results = await extract_entities(
                chunks,
                use_llm_func=self.llm_model_func,
                entity_extract_max_gleaning=self.entity_extract_max_gleaning,
                addon_params=self.addon_params,
                llm_model_max_async=self.llm_model_max_async,
                lightrag_logger=lightrag_logger,
            )

            # 2. Process each component group with its own lock scope
            result = await self._grouping_process_chunk_results(chunk_results, collection_id)

            # Count total results
            entity_count = sum(len(nodes) for nodes, _ in chunk_results)
            relation_count = sum(len(edges) for _, edges in chunk_results)

            lightrag_logger.info(
                f"Graph indexing completed: {entity_count} entities, {relation_count} relations "
                f"in {result['groups_processed']} groups"
            )

            return {
                "status": "success",
                "chunks_processed": len(chunks),
                "entities_extracted": entity_count,
                "relations_extracted": relation_count,
                "groups_processed": result["groups_processed"],
                "collection_id": collection_id,
            }

        except Exception as e:
            if "lightrag_logger" in locals():
                lightrag_logger.error(f"Graph indexing failed: {str(e)}")
            else:
                logger.error(f"Graph indexing failed: {str(e)}", exc_info=True)
            raise e

    # ============= End of New Stateless Interfaces =============

    async def aquery_context(
        self,
        query: str,
        param: QueryParam = QueryParam(),
    ):
        param.original_query = query
        entities_context, relations_context, text_units_context = await build_query_context(
            query.strip(),
            self.chunk_entity_relation_graph,
            self.entities_vdb,
            self.relationships_vdb,
            self.text_chunks,
            param,
            self.tokenizer,
            self.llm_model_func,
            self.addon_params,
            chunks_vdb=self.chunks_vdb,
        )

        # Remove file_path from all contexts before serialization
        def remove_file_path_from_context(context_list):
            """Remove file_path field from each item in the context list"""
            if not context_list:
                return context_list

            cleaned_context = []
            for item in context_list:
                if isinstance(item, dict) and "file_path" in item:
                    # Create a copy without file_path
                    cleaned_item = {k: v for k, v in item.items() if k != "file_path"}
                    cleaned_context.append(cleaned_item)
                else:
                    cleaned_context.append(item)
            return cleaned_context

        # Clean all contexts
        entities_context = remove_file_path_from_context(entities_context)
        relations_context = remove_file_path_from_context(relations_context)
        text_units_context = remove_file_path_from_context(text_units_context)

        import json

        entities_str = json.dumps(entities_context, ensure_ascii=False)
        relations_str = json.dumps(relations_context, ensure_ascii=False)
        text_units_str = json.dumps(text_units_context, ensure_ascii=False)

        context = f"""-----Entities(KG)-----

```json
{entities_str}
```

-----Relationships(KG)-----

```json
{relations_str}
```

-----Document Chunks(DC)-----

```json
{text_units_str}
```

"""
        return context

    async def aquery(
        self,
        query: str,
        param: QueryParam = QueryParam(),
        system_prompt: str | None = None,
    ) -> str | AsyncIterator[str] | Tuple[dict, dict, dict]:
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

    # TODO: Deprecated (Deleting documents can cause hallucinations in RAG.)
    # Document delete is not working properly for most of the storage implementations.
    async def adelete_by_doc_id(self, doc_id: str) -> None:
        """Delete a document and all its related data

        Args:
            doc_id: Document ID to delete
        """
        try:
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
