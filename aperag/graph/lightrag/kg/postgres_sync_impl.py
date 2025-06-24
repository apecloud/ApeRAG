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

import asyncio
import datetime
from dataclasses import dataclass
from datetime import timezone
from typing import Any, final

import numpy as np

from ..base import (
    BaseKVStorage,
    BaseVectorStorage,
)
from ..utils import logger


@final
@dataclass
class PGOpsSyncKVStorage(BaseKVStorage):
    """PostgreSQL KV Storage using DatabaseOps with sync interface."""

    async def initialize(self):
        """Initialize storage."""
        logger.debug(f"PGOpsSyncKVStorage initialized for workspace '{self.workspace}'")

    async def finalize(self):
        """Clean up resources."""
        logger.debug(f"PGOpsSyncKVStorage finalized for workspace '{self.workspace}'")

    async def get_all(self) -> dict[str, Any]:
        """Get all data from storage"""

        def _sync_get_all():
            # Import here to avoid circular imports
            from aperag.db.ops import db_ops
            from aperag.graph.lightrag.namespace import NameSpace, is_namespace

            # Determine which table to query based on namespace
            if is_namespace(self.namespace, NameSpace.KV_STORE_FULL_DOCS):
                models = db_ops.query_lightrag_doc_full_all(self.workspace)
                return {doc_id: {"id": doc_id, "content": model.content or ""} for doc_id, model in models.items()}
            elif is_namespace(self.namespace, NameSpace.KV_STORE_TEXT_CHUNKS):
                models = db_ops.query_lightrag_doc_chunks_all(self.workspace)
                return {
                    chunk_id: {
                        "id": chunk_id,
                        "tokens": model.tokens,
                        "content": model.content or "",
                        "chunk_order_index": model.chunk_order_index,
                        "full_doc_id": model.full_doc_id,
                        "content_vector": model.content_vector,  # Now returns list[float] directly
                        "file_path": model.file_path,
                    }
                    for chunk_id, model in models.items()
                }
            else:
                logger.error(f"Unknown namespace for get_all: {self.namespace}")
                return {}

        return await asyncio.to_thread(_sync_get_all)

    async def get_by_id(self, id: str) -> dict[str, Any] | None:
        """Get data by id"""

        def _sync_get_by_id():
            # Import here to avoid circular imports
            from aperag.db.ops import db_ops
            from aperag.graph.lightrag.namespace import NameSpace, is_namespace

            if is_namespace(self.namespace, NameSpace.KV_STORE_FULL_DOCS):
                model = db_ops.query_lightrag_doc_full_by_id(self.workspace, id)
                if not model:
                    return None
                return {"id": model.id, "content": model.content or ""}
            elif is_namespace(self.namespace, NameSpace.KV_STORE_TEXT_CHUNKS):
                model = db_ops.query_lightrag_doc_chunks_by_id(self.workspace, id)
                if not model:
                    return None
                return {
                    "id": model.id,
                    "tokens": model.tokens,
                    "content": model.content or "",
                    "chunk_order_index": model.chunk_order_index,
                    "full_doc_id": model.full_doc_id,
                    "content_vector": model.content_vector,  # Now returns list[float] directly
                    "file_path": model.file_path,
                }
            else:
                logger.error(f"Unknown namespace for get_by_id: {self.namespace}")
                return None

        return await asyncio.to_thread(_sync_get_by_id)

    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        """Get data by ids"""

        def _sync_get_by_ids():
            if not ids:
                return []

            # Import here to avoid circular imports
            from aperag.db.ops import db_ops
            from aperag.graph.lightrag.namespace import NameSpace, is_namespace

            if is_namespace(self.namespace, NameSpace.KV_STORE_FULL_DOCS):
                models = db_ops.query_lightrag_doc_full_by_ids(self.workspace, ids)
                return [{"id": model.id, "content": model.content or ""} for model in models]
            elif is_namespace(self.namespace, NameSpace.KV_STORE_TEXT_CHUNKS):
                models = db_ops.query_lightrag_doc_chunks_by_ids(self.workspace, ids)
                return [
                    {
                        "id": model.id,
                        "tokens": model.tokens,
                        "content": model.content or "",
                        "chunk_order_index": model.chunk_order_index,
                        "full_doc_id": model.full_doc_id,
                        "content_vector": model.content_vector,  # Now returns list[float] directly
                        "file_path": model.file_path,
                    }
                    for model in models
                ]
            else:
                logger.error(f"Unknown namespace for get_by_ids: {self.namespace}")
                return []

        return await asyncio.to_thread(_sync_get_by_ids)

    async def filter_keys(self, keys: set[str]) -> set[str]:
        """Filter out existing keys"""

        def _sync_filter_keys():
            if not keys:
                return set()

            # Import here to avoid circular imports
            from aperag.db.ops import db_ops
            from aperag.graph.lightrag.namespace import NameSpace, is_namespace

            keys_list = list(keys)
            if is_namespace(self.namespace, NameSpace.KV_STORE_FULL_DOCS):
                existing_keys = db_ops.filter_lightrag_doc_full_keys(self.workspace, keys_list)
            elif is_namespace(self.namespace, NameSpace.KV_STORE_TEXT_CHUNKS):
                existing_keys = db_ops.filter_lightrag_doc_chunks_keys(self.workspace, keys_list)
            else:
                logger.error(f"Unknown namespace for filter_keys: {self.namespace}")
                return keys

            new_keys = set([s for s in keys if s not in existing_keys])
            return new_keys

        return await asyncio.to_thread(_sync_filter_keys)

    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        """Insert or update data"""

        def _sync_upsert():
            logger.debug(f"Inserting {len(data)} to {self.namespace}")
            if not data:
                return

            # Import here to avoid circular imports
            from aperag.db.ops import db_ops
            from aperag.graph.lightrag.namespace import NameSpace, is_namespace

            if is_namespace(self.namespace, NameSpace.KV_STORE_FULL_DOCS):
                # Transform data for doc_full format
                doc_data = {doc_id: {"content": doc_info.get("content", "")} for doc_id, doc_info in data.items()}
                db_ops.upsert_lightrag_doc_full(self.workspace, doc_data)
            elif is_namespace(self.namespace, NameSpace.KV_STORE_TEXT_CHUNKS):
                # Use data directly for chunks
                db_ops.upsert_lightrag_doc_chunks(self.workspace, data)
            else:
                logger.error(f"Unknown namespace for upsert: {self.namespace}")

        await asyncio.to_thread(_sync_upsert)

    async def delete(self, ids: list[str]) -> None:
        """Delete specific records from storage by their IDs"""

        def _sync_delete():
            if not ids:
                return

            # Import here to avoid circular imports
            from aperag.db.ops import db_ops
            from aperag.graph.lightrag.namespace import NameSpace, is_namespace

            if is_namespace(self.namespace, NameSpace.KV_STORE_FULL_DOCS):
                deleted_count = db_ops.delete_lightrag_doc_full(self.workspace, ids)
                logger.debug(f"Successfully deleted {deleted_count} records from {self.namespace}")
            elif is_namespace(self.namespace, NameSpace.KV_STORE_TEXT_CHUNKS):
                deleted_count = db_ops.delete_lightrag_doc_chunks(self.workspace, ids)
                logger.debug(f"Successfully deleted {deleted_count} records from {self.namespace}")
            else:
                logger.error(f"Unknown namespace for deletion: {self.namespace}")

        await asyncio.to_thread(_sync_delete)

    async def drop(self) -> dict[str, str]:
        """Drop the storage - not implemented for safety"""
        return {"status": "error", "message": "Drop operation not supported for database-backed storage"}


@final
@dataclass
class PGOpsSyncVectorStorage(BaseVectorStorage):
    """PostgreSQL Vector Storage using DatabaseOps with sync interface."""

    async def initialize(self):
        """Initialize storage."""
        logger.debug(f"PGOpsSyncVectorStorage initialized for workspace '{self.workspace}'")

    async def finalize(self):
        """Clean up resources."""
        logger.debug(f"PGOpsSyncVectorStorage finalized for workspace '{self.workspace}'")

    async def get_all(self) -> dict[str, Any]:
        """Get all data from vector storage"""

        def _sync_get_all():
            # Import here to avoid circular imports
            from aperag.db.ops import db_ops
            from aperag.graph.lightrag.namespace import NameSpace, is_namespace

            # Determine which table to query based on namespace
            if is_namespace(self.namespace, NameSpace.VECTOR_STORE_CHUNKS):
                models = db_ops.query_lightrag_doc_chunks_all(self.workspace)
                return {
                    chunk_id: {
                        "id": chunk_id,
                        "tokens": model.tokens,
                        "content": model.content or "",
                        "chunk_order_index": model.chunk_order_index,
                        "full_doc_id": model.full_doc_id,
                        "content_vector": model.content_vector,
                        "file_path": model.file_path,
                        "created_at": int(model.create_time.timestamp()) if model.create_time else None,
                    }
                    for chunk_id, model in models.items()
                }
            elif is_namespace(self.namespace, NameSpace.VECTOR_STORE_ENTITIES):
                models = db_ops.query_lightrag_vdb_entity_all(self.workspace)
                return {
                    entity_id: {
                        "id": entity_id,
                        "entity_name": model.entity_name,
                        "content": model.content or "",
                        "content_vector": model.content_vector,
                        "chunk_ids": model.chunk_ids or [],
                        "file_path": model.file_path,
                        "created_at": int(model.create_time.timestamp()) if model.create_time else None,
                    }
                    for entity_id, model in models.items()
                }
            elif is_namespace(self.namespace, NameSpace.VECTOR_STORE_RELATIONSHIPS):
                models = db_ops.query_lightrag_vdb_relation_all(self.workspace)
                return {
                    relation_id: {
                        "id": relation_id,
                        "source_id": model.source_id,
                        "target_id": model.target_id,
                        "content": model.content or "",
                        "content_vector": model.content_vector,
                        "chunk_ids": model.chunk_ids or [],
                        "file_path": model.file_path,
                        "created_at": int(model.create_time.timestamp()) if model.create_time else None,
                        # Add additional fields that might be expected
                        "src_id": model.source_id,
                        "tgt_id": model.target_id,
                    }
                    for relation_id, model in models.items()
                }
            else:
                logger.error(f"Unknown namespace for get_all: {self.namespace}")
                return {}

        return await asyncio.to_thread(_sync_get_all)

    def _prepare_vector_data(self, item: dict[str, Any], current_time: datetime.datetime) -> dict[str, Any]:
        """Prepare vector data based on namespace."""
        from aperag.graph.lightrag.namespace import NameSpace, is_namespace

        if is_namespace(self.namespace, NameSpace.VECTOR_STORE_CHUNKS):
            return {
                "tokens": item["tokens"],
                "chunk_order_index": item["chunk_order_index"],
                "full_doc_id": item["full_doc_id"],
                "content": item["content"],
                "content_vector": item["__vector__"].tolist()
                if hasattr(item["__vector__"], "tolist")
                else item["__vector__"],
                "file_path": item.get("file_path"),
            }
        elif is_namespace(self.namespace, NameSpace.VECTOR_STORE_ENTITIES):
            source_id = item["source_id"]
            chunk_ids = source_id.split("<SEP>") if isinstance(source_id, str) and "<SEP>" in source_id else [source_id]
            return {
                "entity_name": item["entity_name"],
                "content": item["content"],
                "content_vector": item["__vector__"].tolist()
                if hasattr(item["__vector__"], "tolist")
                else item["__vector__"],
                "chunk_ids": chunk_ids,
                "file_path": item.get("file_path"),
            }
        elif is_namespace(self.namespace, NameSpace.VECTOR_STORE_RELATIONSHIPS):
            source_id = item["source_id"]
            chunk_ids = source_id.split("<SEP>") if isinstance(source_id, str) and "<SEP>" in source_id else [source_id]
            return {
                "source_id": item["src_id"],
                "target_id": item["tgt_id"],
                "content": item["content"],
                "content_vector": item["__vector__"].tolist()
                if hasattr(item["__vector__"], "tolist")
                else item["__vector__"],
                "chunk_ids": chunk_ids,
                "file_path": item.get("file_path"),
            }
        else:
            raise ValueError(f"{self.namespace} is not supported")

    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        """Insert or update vector data"""
        logger.debug(f"Inserting {len(data)} to {self.namespace}")
        if not data:
            return

        # Get current time with UTC timezone
        current_time = datetime.datetime.now(timezone.utc)
        list_data = [
            {
                "__id__": k,
                **{k1: v1 for k1, v1 in v.items()},
            }
            for k, v in data.items()
        ]

        # Compute embeddings first (async)
        contents = [v["content"] for v in data.values()]
        batches = [contents[i : i + self._max_batch_size] for i in range(0, len(contents), self._max_batch_size)]

        embedding_tasks = [self.embedding_func(batch) for batch in batches]
        embeddings_list = await asyncio.gather(*embedding_tasks)
        embeddings = np.concatenate(embeddings_list)

        for i, d in enumerate(list_data):
            d["__vector__"] = embeddings[i]

        def _sync_upsert_with_vectors():
            # Import here to avoid circular imports
            from aperag.db.ops import db_ops
            from aperag.graph.lightrag.namespace import NameSpace, is_namespace

            # Prepare data for each item
            vector_data = {}
            for item in list_data:
                item_id = item["__id__"]
                prepared_data = self._prepare_vector_data(item, current_time)
                vector_data[item_id] = prepared_data

            # Use appropriate DatabaseOps method based on namespace
            if is_namespace(self.namespace, NameSpace.VECTOR_STORE_CHUNKS):
                db_ops.upsert_lightrag_doc_chunks(self.workspace, vector_data)
            elif is_namespace(self.namespace, NameSpace.VECTOR_STORE_ENTITIES):
                db_ops.upsert_lightrag_vdb_entity(self.workspace, vector_data)
            elif is_namespace(self.namespace, NameSpace.VECTOR_STORE_RELATIONSHIPS):
                db_ops.upsert_lightrag_vdb_relation(self.workspace, vector_data)
            else:
                raise ValueError(f"{self.namespace} is not supported")

        await asyncio.to_thread(_sync_upsert_with_vectors)

    async def query(self, query: str, top_k: int, ids: list[str] | None = None) -> list[dict[str, Any]]:
        """Query vectors by similarity"""
        # Compute embedding for query
        embeddings = await self.embedding_func([query])
        embedding = embeddings[0]

        def _sync_query():
            # Import here to avoid circular imports
            from aperag.db.ops import db_ops
            from aperag.graph.lightrag.namespace import NameSpace, is_namespace

            # Convert embedding to list if it's numpy array
            if hasattr(embedding, "tolist"):
                embedding_list = embedding.tolist()
            else:
                embedding_list = list(embedding)

            # Use appropriate similarity search method based on namespace
            if is_namespace(self.namespace, NameSpace.VECTOR_STORE_CHUNKS):
                results = db_ops.query_lightrag_doc_chunks_similarity(
                    self.workspace, embedding_list, top_k, ids, self.cosine_better_than_threshold
                )
                # Convert results to expected format for chunks
                formatted_results = []
                for result in results:
                    if hasattr(result, "_asdict"):
                        # Handle NamedTuple or Row objects
                        row_dict = result._asdict()
                    elif isinstance(result, dict):
                        row_dict = result
                    else:
                        # Convert Row object to dict manually
                        row_dict = {key: getattr(result, key) for key in result.keys()}

                    formatted_results.append(
                        {
                            "id": row_dict.get("id"),
                            "content": row_dict.get("content", ""),
                            "file_path": row_dict.get("file_path"),
                            "created_at": row_dict.get("created_at"),
                        }
                    )
                return formatted_results

            elif is_namespace(self.namespace, NameSpace.VECTOR_STORE_ENTITIES):
                results = db_ops.query_lightrag_vdb_entity_similarity(
                    self.workspace, embedding_list, top_k, ids, self.cosine_better_than_threshold
                )
                # Convert results to expected format for entities
                formatted_results = []
                for result in results:
                    if hasattr(result, "_asdict"):
                        # Handle NamedTuple or Row objects
                        row_dict = result._asdict()
                    elif isinstance(result, dict):
                        row_dict = result
                    else:
                        # Convert Row object to dict manually
                        row_dict = {key: getattr(result, key) for key in result.keys()}

                    formatted_results.append(
                        {"entity_name": row_dict.get("entity_name"), "created_at": row_dict.get("created_at")}
                    )
                return formatted_results

            elif is_namespace(self.namespace, NameSpace.VECTOR_STORE_RELATIONSHIPS):
                results = db_ops.query_lightrag_vdb_relation_similarity(
                    self.workspace, embedding_list, top_k, ids, self.cosine_better_than_threshold
                )
                # Convert results to expected format for relationships
                formatted_results = []
                for result in results:
                    if hasattr(result, "_asdict"):
                        # Handle NamedTuple or Row objects
                        row_dict = result._asdict()
                    elif isinstance(result, dict):
                        row_dict = result
                    else:
                        # Convert Row object to dict manually
                        row_dict = {key: getattr(result, key) for key in result.keys()}

                    formatted_results.append(
                        {
                            "src_id": row_dict.get("src_id"),
                            "tgt_id": row_dict.get("tgt_id"),
                            "created_at": row_dict.get("created_at"),
                        }
                    )
                return formatted_results

            else:
                logger.error(f"Unknown namespace for vector similarity query: {self.namespace}")
                return []

        return await asyncio.to_thread(_sync_query)

    async def delete(self, ids: list[str]) -> None:
        """Delete vectors with specified IDs from the storage."""

        def _sync_delete():
            if not ids:
                return

            # Import here to avoid circular imports
            from aperag.db.ops import db_ops
            from aperag.graph.lightrag.namespace import NameSpace, is_namespace

            if is_namespace(self.namespace, NameSpace.VECTOR_STORE_CHUNKS):
                deleted_count = db_ops.delete_lightrag_doc_chunks(self.workspace, ids)
                logger.debug(f"Successfully deleted {deleted_count} vectors from {self.namespace}")
            elif is_namespace(self.namespace, NameSpace.VECTOR_STORE_ENTITIES):
                deleted_count = db_ops.delete_lightrag_vdb_entity(self.workspace, ids)
                logger.debug(f"Successfully deleted {deleted_count} vectors from {self.namespace}")
            elif is_namespace(self.namespace, NameSpace.VECTOR_STORE_RELATIONSHIPS):
                deleted_count = db_ops.delete_lightrag_vdb_relation(self.workspace, ids)
                logger.debug(f"Successfully deleted {deleted_count} vectors from {self.namespace}")
            else:
                logger.error(f"Unknown namespace for vector deletion: {self.namespace}")

        await asyncio.to_thread(_sync_delete)

    async def delete_entity(self, entity_name: str) -> None:
        """Delete an entity by its name from the vector storage."""

        def _sync_delete_entity():
            # Import here to avoid circular imports
            from aperag.db.ops import db_ops

            try:
                # Use the new delete by name method
                deleted_count = db_ops.delete_lightrag_vdb_entity_by_name(self.workspace, entity_name)
                if deleted_count > 0:
                    logger.debug(f"Successfully deleted entity {entity_name}")
                else:
                    logger.debug(f"Entity {entity_name} not found")
            except Exception as e:
                logger.error(f"Error deleting entity {entity_name}: {e}")

        await asyncio.to_thread(_sync_delete_entity)

    async def delete_entity_relation(self, entity_name: str) -> None:
        """Delete all relations associated with an entity."""

        def _sync_delete_entity_relation():
            # Import here to avoid circular imports
            from aperag.db.ops import db_ops

            try:
                # Use the new delete relations by entity method
                deleted_count = db_ops.delete_lightrag_vdb_relation_by_entity(self.workspace, entity_name)
                logger.debug(f"Successfully deleted {deleted_count} relations for entity {entity_name}")
            except Exception as e:
                logger.error(f"Error deleting relations for entity {entity_name}: {e}")

        await asyncio.to_thread(_sync_delete_entity_relation)

    async def get_by_id(self, id: str) -> dict[str, Any] | None:
        """Get vector data by its ID"""

        def _sync_get_by_id():
            # Import here to avoid circular imports
            from aperag.db.ops import db_ops
            from aperag.graph.lightrag.namespace import NameSpace, is_namespace

            if is_namespace(self.namespace, NameSpace.VECTOR_STORE_CHUNKS):
                model = db_ops.query_lightrag_doc_chunks_by_id(self.workspace, id)
                if not model:
                    return None
                return {
                    "id": model.id,
                    "tokens": model.tokens,
                    "content": model.content or "",
                    "chunk_order_index": model.chunk_order_index,
                    "full_doc_id": model.full_doc_id,
                    "content_vector": model.content_vector,
                    "file_path": model.file_path,
                    "created_at": int(model.create_time.timestamp()) if model.create_time else None,
                }
            elif is_namespace(self.namespace, NameSpace.VECTOR_STORE_ENTITIES):
                model = db_ops.query_lightrag_vdb_entity_by_id(self.workspace, id)
                if not model:
                    return None
                return {
                    "id": model.id,
                    "entity_name": model.entity_name,
                    "content": model.content or "",
                    "content_vector": model.content_vector,
                    "chunk_ids": model.chunk_ids,
                    "file_path": model.file_path,
                    "created_at": int(model.create_time.timestamp()) if model.create_time else None,
                }
            elif is_namespace(self.namespace, NameSpace.VECTOR_STORE_RELATIONSHIPS):
                model = db_ops.query_lightrag_vdb_relation_by_id(self.workspace, id)
                if not model:
                    return None
                return {
                    "id": model.id,
                    "source_id": model.source_id,
                    "target_id": model.target_id,
                    "content": model.content or "",
                    "content_vector": model.content_vector,
                    "chunk_ids": model.chunk_ids,
                    "file_path": model.file_path,
                    "created_at": int(model.create_time.timestamp()) if model.create_time else None,
                }
            else:
                logger.error(f"Unknown namespace for ID lookup: {self.namespace}")
                return None

        return await asyncio.to_thread(_sync_get_by_id)

    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        """Get multiple vector data by their IDs"""

        def _sync_get_by_ids():
            if not ids:
                return []

            # Import here to avoid circular imports
            from aperag.db.ops import db_ops
            from aperag.graph.lightrag.namespace import NameSpace, is_namespace

            if is_namespace(self.namespace, NameSpace.VECTOR_STORE_CHUNKS):
                models = db_ops.query_lightrag_doc_chunks_by_ids(self.workspace, ids)
                return [
                    {
                        "id": model.id,
                        "tokens": model.tokens,
                        "content": model.content or "",
                        "chunk_order_index": model.chunk_order_index,
                        "full_doc_id": model.full_doc_id,
                        "content_vector": model.content_vector,
                        "file_path": model.file_path,
                        "created_at": int(model.create_time.timestamp()) if model.create_time else None,
                    }
                    for model in models
                ]
            elif is_namespace(self.namespace, NameSpace.VECTOR_STORE_ENTITIES):
                # Use the new batch query method for entities
                models = db_ops.query_lightrag_vdb_entity_by_ids(self.workspace, ids)
                return [
                    {
                        "id": model.id,
                        "entity_name": model.entity_name,
                        "content": model.content or "",
                        "content_vector": model.content_vector,
                        "chunk_ids": model.chunk_ids,
                        "file_path": model.file_path,
                        "created_at": int(model.create_time.timestamp()) if model.create_time else None,
                    }
                    for model in models
                ]
            elif is_namespace(self.namespace, NameSpace.VECTOR_STORE_RELATIONSHIPS):
                # Use the new batch query method for relations
                models = db_ops.query_lightrag_vdb_relation_by_ids(self.workspace, ids)
                return [
                    {
                        "id": model.id,
                        "source_id": model.source_id,
                        "target_id": model.target_id,
                        "content": model.content or "",
                        "content_vector": model.content_vector,
                        "chunk_ids": model.chunk_ids,
                        "file_path": model.file_path,
                        "created_at": int(model.create_time.timestamp()) if model.create_time else None,
                    }
                    for model in models
                ]
            else:
                logger.error(f"Unknown namespace for IDs lookup: {self.namespace}")
                return []

        return await asyncio.to_thread(_sync_get_by_ids)

    async def drop(self) -> dict[str, str]:
        """Drop the storage - not implemented for safety"""
        return {"status": "error", "message": "Drop operation not supported for database-backed storage"}
