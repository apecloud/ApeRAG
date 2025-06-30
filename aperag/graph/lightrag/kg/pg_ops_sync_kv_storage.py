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
from dataclasses import dataclass
from typing import Any, final

from ..base import (
    BaseKVStorage,
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
