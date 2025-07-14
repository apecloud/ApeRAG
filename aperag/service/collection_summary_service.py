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

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from aperag.db.models import Collection, CollectionStatus, DocumentIndex, DocumentIndexStatus, DocumentIndexType
from aperag.db.ops import async_db_ops
from aperag.index.summary_index import SummaryIndexer
from aperag.llm.completion.base_completion import get_collection_completion_service_sync
from aperag.schema.utils import parseCollectionConfig

logger = logging.getLogger(__name__)


class CollectionSummaryService:
    """Service for managing collection summaries using map-reduce strategy"""

    def __init__(self):
        self.summary_indexer = SummaryIndexer()

    async def get_collection_with_summary(self, session: AsyncSession, collection_id: str) -> Optional[Collection]:
        """Get collection including its summary field"""
        result = await session.execute(
            select(Collection).where(Collection.id == collection_id, Collection.gmt_deleted.is_(None))
        )
        return result.scalar_one_or_none()

    async def trigger_collection_summary_generation(self, collection_id: str) -> bool:
        """
        Trigger collection summary generation as background task with mutex

        Returns:
            bool: True if task was triggered, False if already running
        """
        # Check current collection status

        async def _get_collection(session: AsyncSession):
            result = await session.execute(
                select(Collection).where(Collection.id == collection_id, Collection.gmt_deleted.is_(None))
            )
            return result.scalar_one_or_none()

        collection = await async_db_ops._execute_query(_get_collection)

        if not collection:
            raise ValueError(f"Collection {collection_id} not found")

        # Check if summary is enabled in collection config
        config = parseCollectionConfig(collection.config)
        if not config.enable_summary:
            logger.info(f"Collection {collection_id} has summary disabled in configuration")
            return False

        # Check if summary generation is already in progress
        if collection.status == CollectionStatus.SUMMARY_GENERATING:
            logger.info(f"Collection {collection_id} summary generation already in progress")
            return False

        # Set status to SUMMARY_GENERATING for mutex
        async def _update_collection_status(session: AsyncSession):
            await session.execute(
                update(Collection).where(Collection.id == collection_id).values(status=CollectionStatus.SUMMARY_GENERATING)
            )

        await async_db_ops.execute_with_transaction(_update_collection_status)

        # Import and trigger async task
        try:
            # Schedule background task (non-blocking)
            asyncio.create_task(self._generate_collection_summary_task(collection_id))
            logger.info(f"Collection summary generation task triggered for {collection_id}")
            return True
        except Exception as e:
            # Rollback status on error
            async def _rollback_collection_status(session: AsyncSession):
                await session.execute(
                    update(Collection).where(Collection.id == collection_id).values(status=CollectionStatus.ACTIVE)
                )

            await async_db_ops.execute_with_transaction(_rollback_collection_status)
            logger.error(f"Failed to trigger collection summary generation for {collection_id}: {e}")
            raise

    async def _generate_collection_summary_task(self, collection_id: str):
        """Background task to generate collection summary using map-reduce strategy"""
        try:
            logger.info(f"Starting collection summary generation for {collection_id}")

            # Get collection
            async def _get_collection(session: AsyncSession):
                result = await session.execute(
                    select(Collection).where(Collection.id == collection_id, Collection.gmt_deleted.is_(None))
                )
                return result.scalar_one_or_none()

            collection = await async_db_ops._execute_query(_get_collection)

            if not collection:
                logger.error(f"Collection {collection_id} not found during summary generation")
                return

            # Get collection configuration
            config = parseCollectionConfig(collection.config)
            completion_service = get_collection_completion_service_sync(config)

            if not completion_service:
                logger.warning(f"No completion service available for collection {collection_id}")
                await self._update_collection_status(collection_id, CollectionStatus.ACTIVE)
                return

            # Step 1: Get all document summaries (Map phase)
            document_summaries = await self._get_all_document_summaries(collection_id)

            if not document_summaries:
                logger.info(f"No document summaries found for collection {collection_id}")
                await self._update_collection_status(collection_id, CollectionStatus.ACTIVE)
                return

            # Step 2: Generate collection summary using map-reduce (Reduce phase)
            collection_summary = await self._reduce_document_summaries(
                completion_service, document_summaries, collection.title
            )

            # Step 3: Update collection with generated summary
            async def _update_collection_summary(session: AsyncSession):
                await session.execute(
                    update(Collection)
                    .where(Collection.id == collection_id)
                    .values(summary=collection_summary, status=CollectionStatus.ACTIVE)
                )

            await async_db_ops.execute_with_transaction(_update_collection_summary)

            logger.info(f"Collection summary generated successfully for {collection_id}")

        except Exception as e:
            logger.error(f"Error generating collection summary for {collection_id}: {e}")
            # Reset status on error
            await self._update_collection_status(collection_id, CollectionStatus.ACTIVE)

    async def _get_all_document_summaries(self, collection_id: str) -> List[Dict[str, Any]]:
        """Get all document summaries for the collection (Map phase)"""
        from aperag.db.models import Document

        # Get all documents with active summary indexes
        # First, get all document IDs that belong to this collection
        async def _get_document_ids(session: AsyncSession):
            doc_result = await session.execute(
                select(Document.id).where(Document.collection_id == collection_id, Document.gmt_deleted.is_(None))
            )
            return [row[0] for row in doc_result.fetchall()]

        document_ids = await async_db_ops._execute_query(_get_document_ids)

        if not document_ids:
            return []

        # Get summary indexes for these documents
        async def _get_summary_indexes(session: AsyncSession):
            result = await session.execute(
                select(DocumentIndex).where(
                    DocumentIndex.document_id.in_(document_ids),
                    DocumentIndex.index_type == DocumentIndexType.SUMMARY,
                    DocumentIndex.status == DocumentIndexStatus.ACTIVE,
                )
            )
            return result.scalars().all()

        summary_indexes = await async_db_ops._execute_query(_get_summary_indexes)
        document_summaries = []

        for summary_index in summary_indexes:
            try:
                # Get document summary from index data
                if summary_index.index_data:
                    index_data = json.loads(summary_index.index_data)
                    summary = index_data.get("summary")
                    if summary:
                        document_summaries.append({"document_id": summary_index.document_id, "summary": summary})
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Failed to parse summary for document {summary_index.document_id}: {e}")
                continue

        return document_summaries

    async def _reduce_document_summaries(
        self, completion_service, document_summaries: List[Dict[str, Any]], collection_title: str
    ) -> str:
        """Reduce multiple document summaries into a single collection summary (Reduce phase)"""
        # If we have many summaries, we might need to do hierarchical reduction
        if len(document_summaries) > 20:
            return await self._hierarchical_reduce(completion_service, document_summaries, collection_title)
        else:
            return await self._simple_reduce(completion_service, document_summaries, collection_title)

    async def _simple_reduce(
        self, completion_service, document_summaries: List[Dict[str, Any]], collection_title: str
    ) -> str:
        """Simple reduction for small number of documents"""
        summaries_text = "\n\n".join(
            [f"Document {i + 1}: {doc['summary']}" for i, doc in enumerate(document_summaries)]
        )

        prompt = f"""You are tasked with creating a comprehensive summary of a document collection titled "{collection_title}".

Below are summaries of individual documents in this collection:

{summaries_text}

Please create a concise but comprehensive summary of the entire collection that:
1. Captures the main themes and topics covered across all documents
2. Highlights key insights and important information
3. Maintains logical flow and coherence
4. Is suitable for helping users understand what this collection contains

Collection Summary:"""

        try:
            response = await completion_service.acomplete(prompt)
            return response.text.strip()
        except Exception as e:
            logger.error(f"Error generating collection summary: {e}")
            raise

    async def _hierarchical_reduce(
        self, completion_service, document_summaries: List[Dict[str, Any]], collection_title: str
    ) -> str:
        """Hierarchical reduction for large number of documents"""
        # Group summaries into chunks of 15
        chunk_size = 15
        intermediate_summaries = []

        for i in range(0, len(document_summaries), chunk_size):
            chunk = document_summaries[i : i + chunk_size]
            chunk_summary = await self._simple_reduce(
                completion_service, chunk, f"{collection_title} (Part {i // chunk_size + 1})"
            )
            intermediate_summaries.append({"summary": chunk_summary})

        # Reduce intermediate summaries
        return await self._simple_reduce(completion_service, intermediate_summaries, collection_title)

    async def _update_collection_status(self, collection_id: str, status: CollectionStatus):
        """Helper method to update collection status"""
        async def _update_collection_status(session: AsyncSession):
            await session.execute(update(Collection).where(Collection.id == collection_id).values(status=status))
        await async_db_ops.execute_with_transaction(_update_collection_status)


# Global service instance
collection_summary_service = CollectionSummaryService()
