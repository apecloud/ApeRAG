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

import json
import logging
from typing import Any, Dict, List, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session

from aperag.config import get_async_session, get_sync_session
from aperag.db.models import (
    Collection,
    CollectionSummary,
    CollectionSummaryStatus,
    Document,
    DocumentIndex,
    DocumentIndexStatus,
    DocumentIndexType,
)
from aperag.db.ops import db_ops
from aperag.index.summary_index import SummaryIndexer
from aperag.llm.completion.base_completion import get_collection_completion_service_sync
from aperag.schema.utils import parseCollectionConfig
from aperag.tasks.reconciler import CollectionSummaryCallbacks

logger = logging.getLogger(__name__)


class CollectionSummaryService:
    """Service for managing collection summaries using reconcile strategy"""

    def __init__(self):
        self.summary_indexer = SummaryIndexer()

    async def trigger_collection_summary_generation(self, collection: Collection) -> bool:
        """
        Trigger collection summary generation based on collection config.
        If enable_summary is true, create/update CollectionSummary.
        If enable_summary is false, delete CollectionSummary.
        The reconciler will pick it up and schedule the actual task.

        Returns:
            bool: True if task was triggered or state changed, False otherwise.
        """
        async for session in get_async_session():
            config = parseCollectionConfig(collection.config)

            record = await self._get_summary_by_collection_id(session, collection.id)

            if config.enable_summary:
                if record:
                    # If summary exists, update its version to trigger reconciliation
                    if record.status != CollectionSummaryStatus.GENERATING:
                        record.update_version()
                        logger.info(f"Triggered re-generation for CollectionSummary of collection {collection.id}")
                    else:
                        logger.info(f"CollectionSummary for {collection.id} is already being processed.")
                        return False
                else:
                    # If summary does not exist, create a new one
                    record = CollectionSummary(collection_id=collection.id, status=CollectionSummaryStatus.PENDING)
                    session.add(record)
                    logger.info(f"Created new CollectionSummary for collection {collection.id}")
                await session.commit()
                return True
            else:
                # If summary is disabled, delete the summary object
                if record:
                    await session.delete(record)
                    await session.commit()
                    logger.info(f"Deleted CollectionSummary for collection {collection.id} as summary is disabled.")
                    return True
                return False

    async def _get_summary_by_collection_id(
        self, session: AsyncSession, collection_id: str
    ) -> Optional[CollectionSummary]:
        result = await session.execute(
            select(CollectionSummary).where(CollectionSummary.collection_id == collection_id)
        )
        return result.scalar_one_or_none()

    def generate_collection_summary_task(self, summary_id: str, collection_id: str, target_version: int):
        """Background task to generate collection summary using map-reduce strategy"""
        try:
            logger.info(
                f"Starting collection summary generation for summary {summary_id} (collection: {collection_id}, v{target_version})"
            )

            # Get collection
            for session in get_sync_session():
                collection_result = session.execute(
                    select(Collection).where(Collection.id == collection_id, Collection.gmt_deleted.is_(None))
                )
                collection = collection_result.scalar_one_or_none()

                summary_result = session.execute(select(CollectionSummary).where(CollectionSummary.id == summary_id))
                summary = summary_result.scalar_one_or_none()

            if not collection:
                logger.error(f"Collection {collection_id} not found during summary generation")
                CollectionSummaryCallbacks.on_summary_failed(summary_id, "Collection not found", target_version)
                return

            if not summary:
                logger.error(f"CollectionSummary {summary_id} not found during summary generation")
                return

            if summary.status != CollectionSummaryStatus.GENERATING or summary.version != target_version:
                raise Exception(
                    f"CollectionSummary {summary_id} status/version mismatch, Status: {summary.status}, Version: {summary.version}, Target: {target_version}, retry... "
                )

            completion_service = get_collection_completion_service_sync(collection)

            if not completion_service:
                logger.warning(f"No completion service available for collection {collection_id}")
                CollectionSummaryCallbacks.on_summary_failed(
                    summary_id, "No completion service available", target_version
                )
                return

            document_summaries = self._get_all_document_summaries(collection_id)

            if not document_summaries:
                logger.info(f"No document summaries found for collection {collection_id}")
                CollectionSummaryCallbacks.on_summary_generated(
                    summary_id, "", target_version
                )  # TODO: should we return empty string?
                return

            collection_summary_text = self._reduce_document_summaries(
                completion_service, document_summaries, collection.title
            )

            CollectionSummaryCallbacks.on_summary_generated(summary_id, collection_summary_text, target_version)
            logger.info(f"Collection summary generated successfully for summary {summary_id} (v{target_version})")

        except Exception as e:
            logger.error(f"Error generating collection summary for {summary_id}: {e}", exc_info=True)
            CollectionSummaryCallbacks.on_summary_failed(summary_id, str(e), target_version)

    def _get_all_document_summaries(self, collection_id: str) -> List[Dict[str, Any]]:
        """Get all document summaries for the collection (Map phase)"""

        # Get all documents with active summary indexes
        # First, get all document IDs that belong to this collection
        def _get_document_ids(session: Session):
            doc_result = session.execute(
                select(Document.id).where(Document.collection_id == collection_id, Document.gmt_deleted.is_(None))
            )
            return [row[0] for row in doc_result.fetchall()]

        document_ids = db_ops._execute_query(_get_document_ids)

        if not document_ids:
            return []

        # Get summary indexes for these documents
        def _get_summary_indexes(session: Session):
            result = session.execute(
                select(DocumentIndex).where(
                    DocumentIndex.document_id.in_(document_ids),
                    DocumentIndex.index_type == DocumentIndexType.SUMMARY,
                    DocumentIndex.status == DocumentIndexStatus.ACTIVE,
                )
            )
            return result.scalars().all()

        summary_indexes = db_ops._execute_query(_get_summary_indexes)
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

    def _reduce_document_summaries(
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
            response = completion_service.generate(history=[], prompt=prompt)
            return response.strip()
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


# Global service instances
collection_summary_service = CollectionSummaryService()
