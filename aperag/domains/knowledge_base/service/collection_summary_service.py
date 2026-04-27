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

import logging
from typing import Any, Callable, Dict, List, Optional

from sqlalchemy import and_, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session

from aperag.config import get_async_session, get_sync_session
from aperag.db.ops import db_ops

# Wave 3 T3.1 chunk 2: ``SummaryIndexer`` (legacy
# ``aperag/domains/indexing/summary_index.py``) was hard-deleted; the
# only reference here was an unused ``self.summary_indexer`` instance
# attribute on ``CollectionSummaryService.__init__``. Removed both.
from aperag.domains.knowledge_base.db.models import (
    Collection,
    CollectionSummary,
    CollectionSummaryStatus,
    Document,
)
from aperag.indexing.models import (
    DocumentIndex,
    IndexStatus,
    Modality,
)
from aperag.llm.completion.base_completion import get_collection_completion_service_sync
from aperag.schema.utils import parseCollectionConfig
from aperag.utils.utils import utc_now

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Collection summary lifecycle callbacks (Wave 3 commit 5 Part 2 chunk 1
# inline migration from legacy ``aperag/tasks/reconciler.py``).
#
# Each callback is the terminal hook that the summary generation task
# invokes on success / failure. They update the ``CollectionSummary``
# row's lifecycle (GENERATING → COMPLETE / FAILED) and, on success
# with the summary feature enabled, propagate the generated text to
# the parent ``Collection.description`` for retrieval-side surfacing.
# All three callbacks are tolerant of token / version mismatches —
# the periodic reconciler may re-issue work in the §F.4 cutover
# transit window and the callback that arrives second silently
# no-ops with a "_describe_summary_callback_mismatch" diagnostic.
# ---------------------------------------------------------------------


class CollectionSummaryCallbacks:
    """Callbacks for collection summary task completion."""

    @staticmethod
    def _describe_summary_callback_mismatch(
        summary_id: str,
        processing_token: str,
        expected_status: CollectionSummaryStatus,
        target_version: int,
    ) -> str:
        try:
            for session in get_sync_session():
                summary_query = select(CollectionSummary).where(CollectionSummary.id == summary_id)
                summary_result = session.execute(summary_query)
                summary_record = summary_result.scalar_one_or_none()
                if not summary_record:
                    return "summary_record_not_found"
                if summary_record.processing_token != processing_token:
                    return "token_mismatch"
                if summary_record.status != expected_status:
                    return f"status_changed_to_{summary_record.status}"
                if summary_record.version != target_version:
                    return f"version_mismatch_expected_{target_version}_current_{summary_record.version}"
                return "unknown_mismatch"
        except Exception:
            logger.exception("Failed to inspect collection summary callback mismatch for %s", summary_id)
        return "unknown_mismatch"

    @staticmethod
    def on_summary_generated(summary_id: str, summary_content: str, target_version: int, processing_token: str):
        """Called when summary generation succeeds.

        Promotes the ``CollectionSummary`` row to ``COMPLETE`` and (if
        the parent collection has summary enabled) writes the generated
        text into ``Collection.description``. Both updates are guarded
        by token / version / unchanged-since-read predicates so a
        stale callback racing a newer generation cannot clobber the
        latest result.
        """
        try:
            for session in get_sync_session():
                summary_query = select(CollectionSummary).where(
                    and_(
                        CollectionSummary.id == summary_id,
                        CollectionSummary.status == CollectionSummaryStatus.GENERATING,
                        CollectionSummary.version == target_version,
                        CollectionSummary.processing_token == processing_token,
                    )
                )
                summary_result = session.execute(summary_query)
                summary_record = summary_result.scalar_one_or_none()

                if not summary_record:
                    reason = CollectionSummaryCallbacks._describe_summary_callback_mismatch(
                        summary_id,
                        processing_token,
                        CollectionSummaryStatus.GENERATING,
                        target_version,
                    )
                    logger.warning(
                        "Summary completion callback ignored for %s (v%s) - %s",
                        summary_id,
                        target_version,
                        reason,
                    )
                    return

                collection_id = summary_record.collection_id

                collection_query = select(Collection).where(
                    and_(Collection.id == collection_id, Collection.gmt_deleted.is_(None))
                )
                collection_result = session.execute(collection_query)
                collection_record = collection_result.scalar_one_or_none()

                if not collection_record:
                    logger.error(f"Collection {collection_id} not found during summary completion")
                    return

                try:
                    config = parseCollectionConfig(collection_record.config)
                    is_summary_enabled = config.enable_summary
                except Exception as e:
                    logger.error(f"Failed to parse collection config for {collection_id}: {e}")
                    is_summary_enabled = False

                current_time = utc_now()
                collection_updated_time = collection_record.gmt_updated

                summary_update_stmt = (
                    update(CollectionSummary)
                    .where(
                        and_(
                            CollectionSummary.id == summary_id,
                            CollectionSummary.status == CollectionSummaryStatus.GENERATING,
                            CollectionSummary.version == target_version,
                            CollectionSummary.processing_token == processing_token,
                        )
                    )
                    .values(
                        status=CollectionSummaryStatus.COMPLETE,
                        summary=summary_content,
                        error_message=None,
                        observed_version=target_version,
                        processing_token=None,
                        lease_expires_at=None,
                        gmt_updated=current_time,
                    )
                )
                summary_update_result = session.execute(summary_update_stmt)

                if summary_update_result.rowcount == 0:
                    session.rollback()
                    reason = CollectionSummaryCallbacks._describe_summary_callback_mismatch(
                        summary_id,
                        processing_token,
                        CollectionSummaryStatus.GENERATING,
                        target_version,
                    )
                    logger.warning(
                        "Summary completion callback ignored for %s (v%s) - %s",
                        summary_id,
                        target_version,
                        reason,
                    )
                    return

                if is_summary_enabled and summary_content:
                    collection_update_stmt = (
                        update(Collection)
                        .where(
                            and_(
                                Collection.id == collection_id,
                                Collection.gmt_updated == collection_updated_time,
                                Collection.gmt_deleted.is_(None),
                            )
                        )
                        .values(
                            description=summary_content,
                            gmt_updated=current_time,
                        )
                    )
                    collection_update_result = session.execute(collection_update_stmt)

                    if collection_update_result.rowcount > 0:
                        logger.info(f"Updated collection {collection_id} description with generated summary")
                    else:
                        logger.warning(
                            f"Failed to update collection {collection_id} description - "
                            "collection may have been modified concurrently"
                        )

                session.commit()
                logger.info(f"Collection summary generation completed for {summary_id} (v{target_version})")

        except Exception as e:
            logger.error(f"Failed to update collection summary completion for {summary_id}: {e}")
            try:
                session.rollback()
            except Exception:
                pass

    @staticmethod
    def on_summary_failed(summary_id: str, error_message: str, target_version: int, processing_token: str):
        """Called when summary generation fails.

        Transitions the row to ``FAILED`` with the error message;
        token / version mismatch silently no-ops so a late callback
        does not overwrite a successful retry's result.
        """
        try:
            for session in get_sync_session():
                update_stmt = (
                    update(CollectionSummary)
                    .where(
                        and_(
                            CollectionSummary.id == summary_id,
                            CollectionSummary.status == CollectionSummaryStatus.GENERATING,
                            CollectionSummary.version == target_version,
                            CollectionSummary.processing_token == processing_token,
                        )
                    )
                    .values(
                        status=CollectionSummaryStatus.FAILED,
                        error_message=error_message,
                        processing_token=None,
                        lease_expires_at=None,
                        gmt_updated=utc_now(),
                    )
                )
                result = session.execute(update_stmt)
                if result.rowcount > 0:
                    session.commit()
                    logger.error(
                        f"Collection summary generation failed for {summary_id} (v{target_version}): {error_message}"
                    )
                else:
                    session.rollback()
                    reason = CollectionSummaryCallbacks._describe_summary_callback_mismatch(
                        summary_id,
                        processing_token,
                        CollectionSummaryStatus.GENERATING,
                        target_version,
                    )
                    logger.warning(
                        "Summary failure callback ignored for %s (v%s) - %s",
                        summary_id,
                        target_version,
                        reason,
                    )
        except Exception as e:
            logger.error(f"Failed to update collection summary failure for {summary_id}: {e}")


# Module-level singleton mirrors the legacy
# ``aperag.tasks.reconciler.collection_summary_callbacks`` so callers
# can swap the import path without changing the call shape.
collection_summary_callbacks = CollectionSummaryCallbacks()


class CollectionSummaryService:
    """Service for managing collection summaries using reconcile strategy"""

    def __init__(self):
        pass

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

    def generate_collection_summary_task(
        self,
        summary_id: str,
        collection_id: str,
        target_version: int,
        processing_token: str,
        callback_allowed: Optional[Callable[[], bool]] = None,
    ):
        """Background task to generate collection summary using map-reduce strategy"""
        callback_allowed = callback_allowed or (lambda: True)

        def can_emit_callback(callback_type: str) -> bool:
            if callback_allowed():
                return True
            logger.warning(
                "Skipping collection summary %s callback for %s because ownership was lost",
                callback_type,
                summary_id,
            )
            return False

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
                if can_emit_callback("failure"):
                    CollectionSummaryCallbacks.on_summary_failed(
                        summary_id,
                        "Collection not found",
                        target_version,
                        processing_token,
                    )
                return

            if not summary:
                logger.error(f"CollectionSummary {summary_id} not found during summary generation")
                return

            if (
                summary.status != CollectionSummaryStatus.GENERATING
                or summary.version != target_version
                or summary.processing_token != processing_token
            ):
                raise Exception(
                    "CollectionSummary "
                    f"{summary_id} state mismatch, Status: {summary.status}, "
                    f"Version: {summary.version}, Token: {summary.processing_token}, "
                    f"Target: {target_version}, retry... "
                )

            completion_service = get_collection_completion_service_sync(collection)

            if not completion_service:
                logger.warning(f"No completion service available for collection {collection_id}")
                if can_emit_callback("failure"):
                    CollectionSummaryCallbacks.on_summary_failed(
                        summary_id,
                        "No completion service available",
                        target_version,
                        processing_token,
                    )
                return

            document_summaries = self._get_all_document_summaries(collection_id)

            if not document_summaries:
                logger.info(f"No document summaries found for collection {collection_id}")
                if can_emit_callback("success"):
                    CollectionSummaryCallbacks.on_summary_generated(
                        summary_id,
                        "",
                        target_version,
                        processing_token,
                    )  # TODO: should we return empty string?
                return

            collection_summary_text = self._reduce_document_summaries(
                completion_service, document_summaries, collection.title
            )

            if can_emit_callback("success"):
                CollectionSummaryCallbacks.on_summary_generated(
                    summary_id,
                    collection_summary_text,
                    target_version,
                    processing_token,
                )
            logger.info(f"Collection summary generated successfully for summary {summary_id} (v{target_version})")

        except Exception as e:
            logger.error(f"Error generating collection summary for {summary_id}: {e}", exc_info=True)
            if can_emit_callback("failure"):
                CollectionSummaryCallbacks.on_summary_failed(
                    summary_id,
                    str(e),
                    target_version,
                    processing_token,
                )

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

        # Wave 3 §F.1 schema migration: legacy
        # ``DocumentIndex.index_data`` JSON blob is gone — per-modality
        # summary text now lives in the ``derived/parse_<v>/
        # summary.json`` artifact on the object store, addressed by
        # the row's ``derived_artifact_path``. Reading the actual
        # summary text per document requires hitting the object store.
        # That object-store read path is a chenyexuan T3.1 commit 4b
        # follow-up (the collection_summary lane is Pattern A/B/C
        # work); for now we return an empty list so callers see
        # "no per-document summaries available" — a degraded but
        # safe behaviour (collection-level summaries that this
        # service rolls up will be regenerated next cycle once the
        # object-store read path lands).
        def _get_summary_indexes(session: Session):
            result = session.execute(
                select(DocumentIndex).where(
                    DocumentIndex.document_id.in_(document_ids),
                    DocumentIndex.modality == Modality.SUMMARY.value,
                    DocumentIndex.status == IndexStatus.ACTIVE.value,
                    DocumentIndex.is_serving.is_(True),
                )
            )
            return result.scalars().all()

        # Run the query so the §F.1 partial-unique invariant is
        # exercised through this caller path; the row pointers are
        # the seam into the object-store read path that T3.1
        # commit 4b will wire.
        _ = db_ops._execute_query(_get_summary_indexes)
        document_summaries: list[dict] = []
        return document_summaries

    def _reduce_document_summaries(
        self, completion_service, document_summaries: List[Dict[str, Any]], collection_title: str
    ) -> str:
        """Simple reduction for small number of documents"""
        summaries_text = "\n\n".join(
            [f"Document {i + 1}: {doc['summary']}" for i, doc in enumerate(document_summaries)]
        )

        prompt = f"""You are tasked with creating a concise summary of a document collection titled "{collection_title}".

Below are summaries of individual documents in this collection:

{summaries_text}

Please create a brief and focused summary of the entire collection that:
1. Captures the main themes and topics covered across all documents
2. Highlights the most important insights and key information
3. Maintains logical flow and coherence
4. Is suitable for helping users quickly understand what this collection contains

IMPORTANT: Your summary must be no more than 10 sentences. Focus on the most essential information and avoid redundancy.

Collection Summary:"""

        try:
            response = completion_service.generate(history=[], prompt=prompt)
            return response.strip()
        except Exception as e:
            logger.error(f"Error generating collection summary: {e}")
            raise

    def _hierarchical_reduce(
        self, completion_service, document_summaries: List[Dict[str, Any]], collection_title: str
    ) -> str:
        """Hierarchical reduction for large number of documents"""
        # Group summaries into chunks of 15
        chunk_size = 15
        intermediate_summaries = []

        for i in range(0, len(document_summaries), chunk_size):
            chunk = document_summaries[i : i + chunk_size]
            chunk_summary = self._reduce_document_summaries(
                completion_service, chunk, f"{collection_title} (Part {i // chunk_size + 1})"
            )
            intermediate_summaries.append({"summary": chunk_summary})

        # Reduce intermediate summaries
        return self._reduce_document_summaries(completion_service, intermediate_summaries, collection_title)


# Global service instances
collection_summary_service = CollectionSummaryService()
