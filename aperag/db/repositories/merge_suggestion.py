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

from typing import List, Optional

from sqlalchemy import and_, delete, func, select

from aperag.db.models import MergeSuggestion, MergeSuggestionHistory, MergeSuggestionStatus
from aperag.db.repositories.base import AsyncRepositoryProtocol
from aperag.utils.utils import utc_now


class MergeSuggestionRepository(AsyncRepositoryProtocol):
    """
    Unified repository for merge suggestions with clean 2-table design:
    - graph_index_merge_suggestions: Active suggestions (PENDING only)
    - graph_index_merge_suggestions_history: Processed suggestions (ACCEPTED/REJECTED)
    """

    # ========== Active Suggestions (Main Table) ==========

    async def get_active_suggestions(self, collection_id: str) -> List[MergeSuggestion]:
        """Get all active suggestions for a collection"""

        async def _query(session):
            stmt = (
                select(MergeSuggestion)
                .where(MergeSuggestion.collection_id == collection_id)
                .order_by(MergeSuggestion.confidence_score.desc())
            )
            result = await session.execute(stmt)
            return result.scalars().all()

        return await self._execute_query(_query)

    async def get_active_suggestion_by_id(self, suggestion_id: str) -> Optional[MergeSuggestion]:
        """Get an active suggestion by ID"""

        async def _query(session):
            stmt = select(MergeSuggestion).where(MergeSuggestion.id == suggestion_id)
            result = await session.execute(stmt)
            return result.scalar_one_or_none()

        return await self._execute_query(_query)

    async def create_active_suggestions(self, suggestions: List[dict]) -> List[MergeSuggestion]:
        """Create new active suggestions"""

        async def _operation(session):
            suggestion_batch_id = f"batch{self._generate_random_id()}"
            suggestion_records = []

            for suggestion in suggestions:
                entity_ids_hash = MergeSuggestion.generate_entity_ids_hash(suggestion["entity_ids"])
                suggestion_record = MergeSuggestion(
                    collection_id=suggestion["collection_id"],
                    suggestion_batch_id=suggestion_batch_id,
                    entity_ids=suggestion["entity_ids"],
                    entity_ids_hash=entity_ids_hash,
                    confidence_score=suggestion["confidence_score"],
                    merge_reason=suggestion["merge_reason"],
                    suggested_target_entity=suggestion["suggested_target_entity"],
                    status=MergeSuggestionStatus.PENDING,  # Always PENDING for active suggestions
                )
                suggestion_records.append(suggestion_record)

            for record in suggestion_records:
                session.add(record)

            await session.flush()
            for record in suggestion_records:
                await session.refresh(record)
            return suggestion_records

        return await self.execute_with_transaction(_operation)

    async def clear_active_suggestions(self, collection_id: str) -> int:
        """Clear all active suggestions for a collection"""

        async def _operation(session):
            stmt = delete(MergeSuggestion).where(MergeSuggestion.collection_id == collection_id)
            result = await session.execute(stmt)
            await session.flush()
            return result.rowcount

        return await self.execute_with_transaction(_operation)

    # ========== History Management ==========

    async def move_to_history(
        self, suggestion: MergeSuggestion, final_status: MergeSuggestionStatus, operated_by: str = None
    ) -> MergeSuggestionHistory:
        """Move an active suggestion to history table and remove from active"""

        async def _operation(session):
            # Create history record
            history_record = MergeSuggestionHistory(
                original_suggestion_id=suggestion.id,
                collection_id=suggestion.collection_id,
                suggestion_batch_id=suggestion.suggestion_batch_id,
                entity_ids=suggestion.entity_ids,
                entity_ids_hash=suggestion.entity_ids_hash,
                confidence_score=suggestion.confidence_score,
                merge_reason=suggestion.merge_reason,
                suggested_target_entity=suggestion.suggested_target_entity,
                status=final_status,
                gmt_created=suggestion.gmt_created,
                operated_at=utc_now(),
                operated_by=operated_by,
            )
            session.add(history_record)

            # Delete from active suggestions
            delete_stmt = delete(MergeSuggestion).where(MergeSuggestion.id == suggestion.id)
            await session.execute(delete_stmt)

            await session.flush()
            await session.refresh(history_record)
            return history_record

        return await self.execute_with_transaction(_operation)

    # ========== History Queries ==========

    async def get_suggestion_history(
        self,
        collection_id: str,
        status_filter: List[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[MergeSuggestionHistory]:
        """Get suggestion history for a collection"""

        async def _query(session):
            stmt = select(MergeSuggestionHistory).where(MergeSuggestionHistory.collection_id == collection_id)

            if status_filter:
                stmt = stmt.where(MergeSuggestionHistory.status.in_(status_filter))

            stmt = stmt.order_by(MergeSuggestionHistory.operated_at.desc()).limit(limit).offset(offset)

            result = await session.execute(stmt)
            return result.scalars().all()

        return await self._execute_query(_query)

    async def get_history_stats(self, collection_id: str) -> dict:
        """Get history statistics by status"""

        async def _query(session):
            stmt = (
                select(MergeSuggestionHistory.status, func.count(MergeSuggestionHistory.id))
                .where(MergeSuggestionHistory.collection_id == collection_id)
                .group_by(MergeSuggestionHistory.status)
            )

            result = await session.execute(stmt)
            return {status: count for status, count in result.fetchall()}

        return await self._execute_query(_query)

    # ========== Utility Methods ==========

    def _generate_random_id(self) -> str:
        """Generate a random ID for batch operations"""
        import random
        import uuid

        return "".join(random.sample(uuid.uuid4().hex, 16))

    async def has_active_suggestions(self, collection_id: str) -> bool:
        """Check if there are any active suggestions"""

        async def _query(session):
            from sqlalchemy import exists

            stmt = select(exists().where(MergeSuggestion.collection_id == collection_id))
            result = await session.execute(stmt)
            return result.scalar()

        return await self._execute_query(_query)

    # ========== Cleanup Methods ==========

    async def cleanup_old_history(self, collection_id: str, days_to_keep: int = 90) -> int:
        """Clean up old history records (optional maintenance)"""
        from datetime import timedelta

        cutoff_date = utc_now() - timedelta(days=days_to_keep)

        async def _operation(session):
            stmt = delete(MergeSuggestionHistory).where(
                and_(
                    MergeSuggestionHistory.collection_id == collection_id,
                    MergeSuggestionHistory.operated_at < cutoff_date,
                )
            )
            result = await session.execute(stmt)
            await session.flush()
            return result.rowcount

        return await self.execute_with_transaction(_operation)


# Singleton instance
merge_suggestion_repo = MergeSuggestionRepository()
