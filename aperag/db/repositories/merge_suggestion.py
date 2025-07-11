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

from datetime import datetime, timedelta
from typing import List, Optional

from sqlalchemy import and_, select, update

from aperag.db.models import MergeSuggestion, MergeSuggestionStatus
from aperag.db.repositories.base import AsyncRepositoryProtocol
from aperag.utils.utils import utc_now


class AsyncMergeSuggestionRepositoryMixin(AsyncRepositoryProtocol):
    """Repository mixin for MergeSuggestion operations"""

    async def get_valid_suggestions(self, collection_id: str) -> List[MergeSuggestion]:
        """Get all valid (non-expired, non-deleted) suggestions for a collection"""

        async def _query(session):
            now = utc_now()
            stmt = (
                select(MergeSuggestion)
                .where(
                    and_(
                        MergeSuggestion.collection_id == collection_id,
                        MergeSuggestion.gmt_deleted.is_(None),
                        MergeSuggestion.expires_at > now,
                        MergeSuggestion.status.in_(
                            [
                                MergeSuggestionStatus.PENDING,
                                MergeSuggestionStatus.ACCEPTED,
                                MergeSuggestionStatus.REJECTED,
                            ]
                        ),
                    )
                )
                .order_by(MergeSuggestion.confidence_score.desc())
            )

            result = await session.execute(stmt)
            return result.scalars().all()

        return await self._execute_query(_query)

    async def get_suggestions_by_ids(self, suggestion_ids: List[str]) -> List[MergeSuggestion]:
        """Get suggestions by their IDs"""

        async def _query(session):
            stmt = select(MergeSuggestion).where(
                and_(MergeSuggestion.id.in_(suggestion_ids), MergeSuggestion.gmt_deleted.is_(None))
            )

            result = await session.execute(stmt)
            return result.scalars().all()

        return await self._execute_query(_query)

    async def get_suggestions_containing_entities(
        self, collection_id: str, entity_ids: List[str]
    ) -> List[MergeSuggestion]:
        """Get suggestions that contain any of the specified entities"""

        async def _query(session):
            stmt = select(MergeSuggestion).where(
                and_(
                    MergeSuggestion.collection_id == collection_id,
                    MergeSuggestion.gmt_deleted.is_(None),
                    MergeSuggestion.entity_ids.overlap(entity_ids),  # PostgreSQL array overlap
                )
            )

            result = await session.execute(stmt)
            return result.scalars().all()

        return await self._execute_query(_query)

    async def batch_create_suggestions(self, suggestions: List[dict]) -> List[MergeSuggestion]:
        """Batch create suggestions with the same batch_id"""

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
                    expires_at=suggestion.get("expires_at", utc_now() + timedelta(days=7)),
                )
                suggestion_records.append(suggestion_record)

            for record in suggestion_records:
                session.add(record)

            await session.flush()
            for record in suggestion_records:
                await session.refresh(record)
            return suggestion_records

        return await self.execute_with_transaction(_operation)

    async def update_suggestion_status(
        self, suggestion_id: str, status: MergeSuggestionStatus, operated_at: Optional[datetime] = None
    ) -> bool:
        """Update suggestion status"""

        async def _operation(session):
            update_values = {"status": status, "gmt_updated": utc_now()}

            if operated_at:
                update_values["operated_at"] = operated_at

            stmt = (
                update(MergeSuggestion)
                .where(and_(MergeSuggestion.id == suggestion_id, MergeSuggestion.gmt_deleted.is_(None)))
                .values(**update_values)
            )

            result = await session.execute(stmt)
            await session.flush()
            return result.rowcount > 0

        return await self.execute_with_transaction(_operation)

    async def batch_update_suggestion_status(
        self, suggestion_ids: List[str], status: MergeSuggestionStatus, operated_at: Optional[datetime] = None
    ) -> int:
        """Batch update suggestion status"""

        async def _operation(session):
            update_values = {"status": status, "gmt_updated": utc_now()}

            if operated_at:
                update_values["operated_at"] = operated_at

            stmt = (
                update(MergeSuggestion)
                .where(and_(MergeSuggestion.id.in_(suggestion_ids), MergeSuggestion.gmt_deleted.is_(None)))
                .values(**update_values)
            )

            result = await session.execute(stmt)
            await session.flush()
            return result.rowcount

        return await self.execute_with_transaction(_operation)

    async def expire_related_suggestions(self, collection_id: str, entity_ids: List[str]) -> int:
        """Expire suggestions that contain any of the specified entities"""
        suggestions = await self.get_suggestions_containing_entities(collection_id, entity_ids)

        # Only expire pending suggestions
        pending_suggestion_ids = [s.id for s in suggestions if s.status == MergeSuggestionStatus.PENDING]

        if pending_suggestion_ids:
            return await self.batch_update_suggestion_status(
                pending_suggestion_ids, MergeSuggestionStatus.EXPIRED, utc_now()
            )

        return 0

    async def cleanup_expired_suggestions(self, collection_id: Optional[str] = None) -> int:
        """Clean up expired suggestions (soft delete)"""

        async def _operation(session):
            now = utc_now()

            conditions = [MergeSuggestion.gmt_deleted.is_(None), MergeSuggestion.expires_at <= now]

            if collection_id:
                conditions.append(MergeSuggestion.collection_id == collection_id)

            stmt = update(MergeSuggestion).where(and_(*conditions)).values(gmt_deleted=now, gmt_updated=now)

            result = await session.execute(stmt)
            await session.flush()
            return result.rowcount

        return await self.execute_with_transaction(_operation)

    async def delete_all_suggestions_for_collection(self, collection_id: str) -> int:
        """Physically delete all suggestions for a collection"""

        async def _operation(session):
            from sqlalchemy import delete

            stmt = delete(MergeSuggestion).where(MergeSuggestion.collection_id == collection_id)

            result = await session.execute(stmt)
            await session.flush()
            return result.rowcount

        return await self.execute_with_transaction(_operation)

    def _generate_random_id(self) -> str:
        """Generate a random ID for batch operations"""
        import random
        import uuid

        return "".join(random.sample(uuid.uuid4().hex, 16))


# Keep the original class for backwards compatibility if needed
class MergeSuggestionRepository(AsyncMergeSuggestionRepositoryMixin):
    """Legacy repository class - use AsyncMergeSuggestionRepositoryMixin instead"""

    def __init__(self, session):
        # Initialize the parent mixin properly
        super().__init__()
        self._session = session
