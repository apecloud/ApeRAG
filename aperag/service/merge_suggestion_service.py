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
from typing import Any, Dict, List, Optional

from aperag.db.models import MergeSuggestionStatus
from aperag.db.ops import get_async_db_ops

logger = logging.getLogger(__name__)


class MergeSuggestionService:
    """Service for managing merge suggestions"""

    def __init__(self):
        self.db_ops = get_async_db_ops()

    async def get_valid_suggestions(self, collection_id: str) -> List:
        """Get all valid (non-expired) suggestions for a collection"""
        from aperag.db.repositories.merge_suggestion import MergeSuggestionRepository

        session = self.db_ops.get_session()
        suggestion_repo = MergeSuggestionRepository(session)
        return await suggestion_repo.get_valid_suggestions(collection_id)

    async def batch_create_suggestions(self, suggestion_data: List[Dict[str, Any]]) -> List:
        """Batch create suggestions"""
        from aperag.db.repositories.merge_suggestion import MergeSuggestionRepository

        session = self.db_ops.get_session()
        suggestion_repo = MergeSuggestionRepository(session)
        return await suggestion_repo.batch_create(suggestion_data)

    async def get_suggestions_by_ids(self, suggestion_ids: List[str]) -> List:
        """Get suggestions by their IDs"""
        from aperag.db.repositories.merge_suggestion import MergeSuggestionRepository

        session = self.db_ops.get_session()
        suggestion_repo = MergeSuggestionRepository(session)
        return await suggestion_repo.get_suggestions_by_ids(suggestion_ids)

    async def update_suggestion_status(
        self, suggestion_id: str, status: MergeSuggestionStatus, operated_at: Optional[Any] = None
    ) -> None:
        """Update suggestion status"""
        from aperag.db.repositories.merge_suggestion import MergeSuggestionRepository

        session = self.db_ops.get_session()
        suggestion_repo = MergeSuggestionRepository(session)
        return await suggestion_repo.update_status(suggestion_id, status, operated_at)

    async def expire_related_suggestions(self, collection_id: str, entity_ids: List[str]) -> None:
        """Expire suggestions related to the given entity IDs"""
        from aperag.db.repositories.merge_suggestion import MergeSuggestionRepository

        session = self.db_ops.get_session()
        suggestion_repo = MergeSuggestionRepository(session)
        return await suggestion_repo.expire_related_suggestions(collection_id, entity_ids)


# Global service instance
merge_suggestion_service = MergeSuggestionService()
