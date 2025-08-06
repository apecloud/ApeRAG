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

from datetime import datetime

from sqlalchemy import and_, func, select

from aperag.db.models import Document, DocumentIndex, DocumentIndexStatus, DocumentIndexType
from aperag.db.repositories.base import AsyncRepositoryProtocol


class AsyncDocumentIndexRepositoryMixin(AsyncRepositoryProtocol):
    """Repository mixin for DocumentIndex operations"""

    async def has_recent_graph_index_updates(self, collection_id: str, since_time: datetime) -> int:
        """Count the number of successful graph index updates since a given time."""

        async def _query(session):
            stmt = select(func.count()).where(
                and_(
                    Document.id == DocumentIndex.document_id,
                    Document.collection_id == collection_id,
                    DocumentIndex.index_type == DocumentIndexType.GRAPH,
                    DocumentIndex.status == DocumentIndexStatus.ACTIVE,
                    DocumentIndex.gmt_updated > since_time,
                )
            )
            result = await session.execute(stmt)
            return result.scalar()

        return await self._execute_query(_query)
