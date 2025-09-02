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
from typing import List, Optional

from sqlalchemy import and_, func, select

from aperag.db.models import Document, DocumentIndex, DocumentIndexStatus, DocumentIndexType, DocumentStatus
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

    async def query_documents_with_failed_indexes(
        self, user_id: str, collection_id: str, index_types: Optional[List[DocumentIndexType]] = None
    ) -> List[tuple[str, List[DocumentIndexType]]]:
        """
        Query documents that have failed indexes in a collection.

        Args:
            user_id: User ID
            collection_id: Collection ID
            index_types: Optional filter for specific index types

        Returns:
            List of tuples: (document_id, list_of_failed_index_types)
        """

        async def _query(session):
            # Build the base query
            stmt = (
                select(Document.id, DocumentIndex.index_type)
                .join(DocumentIndex, Document.id == DocumentIndex.document_id)
                .where(
                    and_(
                        Document.user == user_id,
                        Document.collection_id == collection_id,
                        Document.status != DocumentStatus.DELETED,
                        DocumentIndex.status == DocumentIndexStatus.FAILED,
                    )
                )
            )

            # Apply index type filter if provided
            if index_types:
                stmt = stmt.where(DocumentIndex.index_type.in_(index_types))

            result = await session.execute(stmt)
            rows = result.fetchall()

            # Group by document_id
            doc_failed_indexes = {}
            for doc_id, index_type in rows:
                if doc_id not in doc_failed_indexes:
                    doc_failed_indexes[doc_id] = []
                doc_failed_indexes[doc_id].append(index_type)

            return [(doc_id, failed_types) for doc_id, failed_types in doc_failed_indexes.items()]

        return await self._execute_query(_query)
