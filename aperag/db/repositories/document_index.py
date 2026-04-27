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

from aperag.db.repositories.base import AsyncRepositoryProtocol
from aperag.domains.knowledge_base.db.models import (
    Document,
    DocumentStatus,
)
from aperag.indexing.models import (
    DocumentIndex,
    IndexStatus,
    Modality,
)


class AsyncDocumentIndexRepositoryMixin(AsyncRepositoryProtocol):
    """Repository mixin for DocumentIndex operations.

    Wave 3 hard-cut migrated this from the legacy
    ``aperag.domains.indexing.db.models`` schema (with
    ``DocumentIndexType.GRAPH`` enum + ``DocumentIndexStatus.ACTIVE``
    enum + ``gmt_updated`` field) to the §F.1 canonical schema
    (``modality`` column carrying :class:`Modality` enum value +
    ``is_serving`` boolean + ``updated_at`` timestamp). The
    "currently-serving" semantic now requires
    ``status=ACTIVE AND is_serving=TRUE`` per §F.3 cutover model;
    a row at ``status=ACTIVE`` but ``is_serving=FALSE`` is in the
    cutover transit window and is NOT yet user-visible.
    """

    async def has_recent_graph_index_updates(self, collection_id: str, since_time: datetime) -> int:
        """Count the number of successful graph index updates since a given time."""

        async def _query(session):
            stmt = select(func.count()).where(
                and_(
                    Document.id == DocumentIndex.document_id,
                    Document.collection_id == collection_id,
                    DocumentIndex.modality == Modality.GRAPH.value,
                    DocumentIndex.status == IndexStatus.ACTIVE.value,
                    DocumentIndex.is_serving.is_(True),
                    DocumentIndex.updated_at > since_time,
                )
            )
            result = await session.execute(stmt)
            return result.scalar()

        return await self._execute_query(_query)

    async def query_documents_with_failed_indexes(
        self, user_id: str, collection_id: str, index_types: Optional[List[Modality]] = None
    ) -> List[tuple[str, List[str]]]:
        """
        Query documents that have failed indexes in a collection.

        Args:
            user_id: User ID
            collection_id: Collection ID
            index_types: Optional filter for specific :class:`Modality`
                values; pass the modality enum (e.g.,
                ``[Modality.VECTOR, Modality.GRAPH]``).

        Returns:
            List of tuples: ``(document_id,
            list_of_failed_modality_values)``. Each modality value is
            a plain string (``"vector"`` / ``"fulltext"`` / etc.) per
            the §F.1 column type.
        """

        async def _query(session):
            # Build the base query
            stmt = (
                select(Document.id, DocumentIndex.modality)
                .join(DocumentIndex, Document.id == DocumentIndex.document_id)
                .where(
                    and_(
                        Document.user == user_id,
                        Document.collection_id == collection_id,
                        Document.status != DocumentStatus.DELETED,
                        DocumentIndex.status == IndexStatus.FAILED.value,
                    )
                )
            )

            # Apply modality filter if provided.
            if index_types:
                modality_values = [m.value if isinstance(m, Modality) else m for m in index_types]
                stmt = stmt.where(DocumentIndex.modality.in_(modality_values))

            result = await session.execute(stmt)
            rows = result.fetchall()

            # Group by document_id
            doc_failed_indexes: dict[str, list[str]] = {}
            for doc_id, modality_value in rows:
                if doc_id not in doc_failed_indexes:
                    doc_failed_indexes[doc_id] = []
                doc_failed_indexes[doc_id].append(modality_value)

            return [(doc_id, failed_modalities) for doc_id, failed_modalities in doc_failed_indexes.items()]

        return await self._execute_query(_query)
