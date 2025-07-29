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

from sqlalchemy import desc, select

from aperag.db.models import Document, DocumentStatus
from aperag.db.repositories.base import (
    AsyncRepositoryProtocol,
    SyncRepositoryProtocol,
)


class DocumentRepositoryMixin(SyncRepositoryProtocol):
    def query_document_by_id(self, document_id: str, ignore_deleted: bool = True) -> Document:
        def _query(session):
            stmt = select(Document).where(Document.id == document_id)
            if ignore_deleted:
                stmt = stmt.where(Document.status != DocumentStatus.DELETED)
            result = session.execute(stmt)
            return result.scalars().first()

        return self._execute_query(_query)

    def update_document(self, document: Document):
        session = self._get_session()
        session.add(document)
        session.commit()
        session.refresh(document)
        return document

    def query_documents(self, users: List[str], collection_id: str):
        """Query documents by users and collection ID (sync version)"""

        def _query(session):
            stmt = (
                select(Document)
                .where(
                    Document.user.in_(users),
                    Document.collection_id == collection_id,
                    Document.status != DocumentStatus.DELETED,
                )
                .order_by(desc(Document.gmt_created))
            )
            result = session.execute(stmt)
            return result.scalars().all()

        return self._execute_query(_query)


class AsyncDocumentRepositoryMixin(AsyncRepositoryProtocol):
    # Document Operations
    async def create_document(
        self, user: str, collection_id: str, name: str, size: int, object_path: str = None, metadata: str = None
    ) -> Document:
        """Create a new document in database"""

        async def _operation(session):
            instance = Document(
                user=user,
                name=name,
                status=DocumentStatus.PENDING,
                size=size,
                collection_id=collection_id,
                object_path=object_path,
                doc_metadata=metadata,
            )
            session.add(instance)
            await session.flush()
            await session.refresh(instance)
            return instance

        return await self.execute_with_transaction(_operation)

    async def update_document_by_id(
        self, user: str, collection_id: str, document_id: str, metadata: str = None
    ) -> Optional[Document]:
        """Update document by ID"""

        async def _operation(session):
            stmt = select(Document).where(
                Document.id == document_id,
                Document.collection_id == collection_id,
                Document.user == user,
                Document.status != DocumentStatus.DELETED,
            )
            result = await session.execute(stmt)
            instance = result.scalars().first()

            if instance and metadata is not None:
                instance.doc_metadata = metadata
                session.add(instance)
                await session.flush()
                await session.refresh(instance)

            return instance

        return await self.execute_with_transaction(_operation)

    async def delete_document_by_id(self, user: str, collection_id: str, document_id: str) -> Optional[Document]:
        """Soft delete document by ID"""
        from aperag.db.models import DocumentStatus

        async def _operation(session):
            stmt = select(Document).where(
                Document.id == document_id,
                Document.collection_id == collection_id,
                Document.user == user,
                Document.status != DocumentStatus.DELETED,
            )
            result = await session.execute(stmt)
            instance = result.scalars().first()

            if instance:
                from aperag.db.models import utc_now

                instance.status = DocumentStatus.DELETED
                instance.gmt_deleted = utc_now()
                session.add(instance)
                await session.flush()
                await session.refresh(instance)

            return instance

        return await self.execute_with_transaction(_operation)

    async def delete_documents_by_ids(self, user: str, collection_id: str, document_ids: List[str]) -> tuple:
        """Bulk soft delete documents by IDs"""
        from aperag.db.models import DocumentStatus

        async def _operation(session):
            stmt = select(Document).where(
                Document.id.in_(document_ids),
                Document.collection_id == collection_id,
                Document.user == user,
                Document.status != DocumentStatus.DELETED,
            )
            result = await session.execute(stmt)
            instances = result.scalars().all()

            if not instances:
                return [], []

            from aperag.db.models import utc_now

            deleted_instances = []
            for instance in instances:
                instance.status = DocumentStatus.DELETED
                instance.gmt_deleted = utc_now()
                session.add(instance)
                deleted_instances.append(instance.id)

            await session.flush()

            # Return the IDs of deleted and not-found documents
            not_found_ids = list(set(document_ids) - set(deleted_instances))
            return deleted_instances, not_found_ids

        return await self.execute_with_transaction(_operation)

    async def query_document(self, user: str, collection_id: str, document_id: str) -> Document:
        async def _query(session):
            stmt = select(Document).where(
                Document.id == document_id,
                Document.collection_id == collection_id,
                Document.user == user,
                Document.status != DocumentStatus.DELETED,
            )
            result = await session.execute(stmt)
            return result.scalars().first()

        return await self._execute_query(_query)

    async def query_documents(self, users: List[str], collection_id: str):
        async def _query(session):
            stmt = select(Document).where(
                Document.user.in_(users),
                Document.collection_id == collection_id,
                Document.status != DocumentStatus.DELETED,
            )
            result = await session.execute(stmt)
            return result.scalars().all()

        return await self._execute_query(_query)

    async def query_documents_count(self, user: str, collection_id: str):
        async def _query(session):
            from sqlalchemy import func

            stmt = (
                select(func.count())
                .select_from(Document)
                .where(
                    Document.user == user,
                    Document.collection_id == collection_id,
                    Document.status != DocumentStatus.DELETED,
                )
            )
            return await session.scalar(stmt)

        return await self._execute_query(_query)
