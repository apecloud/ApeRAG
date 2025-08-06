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

from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession

from aperag.db import models as db_models
from aperag.db.ops import AsyncDatabaseOps, async_db_ops
from aperag.exceptions import (
    CollectionMarketplaceAccessDeniedError,
    CollectionNotFoundException,
    CollectionNotPublishedError,
)
from aperag.schema import view_models


class MarketplaceCollectionService:
    """
    MarketplaceCollection business logic service
    Responsibilities: Handle subscription access permissions and read-only access for marketplace collections
    """

    def __init__(self, session: AsyncSession = None):
        # Use global db_ops instance by default, or create custom one with provided session
        if session is None:
            self.db_ops = async_db_ops  # Use global instance
        else:
            self.db_ops = AsyncDatabaseOps(session)  # Create custom instance for transaction control

    async def _check_subscription_access(self, user_id: str, collection_id: str) -> dict:
        """Check if user has valid subscription access to a collection"""
        has_access, subscription_info = await self.db_ops.check_subscription_access(user_id, collection_id)

        if not has_access:
            # Try to get more specific error information
            marketplace = await self.db_ops.get_collection_marketplace_by_collection_id(collection_id)
            if marketplace is None or marketplace.status != db_models.CollectionMarketplaceStatusEnum.PUBLISHED.value:
                raise CollectionNotPublishedError(collection_id)
            else:
                raise CollectionMarketplaceAccessDeniedError(
                    collection_id, "You need to subscribe to this collection first"
                )

        return subscription_info

    async def _check_marketplace_access(self, user_id: str, collection_id: str) -> dict:
        """Check if user can access marketplace collection (all logged-in users can view published collections)"""
        # Check if collection is published in marketplace
        marketplace = await self.db_ops.get_collection_marketplace_by_collection_id(collection_id)
        if marketplace is None or marketplace.status != db_models.CollectionMarketplaceStatusEnum.PUBLISHED.value:
            raise CollectionNotPublishedError(collection_id)

        # Get collection info
        collection = await self.db_ops.query_collection_without_user(collection_id)
        if collection is None:
            raise CollectionNotFoundException(collection_id)

        # Get owner info
        owner = await self.db_ops.query_user_by_id(collection.user)
        if owner is None:
            raise CollectionNotFoundException(f"Collection owner not found for collection {collection_id}")

        # Check subscription status (optional - might be None)
        subscription = await self.db_ops.get_user_subscription_by_collection_id(user_id, collection_id)

        return {
            "collection_id": collection.id,
            "collection_title": collection.title,
            "collection_description": collection.description,
            "owner_user_id": collection.user,
            "owner_username": owner.username,
            "subscription_id": subscription.id if subscription else None,
            "gmt_subscribed": subscription.gmt_subscribed if subscription else None,
            "is_subscribed": subscription is not None,
            "is_owner": collection.user == user_id,
        }

    async def get_marketplace_collection(self, user_id: str, collection_id: str) -> view_models.SharedCollection:
        """Get MarketplaceCollection details"""
        # Call _check_marketplace_access to verify permissions (all logged-in users can view)
        marketplace_info = await self._check_marketplace_access(user_id, collection_id)

        # Return SharedCollection data with subscription status
        return view_models.SharedCollection(
            id=marketplace_info["collection_id"],
            title=marketplace_info["collection_title"],
            description=marketplace_info["collection_description"],
            owner_user_id=marketplace_info["owner_user_id"],
            owner_username=marketplace_info["owner_username"],
            subscription_id=marketplace_info["subscription_id"],
            gmt_subscribed=marketplace_info["gmt_subscribed"],
        )

    async def list_marketplace_collection_documents(
        self,
        user_id: str,
        collection_id: str,
        page: int = 1,
        page_size: int = 20,
        search: str = None,
        file_type: str = None,
    ) -> view_models.DocumentList:
        """List documents in MarketplaceCollection (read-only mode)"""
        # Call _check_marketplace_access to verify permissions (all logged-in users can view)
        await self._check_marketplace_access(user_id, collection_id)

        # Get the actual collection object for document queries
        collection = await self.db_ops.query_collection_without_user(collection_id)
        if collection is None:
            raise CollectionNotFoundException(collection_id)

        # Use existing document repository methods to query documents
        documents = await self.db_ops.query_documents([collection.user], collection_id)

        # Simple pagination logic
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        paginated_docs = documents[start_idx:end_idx] if len(documents) > start_idx else []

        # Convert to read-only document objects (hide sensitive fields)
        document_items = []
        for doc in paginated_docs:
            # Create document object but hide internal fields
            document = view_models.Document(
                id=doc.id,
                name=doc.name,
                file_type=getattr(doc, "file_type", None),
                size=getattr(doc, "size", None),
                status=getattr(doc, "status", None),
                created=doc.gmt_created.isoformat() if doc.gmt_created else None,
                # Hide user, config, updated time, error messages, etc.
            )
            document_items.append(document)

        return view_models.DocumentList(items=document_items, total=len(documents), page=page, page_size=page_size)

    async def get_marketplace_collection_document_preview(
        self, user_id: str, collection_id: str, document_id: str
    ) -> view_models.DocumentPreview:
        """Preview document in MarketplaceCollection"""
        # Call _check_marketplace_access to verify permissions (all logged-in users can view)
        await self._check_marketplace_access(user_id, collection_id)

        # Get document preview data (same format as original interface)
        # This reuses existing document preview logic
        document = await self.db_ops.query_document_by_id(document_id, ignore_deleted=True)
        if document is None or document.collection_id != collection_id:
            raise CollectionNotFoundException(f"Document {document_id} not found in collection {collection_id}")

        # Return document preview data
        # Note: Actual preview logic would depend on existing document service implementation
        # Here we provide a simplified version
        return view_models.DocumentPreview(
            id=document.id,
            name=document.name,
            content=document.content if hasattr(document, "content") else None,
            file_type=document.file_type,
            size=document.size,
        )

    async def get_marketplace_collection_graph(
        self, user_id: str, collection_id: str, node_limit: int = 100, depth: int = 2, **params
    ) -> dict:
        """Get knowledge graph for MarketplaceCollection (read-only mode)"""
        # Call _check_marketplace_access to verify permissions (all logged-in users can view)
        await self._check_marketplace_access(user_id, collection_id)

        # Get collection object for graph queries
        collection = await self.db_ops.query_collection_without_user(collection_id)
        if collection is None:
            raise CollectionNotFoundException(collection_id)

        # Return knowledge graph data (read-only mode)
        # Note: Does not provide graph editing related interfaces (like merge suggestions, node editing, etc.)
        # Actual implementation would depend on existing graph service
        # For now, return a placeholder structure
        return {
            "nodes": [],
            "edges": [],
            "collection_id": collection_id,
            "read_only": True,
        }

    async def verify_subscription_access(self, user_id: str, collection_id: str) -> bool:
        """Verify user has valid subscription to the collection"""
        try:
            await self._check_subscription_access(user_id, collection_id)
            return True
        except (CollectionNotPublishedError, CollectionMarketplaceAccessDeniedError):
            return False

    async def get_subscription_info(self, user_id: str, collection_id: str) -> Optional[dict]:
        """Get subscription information if user has valid access"""
        try:
            return await self._check_subscription_access(user_id, collection_id)
        except (CollectionNotPublishedError, CollectionMarketplaceAccessDeniedError):
            return None


# Global marketplace collection service instance
marketplace_collection_service = MarketplaceCollectionService()
