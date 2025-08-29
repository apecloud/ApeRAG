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
from typing import Optional

from aperag.db.models import Collection, CollectionStatus, CollectionType, User
from aperag.db.ops import async_db_ops
from aperag.schema.view_models import CollectionConfig, CollectionCreate, ModelSpec, TagFilterRequest, TagFilterCondition
from aperag.service.collection_service import collection_service
from aperag.service.llm_available_model_service import llm_available_model_service
from aperag.utils.utils import utc_now

logger = logging.getLogger(__name__)


class ChatCollectionService:
    """
    Chat collection management service
    Handles creation and management of chat-specific collections for users
    """

    def __init__(self):
        self.db_ops = async_db_ops

    async def get_user_chat_collection(self, user_id: str) -> Optional[Collection]:
        """Get user's chat collection"""
        user = await self.db_ops.query_user_by_id(user_id)
        if not user or not user.chat_collection_id:
            return None

        collection = await self.db_ops.query_collection_by_id(user.chat_collection_id)
        if collection and collection.status != CollectionStatus.DELETED:
            return collection

        return None

    async def _get_default_embedding_model(self, user_id: str) -> Optional[ModelSpec]:
        """Get default embedding model for chat collection"""
        try:
            # First, try to get models with default_for_embedding tag
            tag_filter_request = TagFilterRequest(
                tag_filters=[TagFilterCondition(operation="AND", tags=["default_for_embedding"])]
            )
            models = await llm_available_model_service.get_available_models(user_id, tag_filter_request)
            
            # Find first embedding model with default_for_embedding tag
            for provider in models.items or []:
                for embedding_model in provider.embedding or []:
                    return ModelSpec(
                        model=embedding_model.model,
                        model_service_provider=provider.name,
                        custom_llm_provider=embedding_model.custom_llm_provider,
                    )
            
            # If no default_for_embedding models found, try enable_for_collection tag
            tag_filter_request = TagFilterRequest(
                tag_filters=[TagFilterCondition(operation="AND", tags=["enable_for_collection"])]
            )
            models = await llm_available_model_service.get_available_models(user_id, tag_filter_request)
            
            # Find first embedding model with enable_for_collection tag
            for provider in models.items or []:
                for embedding_model in provider.embedding or []:
                    return ModelSpec(
                        model=embedding_model.model,
                        model_service_provider=provider.name,
                        custom_llm_provider=embedding_model.custom_llm_provider,
                    )
            
            logger.warning(f"No suitable embedding model found for user {user_id}")
            return None
            
        except Exception as e:
            logger.error(f"Failed to get default embedding model for user {user_id}: {e}")
            return None

    async def create_user_chat_collection(self, user_id: str) -> Collection:
        """Create chat collection for user"""
        # Get default embedding model
        embedding_model = await self._get_default_embedding_model(user_id)
        
        if not embedding_model:
            raise ValueError("No suitable embedding model found for chat collection")
        
        # Create collection config
        config = CollectionConfig(
            source="system",
            enable_vector=True,
            enable_fulltext=True,
            enable_knowledge_graph=False,
            enable_summary=False,
            enable_vision=False,
            embedding=embedding_model,
        )
        
        # Create collection using collection_service
        collection_create = CollectionCreate(
            title="Chat Documents",
            description="Documents uploaded in chat sessions",
            type="document",
            config=config,
        )
        
        collection_response = await collection_service.create_collection(user_id, collection_create)
        
        # Get the actual Collection model instance
        collection = await self.db_ops.query_collection_by_id(collection_response.id)
        
        # Mark as chat collection and update User table
        async def _mark_as_chat_collection(session):
            # Update collection to mark as chat collection
            collection_obj = await session.get(Collection, collection_response.id)
            if collection_obj:
                collection_obj.type = CollectionType.CHAT
                session.add(collection_obj)
                await session.flush()
            
            # Update User table to link chat collection
            user = await session.get(User, user_id)
            if user:
                user.chat_collection_id = collection_response.id
                session.add(user)
                await session.flush()
        
        await self.db_ops.execute_with_transaction(_mark_as_chat_collection)
        
        # Refresh collection to get updated data
        collection = await self.db_ops.query_collection_by_id(collection_response.id)
        
        logger.info(f"Created chat collection {collection.id} for user {user_id}")
        return collection

    async def initialize_user_chat_collection(self, user_id: str) -> Collection:
        """Initialize chat collection for user during registration"""
        existing_collection = await self.get_user_chat_collection(user_id)
        if existing_collection:
            logger.info(f"User {user_id} already has chat collection {existing_collection.id}")
            return existing_collection

        return await self.create_user_chat_collection(user_id)

    async def get_user_chat_collection_id(self, user_id: str) -> Optional[str]:
        """Get user chat collection ID"""
        collection = await self.get_user_chat_collection(user_id)
        return collection.id if collection else None


# Create a global service instance
chat_collection_service = ChatCollectionService()
