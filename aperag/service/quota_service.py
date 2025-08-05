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

"""
Quota service for managing user quotas and usage tracking.
"""
import logging
from typing import Dict, List, Optional

from aperag.db.ops import AsyncDatabaseOps, async_db_ops
from aperag.exceptions import QuotaExceededException

logger = logging.getLogger(__name__)


class QuotaService:
    """Service for managing user quotas."""
    
    def __init__(self, db_ops: AsyncDatabaseOps = None):
        self.db_ops = db_ops or async_db_ops

    async def get_user_quotas(self, user_id: str) -> Dict[str, Dict[str, int]]:
        """Get all quotas for a user as a dictionary."""
        async def _query(session):
            from aperag.db.models import UserQuota
            from sqlalchemy import select
            
            stmt = select(UserQuota).where(UserQuota.user == user_id)
            result = await session.execute(stmt)
            quotas = result.scalars().all()
            
            quota_dict = {}
            for quota in quotas:
                quota_dict[quota.key] = {
                    'quota_limit': quota.quota_limit,
                    'current_usage': quota.current_usage,
                    'remaining': max(0, quota.quota_limit - quota.current_usage)
                }
            
            return quota_dict
        
        return await self.db_ops._execute_query(_query)

    async def get_all_users_quotas(self, search_term: str = None) -> List[Dict]:
        """Get quotas for all users (admin only)."""
        async def _query(session):
            from aperag.db.models import User, UserQuota
            from sqlalchemy import select, or_
            
            # Build query for users
            stmt = select(User).where(User.gmt_deleted.is_(None))
            
            # Add search filter if provided
            if search_term and search_term.strip():
                search_value = search_term.strip()
                stmt = stmt.where(
                    or_(
                        User.username == search_value,
                        User.email == search_value,
                        User.id == search_value
                    )
                )
            
            result = await session.execute(stmt)
            users = result.scalars().all()
            
            result_list = []
            for user in users:
                # Get quotas for this user
                quota_stmt = select(UserQuota).where(UserQuota.user == user.id)
                quota_result = await session.execute(quota_stmt)
                quotas = quota_result.scalars().all()
                
                quota_dict = {}
                for quota in quotas:
                    quota_dict[quota.key] = {
                        'quota_limit': quota.quota_limit,
                        'current_usage': quota.current_usage,
                        'remaining': max(0, quota.quota_limit - quota.current_usage)
                    }
                
                result_list.append({
                    'user_id': user.id,
                    'username': user.username,
                    'email': user.email,
                    'role': user.role,
                    'quotas': quota_dict
                })
            
            return result_list
        
        return await self.db_ops._execute_query(_query)

    async def update_user_quota(self, user_id: str, quota_updates: Dict[str, int]) -> Dict[str, any]:
        """Update quota limits for a user (supports both single and batch updates)."""
        async def _operation(session):
            from aperag.db.models import UserQuota
            from sqlalchemy import select
            from aperag.utils.utils import utc_now
            
            updated_quotas = []
            
            for quota_type, new_limit in quota_updates.items():
                if new_limit is None:
                    continue  # Skip null values
                    
                stmt = select(UserQuota).where(
                    UserQuota.user == user_id,
                    UserQuota.key == quota_type
                )
                result = await session.execute(stmt)
                quota = result.scalars().first()
                
                old_limit = 0
                if not quota:
                    # Create new quota if it doesn't exist
                    quota = UserQuota(
                        user=user_id,
                        key=quota_type,
                        quota_limit=new_limit,
                        current_usage=0,
                        gmt_created=utc_now(),
                        gmt_updated=utc_now()
                    )
                    session.add(quota)
                else:
                    old_limit = quota.quota_limit
                    quota.quota_limit = new_limit
                    quota.gmt_updated = utc_now()
                
                updated_quotas.append({
                    'quota_type': quota_type,
                    'old_limit': old_limit,
                    'new_limit': new_limit
                })
            
            await session.flush()
            return {
                "success": True,
                "message": "Quotas updated successfully",
                "user_id": user_id,
                "updated_quotas": updated_quotas
            }
        
        return await self.db_ops.execute_with_transaction(_operation)


    async def recalculate_user_usage(self, user_id: str) -> Dict[str, int]:
        """Recalculate actual usage for all quotas of a user."""
        async def _operation(session):
            from aperag.db.models import Collection, Document, Bot, UserQuota
            from sqlalchemy import select, func
            from aperag.utils.utils import utc_now
            
            # Calculate actual usage
            usage_data = {}
            
            # Collection count
            stmt = select(func.count()).select_from(Collection).where(
                Collection.user == user_id,
                Collection.status != 'DELETED'
            )
            collection_count = await session.scalar(stmt)
            usage_data['max_collection_count'] = collection_count
            
            # Total document count across all collections
            stmt = select(func.count(Document.id)).select_from(
                Document.__table__.join(Collection.__table__, Document.collection_id == Collection.id)
            ).where(
                Collection.user == user_id,
                Document.status != 'DELETED',
                Collection.status != 'DELETED'
            )
            total_document_count = await session.scalar(stmt)
            usage_data['max_document_count'] = total_document_count
            
            # Bot count (exclude system default bot)
            stmt = select(func.count()).select_from(Bot).where(
                Bot.user == user_id,
                Bot.gmt_deleted.is_(None),
                Bot.title != "Default Agent Bot"  # Exclude system default bot
            )
            bot_count = await session.scalar(stmt)
            usage_data['max_bot_count'] = bot_count
            
            # Update quotas with recalculated usage
            for quota_type, actual_usage in usage_data.items():
                stmt = select(UserQuota).where(
                    UserQuota.user == user_id,
                    UserQuota.key == quota_type
                )
                result = await session.execute(stmt)
                quota = result.scalars().first()
                
                if quota:
                    quota.current_usage = actual_usage
                    quota.gmt_updated = utc_now()
            
            await session.flush()
            return usage_data
        
        return await self.db_ops.execute_with_transaction(_operation)

    async def check_and_consume_quota(self, user_id: str, quota_type: str, amount: int = 1, session=None) -> None:
        """
        Check quota availability and consume it atomically.
        Raises QuotaExceededException if quota would be exceeded.
        This should be called within the same transaction as the resource creation.
        
        Args:
            user_id: User ID
            quota_type: Type of quota to check
            amount: Amount to consume
            session: Optional session to use. If None, creates a new transaction.
        """
        async def _operation(session):
            from aperag.db.models import UserQuota
            from sqlalchemy import select
            from aperag.utils.utils import utc_now
            
            # Use SELECT FOR UPDATE to prevent race conditions
            stmt = select(UserQuota).where(
                UserQuota.user == user_id,
                UserQuota.key == quota_type
            ).with_for_update()
            
            result = await session.execute(stmt)
            quota = result.scalars().first()
            
            if not quota:
                # Use a different exception for quota not found case
                from aperag.exceptions import ResourceNotFoundException
                raise ResourceNotFoundException("quota", f"{quota_type} for user {user_id}")
            
            # Check if consuming this amount would exceed the limit
            if quota.current_usage + amount > quota.quota_limit:
                raise QuotaExceededException(quota_type, quota.quota_limit, quota.current_usage)
            
            # Update usage
            quota.current_usage += amount
            quota.gmt_updated = utc_now()
            
            await session.flush()
        
        if session is not None:
            # Use the provided session (within existing transaction)
            return await _operation(session)
        else:
            # Create new transaction
            return await self.db_ops.execute_with_transaction(_operation)

    async def release_quota(self, user_id: str, quota_type: str, amount: int = 1, session=None) -> None:
        """
        Release quota (decrease usage).
        This should be called within the same transaction as the resource deletion.
        
        Args:
            user_id: User ID
            quota_type: Type of quota to release
            amount: Amount to release
            session: Optional session to use. If None, creates a new transaction.
        """
        async def _operation(session):
            from aperag.db.models import UserQuota
            from sqlalchemy import select
            from aperag.utils.utils import utc_now
            
            stmt = select(UserQuota).where(
                UserQuota.user == user_id,
                UserQuota.key == quota_type
            ).with_for_update()
            
            result = await session.execute(stmt)
            quota = result.scalars().first()
            
            if quota:
                # Ensure we don't go below 0
                quota.current_usage = max(0, quota.current_usage - amount)
                quota.gmt_updated = utc_now()
                await session.flush()
        
        if session is not None:
            # Use the provided session (within existing transaction)
            return await _operation(session)
        else:
            # Create new transaction
            return await self.db_ops.execute_with_transaction(_operation)

    async def get_system_default_quotas(self) -> Dict[str, int]:
        """Get system default quotas from config table."""
        async def _query(session):
            from aperag.db.models import ConfigModel
            from sqlalchemy import select
            import json
            
            stmt = select(ConfigModel).where(ConfigModel.key == 'system_default_quotas')
            result = await session.execute(stmt)
            config = result.scalars().first()
            
            if config:
                try:
                    return json.loads(config.value)
                except json.JSONDecodeError:
                    pass
            
            # Return hardcoded defaults if not found in config
            return {
                "max_collection_count": 10,
                "max_document_count": 1000,
                "max_document_count_per_collection": 100,
                "max_bot_count": 5,
            }
        
        return await self.db_ops._execute_query(_query)

    async def update_system_default_quotas(self, quotas: Dict[str, int]) -> bool:
        """Update system default quotas in config table."""
        async def _operation(session):
            from aperag.db.models import ConfigModel
            from sqlalchemy import select
            from aperag.utils.utils import utc_now
            import json
            
            stmt = select(ConfigModel).where(ConfigModel.key == 'system_default_quotas')
            result = await session.execute(stmt)
            config = result.scalars().first()
            
            if config:
                config.value = json.dumps(quotas)
                config.gmt_updated = utc_now()
            else:
                config = ConfigModel(
                    key='system_default_quotas',
                    value=json.dumps(quotas),
                    gmt_created=utc_now(),
                    gmt_updated=utc_now()
                )
                session.add(config)
            
            await session.flush()
            return True
        
        return await self.db_ops.execute_with_transaction(_operation)

    async def initialize_user_quotas(self, user_id: str) -> None:
        """Initialize default quotas for a new user from system defaults."""
        async def _operation(session):
            from aperag.db.models import UserQuota
            from sqlalchemy import select
            from aperag.utils.utils import utc_now
            
            # Get default quotas from system config
            default_quotas = await self.get_system_default_quotas()
            
            for quota_type, limit in default_quotas.items():
                # Check if quota already exists
                stmt = select(UserQuota).where(
                    UserQuota.user == user_id,
                    UserQuota.key == quota_type
                )
                result = await session.execute(stmt)
                existing_quota = result.scalars().first()
                
                if not existing_quota:
                    quota = UserQuota(
                        user=user_id,
                        key=quota_type,
                        quota_limit=limit,
                        current_usage=0,
                        gmt_created=utc_now(),
                        gmt_updated=utc_now()
                    )
                    session.add(quota)
            
            await session.flush()
        
        return await self.db_ops.execute_with_transaction(_operation)


# Create a global service instance
quota_service = QuotaService()
