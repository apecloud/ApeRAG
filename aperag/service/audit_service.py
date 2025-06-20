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
import re
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import and_, desc, select

from aperag.config import get_async_session
from aperag.db.models import AuditLog, AuditResource

logger = logging.getLogger(__name__)


class AuditService:
    """Service for handling audit logs"""

    def __init__(self):
        self.enabled = True
        # Sensitive fields that should be filtered from logs
        self.sensitive_fields = {
            "password", "token", "api_key", "secret", "authorization",
            "access_token", "refresh_token", "private_key", "credential"
        }
        
        # Map FastAPI tags to audit resources
        self.tag_resource_map = {
            # Support both singular and plural forms
            "collection": AuditResource.COLLECTION,
            "collections": AuditResource.COLLECTION,
            "document": AuditResource.DOCUMENT,
            "documents": AuditResource.DOCUMENT,
            "bot": AuditResource.BOT,
            "bots": AuditResource.BOT,
            "chat": AuditResource.CHAT,
            "chats": AuditResource.CHAT,
            "message": AuditResource.MESSAGE,
            "messages": AuditResource.MESSAGE,
            "apikey": AuditResource.API_KEY,
            "apikeys": AuditResource.API_KEY,
            "llm_provider": AuditResource.LLM_PROVIDER,
            "llm_providers": AuditResource.LLM_PROVIDER,
            "llm_provider_model": AuditResource.LLM_PROVIDER_MODEL,
            "llm_provider_models": AuditResource.LLM_PROVIDER_MODEL,
            "user": AuditResource.USER,
            "users": AuditResource.USER,
            "config": AuditResource.CONFIG,
        }

    def _filter_sensitive_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Filter sensitive information from data"""
        if not isinstance(data, dict):
            return data

        filtered = {}
        for key, value in data.items():
            lower_key = key.lower()
            if any(sensitive in lower_key for sensitive in self.sensitive_fields):
                filtered[key] = "***FILTERED***"
            elif isinstance(value, dict):
                filtered[key] = self._filter_sensitive_data(value)
            elif isinstance(value, list):
                filtered[key] = [
                    self._filter_sensitive_data(item) if isinstance(item, dict) else item
                    for item in value
                ]
            else:
                filtered[key] = value
        return filtered

    def _safe_json_serialize(self, data: Any) -> str:
        """Safely serialize data to JSON string"""
        if data is None:
            return None
        
        try:
            # Filter sensitive data first
            if isinstance(data, dict):
                data = self._filter_sensitive_data(data)
            
            # Handle special types that aren't JSON serializable
            def json_serializer(obj):
                if hasattr(obj, 'dict'):  # Pydantic models
                    return obj.dict()
                elif hasattr(obj, '__dict__'):  # Regular objects
                    return obj.__dict__
                else:
                    return str(obj)
            
            return json.dumps(data, default=json_serializer, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"Failed to serialize data: {e}")
            return str(data)

    def _extract_client_info(self, request) -> tuple[Optional[str], Optional[str]]:
        """Extract client IP and User-Agent from request"""
        try:
            # Get IP address
            ip_address = None
            if hasattr(request, 'client') and request.client:
                ip_address = request.client.host
            
            # Check for forwarded headers
            if hasattr(request, 'headers'):
                forwarded_for = request.headers.get('X-Forwarded-For')
                if forwarded_for:
                    ip_address = forwarded_for.split(',')[0].strip()
                elif request.headers.get('X-Real-IP'):
                    ip_address = request.headers.get('X-Real-IP')
            
            # Get User-Agent
            user_agent = None
            if hasattr(request, 'headers'):
                user_agent = request.headers.get('User-Agent')
            
            return ip_address, user_agent
        except Exception as e:
            logger.warning(f"Failed to extract client info: {e}")
            return None, None

    def get_resource_type_from_tags(self, tags: List[str]) -> Optional[AuditResource]:
        """Get resource type from FastAPI tags"""
        if not tags:
            return None
            
        # Find the first tag that matches our resource mapping
        for tag in tags:
            if tag in self.tag_resource_map:
                return self.tag_resource_map[tag]
        
        return None

    def extract_resource_id_from_path(self, path: str, resource_type: AuditResource) -> Optional[str]:
        """Extract resource ID from path - called during query time"""
        try:
            # Define ID extraction patterns for different resource types
            id_patterns = {
                AuditResource.MESSAGE: r'/messages/([^/]+)',
                AuditResource.CHAT: r'/chats/([^/]+)',
                AuditResource.DOCUMENT: r'/documents/([^/]+)',
                AuditResource.BOT: r'/bots/([^/]+)',
                AuditResource.COLLECTION: r'/collections/([^/]+)',
                AuditResource.API_KEY: r'/apikeys/([^/]+)',
                AuditResource.LLM_PROVIDER: r'/llm_providers/([^/]+)',
                AuditResource.LLM_PROVIDER_MODEL: r'/models/([^/]+/[^/]+)',
                AuditResource.USER: r'/users/([^/]+)',
            }

            pattern = id_patterns.get(resource_type)
            if pattern:
                match = re.search(pattern, path)
                if match:
                    return match.group(1)
                    
        except Exception as e:
            logger.warning(f"Failed to extract resource ID: {e}")
            
        return None

    async def log_audit(
        self,
        user_id: Optional[str],
        username: Optional[str],
        resource_type: AuditResource,
        api_name: str,
        http_method: str,
        path: str,
        status_code: int,
        start_time: int,
        end_time: Optional[int] = None,
        request_data: Optional[Dict[str, Any]] = None,
        response_data: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        request_id: Optional[str] = None
    ):
        """Log an audit entry"""
        if not self.enabled:
            return

        try:
            # Create audit log entry
            audit_log = AuditLog(
                id=str(uuid.uuid4()),
                user_id=user_id,
                username=username,
                resource_type=resource_type,
                api_name=api_name,
                http_method=http_method,
                path=path,
                status_code=status_code,
                start_time=start_time,
                end_time=end_time,
                request_data=self._safe_json_serialize(request_data),
                response_data=self._safe_json_serialize(response_data),
                error_message=error_message,
                ip_address=ip_address,
                user_agent=user_agent,
                request_id=request_id or str(uuid.uuid4())
            )

            # Save to database asynchronously
            async for session in get_async_session():
                session.add(audit_log)
                await session.commit()

        except Exception as e:
            logger.error(f"Failed to log audit: {e}")

    async def list_audit_logs(
        self,
        user_id: Optional[str] = None,
        resource_type: Optional[AuditResource] = None,
        api_name: Optional[str] = None,
        http_method: Optional[str] = None,
        status_code: Optional[int] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 1000
    ) -> List[AuditLog]:
        """List audit logs with filtering"""
        async for session in get_async_session():
            # Build query
            stmt = select(AuditLog)
            
            # Add filters
            conditions = []
            if user_id:
                conditions.append(AuditLog.user_id == user_id)
            if resource_type:
                conditions.append(AuditLog.resource_type == resource_type)
            if api_name:
                conditions.append(AuditLog.api_name.like(f"%{api_name}%"))
            if http_method:
                conditions.append(AuditLog.http_method == http_method)
            if status_code:
                conditions.append(AuditLog.status_code == status_code)
            if start_date:
                conditions.append(AuditLog.gmt_created >= start_date)
            if end_date:
                conditions.append(AuditLog.gmt_created <= end_date)
            
            if conditions:
                stmt = stmt.where(and_(*conditions))
            
            # Order by creation time (newest first) and limit
            stmt = stmt.order_by(desc(AuditLog.gmt_created)).limit(limit)
            
            # Execute query
            result = await session.execute(stmt)
            audit_logs = result.scalars().all()
            
            # Extract resource_id for each log during query time
            for log in audit_logs:
                if log.resource_type and log.path:
                    log.resource_id = self.extract_resource_id_from_path(log.path, log.resource_type)
                
                # Calculate duration if both times are available
                if log.start_time and log.end_time:
                    log.duration_ms = log.end_time - log.start_time
                else:
                    log.duration_ms = None
            
            return audit_logs


# Global audit service instance
audit_service = AuditService() 