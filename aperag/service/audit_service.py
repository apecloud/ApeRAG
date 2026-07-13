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
import uuid
from datetime import datetime
from typing import Any, Dict, Optional

from sqlalchemy import and_, desc, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import sessionmaker

from aperag.config import async_engine
from aperag.db.models import AuditLog, AuditResource

logger = logging.getLogger(__name__)


class AuditService:
    """Service for handling audit logs"""

    def __init__(self):
        self.enabled = True
        # Sensitive fields that should be filtered from logs
        self.sensitive_fields = {
            "password",
            "token",
            "api_key",
            "secret",
            "authorization",
            "access_token",
            "refresh_token",
            "private_key",
            "credential",
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
                    self._filter_sensitive_data(item) if isinstance(item, dict) else item for item in value
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
                if hasattr(obj, "dict"):  # Pydantic models
                    return obj.dict()
                elif hasattr(obj, "__dict__"):  # Regular objects
                    return obj.__dict__
                else:
                    return str(obj)

            return json.dumps(data, default=json_serializer, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"Failed to serialize data: {e}")
            return str(data)

    def extract_resource_id_from_path(self, path: str, resource_type: AuditResource) -> Optional[str]:
        """Extract resource ID from path - called during query time"""
        try:
            # Define ID extraction patterns for different resource types
            id_patterns = {
                AuditResource.MESSAGE: r"/messages/([^/]+)",
                AuditResource.CHAT: r"/chats/([^/]+)",
                AuditResource.DOCUMENT: r"/documents/([^/]+)",
                AuditResource.BOT: r"/bots/([^/]+)",
                AuditResource.COLLECTION: r"/collections/([^/]+)",
                AuditResource.API_KEY: r"/apikeys/([^/]+)",
                AuditResource.LLM_PROVIDER: r"/llm_providers/([^/]+)",
                AuditResource.LLM_PROVIDER_MODEL: r"/models/([^/]+/[^/]+)",
                AuditResource.USER: r"/users/([^/]+)",
            }

            pattern = id_patterns.get(resource_type)
            if pattern:
                match = re.search(pattern, path)
                if match:
                    return match.group(1)

        except Exception as e:
            logger.warning(f"Failed to extract resource ID: {e}")

        return None

    def _make_session(self) -> AsyncSession:
        """Create a short-lived async session for write operations.

        Using a dedicated factory here means we open the session only for the
        duration of the DB write, avoiding the ``async for ... break`` generator
        antipattern and making the lifecycle explicit.
        """
        factory = sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)
        return factory()

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
        request_id: Optional[str] = None,
    ):
        """Log an audit entry and persist it to the database.

        All expensive work (serialization, duration calculation) is done
        *before* the session is opened so the connection is held for the
        minimum time possible.
        """
        if not self.enabled:
            return

        try:
            # Compute duration before touching the DB.
            duration_ms: Optional[int] = None
            if start_time is not None and end_time is not None:
                duration_ms = end_time - start_time

            # Serialize request/response data outside the session.
            serialized_request = self._safe_json_serialize(request_data)
            serialized_response = self._safe_json_serialize(response_data)

            # Emit a structured log line so operators can tail logs without
            # querying the database.
            logger.info(
                "audit %s %s %s status=%s duration_ms=%s",
                http_method,
                path,
                api_name,
                status_code,
                duration_ms,
            )

            # Build the ORM object outside the session — no DB access needed.
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
                duration_ms=duration_ms,
                request_data=serialized_request,
                response_data=serialized_response,
                error_message=error_message,
                ip_address=ip_address,
                user_agent=user_agent,
                request_id=request_id or str(uuid.uuid4()),
            )

            # Open the session only for the DB write; close it immediately after.
            async with self._make_session() as session:
                session.add(audit_log)
                await session.commit()

        except Exception as e:
            logger.error(f"Failed to log audit: {e}")

    async def list_audit_logs(
        self,
        page: int = 1,
        page_size: int = 20,
        sort_by: str = None,
        sort_order: str = "desc",
        search: str = None,
        user_id: Optional[str] = None,
        resource_type: Optional[AuditResource] = None,
        api_name: Optional[str] = None,
        http_method: Optional[str] = None,
        status_code: Optional[int] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ):
        """List audit logs with pagination, sorting, and filtering"""

        # Define sort field mapping
        sort_mapping = {
            "created": AuditLog.gmt_created,
            "duration": AuditLog.duration_ms,  # Stored column — sortable without expression
            "status_code": AuditLog.status_code,
            "api_name": AuditLog.api_name,
        }

        # Define search fields mapping
        search_fields = {"api_name": AuditLog.api_name, "path": AuditLog.path}

        async def _list_audit_logs(session):
            from aperag.utils.pagination import ListParams, PaginationHelper, PaginationParams, SearchParams, SortParams

            # Build base query
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

            # Build query parameters
            params = ListParams(
                pagination=PaginationParams(page=page, page_size=page_size),
                sort=SortParams(sort_by=sort_by, sort_order=sort_order) if sort_by else None,
                search=SearchParams(search=search, search_fields=["api_name", "path"]) if search else None,
            )

            # Use pagination helper
            items, total = await PaginationHelper.paginate_query(
                query=stmt,
                session=session,
                params=params,
                sort_mapping=sort_mapping,
                search_fields=search_fields,
                default_sort=desc(AuditLog.gmt_created),
            )

            return items, total

        # Execute query with proper session management.
        # Open the session, execute the lightweight query, and close it before
        # any post-processing so the connection is returned to the pool quickly.
        audit_logs = None
        total = 0
        async with self._make_session() as session:
            audit_logs, total = await _list_audit_logs(session)

        # Post-process audit logs outside of session to avoid long session occupation
        processed_logs = []
        for log in audit_logs:
            if log.resource_type and log.path:
                # Convert string to enum if needed
                resource_type_enum = log.resource_type
                if isinstance(log.resource_type, str):
                    try:
                        resource_type_enum = AuditResource(log.resource_type)
                    except ValueError:
                        resource_type_enum = None

                if resource_type_enum:
                    log.resource_id = self.extract_resource_id_from_path(log.path, resource_type_enum)
                else:
                    log.resource_id = None

            # duration_ms is now stored on the row; fall back to computing
            # it on the fly for rows written before this migration.
            if log.duration_ms is None and log.start_time and log.end_time:
                log.duration_ms = log.end_time - log.start_time

            processed_logs.append(log)

        # Build paginated response
        from aperag.utils.pagination import PaginationHelper

        return PaginationHelper.build_response(items=processed_logs, total=total, page=page, page_size=page_size)


# Global audit service instance
audit_service = AuditService()
