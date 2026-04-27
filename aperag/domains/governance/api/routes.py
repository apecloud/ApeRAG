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

"""Governance HTTP router (Phase 4 Step 4-S5).

Merges ``aperag/views/audit.py`` + ``aperag/views/api_key.py`` into a
single canonical ``APIRouter`` at the governance domain location.
Handler URLs / methods / response_model / status codes preserved so
``scripts/export_openapi.py --check`` stays byte-stable.

G15 / G16 compliance:

* ``Depends(required_user)`` parameters typed as the per-domain
  ``AuthenticatedUser(Protocol)`` (``id`` + ``role: str``).
* Admin check uses literal string compare
  ``user.role == "admin"`` — the identity-domain ``Role`` enum is
  NEVER imported into governance (per G15).
"""

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy import select

from aperag.config import get_async_session
from aperag.domains.governance.db.models import AuditLog, AuditResource
from aperag.domains.governance.ports import AuthenticatedUser
from aperag.domains.governance.schemas import ApiKeyCreate, ApiKeyList, ApiKeyUpdate, AuditLogList
from aperag.domains.governance.schemas import AuditLog as AuditLogView
from aperag.domains.governance.service.api_key_service import api_key_service
from aperag.domains.governance.service.audit_service import audit_service
from aperag.utils.audit_decorator import audit
from aperag.views.auth import required_user

router = APIRouter()


# ---------- API key routes (from former aperag/views/api_key.py) ---------- #


@router.get("/apikeys", tags=["api_keys"])
async def list_api_keys_view(request: Request, user: AuthenticatedUser = Depends(required_user)) -> ApiKeyList:
    """List all API keys for the current user"""
    return await api_key_service.list_api_keys(str(user.id))


@router.post("/apikeys", tags=["api_keys"])
@audit(resource_type="api_key", api_name="CreateApiKey")
async def create_api_key_view(
    request: Request,
    api_key_create: ApiKeyCreate,
    user: AuthenticatedUser = Depends(required_user),
):
    """Create a new API key"""
    return await api_key_service.create_api_key(str(user.id), api_key_create)


@router.delete("/apikeys/{apikey_id}", tags=["api_keys"])
@audit(resource_type="api_key", api_name="DeleteApiKey")
async def delete_api_key_view(request: Request, apikey_id: str, user: AuthenticatedUser = Depends(required_user)):
    """Delete an API key"""
    return await api_key_service.delete_api_key(str(user.id), apikey_id)


@router.put("/apikeys/{apikey_id}", tags=["api_keys"])
@audit(resource_type="api_key", api_name="UpdateApiKey")
async def update_api_key_view(
    request: Request,
    apikey_id: str,
    api_key_update: ApiKeyUpdate,
    user: AuthenticatedUser = Depends(required_user),
):
    """Update an API key"""
    return await api_key_service.update_api_key(str(user.id), apikey_id, api_key_update)


# ---------- Audit log routes (from former aperag/views/audit.py) ---------- #


@router.get("/audit-logs", tags=["audit"])
async def list_audit_logs(
    user_id: Optional[str] = Query(None, description="Filter by user ID"),
    username: Optional[str] = Query(None, description="Filter by username"),
    resource_type: Optional[str] = Query(None, description="Filter by resource type"),
    resource_id: Optional[str] = Query(None, description="Filter by resource ID"),
    api_name: Optional[str] = Query(None, description="Filter by API name"),
    http_method: Optional[str] = Query(None, description="Filter by HTTP method"),
    status_code: Optional[int] = Query(None, description="Filter by status code"),
    start_date: Optional[datetime] = Query(None, description="Filter by start date"),
    end_date: Optional[datetime] = Query(None, description="Filter by end date"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Page size"),
    sort_by: Optional[str] = Query(None, description="Sort field"),
    sort_order: str = Query("desc", description="Sort order: asc or desc"),
    search: Optional[str] = Query(None, description="Search term"),
    user: AuthenticatedUser = Depends(required_user),
):
    """List audit logs with filtering"""

    # Convert string enums to actual enum values
    audit_resource = None

    if resource_type:
        try:
            audit_resource = AuditResource(resource_type)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid resource_type: {resource_type}")

    # Get audit logs. Non-admin users can only see their own logs.
    # G15 literal compare — never import ``Role.ADMIN`` from identity.
    filter_user_id = user_id
    if user.role != "admin":
        filter_user_id = user.id

    result = await audit_service.list_audit_logs(
        user_id=filter_user_id,
        resource_type=audit_resource,
        api_name=api_name,
        http_method=http_method,
        status_code=status_code,
        start_date=start_date,
        end_date=end_date,
        page=page,
        page_size=page_size,
        sort_by=sort_by,
        sort_order=sort_order,
        search=search,
    )

    # Convert to view models
    items = []
    for log in result.items:
        items.append(
            AuditLogView(
                id=str(log.id),
                user_id=log.user_id,
                username=log.username,
                resource_type=log.resource_type.value if hasattr(log.resource_type, "value") else log.resource_type,
                resource_id=getattr(log, "resource_id", None),  # This is set during query
                api_name=log.api_name,
                http_method=log.http_method,
                path=log.path,
                status_code=log.status_code,
                start_time=log.start_time,
                end_time=log.end_time,
                duration_ms=getattr(log, "duration_ms", None),  # Calculated during query
                request_data=log.request_data,
                response_data=log.response_data,
                error_message=log.error_message,
                ip_address=log.ip_address,
                user_agent=log.user_agent,
                request_id=log.request_id,
                created=log.gmt_created,
            )
        )

    return AuditLogList(
        items=items,
        total=result.total,
        page=result.page,
        page_size=result.page_size,
        total_pages=result.total_pages,
        has_next=result.has_next,
        has_prev=result.has_prev,
    )


@router.get("/audit-logs/{audit_id}", tags=["audit"])
async def get_audit_log(audit_id: str, user: AuthenticatedUser = Depends(required_user)):
    """Get a specific audit log by ID"""

    async for session in get_async_session():
        # G15 literal compare — admin sees all, non-admin sees own.
        if user.role == "admin":
            stmt = select(AuditLog).where(AuditLog.id == audit_id)
        else:
            stmt = select(AuditLog).where(AuditLog.id == audit_id, AuditLog.user_id == user.id)
        result = await session.execute(stmt)
        audit_log = result.scalar_one_or_none()

        if not audit_log:
            raise HTTPException(status_code=404, detail="Audit log not found")

        # Extract resource_id for this specific log
        resource_id = None
        if audit_log.resource_type and audit_log.path:
            # Convert string to enum if needed
            resource_type_enum = audit_log.resource_type
            if isinstance(audit_log.resource_type, str):
                try:
                    resource_type_enum = AuditResource(audit_log.resource_type)
                except ValueError:
                    resource_type_enum = None

            if resource_type_enum:
                resource_id = audit_service.extract_resource_id_from_path(audit_log.path, resource_type_enum)

        # Calculate duration if both times are available
        duration_ms = None
        if audit_log.start_time and audit_log.end_time:
            duration_ms = audit_log.end_time - audit_log.start_time

        return AuditLogView(
            id=str(audit_log.id),
            user_id=audit_log.user_id,
            username=audit_log.username,
            resource_type=audit_log.resource_type.value
            if hasattr(audit_log.resource_type, "value")
            else audit_log.resource_type,
            resource_id=resource_id,
            api_name=audit_log.api_name,
            http_method=audit_log.http_method,
            path=audit_log.path,
            status_code=audit_log.status_code,
            start_time=audit_log.start_time,
            end_time=audit_log.end_time,
            duration_ms=duration_ms,
            request_data=audit_log.request_data,
            response_data=audit_log.response_data,
            error_message=audit_log.error_message,
            ip_address=audit_log.ip_address,
            user_agent=audit_log.user_agent,
            request_id=audit_log.request_id,
            created=audit_log.gmt_created,
        )
