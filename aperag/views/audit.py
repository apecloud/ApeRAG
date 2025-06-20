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

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import desc, select

from aperag.db.models import AuditLog, AuditResource, User
from aperag.config import get_async_session
from aperag.schema import view_models
from aperag.service.audit_service import audit_service
from aperag.views.auth import current_user, get_current_admin

router = APIRouter()


@router.get("/audit-logs", tags=["audit"], name="ListAuditLogs", response_model=view_models.AuditLogList)
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
    limit: int = Query(1000, le=5000, description="Maximum number of records"),
    user: User = Depends(current_user)
):
    """List audit logs with filtering"""
    
    # Convert string enums to actual enum values
    audit_resource = None
    
    if resource_type:
        try:
            audit_resource = AuditResource(resource_type)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid resource_type: {resource_type}")

    # Get audit logs
    audit_logs = await audit_service.list_audit_logs(
        user_id=user_id,
        resource_type=audit_resource,
        api_name=api_name,
        http_method=http_method,
        status_code=status_code,
        start_date=start_date,
        end_date=end_date,
        limit=limit
    )

    # Convert to view models
    items = []
    for log in audit_logs:
        items.append(view_models.AuditLog(
            id=str(log.id),
            user_id=log.user_id,
            username=log.username,
            resource_type=log.resource_type.value if hasattr(log.resource_type, 'value') else log.resource_type,
            resource_id=getattr(log, 'resource_id', None),  # This is set during query
            api_name=log.api_name,
            http_method=log.http_method,
            path=log.path,
            status_code=log.status_code,
            start_time=log.start_time,
            end_time=log.end_time,
            duration_ms=getattr(log, 'duration_ms', None),  # Calculated during query
            request_data=log.request_data,
            response_data=log.response_data,
            error_message=log.error_message,
            ip_address=log.ip_address,
            user_agent=log.user_agent,
            request_id=log.request_id,
            created=log.gmt_created
        ))

    return view_models.AuditLogList(items=items)


@router.get("/audit-logs/{audit_id}", tags=["audit"], name="GetAuditLog", response_model=view_models.AuditLog)
async def get_audit_log(
    audit_id: str,
    user: User = Depends(current_user)
):
    """Get a specific audit log by ID"""
    
    async with get_async_session() as session:
        stmt = select(AuditLog).where(AuditLog.id == audit_id)
        result = await session.execute(stmt)
        audit_log = result.scalar_one_or_none()
        
        if not audit_log:
            raise HTTPException(status_code=404, detail="Audit log not found")
        
        # Extract resource_id for this specific log
        resource_id = None
        if audit_log.resource_type and audit_log.path:
            resource_id = audit_service.extract_resource_id_from_path(audit_log.path, audit_log.resource_type)
        
        # Calculate duration if both times are available
        duration_ms = None
        if audit_log.start_time and audit_log.end_time:
            duration_ms = audit_log.end_time - audit_log.start_time
        
        return view_models.AuditLog(
            id=str(audit_log.id),
            user_id=audit_log.user_id,
            username=audit_log.username,
            resource_type=audit_log.resource_type.value if audit_log.resource_type else None,
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
            created=audit_log.gmt_created
        )


@router.get("/audit/logs", tags=["audit"], name="ListAuditLogs")
async def list_audit_logs_view(
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(20, ge=1, le=100, description="Items per page"),
    resource_type: Optional[str] = Query(None, description="Filter by resource type"),
    api_name: Optional[str] = Query(None, description="Filter by API name"),
    user: User = Depends(current_user),
) -> view_models.AuditLogList:
    """List audit logs with filtering and pagination"""
    return await audit_service.list_audit_logs(
        page=page,
        limit=limit,
        resource_type=resource_type,
        api_name=api_name,
    )


 