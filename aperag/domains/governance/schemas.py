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

"""Canonical Pydantic view models for the ``governance`` domain.

Phase 4 Step 4-S3b carves the API-key + audit-log envelope schemas
out of ``aperag.schema.view_models``. Dual-hook symmetric re-export
pattern (Phase 3 Step 4b) keeps pre-migration callers working.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field, conint

from aperag.schema.common import PageResult, PaginatedResponse

__all__ = [
    "ApiKey",
    "ApiKeyList",
    "ApiKeyCreate",
    "ApiKeyUpdate",
    "AuditLog",
    "AuditLogList",
    "QuotaInfo",
    "QuotaUpdateRequest",
    "QuotaUpdateResponse",
    "SystemDefaultQuotas",
    "SystemDefaultQuotasResponse",
    "SystemDefaultQuotasUpdateRequest",
    "SystemDefaultQuotasUpdateResponse",
    "UpdatedQuota",
    "UserQuotaInfo",
    "UserQuotaList",
]


class ApiKey(BaseModel):
    id: Optional[str] = None
    key: Optional[str] = None
    description: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    last_used_at: Optional[datetime] = None


class ApiKeyList(BaseModel):
    """
    A list of API keys
    """

    items: Optional[list[ApiKey]] = None
    pageResult: Optional[PageResult] = None


class ApiKeyCreate(BaseModel):
    description: Optional[str] = None


class ApiKeyUpdate(BaseModel):
    description: Optional[str] = None


class QuotaInfo(BaseModel):
    """
    Quota information for a specific quota type
    """

    quota_type: str = Field(..., description="Type of quota", examples=["max_collection_count"])
    quota_limit: int = Field(..., description="Maximum allowed usage", examples=[10])
    current_usage: int = Field(..., description="Current usage count", examples=[3])
    remaining: int = Field(..., description="Remaining quota available", examples=[7])


class UserQuotaInfo(BaseModel):
    """
    Complete quota information for a user
    """

    user_id: str = Field(..., description="User ID", examples=["user123"])
    username: Optional[str] = Field(None, description="Username", examples=["john_doe"])
    email: Optional[str] = Field(None, description="User email", examples=["john@example.com"])
    role: str = Field(..., description="User role", examples=["rw"])
    quotas: list[QuotaInfo] = Field(..., description="List of quota information")


class UserQuotaList(BaseModel):
    """
    List of user quota information (admin view)
    """

    items: list[UserQuotaInfo] = Field(..., description="List of user quota information")


class QuotaUpdateRequest(BaseModel):
    """
    Request to update user quotas (supports both single and batch updates)
    """

    max_collection_count: Optional[conint(ge=0)] = Field(None, description="New limit for collection count")
    max_document_count: Optional[conint(ge=0)] = Field(None, description="New limit for document count")
    max_document_count_per_collection: Optional[conint(ge=0)] = Field(
        None, description="New limit for documents per collection"
    )
    max_bot_count: Optional[conint(ge=0)] = Field(None, description="New limit for bot count")


class UpdatedQuota(BaseModel):
    quota_type: str = Field(
        ...,
        description="Type of quota that was updated",
        examples=["max_collection_count"],
    )
    old_limit: int = Field(..., description="Previous quota limit", examples=[10])
    new_limit: int = Field(..., description="New quota limit", examples=[20])


class QuotaUpdateResponse(BaseModel):
    """
    Response after updating user quotas (supports both single and batch updates)
    """

    success: bool = Field(..., description="Whether the update was successful", examples=[True])
    message: str = Field(..., description="Status message", examples=["Quotas updated successfully"])
    user_id: str = Field(..., description="User ID that was updated", examples=["user123"])
    updated_quotas: list[UpdatedQuota] = Field(..., description="List of updated quotas")


class SystemDefaultQuotas(BaseModel):
    """
    System default quota configuration
    """

    max_collection_count: conint(ge=0) = Field(..., description="Default maximum collection count", examples=[20])
    max_document_count: conint(ge=0) = Field(..., description="Default maximum document count", examples=[4000])
    max_document_count_per_collection: conint(ge=0) = Field(
        ..., description="Default maximum documents per collection", examples=[200]
    )
    max_bot_count: conint(ge=0) = Field(..., description="Default maximum bot count", examples=[10])


class SystemDefaultQuotasResponse(BaseModel):
    """
    Response containing system default quotas
    """

    quotas: SystemDefaultQuotas


class SystemDefaultQuotasUpdateRequest(BaseModel):
    """
    Request to update system default quotas
    """

    quotas: SystemDefaultQuotas


class SystemDefaultQuotasUpdateResponse(BaseModel):
    """
    Response after updating system default quotas
    """

    success: bool = Field(..., description="Whether the update was successful", examples=[True])
    message: str = Field(
        ...,
        description="Status message",
        examples=["System default quotas updated successfully"],
    )
    quotas: SystemDefaultQuotas


class AuditLog(BaseModel):
    """
    Audit log entry
    """

    id: Optional[str] = Field(None, description="Audit log ID")
    user_id: Optional[str] = Field(None, description="User ID who performed the action")
    username: Optional[str] = Field(None, description="Username for display")
    resource_type: Optional[
        Literal[
            "collection",
            "document",
            "bot",
            "chat",
            "message",
            "api_key",
            "llm",
            "model_provider",
            "model_account",
            "model",
            "model_use",
            "user",
            "flow",
            "search",
            "index",
        ]
    ] = Field(None, description="Type of resource")
    resource_id: Optional[str] = Field(None, description="ID of the resource (extracted at query time)")
    api_name: Optional[str] = Field(None, description="API operation name")
    http_method: Optional[str] = Field(None, description="HTTP method (POST, PUT, DELETE)")
    path: Optional[str] = Field(None, description="API path")
    status_code: Optional[int] = Field(None, description="HTTP status code")
    start_time: Optional[int] = Field(None, description="Request start time (milliseconds since epoch)")
    end_time: Optional[int] = Field(None, description="Request end time (milliseconds since epoch)")
    duration_ms: Optional[int] = Field(None, description="Request duration in milliseconds (calculated)")
    request_data: Optional[str] = Field(None, description="Request data (JSON string)")
    response_data: Optional[str] = Field(None, description="Response data (JSON string)")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    ip_address: Optional[str] = Field(None, description="Client IP address")
    user_agent: Optional[str] = Field(None, description="User agent string")
    request_id: Optional[str] = Field(None, description="Request ID for tracking")
    created: Optional[datetime] = Field(None, description="Created timestamp")


class AuditLogList(PaginatedResponse):
    """
    List of audit logs with pagination
    """

    items: Optional[list[AuditLog]] = Field(None, description="Audit log entries")
