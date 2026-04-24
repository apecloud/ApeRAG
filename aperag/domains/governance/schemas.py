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

from pydantic import BaseModel, Field

from aperag.schema.common import PageResult, PaginatedResponse

__all__ = [
    "ApiKey",
    "ApiKeyList",
    "ApiKeyCreate",
    "ApiKeyUpdate",
    "AuditLog",
    "AuditLogList",
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
            "llm_provider",
            "llm_provider_model",
            "model_service_provider",
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


def _bind_view_models_reexports() -> None:
    """Phase 3 / Phase 4 dual-hook pattern — see identity/schemas.py
    for the full symmetric-load-order explanation."""

    import sys

    _vm = sys.modules.get("aperag.schema.view_models")
    if _vm is None:
        return
    for _name in __all__:
        setattr(_vm, _name, globals()[_name])


_bind_view_models_reexports()
