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

"""Canonical Pydantic view models for the ``marketplace`` domain.

Phase 4 Step 4-S3c carves the published-collection surface schemas
out of ``aperag.schema.view_models``. Dual-hook symmetric re-export
keeps pre-migration callers working.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field

__all__ = [
    "SharedCollectionConfig",
    "SharedCollection",
    "SharedCollectionList",
]


class SharedCollectionConfig(BaseModel):
    """
    Configuration settings for shared collection features
    """

    enable_vector: bool = Field(..., description="Whether vector search is enabled")
    enable_fulltext: bool = Field(..., description="Whether fulltext search is enabled")
    enable_knowledge_graph: bool = Field(..., description="Whether knowledge graph is enabled")
    enable_summary: bool = Field(..., description="Whether summary generation is enabled")
    enable_vision: bool = Field(..., description="Whether vision processing is enabled")


class SharedCollection(BaseModel):
    """
    Shared Collection information for marketplace users
    """

    id: str = Field(..., description="Collection ID")
    title: str = Field(..., description="Collection title")
    description: Optional[str] = Field(None, description="Collection description")
    owner_user_id: str = Field(..., description="Original owner user ID")
    owner_username: Optional[str] = Field(None, description="Original owner username")
    subscription_id: Optional[str] = Field(
        None,
        description="Subscription record ID (has value if subscribed, null if not subscribed)",
    )
    gmt_subscribed: Optional[datetime] = Field(None, description="Subscription time (only has value when subscribed)")
    subscription_count: int = Field(..., description="Total number of subscriptions")
    config: SharedCollectionConfig = Field(..., description="Collection configuration settings")


class SharedCollectionList(BaseModel):
    """
    Shared Collection list response
    """

    items: list[SharedCollection] = Field(..., description="List of shared Collections")
    total: int = Field(..., description="Total count (for pagination)")
    page: int = Field(..., description="Current page number")
    page_size: int = Field(..., description="Page size")


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
