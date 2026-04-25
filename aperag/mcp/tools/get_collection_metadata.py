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

"""D10 §A.4 — ``get_collection_metadata`` read primitive.

Strict call sequence:

1. ``resolve_authenticated_user()`` — D9 base
2. ``tenancy_gate(user, collection_id)`` — D9 base canonical SoT
3. ``authorization_gate(user, "get_collection_metadata")`` — D9 §2
4. fetch authoritative — un-cached; D10.g wraps with cache.
"""

from __future__ import annotations

import json
from typing import Optional

from sqlalchemy import and_, func, select

from aperag.config import get_async_session
from aperag.domains.knowledge_base.db.models import (
    Document,
    DocumentStatus,
)
from aperag.mcp.tools._d9_base import (
    authorization_gate,
    resolve_authenticated_user,
    tenancy_gate,
)
from aperag.mcp.tools.schemas import CollectionDetailMetadata


def _index_modes_from_config(config_blob: Optional[str]) -> list[str]:
    """Extract enabled index modes from a Collection.config JSON blob.

    Best-effort: if the blob is missing or malformed, returns an empty
    list rather than raising. The §A.4 spec treats this as informational.
    """

    if not config_blob:
        return []
    try:
        cfg = json.loads(config_blob)
    except (TypeError, json.JSONDecodeError):
        return []
    modes: list[str] = []
    # Common collection-config fields per existing CollectionService:
    if cfg.get("enable_vector_index", True):
        modes.append("vector")
    if cfg.get("enable_fulltext_index", True):
        modes.append("fulltext")
    if cfg.get("enable_graph_index"):
        modes.append("graph")
    if cfg.get("enable_summary"):
        modes.append("summary")
    if cfg.get("enable_vision_index"):
        modes.append("vision")
    return modes


async def get_collection_metadata(
    collection_id: str,
) -> CollectionDetailMetadata:
    """Get metadata for a specific collection.

    Per ``docs/modularization/d10-design-pack.md`` §A.4.
    """

    # 1. D9 base: authenticated user.
    user = await resolve_authenticated_user()

    # 2. D9 base: canonical tenancy gate (raises CollectionNotFoundException).
    collection = await tenancy_gate(user, collection_id)

    # 3. D9 §2 three-level authorization (READ_ONLY → auto-invoke).
    await authorization_gate(user, "get_collection_metadata")

    # 4. Fetch authoritative — count active documents in the collection.
    async for session in get_async_session():
        stmt = (
            select(func.count())
            .select_from(Document)
            .where(
                and_(
                    Document.collection_id == collection_id,
                    Document.status != DocumentStatus.DELETED,
                )
            )
        )
        document_count = int((await session.execute(stmt)).scalar() or 0)
        break

    return CollectionDetailMetadata(
        collection_id=collection.id,
        title=collection.title,
        description=collection.description,
        document_count=document_count,
        index_modes_available=_index_modes_from_config(collection.config),
        permission_model="owner",
        created_at=collection.gmt_created,
        updated_at=collection.gmt_updated,
    )


__all__ = ["get_collection_metadata"]
