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

"""D10 §A.1 — ``list_collections`` read primitive.

Strict call sequence (per cuiwenbo msg=13a4139a + 明书 msg=c5e3be15):

1. ``resolve_authenticated_user()`` — D9 base
2. ``authorization_gate(user, "list_collections")`` — D9 §2 3-level
3. fetch authoritative — un-cached; D10.g (#99 @明书) wraps with cache

No tenancy gate here because the primitive is naturally tenant-scoped:
``async_db_ops.query_collections([user_id])`` only returns the caller's
own collections (§F.4 lock — server ignores any client-asserted scope).

Cursor / total_count: opaque base64 cursor produced by
:mod:`aperag.service.pagination`, which wraps the canonical
:mod:`aperag.mcp.cursor` codec around the offset bookkeeping below.
Malformed / expired / scope-mismatched cursors surface as canonical
``CursorError`` per §C.3.
"""

from __future__ import annotations

from typing import Literal, Optional

from aperag.db.ops import async_db_ops
from aperag.mcp.tools._d9_base import authorization_gate, resolve_authenticated_user
from aperag.mcp.tools.schemas import CollectionList, CollectionMetadata
from aperag.service.pagination import decode_offset_cursor, encode_offset_cursor


def _sort_key(c, sort_by: str):
    if sort_by == "title":
        return c.title or ""
    if sort_by == "updated_at":
        return c.gmt_updated
    # default: created_at
    return c.gmt_created


async def list_collections(
    *,
    cursor: Optional[str] = None,
    limit: int = 50,
    sort_by: Literal["created_at", "updated_at", "title"] = "created_at",
    sort_order: Literal["asc", "desc"] = "desc",
    title_filter: Optional[str] = None,
) -> CollectionList:
    """List collections accessible by current user (tenant-scoped).

    Per ``docs/modularization/d10-design-pack.md`` §A.1.
    """

    # 1. D9 base: resolve authenticated user (raises on missing/invalid key).
    user = await resolve_authenticated_user()

    # 2. D9 §2 three-level authorization gate (READ_ONLY → auto-invoke).
    await authorization_gate(user, "list_collections")

    # 3. Fetch authoritative — tenant-scoped by construction.
    rows = await async_db_ops.query_collections([str(user.id)])

    # Apply title filter (case-insensitive contains) before pagination.
    if title_filter:
        needle = title_filter.lower()
        rows = [c for c in rows if (c.title or "").lower().find(needle) >= 0]

    rows = list(rows)
    rows.sort(key=lambda c: _sort_key(c, sort_by) or 0, reverse=(sort_order == "desc"))

    total = len(rows)
    limit = max(1, min(int(limit), 200))

    cursor_filters = {
        "title_filter": title_filter,
        "sort_order": sort_order,
    }
    cursor_kwargs = dict(
        sort_key=sort_by,
        filters=cursor_filters,
        collection_id=None,
        tenant_id=str(user.id),
    )
    offset = decode_offset_cursor(cursor, **cursor_kwargs)
    page = rows[offset : offset + limit]

    items = [
        CollectionMetadata(
            collection_id=c.id,
            title=c.title,
            description=c.description,
            created_at=c.gmt_created,
            updated_at=c.gmt_updated,
        )
        for c in page
    ]

    next_offset = offset + len(items)
    next_cursor = encode_offset_cursor(offset=next_offset, **cursor_kwargs) if next_offset < total else None
    return CollectionList(items=items, next_cursor=next_cursor, total_count=total)


__all__ = ["list_collections"]
