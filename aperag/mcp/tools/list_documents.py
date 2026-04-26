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

"""D10 §A.2 — ``list_documents`` read primitive.

Strict call sequence:

1. ``resolve_authenticated_user()`` — D9 base
2. ``tenancy_gate(user, collection_id)`` — D9 base canonical SoT
3. ``authorization_gate(user, "list_documents")`` — D9 §2
4. fetch authoritative — un-cached.

Cursor / total_count: opaque base64 cursor produced by
:mod:`aperag.service.pagination`, which wraps the canonical
:mod:`aperag.mcp.cursor` codec around the offset bookkeeping below.
Malformed / expired / scope-mismatched cursors surface as canonical
``CursorError`` per §C.3.
"""

from __future__ import annotations

import mimetypes
from typing import Literal, Optional

from sqlalchemy import and_, func, select

from aperag.config import get_async_session
from aperag.domains.knowledge_base.db.models import (
    Document,
    DocumentStatus,
)
from aperag.indexing.models import (
    DocumentIndex,
    IndexStatus,
)
from aperag.mcp.tools._d9_base import (
    authorization_gate,
    resolve_authenticated_user,
    tenancy_gate,
)
from aperag.mcp.tools.schemas import DocumentList, DocumentMetadata
from aperag.service.pagination import decode_offset_cursor, encode_offset_cursor

_DOCUMENT_STATUS_TO_INDEXING = {
    DocumentStatus.PENDING: "pending",
    DocumentStatus.RUNNING: "indexing",
    DocumentStatus.COMPLETE: "complete",
    DocumentStatus.FAILED: "failed",
    DocumentStatus.UPLOADED: "pending",
    DocumentStatus.EXPIRED: "failed",
    DocumentStatus.DELETED: "failed",
}


def _media_type_for(name: Optional[str]) -> str:
    if not name:
        return "application/octet-stream"
    mt, _ = mimetypes.guess_type(name)
    return mt or "application/octet-stream"


_SORT_COLS = {
    "created_at": Document.gmt_created,
    "title": Document.name,
    "size_bytes": Document.size,
}


async def list_documents(
    collection_id: str,
    *,
    cursor: Optional[str] = None,
    limit: int = 50,
    sort_by: Literal["created_at", "title", "size_bytes"] = "created_at",
    sort_order: Literal["asc", "desc"] = "desc",
    title_filter: Optional[str] = None,
    type_filter: Optional[list[str]] = None,
    indexed_only: bool = False,
) -> DocumentList:
    """List documents within a collection.

    Per ``docs/modularization/d10-design-pack.md`` §A.2.
    """

    # 1. D9 base: authenticated user.
    user = await resolve_authenticated_user()

    # 2. D9 base: canonical tenancy gate.
    await tenancy_gate(user, collection_id)

    # 3. D9 §2 three-level authorization.
    await authorization_gate(user, "list_documents")

    # 4. Fetch authoritative.
    limit = max(1, min(int(limit), 200))

    cursor_filters = {
        "title_filter": title_filter,
        "type_filter": sorted(t.lower() for t in type_filter) if type_filter else None,
        "indexed_only": bool(indexed_only),
        "sort_order": sort_order,
    }
    cursor_kwargs = dict(
        sort_key=sort_by,
        filters=cursor_filters,
        collection_id=collection_id,
        tenant_id=str(user.id),
    )
    offset = decode_offset_cursor(cursor, **cursor_kwargs)

    sort_col = _SORT_COLS.get(sort_by, Document.gmt_created)
    sort_clause = sort_col.asc() if sort_order == "asc" else sort_col.desc()

    base_filters = [
        Document.user == str(user.id),
        Document.collection_id == collection_id,
        Document.status != DocumentStatus.DELETED,
    ]
    if title_filter:
        base_filters.append(Document.name.ilike(f"%{title_filter}%"))
    if indexed_only:
        base_filters.append(Document.status == DocumentStatus.COMPLETE)

    async for session in get_async_session():
        if type_filter:
            # media_type is computed from filename via mimetypes.guess_type
            # — there is no Document.mimetype column to push the filter to
            # SQL. Fetch the whole filtered+sorted result set, apply the
            # mimetype filter in Python, THEN count / slice. This keeps
            # total_count, offset/limit, and next_cursor coherent (Weston
            # msg=246c84d3 二线 sanity).
            full_stmt = select(Document).where(and_(*base_filters)).order_by(sort_clause)
            all_rows = list((await session.execute(full_stmt)).scalars().all())
            allowed = {t.lower() for t in type_filter}
            filtered = [d for d in all_rows if _media_type_for(d.name).lower() in allowed]
            total = len(filtered)
            documents = filtered[offset : offset + limit]
        else:
            count_stmt = select(func.count()).select_from(Document).where(and_(*base_filters))
            total = int((await session.execute(count_stmt)).scalar() or 0)

            page_stmt = select(Document).where(and_(*base_filters)).order_by(sort_clause).offset(offset).limit(limit)
            documents = list((await session.execute(page_stmt)).scalars().all())

        # Wave 3 hard-cut: legacy ``DocumentIndex.index_data`` JSON
        # blob is gone (the new §F.1 schema decomposes that
        # information across per-modality rows + the ``derived/`` /
        # backend stores). Chunk count is no longer a per-document
        # scalar; surface 0 here to keep the page response shape
        # stable. Future T3.x can plumb a real chunk count through
        # ``chunks.jsonl`` artifact when an MCP client actually
        # consumes the field.
        chunk_counts: dict[str, int] = {}
        if documents:
            doc_ids = [d.id for d in documents]
            # Run a serving-row query so the §F.1 partial-unique
            # invariant is exercised through this caller path even
            # though we no longer compose chunk_count from it.
            idx_stmt = select(DocumentIndex.document_id).where(
                DocumentIndex.document_id.in_(doc_ids),
                DocumentIndex.status == IndexStatus.ACTIVE.value,
                DocumentIndex.is_serving.is_(True),
            )
            for doc_id in (await session.execute(idx_stmt)).scalars().all():
                chunk_counts.setdefault(doc_id, 0)
        break

    items = [
        DocumentMetadata(
            document_id=d.id,
            collection_id=d.collection_id,
            title=d.name or d.id,
            media_type=_media_type_for(d.name),
            size_bytes=int(d.size or 0),
            indexed_chunks_count=int(chunk_counts.get(d.id, 0)),
            indexing_status=_DOCUMENT_STATUS_TO_INDEXING.get(d.status, "pending"),
            failure_reason=None,
            created_at=d.gmt_created,
            updated_at=d.gmt_updated,
        )
        for d in documents
    ]

    next_offset = offset + len(items)
    next_cursor = encode_offset_cursor(offset=next_offset, **cursor_kwargs) if next_offset < total else None
    return DocumentList(items=items, next_cursor=next_cursor, total_count=total)


__all__ = ["list_documents"]
