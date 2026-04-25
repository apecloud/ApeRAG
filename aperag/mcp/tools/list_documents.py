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

Cursor / total_count: opaque base64 ``{"offset": int}`` for the first
cut; D10.e (#97 @Bryce) replaces the codec.
"""

from __future__ import annotations

import base64
import json
import mimetypes
from typing import Literal, Optional

from sqlalchemy import and_, func, select

from aperag.config import get_async_session
from aperag.domains.indexing.db.models import (
    DocumentIndex,
    DocumentIndexStatus,
)
from aperag.domains.knowledge_base.db.models import (
    Document,
    DocumentStatus,
)
from aperag.mcp.tools._d9_base import (
    authorization_gate,
    resolve_authenticated_user,
    tenancy_gate,
)
from aperag.mcp.tools.schemas import DocumentList, DocumentMetadata


def _decode_cursor(cursor: Optional[str]) -> int:
    """Decode the placeholder D10.c offset cursor.

    Returns 0 only when ``cursor`` is ``None`` or empty; raises
    ``ValueError`` on malformed cursor per §C explicit-not-silent. Bryce's
    D10.e cursor codec replaces this placeholder.

    # TODO(D10.e #97): replace with canonical CursorError after #97 integration
    """

    if cursor is None or cursor == "":
        return 0
    try:
        decoded = base64.urlsafe_b64decode(cursor.encode("ascii")).decode("utf-8")
        payload = json.loads(decoded)
    except Exception as exc:
        raise ValueError(f"cursor decode failed: {exc}") from exc
    if not isinstance(payload, dict) or "offset" not in payload:
        raise ValueError("cursor decode failed: missing 'offset' key")
    raw_offset = payload["offset"]
    if isinstance(raw_offset, bool) or not isinstance(raw_offset, int):
        raise ValueError("cursor decode failed: 'offset' must be a non-negative int")
    if raw_offset < 0:
        raise ValueError("cursor decode failed: 'offset' must be non-negative")
    return raw_offset


def _encode_cursor(offset: int) -> str:
    return base64.urlsafe_b64encode(json.dumps({"offset": offset}, separators=(",", ":")).encode("utf-8")).decode(
        "ascii"
    )


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
    offset = _decode_cursor(cursor)

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

        # Batch-fetch indexed chunk counts for the page.
        chunk_counts: dict[str, int] = {}
        if documents:
            doc_ids = [d.id for d in documents]
            idx_stmt = select(DocumentIndex).where(
                DocumentIndex.document_id.in_(doc_ids),
                DocumentIndex.status == DocumentIndexStatus.ACTIVE,
            )
            for idx_row in (await session.execute(idx_stmt)).scalars().all():
                count = 0
                if idx_row.index_data:
                    try:
                        data = json.loads(idx_row.index_data)
                        count = len(data.get("context_ids") or [])
                    except (TypeError, json.JSONDecodeError):
                        count = 0
                chunk_counts[idx_row.document_id] = max(chunk_counts.get(idx_row.document_id, 0), count)
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

    next_cursor = _encode_cursor(offset + len(items)) if (offset + len(items)) < total else None
    return DocumentList(items=items, next_cursor=next_cursor, total_count=total)


__all__ = ["list_documents"]
