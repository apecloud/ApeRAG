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

"""D10 §A.3 — ``get_document_metadata`` read primitive.

Strict call sequence:

1. ``resolve_authenticated_user()`` — D9 base
2. ``tenancy_gate(user, collection_id)`` — D9 base canonical SoT
3. ``authorization_gate(user, "get_document_metadata")`` — D9 §2
4. fetch authoritative — un-cached.
"""

from __future__ import annotations

import json
import mimetypes

from sqlalchemy import select

from aperag.config import get_async_session
from aperag.domains.indexing.db.models import (
    DocumentIndex,
    DocumentIndexStatus,
)
from aperag.domains.knowledge_base.db.models import (
    Document,
    DocumentStatus,
)
from aperag.exceptions import DocumentNotFoundException
from aperag.mcp.tools._d9_base import (
    authorization_gate,
    resolve_authenticated_user,
    tenancy_gate,
)
from aperag.mcp.tools.list_documents import _DOCUMENT_STATUS_TO_INDEXING
from aperag.mcp.tools.schemas import DocumentMetadata


async def get_document_metadata(
    collection_id: str,
    document_id: str,
) -> DocumentMetadata:
    """Get metadata for a specific document.

    Per ``docs/modularization/d10-design-pack.md`` §A.3.
    """

    # 1. D9 base: authenticated user.
    user = await resolve_authenticated_user()

    # 2. D9 base: canonical tenancy gate (collection scope).
    await tenancy_gate(user, collection_id)

    # 3. D9 §2 three-level authorization.
    await authorization_gate(user, "get_document_metadata")

    # 4. Fetch authoritative — document must be tenant-owned + collection-scoped.
    async for session in get_async_session():
        stmt = select(Document).where(
            Document.id == document_id,
            Document.collection_id == collection_id,
            Document.user == str(user.id),
            Document.status != DocumentStatus.DELETED,
        )
        document = (await session.execute(stmt)).scalars().first()
        if not document:
            raise DocumentNotFoundException(document_id)

        chunk_count = 0
        idx_stmt = select(DocumentIndex).where(
            DocumentIndex.document_id == document_id,
            DocumentIndex.status == DocumentIndexStatus.ACTIVE,
        )
        for idx_row in (await session.execute(idx_stmt)).scalars().all():
            if idx_row.index_data:
                try:
                    data = json.loads(idx_row.index_data)
                    chunk_count = max(chunk_count, len(data.get("context_ids") or []))
                except (TypeError, json.JSONDecodeError):
                    continue
        break

    media_type, _ = mimetypes.guess_type(document.name or "")
    return DocumentMetadata(
        document_id=document.id,
        collection_id=document.collection_id,
        title=document.name or document.id,
        media_type=media_type or "application/octet-stream",
        size_bytes=int(document.size or 0),
        indexed_chunks_count=chunk_count,
        indexing_status=_DOCUMENT_STATUS_TO_INDEXING.get(document.status, "pending"),
        failure_reason=None,
        created_at=document.gmt_created,
        updated_at=document.gmt_updated,
        outline_summary=None,
    )


__all__ = ["get_document_metadata"]
