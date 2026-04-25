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

"""D10 §A.8 — ``read_document_chunk`` read primitive.

``chunk_id`` is immutable per §E.6 — no parse_version weighting needed
for the cache key, but the response envelope still carries
``parse_version`` of the document the chunk belongs to.

Strict call sequence:

1. ``resolve_authenticated_user()`` — D9 base
2. ``tenancy_gate(user, collection_id)`` — D9 base canonical SoT
3. ``authorization_gate(user, "read_document_chunk")`` — D9 §2
4. compute parse_version (§E.2)
5. fetch authoritative chunk — un-cached.
"""

from __future__ import annotations

from sqlalchemy import select

from aperag.cache import get_read_primitive_cache
from aperag.config import get_async_session
from aperag.domains.knowledge_base.db.models import (
    Document,
    DocumentStatus,
)
from aperag.domains.knowledge_base.service.document_service import document_service
from aperag.exceptions import DocumentNotFoundException
from aperag.mcp.tools._d9_base import (
    authorization_gate,
    resolve_authenticated_user,
    tenancy_gate,
)
from aperag.mcp.tools._parsed_doc import resolve_parse_version
from aperag.mcp.tools.handles import ChunkId
from aperag.mcp.tools.schemas import DocumentChunk


async def read_document_chunk(
    collection_id: str,
    document_id: str,
    chunk_id: ChunkId,
) -> DocumentChunk:
    """Read content of a specific chunk by stable ``chunk_id``.

    Per ``docs/modularization/d10-design-pack.md`` §A.8.
    """

    # 1. D9 base: authenticated user.
    user = await resolve_authenticated_user()

    # 2. D9 base: canonical tenancy gate.
    collection = await tenancy_gate(user, collection_id)

    # 3. D9 §2 three-level authorization.
    await authorization_gate(user, "read_document_chunk")

    # 4. Resolve document + compute parse_version.
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
        break

    parse_version = resolve_parse_version(document, collection)

    # 5. Fetch authoritative chunk — D10.g cache-wrapped.
    # §E.6: chunk_id is indexing-immutable, so the cache key is
    # ``(chunk_id,)`` only — no parse_version weighting. Tenancy/auth
    # above are NEVER skipped (§E.7 hard lock).
    cache = await get_read_primitive_cache()

    async def _compute() -> DocumentChunk:
        # Delegate to the existing tenant-aware service-layer helper which
        # routes through the vector-store connector with the multi-tenant
        # guard. We then locate the requested chunk by id within the result.
        all_chunks = await document_service.get_document_chunks(str(user.id), collection_id, document_id)
        matched = next((c for c in all_chunks if c.id == chunk_id), None)
        if matched is None:
            raise DocumentNotFoundException(f"Chunk not found in document {document_id}: chunk_id={chunk_id!r}")

        chunk_index = 0
        for i, c in enumerate(all_chunks):
            if c.id == chunk_id:
                chunk_index = i
                break

        metadata = matched.metadata or {}
        return DocumentChunk(
            chunk_id=matched.id,
            document_id=document.id,
            collection_id=document.collection_id,
            parsed_markdown=matched.text or "",
            section_path=metadata.get("section_path"),
            chunk_index=chunk_index,
            chunk_total=len(all_chunks),
            parse_version=parse_version,
        )

    return await cache.get_or_compute_chunk(
        chunk_id=chunk_id,
        compute=_compute,
        model_cls=DocumentChunk,
    )


__all__ = ["read_document_chunk"]
