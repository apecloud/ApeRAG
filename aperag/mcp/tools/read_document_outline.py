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

"""D10 §A.6 — ``read_document_outline`` read primitive.

Highest-value primitive per inventory §C.3 gold mine. Agents call this
to decide which section / chunk to read next.

Strict call sequence:

1. ``resolve_authenticated_user()`` — D9 base
2. ``tenancy_gate(user, collection_id)`` — D9 base canonical SoT
3. ``authorization_gate(user, "read_document_outline")`` — D9 §2
4. compute parse_version (§E.2)
5. fetch + walk authoritative markdown — un-cached.
"""

from __future__ import annotations

from sqlalchemy import select

from aperag.cache import get_read_primitive_cache
from aperag.config import get_async_session
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
from aperag.mcp.tools._parsed_doc import (
    build_outline_from_markdown,
    read_parsed_markdown,
    resolve_parse_version,
)
from aperag.mcp.tools.schemas import DocumentOutline


async def read_document_outline(
    collection_id: str,
    document_id: str,
    *,
    max_depth: int = 6,
) -> DocumentOutline:
    """Read heading tree (table of contents) of a document.

    Per ``docs/modularization/d10-design-pack.md`` §A.6.
    """

    # 1. D9 base: authenticated user.
    user = await resolve_authenticated_user()

    # 2. D9 base: canonical tenancy gate.
    collection = await tenancy_gate(user, collection_id)

    # 3. D9 §2 three-level authorization.
    await authorization_gate(user, "read_document_outline")

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

    # 5. Fetch + walk authoritative markdown — D10.g cache-wrapped.
    # The cache only accelerates; tenancy/auth above are NEVER skipped
    # (§E.7 hard lock).
    cache = await get_read_primitive_cache()

    async def _compute() -> DocumentOutline:
        markdown = await read_parsed_markdown(document)
        headings = build_outline_from_markdown(markdown, max_depth=max_depth)
        return DocumentOutline(
            document_id=document.id,
            headings=headings,
            parse_version=parse_version,
        )

    return await cache.get_or_compute_outline(
        document_id=document.id,
        parse_version=parse_version,
        compute=_compute,
        model_cls=DocumentOutline,
    )


__all__ = ["read_document_outline"]
