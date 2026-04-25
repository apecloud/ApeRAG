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

"""D10 §A.5 — ``read_document`` read primitive.

Strict call sequence (per cuiwenbo msg=13a4139a + 明书 msg=c5e3be15):

1. ``resolve_authenticated_user()`` — D9 base
2. ``tenancy_gate(user, collection_id)`` — D9 base canonical SoT
3. ``authorization_gate(user, "read_document")`` — D9 §2
4. compute parse_version (§E.2)
5. fetch authoritative parsed markdown — un-cached.
"""

from __future__ import annotations

from typing import Optional

from sqlalchemy import select

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
    read_parsed_markdown,
    resolve_parse_version,
)
from aperag.mcp.tools.schemas import ByteRange, DocumentContent


async def read_document(
    collection_id: str,
    document_id: str,
    *,
    range: Optional[ByteRange] = None,
) -> DocumentContent:
    """Read parsed markdown content of a document.

    Per ``docs/modularization/d10-design-pack.md`` §A.5. ``range`` is
    optional / best-effort and NOT stable across re-parse (see §A.9).
    """

    # 1. D9 base: authenticated user.
    user = await resolve_authenticated_user()

    # 2. D9 base: canonical tenancy gate.
    collection = await tenancy_gate(user, collection_id)

    # 3. D9 §2 three-level authorization.
    await authorization_gate(user, "read_document")

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

    # 5. Fetch authoritative parsed markdown — un-cached body work.
    parsed_markdown = await read_parsed_markdown(document)

    truncated = False
    truncation_reason: Optional[str] = None
    if range is not None:
        encoded = parsed_markdown.encode("utf-8")
        start = max(0, int(range.start))
        end = min(len(encoded), int(range.end))
        if end < start:
            end = start
        try:
            sliced = encoded[start:end].decode("utf-8")
        except UnicodeDecodeError:
            # Best-effort: byte boundaries may split a multi-byte char.
            sliced = encoded[start:end].decode("utf-8", errors="replace")
        if start != 0 or end != len(encoded):
            truncated = True
            truncation_reason = f"byte_range[{start}:{end}]"
        parsed_markdown = sliced

    return DocumentContent(
        document_id=document.id,
        collection_id=document.collection_id,
        parsed_markdown=parsed_markdown,
        parse_version=parse_version,
        truncated=truncated,
        truncation_reason=truncation_reason,
    )


__all__ = ["read_document"]
