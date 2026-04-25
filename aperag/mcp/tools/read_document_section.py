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

"""D10 §A.7 — ``read_document_section`` read primitive.

Strict call sequence:

1. ``resolve_authenticated_user()`` — D9 base
2. ``tenancy_gate(user, collection_id)`` — D9 base canonical SoT
3. ``authorization_gate(user, "read_document_section")`` — D9 §2
4. compute parse_version (§E.2)
5. fetch + slice authoritative markdown — un-cached.
"""

from __future__ import annotations

from typing import Optional

from sqlalchemy import select

from aperag.config import get_async_session
from aperag.domains.knowledge_base.db.models import (
    Document,
    DocumentStatus,
)
from aperag.exceptions import DocumentNotFoundException, InvalidParameterException
from aperag.mcp.tools._d9_base import (
    authorization_gate,
    resolve_authenticated_user,
    tenancy_gate,
)
from aperag.mcp.tools._parsed_doc import (
    build_outline_from_markdown,
    find_section_in_outline,
    read_parsed_markdown,
    resolve_parse_version,
    slice_section_markdown,
)
from aperag.mcp.tools.handles import HeadingAnchor, SectionPath
from aperag.mcp.tools.schemas import DocumentSection


async def read_document_section(
    collection_id: str,
    document_id: str,
    *,
    section_path: Optional[SectionPath] = None,
    heading_anchor: Optional[HeadingAnchor] = None,
) -> DocumentSection:
    """Read content of a specific section by section_path or heading_anchor.

    Per ``docs/modularization/d10-design-pack.md`` §A.7. At least one of
    ``section_path`` / ``heading_anchor`` is required; if both are provided,
    server prefers ``section_path``.
    """

    if not section_path and not heading_anchor:
        raise InvalidParameterException("read_document_section requires section_path or heading_anchor")

    # 1. D9 base: authenticated user.
    user = await resolve_authenticated_user()

    # 2. D9 base: canonical tenancy gate.
    collection = await tenancy_gate(user, collection_id)

    # 3. D9 §2 three-level authorization.
    await authorization_gate(user, "read_document_section")

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

    # 5. Fetch + slice authoritative markdown.
    markdown = await read_parsed_markdown(document)
    outline = build_outline_from_markdown(markdown)

    heading = find_section_in_outline(
        outline,
        section_path=section_path,
        heading_anchor=heading_anchor,
    )
    if heading is None:
        raise DocumentNotFoundException(
            f"Section not found in document {document_id}: "
            f"section_path={section_path!r} heading_anchor={heading_anchor!r}"
        )

    section_md = slice_section_markdown(markdown, heading)
    parent_path: Optional[str] = None
    if "/" in heading.section_path:
        parent_path = "/".join(heading.section_path.split("/")[:-1])

    # Sibling count: peers at the same level in the parent's children list.
    sibling_count = 0
    if parent_path is None:
        sibling_count = max(0, len(outline) - 1)
    else:
        parent = find_section_in_outline(outline, section_path=parent_path)
        if parent is not None:
            sibling_count = max(0, len(parent.children) - 1)

    return DocumentSection(
        document_id=document.id,
        collection_id=document.collection_id,
        section_path=heading.section_path,
        heading_anchor=heading.heading_anchor,
        heading_text=heading.text,
        parsed_markdown=section_md,
        parse_version=parse_version,
        parent_section_path=parent_path,
        sibling_count=sibling_count,
    )


__all__ = ["read_document_section"]
