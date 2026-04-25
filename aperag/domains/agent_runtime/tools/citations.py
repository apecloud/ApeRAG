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

"""RAG tool result -> Anthropic-shape citation transform (D8 §2.5).

The agent runtime gets back tool results in a variety of shapes
(``search_collection`` returns ``{items: [{content, metadata,
score, ...}]}``, ``web_search`` returns ``{results: [{title, url,
snippet, ...}]}``, ``read_document`` returns its own struct). The
existing wire translator already turns
``artifact.created(reference_bundle)`` envelopes into
``source-url`` + ``data-citation`` parts, but it does so by
inferring locations from item metadata in a best-effort, default-
to-``url_citation`` way (per
``aperag/domains/agent_runtime/wire/translator.py::_build_citation_location``).

This module owns the **typed** Anthropic-shape transform:

* ``Char location`` -- when the source carries ``start_char_index``
  + ``end_char_index`` (RAG chunk-level snippets land here).
* ``Page location`` -- when the source carries ``page_number`` (PDF
  / paginated docs).
* ``Content block location`` -- when the source carries
  ``content_block_index`` (structured doc parsers).
* ``URL citation`` -- the fallback (web search results, plain URLs).

The module is consumed by the runtime when it converts a RAG tool
result into a ``reference_bundle`` artifact: instead of letting the
translator's heuristic pick the location type, the runtime builds
typed :class:`CitationData` instances directly so the wire frame is
deterministic and forward-compatible (no future translator change
can silently downgrade a known page-citation to a url-citation).

Inner data classes (``CitationData`` + ``CitationLocation``) come
from :mod:`aperag.domains.agent_runtime.uimessage` to enforce
single-source-of-truth on the wire/at-rest shape (per D8 §2 same-
schema canonical).
"""

from __future__ import annotations

from typing import Any, Optional

from aperag.domains.agent_runtime.uimessage import (
    CharLocation,
    CitationData,
    CitationLocation,
    ContentBlockLocation,
    PageLocation,
    UrlCitationLocation,
)


def build_citation(
    *,
    cited_text: str,
    location_metadata: dict[str, Any],
) -> CitationData:
    """Build one ``CitationData`` from a snippet + source metadata.

    The metadata schema is the loose ``ReferenceBundleItem.metadata``
    bag the runtime already populates. Field probing order:

    1. ``start_char_index`` + ``end_char_index`` -> ``CharLocation``
    2. ``page_number`` (or ``page``) -> ``PageLocation``
    3. ``content_block_index`` -> ``ContentBlockLocation``
    4. ``url`` (or any URL-shaped field) -> ``UrlCitationLocation``
    5. else -> ``UrlCitationLocation`` with empty url (the fallback
       so the part still validates; FE renderer treats empty url as
       "metadata-only citation" per D8 §2.5)
    """

    location = _detect_location(location_metadata)
    return CitationData(cited_text=cited_text, location=location)


def transform_reference_bundle_items(
    items: list[dict[str, Any]],
) -> list[CitationData]:
    """Project a ``ReferenceBundleItem``-shaped list into typed citations.

    Convenience wrapper for the runtime's reference-bundle artifact
    construction path. Skips items with no usable snippet (the wire
    spec requires ``cited_text`` to be non-empty; a blank snippet
    would fail Pydantic validation downstream).
    """

    out: list[CitationData] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        snippet = item.get("snippet") or item.get("content") or ""
        if not snippet:
            continue
        metadata = dict(item.get("metadata") or {})
        # Promote a top-level ``uri`` so callers don't have to nest
        # it inside ``metadata`` for every search-result shape.
        if "uri" in item and "url" not in metadata:
            metadata["url"] = item["uri"]
        if "title" in item and "title" not in metadata:
            metadata["title"] = item["title"]
        out.append(build_citation(cited_text=str(snippet), location_metadata=metadata))
    return out


def _detect_location(metadata: dict[str, Any]) -> CitationLocation:
    title = _string_or_none(metadata.get("title"))
    doc_index = _int_or_default(metadata.get("doc_index"), 0)

    start = metadata.get("start_char_index")
    end = metadata.get("end_char_index")
    if isinstance(start, int) and isinstance(end, int):
        return CharLocation(
            doc_index=doc_index,
            doc_title=title,
            start_char=start,
            end_char=end,
        )

    page_number = metadata.get("page_number") or metadata.get("page")
    if isinstance(page_number, int):
        return PageLocation(
            doc_index=doc_index,
            doc_title=title,
            page_index=page_number,
        )

    block_index = metadata.get("content_block_index")
    if isinstance(block_index, int):
        return ContentBlockLocation(
            doc_index=doc_index,
            doc_title=title,
            block_index=block_index,
        )

    url = _string_or_none(metadata.get("url")) or ""
    return UrlCitationLocation(url=url, title=title)


def _string_or_none(value: Any) -> Optional[str]:
    return value if isinstance(value, str) and value else None


def _int_or_default(value: Any, default: int) -> int:
    return value if isinstance(value, int) else default


__all__ = [
    "build_citation",
    "transform_reference_bundle_items",
]
