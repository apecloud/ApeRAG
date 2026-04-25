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

"""Shared helpers for the 4 parse_version-keyed read primitives.

These helpers live behind the D9 base gates — they are NEVER called
before tenancy + authorization. Each public primitive composes the
D9 gates first, then calls into here for the un-cached body work.

Per ``docs/modularization/d10-design-pack.md`` §E.2 / §E.6:

- ``resolve_parse_version(document, collection)`` produces the 16-char
  hex composite of ``(parser_pipeline, document_md5, chunking_config)``.
- ``read_parsed_markdown(document)`` streams the persisted parsed.md
  blob from the object store (existing canonical location used by
  ``DocumentService.get_document_preview``).
- ``build_outline_from_markdown(markdown)`` walks the markdown headings
  to produce the §A.6 outline tree (no DocParser dependency — pure
  string scan over the already-parsed markdown).

Implementation note: ApeRAG's ``Document`` ORM does not yet carry a
stable ``parser_pipeline`` identifier. We derive a deterministic proxy
from the collection's parser config blob (``Collection.config``) so the
``parse_version`` is stable for unchanged config + content; D10.g (#99)
re-revisits this when the index/parser metadata is promoted into a
first-class column. Flagged in the PR body for follow-up.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Optional

from aperag.mcp.tools.parse_version import compute_parse_version
from aperag.mcp.tools.schemas import OutlineHeading


def _stable_collection_parser_signature(collection) -> str:
    """Produce a deterministic parser-pipeline signature from
    ``Collection.config``.

    Hashes the config JSON (sorted keys) so any change in parser /
    chunking / embedding config produces a fresh ``parse_version``.
    """

    blob = collection.config or "{}"
    try:
        canonical = json.dumps(json.loads(blob), sort_keys=True, separators=(",", ":"))
    except (TypeError, json.JSONDecodeError):
        canonical = blob
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _stable_chunking_config(collection) -> str:
    """Extract chunking-relevant slice of the collection config.

    Falls back to the full config signature when no obvious chunking
    knobs are present so a chunking-config change still rolls
    parse_version even if the JSON shape evolves.
    """

    blob = collection.config or "{}"
    try:
        cfg = json.loads(blob)
    except (TypeError, json.JSONDecodeError):
        return "default"
    chunk_block = {k: cfg.get(k) for k in ("chunk_size", "chunk_overlap", "chunking_strategy", "tokenizer") if k in cfg}
    if not chunk_block:
        return "default"
    return json.dumps(chunk_block, sort_keys=True, separators=(",", ":"))


def resolve_parse_version(document, collection) -> str:
    """Compute the §E.2 ``parse_version`` for ``document``.

    Inputs (sourced from existing ORM rows — no schema change):

    - ``parser_pipeline`` ← sha256(Collection.config sorted JSON)[:16]
    - ``document_md5``    ← ``Document.content_hash`` (NULL → fallback to
      ``id|gmt_updated.isoformat()`` digest so the value is still
      deterministic for unchanged rows).
    - ``chunking_config`` ← sliced + canonicalized chunking config; or
      the empty literal ``"default"`` when none is present.
    """

    parser_pipeline = _stable_collection_parser_signature(collection)
    md5 = document.content_hash
    if not md5:
        proxy = f"{document.id}|{document.gmt_updated.isoformat() if document.gmt_updated else ''}"
        md5 = hashlib.md5(proxy.encode("utf-8")).hexdigest()
    chunking = _stable_chunking_config(collection)
    return compute_parse_version(
        parser_pipeline=parser_pipeline,
        document_md5=md5,
        chunking_config=chunking,
    )


async def read_parsed_markdown(document) -> str:
    """Return the parsed markdown for ``document`` from the object store.

    Returns an empty string when the parsed.md blob is missing — that
    matches the behavior of ``DocumentService.get_document_preview``
    (warns + continues with empty content rather than 500ing).
    """

    from aperag.objectstore.base import get_async_object_store

    async_obj_store = get_async_object_store()
    markdown_path = f"{document.object_store_base_path()}/parsed.md"
    try:
        result = await async_obj_store.get(markdown_path)
    except Exception:
        return ""
    if not result:
        return ""
    md_stream, _ = result
    content = b""
    async for data in md_stream:
        content += data
    try:
        return content.decode("utf-8")
    except UnicodeDecodeError:
        return content.decode("utf-8", errors="replace")


# --- outline derivation ------------------------------------------------------


_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)


def _slugify(text: str) -> str:
    """Best-effort slug for ``heading_anchor``. Stable per-text only."""

    s = text.lower().strip()
    s = re.sub(r"[^\w\s\-一-鿿]+", "", s)
    s = re.sub(r"\s+", "-", s)
    return f"#{s}" if s else "#section"


def build_outline_from_markdown(markdown: str, max_depth: int = 6) -> list[OutlineHeading]:
    """Walk ``markdown`` and produce the §A.6 outline tree.

    ``section_path`` is a slash-separated counter per level
    (e.g. ``"1/2/3"``) — the LOCKED primary stable handle per §A.9.
    Headings beyond ``max_depth`` are skipped.
    """

    if not markdown:
        return []
    matches = _HEADING_RE.findall(markdown)
    if not matches:
        return []

    # Counter per heading level so siblings get monotonic ids; deeper
    # levels reset on a new parent.
    flat: list[OutlineHeading] = []
    counters = [0] * 7  # indexes 1..6
    for hashes, raw_text in matches:
        level = len(hashes)
        if level > max_depth:
            continue
        counters[level] += 1
        # Reset deeper-level counters whenever a shallower heading appears.
        for deeper in range(level + 1, 7):
            counters[deeper] = 0
        path_parts = [str(counters[i]) for i in range(1, level + 1) if counters[i] > 0]
        section_path = "/".join(path_parts) if path_parts else str(counters[level])
        flat.append(
            OutlineHeading(
                level=level,
                text=raw_text.strip(),
                section_path=section_path,
                heading_anchor=_slugify(raw_text),
                chunk_id=None,
                children=[],
            )
        )

    # Build the tree from the flat list using path prefixes.
    by_path: dict[str, OutlineHeading] = {h.section_path: h for h in flat}
    roots: list[OutlineHeading] = []
    for heading in flat:
        parts = heading.section_path.split("/")
        if len(parts) == 1:
            roots.append(heading)
            continue
        parent_path = "/".join(parts[:-1])
        parent = by_path.get(parent_path)
        if parent:
            parent.children.append(heading)
        else:
            roots.append(heading)
    return roots


def find_section_in_outline(
    outline: list[OutlineHeading],
    *,
    section_path: Optional[str] = None,
    heading_anchor: Optional[str] = None,
) -> Optional[OutlineHeading]:
    """DFS for the first heading that matches ``section_path`` (preferred)
    or ``heading_anchor`` (fallback). Returns ``None`` on miss."""

    for heading in outline:
        if section_path and heading.section_path == section_path:
            return heading
        if heading_anchor and heading.heading_anchor == heading_anchor:
            return heading
        nested = find_section_in_outline(
            heading.children,
            section_path=section_path,
            heading_anchor=heading_anchor,
        )
        if nested is not None:
            return nested
    return None


def slice_section_markdown(markdown: str, heading: OutlineHeading) -> str:
    """Return the markdown body for ``heading`` — from its heading line up
    to (but not including) the next heading at the same or shallower level.
    """

    if not markdown:
        return ""
    pattern = re.compile(rf"^#{{{heading.level}}}\s+{re.escape(heading.text)}\s*$", re.MULTILINE)
    match = pattern.search(markdown)
    if not match:
        return ""
    start = match.start()
    # Find the next heading at same-or-shallower level.
    next_pattern = re.compile(r"^(#{1," + str(heading.level) + r"})\s+", re.MULTILINE)
    tail = markdown[match.end() :]
    next_match = next_pattern.search(tail)
    end = match.end() + next_match.start() if next_match else len(markdown)
    return markdown[start:end]


__all__ = [
    "build_outline_from_markdown",
    "find_section_in_outline",
    "read_parsed_markdown",
    "resolve_parse_version",
    "slice_section_markdown",
]
