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

"""Parser entry point — celery T1.1 Foundation.

Per ``docs/modularization/indexing-redesign-design-pack.md`` §C.1, the
parser is the producer of the **shared** derived artifacts that every
modality consumes:

- ``markdown.md`` — full-text markdown rendition of the source file.
- ``outline.json`` — heading tree (``OutlineHeading`` per the D10.c §A.6
  schema; reused so search-result navigation back to a section works
  out of the box).
- ``chunks.jsonl`` — one chunk per line, the canonical chunk_id /
  text / section_path payload that vector + fulltext both consume
  (§C.6 conscious trade-off + future shadow split point).

The parser does NOT call any modality-specific embeddings or LLM
extraction. Those are owned by the per-modality ``derive`` workers
(T1.3 / T1.4 / T1.2). This split is what makes idempotent retry cheap
(§C.3): a failed Qdrant sync re-reads ``chunks.jsonl`` instead of
re-running the parser.

T1.1 ships the **interface and a deterministic in-process simulator**
so downstream modalities can wire their tests against a real
``derived/`` layout. Production parser integration (docparser /
Marker / OCR) is intentionally deferred to T2.x — at that point the
parser body becomes a thin shim that calls the existing parsing
pipeline and emits the same artifacts. The simulator proves the
write contract (atomic visibility + parse_version stability +
round-trip fidelity); the real parser will inherit that contract
unchanged.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from typing import Any

from aperag.indexing.object_store import (
    derived_artifact,
    read_or_none,
    write_atomic,
)
from aperag.objectstore.base import ObjectStore as _SyncObjectStore

# Wave 3 T3.1 chunk 2: ``compute_parse_version`` is imported lazily
# inside ``parse_document`` to avoid pulling the entire ``aperag.mcp``
# package (server + tool registry) at module load. Loading mcp.tools.*
# at this level was the root of two circular imports
# (``knowledge_base.db.models`` and ``db.ops``) that surfaced when the
# Wave 3 hard-cut deleted the legacy indexing layer's stub re-exports.

logger = logging.getLogger(__name__)


# Default parser identifier for the T1.1 simulator. Production parsers
# pass their own pipeline identifier (e.g., ``"docparser-v1"``) via
# ``ParseConfig.parser_pipeline`` so the parse_version rolls when the
# parsing pipeline changes.
DEFAULT_PARSER_PIPELINE = "indexing-simulator-v1"

# Default chunking knobs for the simulator. Real callers should pass
# the collection-specific ``ChunkingConfig`` so a chunking change
# rolls the parse_version (per design pack §E.2 hash inputs).
DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 80


@dataclass(frozen=True)
class ChunkingConfig:
    """Chunking knobs that participate in the parse_version hash.

    Stored as a deterministic ``serialize()`` string fed into
    :func:`aperag.mcp.tools.parse_version.compute_parse_version` so any
    knob change rolls the parse_version and triggers a fresh
    ``derived/`` directory.
    """

    chunk_size: int = DEFAULT_CHUNK_SIZE
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
    strategy: str = "section-aware"

    def serialize(self) -> str:
        return f"size={self.chunk_size}|overlap={self.chunk_overlap}|strategy={self.strategy}"


@dataclass(frozen=True)
class ParseConfig:
    """Inputs to :func:`parse_document` that influence parse_version.

    Two parses with the same ``ParseConfig`` and the same source bytes
    produce the same parse_version (and therefore the same
    ``derived/`` directory) — the basis of the §C.3 "idempotent retry
    is trivial" guarantee.
    """

    parser_pipeline: str = DEFAULT_PARSER_PIPELINE
    chunking: ChunkingConfig = ChunkingConfig()


@dataclass(frozen=True)
class ParseResult:
    """Outcome of :func:`parse_document`.

    ``parse_version`` pins the (parser, content, chunking) triple so
    callers can persist it on the ``DocumentIndex`` row. The three
    artifact paths are the canonical names downstream modalities
    expect to find under ``derived/parse_<version>/``.
    """

    parse_version: str
    markdown_path: str
    outline_path: str
    chunks_path: str


# ---------------------------------------------------------------------
# Internals — markdown → outline + chunks
# ---------------------------------------------------------------------


_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)


def _slugify_anchor(text: str) -> str:
    """Best-effort slug for ``OutlineHeading.heading_anchor``.

    Mirrors the D10.c §A.6 anchor convention (lowercase, ASCII letters
    + digits + hyphens; CJK characters preserved). T1.2 graph and
    later read-path tooling depend on this anchor format.
    """
    lowered = text.strip().lower()
    # Keep ASCII alnum, replace everything else with '-'; collapse
    # multiple hyphens and strip edges. CJK characters fall through
    # because ``str.lower`` is identity for them and the regex below
    # treats them as word characters via the unicode category check.
    pieces: list[str] = []
    for ch in lowered:
        if ch.isalnum():
            pieces.append(ch)
        elif ch.isspace() or ch in "-_":
            pieces.append("-")
        # Any other punctuation drops out.
    slug = "".join(pieces)
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug.strip("-") or "section"


def _build_outline(markdown: str) -> list[dict[str, Any]]:
    """Build the outline tree from markdown headings.

    The shape mirrors :class:`aperag.mcp.tools.schemas.OutlineHeading`
    so downstream read primitives can hand it back to clients
    unchanged. ``section_path`` is a slash-separated counter (e.g.,
    ``"1/2/1"``) per the D10.c §A.9 R1 lock.
    """
    headings = list(_HEADING_RE.finditer(markdown))
    if not headings:
        return []

    # We build the tree with a level stack: each entry is a tuple of
    # (level, current_counter_at_that_level, list_of_children).
    root: list[dict[str, Any]] = []
    stack: list[tuple[int, list[dict[str, Any]], list[int]]] = [(0, root, [])]
    seen_anchors: dict[str, int] = {}

    for match in headings:
        level = len(match.group(1))
        text = match.group(2).strip()

        # Pop deeper levels off the stack until we are at a parent.
        while stack and stack[-1][0] >= level:
            stack.pop()
        if not stack:
            stack.append((0, root, []))

        parent_level, parent_children, parent_path_counter = stack[-1]

        # Bump our slot at this level.
        slot = len(parent_children) + 1
        section_path = "/".join(str(c) for c in parent_path_counter + [slot])

        anchor = _slugify_anchor(text)
        # Disambiguate duplicate anchors with a numeric suffix.
        bump = seen_anchors.get(anchor, 0)
        if bump:
            heading_anchor = f"{anchor}-{bump}"
        else:
            heading_anchor = anchor
        seen_anchors[anchor] = bump + 1

        node: dict[str, Any] = {
            "level": level,
            "text": text,
            "section_path": section_path,
            "heading_anchor": heading_anchor,
            "chunk_id": None,
            "children": [],
        }
        parent_children.append(node)
        stack.append((level, node["children"], parent_path_counter + [slot]))

    return root


def _split_chunks(markdown: str, chunking: ChunkingConfig) -> list[dict[str, Any]]:
    """Split the parsed markdown into chunks following the outline order.

    The simulator uses a deterministic byte-window strategy: walk the
    markdown, slice into windows of approximately ``chunk_size``
    characters honouring the nearest paragraph break, with
    ``chunk_overlap`` characters of carry-over for context. The
    chunk_id is ``"<sha256-prefix>:<index>"`` so the id is stable
    across retries (depends only on content + chunking).

    Real parser integration (T2.x) replaces this with a tokeniser-
    aware splitter; the chunk record schema is the contract that must
    not change.
    """
    text = markdown.strip()
    if not text:
        return []

    chunks: list[dict[str, Any]] = []
    cursor = 0
    chunk_index = 0
    content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

    while cursor < len(text):
        # Aim for ``chunk_size`` characters, but back off to the
        # nearest paragraph break so a chunk does not slice mid-
        # sentence when we can avoid it.
        end = min(cursor + chunking.chunk_size, len(text))
        if end < len(text):
            paragraph_break = text.rfind("\n\n", cursor, end)
            if paragraph_break > cursor:
                end = paragraph_break

        chunk_text = text[cursor:end].strip()
        if chunk_text:
            chunks.append(
                {
                    "chunk_id": f"{content_hash}:{chunk_index:04d}",
                    "text": chunk_text,
                    "section_path": None,
                    "heading_anchor": None,
                    "page_idx": None,
                }
            )
            chunk_index += 1

        if end >= len(text):
            break
        cursor = max(end - chunking.chunk_overlap, cursor + 1)

    return chunks


# ---------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------


def _document_md5(source_bytes: bytes) -> str:
    return hashlib.md5(source_bytes).hexdigest()


def parse_document(
    *,
    store: _SyncObjectStore,
    collection_id: str,
    document_id: str,
    source_bytes: bytes,
    config: ParseConfig | None = None,
) -> ParseResult:
    """Parse the source bytes and persist the three shared artifacts.

    Idempotent at the ``parse_version`` level: re-running with
    identical inputs produces identical artifacts (and overwrites
    them atomically). The artifact paths are the canonical
    ``derived/parse_<version>/`` layout per design pack §C.1.

    Args:
        store: Object store handle (``LocalObjectStore`` /
            ``S3ObjectStore`` / :class:`InMemoryObjectStore` for tests).
        collection_id: Owning collection.
        document_id: Document being parsed.
        source_bytes: Raw bytes of the source document. The simulator
            interprets these as UTF-8 markdown; production parsers
            would invoke docparser / Marker / OCR here.
        config: Parsing knobs that influence the parse_version. Pass
            ``None`` to use simulator defaults.
    """
    from aperag.mcp.tools.parse_version import compute_parse_version

    cfg = config or ParseConfig()

    document_md5 = _document_md5(source_bytes)
    parse_version = compute_parse_version(
        parser_pipeline=cfg.parser_pipeline,
        document_md5=document_md5,
        chunking_config=cfg.chunking.serialize(),
    )

    try:
        markdown = source_bytes.decode("utf-8")
    except UnicodeDecodeError:
        # The simulator assumes UTF-8 markdown. Production parsers
        # convert binary inputs first; that conversion lives outside
        # this T1.1 surface.
        raise ValueError(
            "indexing simulator parser only handles UTF-8 markdown bodies; "
            "wire docparser/Marker before T2.x for non-text documents"
        )

    outline = _build_outline(markdown)
    chunks = _split_chunks(markdown, cfg.chunking)

    markdown_path = derived_artifact(
        collection_id=collection_id,
        document_id=document_id,
        parse_version=parse_version,
        filename="markdown.md",
    )
    outline_path = derived_artifact(
        collection_id=collection_id,
        document_id=document_id,
        parse_version=parse_version,
        filename="outline.json",
    )
    chunks_path = derived_artifact(
        collection_id=collection_id,
        document_id=document_id,
        parse_version=parse_version,
        filename="chunks.jsonl",
    )

    write_atomic(store, markdown_path, markdown.encode("utf-8"))
    write_atomic(
        store,
        outline_path,
        json.dumps(outline, ensure_ascii=False, indent=2).encode("utf-8"),
    )
    write_atomic(
        store,
        chunks_path,
        ("\n".join(json.dumps(c, ensure_ascii=False) for c in chunks) + "\n").encode("utf-8"),
    )

    logger.info(
        "indexing parser produced derived artifacts collection=%s document=%s "
        "parse_version=%s outline_size=%d chunk_count=%d",
        collection_id,
        document_id,
        parse_version,
        len(outline),
        len(chunks),
    )

    return ParseResult(
        parse_version=parse_version,
        markdown_path=markdown_path,
        outline_path=outline_path,
        chunks_path=chunks_path,
    )


def read_chunks(store: _SyncObjectStore, chunks_path: str) -> list[dict[str, Any]]:
    """Read ``chunks.jsonl`` back as a list of chunk records.

    Returns an empty list if the file is missing or empty (per
    §C.7 read contract — partial-derive state should reschedule, not
    raise).
    """
    body = read_or_none(store, chunks_path)
    if body is None:
        return []
    return [json.loads(line) for line in body.splitlines() if line.strip()]


__all__ = [
    "ChunkingConfig",
    "ParseConfig",
    "ParseResult",
    "DEFAULT_PARSER_PIPELINE",
    "DEFAULT_CHUNK_SIZE",
    "DEFAULT_CHUNK_OVERLAP",
    "parse_document",
    "read_chunks",
]
