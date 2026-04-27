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

"""Parser entry point — celery T1.1 Foundation + Wave 4 T3 real wire.

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

T1.1 shipped a **deterministic in-process simulator** for UTF-8
markdown so downstream modalities could wire their tests against a
real ``derived/`` layout before the production parser landed. Wave 4
T3 chunk 1 wires the real :class:`aperag.docparser.doc_parser.DocParser`
(MarkItDown + MinerU + ImageParser + AudioParser) through the same
entry point — the simulator path stays for ``.md`` / ``.markdown`` /
``.txt`` / no-extension inputs and tests, while non-text extensions
(``.pdf`` / ``.docx`` / ``.doc`` / ``.pptx`` / ``.xlsx`` / ``.png``
/ ``.jpg`` / ``.epub`` / ``.html`` / ...) materialise a tempfile,
hand it to ``DocParser.parse_file``, concatenate the resulting
:class:`MarkdownPart` bodies, and run the same outline + chunking
pipeline so the artifact schema stays unchanged.

production-readiness invariant (Wave 3 lesson #10):
- must-be-real: ``DocParser`` chain dispatches on extension and runs
  real PDF / Office / image / audio parsers on production deployments.
- may-be-gated: simulator path stays the default for text-only inputs
  (``.md`` / ``.markdown`` / ``.txt`` / no-extension hint) so unit
  tests + dev workflows that pass UTF-8 markdown bytes without a
  filename keep working unchanged.
- partially-resolves: Wave 4 backlog #4. Chunk 2 promotes the parser
  to its own ``q:parse`` async queue (per §E.2) so an upload handler
  no longer blocks on a 30-second OCR run inside the request thread.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
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


# Extensions that the simulator can decode directly (UTF-8 markdown
# /text). Every other extension (``.pdf`` / ``.docx`` / ``.doc`` /
# ``.png`` / ``.jpg`` / ...) routes through ``DocParser`` so the
# production parser chain owns binary + Office + image inputs.
_SIMULATOR_EXTENSIONS = frozenset(
    {
        ".md",
        ".markdown",
        ".txt",
        ".text",
    }
)


def _normalise_extension(source_filename: str | None) -> str | None:
    """Lowercase + dotted extension from a filename hint, or ``None``.

    ``None`` / empty / no-dot inputs return ``None`` so callers can
    keep the legacy simulator behaviour (assume UTF-8 markdown). All
    other strings normalise to ``.<ext>`` lowercase.
    """
    if not source_filename:
        return None
    suffix = Path(source_filename).suffix
    if not suffix:
        return None
    return suffix.lower()


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
    callers can persist it on the ``DocumentIndex`` row. The artifact
    paths are the canonical names downstream modalities expect to find
    under ``derived/parse_<version>/``.

    Wave 5 P2 chunk 2 (per §G.2.5.1 spec amend item 2): when DocParser
    produces ``AssetBinPart`` payloads (PDF page images / single-image
    inputs / data-URI extracted images), the parser writes each blob to
    ``derived/parse_<v>/vision/images/<image_id>.<ext>`` and lands a
    ``vision/source.jsonl`` descriptor enumerating them. The vision
    worker consumes the descriptor (chunk 4 callsite rewrite) instead
    of the T1 simulator's synthetic ``images.json`` companion. The
    descriptor path is empty when the parsed document has no image
    assets — vision modality short-circuits to the no-image FAILED
    handling already in place.
    """

    parse_version: str
    markdown_path: str
    outline_path: str
    chunks_path: str
    vision_source_path: str = ""
    vision_image_count: int = 0


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


def _all_artifacts_present(
    *,
    store: _SyncObjectStore,
    markdown_path: str,
    outline_path: str,
    chunks_path: str,
) -> bool:
    """Wave 5 P4 short-circuit predicate: all three canonical
    derived artifacts must exist for the cached parse to be valid.

    Uses ``ObjectStore.obj_exists`` (cheap metadata check) rather
    than ``read_or_none`` so the predicate stays cost-bounded for
    every parse call. ``chunks.jsonl`` is checked last because it is
    the only artifact downstream modality workers actually read; if
    it is missing the previous parse was interrupted mid-write and
    re-parsing is required regardless.
    """
    try:
        return store.obj_exists(markdown_path) and store.obj_exists(outline_path) and store.obj_exists(chunks_path)
    except Exception:  # noqa: BLE001 — predicate fails closed (re-parse)
        return False


# ---------------------------------------------------------------------
# Vision asset extraction helpers — Wave 5 P2 chunk 2
# ---------------------------------------------------------------------


# MIME-type → file extension lookup. Provider-specific extras (HEIC,
# AVIF, etc.) drop through to the generic ``.bin`` fallback rather
# than rejecting the asset — the vision worker uses ``imghdr`` on the
# image bytes themselves at embed time, so the on-disk filename
# extension is informational only.
_MIME_EXTENSION_MAP: dict[str, str] = {
    "image/jpeg": "jpg",
    "image/jpg": "jpg",
    "image/png": "png",
    "image/webp": "webp",
    "image/gif": "gif",
    "image/bmp": "bmp",
    "image/tiff": "tiff",
    "image/svg+xml": "svg",
}


def _vision_image_extension(mime_type: str | None) -> str:
    if not mime_type:
        return "bin"
    key = mime_type.split(";", 1)[0].strip().lower()
    return _MIME_EXTENSION_MAP.get(key, "bin")


@dataclass(frozen=True)
class _VisionImageAsset:
    """Internal carrier for a single extracted image asset.

    Holds enough to land both the blob (under ``vision/images/``) and
    its row in the ``vision/source.jsonl`` descriptor consumed by the
    vision worker (chunk 4). ``image_id`` is the canonical ``asset_id``
    DocParser already computes (md5 of the image bytes), so two parses
    of the same content produce the same image_id and downstream
    identity (``vision:<doc_id>:<parse_v>:<image_id>`` Qdrant point id)
    stays stable across retries.
    """

    image_id: str
    data: bytes
    mime_type: str
    alt_text: str
    page_idx: int | None
    bbox: list[float] | None


def _docparser_extract_markdown(
    *,
    source_bytes: bytes,
    extension: str,
    parser_config: dict[str, Any] | None,
) -> tuple[str, list[_VisionImageAsset]]:
    """Run :class:`DocParser` on ``source_bytes`` and return both
    concatenated markdown AND the extracted vision image assets.

    Wave 5 P2 chunk 2 (per §G.2.5.1 item 2): the assets list carries
    every :class:`AssetBinPart` whose ``mime_type`` is a recognised
    image type. Audio / video / PDF-data assets are dropped — only
    images participate in the vision modality. The caller writes the
    blobs out under ``derived/parse_<v>/vision/images/`` plus a
    descriptor JSONL line for each.

    Materialises the bytes into a tempfile (DocParser only accepts a
    real path on disk because MarkItDown / MinerU / OCR all stream
    from disk), runs the parser chain, then collects every
    :class:`MarkdownPart` body in order. Non-image asset parts (PDF
    data, audio, etc.) belong to other modalities or to the cleanup
    loop's asset GC (T2.1), not the shared markdown contract.

    DocParser is imported lazily so the indexing package's __init__
    does not pull MarkItDown / MinerU / pikepdf at import time
    (matches the existing T1.1 lazy-import discipline that kept the
    Wave 3 hard-cut circular-import-free).
    """
    from aperag.docparser.base import AssetBinPart, MarkdownPart
    from aperag.docparser.doc_parser import DocParser

    # Use the suffix the caller already normalised so the temp filename
    # carries the right extension for DocParser's per-extension
    # dispatch (markitdown_parser / mineru_parser / image_parser).
    suffix = extension if extension.startswith(".") else f".{extension}"
    parser = DocParser(parser_config=parser_config or {})
    if not parser.accept(suffix):
        raise ValueError(
            f"DocParser does not accept extension {suffix!r}; "
            "supported list: " + ", ".join(parser.supported_extensions())
        )

    # Tempfile lifecycle: write source_bytes → DocParser reads via
    # path → unlink in finally. ``delete=False`` because Python's
    # NamedTemporaryFile on POSIX leaves the file open for the parser
    # to re-open by path; we close + delete explicitly.
    fd, tmp_path = tempfile.mkstemp(suffix=suffix, prefix="aperag-parse-")
    os.close(fd)
    try:
        with open(tmp_path, "wb") as fh:
            fh.write(source_bytes)
        parts = parser.parse_file(Path(tmp_path))
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp_path)

    markdown_parts = [p.markdown for p in parts if isinstance(p, MarkdownPart) and p.markdown]
    seen_image_ids: set[str] = set()
    image_assets: list[_VisionImageAsset] = []
    for part in parts:
        if not isinstance(part, AssetBinPart):
            continue
        mime = (part.mime_type or "").lower()
        if not mime.startswith("image/"):
            continue
        if not part.data:
            continue
        if part.asset_id in seen_image_ids:
            # Same image referenced twice in the document → keep the
            # first record + drop the duplicate so the descriptor has a
            # single canonical row per image_id. The Qdrant point id
            # would have collided otherwise.
            continue
        seen_image_ids.add(part.asset_id)
        metadata = part.metadata or {}
        image_assets.append(
            _VisionImageAsset(
                image_id=part.asset_id,
                data=part.data,
                mime_type=mime,
                alt_text=str(metadata.get("alt_text") or part.content or ""),
                page_idx=metadata.get("page_idx") if isinstance(metadata.get("page_idx"), int) else None,
                bbox=metadata.get("bbox") if isinstance(metadata.get("bbox"), list) else None,
            )
        )

    if not markdown_parts:
        # Image-only / audio-only inputs (no MarkdownPart) currently
        # have nothing for the outline + chunks pipeline to emit; we
        # return an empty body so the artifacts exist but downstream
        # vector / fulltext modalities see zero chunks. Image-only
        # uploads still land their assets via the descriptor below so
        # the vision modality has bytes to embed.
        return ("", image_assets)
    return ("\n\n".join(markdown_parts), image_assets)


def parse_document(
    *,
    store: _SyncObjectStore,
    collection_id: str,
    document_id: str,
    source_bytes: bytes,
    source_filename: str | None = None,
    parser_config: dict[str, Any] | None = None,
    config: ParseConfig | None = None,
    short_circuit_if_artifacts_exist: bool = True,
) -> ParseResult:
    """Parse the source bytes and persist the three shared artifacts.

    Idempotent at the ``parse_version`` level: re-running with
    identical inputs produces identical artifacts (and overwrites
    them atomically). The artifact paths are the canonical
    ``derived/parse_<version>/`` layout per design pack §C.1.

    Wave 5 P4 short-circuit: when ``short_circuit_if_artifacts_exist``
    is True (default) and all three derived artifacts (``markdown.md``
    / ``outline.json`` / ``chunks.jsonl``) already exist in the object
    store under the canonical ``derived/parse_<version>/`` path, the
    parser **skips DocParser + writes entirely** and returns the
    existing :class:`ParseResult`. This eliminates the ~30s OCR / Word
    rerun cost when a document is re-uploaded with identical content
    or a rebuild is dispatched against an already-parsed version
    (per huangheng T3 chunk 2 obs B + architect Wave 5 P4 lock).

    Dispatch (Wave 4 T3 chunk 1):
    - ``source_filename`` ends in a known text extension (``.md`` /
      ``.markdown`` / ``.txt`` / ``.text``) **or** is ``None`` →
      decode as UTF-8 markdown via the simulator path. Backward-
      compatible with every existing caller that just hands bytes.
    - any other extension → run the real ``DocParser`` chain
      (MarkItDown / MinerU / ImageParser / AudioParser per
      ``parser_config``), concatenate every produced ``MarkdownPart``,
      then continue with the existing outline + chunk pipeline. The
      ``derived/`` artifact schema is unchanged.

    Args:
        store: Object store handle (``LocalObjectStore`` /
            ``S3ObjectStore`` / :class:`InMemoryObjectStore` for tests).
        collection_id: Owning collection.
        document_id: Document being parsed.
        source_bytes: Raw bytes of the source document.
        source_filename: Optional filename hint (e.g. ``"report.pdf"``
            or just ``"report.pdf"`` — only the suffix is consumed).
            Used to dispatch on extension; ``None`` keeps legacy
            simulator behaviour.
        parser_config: Optional collection-level parser config dict
            (e.g. ``{"use_mineru": True, "mineru_api_token": "..."}``).
            Forwarded to :class:`DocParser` when the dispatcher routes
            to the real parser chain. Ignored on the simulator path.
        config: Parsing knobs that influence the parse_version. Pass
            ``None`` to use simulator defaults.
        short_circuit_if_artifacts_exist: When True (default), reuse
            existing canonical artifacts if all three are already in
            the object store under the resolved
            ``derived/parse_<version>/`` path. Pass ``False`` to force
            a re-parse + re-write (used by tests pinning DocParser
            invocation count).
    """
    from aperag.mcp.tools.parse_version import compute_parse_version

    cfg = config or ParseConfig()
    extension = _normalise_extension(source_filename)

    document_md5 = _document_md5(source_bytes)
    parse_version = compute_parse_version(
        parser_pipeline=cfg.parser_pipeline,
        document_md5=document_md5,
        chunking_config=cfg.chunking.serialize(),
    )

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

    if short_circuit_if_artifacts_exist and _all_artifacts_present(
        store=store,
        markdown_path=markdown_path,
        outline_path=outline_path,
        chunks_path=chunks_path,
    ):
        logger.info(
            "indexing parser short-circuit collection=%s document=%s parse_version=%s "
            "(all derived artifacts already present; skipping DocParser + writes)",
            collection_id,
            document_id,
            parse_version,
        )
        return ParseResult(
            parse_version=parse_version,
            markdown_path=markdown_path,
            outline_path=outline_path,
            chunks_path=chunks_path,
        )

    image_assets: list[_VisionImageAsset] = []
    if extension is None or extension in _SIMULATOR_EXTENSIONS:
        try:
            markdown = source_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            # Caller passed text-extension or no extension hint but the
            # bytes are not UTF-8. Surface the contract gap clearly so
            # the upload route logs a real cause; production callers
            # always pass an accurate ``source_filename`` so this only
            # fires on test misuse.
            raise ValueError(
                f"simulator parser path requires UTF-8 markdown bytes "
                f"(extension={extension or 'none'}); pass source_filename "
                f"with the real extension to dispatch to DocParser"
            ) from exc
    else:
        markdown, image_assets = _docparser_extract_markdown(
            source_bytes=source_bytes,
            extension=extension,
            parser_config=parser_config,
        )

    outline = _build_outline(markdown)
    chunks = _split_chunks(markdown, cfg.chunking)

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

    vision_source_path = ""
    if image_assets:
        vision_source_path = _write_vision_assets(
            store=store,
            collection_id=collection_id,
            document_id=document_id,
            parse_version=parse_version,
            assets=image_assets,
        )

    logger.info(
        "indexing parser produced derived artifacts collection=%s document=%s "
        "parse_version=%s outline_size=%d chunk_count=%d vision_image_count=%d",
        collection_id,
        document_id,
        parse_version,
        len(outline),
        len(chunks),
        len(image_assets),
    )

    return ParseResult(
        parse_version=parse_version,
        markdown_path=markdown_path,
        outline_path=outline_path,
        chunks_path=chunks_path,
        vision_source_path=vision_source_path,
        vision_image_count=len(image_assets),
    )


def _write_vision_assets(
    *,
    store: _SyncObjectStore,
    collection_id: str,
    document_id: str,
    parse_version: str,
    assets: list[_VisionImageAsset],
) -> str:
    """Persist extracted image bytes + descriptor under
    ``derived/parse_<v>/vision/``.

    Each asset is written to ``vision/images/<image_id>.<ext>`` and
    enumerated in a ``vision/source.jsonl`` descriptor (one record per
    line, schema ``{image_id, image_path, mime_type, alt_text,
    page_idx, bbox}``). Returns the descriptor path so the caller can
    pin it on :class:`ParseResult` and the orchestrator can hand it to
    the vision worker (chunk 4 callsite rewrite).
    """
    descriptor_lines: list[str] = []
    for asset in assets:
        ext = _vision_image_extension(asset.mime_type)
        image_path = derived_artifact(
            collection_id=collection_id,
            document_id=document_id,
            parse_version=parse_version,
            filename=f"vision/images/{asset.image_id}.{ext}",
        )
        write_atomic(store, image_path, asset.data)
        descriptor_lines.append(
            json.dumps(
                {
                    "image_id": asset.image_id,
                    "image_path": image_path,
                    "mime_type": asset.mime_type,
                    "alt_text": asset.alt_text,
                    "page_idx": asset.page_idx,
                    "bbox": asset.bbox,
                },
                ensure_ascii=False,
            )
        )

    descriptor_path = derived_artifact(
        collection_id=collection_id,
        document_id=document_id,
        parse_version=parse_version,
        filename="vision/source.jsonl",
    )
    write_atomic(store, descriptor_path, ("\n".join(descriptor_lines) + "\n").encode("utf-8"))
    return descriptor_path


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
