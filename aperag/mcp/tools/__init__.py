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

"""ApeRAG D10 MCP tools package — read primitives surface stub.

Phase 9 D10.c (#95) lands the 8 read primitive function signatures, the
typed Pydantic request/response shapes, and the stable handle types
(``ChunkId`` / ``SectionPath`` / ``HeadingAnchor``) per
``docs/modularization/d10-design-pack.md`` §A. Bodies raise
``NotImplementedError`` — full implementation lands in a follow-up PR
within the same task lane.

Cross-lane consumers — D10.d (#96) / D10.e (#97) / D10.g (#99) — import
from this package to statically reference the surface contract.
"""

from aperag.mcp.tools.get_collection_metadata import get_collection_metadata
from aperag.mcp.tools.get_document_metadata import get_document_metadata
from aperag.mcp.tools.handles import ChunkId, HeadingAnchor, SectionPath
from aperag.mcp.tools.list_collections import list_collections
from aperag.mcp.tools.list_documents import list_documents
from aperag.mcp.tools.parse_version import ParseVersionT, compute_parse_version
from aperag.mcp.tools.read_document import read_document
from aperag.mcp.tools.read_document_chunk import read_document_chunk
from aperag.mcp.tools.read_document_outline import read_document_outline
from aperag.mcp.tools.read_document_section import read_document_section
from aperag.mcp.tools.schemas import (
    ByteRange,
    CollectionDetailMetadata,
    CollectionList,
    CollectionMetadata,
    DocumentChunk,
    DocumentContent,
    DocumentList,
    DocumentMetadata,
    DocumentOutline,
    DocumentSection,
    OutlineHeading,
)

__all__ = [
    # Stable handle types (LOCKED per §A.9)
    "ChunkId",
    "HeadingAnchor",
    "SectionPath",
    # parse_version helpers (§E.2)
    "ParseVersionT",
    "compute_parse_version",
    # Read primitive functions (per §A.1 - §A.8)
    "list_collections",
    "list_documents",
    "get_collection_metadata",
    "get_document_metadata",
    "read_document",
    "read_document_outline",
    "read_document_section",
    "read_document_chunk",
    # Pydantic response shapes
    "ByteRange",
    "CollectionDetailMetadata",
    "CollectionList",
    "CollectionMetadata",
    "DocumentChunk",
    "DocumentContent",
    "DocumentList",
    "DocumentMetadata",
    "DocumentOutline",
    "DocumentSection",
    "OutlineHeading",
]
