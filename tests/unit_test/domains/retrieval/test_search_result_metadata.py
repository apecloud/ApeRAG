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

"""D10.h cutover (#100) — SearchResultMetadata chunk_id / section_path /
heading_anchor exposure tests (per amendment-#2 Drift #1).

Locks the retrieval-domain public allowlist:

* The 3 D10 §A.9 stable handle fields are part of the
  ``SearchResultMetadata`` Pydantic model (so callers can read them
  off ``SearchResultItem.metadata``).
* ``SearchResultMetadata.from_raw()`` extracts each of the 3 fields
  from a raw upstream metadata dict.
* The model still ``extra="forbid"``s unknown keys — adding the 3
  fields does NOT relax the allowlist.

These are static contract guards — wire propagation behavior is exercised
in the search-pipeline tests + e2e hurl. The point of this file is to
prevent a future refactor from silently dropping the 3 LOCKED handle
fields from the public surface.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from aperag.domains.retrieval.schemas import SearchResultMetadata


def test_metadata_exposes_chunk_id_field():
    md = SearchResultMetadata(chunk_id="doc-42_3")
    assert md.chunk_id == "doc-42_3"


def test_metadata_exposes_section_path_field():
    md = SearchResultMetadata(section_path="1/2/3")
    assert md.section_path == "1/2/3"


def test_metadata_exposes_heading_anchor_field():
    md = SearchResultMetadata(heading_anchor="#chapter-2-implementation")
    assert md.heading_anchor == "#chapter-2-implementation"


def test_metadata_unknown_key_still_rejected():
    """Adding the 3 handle fields must not relax ``extra="forbid"``."""
    with pytest.raises(ValidationError):
        SearchResultMetadata(invented_key="oops")


def test_from_raw_extracts_all_three_handles():
    md = SearchResultMetadata.from_raw(
        {
            "chunk_id": "doc-42_3",
            "section_path": "1/2/3",
            "heading_anchor": "#chapter-2-implementation",
            "document_id": "doc-42",
        }
    )
    assert md is not None
    assert md.chunk_id == "doc-42_3"
    assert md.section_path == "1/2/3"
    assert md.heading_anchor == "#chapter-2-implementation"
    assert md.document_id == "doc-42"


def test_from_raw_handles_missing_handles_as_none():
    """Upstream backends that do not yet propagate section_path /
    heading_anchor (the indexing layer enhancement is intentionally out
    of D10.h scope per PM msg=5760999e #4) must not break the schema —
    the field surfaces as ``None`` and the rest of the metadata flows
    through unchanged.
    """
    md = SearchResultMetadata.from_raw({"document_id": "doc-42"})
    assert md is not None
    assert md.document_id == "doc-42"
    assert md.chunk_id is None
    assert md.section_path is None
    assert md.heading_anchor is None


def test_from_raw_ignores_non_string_handle_values():
    """``public_str`` filter rejects non-string / empty values so the
    public surface never carries a numeric chunk_id or empty string.
    """
    md = SearchResultMetadata.from_raw(
        {
            "chunk_id": 42,  # not a string — filtered
            "section_path": "",  # empty — filtered
            "heading_anchor": "valid-anchor",
            "document_id": "doc-1",
        }
    )
    assert md is not None
    assert md.chunk_id is None
    assert md.section_path is None
    assert md.heading_anchor == "valid-anchor"
