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

"""Static (source-level) invariants for the D10.g cache wire-in into
the four parse_version-keyed read primitives.

The dynamic behavior (LRU / Redis / inflight collapsing / decode
failure → miss) is covered in the cache layer's own tests. This file
guards the *integration* invariants that the wire-in PR is responsible
for and that future refactors must not silently break.
"""

from __future__ import annotations

import inspect

from aperag.mcp.tools.read_document import read_document
from aperag.mcp.tools.read_document_chunk import read_document_chunk
from aperag.mcp.tools.read_document_outline import read_document_outline
from aperag.mcp.tools.read_document_section import read_document_section

_PARSE_VERSION_KEYED = (
    read_document,
    read_document_outline,
    read_document_section,
    read_document_chunk,
)


def test_cache_call_appears_after_d9_gates_in_every_primitive_body():
    """§E.7 — tenancy + authorization MUST run before any cache call.

    Static guard: in every parse_version-keyed primitive's source, the
    ``cache.get_or_compute_`` call appears AFTER ``tenancy_gate(`` and
    AFTER ``authorization_gate(``. A future refactor that reorders
    them (cache-first short-circuit) will fail this test loudly.
    """

    for primitive in _PARSE_VERSION_KEYED:
        src = inspect.getsource(primitive)
        idx_tenancy = src.find("tenancy_gate(")
        idx_authz = src.find("authorization_gate(")
        idx_cache = src.find("cache.get_or_compute_")
        assert idx_tenancy != -1, f"{primitive.__name__}: tenancy_gate call missing"
        assert idx_authz != -1, f"{primitive.__name__}: authorization_gate call missing"
        assert idx_cache != -1, (
            f"{primitive.__name__}: cache.get_or_compute_ call missing — "
            "the cache wire-in regressed back to un-cached body work"
        )
        assert idx_tenancy < idx_cache, (
            f"{primitive.__name__}: cache call (@{idx_cache}) must come after tenancy_gate (@{idx_tenancy})"
        )
        assert idx_authz < idx_cache, (
            f"{primitive.__name__}: cache call (@{idx_authz}) must come after authorization_gate (@{idx_authz})"
        )


def test_each_primitive_uses_its_dedicated_cache_method():
    """The wire-in must use the primitive-typed `get_or_compute_*` method
    so the cache namespace is correct (outline / section / content /
    chunk). A regression that picks the wrong helper would produce a
    silently shared key namespace across primitives.
    """

    expected = {
        "read_document_outline": "get_or_compute_outline",
        "read_document_section": "get_or_compute_section",
        "read_document": "get_or_compute_content",
        "read_document_chunk": "get_or_compute_chunk",
    }
    for primitive in _PARSE_VERSION_KEYED:
        src = inspect.getsource(primitive)
        method = expected[primitive.__name__]
        assert f"cache.{method}(" in src, f"{primitive.__name__}: expected cache.{method}( call site"


def test_chunk_primitive_does_not_pass_parse_version_to_cache():
    """§E.6 — chunk_id is indexing-immutable; the chunk cache key MUST
    NOT include parse_version. A regression that passes parse_version
    to the chunk cache would defeat the §E.6 design intent (cached
    chunks would invalidate every parse).
    """

    src = inspect.getsource(read_document_chunk)
    # Locate the cache call segment.
    cache_call_start = src.find("cache.get_or_compute_chunk(")
    assert cache_call_start != -1
    cache_call_end = src.find(")", cache_call_start)
    cache_call_segment = src[cache_call_start:cache_call_end]
    assert "parse_version" not in cache_call_segment, (
        "read_document_chunk cache call must not pass parse_version per §E.6"
    )
