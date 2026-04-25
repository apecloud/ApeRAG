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

"""D10.e (#97) opaque cursor pagination contract for MCP read primitives.

Per ``docs/modularization/d10-design-pack.md`` §C — server-issued
base64 cursor with stability invariants (sort key + filter / tenant
/ index hash) plus 6 explicit error codes; never silently reset to
first page (Weston msg=95b07155 hard lock).

Public surface (D10.c read primitives import from here):

* :class:`CursorPayload` — internal cursor structure (server only;
  client treats wire string as opaque)
* :func:`encode_cursor` / :func:`decode_cursor` — wire codec
* :func:`compute_invariant_hash` — stable hash over cursor scope
  bindings (filters / collection_id / tenant_id)
* :class:`PaginationParams` / :class:`PaginationResult` — typed
  request / response generic over the paginated item type
* :class:`CursorError` + the 6 canonical error codes (pending
  spec amendment double-sign per architect msg=669db73c — error
  module loaded after canonical lock)

Search-rank cursor (vector / fulltext score-based) is intentionally
NOT shared with this module — D10.d carries its own cursor type
with score-boundary invariants (per design pack §G D10.e Forbidden).
"""

from aperag.mcp.cursor.codec import (
    CursorPayload,
    decode_cursor,
    encode_cursor,
)
from aperag.mcp.cursor.invariants import compute_invariant_hash
from aperag.mcp.cursor.schemas import (
    PaginationParams,
    PaginationResult,
)

__all__ = [
    "CursorPayload",
    "PaginationParams",
    "PaginationResult",
    "compute_invariant_hash",
    "decode_cursor",
    "encode_cursor",
]
