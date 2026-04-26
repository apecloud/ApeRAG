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

"""D10.e cursor stability invariants — sha256 over scope bindings.

Per design pack §C.2 + Weston msg=95b07155 stability requirement, a
cursor encodes a hash of the bindings that must NOT change for it
to remain valid:

* ``sort_key`` — primary sort field; switching changes ordering and
  invalidates last_position.
* ``filters`` — any user-supplied filter set, including search /
  list narrowing predicates (mode flags, time windows, tags).
* ``collection_id`` / ``tenant_id`` — tenancy boundary; reusing a
  cursor across tenants is a security boundary violation.
* ``index_id`` (optional) — when paginating against a versioned
  index, the cursor pins to the index version it was issued
  against; reindex bumps the id and any cursor with a stale hash
  fails ``cursor_index_changed``.

The function is intentionally insulated from §C error code naming:
callers compute the hash here and compare; the *response* mapping
into ``cursor_filter_mismatch`` / ``cursor_tenant_mismatch`` /
``cursor_index_changed`` lives in ``aperag.mcp.cursor.errors``.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


def compute_invariant_hash(
    *,
    sort_key: str,
    filters: dict[str, Any],
    collection_id: str | None,
    tenant_id: str,
    index_id: str | None = None,
) -> str:
    """Return the canonical sha256 hex digest for these bindings.

    Inputs are normalised via ``json.dumps(sort_keys=True)`` so the
    hash is stable across dict ordering and Python re-serialisation.
    """

    payload = {
        "sort_key": sort_key,
        "filters": filters,
        "collection_id": collection_id,
        "tenant_id": tenant_id,
        "index_id": index_id,
    }
    raw = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()
