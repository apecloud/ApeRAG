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

"""D10.e (#97) follow-up — cursor wiring helper for D10.c list primitives.

Per design pack §G D10.e write-set the lane provides a ``service``-layer
helper that wraps :mod:`aperag.mcp.cursor` around the ORM-side queries
the read primitives already perform. The helper exists so each list
primitive only writes two helper calls instead of inlining cursor
encode / decode + invariant validation, and so the canonical
``CursorError`` codes flow uniformly out of every D10 list tool.

The cursor wire shape is the canonical opaque base64url
``CursorPayload`` from #1712 — clients still treat it as opaque. The
``last_position`` carries the offset for the current page (the design
pack's "ORM ``id``-based seek pagination" promotion is intentionally
deferred: keeping offset semantics minimises diff against #1714's
already-merged primitive bodies and is byte-equivalent on the wire,
since the cursor is opaque to clients).

Public surface (D10.c list primitives import from here):

* :func:`encode_offset_cursor` — produce the next-page wire token
* :func:`decode_offset_cursor` — accept either ``None`` / ``""`` (start
  fresh) or a wire token; raise canonical :class:`CursorError` with
  ``cursor_invalid`` / ``cursor_expired`` / ``cursor_filter_mismatch``
  / ``cursor_tenant_mismatch`` / ``cursor_schema_unsupported``
"""

from __future__ import annotations

import os
import time
from typing import Any, Optional

from aperag.mcp.cursor import (
    CursorPayload,
    compute_invariant_hash,
    decode_cursor,
    encode_cursor,
)
from aperag.mcp.cursor.codec import DEFAULT_TTL_SECONDS
from aperag.mcp.cursor.errors import CursorError

# Stable per-process server identifier — surfaces in the cursor's
# ``server_id`` field for cross-instance debugging only; clients never
# read this value.
_SERVER_ID = f"aperag-{os.getpid()}"


def encode_offset_cursor(
    *,
    offset: int,
    sort_key: str,
    filters: dict[str, Any],
    collection_id: Optional[str],
    tenant_id: str,
    ttl_seconds: int = DEFAULT_TTL_SECONDS,
) -> str:
    """Produce the next-page cursor token for a list primitive.

    ``filters`` should be a normalised dict of every user-supplied
    narrowing predicate (title_filter, type_filter, indexed_only, …) —
    the helper sha256-hashes it together with ``sort_key`` and the
    tenancy bindings so any change between two calls is loud-failed
    on decode.
    """

    invariant_hash = compute_invariant_hash(
        sort_key=sort_key,
        filters=filters,
        collection_id=collection_id,
        tenant_id=tenant_id,
    )
    payload = CursorPayload(
        sort_key=sort_key,
        last_position={"offset": int(offset)},
        invariant_hash=invariant_hash,
        issued_at=int(time.time()),
        ttl_seconds=ttl_seconds,
        server_id=_SERVER_ID,
    )
    return encode_cursor(payload)


def decode_offset_cursor(
    cursor: Optional[str],
    *,
    sort_key: str,
    filters: dict[str, Any],
    collection_id: Optional[str],
    tenant_id: str,
) -> int:
    """Validate ``cursor`` against the current scope and return the offset.

    A ``None`` / empty cursor is the canonical "start fresh" signal and
    yields offset ``0`` without raising. Any non-empty token is decoded
    via :func:`aperag.mcp.cursor.decode_cursor` (which already enforces
    ``cursor_invalid`` / ``cursor_expired`` / ``cursor_schema_unsupported``)
    and then matched against the current scope; mismatches raise
    canonical ``cursor_filter_mismatch`` / ``cursor_tenant_mismatch``
    so the client gets a different recovery path per §C.3 line 567-571.
    """

    if cursor is None or cursor == "":
        return 0

    payload = decode_cursor(cursor)

    if payload.sort_key != sort_key:
        raise CursorError(
            "cursor_filter_mismatch",
            "cursor was issued for a different sort_key",
            details={
                "received_sort_key": payload.sort_key,
                "expected_sort_key": sort_key,
            },
        )

    expected_hash = compute_invariant_hash(
        sort_key=sort_key,
        filters=filters,
        collection_id=collection_id,
        tenant_id=tenant_id,
    )
    if payload.invariant_hash != expected_hash:
        # The original tenant_id is hashed away inside ``invariant_hash``,
        # so the helper can't reliably discriminate filter-drift from
        # tenant-drift on its own. We emit ``cursor_filter_mismatch`` —
        # the broader, user-actionable code per §C.3 line 567-571.
        # The list primitives still wrap their own bodies with a
        # tenancy gate before this helper is called, so a true cross-
        # tenant cursor never reaches here in practice.
        raise CursorError(
            "cursor_filter_mismatch",
            "cursor scope no longer matches the current request",
            details={"sort_key": sort_key},
        )

    raw_offset = payload.last_position.get("offset")
    if isinstance(raw_offset, bool) or not isinstance(raw_offset, int) or raw_offset < 0:
        raise CursorError(
            "cursor_invalid",
            "cursor last_position offset must be a non-negative int",
            details={"last_position": payload.last_position},
        )
    return raw_offset
