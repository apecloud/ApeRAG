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

"""D10.e cursor wire codec — opaque base64url-encoded JSON envelope.

Per design pack §C.1: cursor is server-issued, treated by the
client as an opaque string. This module exposes:

* :class:`CursorPayload` — strongly typed internal structure
* :func:`encode_cursor` — payload → wire string
* :func:`decode_cursor` — wire string → payload, raising the
  appropriate :class:`CursorError` on malformed / expired /
  unsupported-schema input

The codec is stateless: TTL and invariants live inside the payload
and are checked at decode time. There is no server-side store —
that is intentional (§C.0 idempotent + restart-stable).
"""

from __future__ import annotations

import base64
import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any

# CURSOR_SCHEMA_VERSION bumps when the on-wire payload shape changes
# in an incompatible way. Decoders treat any `schema_version` they
# don't know about as `cursor_schema_unsupported` (§C.3) — pending
# the spec amendment double-sign on the canonical error code names
# per architect msg=669db73c, the literal raised here is owned by
# ``aperag.mcp.cursor.errors`` once that module lands.
CURSOR_SCHEMA_VERSION: int = 1

DEFAULT_TTL_SECONDS: int = 3600  # §C.4 1h default; per-tool override


@dataclass
class CursorPayload:
    """Server-only structured cursor (never deserialized client-side)."""

    sort_key: str
    last_position: dict[str, Any]
    invariant_hash: str
    issued_at: int
    server_id: str
    schema_version: int = CURSOR_SCHEMA_VERSION
    ttl_seconds: int = DEFAULT_TTL_SECONDS
    extra: dict[str, Any] = field(default_factory=dict)

    def is_expired(self, *, now: int | None = None) -> bool:
        clock = time.time() if now is None else now
        return int(clock) >= self.issued_at + self.ttl_seconds


def encode_cursor(payload: CursorPayload) -> str:
    """Encode a CursorPayload to its on-wire base64url JSON form.

    The output is URL-safe and unpadded so it round-trips cleanly
    through MCP request/response JSON without quoting or escaping.
    """

    raw = json.dumps(asdict(payload), separators=(",", ":"), sort_keys=True).encode("utf-8")
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _b64url_decode(token: str) -> bytes:
    pad = "=" * (-len(token) % 4)
    return base64.urlsafe_b64decode(token + pad)


def decode_cursor(token: str) -> CursorPayload:
    """Decode a wire cursor string back to a CursorPayload.

    On malformed / unsupported / expired input the caller is
    responsible for raising the canonical CursorError code
    (``cursor_invalid`` / ``cursor_schema_unsupported`` / etc) —
    this codec only surfaces structural issues via ValueError /
    KeyError; the canonical error mapping lives in
    :mod:`aperag.mcp.cursor.errors` (pending spec amendment lock).
    """

    raw = _b64url_decode(token)
    obj = json.loads(raw)
    return CursorPayload(
        schema_version=obj["schema_version"],
        sort_key=obj["sort_key"],
        last_position=obj["last_position"],
        invariant_hash=obj["invariant_hash"],
        issued_at=obj["issued_at"],
        ttl_seconds=obj.get("ttl_seconds", DEFAULT_TTL_SECONDS),
        server_id=obj["server_id"],
        extra=obj.get("extra", {}),
    )
