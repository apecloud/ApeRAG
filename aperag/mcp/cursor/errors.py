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

"""D10.e canonical cursor error codes (§C.3 contract).

Per architect canonical lock (msg=77bd1f6a / msg=05063521 /
#1710) the 6 snake_case codes from design pack §C.3 are the only
identifiers the wire surface uses; the SCREAMING_SNAKE shorthand
that appeared in the §G summary was drafting noise and is
backfilled via #1710.

Each code carries a fixed *client recovery path* (§C.3 lines
567-571) — collapsing distinct codes (e.g., merging the three
invariant-mismatch variants into one) would force the client to
over-react, which is why we keep them split:

* ``cursor_invalid`` / ``cursor_schema_unsupported`` — restart
  pagination from a null cursor.
* ``cursor_expired`` — restart pagination.
* ``cursor_filter_mismatch`` / ``cursor_tenant_mismatch`` —
  client-side bug; surface to the operator.
* ``cursor_index_changed`` — backend ops issue; retry from null.

A single :class:`CursorError` exception carries the code; the wire
mapping happens at the MCP tool boundary (D10.c read primitives /
D10.d search primitives) where ``CursorError`` is caught and
re-emitted as the tool's structured error envelope.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

CursorErrorCode = Literal[
    "cursor_invalid",
    "cursor_expired",
    "cursor_filter_mismatch",
    "cursor_tenant_mismatch",
    "cursor_index_changed",
    "cursor_schema_unsupported",
]

# Anti-pattern guard (§C.3): a server that silently resets to the
# first page on cursor failure violates the explicit-not-silent
# contract. Decoders MUST raise CursorError; callers MUST surface
# the wire envelope rather than swallowing the error and starting
# a fresh pagination on the user's behalf.
SILENT_RESET_FORBIDDEN = True


class CursorErrorEnvelope(BaseModel):
    """Wire envelope for cursor errors emitted by D10 tools."""

    code: CursorErrorCode
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class CursorError(Exception):
    """Raised whenever a cursor cannot be honoured.

    The ``code`` attribute carries one of the six canonical
    :data:`CursorErrorCode` literals. ``details`` is reserved for
    server-diagnostic data that is safe to expose to the client
    (e.g., the offending invariant field name); operator-only
    diagnostics belong in logs, not in the wire envelope.
    """

    def __init__(
        self,
        code: CursorErrorCode,
        message: str,
        *,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.code: CursorErrorCode = code
        self.message: str = message
        self.details: dict[str, Any] = details or {}

    def to_envelope(self) -> CursorErrorEnvelope:
        return CursorErrorEnvelope(code=self.code, message=self.message, details=self.details)
