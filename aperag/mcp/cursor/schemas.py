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

"""D10.e generic ``PaginationParams`` / ``PaginationResult`` shapes.

Per design pack §C.5 SDK type guards, every paginated D10 read
primitive accepts a :class:`PaginationParams` request and returns a
:class:`PaginationResult` parameterised on the per-tool item type
the primitive emits (e.g., ``CollectionItem`` for
``list_collections``, ``DocumentItem`` for ``list_documents``).

The wire ``next_cursor`` is the opaque base64url string produced by
:mod:`aperag.mcp.cursor.codec`. Clients that finish iterating
receive ``next_cursor=None`` — they MUST NOT pass that back as the
next call (decoding ``None`` would be a malformed input).
"""

from __future__ import annotations

from typing import Generic, Optional, TypeVar

from pydantic import BaseModel, Field, conint

T = TypeVar("T")

PER_TOOL_LIMIT_CEILING = 200  # absolute upper bound; per-tool override may lower this


class PaginationParams(BaseModel):
    """Generic paginated-request envelope for D10 read primitives."""

    cursor: Optional[str] = Field(
        default=None,
        description=(
            "Opaque base64url cursor returned by the previous page. "
            "``None`` means start a new pagination from the head."
        ),
    )
    limit: conint(ge=1, le=PER_TOOL_LIMIT_CEILING) = Field(
        default=50,
        description=(
            "Page size hint (1..PER_TOOL_LIMIT_CEILING). Per-tool "
            "implementations may clamp lower; the server-side cap "
            "is authoritative."
        ),
    )


class PaginationResult(BaseModel, Generic[T]):
    """Generic paginated-response envelope for D10 read primitives."""

    items: list[T]
    next_cursor: Optional[str] = Field(
        default=None,
        description=("Opaque cursor for the next page; ``None`` when the underlying iteration is exhausted."),
    )
    total_count: Optional[int] = Field(
        default=None,
        description=(
            "Optional total count; tools that can't compute it "
            "cheaply omit the field rather than emitting a stale "
            "or expensive estimate."
        ),
    )
