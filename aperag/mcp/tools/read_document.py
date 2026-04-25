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

"""D10 §A.5 — ``read_document`` read primitive (stub)."""

from __future__ import annotations

from typing import Optional

from aperag.mcp.tools.schemas import ByteRange, DocumentContent

_NOT_IMPL = "D10.c: implementation pending in follow-up PR; surface only for cross-lane import"


async def read_document(
    collection_id: str,
    document_id: str,
    *,
    range: Optional[ByteRange] = None,
) -> DocumentContent:
    """Read parsed markdown content of a document.

    Per ``docs/modularization/d10-design-pack.md`` §A.5. ``range`` is
    optional / best-effort and NOT stable across re-parse (see §A.9).
    """
    raise NotImplementedError(_NOT_IMPL)


__all__ = ["read_document"]
