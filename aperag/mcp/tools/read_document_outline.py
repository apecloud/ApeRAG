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

"""D10 §A.6 — ``read_document_outline`` read primitive (stub).

Highest-value primitive per inventory §C.3 gold mine. Agents call this
to decide which section / chunk to read next.
"""

from __future__ import annotations

from aperag.mcp.tools.schemas import DocumentOutline

_NOT_IMPL = "D10.c: implementation pending in follow-up PR; surface only for cross-lane import"


async def read_document_outline(
    collection_id: str,
    document_id: str,
    *,
    max_depth: int = 6,
) -> DocumentOutline:
    """Read heading tree (table of contents) of a document.

    Per ``docs/modularization/d10-design-pack.md`` §A.6.
    """
    raise NotImplementedError(_NOT_IMPL)


__all__ = ["read_document_outline"]
