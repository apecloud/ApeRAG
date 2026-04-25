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

"""D10 §A.3 — ``get_document_metadata`` read primitive (stub)."""

from __future__ import annotations

from aperag.mcp.tools.schemas import DocumentMetadata

_NOT_IMPL = "D10.c: implementation pending in follow-up PR; surface only for cross-lane import"


async def get_document_metadata(
    collection_id: str,
    document_id: str,
) -> DocumentMetadata:
    """Get metadata for a specific document.

    Per ``docs/modularization/d10-design-pack.md`` §A.3.
    """
    raise NotImplementedError(_NOT_IMPL)


__all__ = ["get_document_metadata"]
