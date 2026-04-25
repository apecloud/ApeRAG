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

"""parse_version helper for D10.c read primitives.

Per ``docs/modularization/d10-design-pack.md`` §E.2: ``parse_version`` is
the composite cache key that pins a parsed-markdown view of a document
to a specific ``(parser_pipeline, document_md5, chunking_config)`` triple.
Two reads of the same document with the same triple MUST return byte-
identical content; any change in the triple produces a fresh
``parse_version``.

Format: ``sha256("{parser_pipeline}|{document_md5}|{chunking_config}")[:16]`` —
16 lowercase hex chars (architect canonical lock msg=a67974b3 Q1).

D10.g (#99 @明书) may move/wrap this helper; the public symbol names
(``ParseVersionT`` / ``compute_parse_version``) are part of the cross-
lane contract and must remain stable.
"""

from __future__ import annotations

import hashlib
from typing import Annotated

from pydantic import StringConstraints

# 16 lowercase hex chars per architect msg=a67974b3 Q1 lock.
ParseVersionT = Annotated[
    str,
    StringConstraints(pattern=r"^[0-9a-f]{16}$", min_length=16, max_length=16),
]


def compute_parse_version(
    *,
    parser_pipeline: str,
    document_md5: str,
    chunking_config: str,
) -> str:
    """Compute the deterministic ``parse_version`` for a parsed document view.

    Args:
        parser_pipeline: identifier for the parser pipeline that produced the
            parsed markdown (e.g., ``"docparser-v1"`` or a serialized
            collection parser config blob).
        document_md5: stable content hash of the source document (typically
            ``Document.content_hash``).
        chunking_config: serialized chunking config (chunk size + overlap
            + strategy). Must change when chunking changes so the
            ``parse_version`` rolls.

    Returns:
        A 16-char lowercase hex string (first 16 chars of sha256).
    """

    payload = f"{parser_pipeline}|{document_md5}|{chunking_config}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


__all__ = ["ParseVersionT", "compute_parse_version"]
