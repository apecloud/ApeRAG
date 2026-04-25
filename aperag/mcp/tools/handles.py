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

"""Stable handle types for D10 read primitives — LOCKED per §A.9.

These type aliases are the stable cross-lane contract that D10.d / D10.e /
D10.g import. Field/handle names are LOCKED — do NOT rename without an
``[D10 spec amendment]`` thread per the design pack.

Per ``docs/modularization/d10-design-pack.md`` §A.9 (R1 Lock):

- ``SectionPath`` — primary stable handle; slash-separated heading position
  (e.g., ``"1/2.3/4"``).
- ``ChunkId`` — indexing-layer-issued stable id (shared with existing
  vector / full-text / graph indexes).
- ``HeadingAnchor`` — slug-style alternative (e.g.,
  ``"#chapter-2-implementation"``).

All stable handles are byte-stable within the same
``(document_id, parse_version)``; cross-``parse_version`` stability is not
guaranteed (``section_path`` is typically robust, ``chunk_id`` depends on
indexing chunker determinism — D10.d implementation decides).

For now these are exposed as ``str`` aliases; if future hardening adopts
``NewType`` or branded ``Annotated`` types, the symbol names must remain.
"""

from __future__ import annotations

from typing import TypeAlias

# §A.9: section_path — primary stable handle, slash-separated heading position.
SectionPath: TypeAlias = str

# §A.9: chunk_id — indexing-layer immutable id (per #74 D8.2 + #75 D8.3).
ChunkId: TypeAlias = str

# §A.9: heading_anchor — slug-style alternative (e.g., "#chapter-2-implementation").
HeadingAnchor: TypeAlias = str

__all__ = ["ChunkId", "HeadingAnchor", "SectionPath"]
