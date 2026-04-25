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

"""ApeRAG SDK helpers for MCP clients (Phase 9 D10.f #98 +).

The SDK package collects pure-Python helpers that an external MCP-aware
client (Claude Code / Codex / Cursor / ApeRAG own-Agent) can vendor or
import to consume the ApeRAG MCP surface without re-implementing the
same logic. The first member is :mod:`capability_filter` (D10 §D
Option A canonical).
"""
