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

"""Pin the tool-call title-resolution contract.

earayu2 directive (msg=077c88fd 都展示名称而不是 id): MCP read
primitives like ``read_document(document_id, ...)`` and
``get_collection_metadata(collection_id)`` only carry opaque IDs in
their input. Without a BE-side title lookup the FE activity narration
ends up showing ``"已查看文档：doc12a626b4..."`` to users, which
defeats the whole "user-friendly tool labels" feature.

The runtime now resolves titles into ``ToolPart.metadata`` keyed by
``document_title`` / ``collection_title`` — exact match for the FE
renderer's ``extractStringField(part.metadata, [...])`` lookup
list in ``agent-turn-renderer.tsx`` (PR #1826), so dongdong's FE
picks them up automatically with zero extra mapping.
"""

from __future__ import annotations

from aperag.domains.agent_runtime.runtime import (
    _compose_assistant_parts,
    _PersistedToolCall,
)
from aperag.domains.agent_runtime.uimessage import ToolPart


def test_persisted_tool_call_titles_flow_into_toolpart_metadata():
    """The renderer reads ``part.metadata.document_title`` /
    ``collection_title``; verify the at-rest composer merges
    ``_PersistedToolCall.titles`` into the metadata dict alongside
    ``mcpToolName``."""

    parts = _compose_assistant_parts(
        turn_id="turn-1",
        answer_text="",
        references=[],
        tool_calls=[
            _PersistedToolCall(
                tool_call_id="call-1",
                tool_name="read_document",
                state="output-available",
                titles={
                    "document_title": "03-非洲猪瘟常态化防控技术指南.md",
                    "collection_title": "技术文档库",
                },
            )
        ],
    )
    assert len(parts) == 1
    assert isinstance(parts[0], ToolPart)
    assert parts[0].metadata == {
        "mcpToolName": "read_document",
        "document_title": "03-非洲猪瘟常态化防控技术指南.md",
        "collection_title": "技术文档库",
    }


def test_persisted_tool_call_with_no_titles_keeps_metadata_minimal():
    """When the tool args don't have any opaque IDs (e.g., ``web_search``
    with just a query) the resolver returns ``{}`` — metadata stays
    on the original ``mcpToolName``-only shape so we don't bloat
    ``agent_message.parts`` with empty title fields."""

    parts = _compose_assistant_parts(
        turn_id="turn-1",
        answer_text="",
        references=[],
        tool_calls=[
            _PersistedToolCall(
                tool_call_id="call-1",
                tool_name="web_search",
                state="output-available",
                summary="搜索:中国蜜蜂产业",
            )
        ],
    )
    assert parts[0].metadata == {"mcpToolName": "web_search"}, "no titles resolved → metadata should stay minimal"


def test_persisted_tool_call_titles_applied_on_timeline_path_too():
    """The chronological timeline path (Wave 9 reasoning interleave)
    also needs to propagate titles — both ``timeline=`` and legacy
    ``tool_calls=`` callers must agree."""

    parts = _compose_assistant_parts(
        turn_id="turn-1",
        answer_text="",
        references=[],
        timeline=[
            _PersistedToolCall(
                tool_call_id="call-1",
                tool_name="get_document_metadata",
                state="output-available",
                titles={"document_title": "中国蜜蜂产业概况"},
            ),
        ],
    )
    tool_parts = [p for p in parts if isinstance(p, ToolPart)]
    assert len(tool_parts) == 1
    assert tool_parts[0].metadata == {
        "mcpToolName": "get_document_metadata",
        "document_title": "中国蜜蜂产业概况",
    }
