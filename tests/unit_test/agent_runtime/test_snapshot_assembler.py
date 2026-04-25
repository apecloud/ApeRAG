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

"""Snapshot assembler contract tests (Phase 8 D8.4d / #90).

Pins the legacy artifact -> ``UIMessagePart`` projection that the
snapshot endpoint relies on until the wire emitter starts persisting
``agent_message.parts`` directly (D8.6 / #80 cleanup). Mirrors the FE
side ``snapshot-fallback.ts`` adapter so deleting both sides post-D8.6
is mechanical.
"""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

from aperag.domains.agent_runtime.db.models import AgentArtifactType
from aperag.domains.agent_runtime.snapshot_assembler import (
    assemble_parts_from_artifacts,
    extract_error_text,
)
from aperag.domains.agent_runtime.uimessage import (
    DataCitationPart,
    SourceUrlPart,
    TextPart,
)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _answer(text: str = "Hello world") -> SimpleNamespace:
    return SimpleNamespace(
        id="art-answer",
        turn_id="turn-1",
        artifact_type=AgentArtifactType.ANSWER,
        summary=text[:200],
        payload={"text": text, "query": "hi"},
        storage_ref=None,
        gmt_created=_now(),
        gmt_updated=_now(),
    )


def _reference_bundle(items: list[dict]) -> SimpleNamespace:
    return SimpleNamespace(
        id="art-bundle",
        turn_id="turn-1",
        artifact_type=AgentArtifactType.REFERENCE_BUNDLE,
        summary=f"{len(items)} references",
        payload={"items": items},
        storage_ref=None,
        gmt_created=_now(),
        gmt_updated=_now(),
    )


def _error_summary(message: str) -> SimpleNamespace:
    return SimpleNamespace(
        id="art-error",
        turn_id="turn-1",
        artifact_type=AgentArtifactType.ERROR_SUMMARY,
        summary=message[:200],
        payload={"message": message},
        storage_ref=None,
        gmt_created=_now(),
        gmt_updated=_now(),
    )


def test_answer_artifact_projects_to_text_part():
    parts = assemble_parts_from_artifacts([_answer("ApeRAG is great")])
    assert len(parts) == 1
    assert isinstance(parts[0], TextPart)
    assert parts[0].text == "ApeRAG is great"


def test_reference_bundle_projects_to_source_url_plus_data_citation():
    items = [
        {
            "source_type": "web_page",
            "source_id": "src-1",
            "title": "Example",
            "snippet": "ApeRAG is a RAG platform",
            "uri": "https://example.com/a",
            "metadata": {},
        },
        {
            "source_type": "web_page",
            "source_id": "src-2",
            "title": "Second",
            "snippet": "Second snippet",
            "uri": "https://example.com/b",
            "metadata": {},
        },
    ]
    parts = assemble_parts_from_artifacts([_reference_bundle(items)])
    types = [getattr(part, "type", None) for part in parts]
    assert types == ["source-url", "data-citation", "source-url", "data-citation"]
    assert isinstance(parts[0], SourceUrlPart)
    assert parts[0].source_id == "src-1"
    assert parts[0].url == "https://example.com/a"
    assert isinstance(parts[1], DataCitationPart)
    assert parts[1].data.cited_text == "ApeRAG is a RAG platform"
    assert parts[1].data.location.url == "https://example.com/a"


def test_reference_item_without_uri_emits_citation_only():
    """Pre-D8.6 ApeRAG reference items may be document-based with no
    URL. The citation still serialises (with empty ``url``) so the FE
    can show snippet/title; D8.6 will replace this with the canonical
    at-rest read where citation locations carry full doc/page info.
    """

    items = [{"source_id": "doc-1", "title": "PDF chapter", "snippet": "abc"}]
    parts = assemble_parts_from_artifacts([_reference_bundle(items)])
    assert [getattr(part, "type", None) for part in parts] == ["data-citation"]
    assert parts[0].data.location.url == ""
    assert parts[0].data.location.title == "PDF chapter"


def test_answer_then_reference_bundle_ordering():
    items = [{"source_id": "src-1", "uri": "https://example.com/", "title": "x", "snippet": "y"}]
    parts = assemble_parts_from_artifacts([_reference_bundle(items), _answer("done")])
    types = [getattr(part, "type", None) for part in parts]
    # TextPart from answer comes first regardless of input ordering.
    assert types == ["text", "source-url", "data-citation"]


def test_unknown_artifact_types_are_skipped():
    tool_summary = SimpleNamespace(
        id="art-tool",
        turn_id="turn-1",
        artifact_type=AgentArtifactType.TOOL_RESULT_SUMMARY,
        summary="tool output",
        payload={"text": "tool output"},
        storage_ref=None,
        gmt_created=_now(),
        gmt_updated=_now(),
    )
    parts = assemble_parts_from_artifacts([tool_summary])
    assert parts == []


def test_extract_error_text_prefers_payload_message():
    text = extract_error_text([_error_summary("upstream timeout")])
    assert text == "upstream timeout"


def test_extract_error_text_falls_back_to_summary_when_payload_empty():
    artifact = _error_summary("ignored")
    artifact.payload = {}
    artifact.summary = "from-summary"
    assert extract_error_text([artifact]) == "from-summary"


def test_extract_error_text_returns_none_without_error_summary():
    assert extract_error_text([_answer()]) is None
