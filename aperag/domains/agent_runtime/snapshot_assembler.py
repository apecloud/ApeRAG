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

"""Snapshot endpoint UIMessage assembler (Phase 8 D8.4d / #90).

The agent runtime's snapshot endpoint historically returned the legacy
``{turn, timeline, artifacts}`` shape. D8 §2 locks the wire and at-rest
``UIMessage`` parts as same-schema, so the snapshot endpoint must
expose ``parts: UIMessagePart[]`` for the FE renderer (#76 / #77 / #78)
to consume from a single code path.

This module projects the legacy ``AgentArtifact`` rows into the
canonical at-rest part list, mirroring the FE-side
``snapshot-fallback.ts`` adapter that #77 huangheng landed as a
transitional bridge:

* ``answer`` artifact -> single ``TextPart``
* ``reference_bundle`` artifact -> N x ``SourceUrlPart`` plus
  N x ``DataCitationPart`` (Anthropic-shape)
* ``error_summary`` artifact -> not a part; surfaced via the snapshot
  envelope's ``error_text`` field

Other artifact types (``tool_result_summary`` /
``search_result_summary``) are skipped: tool lifecycle and search
results are surfaced by the tool/citation parts emitted directly during
the live turn (#73 wire emitter + #75 tool lifecycle), so projecting
them here would double-count once the wire emitter starts persisting
``agent_message.parts`` directly (D8.6 / #80 cleanup).

When that cleanup lands and ``UIMessageStore.read(turn_id)`` returns a
populated row, the snapshot service short-circuits to the at-rest read
and this assembler is no longer hit. At that point the FE
``snapshot-fallback.ts`` and this module can both be deleted in a
single follow-up.
"""

from __future__ import annotations

from typing import Iterable, Optional

from aperag.domains.agent_runtime.db.models import AgentArtifact, AgentArtifactType
from aperag.domains.agent_runtime.uimessage import (
    CitationData,
    DataCitationPart,
    SourceUrlPart,
    TextPart,
    UIMessagePart,
    UrlCitationLocation,
)


def assemble_parts_from_artifacts(
    artifacts: Iterable[AgentArtifact],
) -> list[UIMessagePart]:
    """Project legacy ``AgentArtifact`` rows into a UIMessage parts list.

    Order: ``TextPart`` (answer) first, then ``SourceUrlPart`` /
    ``DataCitationPart`` pairs from ``reference_bundle.items``. The
    answer text comes from ``payload.text`` (matches the runtime's
    artifact write at ``runtime.py:441``); reference items are read
    from ``payload.items`` (matches ``runtime.py:449``).

    Items that lack a URI are still surfaced as ``DataCitationPart``
    with an empty ``url`` so the FE renderer can show snippet/title;
    the upcoming D8.6 cleanup will replace this transitional adapter
    with the canonical at-rest ``UIMessageStore.read()`` path.
    """

    answer: Optional[AgentArtifact] = None
    reference_bundle: Optional[AgentArtifact] = None
    for artifact in artifacts:
        if artifact.artifact_type == AgentArtifactType.ANSWER and answer is None:
            answer = artifact
        elif artifact.artifact_type == AgentArtifactType.REFERENCE_BUNDLE and reference_bundle is None:
            reference_bundle = artifact

    parts: list[UIMessagePart] = []

    if answer is not None:
        text = _extract_answer_text(answer)
        if text:
            parts.append(TextPart(text=text))

    if reference_bundle is not None:
        for source_part, citation_part in _project_reference_items(reference_bundle):
            if source_part is not None:
                parts.append(source_part)
            parts.append(citation_part)

    return parts


def extract_error_text(artifacts: Iterable[AgentArtifact]) -> Optional[str]:
    """Return the human-readable error message for a failed turn, or None.

    Looks up the ``error_summary`` artifact and pulls
    ``payload.message`` / ``payload.text`` / ``summary`` in that order.
    Mirrors the FE adapter's ``extractErrorTextFromSnapshot`` so a
    historical FAILED turn renders the same error string from either
    side of the new boundary.
    """

    for artifact in artifacts:
        if artifact.artifact_type != AgentArtifactType.ERROR_SUMMARY:
            continue
        payload = artifact.payload if isinstance(artifact.payload, dict) else {}
        message = payload.get("message") or payload.get("text") or payload.get("summary")
        if message:
            return str(message)
        if artifact.summary:
            return str(artifact.summary)
    return None


def _extract_answer_text(artifact: AgentArtifact) -> str:
    payload = artifact.payload if isinstance(artifact.payload, dict) else {}
    text = payload.get("text") or payload.get("content") or payload.get("answer")
    if text:
        return str(text)
    if artifact.summary:
        return str(artifact.summary)
    return ""


def _project_reference_items(
    artifact: AgentArtifact,
) -> Iterable[tuple[Optional[SourceUrlPart], DataCitationPart]]:
    payload = artifact.payload if isinstance(artifact.payload, dict) else {}
    raw_items = payload.get("items")
    if not isinstance(raw_items, list):
        return

    for index, raw in enumerate(raw_items):
        if not isinstance(raw, dict):
            continue
        metadata = raw.get("metadata") if isinstance(raw.get("metadata"), dict) else {}
        url = raw.get("uri") or metadata.get("url")
        title = raw.get("title") or metadata.get("title")
        snippet = raw.get("snippet") or raw.get("content") or ""
        source_id = raw.get("source_id") or raw.get("id") or f"{artifact.id}-ref-{index}"

        source_part: Optional[SourceUrlPart] = None
        if url:
            source_part = SourceUrlPart(source_id=str(source_id), url=str(url), title=title)

        citation_part = DataCitationPart(
            data=CitationData(
                cited_text=str(snippet),
                location=UrlCitationLocation(url=str(url) if url else "", title=title),
            )
        )
        yield source_part, citation_part
