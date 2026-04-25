"""At-rest UIMessage storage contract tests (Phase 8 D8.2 / #74).

These tests pin the at-rest persistence contract Weston named as the
prerequisite for unblocking D8.4b FE message-parts renderer
(msg=50c90f6f / msg=cef89ed8):

1. **Round-trip fidelity** — every persistable ``UIMessagePart``
   variant survives a ``write`` → ``read`` cycle without loss.
2. **Transient exclusion** — parts that carry ``transient=True``
   (currently only ``data-activity``) are stripped before write and
   never reappear on read.
3. **Snapshot consistency** — Redis cache and DB row hold the same
   shape; loss of one is recoverable from the other.

The tests use in-memory fakes for both the SQLAlchemy session
boundary and the Redis client so they are hermetic and fast — the
real DB / Redis behaviour is exercised by Phase 8 e2e tests once
the wire emitter (D8.1, #73) starts producing UIMessage payloads.
"""

from __future__ import annotations

from typing import Any, Optional

import pytest

from aperag.domains.agent_runtime.schemas import (
    AGENT_RUNTIME_SCHEMA_VERSION,
    UserActivityEnvelope,
    UserActivityIntent,
)
from aperag.domains.agent_runtime.uimessage import (
    ActivityData,
    CitationData,
    DataActivityPart,
    DataCitationPart,
    DataElicitationPart,
    DataToolConsentPart,
    ElicitationData,
    SourceDocumentPart,
    SourceUrlPart,
    TextPart,
    ToolConsentData,
    ToolPart,
    UIMessage,
    UrlCitationLocation,
    args_hash,
    args_preview,
    persistable_parts,
)
from aperag.domains.agent_runtime.uimessage_store import UIMessageStore

# ---------------------------------------------------------------------
# In-memory fakes
# ---------------------------------------------------------------------


class _FakeDbOps:
    """Minimal in-memory stand-in for ``UIMessageDbOps``.

    Only implements the three calls :class:`UIMessageStore` needs.
    Keyed by ``turn_id`` since the real ``agent_message`` table is
    1:1 with ``agent_turn``.
    """

    def __init__(self) -> None:
        self.rows: dict[str, dict[str, Any]] = {}

    async def upsert_agent_message(
        self,
        *,
        turn_id: str,
        chat_id: str,
        role: str,
        schema_version: str,
        parts: list[dict[str, Any]],
        gmt_updated: Any,
    ) -> None:
        existing = self.rows.get(turn_id, {})
        message_id = existing.get("id", f"msg-{turn_id}")
        self.rows[turn_id] = {
            "id": message_id,
            "turn_id": turn_id,
            "chat_id": chat_id,
            "role": role,
            "schema_version": schema_version,
            "parts": parts,
            "gmt_updated": gmt_updated,
        }

    async def query_agent_message(self, turn_id: str) -> Optional[Any]:
        row = self.rows.get(turn_id)
        if row is None:
            return None

        class _Row:
            def __init__(self, data: dict[str, Any]) -> None:
                for key, value in data.items():
                    setattr(self, key, value)

        return _Row(row)

    async def delete_agent_message(self, turn_id: str) -> None:
        self.rows.pop(turn_id, None)


class _FakeRedisStore:
    """Minimal in-memory stand-in for ``AgentRuntimeRedisStore``.

    Only the at-rest snapshot keys are implemented; the live event
    buffer / lease logic is exercised elsewhere.
    """

    def __init__(self) -> None:
        self.snapshots: dict[str, dict[str, Any]] = {}

    async def write_message_snapshot(self, turn_id: str, payload: dict[str, Any]) -> None:
        self.snapshots[turn_id] = payload

    async def read_message_snapshot(self, turn_id: str) -> Optional[dict[str, Any]]:
        return self.snapshots.get(turn_id)

    async def delete_message_snapshot(self, turn_id: str) -> None:
        self.snapshots.pop(turn_id, None)


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------


def _every_part_message() -> UIMessage:
    """Build a UIMessage that exercises every persistable part variant
    plus one transient ``data-activity`` part. Used to anchor the
    round-trip + transient-exclusion contract.
    """

    raw_args = {"query": "ApeRAG", "top_k": 3}
    return UIMessage(
        id="msg-fixture-1",
        role="assistant",
        parts=[
            TextPart(text="Here is what I found:"),
            ToolPart(
                type="tool-aperag_knowledge_base_search_collection",
                tool_call_id="tc-1",
                state="output-available",
                input=raw_args,
                output={"results": [{"id": "doc-1", "score": 0.92}]},
                metadata={"mcpServer": "aperag", "mcpToolName": "knowledge-base.search_collection"},
            ),
            SourceUrlPart(source_id="src-1", url="https://example.com/a", title="Example A"),
            SourceDocumentPart(source_id="doc-1", media_type="application/pdf", title="ApeRAG paper"),
            DataCitationPart(
                data=CitationData(
                    cited_text="ApeRAG is a retrieval augmented generation platform.",
                    location=UrlCitationLocation(url="https://example.com/a", title="Example A"),
                ),
            ),
            DataActivityPart(
                data=ActivityData(
                    activity=UserActivityEnvelope(
                        intent=UserActivityIntent.SEARCHING_KNOWLEDGE,
                        title_key="searching.title",
                        subtitle_key="searching.subtitle",
                    ),
                ),
            ),
            DataToolConsentPart(
                data=ToolConsentData(
                    tool_call_id="tc-2",
                    tool_name="aperag_admin_modify_system",
                    args_preview="{...}",
                    args_hash="0" * 64,
                    risk="modifies_system",
                    requested_at="2026-04-25T00:00:00Z",
                    state="pending",
                ),
            ),
            DataElicitationPart(
                data=ElicitationData(
                    elicitation_id="el-1",
                    server_name="aperag-knowledge-base",
                    prompt="Please confirm the date range",
                    schema={"type": "object", "properties": {"from": {"type": "string"}}},
                    response=None,
                    state="pending",
                ),
            ),
        ],
    )


# ---------------------------------------------------------------------
# Contract tests
# ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_uimessage_round_trip_preserves_persistable_parts():
    """Every non-transient ``UIMessagePart`` variant survives write→read."""

    db = _FakeDbOps()
    redis = _FakeRedisStore()
    store = UIMessageStore(db_ops=db, redis_store=redis)

    original = _every_part_message()
    await store.write(turn_id="turn-1", chat_id="chat-1", message=original)
    reloaded = await store.read("turn-1")

    assert reloaded is not None
    expected_types = [
        "text",
        "tool-aperag_knowledge_base_search_collection",
        "source-url",
        "source-document",
        "data-citation",
        "data-tool-consent",
        "data-elicitation",
    ]
    actual_types = [getattr(part, "type", None) for part in reloaded.parts]
    assert actual_types == expected_types, "round-trip must preserve every persistable part variant"
    assert reloaded.role == original.role
    assert reloaded.schema_version == AGENT_RUNTIME_SCHEMA_VERSION


@pytest.mark.asyncio
async def test_transient_parts_are_stripped_before_persistence():
    """``data-activity`` parts (transient=True) must never reach the at-rest store."""

    db = _FakeDbOps()
    redis = _FakeRedisStore()
    store = UIMessageStore(db_ops=db, redis_store=redis)

    original = _every_part_message()
    written = await store.write(turn_id="turn-2", chat_id="chat-2", message=original)

    written_types = [getattr(part, "type", None) for part in written.parts]
    assert "data-activity" not in written_types

    redis_payload = redis.snapshots["turn-2"]
    redis_types = [part.get("type") for part in redis_payload["parts"]]
    assert "data-activity" not in redis_types

    db_row = db.rows["turn-2"]
    db_types = [part.get("type") for part in db_row["parts"]]
    assert "data-activity" not in db_types


@pytest.mark.asyncio
async def test_snapshot_consistency_db_and_redis_agree():
    """Both stores hold the same shape after write; either can serve a read."""

    db = _FakeDbOps()
    redis = _FakeRedisStore()
    store = UIMessageStore(db_ops=db, redis_store=redis)

    await store.write(turn_id="turn-3", chat_id="chat-3", message=_every_part_message())

    redis_parts = redis.snapshots["turn-3"]["parts"]
    db_parts = db.rows["turn-3"]["parts"]
    assert redis_parts == db_parts

    # Redis goes cold (TTL expiry simulation) — read still works via DB fallback.
    await redis.delete_message_snapshot("turn-3")
    fallback = await store.read("turn-3")
    assert fallback is not None
    assert [getattr(part, "type", None) for part in fallback.parts] == [part["type"] for part in db_parts]


@pytest.mark.asyncio
async def test_write_is_idempotent_on_turn_id():
    """A second write to the same turn_id refreshes the row instead of duplicating it."""

    db = _FakeDbOps()
    redis = _FakeRedisStore()
    store = UIMessageStore(db_ops=db, redis_store=redis)

    first = UIMessage(id="msg-x", role="assistant", parts=[TextPart(text="initial")])
    second = UIMessage(id="msg-x", role="assistant", parts=[TextPart(text="updated")])

    await store.write(turn_id="turn-4", chat_id="chat-4", message=first)
    await store.write(turn_id="turn-4", chat_id="chat-4", message=second)

    assert len(db.rows) == 1
    reloaded = await store.read("turn-4")
    assert reloaded is not None
    assert isinstance(reloaded.parts[0], TextPart)
    assert reloaded.parts[0].text == "updated"


def test_args_preview_truncates_long_payloads():
    long_blob = {"data": "x" * 2000}
    preview = args_preview(long_blob, limit=200)
    assert preview.endswith("...<truncated>")
    assert len(preview) <= 200 + len("...<truncated>")


def test_args_hash_is_stable_canonical_sha256():
    a = {"top_k": 3, "query": "ApeRAG"}
    b = {"query": "ApeRAG", "top_k": 3}
    assert args_hash(a) == args_hash(b)
    assert len(args_hash(a)) == 64
    assert all(ch in "0123456789abcdef" for ch in args_hash(a))


@pytest.mark.asyncio
async def test_tool_part_type_uses_safe_tool_name_form():
    """``ToolPart.type`` must be ``tool-<SafeToolName>`` per D8 §2.4 / D9 §A1+A6.

    Architect lock 2026-04-25 (msg=8412dce5): the at-rest tool part
    encodes the SafeToolName in its ``type`` discriminator (matching
    the AI SDK v5 consolidated form) — there is no separate
    ``tool_name`` field, and the wire emitter (#73) plus the FE
    consumer (#76/#77) must round-trip against the same shape.
    """

    db = _FakeDbOps()
    redis = _FakeRedisStore()
    store = UIMessageStore(db_ops=db, redis_store=redis)

    await store.write(turn_id="turn-tool", chat_id="chat-tool", message=_every_part_message())

    tool_parts = [part for part in db.rows["turn-tool"]["parts"] if part["type"].startswith("tool-")]
    assert len(tool_parts) == 1
    tool = tool_parts[0]
    assert tool["type"] == "tool-aperag_knowledge_base_search_collection"
    assert "tool_name" not in tool, "tool_name must not be a top-level field; SafeToolName is encoded in 'type'"
    assert tool["state"] == "output-available"
    assert tool["toolCallId"] == "tc-1"
    assert tool["metadata"] == {
        "mcpServer": "aperag",
        "mcpToolName": "knowledge-base.search_collection",
    }


@pytest.mark.asyncio
async def test_data_parts_use_wrapped_data_shape():
    """``data-*`` parts persist as ``{type, data: {...}}`` per D8 §2 same-schema canonical.

    Architect lock 2026-04-25 (msg=ad6168e7 / msg=1ff7ed9e): wire (cuiwenbo
    #73) and at-rest (this PR) must round-trip byte-for-byte with no
    converter layer. FE renderer (huangheng #76/#77) consumes the same
    discriminated-union shape from both sources.
    """

    db = _FakeDbOps()
    redis = _FakeRedisStore()
    store = UIMessageStore(db_ops=db, redis_store=redis)

    await store.write(turn_id="turn-shape", chat_id="chat-shape", message=_every_part_message())

    persisted = {part["type"]: part for part in db.rows["turn-shape"]["parts"]}
    for wrapped_type in ("data-citation", "data-tool-consent", "data-elicitation"):
        part = persisted[wrapped_type]
        assert set(part.keys()) == {"type", "data"}, f"{wrapped_type} must be wrapped, got keys {part.keys()}"
        assert isinstance(part["data"], dict) and part["data"], f"{wrapped_type}.data must be a non-empty object"

    citation = persisted["data-citation"]["data"]
    assert "cited_text" in citation and "location" in citation  # Anthropic-shape stays snake_case per D8 §2

    consent = persisted["data-tool-consent"]["data"]
    assert {"toolCallId", "toolName", "argsPreview", "argsHash", "requestedAt", "risk", "state"} <= consent.keys()

    elicitation = persisted["data-elicitation"]["data"]
    assert {"elicitationId", "serverName", "prompt", "schema", "state"} <= elicitation.keys()


@pytest.mark.asyncio
async def test_persisted_keys_use_canonical_camelcase():
    """AI SDK v5 standard parts + custom data payloads must serialize with
    canonical camelCase keys per D8 §2, matching the wire produced by #73.

    Architect lock 2026-04-25 (msg=935bdb9d follow-up / Weston msg=59a459c6):
    same-schema invariant covers key casing, not just nested structure.
    Failing this regression would force #76/#77 FE renderer to handle
    snake_case-from-storage and camelCase-from-stream as two distinct
    shapes.
    """

    db = _FakeDbOps()
    redis = _FakeRedisStore()
    store = UIMessageStore(db_ops=db, redis_store=redis)

    await store.write(turn_id="turn-casing", chat_id="chat-casing", message=_every_part_message())

    by_type = {part["type"]: part for part in db.rows["turn-casing"]["parts"]}

    tool = by_type["tool-aperag_knowledge_base_search_collection"]
    assert "toolCallId" in tool and "tool_call_id" not in tool
    assert "error_text" not in tool  # snake_case form must never appear (errorText is the canonical alias)

    source_url = by_type["source-url"]
    assert "sourceId" in source_url and "source_id" not in source_url

    source_doc = by_type["source-document"]
    assert "sourceId" in source_doc and "source_id" not in source_doc
    assert "mediaType" in source_doc and "media_type" not in source_doc

    consent = by_type["data-tool-consent"]["data"]
    snake_forbidden = {"tool_call_id", "tool_name", "args_preview", "args_hash", "requested_at"}
    assert snake_forbidden.isdisjoint(consent.keys()), (
        f"snake_case keys leaked into data-tool-consent.data: {snake_forbidden & consent.keys()}"
    )

    elicitation = by_type["data-elicitation"]["data"]
    assert "elicitationId" in elicitation and "elicitation_id" not in elicitation
    assert "serverName" in elicitation and "server_name" not in elicitation
    assert elicitation["state"] in {"pending", "answered", "cancelled"}, (
        f"data-elicitation.state must be canonical per D9 §5.1, got {elicitation['state']!r}"
    )


@pytest.mark.asyncio
async def test_data_elicitation_answered_state_round_trip():
    """``DataElicitationPart`` must accept the canonical ``answered`` state per D9 §5.1.

    Architect lock 2026-04-25 (Weston msg=51dffdc9 / PM msg=042b0a7b):
    state literal is ``pending | answered | cancelled`` — the previous
    ``submitted`` value was non-canonical and would have forced #75 to
    rewrite state on emit.
    """

    db = _FakeDbOps()
    redis = _FakeRedisStore()
    store = UIMessageStore(db_ops=db, redis_store=redis)

    answered = UIMessage(
        id="msg-elicit",
        role="assistant",
        parts=[
            DataElicitationPart(
                data=ElicitationData(
                    elicitation_id="el-2",
                    server_name="aperag-knowledge-base",
                    prompt="Pick a date",
                    schema={"type": "object"},
                    response={"value": "2026-04-25"},
                    state="answered",
                ),
            ),
        ],
    )
    await store.write(turn_id="turn-elicit", chat_id="chat-elicit", message=answered)
    reloaded = await store.read("turn-elicit")

    assert reloaded is not None
    assert isinstance(reloaded.parts[0], DataElicitationPart)
    assert reloaded.parts[0].data.state == "answered"
    assert reloaded.parts[0].data.server_name == "aperag-knowledge-base"


def test_persistable_parts_helper_strips_only_transient():
    parts = _every_part_message().parts
    persisted = persistable_parts(parts)
    assert len(persisted) == len(parts) - 1
    assert all(getattr(part, "transient", False) is not True for part in persisted)
