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

"""Contract tests for D9 §3 + §A7 consent service (Phase 8 #75)."""

from __future__ import annotations

import asyncio

import pytest

from aperag.domains.agent_runtime.tools.consent import ConsentService


@pytest.mark.asyncio
async def test_request_consent_emits_pending_payload_with_redaction():
    audit: list[tuple[str, dict]] = []
    svc = ConsentService(audit_logger=lambda evt, payload: audit.append((evt, payload)))
    raw = {"path": "/etc/passwd", "content": "secret-token"}
    result = await svc.request_consent(
        consent_id="call-1",
        tool_name="aperag_fs_write_file",
        raw_args=raw,
        risk="writes_user_data",
        metadata={"mcpServer": "aperag-fs", "mcpToolName": "write_file"},
    )
    assert result.consent_id == "call-1"
    assert result.payload.state == "pending"
    assert result.payload.tool_name == "aperag_fs_write_file"
    # argsPreview is the JSON-stringified raw args; raw values must
    # appear within the limit, not the full payload.
    assert "secret-token" in result.payload.args_preview
    # argsHash is sha256 (64 hex chars)
    assert len(result.payload.args_hash) == 64
    assert any(evt == "consent.requested" for evt, _ in audit)


@pytest.mark.asyncio
async def test_request_consent_rejects_duplicate_id():
    svc = ConsentService()
    await svc.request_consent(
        consent_id="dup",
        tool_name="t",
        raw_args={},
        risk="writes_user_data",
    )
    with pytest.raises(ValueError):
        await svc.request_consent(
            consent_id="dup",
            tool_name="t",
            raw_args={},
            risk="writes_user_data",
        )


@pytest.mark.asyncio
async def test_decide_approves_and_runtime_can_consume_raw_args():
    svc = ConsentService()
    raw = {"command": "rm -rf /tmp/scratch"}
    await svc.request_consent(
        consent_id="c1",
        tool_name="cli",
        raw_args=raw,
        risk="modifies_system",
    )

    async def runtime_waiter():
        return await svc.wait_for_decision("c1")

    waiter_task = asyncio.create_task(runtime_waiter())
    await asyncio.sleep(0)  # let the waiter park
    await svc.decide("c1", "approved", actor_user_id="user-42")
    decision = await waiter_task
    assert decision.decision == "approved"
    # Raw args fetched once -> dispatched
    assert await svc.consume_raw_args("c1") == raw
    # Second fetch returns None (single-use)
    assert await svc.consume_raw_args("c1") is None


@pytest.mark.asyncio
async def test_denial_drops_raw_args_immediately():
    svc = ConsentService()
    await svc.request_consent(
        consent_id="c1",
        tool_name="t",
        raw_args={"x": 1},
        risk="writes_user_data",
    )
    await svc.decide("c1", "denied", actor_user_id="user")
    assert await svc.consume_raw_args("c1") is None


@pytest.mark.asyncio
async def test_decide_rejects_invalid_decision():
    svc = ConsentService()
    await svc.request_consent(
        consent_id="c1",
        tool_name="t",
        raw_args={},
        risk="writes_user_data",
    )
    with pytest.raises(ValueError):
        await svc.decide("c1", "later", actor_user_id="u")


@pytest.mark.asyncio
async def test_decide_unknown_consent_raises_keyerror():
    svc = ConsentService()
    with pytest.raises(KeyError):
        await svc.decide("never", "approved", actor_user_id="u")


@pytest.mark.asyncio
async def test_decide_twice_rejected():
    svc = ConsentService()
    await svc.request_consent(
        consent_id="c1",
        tool_name="t",
        raw_args={},
        risk="writes_user_data",
    )
    await svc.decide("c1", "approved", actor_user_id="u")
    with pytest.raises(ValueError):
        await svc.decide("c1", "denied", actor_user_id="u")


@pytest.mark.asyncio
async def test_wait_for_decision_expires_on_timeout():
    svc = ConsentService(default_timeout_seconds=1)
    await svc.request_consent(
        consent_id="c1",
        tool_name="t",
        raw_args={},
        risk="writes_user_data",
    )
    decision = await svc.wait_for_decision("c1", timeout_seconds=0.05)
    assert decision.decision == "expired"
    # Raw args dropped on expiry
    assert await svc.consume_raw_args("c1") is None


@pytest.mark.asyncio
async def test_args_hash_stable_across_consents_for_same_args():
    svc = ConsentService()
    a = await svc.request_consent(
        consent_id="a",
        tool_name="t",
        raw_args={"k": "v"},
        risk="writes_user_data",
    )
    b = await svc.request_consent(
        consent_id="b",
        tool_name="t",
        raw_args={"k": "v"},
        risk="writes_user_data",
    )
    assert a.payload.args_hash == b.payload.args_hash
