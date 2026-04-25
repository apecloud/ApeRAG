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

"""Contract tests for D9 §5 elicitation service (Phase 8 #75)."""

from __future__ import annotations

import asyncio

import pytest

from aperag.domains.agent_runtime.tools.elicitation import ElicitationService

_REQUIRED_SCHEMA = {
    "type": "object",
    "required": ["path"],
    "properties": {"path": {"type": "string"}},
}


@pytest.mark.asyncio
async def test_request_input_emits_pending_payload():
    audit: list[tuple[str, dict]] = []
    svc = ElicitationService(audit_logger=lambda evt, payload: audit.append((evt, payload)))
    result = await svc.request_input(
        elicitation_id="e1",
        server_name="aperag-fs",
        prompt="Which file should I write to?",
        schema=_REQUIRED_SCHEMA,
    )
    assert result.payload.elicitation_id == "e1"
    assert result.payload.state == "pending"
    assert result.payload.schema_["required"] == ["path"]
    assert any(evt == "elicitation.requested" for evt, _ in audit)


@pytest.mark.asyncio
async def test_submit_validates_required_fields_and_records_response():
    svc = ElicitationService()
    await svc.request_input(
        elicitation_id="e1",
        server_name="aperag-fs",
        prompt="Provide path",
        schema=_REQUIRED_SCHEMA,
    )

    async def runtime_waiter():
        return await svc.wait_for_input("e1")

    waiter_task = asyncio.create_task(runtime_waiter())
    await asyncio.sleep(0)
    await svc.submit(
        "e1",
        {"path": "/tmp/notes.md"},
        actor_user_id="user",
    )
    result = await waiter_task
    assert result.outcome == "answered"
    assert result.payload.state == "answered"
    assert result.payload.response == {"path": "/tmp/notes.md"}


@pytest.mark.asyncio
async def test_submit_rejects_response_missing_required_field():
    svc = ElicitationService()
    await svc.request_input(
        elicitation_id="e1",
        server_name="aperag-fs",
        prompt="Provide path",
        schema=_REQUIRED_SCHEMA,
    )
    with pytest.raises(ValueError, match="missing required fields"):
        await svc.submit("e1", {"other": "x"}, actor_user_id="user")
    # Elicitation stays pending after a validation failure -- caller
    # can re-prompt without a separate cancel + re-request.
    assert svc.get_payload("e1").state == "pending"


@pytest.mark.asyncio
async def test_submit_unknown_elicitation_raises_keyerror():
    svc = ElicitationService()
    with pytest.raises(KeyError):
        await svc.submit("never", {}, actor_user_id="user")


@pytest.mark.asyncio
async def test_submit_after_resolve_rejected():
    svc = ElicitationService()
    await svc.request_input(elicitation_id="e1", server_name="aperag-fs", prompt="?", schema={})
    await svc.submit("e1", {}, actor_user_id="user")
    with pytest.raises(ValueError, match="already resolved"):
        await svc.submit("e1", {}, actor_user_id="user")


@pytest.mark.asyncio
async def test_cancel_sets_state_cancelled():
    svc = ElicitationService()
    await svc.request_input(elicitation_id="e1", server_name="aperag-fs", prompt="?", schema={})
    result = await svc.cancel("e1", actor_user_id="user", reason="user-aborted")
    assert result.outcome == "cancelled"
    assert result.payload.state == "cancelled"


@pytest.mark.asyncio
async def test_wait_for_input_cancels_on_timeout():
    svc = ElicitationService(default_timeout_seconds=1)
    await svc.request_input(elicitation_id="e1", server_name="aperag-fs", prompt="?", schema={})
    result = await svc.wait_for_input("e1", timeout_seconds=0.05)
    assert result.outcome == "cancelled"


@pytest.mark.asyncio
async def test_custom_validator_can_override_default():
    svc = ElicitationService()
    await svc.request_input(
        elicitation_id="e1",
        server_name="aperag-fs",
        prompt="?",
        schema={"required": ["x"]},
    )

    def reject_all(schema, response):
        raise ValueError("custom validator says no")

    with pytest.raises(ValueError, match="custom validator says no"):
        await svc.submit("e1", {"x": "v"}, actor_user_id="user", validator=reject_all)


@pytest.mark.asyncio
async def test_request_input_rejects_non_dict_schema():
    svc = ElicitationService()
    with pytest.raises(TypeError):
        await svc.request_input(
            elicitation_id="e1",
            server_name="aperag-fs",
            prompt="?",
            schema=["bad"],  # type: ignore[arg-type]
        )


@pytest.mark.asyncio
async def test_request_input_rejects_empty_server_name():
    svc = ElicitationService()
    with pytest.raises(ValueError, match="server_name"):
        await svc.request_input(elicitation_id="e1", server_name="", prompt="?", schema={})


@pytest.mark.asyncio
async def test_payload_carries_canonical_server_name():
    svc = ElicitationService()
    result = await svc.request_input(
        elicitation_id="e1",
        server_name="aperag-fs",
        prompt="?",
        schema={},
    )
    assert result.payload.server_name == "aperag-fs"
