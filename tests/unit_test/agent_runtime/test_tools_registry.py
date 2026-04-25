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

"""Contract tests for D9 §1.1 + §A5 MCP server registry (Phase 8 #75)."""

from __future__ import annotations

import pytest

from aperag.domains.agent_runtime.tools.registry import (
    MCPServerEntry,
    MCPServerRegistry,
    RegistryConflictError,
    RegistryScope,
)


def _make_entry(
    scope: RegistryScope, name: str, *, scope_ref: str | None = None, enabled: bool = True
) -> MCPServerEntry:
    return MCPServerEntry(
        scope=scope,
        name=name,
        url=f"https://mcp.test/{name}",
        scope_ref=scope_ref,
        enabled=enabled,
    )


def test_register_system_entry_audit_logged():
    audit: list[tuple[str, dict]] = []
    registry = MCPServerRegistry(audit_logger=lambda evt, payload: audit.append((evt, payload)))
    entry = _make_entry(RegistryScope.SYSTEM, "aperag-knowledge-base")
    registry.register(entry)
    assert audit == [
        (
            "registry.registered",
            {
                "scope": "system",
                "scope_ref": None,
                "name": "aperag-knowledge-base",
                "url": "https://mcp.test/aperag-knowledge-base",
                "enabled": True,
            },
        )
    ]


def test_user_cannot_silently_register_in_system_namespace():
    audit: list[tuple[str, dict]] = []
    registry = MCPServerRegistry(audit_logger=lambda evt, payload: audit.append((evt, payload)))
    registry.register(_make_entry(RegistryScope.SYSTEM, "aperag-knowledge-base"))

    with pytest.raises(RegistryConflictError):
        registry.register(_make_entry(RegistryScope.USER, "aperag-knowledge-base", scope_ref="user-1"))

    # Audit trail captures the rejection so the caller can flag namespace pressure.
    assert any(evt == "registry.system_namespace_rejected" for evt, _ in audit)


def test_bot_cannot_silently_register_in_system_namespace():
    registry = MCPServerRegistry()
    registry.register(_make_entry(RegistryScope.SYSTEM, "system-tool"))
    with pytest.raises(RegistryConflictError):
        registry.register(_make_entry(RegistryScope.BOT, "system-tool", scope_ref="bot-1"))


def test_admin_alias_audit_logs_with_actor_and_reason():
    audit: list[tuple[str, dict]] = []
    registry = MCPServerRegistry(audit_logger=lambda evt, payload: audit.append((evt, payload)))
    registry.register(_make_entry(RegistryScope.SYSTEM, "system-tool"))
    alias = MCPServerEntry(
        scope=RegistryScope.BOT,
        name="system-tool",
        url="https://override.test/system-tool",
        scope_ref="bot-1",
    )
    registry.register_admin_alias(
        target_scope=RegistryScope.BOT,
        target_scope_ref="bot-1",
        alias=alias,
        admin_user_id="admin-42",
        reason="bot-1 ships custom build of system-tool with hotfix",
    )
    alias_evts = [payload for evt, payload in audit if evt == "registry.admin_alias_registered"]
    assert len(alias_evts) == 1
    assert alias_evts[0]["admin_user_id"] == "admin-42"
    assert "hotfix" in alias_evts[0]["reason"]


def test_admin_alias_rejects_scope_mismatch():
    registry = MCPServerRegistry()
    alias = MCPServerEntry(
        scope=RegistryScope.USER,
        name="x",
        url="https://x",
    )
    with pytest.raises(ValueError):
        registry.register_admin_alias(
            target_scope=RegistryScope.BOT,
            target_scope_ref="bot-1",
            alias=alias,
            admin_user_id="admin",
            reason="r",
        )


def test_effective_servers_system_takes_precedence_over_bot_user():
    registry = MCPServerRegistry()
    registry.register(_make_entry(RegistryScope.SYSTEM, "shared-name"))
    # Try to register bot/user entries with the same name; should fail
    # at register time per no-silent-override.
    with pytest.raises(RegistryConflictError):
        registry.register(_make_entry(RegistryScope.BOT, "shared-name", scope_ref="bot"))


def test_effective_servers_iteration_order_is_system_bot_user():
    registry = MCPServerRegistry()
    registry.register(_make_entry(RegistryScope.SYSTEM, "system-only"))
    registry.register(_make_entry(RegistryScope.BOT, "bot-only", scope_ref="bot-1"))
    registry.register(_make_entry(RegistryScope.USER, "user-only", scope_ref="user-1"))

    out = registry.effective_servers(user_id="user-1", bot_id="bot-1")
    assert [e.name for e in out] == ["system-only", "bot-only", "user-only"]


def test_effective_servers_filters_disabled_entries():
    registry = MCPServerRegistry()
    registry.register(_make_entry(RegistryScope.SYSTEM, "live"))
    registry.register(_make_entry(RegistryScope.SYSTEM, "dead", enabled=False))
    out = registry.effective_servers(user_id="u", bot_id="b")
    assert [e.name for e in out] == ["live"]


def test_effective_servers_scopes_bot_entries_to_owning_bot():
    registry = MCPServerRegistry()
    registry.register(_make_entry(RegistryScope.BOT, "bot-a-tool", scope_ref="bot-A"))
    registry.register(_make_entry(RegistryScope.BOT, "bot-b-tool", scope_ref="bot-B"))
    out = registry.effective_servers(user_id="u", bot_id="bot-A")
    assert [e.name for e in out] == ["bot-a-tool"]


def test_effective_servers_scopes_user_entries_to_owning_user():
    registry = MCPServerRegistry()
    registry.register(_make_entry(RegistryScope.USER, "u1-tool", scope_ref="user-1"))
    registry.register(_make_entry(RegistryScope.USER, "u2-tool", scope_ref="user-2"))
    out = registry.effective_servers(user_id="user-1", bot_id=None)
    assert [e.name for e in out] == ["u1-tool"]


def test_unregister_returns_false_for_unknown_name():
    registry = MCPServerRegistry()
    assert registry.unregister(RegistryScope.SYSTEM, "never") is False


def test_unregister_removes_and_audit_logs():
    audit: list[tuple[str, dict]] = []
    registry = MCPServerRegistry(audit_logger=lambda evt, payload: audit.append((evt, payload)))
    registry.register(_make_entry(RegistryScope.SYSTEM, "tmp"))
    assert registry.unregister(RegistryScope.SYSTEM, "tmp") is True
    assert any(evt == "registry.unregistered" for evt, _ in audit)
