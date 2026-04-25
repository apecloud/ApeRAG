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

"""Contract tests for D9 §2 three-level tool authorization (Phase 8 #75)."""

from __future__ import annotations

from aperag.domains.agent_runtime.tools.authorization import (
    Principal,
    ToolAuthorizationPolicy,
    ToolRiskClassification,
    default_policy,
)


def _policy(
    risk_map: dict[str, ToolRiskClassification], owner_map: dict[str, str] | None = None
) -> ToolAuthorizationPolicy:
    return default_policy(
        risk_resolver=lambda name: risk_map.get(name),
        owner_resolver=(lambda name: (owner_map or {}).get(name)),
    )


def test_read_only_tool_is_visible_and_auto_invocable():
    policy = _policy({"search": ToolRiskClassification.READ_ONLY})
    decision = policy.evaluate(Principal(user_id="u1"), "search")
    assert decision.visible is True
    assert decision.can_invoke_auto is True
    assert decision.requires_consent is False
    assert decision.risk is None


def test_unknown_tool_default_deny_per_security_canonical():
    """Architect canonical lock msg=19f2c9a9: missing risk
    classification MUST surface as consent-required (default-deny)
    so a misclassified side-effect tool cannot silently bypass the
    consent gate."""

    policy = _policy({})
    decision = policy.evaluate(Principal(user_id="u1"), "unknown")
    assert decision.visible is True
    assert decision.can_invoke_auto is False
    assert decision.requires_consent is True
    assert decision.risk == "writes_user_data"
    assert "default-deny" in decision.reason


def test_admin_tool_hidden_from_non_admin_principal():
    policy = _policy({"system_config": ToolRiskClassification.ADMIN_ONLY})
    decision = policy.evaluate(Principal(user_id="u1", is_admin=False), "system_config")
    assert decision.visible is False
    assert decision.can_invoke_auto is False
    assert decision.requires_consent is False


def test_admin_tool_auto_invocable_for_admin_principal():
    policy = _policy({"system_config": ToolRiskClassification.ADMIN_ONLY})
    decision = policy.evaluate(Principal(user_id="admin", is_admin=True), "system_config")
    assert decision.visible is True
    assert decision.can_invoke_auto is True
    assert decision.requires_consent is False


def test_user_personal_tool_visible_only_to_owner():
    policy = _policy(
        {"user_notes_create": ToolRiskClassification.USER_PERSONAL},
        owner_map={"user_notes_create": "user-1"},
    )

    own = policy.evaluate(Principal(user_id="user-1"), "user_notes_create")
    other = policy.evaluate(Principal(user_id="user-2"), "user_notes_create")
    assert own.visible is True
    assert own.can_invoke_auto is True
    assert other.visible is False
    assert other.can_invoke_auto is False


def test_user_personal_tool_with_no_owner_resolver_is_hidden():
    policy = _policy({"orphan": ToolRiskClassification.USER_PERSONAL})
    decision = policy.evaluate(Principal(user_id="u1"), "orphan")
    assert decision.visible is False


def test_side_effect_user_data_tool_requires_consent():
    policy = _policy({"write_file": ToolRiskClassification.SIDE_EFFECT_USER_DATA})
    decision = policy.evaluate(Principal(user_id="u1"), "write_file")
    assert decision.visible is True
    assert decision.can_invoke_auto is False
    assert decision.requires_consent is True
    assert decision.risk == "writes_user_data"


def test_side_effect_external_tool_requires_consent_with_distinct_risk():
    policy = _policy({"send_email": ToolRiskClassification.SIDE_EFFECT_EXTERNAL})
    decision = policy.evaluate(Principal(user_id="u1"), "send_email")
    assert decision.requires_consent is True
    assert decision.risk == "calls_external_api"


def test_side_effect_system_tool_requires_consent_for_admin_too():
    policy = _policy({"db_modify": ToolRiskClassification.SIDE_EFFECT_SYSTEM})
    decision = policy.evaluate(Principal(user_id="admin", is_admin=True), "db_modify")
    assert decision.requires_consent is True
    assert decision.risk == "modifies_system"


def test_filter_visible_drops_admin_only_for_user():
    policy = _policy(
        {
            "search": ToolRiskClassification.READ_ONLY,
            "system_config": ToolRiskClassification.ADMIN_ONLY,
            "write_file": ToolRiskClassification.SIDE_EFFECT_USER_DATA,
        }
    )
    visible = policy.filter_visible(
        Principal(user_id="u1"),
        ["search", "system_config", "write_file"],
    )
    assert visible == ["search", "write_file"]


def test_unknown_tool_filter_visible_keeps_consent_required_tool():
    """The unknown-tool default-deny still keeps the tool visible to
    the LLM (consent-required, not invisible). filter_visible only
    drops invisible ones."""

    policy = _policy({})
    visible = policy.filter_visible(Principal(user_id="u1"), ["mystery"])
    assert visible == ["mystery"]
