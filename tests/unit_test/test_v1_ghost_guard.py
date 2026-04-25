"""Guard against regrowth of internal ``/api/v1`` references in e2e Hurl tests.

The end-state allowlist (canonical D7-1 / msg=8b6b4bc3) is:

- ``/api/v1/embeddings`` — OpenAI-compatible public endpoint, permanent ``/api/v1`` mount
- ``/api/v1/rerank``     — OpenAI-compatible public endpoint, permanent ``/api/v1`` mount

Phase 8 task #44 (H3) cleaned the provider-aware Hurl suite to use
``/api/v2/providers/*`` so it no longer hits dead v1 provider CRUD routes.
A few other domains (apikeys, audit, marketplace, chat ops, export,
document upload variants) still legitimately live at
``/api/v1`` until tasks #50–#52 / G4d / G5 land. Those paths are listed
in ``TRANSITIONAL_V1_PREFIXES`` below and each follow-up PR is expected
to shrink the set as it migrates the corresponding domain to ``/api/v2``.
Phase 8 #49 G3 already removed ``/api/v1/prompts`` (carved to
``aperag/domains/model_platform/api/prompts_routes.py`` at
``/api/v2/prompts*``).

This test scans every ``.hurl`` file under ``tests/e2e_http/`` and asserts
each ``/api/v1/...`` literal matches either the permanent allowlist or a
transitional prefix, so a future PR can not silently re-introduce dead
internal v1 CRUD calls.
"""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
HURL_ROOT = REPO_ROOT / "tests" / "e2e_http" / "hurl"

# OpenAI-compat permanent allowlist (canonical D7-1 / msg=8b6b4bc3).
OPENAI_COMPAT_V1_ALLOWLIST: frozenset[str] = frozenset(
    {
        "/api/v1/embeddings",
        "/api/v1/rerank",
    }
)

# Transitional prefixes — these v1 routes are still mounted in main and used
# by e2e Hurl as long as the corresponding G* migration tasks have not landed.
# Each follow-up PR (G4b apikeys / G4d chat ops) is expected to remove
# the matching entry from this set together with its Hurl rewrite.
# Phase 8 #1 (G1 export, #47), #2 (G2 settings, #48), #3 (G3 prompts,
# #49), #50 (G4a audit-logs), and #52 (G4c marketplace) have already
# shrunk this set to remove ``/api/v1/export*``, ``/api/v1/settings*``,
# ``/api/v1/prompts*``, ``/api/v1/audit-logs``, and ``/api/v1/marketplace``.
#
# Anything outside both sets is an unrecognised v1 reference and must be
# either migrated or moved into one of these sets in the same PR.
TRANSITIONAL_V1_PREFIXES: frozenset[str] = frozenset(
    {
        "/api/v1/apikeys",  # G4b → /api/v2/apikeys
        "/api/v1/collections",  # G1 (export) + G5 (document upload variants)
        "/api/v1/export-tasks",  # G1  → /api/v2/export-tasks
        "/api/v1/chats",  # G4d → /api/v2/chats
        "/api/v1/quotas",  # G5 caller cleanup (FE references only)
        "/api/v1/system",  # G5 caller cleanup (admin default-quotas)
        "/api/v1/bots",  # G5 caller cleanup (pytest fixtures)
        "/api/v1/test",  # G5 caller cleanup (test fixtures)
    }
)

V1_PATH_RE = re.compile(r"/api/v1/[A-Za-z0-9_\-/{}.%]+")


def _normalise(path: str) -> str:
    return path.rstrip(",;)\"'")


def _is_allowed(path: str) -> bool:
    if path in OPENAI_COMPAT_V1_ALLOWLIST:
        return True
    for prefix in OPENAI_COMPAT_V1_ALLOWLIST | TRANSITIONAL_V1_PREFIXES:
        if path == prefix or path.startswith(prefix + "/") or path.startswith(prefix + "?"):
            return True
    return False


def test_e2e_hurl_v1_routes_match_allowlist():
    offenders: dict[str, set[str]] = {}
    for hurl in HURL_ROOT.rglob("*.hurl"):
        text = hurl.read_text()
        seen: set[str] = set()
        for raw_match in V1_PATH_RE.findall(text):
            path = _normalise(raw_match)
            if not _is_allowed(path):
                seen.add(path)
        if seen:
            offenders[str(hurl.relative_to(REPO_ROOT))] = seen

    assert not offenders, (
        "tests/e2e_http/hurl/** may only reference /api/v1 paths in the OpenAI-compat "
        "permanent allowlist or the transitional pre-migration set "
        "(see TRANSITIONAL_V1_PREFIXES). Unrecognised v1 references must either be "
        "migrated to /api/v2 or, if they are a new mount, added to the allowlist in "
        f"the same PR. Offenders:\n{offenders}"
    )


def test_provider_v1_crud_routes_are_gone_from_hurl():
    """After H3, no Hurl test references the dead provider CRUD/config/default v1 routes."""
    forbidden_provider_v1 = (
        "/api/v1/llm_configuration",
        "/api/v1/llm_providers",
        "/api/v1/available_models",
        "/api/v1/default_models",
    )
    offenders: dict[str, set[str]] = {}
    for hurl in HURL_ROOT.rglob("*.hurl"):
        text = hurl.read_text()
        seen = {p for p in forbidden_provider_v1 if p in text}
        if seen:
            offenders[str(hurl.relative_to(REPO_ROOT))] = seen
    assert not offenders, (
        "Provider v1 CRUD/config/default routes have been removed from main; "
        f"Hurl tests must not reference them. Offenders:\n{offenders}"
    )
