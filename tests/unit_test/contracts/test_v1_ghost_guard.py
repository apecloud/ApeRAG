"""Guard against regrowth of internal ``/api/v1`` references in e2e Hurl tests.

The end-state allowlist (canonical D7-1 / msg=8b6b4bc3) is:

- ``/api/v1/embeddings`` — OpenAI-compatible public endpoint, permanent ``/api/v1`` mount

Phase 8 task #44 (H3) cleaned the provider-aware Hurl suite to use
``/api/v2/providers/*``. The G* hard-cut series (#1 G1 export, #2 G2
settings, #3 G3 prompts, #50 G4a audit-logs, #51 G4b apikeys, #52 G4c
marketplace, G4d chat ops, #63 G5a, #65 G5c, and #66 G5b) retired the
temporary ``TRANSITIONAL_V1_PREFIXES`` ledger. No internal ``/api/v1``
Hurl references may remain after #66; only the OpenAI-compatible public
allowlist below is permanent.

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
    }
)

# Transitional prefixes — after #66 there are no temporary internal v1
# prefixes left. Any new item here must come with a new canonical decision
# explaining why it is not migrated to ``/api/v2`` in the same PR.
TRANSITIONAL_V1_PREFIXES: frozenset[str] = frozenset()

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
