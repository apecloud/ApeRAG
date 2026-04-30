from __future__ import annotations

import re
from pathlib import Path

from aperag.domains.knowledge_graph.db.models import GraphCurationSuggestionStatus

ROOT = Path(__file__).resolve().parents[3]


def test_graph_curation_suggestion_status_values_include_async_apply_states():
    values = {status.value for status in GraphCurationSuggestionStatus}

    assert {
        "PENDING",
        "APPLY_PENDING",
        "APPLYING",
        "APPLIED",
        "APPLY_FAILED",
        "ACCEPTED",
        "REJECTED",
        "DISMISSED",
        "EXPIRED",
        "SUPERSEDED",
    } <= values


def test_accepted_status_write_is_legacy_service_only():
    """New async apply code must not reuse legacy ACCEPTED as a pending-apply state."""

    write_pattern = re.compile(r"(?:suggestion\.status|status)\s*=\s*GraphCurationSuggestionStatus\.ACCEPTED")
    unexpected: list[str] = []

    for path in (ROOT / "aperag").rglob("*.py"):
        rel = path.relative_to(ROOT).as_posix()
        if rel.startswith("aperag/migration/"):
            continue
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not write_pattern.search(line):
                continue
            if rel == "aperag/graph_curation/service.py" and "suggestion.status" in line:
                continue
            unexpected.append(f"{rel}:{lineno}: {line.strip()}")

    assert unexpected == []
