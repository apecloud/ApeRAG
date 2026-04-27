import re
from pathlib import Path

# After PR splitting the suite by deployment shape, the per-shape logic
# (smoke + provider-preflight + provider-aware) lives in the reusable
# `e2e-http-shape.yml` and per-shape callers (e2e-http-lite.yml etc.)
# bind it via workflow_call.
SHAPE_WORKFLOW_PATH = Path(".github/workflows/e2e-http-shape.yml")
SHAPE_CALLER_PATHS = [
    Path(".github/workflows/e2e-http-lite.yml"),
    Path(".github/workflows/e2e-http-qdrant-neo4j.yml"),
    Path(".github/workflows/e2e-http-qdrant-nebula.yml"),
]


def _job_section(workflow_text: str, job_name: str) -> str:
    match = re.search(
        rf"^  {re.escape(job_name)}:\n(?P<body>.*?)(?=^  [A-Za-z0-9_-]+:\n|\Z)",
        workflow_text,
        flags=re.MULTILINE | re.DOTALL,
    )
    assert match is not None, f"Missing workflow job: {job_name}"
    return match.group("body")


def test_e2e_http_shape_workflow_splits_smoke_from_provider_suite():
    workflow_text = SHAPE_WORKFLOW_PATH.read_text()

    smoke_job = _job_section(workflow_text, "e2e-http-smoke")
    provider_job = _job_section(workflow_text, "e2e-http-provider")

    assert "run_smoke.sh" in smoke_job
    assert "run_full.sh" not in smoke_job

    assert "provider-preflight" in workflow_text
    assert "needs.provider-preflight.outputs.available == 'true'" in provider_job
    assert "run_full.sh" in provider_job


def test_each_shape_caller_invokes_shape_workflow_with_a_shape_input():
    for caller in SHAPE_CALLER_PATHS:
        text = caller.read_text()
        assert "uses: ./.github/workflows/e2e-http-shape.yml" in text, f"{caller} must invoke the shared shape workflow"
        assert re.search(r"^\s+shape:\s+\S+", text, flags=re.MULTILINE), f"{caller} must pass a `shape` input"
