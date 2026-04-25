import re
from pathlib import Path

WORKFLOW_PATH = Path(".github/workflows/e2e-http-smoke.yml")


def _job_section(workflow_text: str, job_name: str) -> str:
    match = re.search(
        rf"^  {re.escape(job_name)}:\n(?P<body>.*?)(?=^  [A-Za-z0-9_-]+:\n|\Z)",
        workflow_text,
        flags=re.MULTILINE | re.DOTALL,
    )
    assert match is not None, f"Missing workflow job: {job_name}"
    return match.group("body")


def test_e2e_http_workflow_splits_binding_smoke_from_provider_suite():
    workflow_text = WORKFLOW_PATH.read_text()

    smoke_job = _job_section(workflow_text, "e2e-http-smoke")
    provider_job = _job_section(workflow_text, "e2e-http-provider")

    assert "run_smoke.sh" in smoke_job
    assert "run_full.sh" not in smoke_job

    assert "provider-preflight" in workflow_text
    assert "needs.provider-preflight.outputs.available == 'true'" in provider_job
    assert "run_full.sh" in provider_job
