import re
import shlex
from pathlib import Path

# After PR splitting the suite by deployment shape, the per-shape logic
# (smoke + provider-preflight + provider-aware) lives in the reusable
# `e2e-http-shape.yml` and per-shape callers (e2e-http-lite.yml etc.)
# bind it via workflow_call.
SHAPE_WORKFLOW_PATH = Path(".github/workflows/e2e-http-shape.yml")
SHAPES_DIR = Path("tests/e2e_http/shapes")
EXPECTED_SHAPE_BACKENDS = {
    "lite": ("pgvector", "postgresql"),
    "qdrant-postgres": ("qdrant", "postgresql"),
    "qdrant-neo4j": ("qdrant", "neo4j"),
    "qdrant-nebula": ("qdrant", "nebula"),
    "pgvector-neo4j": ("pgvector", "neo4j"),
    "pgvector-nebula": ("pgvector", "nebula"),
}
SHAPE_CALLER_PATHS = [Path(f".github/workflows/e2e-http-{shape}.yml") for shape in EXPECTED_SHAPE_BACKENDS]
EXTENDED_SHAPES = {"qdrant-postgres", "pgvector-neo4j", "pgvector-nebula"}


def _job_section(workflow_text: str, job_name: str) -> str:
    match = re.search(
        rf"^  {re.escape(job_name)}:\n(?P<body>.*?)(?=^  [A-Za-z0-9_-]+:\n|\Z)",
        workflow_text,
        flags=re.MULTILINE | re.DOTALL,
    )
    assert match is not None, f"Missing workflow job: {job_name}"
    return match.group("body")


def _shape_env(shape: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in (SHAPES_DIR / f"{shape}.env").read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        key, value = stripped.split("=", 1)
        result[key] = shlex.split(value)[0]
    return result


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


def test_e2e_http_shape_files_cover_full_vector_graph_matrix():
    actual_shapes = {path.stem for path in SHAPES_DIR.glob("*.env")}
    assert actual_shapes == set(EXPECTED_SHAPE_BACKENDS)

    for shape, (vector_backend, graph_backend) in EXPECTED_SHAPE_BACKENDS.items():
        env = _shape_env(shape)
        assert env["SHAPE_VECTOR_DB_TYPE"] == vector_backend
        assert env["SHAPE_GRAPH_DB_TYPE"] == graph_backend

        services = env["SHAPE_COMPOSE_SERVICES"].split()
        profiles = env["SHAPE_COMPOSE_PROFILES"].split()

        if vector_backend == "qdrant":
            assert "qdrant" in services, f"{shape} must start qdrant"
        else:
            assert "qdrant" not in services, f"{shape} must not start qdrant in pgvector mode"

        if graph_backend == "neo4j":
            assert profiles == ["--profile", "neo4j"]
        elif graph_backend == "nebula":
            assert profiles == ["--profile", "nebula"]
        else:
            assert profiles == []


def test_extended_shape_callers_are_backend_surface_targeted():
    for shape in EXTENDED_SHAPES:
        text = Path(f".github/workflows/e2e-http-{shape}.yml").read_text()
        assert "workflow_dispatch:" in text
        assert "paths:" in text, f"{shape} must not run on every PR"
        assert "tests/e2e_http/**" in text
        assert "envs/**" in text
        assert "aperag/config.py" in text
        assert "aperag/vectorstore/**" in text
        assert "aperag/indexing/**" in text
        assert "aperag/graph_curation/**" in text
        assert "aperag/domains/knowledge_graph/**" in text
