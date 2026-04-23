from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_server_typed_client_uses_api_root_not_legacy_v1_base_path():
    server_client = REPO_ROOT / "web/src/lib/api/typed/server.ts"
    env_template = REPO_ROOT / "web/deploy/env.local.template"
    configmap = REPO_ROOT / "web/deploy/yaml/configmap.yaml"

    server_source = server_client.read_text()

    assert "API_SERVER_BASE_PATH" in env_template.read_text()
    assert "API_SERVER_BASE_PATH" in configmap.read_text()
    assert "API_SERVER_BASE_PATH" not in server_source
    assert "process.env.API_SERVER_ENDPOINT || 'http://localhost:8000'" in server_source
    assert "if (!response.ok)" in server_source
