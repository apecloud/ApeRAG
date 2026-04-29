from fastapi import FastAPI
from fastapi.testclient import TestClient

from aperag.server import health


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(health.router, prefix="/health")
    return TestClient(app)


def test_legacy_health_endpoint_keeps_existing_payload():
    response = _client().get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "service": "aperag-api"}


def test_legacy_health_endpoint_does_not_redirect():
    response = _client().get("/health", follow_redirects=False)

    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "service": "aperag-api"}


def test_live_and_ready_are_lightweight():
    client = _client()

    assert client.get("/health/live").json() == {"status": "live", "service": "aperag-api"}
    assert client.get("/health/ready").json() == {"status": "ready", "service": "aperag-api"}


def test_diagnostics_is_hidden_from_openapi():
    schema = _client().get("/openapi.json").json()

    assert "/health/diagnostics" not in schema["paths"]


def test_diagnostics_requires_configured_token(monkeypatch):
    monkeypatch.delenv(health.DIAGNOSTICS_TOKEN_ENV, raising=False)

    response = _client().get("/health/diagnostics")

    assert response.status_code == 503
    assert response.json()["detail"] == f"{health.DIAGNOSTICS_TOKEN_ENV} is not configured"


def test_diagnostics_requires_matching_token(monkeypatch):
    monkeypatch.setenv(health.DIAGNOSTICS_TOKEN_ENV, "secret")

    response = _client().get("/health/diagnostics", headers={"X-Internal-Token": "wrong"})

    assert response.status_code == 401


def test_diagnostics_uses_isolated_probe_when_authorized(monkeypatch):
    monkeypatch.setenv(health.DIAGNOSTICS_TOKEN_ENV, "secret")

    async def fake_probe():
        return {"ok": True, "elapsed_ms": 1.0}

    monkeypatch.setattr(health, "_run_database_probe", fake_probe)

    response = _client().get("/health/diagnostics", headers={"X-Internal-Token": "secret"})

    assert response.status_code == 200
    assert response.json()["checks"]["database"] == {"ok": True, "elapsed_ms": 1.0}
