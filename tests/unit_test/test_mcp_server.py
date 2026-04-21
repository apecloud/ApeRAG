import pytest

from aperag.mcp import server as mcp_server


def test_get_api_key_prefers_authorization_header(monkeypatch):
    monkeypatch.delenv("APERAG_API_KEY", raising=False)
    monkeypatch.setattr(
        mcp_server,
        "get_http_headers",
        lambda include_all=False, include=None: {"authorization": "Bearer header-token"},
    )

    assert mcp_server.get_api_key() == "header-token"


def test_get_api_key_falls_back_to_environment(monkeypatch):
    monkeypatch.setenv("APERAG_API_KEY", "env-token")
    monkeypatch.setattr(mcp_server, "get_http_headers", lambda include_all=False, include=None: {})

    assert mcp_server.get_api_key() == "env-token"


def test_get_api_key_raises_when_header_and_environment_missing(monkeypatch):
    monkeypatch.delenv("APERAG_API_KEY", raising=False)
    monkeypatch.setattr(mcp_server, "get_http_headers", lambda include_all=False, include=None: {})

    with pytest.raises(ValueError, match="API key not found"):
        mcp_server.get_api_key()
