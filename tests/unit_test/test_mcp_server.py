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


def test_list_collections_docstring_explains_empty_and_user_step_language():
    doc = mcp_server.list_collections.__doc__

    assert "What an empty result means:" in doc
    assert "It does not automatically mean the system is broken." in doc
    assert "Checking which knowledge bases are available." in doc


def test_search_collection_docstring_explains_step_and_failure_meaning():
    doc = mcp_server.search_collection.__doc__

    assert "What failure may mean:" in doc
    assert "Reviewed results from the selected knowledge base." in doc


def test_web_tools_docstrings_match_user_visible_step_language():
    assert "Searching the web for current or missing information." in mcp_server.web_search.__doc__
    assert "Reading content from web sources." in mcp_server.web_read.__doc__
