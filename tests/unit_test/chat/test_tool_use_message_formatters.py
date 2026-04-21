from aperag.agent.tool_use_message_formatters import ToolResultFormatter
from aperag.schema.view_models import WebSearchMeta, WebSearchResponse


def test_web_search_formatter_distinguishes_unavailable_from_empty():
    formatter = ToolResultFormatter(language="en-US")
    response = WebSearchResponse(
        query="latest chip news",
        results=[],
        total_results=0,
        search_time=0.2,
        meta=WebSearchMeta(
            search_status="unavailable",
            provider_used=["jina", "duckduckgo"],
            backend_used=["duckduckgo:auto", "duckduckgo:html"],
            fallback_used=True,
            error_code="duckduckgo_unavailable",
        ),
    )

    formatted = formatter._format_web_search(response)

    assert "Web search was unavailable for this step" in formatted
    assert "duckduckgo_unavailable" in formatted
    assert "duckduckgo:auto" in formatted


def test_web_search_formatter_reports_disabled_state():
    formatter = ToolResultFormatter(language="en-US")
    response = WebSearchResponse(
        query="latest chip news",
        results=[],
        total_results=0,
        search_time=0.2,
        meta=WebSearchMeta(
            search_status="disabled",
            provider_used=[],
            backend_used=[],
            fallback_used=False,
            error_code=None,
        ),
    )

    formatted = formatter._format_web_search(response)

    assert "Web search was disabled for this turn" in formatted
