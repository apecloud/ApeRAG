"""
LLM.txt Parsing Tests

Tests the URL parsing and content extraction from LLM.txt files
using real-world data formats.
"""

import pytest

from aperag.websearch.search.providers.llm_txt_search_provider import LLMTxtSearchProvider


class TestLLMTxtParsing:
    """Test LLM.txt content parsing with real-world data."""

    @pytest.fixture
    def provider(self):
        """Create provider instance for testing."""
        return LLMTxtSearchProvider()

    @pytest.fixture
    def sample_llm_txt_content(self):
        """Sample LLM.txt content from Model Context Protocol."""
        return """# Model Context Protocol

## Docs

- [Example Clients](https://modelcontextprotocol.io/clients.md): A list of applications that support MCP integrations
- [Contributing](https://modelcontextprotocol.io/development/contributing.md): How to participate in Model Context Protocol development
- [Roadmap](https://modelcontextprotocol.io/development/roadmap.md): Our plans for evolving Model Context Protocol
- [Core architecture](https://modelcontextprotocol.io/docs/concepts/architecture.md): Understand how MCP connects clients, servers, and LLMs
- [Elicitation](https://modelcontextprotocol.io/docs/concepts/elicitation.md): Interactive information gathering in MCP
- [Prompts](https://modelcontextprotocol.io/docs/concepts/prompts.md): Create reusable prompt templates and workflows
- [Resources](https://modelcontextprotocol.io/docs/concepts/resources.md): Expose data and content from your servers to LLMs
- [Roots](https://modelcontextprotocol.io/docs/concepts/roots.md): Understanding roots in MCP
- [Sampling](https://modelcontextprotocol.io/docs/concepts/sampling.md): Let your servers request completions from LLMs
- [Tools](https://modelcontextprotocol.io/docs/concepts/tools.md): Enable LLMs to perform actions through your server
- [Transports](https://modelcontextprotocol.io/docs/concepts/transports.md): Learn about MCP's communication mechanisms
- [Debugging](https://modelcontextprotocol.io/docs/tools/debugging.md): A comprehensive guide to debugging Model Context Protocol (MCP) integrations
- [Inspector](https://modelcontextprotocol.io/docs/tools/inspector.md): In-depth guide to using the MCP Inspector for testing and debugging Model Context Protocol servers
- [Example Servers](https://modelcontextprotocol.io/examples.md): A list of example servers and implementations
- [FAQs](https://modelcontextprotocol.io/faqs.md): Explaining MCP and why it matters in simple terms
- [Introduction](https://modelcontextprotocol.io/introduction.md): Get started with the Model Context Protocol (MCP)
- [C# SDK](https://modelcontextprotocol.io/links/sdks/csharp.md)
- [Java SDK](https://modelcontextprotocol.io/links/sdks/java.md)
- [Kotlin SDK](https://modelcontextprotocol.io/links/sdks/kotlin.md)
- [Python SDK](https://modelcontextprotocol.io/links/sdks/python.md)
- [Ruby SDK](https://modelcontextprotocol.io/links/sdks/ruby.md)
- [Swift SDK](https://modelcontextprotocol.io/links/sdks/swift.md)
- [TypeScript SDK](https://modelcontextprotocol.io/links/sdks/typescript.md)
- [For Client Developers](https://modelcontextprotocol.io/quickstart/client.md): Get started building your own client that can integrate with all MCP servers.
- [For Server Developers](https://modelcontextprotocol.io/quickstart/server.md): Get started building your own server to use in Claude for Desktop and other clients.
- [For Claude Desktop Users](https://modelcontextprotocol.io/quickstart/user.md): Get started using pre-built servers in Claude for Desktop.
- [Architecture](https://modelcontextprotocol.io/specification/2025-06-18/architecture/index.md)
- [Authorization](https://modelcontextprotocol.io/specification/2025-06-18/basic/authorization.md)
- [Overview](https://modelcontextprotocol.io/specification/2025-06-18/basic/index.md)
- [Lifecycle](https://modelcontextprotocol.io/specification/2025-06-18/basic/lifecycle.md)
- [Security Best Practices](https://modelcontextprotocol.io/specification/2025-06-18/basic/security_best_practices.md)
- [Transports](https://modelcontextprotocol.io/specification/2025-06-18/basic/transports.md)
- [Cancellation](https://modelcontextprotocol.io/specification/2025-06-18/basic/utilities/cancellation.md)
- [Ping](https://modelcontextprotocol.io/specification/2025-06-18/basic/utilities/ping.md)
- [Progress](https://modelcontextprotocol.io/specification/2025-06-18/basic/utilities/progress.md)
- [Key Changes](https://modelcontextprotocol.io/specification/2025-06-18/changelog.md)
- [Elicitation](https://modelcontextprotocol.io/specification/2025-06-18/client/elicitation.md)
- [Roots](https://modelcontextprotocol.io/specification/2025-06-18/client/roots.md)
- [Sampling](https://modelcontextprotocol.io/specification/2025-06-18/client/sampling.md)
- [Specification](https://modelcontextprotocol.io/specification/2025-06-18/index.md)
- [Overview](https://modelcontextprotocol.io/specification/2025-06-18/server/index.md)
- [Prompts](https://modelcontextprotocol.io/specification/2025-06-18/server/prompts.md)
- [Resources](https://modelcontextprotocol.io/specification/2025-06-18/server/resources.md)
- [Tools](https://modelcontextprotocol.io/specification/2025-06-18/server/tools.md)
- [Completion](https://modelcontextprotocol.io/specification/2025-06-18/server/utilities/completion.md)
- [Logging](https://modelcontextprotocol.io/specification/2025-06-18/server/utilities/logging.md)
- [Pagination](https://modelcontextprotocol.io/specification/2025-06-18/server/utilities/pagination.md)
- [Versioning](https://modelcontextprotocol.io/specification/versioning.md)
- [Building MCP with LLMs](https://modelcontextprotocol.io/tutorials/building-mcp-with-llms.md): Speed up your MCP development using LLMs such as Claude!
"""

    def test_parse_real_llm_txt_content(self, provider, sample_llm_txt_content):
        """Test parsing real LLM.txt content."""
        url_data_list = provider._parse_urls_from_llm_txt(sample_llm_txt_content)

        # Should extract all valid URLs
        assert len(url_data_list) == 49  # Total number of markdown links

        # Check first few entries
        first_entry = url_data_list[0]
        assert first_entry["url"] == "https://modelcontextprotocol.io/clients.md"
        assert first_entry["title"] == "Example Clients"
        assert "A list of applications that support MCP integrations" in first_entry["line_content"]

        second_entry = url_data_list[1]
        assert second_entry["url"] == "https://modelcontextprotocol.io/development/contributing.md"
        assert second_entry["title"] == "Contributing"
        assert "How to participate in Model Context Protocol development" in second_entry["line_content"]

    def test_extract_url_and_title_from_markdown_lines(self, provider):
        """Test URL and title extraction from different markdown formats."""
        test_cases = [
            # Standard markdown with description
            {
                "line": "- [Example Clients](https://modelcontextprotocol.io/clients.md): A list of applications that support MCP integrations",
                "expected_url": "https://modelcontextprotocol.io/clients.md",
                "expected_title": "Example Clients",
            },
            # Simple markdown without description
            {
                "line": "- [C# SDK](https://modelcontextprotocol.io/links/sdks/csharp.md)",
                "expected_url": "https://modelcontextprotocol.io/links/sdks/csharp.md",
                "expected_title": "C# SDK",
            },
            # Complex description
            {
                "line": "- [For Client Developers](https://modelcontextprotocol.io/quickstart/client.md): Get started building your own client that can integrate with all MCP servers.",
                "expected_url": "https://modelcontextprotocol.io/quickstart/client.md",
                "expected_title": "For Client Developers",
            },
            # Invalid line (should return None, None)
            {"line": "## This is just a heading", "expected_url": None, "expected_title": None},
        ]

        for case in test_cases:
            url, title = provider._extract_url_and_title_from_line(case["line"])
            assert url == case["expected_url"], f"URL mismatch for line: {case['line']}"
            assert title == case["expected_title"], f"Title mismatch for line: {case['line']}"

    def test_generate_title_and_snippet_from_real_data(self, provider):
        """Test title and snippet generation from real LLM.txt line data."""
        test_data = [
            {
                "url": "https://modelcontextprotocol.io/clients.md",
                "line_content": "- [Example Clients](https://modelcontextprotocol.io/clients.md): A list of applications that support MCP integrations",
                "title": "Example Clients",
            },
            {
                "url": "https://modelcontextprotocol.io/development/contributing.md",
                "line_content": "- [Contributing](https://modelcontextprotocol.io/development/contributing.md): How to participate in Model Context Protocol development",
                "title": "Contributing",
            },
            {
                "url": "https://modelcontextprotocol.io/links/sdks/csharp.md",
                "line_content": "- [C# SDK](https://modelcontextprotocol.io/links/sdks/csharp.md)",
                "title": "C# SDK",
            },
        ]

        for data in test_data:
            title, snippet = provider._generate_title_and_snippet_from_line(data)

            # Title should be extracted from line content
            assert title == data["title"]

            # Snippet should include both title and description (if available)
            assert data["title"] in snippet
            if ": " in data["line_content"]:
                # Should include description after the colon
                description_part = data["line_content"].split(": ", 1)[1]
                assert description_part in snippet

    def test_clean_line_content_for_snippet(self, provider):
        """Test cleaning line content to create readable snippets."""
        test_cases = [
            {
                "line_content": "- [Example Clients](https://modelcontextprotocol.io/clients.md): A list of applications that support MCP integrations",
                "expected_snippet": "Example Clients: A list of applications that support MCP integrations",
            },
            {
                "line_content": "- [C# SDK](https://modelcontextprotocol.io/links/sdks/csharp.md)",
                "expected_snippet": "C# SDK",
            },
            {
                "line_content": "- [Building MCP with LLMs](https://modelcontextprotocol.io/tutorials/building-mcp-with-llms.md): Speed up your MCP development using LLMs such as Claude!",
                "expected_snippet": "Building MCP with LLMs: Speed up your MCP development using LLMs such as Claude!",
            },
            {"line_content": "", "expected_snippet": "LLM-optimized content"},
        ]

        for case in test_cases:
            snippet = provider._clean_line_content_for_snippet(case["line_content"])
            assert snippet == case["expected_snippet"]

    def test_handle_various_url_formats(self, provider):
        """Test handling of various URL formats in LLM.txt."""
        test_content = """
# Different URL formats

- [Standard](https://example.com/page.md): Description here
- [With Path](https://example.com/docs/concepts/architecture.md): Multi-level path
- [With Query](https://example.com/search?q=test): URL with query parameters
- [Special chars](https://example.com/spec-2025-06-18/index.md): URL with special characters
- Not a URL line
- [Broken link](not-a-valid-url): This should be ignored
"""

        url_data_list = provider._parse_urls_from_llm_txt(test_content)

        # Should only extract valid URLs
        assert len(url_data_list) == 4

        urls = [data["url"] for data in url_data_list]
        expected_urls = [
            "https://example.com/page.md",
            "https://example.com/docs/concepts/architecture.md",
            "https://example.com/search?q=test",
            "https://example.com/spec-2025-06-18/index.md",
        ]

        assert urls == expected_urls

    def test_performance_with_large_content(self, provider, sample_llm_txt_content):
        """Test parsing performance with larger content."""
        import time

        # Repeat the content to make it larger
        large_content = sample_llm_txt_content * 5

        start_time = time.time()
        url_data_list = provider._parse_urls_from_llm_txt(large_content)
        end_time = time.time()

        # Should complete quickly (under 1 second)
        assert end_time - start_time < 1.0

        # Should extract all URLs correctly (49 * 5 = 245)
        assert len(url_data_list) == 245
