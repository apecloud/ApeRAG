"""
DuckDuckGo provider resilience tests (issue #1722).

The DuckDuckGo fallback used to die with ``DuckDuckGoSearchException("<url>
return None")`` on a single empty backend response, which surfaced to the user
as ``Both JINA and DuckDuckGo search failed`` even though JINA had never been
attempted. These tests pin down the retry / error behaviour.
"""

from unittest.mock import patch

import pytest

from aperag.websearch.search.providers import duckduckgo_search_provider
from aperag.websearch.search.providers.duckduckgo_search_provider import (
    DuckDuckGoProvider,
    DuckDuckGoSearchError,
)

MODULE = "aperag.websearch.search.providers.duckduckgo_search_provider"
BACKEND_ERROR = RuntimeError("https://www.bing.com/search?q=Aperag+AI+model information return None")


def _item(href="https://github.com/apecloud/ApeRAG"):
    return {"href": href, "title": "ApeRAG", "body": "ApeRAG knowledge base"}


class _FakeDDGS:
    """Replays scripted outcomes: a list of results, or an exception to raise."""

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def text(self, *args, **kwargs):
        self.calls += 1
        outcome = self.outcomes.pop(0) if self.outcomes else []
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _patch(outcomes):
    fake = _FakeDDGS(outcomes)
    patcher = patch(f"{MODULE}.DDGS", return_value=fake)
    return fake, patcher


class TestDuckDuckGoRetry:
    @pytest.mark.asyncio
    async def test_recovers_after_transient_backend_error(self):
        """A first failing attempt must not fail the whole search."""
        fake, patcher = _patch([BACKEND_ERROR, [_item()]])

        with patcher, patch(f"{MODULE}.time.sleep") as mock_sleep:
            results = await DuckDuckGoProvider().search("Aperag AI model")

        assert [r.url for r in results] == ["https://github.com/apecloud/ApeRAG"]
        assert fake.calls == 2
        assert mock_sleep.call_count == 1

    @pytest.mark.asyncio
    async def test_raises_provider_error_after_all_attempts_fail(self):
        provider = DuckDuckGoProvider()
        fake, patcher = _patch([BACKEND_ERROR] * provider.max_attempts)

        with patcher, patch(f"{MODULE}.time.sleep"):
            with pytest.raises(DuckDuckGoSearchError) as excinfo:
                await provider.search("Aperag AI model")

        assert fake.calls == provider.max_attempts
        assert "return None" in str(excinfo.value)
        # The caller must be able to see it was the DuckDuckGo side that broke.
        assert "DuckDuckGo" in str(excinfo.value)

    @pytest.mark.asyncio
    async def test_empty_result_is_not_retried(self):
        """No results is a valid answer, not a backend failure."""
        fake, patcher = _patch([[]])

        with patcher, patch(f"{MODULE}.time.sleep") as mock_sleep:
            results = await DuckDuckGoProvider().search("Aperag AI model")

        assert results == []
        assert fake.calls == 1
        mock_sleep.assert_not_called()


class TestDuckDuckGoBackendPackage:
    def test_prefers_maintained_ddgs_package(self):
        """``duckduckgo-search`` 8.x pins a single bing backend (#1722)."""
        pytest.importorskip("ddgs")
        assert duckduckgo_search_provider.DDGS.__module__.split(".")[0] == "ddgs"
