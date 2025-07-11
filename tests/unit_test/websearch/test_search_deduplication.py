"""
Unit tests for web search result deduplication.

Tests the deduplication logic in web.py including:
- URL deduplication with priority rules
- LLM.txt results placement
- Regular search priority for conflicts
- Edge cases handling
"""

from datetime import datetime

from aperag.schema.view_models import WebSearchResultItem
from aperag.views.web import deduplicate_search_results


class TestSearchDeduplication:
    """Test cases for search result deduplication"""

    def create_result(self, rank: int, url: str, title: str, domain: str, snippet: str = "Test snippet") -> WebSearchResultItem:
        """Helper method to create WebSearchResultItem for testing"""
        return WebSearchResultItem(
            rank=rank,
            title=title,
            url=url,
            snippet=snippet,
            domain=domain,
            timestamp=datetime.now()
        )

    def test_no_duplicates(self):
        """Test deduplication when there are no URL conflicts"""
        llm_txt_results = [
            self.create_result(1, "https://example.com/llms.txt", "LLM.txt File", "example.com"),
            self.create_result(2, "https://docs.com/llms.txt", "Docs LLM.txt", "docs.com"),
        ]
        
        regular_results = [
            self.create_result(1, "https://different.com/page", "Different Page", "different.com"),
            self.create_result(2, "https://another.com/content", "Another Content", "another.com"),
        ]
        
        result = deduplicate_search_results(llm_txt_results, regular_results)
        
        # Should have all 4 results, LLM.txt first
        assert len(result) == 4
        assert result[0].url == "https://example.com/llms.txt"
        assert result[1].url == "https://docs.com/llms.txt"
        assert result[2].url == "https://different.com/page"
        assert result[3].url == "https://another.com/content"

    def test_url_conflicts_priority_to_regular(self):
        """Test that regular search results take priority in URL conflicts"""
        llm_txt_results = [
            self.create_result(1, "https://example.com/page", "LLM.txt Page Title", "example.com", "Brief LLM snippet"),
            self.create_result(2, "https://unique.com/llms.txt", "Unique LLM.txt", "unique.com"),
        ]
        
        regular_results = [
            self.create_result(1, "https://example.com/page", "Detailed Page Title", "example.com", "Comprehensive description with more details"),
            self.create_result(2, "https://other.com/content", "Other Content", "other.com"),
        ]
        
        result = deduplicate_search_results(llm_txt_results, regular_results)
        
        # Should have 3 results total (one duplicate removed)
        assert len(result) == 3
        
        # First result should be the regular search result (better description)
        assert result[0].url == "https://example.com/page"
        assert result[0].title == "Detailed Page Title"
        assert result[0].snippet == "Comprehensive description with more details"
        
        # LLM.txt unique result should be second
        assert result[1].url == "https://unique.com/llms.txt"
        
        # Regular unique result should be third
        assert result[2].url == "https://other.com/content"

    def test_only_llm_txt_results(self):
        """Test when only LLM.txt results are provided"""
        llm_txt_results = [
            self.create_result(1, "https://example.com/llms.txt", "LLM.txt File", "example.com"),
            self.create_result(2, "https://docs.com/llms.txt", "Docs LLM.txt", "docs.com"),
        ]
        
        result = deduplicate_search_results(llm_txt_results, [])
        
        assert len(result) == 2
        assert result[0].url == "https://example.com/llms.txt"
        assert result[1].url == "https://docs.com/llms.txt"

    def test_only_regular_results(self):
        """Test when only regular search results are provided"""
        regular_results = [
            self.create_result(1, "https://example.com/page", "Example Page", "example.com"),
            self.create_result(2, "https://docs.com/content", "Docs Content", "docs.com"),
        ]
        
        result = deduplicate_search_results([], regular_results)
        
        assert len(result) == 2
        assert result[0].url == "https://example.com/page"
        assert result[1].url == "https://docs.com/content"

    def test_both_empty_lists(self):
        """Test when both result lists are empty"""
        result = deduplicate_search_results([], [])
        assert len(result) == 0

    def test_multiple_conflicts(self):
        """Test multiple URL conflicts"""
        llm_txt_results = [
            self.create_result(1, "https://example.com/page1", "LLM Page 1", "example.com", "LLM snippet 1"),
            self.create_result(2, "https://example.com/page2", "LLM Page 2", "example.com", "LLM snippet 2"),
            self.create_result(3, "https://unique-llm.com/content", "Unique LLM", "unique-llm.com"),
        ]
        
        regular_results = [
            self.create_result(1, "https://example.com/page1", "Regular Page 1", "example.com", "Detailed snippet 1"),
            self.create_result(2, "https://example.com/page2", "Regular Page 2", "example.com", "Detailed snippet 2"),
            self.create_result(3, "https://unique-regular.com/content", "Unique Regular", "unique-regular.com"),
        ]
        
        result = deduplicate_search_results(llm_txt_results, regular_results)
        
        # Should have 4 results (2 conflicts resolved, 2 unique)
        assert len(result) == 4
        
        # First two should be regular results (conflict resolution)
        assert result[0].url == "https://example.com/page1"
        assert result[0].title == "Regular Page 1"
        assert result[1].url == "https://example.com/page2"
        assert result[1].title == "Regular Page 2"
        
        # Third should be unique LLM.txt result
        assert result[2].url == "https://unique-llm.com/content"
        
        # Fourth should be unique regular result
        assert result[3].url == "https://unique-regular.com/content"

    def test_order_preservation_within_categories(self):
        """Test that relative order is preserved within each category"""
        llm_txt_results = [
            self.create_result(1, "https://llm1.com/page", "LLM 1", "llm1.com"),
            self.create_result(2, "https://llm2.com/page", "LLM 2", "llm2.com"),
            self.create_result(3, "https://llm3.com/page", "LLM 3", "llm3.com"),
        ]
        
        regular_results = [
            self.create_result(1, "https://reg1.com/page", "Regular 1", "reg1.com"),
            self.create_result(2, "https://reg2.com/page", "Regular 2", "reg2.com"),
            self.create_result(3, "https://reg3.com/page", "Regular 3", "reg3.com"),
        ]
        
        result = deduplicate_search_results(llm_txt_results, regular_results)
        
        # Should maintain order: all LLM.txt first, then all regular
        assert len(result) == 6
        assert result[0].title == "LLM 1"
        assert result[1].title == "LLM 2"
        assert result[2].title == "LLM 3"
        assert result[3].title == "Regular 1"
        assert result[4].title == "Regular 2"
        assert result[5].title == "Regular 3"

    def test_identical_urls_different_casing(self):
        """Test URL conflicts with different casing (should be treated as different URLs)"""
        llm_txt_results = [
            self.create_result(1, "https://Example.com/Page", "LLM Page", "Example.com"),
        ]
        
        regular_results = [
            self.create_result(1, "https://example.com/page", "Regular Page", "example.com"),
        ]
        
        result = deduplicate_search_results(llm_txt_results, regular_results)
        
        # URLs are case-sensitive, so should have both results
        assert len(result) == 2
        assert result[0].url == "https://Example.com/Page"
        assert result[1].url == "https://example.com/page" 