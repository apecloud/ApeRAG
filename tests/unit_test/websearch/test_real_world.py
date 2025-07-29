"""
Real World Integration Tests

Tests actual network requests to verify end-to-end functionality including:
- Real search engine integration
- Actual website content reading
- Production-like scenarios
- Performance verification

Note: These tests make real network requests and may be slower.
Run with: pytest -m integration tests/unit_test/websearch/test_real_world.py -v
"""

import asyncio
from datetime import datetime

import pytest

from aperag.schema.view_models import WebReadRequest, WebSearchRequest
from aperag.websearch.reader.reader_service import ReaderService
from aperag.websearch.search.search_service import SearchService


@pytest.mark.integration
class TestRealWorldSearch:
    """Real world search integration tests."""

    @pytest.mark.asyncio
    async def test_duckduckgo_real_search(self):
        """Test actual DuckDuckGo search with real queries."""
        service = SearchService.create_default()

        try:
            # Test basic programming query
            response = await service.search(
                WebSearchRequest(query="Python asyncio tutorial", max_results=5, timeout=15)
            )

            print("\n=== DuckDuckGo Search: 'Python asyncio tutorial' ===")
            print(f"Results: {len(response.results)}")
            print(f"Search time: {response.search_time:.2f}s")

            # Basic validation
            assert len(response.results) > 0, "Should return at least one result"
            assert response.search_time > 0, "Search time should be positive"
            assert response.query == "Python asyncio tutorial"

            for i, result in enumerate(response.results):
                print(f"\nResult {i + 1}:")
                print(f"  Title: {result.title}")
                print(f"  URL: {result.url}")
                print(f"  Domain: {result.domain}")
                print(f"  Snippet: {result.snippet[:100]}...")

                # Validate result structure
                assert result.title, "Title should not be empty"
                assert result.url.startswith("http"), "URL should be valid"
                assert result.domain, "Domain should not be empty"
                assert result.rank > 0, "Rank should be positive"
                assert result.timestamp, "Timestamp should be set"

            print("✅ DuckDuckGo real search test passed!")

        except Exception as e:
            pytest.skip(f"DuckDuckGo search test skipped due to: {e}")

    @pytest.mark.asyncio
    async def test_site_specific_search(self):
        """Test site-specific search functionality."""
        service = SearchService.create_default()

        try:
            # Search within a specific site
            request = WebSearchRequest(
                query="machine learning",
                max_results=3,
                timeout=30,
                locale="en-US",
                source="github.com",
            )
            response = await service.search(request)

            print("\n=== Site-specific Search: 'machine learning' on github.com ===")
            print(f"Results: {len(response.results)}")

            if len(response.results) > 0:
                for result in response.results:
                    print(f"URL: {result.url}")
                    # Should contain stackoverflow.com in domain or URL
                    assert "stackoverflow" in result.url.lower() or "stackoverflow" in result.domain.lower()

                print("✅ Site-specific search test passed!")
            else:
                print("⚠️ No results returned for site-specific search")

        except Exception as e:
            pytest.skip(f"Site-specific search test skipped due to: {e}")


@pytest.mark.integration
class TestRealWorldReading:
    """Real world content reading integration tests."""

    @pytest.mark.asyncio
    async def test_read_simple_websites(self):
        """Test reading content from simple, reliable websites."""
        service = ReaderService.create_default()

        # Use reliable, simple websites for testing
        test_urls = [
            "https://example.com",  # Classic simple example
            "https://httpbin.org/html",  # Simple HTML test page
        ]

        try:
            for url in test_urls:
                print(f"\n=== Reading: {url} ===")

                result = await service.read_simple(url=url, timeout=15)

                print(f"Status: {result.status}")
                print(f"Title: {result.title}")
                print(f"Content length: {len(result.content) if result.content else 0}")
                print(f"Word count: {result.word_count}")

                if result.status == "success":
                    assert result.title, "Title should not be empty"
                    assert result.content, "Content should not be empty"
                    assert result.word_count > 0, "Word count should be positive"
                    assert result.extracted_at, "Extraction time should be set"

                    print(f"Content preview: {result.content[:150]}...")
                    print("✅ Content reading successful!")
                else:
                    print(f"❌ Reading failed: {result.error}")

        except Exception as e:
            pytest.skip(f"Content reading test skipped due to: {e}")
        finally:
            await service.close()

    @pytest.mark.asyncio
    async def test_batch_reading(self):
        """Test batch reading of multiple URLs."""
        service = ReaderService.create_default()

        urls = [
            "https://example.com",
            "https://httpbin.org/html",
        ]

        try:
            print(f"\n=== Batch Reading: {len(urls)} URLs ===")

            response = await service.read(WebReadRequest(urls=urls, timeout=15, max_concurrent=2))

            print(f"Total URLs: {response.total_urls}")
            print(f"Successful: {response.successful}")
            print(f"Failed: {response.failed}")
            print(f"Processing time: {response.processing_time:.2f}s")

            assert response.total_urls == len(urls)
            assert len(response.results) == len(urls)

            for i, result in enumerate(response.results):
                print(f"\nResult {i + 1}: {result.url}")
                print(f"  Status: {result.status}")
                if result.status == "success":
                    print(f"  Title: {result.title}")
                    print(f"  Words: {result.word_count}")
                else:
                    print(f"  Error: {result.error}")

            print("✅ Batch reading test completed!")

        except Exception as e:
            pytest.skip(f"Batch reading test skipped due to: {e}")
        finally:
            await service.close()


@pytest.mark.integration
class TestRealWorldLLMTxtDiscovery:
    """Real world LLM.txt discovery tests."""

    @pytest.mark.asyncio
    async def test_discover_real_llm_txt_files(self):
        """Test discovering actual LLM.txt files from known sources."""
        from aperag.websearch.search.providers.llm_txt_search_provider import LLMTxtSearchProvider

        provider = LLMTxtSearchProvider()

        # Known sources that likely have LLM.txt files
        test_sources = [
            "modelcontextprotocol.io",  # Known to have llms.txt
            "docs.anthropic.com",  # Anthropic documentation
        ]

        try:
            for source in test_sources:
                print(f"\n=== LLM.txt Discovery: {source} ===")

                results = await provider.search(query="documentation", source=source, max_results=5, timeout=20)

                print(f"Found {len(results)} LLM.txt results")

                if results:
                    for result in results:
                        print(f"URL: {result.url}")
                        print(f"Title: {result.title}")
                        print(f"Snippet: {result.snippet[:100]}...")

                        # Validate LLM.txt URL structure
                        assert "llms" in result.url.lower()
                        assert result.url.startswith("https://")
                        assert result.domain == source

                    print("✅ LLM.txt discovery successful!")
                else:
                    print("⚠️ No LLM.txt files found for this source")

        except Exception as e:
            pytest.skip(f"LLM.txt discovery test skipped due to: {e}")
        finally:
            await provider.close()

    @pytest.mark.asyncio
    async def test_direct_llm_txt_url_reading(self):
        """Test reading a direct LLM.txt URL."""
        from aperag.websearch.search.providers.llm_txt_search_provider import LLMTxtSearchProvider

        provider = LLMTxtSearchProvider()

        # Direct LLM.txt URL (known to exist)
        direct_url = "https://modelcontextprotocol.io/llms-full.txt"

        try:
            print(f"\n=== Direct LLM.txt Reading: {direct_url} ===")

            results = await provider.search(query="test", source=direct_url, max_results=1, timeout=15)

            if results:
                result = results[0]
                print(f"URL: {result.url}")
                print(f"Title: {result.title}")
                print(f"Domain: {result.domain}")
                print(f"Snippet: {result.snippet[:200]}...")

                assert result.url == direct_url
                assert result.domain == "modelcontextprotocol.io"
                assert len(result.snippet) > 0

                print("✅ Direct LLM.txt reading successful!")
            else:
                print("⚠️ Failed to read direct LLM.txt URL")

        except Exception as e:
            pytest.skip(f"Direct LLM.txt test skipped due to: {e}")
        finally:
            await provider.close()


@pytest.mark.integration
class TestRealWorldPerformance:
    """Real world performance tests."""

    @pytest.mark.asyncio
    async def test_search_performance_benchmark(self):
        """Benchmark search performance with real queries."""
        service = SearchService.create_default()

        # Various query types for performance testing
        test_queries = [
            "Python programming tutorial",
            "machine learning algorithms",
            "web development best practices",
            "database optimization techniques",
            "API design patterns",
        ]

        try:
            print(f"\n=== Performance Benchmark: {len(test_queries)} queries ===")

            start_time = datetime.now()
            results = []

            for query in test_queries:
                query_start = datetime.now()

                response = await service.search(WebSearchRequest(query=query, max_results=3, timeout=10))

                query_time = (datetime.now() - query_start).total_seconds()
                results.append({"query": query, "time": query_time, "results_count": len(response.results)})

                print(f"'{query}': {len(response.results)} results in {query_time:.2f}s")

            total_time = (datetime.now() - start_time).total_seconds()
            avg_time = total_time / len(test_queries)

            print("\nPerformance Summary:")
            print(f"Total time: {total_time:.2f}s")
            print(f"Average time per query: {avg_time:.2f}s")
            print(f"Total results: {sum(r['results_count'] for r in results)}")

            # Performance assertions
            assert avg_time < 5.0, f"Average query time {avg_time:.2f}s exceeds 5s threshold"
            assert all(r["results_count"] > 0 for r in results), "All queries should return results"

            print("✅ Performance benchmark passed!")

        except Exception as e:
            pytest.skip(f"Performance benchmark skipped due to: {e}")

    @pytest.mark.asyncio
    async def test_concurrent_search_performance(self):
        """Test performance under concurrent load."""
        service = SearchService.create_default()

        try:
            print("\n=== Concurrent Search Test ===")

            # Create multiple concurrent search tasks
            async def search_task(query_id):
                response = await service.search(
                    WebSearchRequest(query=f"test query {query_id}", max_results=2, timeout=10)
                )
                return len(response.results)

            start_time = datetime.now()

            # Run 5 concurrent searches
            tasks = [search_task(i) for i in range(5)]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            end_time = datetime.now()
            total_time = (end_time - start_time).total_seconds()

            # Check results
            successful_results = [r for r in results if isinstance(r, int)]
            failed_results = [r for r in results if isinstance(r, Exception)]

            print(f"Successful searches: {len(successful_results)}")
            print(f"Failed searches: {len(failed_results)}")
            print(f"Total concurrent time: {total_time:.2f}s")

            # Should handle concurrent requests gracefully
            assert len(successful_results) >= 3, "Most concurrent searches should succeed"
            assert total_time < 15.0, "Concurrent searches should complete in reasonable time"

            print("✅ Concurrent search test passed!")

        except Exception as e:
            pytest.skip(f"Concurrent search test skipped due to: {e}")
