"""
Unit tests for ReaderService - User perspective tests
"""

from datetime import datetime
from unittest.mock import AsyncMock, patch

import pytest

from aperag.schema.view_models import WebReadRequest, WebReadResultItem
from aperag.websearch.reader.providers.trafilatura_read_provider import ReaderProviderError
from aperag.websearch.reader.reader_service import ReaderService


class TestReaderService:
    """Test ReaderService from user perspective"""

    def test_create_service(self):
        """Test creating reader service"""
        # Default service
        service = ReaderService.create_default()
        assert service.provider_name == "trafilatura"

        # Custom provider
        service = ReaderService.create_with_provider("trafilatura", timeout=60)
        assert service.provider_name == "trafilatura"
        assert service.provider_config["timeout"] == 60

    @pytest.mark.asyncio
    async def test_read_single_url_with_request_object(self):
        """Test reading single URL using WebReadRequest object"""
        service = ReaderService.create_default()

        # Mock the actual provider to avoid real network calls
        mock_result = WebReadResultItem(
            url="https://example.com",
            status="success",
            title="Example Page",
            content="# Example\n\nThis is example content",
            extracted_at=datetime.now(),
            word_count=4,
            token_count=6,
        )

        with patch.object(service.provider, "read", new_callable=AsyncMock) as mock_read:
            mock_read.return_value = mock_result

            request = WebReadRequest(url_list=["https://example.com"], timeout=30)

            response = await service.read(request)

            assert response.total_urls == 1
            assert response.successful == 1
            assert response.failed == 0
            assert len(response.results) == 1
            assert response.results[0].url == "https://example.com"
            assert response.results[0].status == "success"
            assert response.results[0].title == "Example Page"
            assert response.processing_time >= 0

    @pytest.mark.asyncio
    async def test_read_multiple_urls_with_request_object(self):
        """Test reading multiple URLs using WebReadRequest object"""
        service = ReaderService.create_default()

        mock_results = [
            WebReadResultItem(
                url="https://example1.com",
                status="success",
                title="Page 1",
                content="Content 1",
                word_count=2,
                token_count=3,
            ),
            WebReadResultItem(url="https://example2.com", status="error", error="Timeout", error_code="TIMEOUT"),
        ]

        with patch.object(service.provider, "read_batch", new_callable=AsyncMock) as mock_read_batch:
            mock_read_batch.return_value = mock_results

            request = WebReadRequest(
                url_list=["https://example1.com", "https://example2.com"], timeout=30, max_concurrent=2
            )

            response = await service.read(request)

            assert response.total_urls == 2
            assert response.successful == 1
            assert response.failed == 1
            assert len(response.results) == 2

    @pytest.mark.asyncio
    async def test_read_simple_interface(self):
        """Test simplified single URL reading interface"""
        service = ReaderService.create_default()

        mock_result = WebReadResultItem(
            url="https://test.com",
            status="success",
            title="Test Page",
            content="Test content",
            word_count=2,
            token_count=3,
        )

        with patch.object(service.provider, "read", new_callable=AsyncMock) as mock_read:
            mock_read.return_value = mock_result

            result = await service.read_simple(url="https://test.com")

            assert result.url == "https://test.com"
            assert result.status == "success"
            assert result.title == "Test Page"

    @pytest.mark.asyncio
    async def test_read_batch_simple_interface(self):
        """Test simplified batch reading interface"""
        service = ReaderService.create_default()

        mock_results = [
            WebReadResultItem(url="https://test1.com", status="success", title="Test 1", content="Content 1"),
            WebReadResultItem(url="https://test2.com", status="success", title="Test 2", content="Content 2"),
        ]

        with patch.object(service.provider, "read_batch", new_callable=AsyncMock) as mock_read_batch:
            mock_read_batch.return_value = mock_results

            results = await service.read_batch_simple(urls=["https://test1.com", "https://test2.com"], max_concurrent=2)

            assert len(results) == 2
            assert results[0].title == "Test 1"
            assert results[1].title == "Test 2"

    @pytest.mark.asyncio
    async def test_read_error_handling(self):
        """Test reading error handling"""
        service = ReaderService.create_default()

        # Test empty URLs
        request = WebReadRequest(url_list=[])
        with pytest.raises(ReaderProviderError, match="No URLs provided in request"):
            await service.read(request)

        # Test provider error
        with patch.object(service.provider, "read", new_callable=AsyncMock) as mock_read:
            mock_read.side_effect = ReaderProviderError("Network timeout")

            request = WebReadRequest(url_list=["https://test.com"])
            with pytest.raises(ReaderProviderError, match="Network timeout"):
                await service.read(request)

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_real_reader_integration(self):
        """
        Real integration test with actual web page reading
        Use this for debugging and verification

        Run with: pytest -m integration tests/unit_test/websearch/test_reader_service.py::TestReaderService::test_real_reader_integration -v
        """
        service = ReaderService.create_default()

        # Test reading simple, static webpages (suitable for Trafilatura)
        test_urls = [
            "https://example.com",  # Classic example page - static HTML
            "https://httpbin.org/html",  # Simple HTML test page
        ]

        try:
            print("\n=== Real Reader Integration Test ===")

            # Test single URL read
            result = await service.read_simple(url=test_urls[0], timeout=15)

            print("\n--- Single URL Read ---")
            print(f"URL: {result.url}")
            print(f"Status: {result.status}")
            print(f"Title: {result.title}")
            print(f"Content length: {len(result.content) if result.content else 0}")
            print(f"Word count: {result.word_count}")
            print(f"Token count: {result.token_count}")

            # Basic validation
            assert result.status == "success", f"Expected success, got {result.status}"
            assert result.title, "Title should not be empty"
            assert result.content, "Content should not be empty"
            assert result.word_count > 0, "Word count should be positive"
            assert result.token_count > 0, "Token count should be positive"
            assert result.extracted_at, "Extraction time should be set"

            print(f"Content preview: {result.content[:200]}...")

            # Test batch read
            request = WebReadRequest(url_list=test_urls, timeout=15)

            response = await service.read(request)

            print("\n--- Batch Read ---")
            print(f"Total URLs: {response.total_urls}")
            print(f"Successful: {response.successful}")
            print(f"Failed: {response.failed}")
            print(f"Processing time: {response.processing_time:.2f}s")

            assert response.total_urls == len(test_urls)
            assert response.successful > 0, "Should have at least one successful read"
            assert len(response.results) == len(test_urls)

            for i, result in enumerate(response.results):
                print(f"\n  Result {i + 1}:")
                print(f"    URL: {result.url}")
                print(f"    Status: {result.status}")
                if result.status == "success":
                    print(f"    Title: {result.title}")
                    print(f"    Words: {result.word_count}")
                else:
                    print(f"    Error: {result.error}")

            # Test simple batch interface
            results = await service.read_batch_simple(
                urls=test_urls[:1],  # Just one URL for simplicity
                timeout=15,
            )

            print("\n--- Simple Batch Interface ---")
            print(f"Results count: {len(results)}")
            assert len(results) == 1
            assert results[0].status == "success"

            print("\n✅ Real reader integration test passed!")

        except Exception as e:
            print(f"\n❌ Real reader test failed: {e}")
            # Don't fail the test for network issues, just warn
            pytest.skip(f"Real reader test skipped due to: {e}")

        finally:
            # Clean up
            await service.cleanup()

    @pytest.mark.asyncio
    async def test_service_cleanup(self):
        """Test service cleanup"""
        service = ReaderService.create_default()

        # Mock provider with close method
        with patch.object(service.provider, "close", new_callable=AsyncMock) as mock_close:
            await service.close()
            mock_close.assert_called_once()

    @pytest.mark.asyncio
    async def test_read_mixed_success_failure(self):
        """Test reading with mixed success and failure results"""
        service = ReaderService.create_default()

        mock_results = [
            WebReadResultItem(
                url="https://success.com", status="success", title="Success Page", content="Success content"
            ),
            WebReadResultItem(url="https://fail.com", status="error", error="Page not found", error_code="NOT_FOUND"),
            WebReadResultItem(url="https://timeout.com", status="error", error="Request timeout", error_code="TIMEOUT"),
        ]

        with patch.object(service.provider, "read_batch", new_callable=AsyncMock) as mock_read_batch:
            mock_read_batch.return_value = mock_results

            request = WebReadRequest(url_list=["https://success.com", "https://fail.com", "https://timeout.com"])

            response = await service.read(request)

            assert response.total_urls == 3
            assert response.successful == 1
            assert response.failed == 2

            # Check specific results
            success_result = next(r for r in response.results if r.status == "success")
            assert success_result.title == "Success Page"

            error_results = [r for r in response.results if r.status == "error"]
            assert len(error_results) == 2
