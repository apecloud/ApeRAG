# Copyright 2025 ApeCloud, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
End-to-End Test Suite for Embedding Service Cache Functionality

This module contains comprehensive tests for the LiteLLM caching functionality
integrated with the EmbeddingService. It validates both single query and batch
document embedding requests to ensure cache hit/miss behavior works correctly.

Test Coverage:
- Single query embedding caching (embed_query)
- Batch document embedding caching (embed_documents)
- Cache performance measurements
- Vector consistency validation
- Error handling scenarios
"""

import logging
import time
from typing import Any, Dict, Optional

from litellm.types.caching import LiteLLMCacheType

from aperag.llm import litellm_cache
from aperag.llm.embed.embedding_service import EmbeddingService
from tests.e2e_test.config import (
    EMBEDDING_MODEL_CUSTOM_PROVIDER,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_MODEL_PROVIDER,
    EMBEDDING_MODEL_PROVIDER_API_KEY,
    EMBEDDING_MODEL_PROVIDER_URL,
)

litellm_cache.setup_litellm_cache(default_type=LiteLLMCacheType.LOCAL)

# Configure logging for better test output
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Test configuration constants
CACHE_WAIT_TIME = 2  # seconds to wait for cache updates
PERFORMANCE_THRESHOLD = 0.5  # Cache hit should be at least 50% faster
EMBEDDING_MAX_CHUNKS = 10
TEST_SINGLE_QUERY = "What is artificial intelligence and machine learning?"
TEST_BATCH_DOCUMENTS = [
    "Artificial intelligence is a branch of computer science.",
    "Machine learning algorithms can learn from data patterns.",
    "Neural networks are inspired by biological brain structures.",
    "Deep learning uses multiple layers for feature extraction.",
    "Natural language processing enables computers to understand text.",
]


class EmbeddingCacheTestResult:
    """Data class to store embedding cache test results for analysis"""

    def __init__(self, operation: str):
        self.operation = operation
        self.first_call_time: Optional[float] = None
        self.second_call_time: Optional[float] = None
        self.first_response: Optional[Any] = None
        self.second_response: Optional[Any] = None
        self.cache_effective: bool = False
        self.responses_match: bool = False
        self.vector_dimensions_match: bool = False
        self.error: Optional[str] = None

    def analyze_performance(self) -> Dict[str, Any]:
        """Analyze cache performance and return results"""
        if not (self.first_call_time and self.second_call_time):
            return {"error": "Incomplete timing data"}

        speedup_ratio = self.first_call_time / self.second_call_time if self.second_call_time > 0 else 0
        self.cache_effective = self.second_call_time < (self.first_call_time * PERFORMANCE_THRESHOLD)

        # Check response consistency
        if self.first_response is not None and self.second_response is not None:
            self.responses_match = self._compare_embeddings(self.first_response, self.second_response)
            self.vector_dimensions_match = self._check_dimensions(self.first_response, self.second_response)

        return {
            "first_call_time": self.first_call_time,
            "second_call_time": self.second_call_time,
            "speedup_ratio": speedup_ratio,
            "cache_effective": self.cache_effective,
            "responses_match": self.responses_match,
            "vector_dimensions_match": self.vector_dimensions_match,
            "performance_improvement": f"{speedup_ratio:.2f}x" if speedup_ratio > 1 else "No improvement",
        }

    def _compare_embeddings(self, first: Any, second: Any) -> bool:
        """Compare embedding vectors for exact equality"""
        try:
            if isinstance(first, list) and isinstance(second, list):
                if len(first) != len(second):
                    return False

                # Handle both single vector and list of vectors
                if len(first) > 0 and isinstance(first[0], (int, float)):
                    # Single vector case
                    return first == second
                else:
                    # Multiple vectors case
                    for v1, v2 in zip(first, second):
                        if v1 != v2:
                            return False
                    return True
            return first == second
        except Exception as e:
            logger.warning(f"Error comparing embeddings: {e}")
            return False

    def _check_dimensions(self, first: Any, second: Any) -> bool:
        """Check if embedding dimensions are consistent"""
        try:
            if isinstance(first, list) and isinstance(second, list):
                if len(first) > 0 and isinstance(first[0], (int, float)):
                    # Single vector case
                    return len(first) == len(second)
                else:
                    # Multiple vectors case
                    if len(first) != len(second):
                        return False
                    for v1, v2 in zip(first, second):
                        if len(v1) != len(v2):
                            return False
                    return True
            return True
        except Exception as e:
            logger.warning(f"Error checking dimensions: {e}")
            return False

    def print_summary(self):
        """Print a formatted summary of the test results"""
        print(f"\n{'=' * 60}")
        print(f"EMBEDDING CACHE TEST SUMMARY - {self.operation.upper()}")
        print(f"{'=' * 60}")

        if self.error:
            print(f"❌ Test failed: {self.error}")
            return

        analysis = self.analyze_performance()
        if "error" in analysis:
            print(f"❌ Analysis failed: {analysis['error']}")
            return

        print("📊 Performance Metrics:")
        print(f"   First call time:  {analysis['first_call_time']:.2f} seconds")
        print(f"   Second call time: {analysis['second_call_time']:.2f} seconds")
        print(f"   Speedup ratio:    {analysis['performance_improvement']}")

        print("\n🎯 Cache Effectiveness:")
        if analysis["cache_effective"]:
            print("   ✅ Cache is working! Second call was significantly faster")
        else:
            print("   ❌ Cache may not be effective - minimal performance improvement")

        print("\n🔍 Response Consistency:")
        if analysis["responses_match"]:
            print("   ✅ Both embedding responses are identical")
        else:
            print("   ❌ Embedding responses differ (may indicate cache miss)")

        print("\n📏 Vector Dimensions:")
        if analysis["vector_dimensions_match"]:
            print("   ✅ Vector dimensions are consistent")
        else:
            print("   ❌ Vector dimensions don't match")


def validate_test_configuration() -> bool:
    """Validate that required test configuration is available"""
    if not EMBEDDING_MODEL_PROVIDER_API_KEY:
        logger.warning("EMBEDDING_MODEL_PROVIDER_API_KEY is not configured")
        return False

    if not EMBEDDING_MODEL_NAME:
        logger.warning("EMBEDDING_MODEL_NAME is not configured")
        return False

    return True


def print_test_configuration():
    """Print current test configuration for debugging"""
    print(f"\n{'=' * 60}")
    print("TEST CONFIGURATION")
    print(f"{'=' * 60}")
    print(f"Model Name:      {EMBEDDING_MODEL_NAME}")
    print(f"Provider:        {EMBEDDING_MODEL_PROVIDER}")
    print(f"Custom Provider: {EMBEDDING_MODEL_CUSTOM_PROVIDER}")
    print(f"API URL:         {EMBEDDING_MODEL_PROVIDER_URL}")
    print(
        f"API Key:         {'***' + EMBEDDING_MODEL_PROVIDER_API_KEY[-4:] if EMBEDDING_MODEL_PROVIDER_API_KEY else 'Not Set'}"
    )
    print(f"Max Chunks:      {EMBEDDING_MAX_CHUNKS}")
    print(f"Single Query:    {TEST_SINGLE_QUERY}")
    print(f"Batch Size:      {len(TEST_BATCH_DOCUMENTS)} documents")


def create_embedding_service() -> EmbeddingService:
    """Factory method to create a configured EmbeddingService instance"""
    return EmbeddingService(
        embedding_provider=EMBEDDING_MODEL_CUSTOM_PROVIDER,
        embedding_model=EMBEDDING_MODEL_NAME,
        embedding_service_url=EMBEDDING_MODEL_PROVIDER_URL,
        embedding_service_api_key=EMBEDDING_MODEL_PROVIDER_API_KEY,
        embedding_max_chunks_in_batch=EMBEDDING_MAX_CHUNKS,
        caching=True,
    )


def test_embedding_query_cache():
    """
    Test single query embedding cache functionality

    This test validates that:
    1. First embed_query call executes normally and caches the result
    2. Second identical call retrieves from cache and is significantly faster
    3. Both embedding vectors are identical
    """
    print(f"\n{'=' * 80}")
    print("TESTING SINGLE QUERY EMBEDDING CACHE")
    print(f"{'=' * 80}")

    # Validate configuration
    if not validate_test_configuration():
        print("❌ Test configuration is incomplete - skipping test")
        return

    print_test_configuration()

    # Initialize test result tracker
    result = EmbeddingCacheTestResult("Single Query Embedding")

    try:
        # === FIRST CALL (should miss cache and populate it) ===
        print("\n🔤 Executing first embed_query call (cache miss expected)...")
        start_time = time.time()

        embedding_service = create_embedding_service()
        result.first_response = embedding_service.embed_query(TEST_SINGLE_QUERY)

        result.first_call_time = time.time() - start_time

        print(f"✅ First call completed in {result.first_call_time:.2f} seconds")
        print(f"📊 Vector dimension: {len(result.first_response) if result.first_response else 'N/A'}")
        print(f"📝 Vector preview: {str(result.first_response[:5]) if result.first_response else 'N/A'}...")

        # === WAIT FOR CACHE UPDATE ===
        print(f"\n⏳ Waiting {CACHE_WAIT_TIME} seconds for cache synchronization...")
        time.sleep(CACHE_WAIT_TIME)

        # === SECOND CALL (should hit cache) ===
        print("\n🔤 Executing second embed_query call (cache hit expected)...")
        start_time = time.time()

        embedding_service = create_embedding_service()
        result.second_response = embedding_service.embed_query(TEST_SINGLE_QUERY)

        result.second_call_time = time.time() - start_time

        print(f"✅ Second call completed in {result.second_call_time:.2f} seconds")
        print(f"📊 Vector dimension: {len(result.second_response) if result.second_response else 'N/A'}")
        print(f"📝 Vector preview: {str(result.second_response[:5]) if result.second_response else 'N/A'}...")

    except Exception as e:
        result.error = str(e)
        logger.error(f"Single query test execution failed: {e}", exc_info=True)

    # Print comprehensive results
    result.print_summary()


def test_embedding_documents_cache():
    """
    Test batch document embedding cache functionality

    This test validates that:
    1. First embed_documents call executes normally and caches the result
    2. Second identical call retrieves from cache and is significantly faster
    3. Both embedding vector lists are identical
    """
    print(f"\n{'=' * 80}")
    print("TESTING BATCH DOCUMENTS EMBEDDING CACHE")
    print(f"{'=' * 80}")

    # Validate configuration
    if not validate_test_configuration():
        print("❌ Test configuration is incomplete - skipping test")
        return

    # Initialize test result tracker
    result = EmbeddingCacheTestResult("Batch Documents Embedding")

    try:
        # === FIRST CALL (should miss cache and populate it) ===
        print("\n📚 Executing first embed_documents call (cache miss expected)...")
        print(f"📄 Processing {len(TEST_BATCH_DOCUMENTS)} documents...")
        start_time = time.time()

        embedding_service = create_embedding_service()
        result.first_response = embedding_service.embed_documents(TEST_BATCH_DOCUMENTS)

        result.first_call_time = time.time() - start_time

        print(f"✅ First call completed in {result.first_call_time:.2f} seconds")
        print(f"📊 Generated {len(result.first_response) if result.first_response else 0} vectors")
        if result.first_response and len(result.first_response) > 0:
            print(f"📏 Vector dimension: {len(result.first_response[0])}")
            print(f"📝 First vector preview: {str(result.first_response[0][:5])}...")

        # === WAIT FOR CACHE UPDATE ===
        print(f"\n⏳ Waiting {CACHE_WAIT_TIME} seconds for cache synchronization...")
        time.sleep(CACHE_WAIT_TIME)

        # === SECOND CALL (should hit cache) ===
        print("\n📚 Executing second embed_documents call (cache hit expected)...")
        start_time = time.time()

        embedding_service = create_embedding_service()
        result.second_response = embedding_service.embed_documents(TEST_BATCH_DOCUMENTS)

        result.second_call_time = time.time() - start_time

        print(f"✅ Second call completed in {result.second_call_time:.2f} seconds")
        print(f"📊 Generated {len(result.second_response) if result.second_response else 0} vectors")
        if result.second_response and len(result.second_response) > 0:
            print(f"📏 Vector dimension: {len(result.second_response[0])}")
            print(f"📝 First vector preview: {str(result.second_response[0][:5])}...")

    except Exception as e:
        result.error = str(e)
        logger.error(f"Batch documents test execution failed: {e}", exc_info=True)

    # Print comprehensive results
    result.print_summary()


def main():
    """Main test execution function"""
    print("🚀 Starting Embedding Service Cache Tests")
    print("=" * 80)

    try:
        # Test single query embedding cache functionality
        test_embedding_query_cache()

        # Test batch documents embedding cache functionality
        test_embedding_documents_cache()

    except KeyboardInterrupt:
        print("\n🛑 Tests interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error during test execution: {e}", exc_info=True)
    finally:
        print(f"\n{'=' * 80}")
        print("🏁 EMBEDDING CACHE TESTING COMPLETED")
        print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
