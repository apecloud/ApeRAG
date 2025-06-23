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
End-to-End Test Suite for Completion Service Cache Functionality

This module contains comprehensive tests for the LiteLLM caching functionality
integrated with the CompletionService. It validates both regular and streaming
completion requests to ensure cache hit/miss behavior works correctly.

Test Coverage:
- Non-streaming completion caching
- Streaming completion caching
- Cache performance measurements
- Error handling scenarios
"""

import logging
import time
from typing import Any, Dict, Optional

import litellm
from litellm import completion
from litellm.caching.caching import Cache
from litellm.types.caching import LiteLLMCacheType

from aperag.llm import litellm_cache
from aperag.llm.completion import CompletionService
from tests.e2e_test.config import (
    COMPLETION_MODEL_CUSTOM_PROVIDER,
    COMPLETION_MODEL_NAME,
    COMPLETION_MODEL_PROVIDER,
    COMPLETION_MODEL_PROVIDER_API_KEY,
    COMPLETION_MODEL_PROVIDER_URL,
)

# Configure logging for better test output
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Test configuration constants
CACHE_WAIT_TIME = 2  # seconds to wait for cache updates
PERFORMANCE_THRESHOLD = 0.5  # Cache hit should be at least 50% faster
TEST_TEMPERATURE = 0.7
TEST_QUERY = "Please provide a brief introduction to the development history of artificial intelligence"


litellm_cache.setup_litellm_cache(default_type=LiteLLMCacheType.LOCAL)


class CacheTestResult:
    """Data class to store cache test results for analysis"""

    def __init__(self, operation: str):
        self.operation = operation
        self.first_call_time: Optional[float] = None
        self.second_call_time: Optional[float] = None
        self.first_response: Optional[str] = None
        self.second_response: Optional[str] = None
        self.cache_effective: bool = False
        self.responses_match: bool = False
        self.error: Optional[str] = None

    def analyze_performance(self) -> Dict[str, Any]:
        """Analyze cache performance and return results"""
        if not (self.first_call_time and self.second_call_time):
            return {"error": "Incomplete timing data"}

        speedup_ratio = self.first_call_time / self.second_call_time if self.second_call_time > 0 else 0
        self.cache_effective = self.second_call_time < (self.first_call_time * PERFORMANCE_THRESHOLD)
        self.responses_match = (
            (self.first_response == self.second_response)
            if both_responses_available(self.first_response, self.second_response)
            else False
        )

        return {
            "first_call_time": self.first_call_time,
            "second_call_time": self.second_call_time,
            "speedup_ratio": speedup_ratio,
            "cache_effective": self.cache_effective,
            "responses_match": self.responses_match,
            "performance_improvement": f"{speedup_ratio:.2f}x" if speedup_ratio > 1 else "No improvement",
        }

    def print_summary(self):
        """Print a formatted summary of the test results"""
        print(f"\n{'=' * 60}")
        print(f"CACHE TEST SUMMARY - {self.operation.upper()}")
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
            print("   ✅ Both responses are identical")
        else:
            print("   ❌ Responses differ (may indicate cache miss)")


def both_responses_available(first: Optional[str], second: Optional[str]) -> bool:
    """Check if both responses are available for comparison"""
    return first is not None and second is not None


def validate_test_configuration() -> bool:
    """Validate that required test configuration is available"""
    if not COMPLETION_MODEL_PROVIDER_API_KEY:
        logger.warning("COMPLETION_MODEL_PROVIDER_API_KEY is not configured")
        return False

    if not COMPLETION_MODEL_NAME:
        logger.warning("COMPLETION_MODEL_NAME is not configured")
        return False

    return True


def print_test_configuration():
    """Print current test configuration for debugging"""
    print(f"\n{'=' * 60}")
    print("TEST CONFIGURATION")
    print(f"{'=' * 60}")
    print(f"Model Name:     {COMPLETION_MODEL_NAME}")
    print(f"Provider:       {COMPLETION_MODEL_PROVIDER}")
    print(f"Custom Provider: {COMPLETION_MODEL_CUSTOM_PROVIDER}")
    print(f"API URL:        {COMPLETION_MODEL_PROVIDER_URL}")
    print(
        f"API Key:        {'***' + COMPLETION_MODEL_PROVIDER_API_KEY[-4:] if COMPLETION_MODEL_PROVIDER_API_KEY else 'Not Set'}"
    )
    print(f"Temperature:    {TEST_TEMPERATURE}")
    print(f"Test Query:     {TEST_QUERY}")


def create_completion_service() -> CompletionService:
    """Factory method to create a configured CompletionService instance"""
    return CompletionService(
        provider=COMPLETION_MODEL_CUSTOM_PROVIDER,
        model=COMPLETION_MODEL_NAME,
        base_url=COMPLETION_MODEL_PROVIDER_URL,
        api_key=COMPLETION_MODEL_PROVIDER_API_KEY,
        temperature=TEST_TEMPERATURE,
        caching=True,
    )


def test_completion_cache():
    """
    Test non-streaming completion service cache functionality

    This test validates that:
    1. First call executes normally and caches the result
    2. Second identical call retrieves from cache and is significantly faster
    3. Both responses are identical
    """
    print(f"\n{'=' * 80}")
    print("TESTING NON-STREAMING COMPLETION CACHE")
    print(f"{'=' * 80}")

    # Initialize cache (using in-memory cache for testing)
    litellm.cache = Cache()

    # Validate configuration
    if not validate_test_configuration():
        print("❌ Test configuration is incomplete - skipping test")
        return

    print_test_configuration()

    # Initialize test result tracker
    result = CacheTestResult("Non-streaming Completion")

    try:
        # === FIRST CALL (should miss cache and populate it) ===
        print("\n📡 Executing first call (cache miss expected)...")
        start_time = time.time()

        completion_service = create_completion_service()
        result.first_response = completion_service.generate(history=None, prompt=TEST_QUERY, memory=False)

        result.first_call_time = time.time() - start_time

        print(f"✅ First call completed in {result.first_call_time:.2f} seconds")
        print(f"📝 Response preview: {result.first_response[:100]}...")

        # === WAIT FOR CACHE UPDATE ===
        print(f"\n⏳ Waiting {CACHE_WAIT_TIME} seconds for cache synchronization...")
        time.sleep(CACHE_WAIT_TIME)

        # === SECOND CALL (should hit cache) ===
        print("\n📡 Executing second call (cache hit expected)...")
        start_time = time.time()

        completion_service = create_completion_service()
        result.second_response = completion_service.generate(history=None, prompt=TEST_QUERY, memory=False)

        result.second_call_time = time.time() - start_time

        print(f"✅ Second call completed in {result.second_call_time:.2f} seconds")
        print(f"📝 Response preview: {result.second_response[:100]}...")

    except Exception as e:
        result.error = str(e)
        logger.error(f"Test execution failed: {e}", exc_info=True)

    # Print comprehensive results
    result.print_summary()


def test_completion_cache_with_stream():
    """
    Test streaming completion cache functionality

    This test validates that:
    1. First streaming call executes normally and caches the result
    2. Second identical call retrieves from cache and streams back cached content
    3. Both streaming responses produce identical final content
    """
    print(f"\n{'=' * 80}")
    print("TESTING STREAMING COMPLETION CACHE")
    print(f"{'=' * 80}")

    # Initialize cache
    litellm.cache = Cache()

    if not validate_test_configuration():
        print("❌ Test configuration is incomplete - skipping test")
        return

    # Initialize test result tracker
    result = CacheTestResult("Streaming Completion")

    # Test messages for streaming
    test_messages = [{"role": "user", "content": "Explain machine learning in one sentence."}]

    try:
        # === FIRST STREAMING CALL ===
        print("\n📡 Executing first streaming call (cache miss expected)...")
        start_time = time.time()
        first_content = ""

        response1 = completion(
            model=f"{COMPLETION_MODEL_CUSTOM_PROVIDER}/{COMPLETION_MODEL_NAME}",
            messages=test_messages,
            api_key=COMPLETION_MODEL_PROVIDER_API_KEY,
            base_url=COMPLETION_MODEL_PROVIDER_URL,
            stream=True,
            caching=True,
            temperature=TEST_TEMPERATURE,
        )

        print("📡 Streaming response: ", end="", flush=True)
        for chunk in response1:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                first_content += content
                print(content, end="", flush=True)

        result.first_call_time = time.time() - start_time
        result.first_response = first_content

        print(f"\n✅ First streaming call completed in {result.first_call_time:.2f} seconds")

        # === WAIT FOR CACHE UPDATE ===
        print(f"\n⏳ Waiting {CACHE_WAIT_TIME} seconds for cache synchronization...")
        time.sleep(CACHE_WAIT_TIME)

        # === SECOND STREAMING CALL ===
        print("\n📡 Executing second streaming call (cache hit expected)...")
        start_time = time.time()
        second_content = ""

        response2 = completion(
            model=f"{COMPLETION_MODEL_CUSTOM_PROVIDER}/{COMPLETION_MODEL_NAME}",
            messages=test_messages,
            api_key=COMPLETION_MODEL_PROVIDER_API_KEY,
            base_url=COMPLETION_MODEL_PROVIDER_URL,
            stream=True,
            caching=True,
            temperature=TEST_TEMPERATURE,
        )

        print("📡 Streaming response: ", end="", flush=True)
        for chunk in response2:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                second_content += content
                print(content, end="", flush=True)

        result.second_call_time = time.time() - start_time
        result.second_response = second_content

        print(f"\n✅ Second streaming call completed in {result.second_call_time:.2f} seconds")

    except Exception as e:
        result.error = str(e)
        logger.error(f"Streaming test execution failed: {e}", exc_info=True)

    # Print comprehensive results
    result.print_summary()


def main():
    """Main test execution function"""
    print("🚀 Starting Completion Service Cache Tests")
    print("=" * 80)

    try:
        # Test non-streaming cache functionality
        test_completion_cache()

        # Test streaming cache functionality
        test_completion_cache_with_stream()

    except KeyboardInterrupt:
        print("\n🛑 Tests interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error during test execution: {e}", exc_info=True)
    finally:
        print(f"\n{'=' * 80}")
        print("🏁 CACHE TESTING COMPLETED")
        print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
