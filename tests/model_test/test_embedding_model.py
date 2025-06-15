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
Embedding Model Test Script

This script tests all available embedding models in the deployed ApeRAG system.
It's designed to be run after system deployment to verify which models are actually
functional, considering factors like API key configuration, provider availability, etc.

Usage:
    python tests/model_test/test_embedding_model.py

The script will:
1. Fetch all available embedding models from /api/v1/available_models
2. Test each model by calling /api/v1/embeddings with a test sentence
3. Generate a JSON report with test results

Environment Variables:
    APERAG_API_URL: Base URL for the ApeRAG API (default: http://localhost:8000)
    APERAG_USERNAME: Username for authentication
    APERAG_PASSWORD: Password for authentication
    EMBEDDING_TEST_TEXT: Custom text for embedding test (optional)
"""

import json
import os
import time
from typing import Any, Dict, List, Optional

import httpx

# --- Configuration ---
API_BASE_URL = os.getenv("APERAG_API_URL", "http://localhost:8000")
USERNAME = os.getenv("APERAG_USERNAME", "user@nextmail.com")
PASSWORD = os.getenv("APERAG_PASSWORD", "123456")
EMBEDDING_TEST_TEXT = os.getenv(
    "EMBEDDING_TEST_TEXT", "This is a test sentence for embedding generation. 这是一个用于生成嵌入向量的测试句子。"
)
REPORT_FILE = "embedding_model_test_report.json"
REQUEST_TIMEOUT = 60  # seconds

# --- Helper Functions ---


def login_and_get_session() -> Optional[httpx.Client]:
    """Login to the system and return an authenticated httpx client."""
    try:
        client = httpx.Client(base_url=API_BASE_URL, timeout=REQUEST_TIMEOUT)

        # Login to get session cookies
        login_data = {"username": USERNAME, "password": PASSWORD}
        response = client.post("/api/v1/login", json=login_data)
        response.raise_for_status()

        print(f"Successfully logged in as {USERNAME}")
        return client

    except httpx.HTTPError as e:
        print(f"Login failed: {e}")
        if hasattr(e, "response") and e.response is not None:
            print(f"Response: {e.response.text}")
        return None
    except Exception as e:
        print(f"Unexpected error during login: {e}")
        return None


def get_available_models(client: httpx.Client) -> Optional[Dict[str, Any]]:
    """Fetch all available models from the API."""
    try:
        print("Fetching available models...")
        # Get all models (not just recommended ones)
        request_data = {"tag_filters": []}
        response = client.post("/api/v1/available_models", json=request_data)
        response.raise_for_status()

        data = response.json()
        print("Successfully fetched models.")
        return data

    except httpx.HTTPError as e:
        print(f"Error fetching available models: {e}")
        if hasattr(e, "response") and e.response is not None:
            print(f"Response: {e.response.text}")
        return None
    except Exception as e:
        print(f"Unexpected error fetching models: {e}")
        return None


def test_embedding_model(client: httpx.Client, provider: str, model: str, text: str) -> Dict[str, Any]:
    """Test a specific embedding model and return a result dictionary."""
    start_time = time.time()

    try:
        request_body = {"provider": provider, "model": model, "input": text}

        response = client.post("/api/v1/embeddings", json=request_body)
        response.raise_for_status()

        data = response.json()
        end_time = time.time()

        # Validate response structure
        if (
            "data" in data
            and len(data["data"]) > 0
            and "embedding" in data["data"][0]
            and isinstance(data["data"][0]["embedding"], list)
        ):
            dimension = len(data["data"][0]["embedding"])
            return {
                "test_pass": True,
                "dimension": dimension,
                "response_time_seconds": round(end_time - start_time, 2),
                "error_message": None,
            }
        else:
            return {
                "test_pass": False,
                "dimension": None,
                "response_time_seconds": round(end_time - start_time, 2),
                "error_message": "Invalid response format from API.",
            }

    except httpx.HTTPError as e:
        end_time = time.time()
        error_details = f"HTTP Error: {e.response.status_code}"
        try:
            # Try to get more specific error from response body
            error_body = e.response.json()
            if isinstance(error_body, dict) and "message" in error_body:
                error_details += f" - {error_body['message']}"
            else:
                error_details += f" - {error_body}"
        except (json.JSONDecodeError, AttributeError):
            error_details += f" - {e.response.text}"

        return {
            "test_pass": False,
            "dimension": None,
            "response_time_seconds": round(end_time - start_time, 2),
            "error_message": error_details,
        }

    except Exception as e:
        end_time = time.time()
        return {
            "test_pass": False,
            "dimension": None,
            "response_time_seconds": round(end_time - start_time, 2),
            "error_message": f"Unexpected error: {str(e)}",
        }


def extract_embedding_models(available_models_data: Dict[str, Any]) -> List[Dict[str, str]]:
    """Extract embedding models from the available models response."""
    embedding_models = []

    providers = available_models_data.get("items", [])
    for provider in providers:
        provider_name = provider.get("name", "")
        embedding_list = provider.get("embedding", [])

        if embedding_list:
            for model_info in embedding_list:
                if model_info and isinstance(model_info, dict):
                    model_name = model_info.get("model", "")
                    if model_name:
                        embedding_models.append({"provider": provider_name, "model": model_name})

    return embedding_models


def main():
    """Main function to run the embedding model test."""
    print("=" * 60)
    print("ApeRAG Embedding Model Test")
    print("=" * 60)
    print(f"API Base URL: {API_BASE_URL}")
    print(f"Test Text: {EMBEDDING_TEST_TEXT}")
    print(f"Report File: {REPORT_FILE}")
    print("=" * 60)

    # Login and get authenticated session
    client = login_and_get_session()
    if not client:
        print("\nFailed to login. Exiting.")
        return

    try:
        # Get available models
        available_models_data = get_available_models(client)
        if not available_models_data:
            print("\nCould not retrieve available models. Exiting.")
            return

        # Extract embedding models
        embedding_models = extract_embedding_models(available_models_data)

        if not embedding_models:
            print("\nNo embedding models found to test.")
            return

        print(f"\nFound {len(embedding_models)} embedding models to test:")
        for model_info in embedding_models:
            print(f"  - {model_info['provider']} / {model_info['model']}")
        print()

        # Test each embedding model
        report: List[Dict[str, Any]] = []

        for i, model_info in enumerate(embedding_models, 1):
            provider = model_info["provider"]
            model = model_info["model"]

            print(f"[{i}/{len(embedding_models)}] Testing: {provider} / {model}")

            result = test_embedding_model(client, provider, model, EMBEDDING_TEST_TEXT)

            report_entry = {
                "provider": provider,
                "model": model,
                "dimension": result["dimension"],
                "test_pass": result["test_pass"],
                "response_time_seconds": result["response_time_seconds"],
                "error_message": result["error_message"],
            }
            report.append(report_entry)

            # Print result
            status = "✅ PASSED" if result["test_pass"] else "❌ FAILED"
            print(f"  Status: {status}")
            if result["dimension"]:
                print(f"  Dimension: {result['dimension']}")
            if result["response_time_seconds"]:
                print(f"  Response Time: {result['response_time_seconds']}s")
            if not result["test_pass"] and result["error_message"]:
                print(f"  Error: {result['error_message']}")
            print("-" * 50)

        # Generate summary
        passed_count = sum(1 for entry in report if entry["test_pass"])
        failed_count = len(report) - passed_count

        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        print(f"Total Models Tested: {len(report)}")
        print(f"Passed: {passed_count}")
        print(f"Failed: {failed_count}")
        print(f"Success Rate: {passed_count / len(report) * 100:.1f}%")

        # Save report to file
        try:
            with open(REPORT_FILE, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "test_summary": {
                            "total_models": len(report),
                            "passed": passed_count,
                            "failed": failed_count,
                            "success_rate": round(passed_count / len(report) * 100, 1),
                            "test_text": EMBEDDING_TEST_TEXT,
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        },
                        "results": report,
                    },
                    f,
                    indent=2,
                    ensure_ascii=False,
                )

            print(f"\nReport saved to: {os.path.abspath(REPORT_FILE)}")

        except IOError as e:
            print(f"\nError saving report file: {e}")

    finally:
        client.close()


if __name__ == "__main__":
    main()
