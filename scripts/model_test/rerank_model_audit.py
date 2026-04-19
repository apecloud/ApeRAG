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
Rerank Model Test Script

This script tests all available rerank models in the deployed ApeRAG system.
It's designed to be run after system deployment to verify which models are actually
functional, considering factors like API key configuration, provider availability, etc.

Usage:
    python scripts/model_test/rerank_model_audit.py

The script will:
1. Fetch all available rerank models from /api/v1/available_models
2. Test each model by calling /api/v1/rerank with a test query and documents
3. Generate a JSON report with test results

Environment Variables:
    APERAG_API_URL: Base URL for the ApeRAG API (default: http://localhost:8000)
    APERAG_USERNAME: Username for authentication
    APERAG_PASSWORD: Password for authentication
    RERANK_TEST_QUERY: Custom query for rerank test (optional)
    RERANK_TEST_DOCS: Custom documents for rerank test (optional, JSON array)
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

DEFAULT_RERANK_TEST_QUERY = "artificial intelligence machine learning"

DEFAULT_RERANK_TEST_DOCS = [
    "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed.",
    "Natural language processing is a branch of AI that helps computers understand, interpret, and manipulate human language.",
    "Computer vision is an AI technology that enables machines to interpret and understand visual information from the world.",
    "Deep learning is a machine learning technique inspired by the structure and function of the brain called artificial neural networks.",
    "The weather today is sunny with a temperature of 25 degrees Celsius and low humidity.",
    "Cooking pasta requires boiling water with salt and cooking the pasta for 8-12 minutes depending on the type.",
    "Quantum computing uses quantum mechanical phenomena to process information in ways that traditional computers cannot.",
    "Blockchain technology is a distributed ledger system that maintains a continuously growing list of records called blocks.",
    "Renewable energy sources like solar and wind power are becoming increasingly important for sustainable development.",
    "Data science combines statistics, computer science, and domain expertise to extract insights from structured and unstructured data.",
]

RERANK_TEST_QUERY = os.getenv("RERANK_TEST_QUERY", DEFAULT_RERANK_TEST_QUERY)
try:
    RERANK_TEST_DOCS = json.loads(os.getenv("RERANK_TEST_DOCS", "[]")) or DEFAULT_RERANK_TEST_DOCS
except json.JSONDecodeError:
    RERANK_TEST_DOCS = DEFAULT_RERANK_TEST_DOCS

REPORT_FILE = "rerank_model_test_report.json"
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


def test_rerank_model(
    client: httpx.Client, provider: str, model: str, query: str, documents: List[str]
) -> Dict[str, Any]:
    """Test a specific rerank model and return a result dictionary."""
    start_time = time.time()

    try:
        request_body = {
            "provider": provider,
            "model": model,
            "query": query,
            "documents": documents,
            "top_k": min(5, len(documents)),  # Limit to top 5 results
            "return_documents": True,
        }

        response = client.post("/api/v1/rerank", json=request_body)
        response.raise_for_status()

        data = response.json()
        end_time = time.time()

        # Validate response structure
        if (
            "data" in data
            and len(data["data"]) > 0
            and all("index" in item and "relevance_score" in item for item in data["data"])
        ):
            num_results = len(data["data"])
            top_score = data["data"][0]["relevance_score"] if data["data"] else 0.0
            has_documents = all("document" in item for item in data["data"])

            return {
                "test_pass": True,
                "num_results": num_results,
                "top_score": round(top_score, 4),
                "has_documents": has_documents,
                "response_time_seconds": round(end_time - start_time, 2),
                "error_message": None,
            }
        else:
            return {
                "test_pass": False,
                "num_results": None,
                "top_score": None,
                "has_documents": None,
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
            "num_results": None,
            "top_score": None,
            "has_documents": None,
            "response_time_seconds": round(end_time - start_time, 2),
            "error_message": error_details,
        }

    except Exception as e:
        end_time = time.time()
        return {
            "test_pass": False,
            "num_results": None,
            "top_score": None,
            "has_documents": None,
            "response_time_seconds": round(end_time - start_time, 2),
            "error_message": f"Unexpected error: {str(e)}",
        }


def extract_rerank_models(available_models_data: Dict[str, Any]) -> List[Dict[str, str]]:
    """Extract rerank models from the available models response."""
    rerank_models = []

    providers = available_models_data.get("items", [])
    for provider in providers:
        provider_name = provider.get("name", "")
        rerank_list = provider.get("rerank", [])

        if rerank_list:
            for model_info in rerank_list:
                if model_info and isinstance(model_info, dict):
                    model_name = model_info.get("model", "")
                    if model_name:
                        rerank_models.append({"provider": provider_name, "model": model_name})

    return rerank_models


def main():
    """Main function to run the rerank model test."""
    print("=" * 60)
    print("ApeRAG Rerank Model Test")
    print("=" * 60)
    print(f"API Base URL: {API_BASE_URL}")
    print(f"Test Query: {RERANK_TEST_QUERY}")
    print(f"Test Documents: {len(RERANK_TEST_DOCS)} documents")
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

        # Extract rerank models
        rerank_models = extract_rerank_models(available_models_data)

        if not rerank_models:
            print("\nNo rerank models found to test.")
            return

        print(f"\nFound {len(rerank_models)} rerank models to test:")
        for model_info in rerank_models:
            print(f"  - {model_info['provider']} / {model_info['model']}")
        print()

        # Test each rerank model
        report: List[Dict[str, Any]] = []

        for i, model_info in enumerate(rerank_models, 1):
            provider = model_info["provider"]
            model = model_info["model"]

            print(f"[{i}/{len(rerank_models)}] Testing: {provider} / {model}")

            result = test_rerank_model(client, provider, model, RERANK_TEST_QUERY, RERANK_TEST_DOCS)

            report_entry = {
                "provider": provider,
                "model": model,
                "test_pass": result["test_pass"],
                "num_results": result["num_results"],
                "top_score": result["top_score"],
                "has_documents": result["has_documents"],
                "response_time_seconds": result["response_time_seconds"],
                "error_message": result["error_message"],
            }
            report.append(report_entry)

            # Print result
            status = "✅ PASSED" if result["test_pass"] else "❌ FAILED"
            print(f"  Status: {status}")
            if result["num_results"]:
                print(f"  Results: {result['num_results']}")
            if result["top_score"] is not None:
                print(f"  Top Score: {result['top_score']}")
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
                            "test_query": RERANK_TEST_QUERY,
                            "num_test_documents": len(RERANK_TEST_DOCS),
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
