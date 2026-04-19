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
Completion Model Test Script

This script tests completion models by creating a common bot and using the existing chat API.
It allows manual configuration of provider, model, and test prompts to verify model functionality.

Usage:
    python scripts/model_test/completion_model_audit.py

Configuration:
    Edit the TEST_CONFIGS section below to specify the models and prompts you want to test.

Environment Variables:
    APERAG_API_URL: Base URL for the ApeRAG API (default: http://localhost:8000)
    APERAG_USERNAME: Username for authentication (default: user@nextmail.com)
    APERAG_PASSWORD: Password for authentication (default: 123456)
"""

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import httpx
import yaml

# --- Configuration ---
API_BASE_URL = os.getenv("APERAG_API_URL", "http://localhost:8000")
USERNAME = os.getenv("APERAG_USERNAME", "user@nextmail.com")
PASSWORD = os.getenv("APERAG_PASSWORD", "123456")

# Test configurations - Edit this section to test different models
TEST_CONFIGS = [
    {
        "name": "DeepSeek V3 Test",
        "provider": "openrouter",
        "model": "deepseek/deepseek-v3-base:free",
        "prompts": [
            "Hello, how are you today?",
        ],
        "config_overrides": {
            "temperature": 0.7,
            "context_window": 3500,
        },
    },
    {
        "name": "GPT-4o Mini Test",
        "provider": "openrouter",
        "model": "openai/gpt-4o-mini",
        "prompts": [
            "Tell me a joke about programming.",
            "What are the benefits of using RAG systems?",
        ],
        "config_overrides": {
            "temperature": 0.8,
            "context_window": 8000,
        },
    },
    # Add more test configurations here as needed
]

REPORT_FILE = "completion_model_test_report.json"
REQUEST_TIMEOUT = 120  # seconds

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


def create_bot_config(provider: str, model: str, config_overrides: Dict[str, Any] = None) -> Dict[str, Any]:
    """Create bot configuration for the specified model."""
    base_config = {
        "model_name": model,
        "model_service_provider": provider,
        "llm": {
            "context_window": 3500,
            "temperature": 0.7,
        },
    }

    if config_overrides:
        base_config["llm"].update(config_overrides)

    return base_config


def create_common_bot(
    client: httpx.Client, name: str, provider: str, model: str, config_overrides: Dict[str, Any] = None
) -> Optional[Dict[str, Any]]:
    """Create a common bot with the specified model configuration."""
    try:
        config = create_bot_config(provider, model, config_overrides)

        create_data = {
            "title": f"{name} - Test Bot",
            "description": f"Test bot for {provider}/{model}",
            "type": "common",
            "config": json.dumps(config),
            "collection_ids": [],
        }

        response = client.post("/api/v1/bots", json=create_data)
        response.raise_for_status()

        bot = response.json()
        print(f"Created bot: {bot['id']} - {bot['title']}")

        # Configure flow for the bot
        if not configure_bot_flow(client, bot["id"], provider, model):
            print(f"Warning: Failed to configure flow for bot {bot['id']}")
            # Don't fail here, as the bot is created but may not work properly

        return bot

    except httpx.HTTPError as e:
        print(f"Error creating bot: {e}")
        if hasattr(e, "response") and e.response is not None:
            print(f"Response: {e.response.text}")
        return None
    except Exception as e:
        print(f"Unexpected error creating bot: {e}")
        return None


def configure_bot_flow(client: httpx.Client, bot_id: str, provider: str, model: str) -> bool:
    """Configure flow for the bot using basic-flow.yaml template."""
    try:
        # Load basic flow template from same directory
        flow_path = Path(__file__).parent / "basic-flow.yaml"
        with open(flow_path, "r", encoding="utf-8") as f:
            flow = yaml.safe_load(f)

        # Update model configuration in the flow
        for node in flow.get("nodes", []):
            if node.get("type") == "llm":
                if "data" in node and "input" in node["data"] and "values" in node["data"]["input"]:
                    values = node["data"]["input"]["values"]
                    values["model_service_provider"] = provider
                    values["model_name"] = model
                    values["custom_llm_provider"] = provider

        flow_json = json.dumps(flow)
        response = client.put(
            f"/api/v1/bots/{bot_id}/flow", content=flow_json, headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()

        print(f"Configured flow for bot: {bot_id}")
        return True

    except Exception as e:
        print(f"Error configuring bot flow: {e}")
        return False


def create_chat(client: httpx.Client, bot_id: str, title: str) -> Optional[Dict[str, Any]]:
    """Create a chat for the given bot."""
    try:
        data = {"title": title}
        response = client.post(f"/api/v1/bots/{bot_id}/chats", json=data)
        response.raise_for_status()

        chat = response.json()
        print(f"Created chat: {chat['id']} - {chat['title']}")
        return chat

    except httpx.HTTPError as e:
        print(f"Error creating chat: {e}")
        if hasattr(e, "response") and e.response is not None:
            print(f"Response: {e.response.text}")
        return None
    except Exception as e:
        print(f"Unexpected error creating chat: {e}")
        return None


def test_completion_via_openai_api(
    client: httpx.Client, bot_id: str, chat_id: str, prompt: str, test_name: str
) -> Dict[str, Any]:
    """Test completion using the OpenAI-compatible API (non-streaming)."""
    start_time = time.time()

    try:
        # Use OpenAI-compatible API format
        request_data = {
            "model": "aperag",
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        }

        response = client.post(
            "/v1/chat/completions",
            json=request_data,
            params={"bot_id": bot_id, "chat_id": chat_id},
            timeout=REQUEST_TIMEOUT,
        )

        response.raise_for_status()
        end_time = time.time()

        response_data = response.json()

        # Validate OpenAI response structure
        if "error" in response_data:
            return {
                "test_pass": False,
                "response_length": None,
                "response_time_seconds": round(end_time - start_time, 2),
                "error_message": response_data["error"].get("message", "Unknown OpenAI API error"),
                "response_data": None,
            }
        elif "choices" in response_data and len(response_data["choices"]) > 0:
            choice = response_data["choices"][0]
            if "message" in choice and "content" in choice["message"]:
                response_content = choice["message"]["content"]
                return {
                    "test_pass": True,
                    "response_length": len(response_content),
                    "response_time_seconds": round(end_time - start_time, 2),
                    "error_message": None,
                    "response_data": response_content[:200] + "..."
                    if len(response_content) > 200
                    else response_content,
                }

        return {
            "test_pass": False,
            "response_length": None,
            "response_time_seconds": round(end_time - start_time, 2),
            "error_message": "Invalid OpenAI API response structure",
            "response_data": None,
        }

    except httpx.HTTPError as e:
        end_time = time.time()
        error_details = f"HTTP Error: {e.response.status_code}"
        try:
            error_body = e.response.json()
            if isinstance(error_body, dict) and "error" in error_body:
                error_details += f" - {error_body['error'].get('message', 'Unknown error')}"
            else:
                error_details += f" - {error_body}"
        except (json.JSONDecodeError, AttributeError):
            error_details += f" - {e.response.text}"

        return {
            "test_pass": False,
            "response_length": None,
            "response_time_seconds": round(end_time - start_time, 2),
            "error_message": error_details,
            "response_data": None,
        }

    except Exception as e:
        end_time = time.time()
        return {
            "test_pass": False,
            "response_length": None,
            "response_time_seconds": round(end_time - start_time, 2),
            "error_message": f"Unexpected error: {str(e)}",
            "response_data": None,
        }


def cleanup_resources(client: httpx.Client, bot_id: str, chat_id: str = None):
    """Clean up created resources."""
    try:
        if chat_id:
            delete_resp = client.delete(f"/api/v1/bots/{bot_id}/chats/{chat_id}")
            if delete_resp.status_code not in (200, 204, 404):
                print(f"Warning: Failed to delete chat {chat_id}: {delete_resp.status_code}")

        delete_resp = client.delete(f"/api/v1/bots/{bot_id}")
        if delete_resp.status_code not in (200, 204, 404):
            print(f"Warning: Failed to delete bot {bot_id}: {delete_resp.status_code}")
        else:
            print(f"Cleaned up bot: {bot_id}")

    except Exception as e:
        print(f"Warning: Error during cleanup: {e}")


def main():
    """Main function to run the completion model test."""
    print("=" * 80)
    print("ApeRAG Completion Model Test")
    print("=" * 80)
    print(f"API Base URL: {API_BASE_URL}")
    print(f"Test Configurations: {len(TEST_CONFIGS)}")
    print(f"Report File: {REPORT_FILE}")
    print("=" * 80)

    # Login and get authenticated session
    client = login_and_get_session()
    if not client:
        print("\nFailed to login. Exiting.")
        return

    try:
        all_results = []

        for config_idx, test_config in enumerate(TEST_CONFIGS, 1):
            print(f"\n[{config_idx}/{len(TEST_CONFIGS)}] Testing: {test_config['name']}")
            print(f"Provider: {test_config['provider']}")
            print(f"Model: {test_config['model']}")
            print(f"Prompts: {len(test_config['prompts'])}")
            print("-" * 60)

            # Create bot for this configuration
            bot = create_common_bot(
                client,
                test_config["name"],
                test_config["provider"],
                test_config["model"],
                test_config.get("config_overrides", {}),
            )

            if not bot:
                print(f"❌ Failed to create bot for {test_config['name']}")
                continue

            # Create chat
            chat = create_chat(client, bot["id"], f"Test Chat for {test_config['name']}")
            if not chat:
                print(f"❌ Failed to create chat for {test_config['name']}")
                cleanup_resources(client, bot["id"])
                continue

            # Test each prompt
            config_results = {
                "config_name": test_config["name"],
                "provider": test_config["provider"],
                "model": test_config["model"],
                "config_overrides": test_config.get("config_overrides", {}),
                "bot_id": bot["id"],
                "chat_id": chat["id"],
                "prompt_results": [],
                "summary": {
                    "total_prompts": len(test_config["prompts"]),
                    "openai_api_passed": 0,
                    "avg_response_time": 0,
                    "avg_response_length": 0,
                },
            }

            total_response_time = 0
            total_response_length = 0
            total_successful_responses = 0

            for prompt_idx, prompt in enumerate(test_config["prompts"], 1):
                print(f"  [{prompt_idx}/{len(test_config['prompts'])}] Testing prompt: {prompt[:50]}...")

                # Test OpenAI API only
                openai_result = test_completion_via_openai_api(
                    client, bot["id"], chat["id"], prompt, f"{test_config['name']}_prompt_{prompt_idx}"
                )

                # Aggregate results
                prompt_result = {
                    "prompt": prompt,
                    "openai_api": openai_result,
                }

                config_results["prompt_results"].append(prompt_result)

                # Update summary statistics
                if openai_result["test_pass"]:
                    config_results["summary"]["openai_api_passed"] += 1
                    if openai_result["response_length"]:
                        total_response_length += openai_result["response_length"]
                        total_successful_responses += 1

                total_response_time += openai_result["response_time_seconds"]

                # Print prompt result
                openai_status = "✅" if openai_result["test_pass"] else "❌"
                print(f"    OpenAI API: {openai_status}")

            # Calculate averages
            config_results["summary"]["avg_response_time"] = round(total_response_time / len(test_config["prompts"]), 2)

            if total_successful_responses > 0:
                config_results["summary"]["avg_response_length"] = round(
                    total_response_length / total_successful_responses, 0
                )

            all_results.append(config_results)

            # Print configuration summary
            print("\n  Configuration Summary:")
            print(
                f"  OpenAI API: {config_results['summary']['openai_api_passed']}/{config_results['summary']['total_prompts']} passed"
            )
            print(f"  Avg Response Time: {config_results['summary']['avg_response_time']}s")
            print(f"  Avg Response Length: {config_results['summary']['avg_response_length']} chars")

            # Cleanup resources
            cleanup_resources(client, bot["id"], chat["id"])
            print(f"  ✅ Completed testing {test_config['name']}")

        # Generate overall summary
        total_configs = len(all_results)
        total_openai_passed = sum(r["summary"]["openai_api_passed"] for r in all_results)
        total_prompts = sum(r["summary"]["total_prompts"] for r in all_results)

        print("\n" + "=" * 80)
        print("OVERALL TEST SUMMARY")
        print("=" * 80)
        print(f"Configurations Tested: {total_configs}")
        print(f"Total Prompts Tested: {total_prompts}")
        print(
            f"OpenAI API Success: {total_openai_passed}/{total_prompts} ({total_openai_passed / total_prompts * 100:.1f}%)"
        )

        # Save report to file
        try:
            report_data = {
                "test_summary": {
                    "total_configurations": total_configs,
                    "total_prompts": total_prompts,
                    "openai_api_success": total_openai_passed,
                    "openai_api_success_rate": round(total_openai_passed / total_prompts * 100, 1)
                    if total_prompts > 0
                    else 0,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                },
                "configurations": all_results,
            }

            with open(REPORT_FILE, "w", encoding="utf-8") as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False)

            print(f"\nReport saved to: {os.path.abspath(REPORT_FILE)}")

        except IOError as e:
            print(f"\nError saving report file: {e}")

    finally:
        client.close()


if __name__ == "__main__":
    main()
