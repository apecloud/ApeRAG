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

import urllib.parse
from http import HTTPStatus

import pytest


class TestLLMProvider:
    """Test LLM provider management functionality"""

    def test_get_llm_configuration(self, client):
        """Test getting complete LLM configuration"""
        resp = client.get("/api/v1/llm_configuration")
        assert resp.status_code == HTTPStatus.OK, f"Failed to get LLM configuration: {resp.text}"

        data = resp.json()
        assert "providers" in data
        assert "models" in data
        assert isinstance(data["providers"], list)
        assert isinstance(data["models"], list)

    def test_get_provider_models(self, client):
        """Test getting models for specific provider"""
        resp = client.get("/api/v1/llm_providers/siliconflow/models")
        assert resp.status_code == HTTPStatus.OK, f"Failed to get siliconflow models: {resp.text}"

        data = resp.json()
        assert "items" in data
        models = data["items"]

        # Find the Qwen/Qwen3-8B model to verify it exists
        qwen_model = None
        for model in models:
            if model["model"] == "Qwen/Qwen3-8B" and model["api"] == "completion":
                qwen_model = model
                break

        assert qwen_model is not None, "Qwen/Qwen3-8B model not found in siliconflow models"
        assert qwen_model["provider_name"] == "siliconflow"
        assert qwen_model["custom_llm_provider"] == "openai"

    def test_update_model_with_slash_in_name(self, client):
        """Test updating a model with slash in name - this is the main test case for the bug fix"""
        # Create a simple test model first to avoid dependency on existing data
        test_model_name = "test-org/test-model"

        # Create test model
        model_data = {
            "provider_name": "siliconflow",
            "api": "completion",
            "model": test_model_name,
            "custom_llm_provider": "openai",
            "context_window": 4096,
            "tags": ["test"]
        }

        create_resp = client.post("/api/v1/llm_providers/siliconflow/models", json=model_data)

        # If model already exists, that's okay for our test
        if create_resp.status_code == HTTPStatus.BAD_REQUEST and "already exists" in create_resp.text:
            pass  # Continue with the test
        else:
            assert create_resp.status_code == HTTPStatus.OK, f"Failed to create test model: {create_resp.text}"

        # Test updating the model with slash in name - this is the main functionality we're testing
        encoded_model = urllib.parse.quote(test_model_name, safe="")
        update_data = {
            "context_window": 8192,
            "tags": ["test", "updated"]
        }
        
        # This should work now with the :path fix
        resp = client.put(f"/api/v1/llm_providers/siliconflow/models/completion/{encoded_model}", json=update_data)
        assert resp.status_code == HTTPStatus.OK, f"Failed to update model with slash: {resp.text}"

        updated_model = resp.json()
        assert updated_model["model"] == test_model_name
        assert updated_model["context_window"] == 8192
        assert "test" in updated_model["tags"]
        assert "updated" in updated_model["tags"]

        # Cleanup: delete the test model
        delete_resp = client.delete(f"/api/v1/llm_providers/siliconflow/models/completion/{encoded_model}")
        assert delete_resp.status_code == HTTPStatus.OK, f"Failed to delete test model: {delete_resp.text}"

    def test_complex_model_name_with_multiple_slashes(self, client):
        """Test handling of complex model names with multiple slashes"""
        complex_model_name = "org/user/model-name/v1.0"

        # Create a test model with complex name
        model_data = {
            "provider_name": "siliconflow",
            "api": "completion",
            "model": complex_model_name,
            "custom_llm_provider": "openai",
            "context_window": 2048,
            "tags": ["test", "complex"]
        }

        resp = client.post("/api/v1/llm_providers/siliconflow/models", json=model_data)

        # Skip if model already exists
        if resp.status_code == HTTPStatus.BAD_REQUEST and "already exists" in resp.text:
            pytest.skip("Complex test model already exists")

        assert resp.status_code == HTTPStatus.OK, f"Failed to create complex model: {resp.text}"

        # Test updating the complex model
        encoded_model = urllib.parse.quote(complex_model_name, safe="")
        update_data = {"context_window": 4096}
        
        update_resp = client.put(
            f"/api/v1/llm_providers/siliconflow/models/completion/{encoded_model}", json=update_data
        )
        assert update_resp.status_code == HTTPStatus.OK, f"Failed to update complex model: {update_resp.text}"

        updated_model = update_resp.json()
        assert updated_model["model"] == complex_model_name
        assert updated_model["context_window"] == 4096
        
        # Cleanup
        delete_resp = client.delete(f"/api/v1/llm_providers/siliconflow/models/completion/{encoded_model}")
        assert delete_resp.status_code == HTTPStatus.OK, f"Failed to delete complex model: {delete_resp.text}"

    def test_nonexistent_model_with_slash(self, client):
        """Test that nonexistent models with slashes return proper 404"""
        nonexistent_model = "nonexistent/model"
        encoded_model = urllib.parse.quote(nonexistent_model, safe="")
        
        update_data = {"context_window": 1000}
        resp = client.put(
            f"/api/v1/llm_providers/siliconflow/models/completion/{encoded_model}",
            json=update_data
        )
        
        assert resp.status_code == HTTPStatus.NOT_FOUND, f"Expected 404 for nonexistent model, got: {resp.status_code}"
        error_data = resp.json()
        assert "not found" in error_data.get("message", "").lower()

    def test_update_actual_qwen_model(self, client):
        """Test updating the actual Qwen/Qwen3-8B model that was mentioned in the original issue"""
        # First, check if the Qwen/Qwen3-8B model exists
        resp = client.get("/api/v1/llm_providers/siliconflow/models")
        assert resp.status_code == HTTPStatus.OK, f"Failed to get models: {resp.text}"

        models = resp.json()["items"]
        qwen_model = None
        for model in models:
            if model["model"] == "Qwen/Qwen3-8B" and model["api"] == "completion":
                qwen_model = model
                break

        if qwen_model is None:
            pytest.skip("Qwen/Qwen3-8B model not found, skipping update test")

        # Store original values for restoration
        original_context_window = qwen_model.get("context_window")
        original_tags = qwen_model.get("tags", [])

        # Prepare update data
        new_context_window = 8192 if original_context_window != 8192 else 4096
        update_data = {
            "context_window": new_context_window,
            "tags": original_tags + ["e2e-test"]
        }
        
        # Test updating the model - this was the original failing case
        encoded_model = urllib.parse.quote("Qwen/Qwen3-8B", safe="")
        resp = client.put(f"/api/v1/llm_providers/siliconflow/models/completion/{encoded_model}", json=update_data)
        assert resp.status_code == HTTPStatus.OK, f"Failed to update Qwen/Qwen3-8B model: {resp.text}"

        updated_model = resp.json()
        assert updated_model["model"] == "Qwen/Qwen3-8B"
        assert updated_model["context_window"] == new_context_window
        assert "e2e-test" in updated_model["tags"]

        # Restore original values
        restore_data = {
            "context_window": original_context_window,
            "tags": original_tags
        }
        restore_resp = client.put(
            f"/api/v1/llm_providers/siliconflow/models/completion/{encoded_model}", json=restore_data
        )
        assert restore_resp.status_code == HTTPStatus.OK, f"Failed to restore model: {restore_resp.text}"
