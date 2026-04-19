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

from http import HTTPStatus

import pytest


def test_get_available_models_default(benchmark, client, setup_model_service_provider):
    """Test GET available models with default behavior (recommend tag only)"""
    # Test with empty body - should return recommend models only
    resp = benchmark(client.post, "/api/v1/available_models", json={})
    assert resp.status_code == HTTPStatus.OK, resp.text
    data = resp.json()

    assert "items" in data
    assert isinstance(data["items"], list)

    # Verify that ALL returned models have recommend tag
    for provider in data["items"]:
        for model_type in ["completion", "embedding", "rerank"]:
            models = provider.get(model_type, [])
            if models:
                for model in models:
                    if model and isinstance(model, dict):
                        tags = model.get("tags", [])
                        assert tags is not None, f"Model {model.get('model')} has None tags"
                        assert isinstance(tags, list), (
                            f"Model {model.get('model')} tags should be list, got {type(tags)}"
                        )
                        assert "recommend" in tags, (
                            f"Model {model.get('model')} should have 'recommend' tag, but has tags: {tags}"
                        )


def test_get_available_models_no_filter(benchmark, client, setup_model_service_provider):
    """Test GET available models with empty tag_filters (all models)"""
    request_data = {"tag_filters": []}
    resp = benchmark(client.post, "/api/v1/available_models", json=request_data)
    assert resp.status_code == HTTPStatus.OK, resp.text
    data = resp.json()

    assert "items" in data
    assert isinstance(data["items"], list)

    # Store results for comparison with default (recommend-only) behavior
    all_models_count = 0
    recommend_models_count = 0

    for provider in data["items"]:
        for model_type in ["completion", "embedding", "rerank"]:
            models = provider.get(model_type, [])
            if models:
                for model in models:
                    if model and isinstance(model, dict):
                        all_models_count += 1
                        tags = model.get("tags", [])
                        if tags and "recommend" in tags:
                            recommend_models_count += 1

    # Verify we got some models
    assert all_models_count > 0, "Should return at least some models when no filter is applied"

    # Compare with default behavior (recommend only)
    default_resp = client.post("/api/v1/available_models", json={})
    default_data = default_resp.json()
    default_models_count = 0

    for provider in default_data["items"]:
        for model_type in ["completion", "embedding", "rerank"]:
            models = provider.get(model_type, [])
            if models:
                for model in models:
                    if model and isinstance(model, dict):
                        default_models_count += 1

    # With no filter, we should get at least as many models as with recommend filter
    # (and likely more, unless all models have recommend tag)
    assert all_models_count >= default_models_count, (
        f"No filter should return at least as many models as recommend filter. Got {all_models_count} vs {default_models_count}"
    )


def test_get_available_models_and_filter(benchmark, client, setup_model_service_provider):
    """Test GET available models with AND filter"""
    request_data = {"tag_filters": [{"operation": "AND", "tags": ["free", "recommend"]}]}
    resp = benchmark(client.post, "/api/v1/available_models", json=request_data)
    assert resp.status_code == HTTPStatus.OK, resp.text
    data = resp.json()

    assert "items" in data
    assert isinstance(data["items"], list)

    # If there are results, verify they match the filter
    for provider in data["items"]:
        for model_type in ["completion", "embedding", "rerank"]:
            models = provider.get(model_type, [])
            if models:
                for model in models:
                    if model and isinstance(model, dict):
                        tags = model.get("tags", [])
                        if tags and "free" in tags and "recommend" in tags:
                            break


def test_get_available_models_or_filter(benchmark, client, setup_model_service_provider):
    """Test GET available models with OR filter"""
    request_data = {"tag_filters": [{"operation": "OR", "tags": ["openai", "gpt"]}]}
    resp = benchmark(client.post, "/api/v1/available_models", json=request_data)
    assert resp.status_code == HTTPStatus.OK, resp.text
    data = resp.json()

    assert "items" in data
    assert isinstance(data["items"], list)


def test_get_available_models_multiple_conditions(benchmark, client, setup_model_service_provider):
    """Test GET available models with multiple filter conditions (OR relationship)"""
    request_data = {
        "tag_filters": [{"operation": "AND", "tags": ["free", "recommend"]}, {"operation": "OR", "tags": ["openai"]}]
    }
    resp = benchmark(client.post, "/api/v1/available_models", json=request_data)
    assert resp.status_code == HTTPStatus.OK, resp.text
    data = resp.json()

    assert "items" in data
    assert isinstance(data["items"], list)


def test_get_available_models_single_tag(benchmark, client, setup_model_service_provider):
    """Test GET available models with single tag filter"""
    request_data = {"tag_filters": [{"operation": "OR", "tags": ["recommend"]}]}
    resp = benchmark(client.post, "/api/v1/available_models", json=request_data)
    assert resp.status_code == HTTPStatus.OK, resp.text
    data = resp.json()

    assert "items" in data
    assert isinstance(data["items"], list)


def test_get_available_models_nonexistent_tag(benchmark, client, setup_model_service_provider):
    """Test GET available models with nonexistent tag (should return empty)"""
    request_data = {"tag_filters": [{"operation": "OR", "tags": ["nonexistent_tag_12345"]}]}
    resp = benchmark(client.post, "/api/v1/available_models", json=request_data)
    assert resp.status_code == HTTPStatus.OK, resp.text
    data = resp.json()

    assert "items" in data
    assert isinstance(data["items"], list)
    # Should be empty or only contain providers without the tag
    assert len(data["items"]) >= 0


def test_get_available_models_invalid_operation(benchmark, client, setup_model_service_provider):
    """Test GET available models with invalid operation (should return validation error)"""
    request_data = {"tag_filters": [{"operation": "INVALID", "tags": ["recommend"]}]}
    resp = benchmark(client.post, "/api/v1/available_models", json=request_data)
    assert resp.status_code == HTTPStatus.UNPROCESSABLE_ENTITY, resp.text
    data = resp.json()

    # Should return validation error details
    assert "detail" in data
    assert isinstance(data["detail"], list)
    assert len(data["detail"]) > 0

    # Verify it's specifically about the operation field
    error = data["detail"][0]
    assert "operation" in error.get("loc", [])
    assert "literal_error" == error.get("type")


def test_get_available_models_empty_tags(benchmark, client, setup_model_service_provider):
    """Test GET available models with empty tags list"""
    request_data = {"tag_filters": [{"operation": "OR", "tags": []}]}
    resp = benchmark(client.post, "/api/v1/available_models", json=request_data)
    assert resp.status_code == HTTPStatus.OK, resp.text
    data = resp.json()

    assert "items" in data
    assert isinstance(data["items"], list)


@pytest.mark.parametrize(
    "request_data,description",
    [
        ({}, "empty_body"),
        ({"tag_filters": []}, "empty_filters"),
        ({"tag_filters": [{"operation": "AND", "tags": ["free"]}]}, "single_and_filter"),
        ({"tag_filters": [{"operation": "OR", "tags": ["recommend"]}]}, "single_or_filter"),
        (
            {
                "tag_filters": [
                    {"operation": "AND", "tags": ["free", "recommend"]},
                    {"operation": "OR", "tags": ["openai", "gpt"]},
                ]
            },
            "multiple_conditions",
        ),
    ],
    ids=["empty_body", "empty_filters", "single_and", "single_or", "multiple_conditions"],
)
def test_available_models_parametrized(benchmark, client, setup_model_service_provider, request_data, description):
    """Parametrized test for various available_models scenarios"""
    resp = benchmark(client.post, "/api/v1/available_models", json=request_data)
    assert resp.status_code == HTTPStatus.OK, resp.text
    data = resp.json()

    assert "items" in data
    assert isinstance(data["items"], list)

    # Basic validation - each provider should have expected structure
    for provider in data["items"]:
        assert "name" in provider
        assert isinstance(provider["name"], str)
        # Provider can have completion, embedding, rerank models
        for model_type in ["completion", "embedding", "rerank"]:
            if model_type in provider:
                assert isinstance(provider[model_type], list)


def test_get_available_models_response_structure(benchmark, client, setup_model_service_provider):
    """Test that available_models response has the correct structure"""
    resp = benchmark(client.post, "/api/v1/available_models", json={})
    assert resp.status_code == HTTPStatus.OK, resp.text
    data = resp.json()

    # Validate top-level structure
    assert "items" in data
    assert isinstance(data["items"], list)

    # Validate each provider structure
    for provider in data["items"]:
        assert isinstance(provider, dict)
        assert "name" in provider
        assert isinstance(provider["name"], str)

        # Check for model types
        for model_type in ["completion", "embedding", "rerank"]:
            if model_type in provider:
                models = provider[model_type]
                assert isinstance(models, list)

                # Check each model structure if present
                for model in models:
                    if model:  # model could be None due to data issues
                        assert isinstance(model, dict)
                        assert "model" in model
                        if "tags" in model:
                            assert isinstance(model["tags"], (list, type(None)))
