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

from unittest.mock import MagicMock

from aperag.schema import view_models
from aperag.service.llm_available_model_service import _filter_models_by_tags, _filter_providers_by_tags


class TestFilterModelsByTags:
    """Test cases for filter_models_by_tags function"""

    def test_filter_models_by_tags_empty_filters(self):
        """Test with empty filters - should return all models"""
        models = [
            {"name": "model1", "tags": ["free", "recommend"]},
            {"name": "model2", "tags": ["premium"]},
        ]
        result = _filter_models_by_tags(models, None)
        assert result == models

        result = _filter_models_by_tags(models, [])
        assert result == models

    def test_filter_models_by_tags_and_operation(self):
        """Test AND operation - all tags must be present"""
        models = [
            {"name": "model1", "tags": ["free", "recommend"]},
            {"name": "model2", "tags": ["free"]},
            {"name": "model3", "tags": ["recommend"]},
            {"name": "model4", "tags": ["free", "recommend", "premium"]},
        ]

        tag_filter = view_models.TagFilterCondition(operation="AND", tags=["free", "recommend"])
        result = _filter_models_by_tags(models, [tag_filter])

        # Should return models that have both "free" and "recommend" tags
        expected_models = [models[0], models[3]]  # model1 and model4
        assert result == expected_models

    def test_filter_models_by_tags_or_operation(self):
        """Test OR operation - at least one tag must be present"""
        models = [
            {"name": "model1", "tags": ["free"]},
            {"name": "model2", "tags": ["recommend"]},
            {"name": "model3", "tags": ["premium"]},
            {"name": "model4", "tags": ["free", "recommend"]},
        ]

        tag_filter = view_models.TagFilterCondition(operation="OR", tags=["free", "recommend"])
        result = _filter_models_by_tags(models, [tag_filter])

        # Should return models that have either "free" or "recommend" tags
        expected_models = [models[0], models[1], models[3]]  # model1, model2, model4
        assert result == expected_models

    def test_filter_models_by_tags_multiple_conditions(self):
        """Test multiple filter conditions - OR relationship between conditions"""
        models = [
            {"name": "model1", "tags": ["free", "recommend"]},
            {"name": "model2", "tags": ["openai"]},
            {"name": "model3", "tags": ["premium"]},
            {"name": "model4", "tags": ["free"]},
        ]

        filters = [
            view_models.TagFilterCondition(operation="AND", tags=["free", "recommend"]),
            view_models.TagFilterCondition(operation="OR", tags=["openai"]),
        ]
        result = _filter_models_by_tags(models, filters)

        # Should return models that match either condition
        expected_models = [models[0], models[1]]  # model1 (free AND recommend), model2 (openai)
        assert result == expected_models

    def test_filter_models_by_tags_no_tags_field(self):
        """Test models without tags field"""
        models = [
            {"name": "model1"},  # No tags field
            {"name": "model2", "tags": ["free"]},
            {"name": "model3", "tags": None},  # None tags
        ]

        tag_filter = view_models.TagFilterCondition(operation="OR", tags=["free"])
        result = _filter_models_by_tags(models, [tag_filter])

        # Should only return model2
        expected_models = [models[1]]
        assert result == expected_models

    def test_filter_models_by_tags_empty_tags(self):
        """Test filter with empty tags list"""
        models = [
            {"name": "model1", "tags": ["free"]},
            {"name": "model2", "tags": ["recommend"]},
        ]

        tag_filter = view_models.TagFilterCondition(operation="OR", tags=[])
        result = _filter_models_by_tags(models, [tag_filter])

        # Should return empty list since no tags to match
        assert result == []

    def test_filter_models_by_tags_case_sensitivity(self):
        """Test that tag matching is case sensitive"""
        models = [
            {"name": "model1", "tags": ["Free"]},
            {"name": "model2", "tags": ["free"]},
        ]

        tag_filter = view_models.TagFilterCondition(operation="OR", tags=["free"])
        result = _filter_models_by_tags(models, [tag_filter])

        # Should only return model2 (case sensitive match)
        expected_models = [models[1]]
        assert result == expected_models


class TestFilterProvidersByTags:
    """Test cases for _filter_providers_by_tags function"""

    def create_mock_provider(self, name: str, completion_models=None, embedding_models=None, rerank_models=None):
        """Helper to create mock provider"""
        provider_data = {"name": name}
        if completion_models:
            provider_data["completion"] = completion_models
        if embedding_models:
            provider_data["embedding"] = embedding_models
        if rerank_models:
            provider_data["rerank"] = rerank_models

        mock_provider = MagicMock(spec=view_models.ModelConfig)
        mock_provider.model_dump.return_value = provider_data
        return mock_provider

    def test_filter_providers_by_tags_basic(self):
        """Test basic provider filtering - should filter at model level"""
        completion_models = [
            {"model": "gpt-4", "tags": ["recommend", "premium"]},
            {"model": "gpt-3.5", "tags": ["free"]},  # This model should be filtered out
        ]

        provider = self.create_mock_provider("openai", completion_models=completion_models)

        tag_filter = view_models.TagFilterCondition(operation="OR", tags=["recommend"])
        result = _filter_providers_by_tags([provider], [tag_filter])

        # Should return provider with only the models that match the filter
        assert len(result) == 1
        provider_result = result[0]

        # Verify the provider was reconstructed correctly
        assert isinstance(provider_result, view_models.ModelConfig)

        # Verify that provider.model_dump was called to get the original data
        provider.model_dump.assert_called_once()

    def test_filter_providers_by_tags_no_match(self):
        """Test provider filtering with no matching tags - should return empty list"""
        completion_models = [
            {"model": "model1", "tags": ["premium"]},
            {"model": "model2", "tags": ["enterprise"]},
        ]

        provider = self.create_mock_provider("provider1", completion_models=completion_models)

        tag_filter = view_models.TagFilterCondition(operation="OR", tags=["free"])
        result = _filter_providers_by_tags([provider], [tag_filter])

        # Should return empty list since no models match
        assert result == []

    def test_filter_providers_by_tags_partial_match(self):
        """Test provider filtering where only some models match"""
        completion_models = [
            {"model": "model1", "tags": ["recommend"]},
            {"model": "model2", "tags": ["premium"]},
            {"model": "model3", "tags": ["free", "recommend"]},
        ]

        provider = self.create_mock_provider("provider1", completion_models=completion_models)

        tag_filter = view_models.TagFilterCondition(operation="OR", tags=["recommend"])
        result = _filter_providers_by_tags([provider], [tag_filter])

        # Should return provider but verify it was processed
        assert len(result) == 1
        provider.model_dump.assert_called_once()

    def test_filter_providers_by_tags_multiple_model_types(self):
        """Test provider with multiple model types - filter each type separately"""
        completion_models = [{"model": "comp1", "tags": ["premium"]}]
        embedding_models = [
            {"model": "embed1", "tags": ["free", "recommend"]},
            {"model": "embed2", "tags": ["premium"]},
        ]
        rerank_models = [{"model": "rerank1", "tags": ["enterprise"]}]

        provider = self.create_mock_provider(
            "provider1",
            completion_models=completion_models,
            embedding_models=embedding_models,
            rerank_models=rerank_models,
        )

        tag_filter = view_models.TagFilterCondition(operation="OR", tags=["recommend"])
        result = _filter_providers_by_tags([provider], [tag_filter])

        # Should return the provider since one embedding model has "recommend" tag
        assert len(result) == 1

    def test_filter_providers_by_tags_empty_providers(self):
        """Test filtering with empty provider list"""
        tag_filter = view_models.TagFilterCondition(operation="OR", tags=["recommend"])
        result = _filter_providers_by_tags([], [tag_filter])

        assert result == []

    def test_filter_providers_by_tags_model_tags_none(self):
        """Test filtering when model tags field is None"""
        completion_models = [
            {"model": "model1", "tags": None},
            {"model": "model2", "tags": ["recommend"]},
        ]

        provider = self.create_mock_provider("provider1", completion_models=completion_models)

        tag_filter = view_models.TagFilterCondition(operation="OR", tags=["recommend"])
        result = _filter_providers_by_tags([provider], [tag_filter])

        # Should still work and return provider (with model2 matching)
        assert len(result) == 1

    def test_filter_providers_by_tags_all_models_filtered_out(self):
        """Test when all models in a provider are filtered out"""
        completion_models = [
            {"model": "model1", "tags": ["premium"]},
            {"model": "model2", "tags": ["enterprise"]},
        ]
        embedding_models = [
            {"model": "embed1", "tags": ["premium"]},
        ]

        provider = self.create_mock_provider(
            "provider1", completion_models=completion_models, embedding_models=embedding_models
        )

        tag_filter = view_models.TagFilterCondition(operation="OR", tags=["free"])
        result = _filter_providers_by_tags([provider], [tag_filter])

        # Should return empty list since no models match the filter
        assert result == []
