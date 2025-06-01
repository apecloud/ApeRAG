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
# See the for the specific language governing permissions and
# limitations under the License.

import pytest


def assert_dict_subset(subset, superset):
    """
    Assert that all key-value pairs in subset appear in superset (recursively).
    """
    for k, v in subset.items():
        assert k in superset, f"Key '{k}' not found in superset"
        if isinstance(v, dict) and isinstance(superset[k], dict):
            assert_dict_subset(v, superset[k])
        else:
            assert superset[k] == v, f"Value for key '{k}' does not match: {superset[k]} != {v}"

def assert_collection_config(expected, actual):
    """
    Assert that collection_data contains all the expected values from update_data, key by key.
    """
    assert expected["title"] == actual["title"]
    assert expected["description"] == actual["description"]
    for key in ["title", "description"]:
        assert expected[key] == actual[key]

    assert expected["config"]["enable_knowledge_graph"] == actual["config"]["enable_knowledge_graph"]

    for key in ["model", "model_service_provider", "custom_llm_provider", "timeout"]:
        assert expected["config"]["embedding"][key] == actual["config"]["embedding"][key]

    if "completion" in actual["config"]:
        for key in ["model", "model_service_provider", "custom_llm_provider", "timeout"]:
            assert expected["config"]["completion"][key] == actual["config"]["completion"][key]

def assert_search_test_result(expected, actual):
    """
    Assert that result contains all the expected values from search_data, key by key.
    """
    assert expected["query"] == actual["query"]

    if "vector_search" not in expected and "fulltext_search" not in expected and "graph_search" not in expected:
        pytest.fail("No search type specified")
    
    def assert_search_test_result_item(search_type,items):
        for item in items:
            if search_type != "graph_search":
                assert isinstance(item["score"], float)
            assert isinstance(item["content"], str)
            assert isinstance(item["rank"], int)

    if "vector_search" in expected:
        assert actual["vector_search"] is not None
        for key in ["topk", "similarity"]:
            assert expected["vector_search"][key] == actual["vector_search"][key]
        assert_search_test_result_item("vector_search", actual["items"])

    if "fulltext_search" in expected:
        assert actual["fulltext_search"] is not None
        for key in ["topk"]:
            assert expected["fulltext_search"][key] == actual["fulltext_search"][key]
        assert_search_test_result_item("fulltext_search", actual["items"])

    if "graph_search" in expected:
        assert actual["graph_search"] is not None
        for key in ["topk"]:
            assert expected["graph_search"][key] == actual["graph_search"][key]
        assert_search_test_result_item("graph_search", actual["items"])

