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


from tests.e2e_test.config import (
    COMPLETION_MODEL_CUSTOM_PROVIDER,
    COMPLETION_MODEL_NAME,
    COMPLETION_MODEL_PROVIDER,
    EMBEDDING_MODEL_CUSTOM_PROVIDER,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_MODEL_PROVIDER,
)
from tests.e2e_test.utils import assert_collection_config, assert_search_test_result
import pytest


def test_list_collections(client, collection):
    # Get collections list
    resp = client.get("/api/v1/collections")
    assert resp.status_code == 200
    data = resp.json()
    collections = data["items"]
    assert isinstance(collections, list)
    assert len(collections) > 0
    assert any(c["id"] == collection["id"] for c in collections)


def test_update_collection(client, collection):
    # Update collection config
    update_data = {
        "title": "Updated E2E Test Collection",
        "description": "Updated E2E Test Collection Description",
        "config": {
            "source": "system",
            "enable_knowledge_graph": True,
            "embedding": {
                "model": EMBEDDING_MODEL_NAME,
                "model_service_provider": EMBEDDING_MODEL_PROVIDER,
                "custom_llm_provider": EMBEDDING_MODEL_CUSTOM_PROVIDER,
                "timeout": 2000,
            },
            "completion": {
                "model": COMPLETION_MODEL_NAME,
                "model_service_provider": COMPLETION_MODEL_PROVIDER,
                "custom_llm_provider": COMPLETION_MODEL_CUSTOM_PROVIDER,
                "timeout": 3000,
            },
        },
    }
    resp = client.put(f"/api/v1/collections/{collection['id']}", json=update_data)
    assert resp.status_code == 200
    updated = resp.json()
    assert_collection_config(update_data, updated)

    resp = client.get(f"/api/v1/collections/{collection['id']}")
    assert resp.status_code == 200
    got = resp.json()
    assert_collection_config(update_data, got)


def run_search_test(client, collection, document, search_data):
    resp = client.post(f"/api/v1/collections/{collection['id']}/searchTests", json=search_data)
    assert resp.status_code == 200
    result = resp.json()
    assert_search_test_result(search_data, result)

    resp = client.get(f"/api/v1/collections/{collection['id']}/searchTests")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["items"]) == 1
    assert data["items"][0]["id"] == result["id"]

    test_id = result["id"]
    resp = client.delete(f"/api/v1/collections/{collection['id']}/searchTests/{test_id}")
    assert resp.status_code == 200


@pytest.mark.parametrize(
    "search_data",
    [
        {"query": "test", "vector_search": {"topk": 10, "similarity": 0.1}},
        {"query": "test", "fulltext_search": {"topk": 10}},
        {"query": "test", "graph_search": {"topk": 10}},
        {
            "query": "test",
            "vector_search": {"topk": 10, "similarity": 0.1},
            "fulltext_search": {"topk": 10},
            "graph_search": {"topk": 10},
        },
    ],
    ids=["vector", "fulltext", "graph", "hybrid"]
)
def test_search_types(client, collection, document, search_data):
    run_search_test(client, collection, document, search_data)