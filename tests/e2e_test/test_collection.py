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
                "timeout": 2000,
            },
        },
    }
    resp = client.put(f"/api/v1/collections/{collection['id']}", json=update_data)
    assert resp.status_code == 200
    updated = resp.json()
    assert updated["title"] == "Updated E2E Test Collection"
    assert updated["description"] == "Updated E2E Test Collection Description"
    embedding_config = updated["config"]["embedding"]
    assert embedding_config["model"] == EMBEDDING_MODEL_NAME
    assert embedding_config["model_service_provider"] == EMBEDDING_MODEL_PROVIDER
    assert embedding_config["custom_llm_provider"] == EMBEDDING_MODEL_CUSTOM_PROVIDER
    assert embedding_config["timeout"] == 2000
    completion_config = updated["config"]["completion"]
    assert completion_config["model"] == COMPLETION_MODEL_NAME
    assert completion_config["model_service_provider"] == COMPLETION_MODEL_PROVIDER


def test_vector_search(client, collection, document):
    search_data = {"query": "test document", "search_type": "vector", "vector_search": {"topk": 10, "similarity": 0.1}}
    resp = client.post(f"/api/v1/collections/{collection['id']}/searchTests", json=search_data)
    assert resp.status_code == 200
    results = resp.json()
    assert len(results["items"]) > 0
    for item in results["items"]:
        assert "score" in item
        assert "content" in item
        assert "rank" in item


def test_full_text_search(client, collection, document):
    search_data = {"query": "unique test", "search_type": "fulltext", "fulltext_search": {"topk": 10}}
    resp = client.post(f"/api/v1/collections/{collection['id']}/searchTests", json=search_data)
    assert resp.status_code == 200
    results = resp.json()
    assert len(results["items"]) > 0
    for item in results["items"]:
        assert "score" in item
        assert "content" in item
        assert "rank" in item


def test_hybrid_search(client, collection, document):
    search_data = {
        "query": "specialized test",
        "search_type": "hybrid",
        "vector_search": {"topk": 10, "similarity": 0.1},
        "fulltext_search": {"topk": 10},
    }
    resp = client.post(f"/api/v1/collections/{collection['id']}/searchTests", json=search_data)
    assert resp.status_code == 200
    results = resp.json()
    assert len(results["items"]) > 0
    for item in results["items"]:
        assert "score" in item
        assert "content" in item
        assert "rank" in item
