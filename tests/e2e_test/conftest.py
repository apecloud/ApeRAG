import json
import time
from http import HTTPStatus

import httpx
import pytest

from tests.e2e_test.config import (
    API_BASE_URL,
    COMPLETION_MODEL_NAME,
    COMPLETION_MODEL_PROVIDER,
    COMPLETION_MODEL_PROVIDER_API_KEY,
    EMBEDDING_MODEL_CUSTOM_PROVIDER,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_MODEL_PROVIDER,
    EMBEDDING_MODEL_PROVIDER_API_KEY,
    RERANK_MODEL_PROVIDER,
    RERANK_MODEL_PROVIDER_API_KEY,
)
from tests.e2e_test.utils import assert_dict_subset


@pytest.fixture(scope="module")
def api_key(cookie_client):
    """Dynamically create an API key for testing and yield its value, then delete it after tests."""
    resp = cookie_client.post("/api/v1/apikeys", json={"description": "e2e dynamic key"})
    assert resp.status_code == HTTPStatus.OK, f"Failed to create API key: {resp.text}"
    api_key = resp.json()["key"]
    yield api_key
    resp = cookie_client.delete(f"/api/v1/apikeys/{resp.json()['id']}")
    assert resp.status_code == HTTPStatus.OK, f"Failed to delete API key: {resp.text}"


@pytest.fixture(scope="module")
def setup_model_service_provider(cookie_client):
    """Setup completion/embedding/rerank model service provider for testing."""
    resp = cookie_client.put(
        f"/api/v1/llm_providers/{COMPLETION_MODEL_PROVIDER}",
        json={"api_key": COMPLETION_MODEL_PROVIDER_API_KEY},
    )
    assert resp.status_code == HTTPStatus.OK, f"Failed to create completion model service provider: {resp.text}"
    resp = cookie_client.put(
        f"/api/v1/llm_providers/{EMBEDDING_MODEL_PROVIDER}",
        json={"api_key": EMBEDDING_MODEL_PROVIDER_API_KEY},
    )
    assert resp.status_code == HTTPStatus.OK, f"Failed to create embedding model service provider: {resp.text}"
    resp = cookie_client.put(
        f"/api/v1/llm_providers/{RERANK_MODEL_PROVIDER}", json={"api_key": RERANK_MODEL_PROVIDER_API_KEY}
    )
    assert resp.status_code == HTTPStatus.OK, f"Failed to create rerank model service provider: {resp.text}"


@pytest.fixture(scope="module")
def client(cookie_client, api_key, setup_model_service_provider):
    """Return a httpx.Client using api key authentication."""
    headers = {"Authorization": f"Bearer {api_key}"}
    with httpx.Client(base_url=API_BASE_URL, headers=headers) as c:
        yield c


@pytest.fixture
def collection(client):
    # Create collection
    data = {
        "title": "E2E Test Collection",
        "type": "document",
        "config": {
            "source": "system",
            "enable_knowledge_graph": False,
            "embedding": {
                "model": EMBEDDING_MODEL_NAME,
                "model_service_provider": EMBEDDING_MODEL_PROVIDER,
                "custom_llm_provider": EMBEDDING_MODEL_CUSTOM_PROVIDER,
            },
        },
    }
    resp = client.post("/api/v1/collections", json=data)
    assert resp.status_code == HTTPStatus.OK, f"status_code={resp.status_code}, resp={resp.text}"
    collection_data = resp.json()
    collection_id = collection_data["id"]
    assert collection_id is not None
    assert_dict_subset(data, collection_data)

    # Wait for collection to be active
    max_wait = 30
    interval = 2
    for _ in range(max_wait // interval):
        get_resp = client.get(f"/api/v1/collections/{collection_id}")
        assert get_resp.status_code == HTTPStatus.OK, get_resp.text
        got = get_resp.json()
        assert_dict_subset(data, got)
        if got.get("status") == "ACTIVE":
            break
        time.sleep(interval)
    else:
        pytest.fail(f"Collection {collection_id} failed to become active")

    yield collection_data

    # Cleanup: Delete collection
    delete_resp = client.delete(f"/api/v1/collections/{collection_id}")
    assert delete_resp.status_code == HTTPStatus.OK, delete_resp.text

    resp = client.get(f"/api/v1/collections/{collection_id}")
    assert resp.status_code == HTTPStatus.NOT_FOUND, resp.text


@pytest.fixture
def document(client, collection):
    # Upload a test document
    files = {"files": ("test.txt", "This is a test document for e2e.", "text/plain")}
    upload_resp = client.post(f"/api/v1/collections/{collection['id']}/documents", files=files)
    assert upload_resp.status_code == HTTPStatus.OK, upload_resp.text
    resp_data = upload_resp.json()
    assert len(resp_data["items"]) == 1
    doc_id = resp_data["items"][0]["id"]

    # Wait for document to be processed
    max_wait = 10
    interval = 2
    for _ in range(max_wait // interval):
        get_resp = client.get(f"/api/v1/collections/{collection['id']}/documents/{doc_id}")
        assert get_resp.status_code == HTTPStatus.OK, get_resp.text
        data = get_resp.json()
        if data.get("vector_index_status") == "ACTIVE" and data.get("fulltext_index_status") == "ACTIVE":
            break
        time.sleep(interval)
    else:
        pytest.fail(f"Document {doc_id} failed to be processed")

    yield {"id": doc_id, "content": files["files"][1]}

    # Cleanup: Delete document
    delete_resp = client.delete(f"/api/v1/collections/{collection['id']}/documents/{doc_id}")
    assert delete_resp.status_code == HTTPStatus.OK, delete_resp.text

    resp = client.get(f"/api/v1/collections/{collection['id']}/documents")
    assert resp.status_code == HTTPStatus.OK, resp.text
    data = resp.json()
    for item in data["items"]:
        if item["id"] == doc_id:
            assert item["status"] in ["DELETED", "DELETING"]


@pytest.fixture
def bot(client, document, collection):
    config = {
        "model_name": f"{COMPLETION_MODEL_NAME}",
        "model_service_provider": COMPLETION_MODEL_PROVIDER,
        "llm": {"context_window": 3500, "similarity_score_threshold": 0.5, "similarity_topk": 3, "temperature": 0.1},
    }
    create_data = {
        "title": "E2E Test Bot",
        "description": "E2E Bot Description",
        "type": "knowledge",
        "config": json.dumps(config),
        "collection_ids": [collection["id"]],
    }
    resp = client.post("/api/v1/bots", json=create_data)
    assert resp.status_code == HTTPStatus.OK, resp.text
    bot = resp.json()
    yield bot
    resp = client.delete(f"/api/v1/bots/{bot['id']}")
    assert resp.status_code in (200, 204), f"Failed to delete bot: {resp.status_code}, {resp.text}"


@pytest.fixture(scope="module")
def register_user():
    """Register a new user and return user info and password"""
    import random
    import string

    username = f"e2euser_{''.join(random.choices(string.ascii_lowercase, k=6))}"
    email = f"{username}@example.com"
    password = f"TestPwd!{random.randint(1000, 9999)}"
    data = {"username": username, "email": email, "password": password}
    resp = httpx.post(f"{API_BASE_URL}/api/v1/register", json=data)
    assert resp.status_code == HTTPStatus.OK, f"register failed: {resp.text}"
    user = resp.json()
    return {"username": username, "email": email, "password": password, "user": user}


@pytest.fixture(scope="module")
def login_user(register_user):
    """Login with the registered user and return cookies and user info"""
    data = {"username": register_user["username"], "password": register_user["password"]}
    with httpx.Client(base_url=API_BASE_URL) as c:
        resp = c.post("/api/v1/login", json=data)
        assert resp.status_code == HTTPStatus.OK, f"login failed: {resp.text}"
        cookies = c.cookies  # use httpx.Cookies directly
        user = resp.json()
        yield {
            "cookies": cookies,
            "user": user,
            "username": register_user["username"],
            "password": register_user["password"],
        }


@pytest.fixture(scope="module")
def cookie_client(login_user):
    """Return a httpx.Client with cookie-based authentication"""
    with httpx.Client(base_url=API_BASE_URL) as c:
        c.cookies.update(login_user["cookies"])
        yield c
