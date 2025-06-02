from http import HTTPStatus

import httpx
import pytest

from tests.e2e_test.config import API_BASE_URL

API_KEY_ENDPOINT = "/api/v1/apikeys"
CONFIG_ENDPOINT = "/api/v1/collections"


def get_auth_headers(api_key):
    return {"Authorization": f"Bearer {api_key}"}


@pytest.fixture
def new_api_key(client):
    # Create a new API key for testing
    create_data = {"description": "e2e test key"}
    resp = client.post(API_KEY_ENDPOINT, json=create_data)
    assert resp.status_code == HTTPStatus.OK
    resp_data = resp.json()
    yield resp_data
    # Cleanup: delete the key if it still exists
    client.delete(f"{API_KEY_ENDPOINT}/{resp_data['id']}")
    assert resp.status_code == HTTPStatus.OK


def test_create_api_key(new_api_key):
    # Test API key creation
    assert new_api_key["id"] is not None
    assert new_api_key["key"] is not None
    assert new_api_key["description"] == "e2e test key"
    assert new_api_key["created_at"] is not None
    assert new_api_key["updated_at"] is not None
    assert new_api_key["last_used_at"] is None


def test_access_config_with_valid_api_key(client, new_api_key):
    # Test access /api/v1/config with valid API key
    api_key_value = new_api_key["key"]
    with httpx.Client(base_url=API_BASE_URL, headers=get_auth_headers(api_key_value)) as c:
        config_resp = c.get(CONFIG_ENDPOINT)
        assert config_resp.status_code == HTTPStatus.OK

    resp = client.get(f"{API_KEY_ENDPOINT}")
    assert resp.status_code == HTTPStatus.OK
    resp_data = resp.json()
    for k in resp_data["items"]:
        if k["id"] == new_api_key["id"]:
            assert k["last_used_at"] is not None
            break
    else:
        pytest.fail(f"API key {new_api_key['id']} not found in response")


def test_access_config_with_fake_api_key():
    # Test access /api/v1/config with a fake API key
    fake_key = "sk-fakekey1234567890"
    with httpx.Client(base_url=API_BASE_URL, headers=get_auth_headers(fake_key)) as c:
        config_resp = c.get(CONFIG_ENDPOINT)
        assert config_resp.status_code == HTTPStatus.UNAUTHORIZED


def test_update_api_key(client, new_api_key):
    # Test updating API key description
    api_key_id = new_api_key["id"]
    update_data = {"description": "updated desc"}
    resp = client.put(f"{API_KEY_ENDPOINT}/{api_key_id}", json=update_data)
    assert resp.status_code == HTTPStatus.OK
    resp_data = resp.json()
    assert resp_data["description"] == "updated desc"
    assert resp_data["updated_at"] is not None


def test_delete_api_key(client, new_api_key):
    # Test deleting API key
    api_key_id = new_api_key["id"]
    resp = client.delete(f"{API_KEY_ENDPOINT}/{api_key_id}")
    assert resp.status_code == HTTPStatus.OK
    # After deletion, access config should fail
    api_key_value = new_api_key["key"]
    with httpx.Client(base_url=API_BASE_URL, headers=get_auth_headers(api_key_value)) as c:
        config_resp = c.get(CONFIG_ENDPOINT)
        assert config_resp.status_code == HTTPStatus.UNAUTHORIZED
